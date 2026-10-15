
# CELL 2
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

DAY = 86400.0
YEAR = 365.25 * DAY
T_REF = 20.0

METHOD_HWA = "PCM-HWA"
METHOD_DAR = "PCM-DAR-HWA"
METHOD_DS = "DriftSlice-PCM"
MAIN_METHODS = (METHOD_HWA, METHOD_DAR, METHOD_DS)

COLORS = {
    METHOD_HWA: "#167ac6",
    METHOD_DAR: "#e65a26",
    METHOD_DS: "#e6a400",
}

SCHEMA = "driftslice-unified-v3-auditable"
CODE_REVISION = "2026-10-15-v3-fp32-direct-hwa-selection"


@dataclass
class Config:
    # Reproducibility / storage
    seed: int = 42
    fast_mode: bool = False
    require_cuda: bool = True
    run_id: str = "driftslice_smoke_001"
    output_root: str = "driftslice_verified_runs"
    resume_policy: str = "auto_new"   # "auto_new" or "error"
    data_root: str = "data/MNIST/raw"
    allow_download: bool = True

    # Requested training semantics
    training_semantics: str = "fp32_pretrain_then_analog_hwa"
    digital_epochs: int = 30  # reconstruction assumption: paper does not specify digital epochs
    digital_lr: float = 1e-3
    digital_max_lr: float = 0.004
    digital_min_val_accuracy: float = 90.0  # validation only; full MNIST
    transfer_max_val_drop_pp: float = 5.0
    hwa_early_stop_drop_pp: float = 8.0
    hwa_early_stop_patience: int = 2
    batch_size: int = 256
    test_batch_size: int = 512
    epochs: int = 154
    train_repeats: int = 5
    lr: float = 1e-3
    max_lr: float = 0.004
    weight_decay: float = 1e-4
    grad_clip: float = 0.8

    # Paper-aligned HWA
    weight_scaling_omega: float = 0.8
    bits: int = 8
    weight_noise_std: float = 0.02
    output_noise_std: float = 0.04  # standard-HWA preset value
    output_bound: float = 10.0
    input_range_init: float = 3.0
    gmax_us: float = 25.0
    ir_drop: float = 0.0

    # DAR: paper values + explicitly undisclosed conventions
    dar_lambda_init: float = 0.5
    dar_fraction: float = 0.80
    dar_tc: float = 1e-2
    dar_eta: float = 1.0                     # NOT disclosed by paper
    dar_delta_mode: str = "signed_initial"   # NOT disclosed
    dar_g_units: str = "siemens"             # NOT disclosed: "siemens" or "normalized"
    dar_time_mode: str = "literal_seconds"   # source-literal interpretation
    dar_times_s: Tuple[float, ...] = (
        20.0, 60.0, 1800.0, 3600.0, DAY,
        30.0*DAY, 180.0*DAY, YEAR, 2.0*YEAR, 3.0*YEAR, 5.0*YEAR
    )

    # Retention evaluation
    realizations: int = 30
    retention_points: int = 75
    programming_noise_scale: float = 1.0
    read_noise_scale: float = 1.0
    drift_scale: float = 1.0

    # Proposed DriftSlice
    driftslice_bins: int = 4
    reference_cell_budget: int = 64
    refs_per_bin: int = 16
    reference_read_repeats: int = 4
    gain_min: float = 0.50
    gain_max: float = 3.00
    reference_floor_us: float = 0.10

    # Unified method selection and scientific safeguards
    methods: Tuple[str, ...] = (
        "PCM-HWA", "PCM-DAR-HWA", "PCM-DriftAttention-HWA",
        "PCM-NSR-HWA", "PCM-AttentionOnly-HWA", "PCM-GainOnly-HWA",
    )
    attention_lr: float = 5e-5
    attention_lambda: float = 0.5
    attention_input_gain: float = 0.9
    nsr_lambda: float = 0.1
    validation_reads: int = 3
    driftslice_selection_realizations: int = 3
    strict_154_epochs: bool = True

    # Diagnostics / ablations
    run_gdc_baselines: bool = True
    run_bin_ablation: bool = True
    ablation_bins: Tuple[int, ...] = (1, 2, 4, 8)
    cdf_times_s: Tuple[float, ...] = (20.0, 5.0*DAY, 5.0*YEAR)

    def resolved(self) -> "Config":
        c = copy.deepcopy(self)
        if c.fast_mode:
            c.epochs = 2
            c.digital_epochs = 2
            c.train_repeats = 1
            c.realizations = 2
            c.retention_points = 8
            c.refs_per_bin = min(c.refs_per_bin, 4)
            c.reference_read_repeats = 2
        if c.training_semantics != "fp32_pretrain_then_analog_hwa":
            raise ValueError("This notebook requires digital FP32 pretraining followed by analog HWA.")
        if c.digital_epochs < 1 or c.epochs < 1:
            raise ValueError("Both digital_epochs and epochs must be positive")
        if c.hwa_early_stop_patience < 1:
            raise ValueError("hwa_early_stop_patience must be positive")
        if c.bits < 2:
            raise ValueError("bits must be >= 2")
        if not 0 < c.dar_fraction <= 1:
            raise ValueError("dar_fraction must be in (0,1]")
        if c.dar_delta_mode not in ("signed_initial", "absolute_initial"):
            raise ValueError("Unknown dar_delta_mode")
        if c.dar_g_units not in ("siemens", "normalized"):
            raise ValueError("Unknown dar_g_units")
        if c.dar_time_mode not in ("literal_seconds", "schedule_index"):
            raise ValueError("Unknown dar_time_mode")
        if c.driftslice_bins < 1 or c.refs_per_bin < 1:
            raise ValueError("Invalid DriftSlice bank size")
        if not (0 < c.gain_min <= 1 <= c.gain_max):
            raise ValueError("gain bounds must include 1")
        if not c.fast_mode and c.strict_154_epochs:
            if c.digital_epochs != 30 or c.epochs != 154:
                raise ValueError("Full protocol requires 30 FP32 + 154 analog epochs")
        if not c.methods or len(set(c.methods)) != len(c.methods):
            raise ValueError("Training methods must be nonempty and unique")
        if c.validation_reads < 1 or c.driftslice_selection_realizations < 1:
            raise ValueError("Validation read/realization counts must be >= 1")
        if c.resume_policy not in ("auto_new", "error"):
            raise ValueError("resume_policy must be auto_new or error")
        if any(c.reference_cell_budget % b for b in (*c.ablation_bins, c.driftslice_bins)):
            raise ValueError("Reference budget must be divisible by every bin count")
        return c


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def random_stream(seed: int, device: torch.device):
    dev = torch.device(device)
    devices = []
    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        devices = [idx]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if devices:
            torch.cuda.manual_seed(int(seed))
        yield


def atomic_torch_save(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def check_environment(cfg: Config) -> torch.device:
    import aihwkit
    version = aihwkit.__version__
    if not version.startswith("1.1."):
        print(f"WARNING: audited against AIHWKit 1.1.x; found {version}.")
    if cfg.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but PyTorch CUDA is unavailable.")
    device = torch.device("cuda" if cfg.require_cuda else "cpu")
    print({
        "torch": torch.__version__,
        "aihwkit": version,
        "device": str(device),
        "training": cfg.training_semantics,
    })
    return device


def experiment_directory(cfg: Config, base_dir: Path):
    import aihwkit

    root = Path(base_dir) / cfg.output_root / ("smoke" if cfg.fast_mode else "full")
    root.mkdir(parents=True, exist_ok=True)

    contract_cfg = asdict(cfg).copy()
    contract_cfg.pop("run_id", None)
    contract_cfg.pop("resume_policy", None)
    contract = {
        "schema": SCHEMA,
        "code_revision": CODE_REVISION,
        "source_hash": scientific_source_hash(),
        "config": contract_cfg,
        "torch": torch.__version__,
        "aihwkit": aihwkit.__version__,
    }
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, default=str).encode()
    ).hexdigest()

    requested = root / cfg.run_id

    def read_manifest(directory: Path):
        p = directory / "manifest.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    old = read_manifest(requested) if requested.exists() else None

    if old and old.get("fingerprint") == digest:
        out = requested
        print("Resuming matching run:", out)
    elif old is None and (not requested.exists() or not any(requested.iterdir())):
        out = requested
        out.mkdir(parents=True, exist_ok=True)
        print("Starting new run:", out)
    else:
        if cfg.resume_policy == "error":
            raise ValueError("Run contract changed. Choose a new run_id.")
        stem = f"{cfg.run_id}__{digest[:10]}"
        out = root / stem
        i = 1
        while out.exists():
            manifest = read_manifest(out)
            if manifest and manifest.get("fingerprint") == digest:
                break
            out = root / f"{stem}_{i:02d}"
            i += 1
        out.mkdir(parents=True, exist_ok=True)
        print("Scientific contract changed; prior outputs were preserved.")
        print("Using:", out)

    (out / "manifest.json").write_text(
        json.dumps({"fingerprint": digest, **contract}, indent=2, default=str),
        encoding="utf-8",
    )
    return out, digest


def scientific_source_hash():
    """Hash executable definitions, including nested code; ignore cell filenames."""
    import types, marshal
    def canonical(code):
        consts = tuple(canonical(x) if isinstance(x, types.CodeType) else x
                       for x in code.co_consts)
        return code.replace(co_consts=consts, co_filename="<notebook>",
                            co_firstlineno=1, co_linetable=b"")
    digest = hashlib.sha256()
    for name, obj in sorted(list(globals().items())):
        items = [(name, obj)] if isinstance(obj, types.FunctionType) else []
        if isinstance(obj, type) and obj.__module__ == __name__:
            items += [(name+"."+k, v) for k,v in sorted(vars(obj).items())
                      if isinstance(v, types.FunctionType)]
        for key, fn in items:
            if fn.__globals__ is globals():
                digest.update(key.encode()); digest.update(marshal.dumps(canonical(fn.__code__)))
    return digest.hexdigest()


# CELL 4
def _read_idx(path: Path, kind: str):
    """Read an MNIST IDX file, transparently accepting either raw or .gz form."""
    path = Path(path)
    if not path.exists():
        gz = Path(str(path) + ".gz")
        if gz.exists():
            path = gz
        else:
            raise FileNotFoundError(f"Missing IDX file: {path} (or {gz})")

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        data = f.read()

    expected_magic = 2051 if kind == "image" else 2049
    magic = int.from_bytes(data[:4], "big")
    if magic != expected_magic:
        raise ValueError(
            f"Invalid IDX file {path}: magic={magic}, expected={expected_magic}"
        )

    n = int.from_bytes(data[4:8], "big")
    if kind == "image":
        h = int.from_bytes(data[8:12], "big")
        w = int.from_bytes(data[12:16], "big")
        arr = np.frombuffer(data, np.uint8, offset=16).copy().reshape(n, 1, h, w)
        return torch.from_numpy(arr).float() / 255.0

    arr = np.frombuffer(data, np.uint8, offset=8).copy()
    if arr.shape[0] != n:
        raise ValueError(f"Label count mismatch in {path}: header={n}, data={arr.shape[0]}")
    return torch.from_numpy(arr).long()


_MNIST_FILES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)


def _idx_exists(raw_dir: Path, name: str) -> bool:
    p = raw_dir / name
    return p.exists() or Path(str(p) + ".gz").exists()


def _mnist_complete(raw_dir: Path) -> bool:
    return all(_idx_exists(raw_dir, name) for name in _MNIST_FILES)


def resolve_mnist_raw_dir(cfg: Config, base_dir: Path) -> Path:
    """Find or create the MNIST/raw directory.

    Search order:
      1) cfg.data_root relative to BASE_DIR
      2) BASE_DIR/data/MNIST/raw
      3) current-working-directory/data/MNIST/raw

    If absent and cfg.allow_download=True, torchvision downloads MNIST into
    BASE_DIR/data, yielding BASE_DIR/data/MNIST/raw.
    """
    base_dir = Path(base_dir).expanduser().resolve()

    requested = Path(cfg.data_root).expanduser()
    candidates = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.append(base_dir / requested)

    candidates.extend([
        base_dir / "data" / "MNIST" / "raw",
        Path.cwd().resolve() / "data" / "MNIST" / "raw",
    ])

    # De-duplicate without requiring paths to exist.
    unique = []
    seen = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)

    for raw_dir in unique:
        if _mnist_complete(raw_dir):
            print(f"Using MNIST IDX files from: {raw_dir}")
            return raw_dir

    if not cfg.allow_download:
        searched = "\n  - ".join(str(p) for p in unique)
        raise FileNotFoundError(
            "MNIST was not found. Searched:\n"
            f"  - {searched}\n"
            "Either copy the four MNIST IDX files into one of those directories "
            "or set Config(allow_download=True)."
        )

    # Canonical torchvision destination: BASE_DIR/data/MNIST/raw
    dataset_root = base_dir / "data"
    dataset_root.mkdir(parents=True, exist_ok=True)

    print(f"MNIST not found locally. Downloading with torchvision into: {dataset_root}")
    try:
        from torchvision.datasets import MNIST
        MNIST(root=str(dataset_root), train=True, download=True)
        MNIST(root=str(dataset_root), train=False, download=True)
    except Exception as exc:
        raise RuntimeError(
            "Automatic MNIST download failed. Check internet access / proxy settings, "
            f"or manually place the IDX files under {dataset_root / 'MNIST' / 'raw'}."
        ) from exc

    raw_dir = dataset_root / "MNIST" / "raw"
    if not _mnist_complete(raw_dir):
        missing = [name for name in _MNIST_FILES if not _idx_exists(raw_dir, name)]
        raise FileNotFoundError(
            f"torchvision returned without a complete MNIST dataset at {raw_dir}. "
            f"Missing: {missing}"
        )

    print(f"MNIST download complete: {raw_dir}")
    return raw_dir


def build_loaders(cfg: Config, base_dir: Path, fold: int):
    root = resolve_mnist_raw_dir(cfg, base_dir)

    train_ds = TensorDataset(
        _read_idx(root / "train-images-idx3-ubyte", "image"),
        _read_idx(root / "train-labels-idx1-ubyte", "label"),
    )
    test_ds = TensorDataset(
        _read_idx(root / "t10k-images-idx3-ubyte", "image"),
        _read_idx(root / "t10k-labels-idx1-ubyte", "label"),
    )

    if len(train_ds) != 60000:
        raise RuntimeError(f"Expected 60,000 MNIST training samples, found {len(train_ds)}")
    if len(test_ds) != 10000:
        raise RuntimeError(f"Expected 10,000 MNIST test samples, found {len(test_ds)}")

    split_gen = torch.Generator().manual_seed(cfg.seed + fold)
    perm = torch.randperm(len(train_ds), generator=split_gen)
    n_train = int(0.9 * len(train_ds))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    if cfg.fast_mode:
        train_idx = train_idx[:1024]
        val_idx = val_idx[:256]
        test_idx = torch.arange(min(512, len(test_ds)))
    else:
        test_idx = torch.arange(len(test_ds))

    train_sub = torch.utils.data.Subset(train_ds, train_idx.tolist())
    val_sub = torch.utils.data.Subset(train_ds, val_idx.tolist())
    test_sub = torch.utils.data.Subset(test_ds, test_idx.tolist())

    train_gen = torch.Generator().manual_seed(cfg.seed + fold + 1000)
    loader_kwargs = dict(
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(
        train_sub,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=train_gen,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_sub,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_sub,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    split_info = {
        "fold": fold,
        "mnist_raw_dir": str(root),
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
    }
    return train_loader, val_loader, test_loader, split_info

# CELL 7
class DigitalMLP(nn.Module):
    """Paper architecture, digital FP32 pretraining (no analog simulation)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 250)
        self.fc2 = nn.Linear(250, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(1)
        return self.fc2(torch.tanh(self.fc1(x)))


def cpu_cloned_state(model: nn.Module):
    """Snapshot a PyTorch/AIHWKit state_dict without assuming all entries are Tensors.

    Analog tiles include non-Tensor state (e.g., nested dictionaries containing
    RPU configuration and metadata).  deepcopy preserves OrderedDict._metadata
    and non-Tensor structure needed by load_state_dict(); only Tensor leaves are
    detached, cloned, and transferred to CPU for checkpoint isolation.
    """
    state = copy.deepcopy(model.state_dict())

    def move_tensor_leaves(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            # Update in place so OrderedDict._metadata and mapping class survive.
            for key in value:
                value[key] = move_tensor_leaves(value[key])
            return value
        if isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = move_tensor_leaves(item)
            return value
        if isinstance(value, tuple):
            if hasattr(value, '_fields'):  # namedtuple
                return type(value)(*(move_tensor_leaves(x) for x in value))
            return tuple(move_tensor_leaves(x) for x in value)
        return value  # scalars, strings, config objects, and other metadata

    return move_tensor_leaves(state)


def train_digital_pretrained(cfg: Config, fold: int, train_loader, val_loader,
                             device: torch.device, out: Path, fingerprint: str):
    """Train digital FP32 model first, with best-validation checkpoint.

    Digital optimization recipe is an assumption; the 2025 paper only says
    the analog weights were initialized with pre-trained non-HWA weights.
    """
    seed_all(cfg.seed + fold)
    model = DigitalMLP().to(device=device, dtype=torch.float32)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    checkpoint = Path(out) / f"fold{fold}_digital_pretrained.pt"
    history_file = Path(out) / f"fold{fold}_digital_history.csv"
    if checkpoint.exists():
        pack = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if pack["fingerprint"] != fingerprint:
            raise RuntimeError("Digital pretrained checkpoint fingerprint mismatch")
        model.load_state_dict(pack["model"])
        hist = pd.DataFrame(pack["history"])
        _, acc = evaluate_model(model, val_loader, device)
        print(f"Digital fold={fold}: loaded FP32 pretrained checkpoint, val={acc:.2f}%")
        return model.eval(), hist

    train_loader.generator.manual_seed(cfg.seed + fold + 3000)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.digital_lr,
                                 weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.digital_max_lr,
        epochs=cfg.digital_epochs, steps_per_epoch=len(train_loader),
        div_factor=cfg.digital_max_lr / cfg.digital_lr,
    )
    best_acc = -1.0
    best_state = None
    rows = []
    for epoch in range(cfg.digital_epochs):
        model.train()
        total_ce = 0.0
        correct = 0
        n = 0
        for x, y in train_loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = F.cross_entropy(pred, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Digital loss nonfinite at epoch {epoch+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip,
                                           error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            bs = y.numel()
            n += bs
            correct += (pred.argmax(1) == y).sum().item()
            total_ce += loss.item() * bs
        val_ce, val_acc = evaluate_model(model, val_loader, device)
        rows.append({"fold":fold, "epoch":epoch+1,
                     "train_ce":total_ce/n, "train_accuracy":100*correct/n,
                     "val_ce":val_ce, "val_accuracy":val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = cpu_cloned_state(model)
        print(f"DIGITAL FP32 fold={fold} epoch={epoch+1}/{cfg.digital_epochs} "
              f"train={100*correct/n:.2f}% val={val_acc:.2f}%")
        pd.DataFrame(rows).to_csv(history_file, index=False)
    if best_state is None:
        raise RuntimeError("Digital training produced no valid model")
    model.load_state_dict(best_state)
    _, best_val_acc = evaluate_model(model, val_loader, device)
    if not cfg.fast_mode and best_val_acc < cfg.digital_min_val_accuracy:
        raise RuntimeError(f"Digital pretraining underfit: val={best_val_acc:.2f}%; "
                           "increase digital_epochs or inspect preprocessing")
    atomic_torch_save({"fingerprint":fingerprint, "model":best_state,
                       "history":rows, "best_val_accuracy":best_val_acc,
                       "dtype":"torch.float32"}, checkpoint)
    print(f"FP32 PRETRAIN PASS fold={fold}: val={best_val_acc:.2f}%")
    return model.eval(), pd.DataFrame(rows)


# CELL 8
def make_hwa_rpu(cfg: Config):
    from aihwkit.simulator.configs import TorchInferenceRPUConfig
    from aihwkit.simulator.parameters.enums import (
        BoundManagementType,
        NoiseManagementType,
        WeightModifierType,
        WeightClipType,
        WeightRemapType,
    )

    r = TorchInferenceRPUConfig()

    r.mapping.max_input_size = 0
    r.mapping.max_output_size = 0
    r.mapping.digital_bias = True
    r.mapping.weight_scaling_omega = cfg.weight_scaling_omega
    r.mapping.weight_scaling_columnwise = True
    r.mapping.learn_out_scaling = True
    r.mapping.out_scaling_columnwise = True

    levels = float(2**cfg.bits - 2)
    r.forward.inp_res = levels
    r.forward.out_res = levels
    r.forward.bound_management = BoundManagementType.NONE
    r.forward.noise_management = NoiseManagementType.CONSTANT
    r.forward.nm_thres = 1.0
    r.forward.out_noise = cfg.output_noise_std
    r.forward.out_bound = cfg.output_bound
    r.forward.ir_drop = cfg.ir_drop

    r.modifier.type = WeightModifierType.ADD_NORMAL
    r.modifier.std_dev = cfg.weight_noise_std
    r.modifier.rel_to_actual_wmax = True

    r.clip.type = WeightClipType.FIXED_VALUE
    r.clip.fixed_value = 1.0
    r.remap.type = WeightRemapType.NONE

    ir = r.pre_post.input_range
    ir.enable = True
    ir.init_value = cfg.input_range_init
    ir.init_from_data = 100
    ir.init_std_alpha = 3.0
    ir.decay = 0.001
    ir.input_min_percentage = 0.95
    ir.output_min_percentage = 0.95
    ir.manage_output_clipping = False

    r.drift_compensation = None
    return r


class AnalogMLP(nn.Module):
    def __init__(self, cfg: Config, seed: int, device: torch.device,
                 pretrained_state: Optional[dict] = None):
        super().__init__()
        from aihwkit.nn import AnalogLinear

        self.fc1 = AnalogLinear(784, 250, bias=True, rpu_config=make_hwa_rpu(cfg))
        self.fc2 = AnalogLinear(250, 10, bias=True, rpu_config=make_hwa_rpu(cfg))
        if pretrained_state is None:
            # Only used for synthetic preflight; production requires a digital checkpoint.
            gen = torch.Generator().manual_seed(seed)
            for layer in (self.fc1, self.fc2):
                w = torch.empty(layer.out_features, layer.in_features)
                nn.init.xavier_uniform_(w, generator=gen)
                layer.set_weights(w, torch.zeros(layer.out_features))
        else:
            for name, layer in self.layers().items():
                w = pretrained_state[f"{name}.weight"].detach().to("cpu", dtype=torch.float32)
                b = pretrained_state[f"{name}.bias"].detach().to("cpu", dtype=torch.float32)
                if tuple(w.shape) != (layer.out_features, layer.in_features):
                    raise RuntimeError(f"Pretrained {name}: shape mismatch: {tuple(w.shape)}")
                layer.set_weights(w, b)
        self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(1)
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)

    def layers(self):
        return {"fc1": self.fc1, "fc2": self.fc2}


def transfer_pretrained_digital_to_analog(cfg: Config, digital_model: DigitalMLP,
                                          seed: int, device: torch.device,
                                          val_loader=None):
    """Transfer *trained* FP32 weights into HWA layers without reinitialization."""
    state = cpu_cloned_state(digital_model)
    analog_model = AnalogMLP(cfg, seed, device, pretrained_state=state)
    for name, layer in analog_model.layers().items():
        logical_w, logical_b = layer.get_weights(apply_weight_scaling=True)
        torch.testing.assert_close(logical_w.detach().cpu(), state[f"{name}.weight"],
                                   rtol=2e-3, atol=2e-4,
                                   msg=lambda msg: f"Digital→analog transfer failed for {name} weights: {msg}")
        torch.testing.assert_close(logical_b.detach().cpu(), state[f"{name}.bias"],
                                   rtol=2e-3, atol=2e-4,
                                   msg=lambda msg: f"Digital→analog transfer failed for {name} bias: {msg}")
    if val_loader is not None:
        _, digital_acc = evaluate_model(digital_model, val_loader, device)
        _, analog_acc = evaluate_model(analog_model, val_loader, device)
        print(f"TRANSFER check: digital FP32={digital_acc:.2f}% → "
              f"analog HWA (before optimizer)={analog_acc:.2f}%")
        if (not cfg.fast_mode and
                digital_acc - analog_acc > cfg.transfer_max_val_drop_pp):
            raise RuntimeError("Weight transfer changed accuracy excessively; "
                               "the analog training has NOT begun")
    return analog_model


def physical_weight_parameter(layer) -> Tensor:
    tile, = list(layer.analog_tiles())
    w = tile.tile.weight
    if not w.requires_grad:
        raise RuntimeError("DAR requires a differentiable TorchInferenceTile weight tensor.")
    return w


def _sample_pcm_nu(g_rel: Tensor) -> Tensor:
    g_rel = g_rel.detach().clamp_min(1e-7)
    mu = (-0.0155 * g_rel.log() + 0.0244).clamp(0.049, 0.1)
    sigma = (-0.0125 * g_rel.log() - 0.0059).clamp(0.008, 0.045)
    return (mu + sigma * torch.randn_like(g_rel)).abs()


def dar_penalty(w: Tensor, initial_w: Tensor, epoch: int, t_inf_s: float, cfg: Config) -> Tensor:
    if t_inf_s < T_REF:
        raise ValueError("Absolute retention time must be >=20 seconds")
    flat = w.flatten()
    ref = initial_w.flatten().to(w)

    k = max(1, math.ceil(cfg.dar_fraction * flat.numel()))
    idx = flat.abs().topk(k, sorted=False).indices
    selected = flat[idx]

    w_inf = flat.abs().max().clamp_min(1e-8)
    g_rel = (selected.abs() / w_inf).clamp(1e-7, 1.0)
    nu = _sample_pcm_nu(g_rel)

    delta = selected - ref[idx]
    if cfg.dar_delta_mode == "absolute_initial":
        delta = delta.abs()

    if cfg.dar_time_mode == "literal_seconds":
        time_term = float(t_inf_s)
    else:
        time_term = float(cfg.dar_times_s.index(t_inf_s) + 1)

    lam = (
        cfg.dar_lambda_init
        / (epoch + 1.0) ** 1.5
        / (1.0 + cfg.dar_tc * time_term)
    )
    zeta = lam * (nu + cfg.dar_eta * delta / w_inf)

    if cfg.dar_g_units == "siemens":
        g = g_rel * cfg.gmax_us * 1e-6
    else:
        g = g_rel

    ratio = max(float(t_inf_s) / T_REF, 1.0)
    temporal = torch.expm1(-zeta * math.log(ratio)).abs()
    return (g * temporal).sum()


@torch.inference_mode()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite evaluation logits")
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    if not n:
        raise ValueError("Empty evaluation loader")
    return loss_sum / n, 100.0 * correct / n


def snapshot_analog_model(model: AnalogMLP) -> Dict[str, Dict[str, Tensor]]:
    result = {}
    for name, layer in model.layers().items():
        tile, = list(layer.analog_tiles())
        device = physical_weight_parameter(layer).device

        w, b = layer.get_weights(apply_weight_scaling=False)
        result[name] = {
            "weight": w.detach().to(device).clone(),
            "bias": b.detach().to(device).clone(),
            "scale": tile.get_scales().detach().to(device).clone(),
            "input_range": tile.input_range.detach().to(device).clone(),
        }
    return result


def train_analog_method_legacy(
    method: str,
    cfg: Config,
    fold: int,
    digital_model: DigitalMLP,
    train_loader,
    val_loader,
    device: torch.device,
    out: Path,
    fingerprint: str,
):
    from aihwkit.optim import AnalogAdam

    if method not in (METHOD_HWA, METHOD_DAR):
        raise ValueError(method)

    model_seed = cfg.seed + fold
    seed_all(model_seed)
    train_loader.generator.manual_seed(model_seed + 1000)

    model = transfer_pretrained_digital_to_analog(
        cfg, digital_model, model_seed, device,
        val_loader=val_loader if not (Path(out) / f"fold{fold}_{method}_checkpoint.pt").exists() else None,
    )

    analog_weights = [physical_weight_parameter(l) for l in model.layers().values()]
    weight_ids = {id(p) for p in analog_weights}
    other_params = [p for p in model.parameters() if id(p) not in weight_ids]

    optimizer = AnalogAdam(
        [
            {"params": analog_weights, "weight_decay": cfg.weight_decay},
            {"params": other_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )
    optimizer.regroup_param_groups(model)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.max_lr,
        epochs=cfg.epochs,
        steps_per_epoch=len(train_loader),
        div_factor=cfg.max_lr / cfg.lr,
    )

    initial_refs = {
        name: physical_weight_parameter(layer).detach().clone()
        for name, layer in model.layers().items()
    }

    checkpoint = Path(out) / f"fold{fold}_{method}_checkpoint.pt"
    history = []
    start_epoch = 0

    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint scientific contract mismatch.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        initial_refs = {k: v.to(device) for k, v in saved["initial_refs"].items()}
        history = saved["history"]
        start_epoch = saved["next_epoch"]
        train_loader.generator.set_state(saved["loader_rng"].cpu())
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)

    best_val = -1.0
    best_state = None
    best_epoch = 0
    patience = 0
    stopped_early = False
    if start_epoch > 0:
        best_val = float(saved.get("best_val_accuracy", -1.0))
        best_state = saved.get("best_model_state")
        best_epoch = int(saved.get("best_epoch", 0))
        stopped_early = bool(saved.get("stopped_early", False))
    else:
        _, best_val = evaluate_model(model, val_loader, device)
        best_state = cpu_cloned_state(model)
    init_analog_val = best_val if start_epoch == 0 else float(saved.get("initial_analog_val", best_val))
    print(f"{method}: transferred pretrained checkpoint; val before HWA={init_analog_val:.2f}%")
    started = time.perf_counter()

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        ce_sum = 0.0
        dar_sum = 0.0
        total_sum = 0.0
        correct = 0
        n = 0

        t_dar = cfg.dar_times_s[epoch % len(cfg.dar_times_s)]

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            ce = F.cross_entropy(logits, y)

            if method == METHOD_DAR:
                with random_stream(
                    cfg.seed + fold*1_000_000 + epoch*10_000 + batch_idx,
                    device,
                ):
                    dar = sum(
                        dar_penalty(
                            physical_weight_parameter(layer),
                            initial_refs[name],
                            epoch,
                            t_dar,
                            cfg,
                        )
                        for name, layer in model.layers().items()
                    )
            else:
                dar = torch.zeros((), device=device)

            loss = ce + dar
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{method}: non-finite loss epoch={epoch+1}, batch={batch_idx+1}; "
                    f"CE={ce.item()}, DAR={dar.item()}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for name, layer in model.layers().items():
                    tile, = list(layer.analog_tiles())
                    tile.input_range.clamp_(min=1e-3)
                    if not torch.isfinite(physical_weight_parameter(layer)).all():
                        raise FloatingPointError(f"{method}/{name}: non-finite analog weight")
                    if not torch.isfinite(tile.input_range).all():
                        raise FloatingPointError(f"{method}/{name}: non-finite input range")
                    if not torch.isfinite(tile.get_scales()).all():
                        raise FloatingPointError(f"{method}/{name}: non-finite output scale")

            bs = y.numel()
            ce_sum += ce.item() * bs
            dar_sum += dar.item() * bs
            total_sum += loss.item() * bs
            correct += (logits.argmax(1) == y).sum().item()
            n += bs

        val_loss, val_acc = evaluate_model(model, val_loader, device)
        elapsed = time.perf_counter() - started
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch + 1
            best_state = cpu_cloned_state(model)
        # Do not confuse scientific early stopping with full 154-epoch reproduction.
        if not cfg.fast_mode and val_acc < init_analog_val - cfg.hwa_early_stop_drop_pp:
            patience += 1
        else:
            patience = 0
        stopped_early = patience >= cfg.hwa_early_stop_patience

        history.append({
            "method": method,
            "fold": fold,
            "epoch": epoch + 1,
            "t_dar_s": t_dar,
            "train_ce": ce_sum / n,
            "train_dar": dar_sum / n,
            "train_loss": total_sum / n,
            "train_accuracy": 100.0 * correct / n,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "best_val_accuracy": best_val,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "paper_154_epochs_complete": False,
            "eta_seconds": elapsed / (epoch - start_epoch + 1) * (cfg.epochs - epoch - 1),
        })

        pd.DataFrame(history).to_csv(
            Path(out) / f"fold{fold}_{method}_history.csv", index=False
        )

        atomic_torch_save(
            {
                "fingerprint": fingerprint,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "initial_refs": initial_refs,
                "next_epoch": epoch + 1,
                "history": history,
                "best_val_accuracy": best_val,
                "best_epoch": best_epoch,
                "best_model_state": best_state,
                "initial_analog_val": init_analog_val,
                "stopped_early": stopped_early,
                "loader_rng": train_loader.generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            },
            checkpoint,
        )

        print(
            f"{method} fold={fold} epoch={epoch+1}/{cfg.epochs} "
            f"CE={ce_sum/n:.4f} DAR={dar_sum/n:.6g} "
            f"train={100*correct/n:.2f}% val={val_acc:.2f}%"
        )
        if stopped_early:
            print(f"EARLY STOP ({method}): validation fell below the preset threshold; "
                  "restoring best checkpoint. NOT an exact 154-epoch paper run.")
            break

    # Never silently present the original transferred digital model as
    # successful HWA. Only restore best weights for an early-stopped run;
    # otherwise retain the actual weights at the final HWA epoch.
    if stopped_early and best_state is not None:
        model.load_state_dict(best_state)
        if best_epoch == 0:
            print(f"SCIENTIFIC INVALIDITY WARNING ({method}): best validation was "
                  "BEFORE analog training; this selected checkpoint has no HWA updates. "
                  "Do not report it as an HWA baseline.")
    # Save the selected model for subsequent restarts, with explicit scientific status.
    if history:
        completed_154 = bool((history[-1]["epoch"] == 154) and not stopped_early and not cfg.fast_mode)
        selected_epoch = best_epoch if stopped_early else history[-1]["epoch"]
        for row in history:
            row["paper_154_epochs_complete"] = completed_154
            row["selected_epoch"] = selected_epoch
            row["selected_checkpoint_was_HWA_trained"] = bool(selected_epoch > 0)
        pd.DataFrame(history).to_csv(
            Path(out) / f"fold{fold}_{method}_history.csv", index=False)
        # Final checkpoint contains exactly the selected model, not a stale
        # last-epoch model if an early stop restored an earlier checkpoint.
        atomic_torch_save({
                "fingerprint": fingerprint, "model":cpu_cloned_state(model),
                "optimizer":optimizer.state_dict(), "scheduler":scheduler.state_dict(),
                "initial_refs":initial_refs, "next_epoch":cfg.epochs,
                "history":history,"best_val_accuracy":best_val,
                "best_epoch":best_epoch,
                "selected_epoch":selected_epoch,
                "best_model_state":best_state,"initial_analog_val":init_analog_val,
                "stopped_early":stopped_early,
                "loader_rng":train_loader.generator.get_state(),
                "torch_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state(device) if device.type=="cuda" else None,
            },checkpoint)
    return model.eval(), pd.DataFrame(history)


# CELL 10
def elapsed_pcm_time(t_absolute_s):
    """Paper-reference convention: t=20 s maps to AIHWKit elapsed time zero."""
    if float(t_absolute_s) < T_REF:
        raise ValueError("Absolute retention time must be >=20 seconds")
    return float(t_absolute_s) - T_REF


def make_pcm_noise_model(cfg: Config):
    from aihwkit.inference import PCMLikeNoiseModel
    from aihwkit.inference.converter.conductance import SinglePairConductanceConverter

    converter = SinglePairConductanceConverter(g_max=cfg.gmax_us, g_min=0.0)
    noise = PCMLikeNoiseModel(
        g_converter=converter,
        prog_noise_scale=cfg.programming_noise_scale,
        read_noise_scale=cfg.read_noise_scale,
        drift_scale=cfg.drift_scale,
    )
    return noise, converter


class PCMPhysicalMatrix:
    """One differential PCM matrix realization."""
    def __init__(self, logical_weight: Tensor, cfg: Config, seed: int):
        self.cfg = cfg
        self.device = logical_weight.device
        self.noise, self.converter = make_pcm_noise_model(cfg)

        g_target, params = self.converter.convert_to_conductances(logical_weight.detach())
        if len(g_target) != 2:
            raise RuntimeError("Expected one positive and one negative conductance tensor.")

        self.g_target = tuple(g.detach().clone() for g in g_target)
        self.params = params

        with random_stream(seed, self.device):
            self.g_programmed = tuple(
                self.noise.apply_programming_noise_to_conductance(g).clamp(0, cfg.gmax_us)
                for g in self.g_target
            )
            self.nu = tuple(
                self.noise.generate_drift_coefficients(g)
                for g in self.g_target
            )

    def read_conductances(self, t_inf_s: float, seed: int, noiseless: bool = False):
        elapsed = elapsed_pcm_time(t_inf_s)

        if noiseless:
            t = elapsed + self.noise.t_0
            return tuple(
                g * ((t / self.noise.t_0) ** (-nu))
                for g, nu in zip(self.g_programmed, self.nu)
            )

        with random_stream(seed, self.device):
            return tuple(
                self.noise.apply_drift_noise_to_conductance(g, nu, elapsed)
                for g, nu in zip(self.g_programmed, self.nu)
            )

    def logical_weight(self, conductances):
        return self.converter.convert_back_to_weights(list(conductances), self.params)


def conductance_edges(cfg: Config, bins: int, device: torch.device):
    return torch.linspace(0.0, cfg.gmax_us, int(bins) + 1, device=device)


def class_indices(g_target: Tensor, edges: Tensor):
    return torch.bucketize(g_target, edges[1:-1], right=False).long()


class PCMClassReferenceBank:
    """Small class reference bank shared by all layers of one realization."""
    def __init__(self, cfg: Config, bins: int, seed: int, device: torch.device):
        self.cfg = cfg
        self.bins = int(bins)
        self.seed = int(seed)
        self.device = device
        self.noise, _ = make_pcm_noise_model(cfg)
        self.edges = conductance_edges(cfg, self.bins, device)

        refs_per_class = cfg.reference_cell_budget // self.bins
        if cfg.reference_cell_budget % self.bins or refs_per_class < 1:
            raise ValueError("Invalid fixed reference budget")
        refs = []
        for b in range(self.bins):
            lo = float(self.edges[b].item())
            hi = float(self.edges[b+1].item())
            hi_eff = hi - min(1e-4, (hi-lo)/10)
            lo_eff = min(max(lo + 1e-4, cfg.reference_floor_us), hi_eff)
            refs.append(torch.linspace(lo_eff, hi_eff, refs_per_class, device=device))
        self.target = torch.stack(refs, dim=0)  # [B,R]

        with random_stream(seed, device):
            self.programmed = self.noise.apply_programming_noise_to_conductance(
                self.target
            ).clamp(0, cfg.gmax_us)
            self.nu = self.noise.generate_drift_coefficients(self.target)

        # One stored reference measurement at the 20-s reference point.
        self.ref20 = self._read(T_REF, seed + 333)

    def _read(self, t_inf_s: float, seed: int):
        reads = []
        for rep in range(self.cfg.reference_read_repeats):
            with random_stream(seed + rep * 7919, self.device):
                reads.append(
                    self.noise.apply_drift_noise_to_conductance(
                        self.programmed, self.nu, elapsed_pcm_time(t_inf_s)
                    )
                )
        return torch.stack(reads, dim=0).mean(dim=0)

    def gains(self, t_inf_s: float, seed: int):
        now = self._read(t_inf_s, seed)
        numerator = self.ref20.mean(dim=1)
        denominator = now.mean(dim=1).clamp_min(1e-6)
        gain = numerator / denominator
        return gain.clamp(self.cfg.gain_min, self.cfg.gain_max)


def apply_driftslice(
    matrix: PCMPhysicalMatrix,
    current_g: Tuple[Tensor, Tensor],
    gains: Tensor,
    edges: Tensor,
):
    corrected = []
    for target, current in zip(matrix.g_target, current_g):
        idx = class_indices(target, edges)
        corrected.append(current * gains[idx])
    return tuple(corrected)


def global_gdc_gain(w_ref, w_now, snap, io, seed, device):
    """Measured all-ones calibration through the same DAC/ADC path.

    Per-layer functional calibration, not native mapped-tile GDC.
    Bias is excluded from the measurement. Independent repeated read noise.
    """
    calibration = dict(snap)
    calibration["bias"] = torch.zeros_like(snap["bias"])
    ones = torch.ones(1, w_ref.shape[1], device=device, dtype=w_ref.dtype)
    with torch.inference_mode(), random_stream(seed, device):
        baseline = torch.stack([shared_mvm(ones, w_ref, calibration, io).abs().mean()
                                for _ in range(4)]).mean()
        current = torch.stack([shared_mvm(ones, w_now, calibration, io).abs().mean()
                               for _ in range(4)]).mean()
    if not torch.isfinite(current) or current <= 1e-8:
        raise FloatingPointError("Invalid GDC calibration readout")
    return float(baseline/current)


def shared_mvm(
    x: Tensor,
    weight: Tensor,
    snap: Dict[str, Tensor],
    io,
    gain: float = 1.0,
):
    from aihwkit.simulator.tiles.analog_mvm import AnalogMVM

    r = snap["input_range"].abs().clamp_min(1e-8)
    normalized = x.clamp(-r.item(), r.item()) / r
    y = AnalogMVM.matmul(weight, normalized, io, trans=False, is_test=True)
    y = y * r
    y = y * snap["scale"].view(1, -1) * float(gain)
    return y + snap["bias"]


class SnapshotMLP(nn.Module):
    def __init__(self, snapshots, effective_weights, cfg: Config, layer_gains=None):
        super().__init__()
        self.snapshots = snapshots
        self.effective_weights = effective_weights
        self.layer_gains = layer_gains or {"fc1": 1.0, "fc2": 1.0}
        self.io = copy.deepcopy(make_hwa_rpu(cfg).forward)

    def forward(self, x):
        x = shared_mvm(
            x.flatten(1),
            self.effective_weights["fc1"],
            self.snapshots["fc1"],
            self.io,
            self.layer_gains["fc1"],
        )
        x = torch.tanh(x)
        return shared_mvm(
            x,
            self.effective_weights["fc2"],
            self.snapshots["fc2"],
            self.io,
            self.layer_gains["fc2"],
        )


@torch.inference_mode()
def evaluate_snapshot(model: SnapshotMLP, loader: DataLoader, device: torch.device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    return loss_sum / n, 100.0 * correct / n


# CELL 12
def retention_times(cfg: Config):
    anchors = np.array([
        20.0, 60.0, 1800.0, 3600.0, DAY, 5*DAY,
        30*DAY, 180*DAY, YEAR, 2*YEAR, 3*YEAR, 5*YEAR
    ], dtype=float)
    grid = np.geomspace(20.0, 5.0*YEAR, cfg.retention_points)
    return np.unique(np.concatenate([anchors, grid]))


def build_main_pcm(snapshot, cfg: Config, seed_base: int):
    return {
        "fc1": PCMPhysicalMatrix(snapshot["fc1"]["weight"], cfg, seed_base + 101),
        "fc2": PCMPhysicalMatrix(snapshot["fc2"]["weight"], cfg, seed_base + 202),
    }


def plain_effective_weights(main_pcm, t_s: float, seed: int):
    out = {}
    conductances = {}
    for i, name in enumerate(("fc1", "fc2")):
        g = main_pcm[name].read_conductances(t_s, seed + i*1009)
        conductances[name] = g
        out[name] = main_pcm[name].logical_weight(g)
    return out, conductances


def driftslice_effective_weights(main_pcm, conductances, ref_bank, t_s: float, seed: int):
    gains = ref_bank.gains(t_s, seed)
    out = {}
    for name in ("fc1", "fc2"):
        g_corr = apply_driftslice(
            main_pcm[name], conductances[name], gains, ref_bank.edges
        )
        out[name] = main_pcm[name].logical_weight(g_corr)
    return out, gains


def retention_realization_legacy(
    hwa_snap,
    dar_snap,
    cfg: Config,
    fold: int,
    realization: int,
    test_loader,
    device: torch.device,
):
    root = cfg.seed + fold*1_000_000 + realization*10_000

    # Critical fairness constraint:
    # PCM-HWA and DriftSlice share the exact same main-array programming realization.
    hwa_main = build_main_pcm(hwa_snap, cfg, root + 10)
    dar_main = build_main_pcm(dar_snap, cfg, root + 20)
    ref_bank = PCMClassReferenceBank(
        cfg, cfg.driftslice_bins, root + 30, device
    )

    times = retention_times(cfg)

    hwa_ref, _ = plain_effective_weights(hwa_main, T_REF, root + 40)
    dar_ref, _ = plain_effective_weights(dar_main, T_REF, root + 50)

    rows = []
    cdf_states = {}

    for ti, t_s in enumerate(times):
        read_seed = root + 100_000 + ti*100

        hwa_w, hwa_g = plain_effective_weights(hwa_main, float(t_s), read_seed)
        # Same HWA conductance read is reused by DriftSlice.
        ds_w, class_gain = driftslice_effective_weights(
            hwa_main, hwa_g, ref_bank, float(t_s), read_seed + 500_000
        )
        dar_w, _ = plain_effective_weights(dar_main, float(t_s), read_seed + 7)

        methods = {
            METHOD_HWA: (hwa_snap, hwa_w),
            METHOD_DAR: (dar_snap, dar_w),
            METHOD_DS: (hwa_snap, ds_w),
        }

        for method, (snap, weights) in methods.items():
            short_noise_seed = root + 700_000 + ti*31
            with random_stream(short_noise_seed, device):
                model = SnapshotMLP(snap, weights, cfg).to(device)
                loss, acc = evaluate_snapshot(model, test_loader, device)

            row = {
                "fold": fold,
                "realization": realization,
                "method": method,
                "gdc": False,
                "t_s": float(t_s),
                "loss": loss,
                "accuracy": acc,
                "error": 100.0 - acc,
            }
            if method == METHOD_DS:
                for b, value in enumerate(class_gain.detach().cpu().tolist()):
                    row[f"class_gain_{b}"] = value
            rows.append(row)

        if cfg.run_gdc_baselines:
            for method, snap, weights, ref in [
                (METHOD_HWA, hwa_snap, hwa_w, hwa_ref),
                (METHOD_DAR, dar_snap, dar_w, dar_ref),
            ]:
                gains = {
                    name: global_gdc_gain(ref[name], weights[name], snap[name],
                                          make_hwa_rpu(cfg).forward,
                                          root + 850_000 + ti*31 + (name == "fc2"), device)
                    for name in ("fc1", "fc2")
                }
                with random_stream(root + 800_000 + ti*31, device):
                    model = SnapshotMLP(snap, weights, cfg, layer_gains=gains).to(device)
                    loss, acc = evaluate_snapshot(model, test_loader, device)

                rows.append({
                    "fold": fold,
                    "realization": realization,
                    "method": method,
                    "gdc": True,
                    "t_s": float(t_s),
                    "loss": loss,
                    "accuracy": acc,
                    "error": 100.0 - acc,
                })

        for target_t in cfg.cdf_times_s:
            if math.isclose(float(t_s), float(target_t), rel_tol=0.0, abs_tol=1e-6):
                cdf_states[float(target_t)] = {
                    METHOD_HWA: {k: v.detach().cpu() for k, v in hwa_w.items()},
                    METHOD_DAR: {k: v.detach().cpu() for k, v in dar_w.items()},
                    METHOD_DS: {k: v.detach().cpu() for k, v in ds_w.items()},
                }

    return pd.DataFrame(rows), cdf_states


# CELL 14
def driftslice_bin_ablation_legacy(
    hwa_snap,
    cfg: Config,
    fold: int,
    realization: int,
    loader,
    device: torch.device,
    t_s: float = 5*YEAR,
):
    root = cfg.seed + 9_000_000 + fold*100_000 + realization*10_000
    main = build_main_pcm(hwa_snap, cfg, root + 1)

    # One fixed main-array state for every B.
    read_seed = root + 2
    conductances = {}
    for i, name in enumerate(("fc1", "fc2")):
        conductances[name] = main[name].read_conductances(t_s, read_seed + i*13)

    rows = []

    for B in cfg.ablation_bins:
        bank = PCMClassReferenceBank(cfg, int(B), root + 100 + int(B), device)
        gains = bank.gains(t_s, root + 1000 + int(B))

        weights = {}
        for name in ("fc1", "fc2"):
            g_corr = apply_driftslice(
                main[name], conductances[name], gains, bank.edges
            )
            weights[name] = main[name].logical_weight(g_corr)

        with random_stream(root + 5000, device):
            model = SnapshotMLP(hwa_snap, weights, cfg).to(device)
            loss, acc = evaluate_snapshot(model, loader, device)

        rows.append({
            "fold": fold,
            "realization": realization,
            "bins": int(B),
            "t_s": float(t_s),
            "loss": loss,
            "accuracy": acc,
            "error": 100.0 - acc,
            "reference_cells": cfg.reference_cell_budget,
        })

    return pd.DataFrame(rows)


# CELL 16
def _save_figure(fig, stem: Path, caption: str):
    import matplotlib.pyplot as plt

    stem = Path(stem)
    import textwrap
    wrapped = textwrap.fill(caption, width=max(60, int(fig.get_figwidth()*15)))
    margin = min(.42, .10 + .028*len(wrapped.splitlines()))
    fig.subplots_adjust(bottom=margin, wspace=.3)
    fig.text(0.07, 0.015, wrapped, ha="left", va="bottom", fontsize=8)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    stem.with_suffix(".caption.txt").write_text(caption, encoding="utf-8")
    plt.close(fig)


def plot_training(history: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.4), sharex=True)
    panels = [METHOD_HWA, METHOD_DAR, METHOD_DS]

    for i, (ax, method) in enumerate(zip(axes, panels)):
        source = METHOD_HWA if method == METHOD_DS else method
        g = history[history.method == source].groupby("epoch").agg(
            train=("train_accuracy", "mean"),
            train_sd=("train_accuracy", "std"),
            val=("val_accuracy", "mean"),
            val_sd=("val_accuracy", "std"),
        ).fillna(0.0)

        ax.plot(g.index, g.train, color="#167ac6", lw=1.6, label=r"$A_{train}$")
        ax.plot(g.index, g.val, color="#e65a26", lw=1.6, label=r"$A_{val}$")
        ax.fill_between(g.index, g.train-g.train_sd, g.train+g.train_sd,
                        color="#167ac6", alpha=0.13)
        ax.fill_between(g.index, g.val-g.val_sd, g.val+g.val_sd,
                        color="#e65a26", alpha=0.13)
        ax.set_ylim(0, 100)
        ax.set_ylabel(r"$\mathcal{A}$ [%]")
        ax.grid(alpha=0.22)
        ax.text(0.02, 0.86, f"({chr(97+i)}) {method}",
                transform=ax.transAxes, fontsize=11, fontweight="bold")

        if method == METHOD_DS:
            ax.text(
                0.02, 0.08,
                "Same PCM-HWA checkpoint; no additional training",
                transform=ax.transAxes, fontsize=8.5,
                bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.9),
            )

        tail = g.iloc[-max(2, math.ceil(0.2*len(g))):]
        inset = ax.inset_axes([0.66, 0.17, 0.30, 0.38])
        inset.plot(tail.index, tail.train, color="#167ac6", lw=1.0)
        inset.plot(tail.index, tail.val, color="#e65a26", lw=1.0)
        y0 = min(tail.train.min(), tail.val.min())
        inset.set_ylim(max(0, y0 - 2), 100)
        inset.grid(alpha=0.2)

    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    axes[-1].set_xlabel("Epochs [#]")

    caption = (
        "Fig. 1. Analog training and validation accuracy for (a) PCM-HWA, "
        "(b) PCM-DAR-HWA, and (c) the PCM-HWA checkpoint reused by DriftSlice-PCM. "
        "DriftSlice introduces no additional training run."
    )
    _save_figure(fig, Path(out) / "fig1_training", caption)


def _time_ticks():
    vals = np.array([
        20, 60, 1800, 5*3600, DAY, 7*DAY, 30*DAY,
        5*30*DAY, YEAR, 2*YEAR, 5*YEAR
    ], dtype=float)
    labels = ["20s", "1m", "30m", "5h", "1d", "7d", "1M", "5M", "1y", "2y", "5y"]
    return vals, labels


def plot_retention(results: pd.DataFrame, out: Path, cfg: Config):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.3), sharex=True)

    for method in MAIN_METHODS:
        g = results[(results.method == method) & (~results.gdc)].groupby("t_s").error
        mean = g.mean()
        sd = g.std().fillna(0)
        x = mean.index.to_numpy()

        axes[0].plot(x, mean.to_numpy(), color=COLORS[method], lw=1.7, label=method)
        axes[0].fill_between(
            x, (mean-sd).to_numpy(), (mean+sd).to_numpy(),
            color=COLORS[method], alpha=0.13
        )

    axes[0].set_xscale("log")
    axes[0].set_ylabel(r"$\mathcal{E}$ [%]")
    axes[0].grid(alpha=0.22)
    axes[0].text(0.02, 0.84, "(a) Main comparison", transform=axes[0].transAxes, fontsize=11)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=3, frameon=False)

    if cfg.run_gdc_baselines:
        for method in (METHOD_HWA, METHOD_DAR):
            g = results[(results.method == method) & (results.gdc)].groupby("t_s").error
            mean = g.mean()
            sd = g.std().fillna(0)
            axes[1].plot(
                mean.index, mean.values, color=COLORS[method], lw=1.7,
                label=method + " + GDC"
            )
            axes[1].fill_between(
                mean.index, (mean-sd).values, (mean+sd).values,
                color=COLORS[method], alpha=0.13
            )

        g = results[(results.method == METHOD_DS) & (~results.gdc)].groupby("t_s").error
        mean = g.mean()
        sd = g.std().fillna(0)
        axes[1].plot(mean.index, mean.values, color=COLORS[METHOD_DS], lw=1.7, label=METHOD_DS)
        axes[1].fill_between(
            mean.index, (mean-sd).values, (mean+sd).values,
            color=COLORS[METHOD_DS], alpha=0.13
        )
        axes[1].legend(frameon=False, ncol=3, fontsize=8)

    axes[1].set_ylabel(r"$\mathcal{E}$ [%]")
    axes[1].grid(alpha=0.22)
    axes[1].text(0.02, 0.84, "(b) GDC reference comparison",
                 transform=axes[1].transAxes, fontsize=11)

    for method in MAIN_METHODS:
        subset = results[(results.method == method) & (~results.gdc)]
        g = subset.groupby(["fold", "t_s"]).error.var(ddof=1).groupby("t_s").mean().pow(0.5)
        axes[2].plot(g.index, g.values, color=COLORS[method], lw=1.4, label=method)

    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"$\sigma(\mathcal{E})$")
    axes[2].grid(alpha=0.22)
    axes[2].text(0.02, 0.84, "(c)", transform=axes[2].transAxes, fontsize=11)

    ticks, labels = _time_ticks()
    axes[2].set_xticks(ticks)
    axes[2].set_xticklabels(labels, rotation=45, ha="right")
    axes[2].set_xlabel(r"$t_{\mathrm{inf}}$")
    axes[2].set_xscale("log")

    caption = (
        "Fig. 2. Classification error versus PCM retention time. "
        "(a) PCM-HWA, PCM-DAR-HWA, and proposed DriftSlice-PCM without global drift compensation. "
        "(b) GDC-enabled HWA/DAR baselines compared with DriftSlice conductance-class reference equalization. "
        "(c) RMS within-training-seed hardware SD (training variability excluded). "
        "Shaded regions denote pooled ±1 SD including training and hardware variability. "
        "GDC is functional per-layer measured calibration, not native mapped-tile GDC."
    )
    _save_figure(fig, Path(out) / "fig2_retention", caption)


def ecdf(values: np.ndarray):
    x = np.sort(np.asarray(values).reshape(-1))
    y = np.arange(1, len(x)+1) / len(x)
    return x, y


def plot_weight_cdfs(cdf_states, out: Path):
    import matplotlib.pyplot as plt

    methods = [METHOD_HWA, METHOD_DAR, METHOD_DS]
    times = sorted(cdf_states.keys())
    time_colors = ["#167ac6", "#e65a26", "#e6a400"]

    fig, axes = plt.subplots(3, 2, figsize=(8.2, 8.4), sharex="col", sharey=True)

    t0 = min(times)
    norm = {
        method: {
            layer: cdf_states[t0][method][layer].abs().max().item() + 1e-12
            for layer in ("fc1", "fc2")
        }
        for method in methods
    }

    for r, method in enumerate(methods):
        for c, layer in enumerate(("fc1", "fc2")):
            ax = axes[r, c]
            for k, t in enumerate(times):
                w = cdf_states[t][method][layer].numpy() / norm[method][layer]
                x, y = ecdf(w)

                label = None
                if r == 0 and c == 0:
                    if math.isclose(t, 20.0):
                        label = "20 seconds"
                    elif math.isclose(t, 5*DAY):
                        label = "5 days"
                    elif math.isclose(t, 5*YEAR):
                        label = "5 years"

                ax.plot(x, y, color=time_colors[k % len(time_colors)], lw=1.4, label=label)

            ax.grid(alpha=0.22)
            if c == 0:
                ax.set_ylabel(f"{method}\n$F_{{CD}}(W)$")
            if r == 2:
                ax.set_xlabel("First Layer Weights" if c == 0 else "Second Layer Weights")

    axes[0,0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.20), ncol=3, frameon=False)

    caption = (
        "Fig. 3. Cumulative distribution functions of normalized effective layer weights "
        "at 20 s, 5 d, and 5 y. Rows show PCM-HWA, PCM-DAR-HWA, and DriftSlice-PCM; "
        "columns show the first and second layers. DriftSlice uses the class-equalized "
        "effective matrix seen by the shared ADC."
    )
    _save_figure(fig, Path(out) / "fig3_weight_cdfs", caption)


def plot_bin_ablation(ablation: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt

    g = ablation.groupby("bins").agg(
        error=("error", "mean"),
        error_sd=("error", "std"),
        refs=("reference_cells", "mean"),
    ).fillna(0).reset_index()

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(g.bins, g.error, yerr=g.error_sd, marker="o", capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(g.bins)
    ax.set_xticklabels([str(int(x)) for x in g.bins])
    ax.set_xlabel("Number of conductance classes B")
    ax.set_ylabel("5-year validation error [%]")
    ax.grid(alpha=0.25)

    caption = (
        "Fig. 4. Validation-only DriftSlice conductance-class ablation at five-year retention. "
        "The main PCM realization is fixed while the class-reference granularity changes. "
        "The total reference-cell budget is fixed across B; B=1 is the single-class control."
    )
    _save_figure(fig, Path(out) / "fig4_bin_ablation", caption)


def compare_digital_and_hwa(digital, hwa, dar, test_loader, device, cfg, fold):
    """Test-only reporting; never used to select checkpoints or hyperparameters."""
    rows = []
    for label, model in [("FP32 pretrained", digital), (METHOD_HWA, hwa), (METHOD_DAR, dar)]:
        with random_stream(cfg.seed + 600000 + fold, device):
            loss, acc = evaluate_model(model, test_loader, device)
        rows.append(dict(fold=fold, stage=label, accuracy=acc, loss=loss))
    return rows


def plot_digital_hwa(comparison, results, digital_histories, history, out):
    """Separate software reference, HWA eval, and programmed retention results."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    labels = ["FP32 pretrained", METHOD_HWA, METHOD_DAR]
    for i, label in enumerate(labels):
        values = comparison.loc[comparison.stage == label, "accuracy"]
        axes[0].bar(i, values.mean(), color=["#555555", COLORS[METHOD_HWA], COLORS[METHOD_DAR]][i], alpha=.7)
        axes[0].scatter(np.full(len(values), i), values, color="black", s=14, zorder=3)
        if len(values)>1:
            axes[0].errorbar(i, values.mean(), yerr=values.std(ddof=1), color="black", capsize=4)
    axes[0].set_xticks(range(3), labels, rotation=15)
    axes[0].set_ylabel("Test accuracy [%]")
    axes[0].set_title("(a) FP32 pretrained vs HWA fine-tuned")
    baseline = comparison.loc[comparison.stage == "FP32 pretrained", "accuracy"].mean()
    axes[1].axhline(baseline, color="#555555", ls=":", label="FP32 pretrained (software)")
    for method, gdc, linestyle in [(METHOD_HWA, False, "-"), (METHOD_DAR, False, "-"),
                                    (METHOD_HWA, True, "--"), (METHOD_DAR, True, "--"),
                                    (METHOD_DS, False, "-")]:
        sub = results[(results.method==method)&(results.gdc==gdc)]
        if sub.empty: continue
        per_seed = sub.groupby(["fold","t_s"]).accuracy.mean().unstack("fold")
        mean, sd = per_seed.mean(axis=1), per_seed.std(axis=1)
        axes[1].plot(mean.index, mean, linestyle, color=COLORS[method],
                     label=method + (" + measured GDC" if gdc else ""))
        if per_seed.shape[1]>1:
            axes[1].fill_between(mean.index, mean-sd, mean+sd, color=COLORS[method], alpha=.1)
    axes[1].set_xscale("log"); axes[1].set_xlabel("Absolute retention time [s]")
    axes[1].set_ylabel("Test accuracy [%]"); axes[1].set_title("(b) Programmed PCM retention")
    axes[1].legend(fontsize=7)
    for ax in axes: ax.grid(alpha=.2)
    _save_figure(fig, Path(out)/"fig5_digital_vs_hwa",
        "Fig. 5. (a) FP32 pretrained and final HWA test accuracy before PCM programming; dots are training seeds, bars show mean +/- seed SD. "
        "(b) Programmed PCM accuracy versus time; shading is SD of hardware-averaged seed means. The software reference has no device drift.")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, frame, title in [(axes[0], digital_histories, "FP32 pretraining"),
                              (axes[1], history, "HWA fine-tuning")]:
        groups = [("FP32", frame)] if ax is axes[0] else frame.groupby("method")
        for label, data in groups:
            g = data.groupby("epoch")[["train_accuracy","val_accuracy"]].mean()
            ax.plot(g.index, g.train_accuracy, label=label+" train")
            ax.plot(g.index, g.val_accuracy, "--", label=label+" validation")
        ax.set_title(title); ax.set_xlabel("Stage epoch"); ax.set_ylabel("Accuracy [%]")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    _save_figure(fig, Path(out)/"fig6_digital_hwa_learning",
        "Fig. 6. Separate FP32 pretraining and HWA fine-tuning stages. Curves average training seeds; validation, not test, is evaluated each epoch.")


# CELL 18
PAPER_REFERENCE = pd.DataFrame([
    {"method": METHOD_HWA, "t_s": 20.0, "paper_error_pct": 4.82},
    {"method": METHOD_HWA, "t_s": 5*YEAR, "paper_error_pct": 9.14},
    {"method": METHOD_DAR, "t_s": 20.0, "paper_error_pct": 3.52},
    {"method": METHOD_DAR, "t_s": 5*YEAR, "paper_error_pct": 7.95},
])


def summarize_at_anchors(results: pd.DataFrame):
    anchors = [20.0, 5*DAY, YEAR, 5*YEAR]
    x = results[(~results.gdc) & (results.t_s.isin(anchors))].copy()
    return (
        x.groupby(["method", "t_s"])
        .error.agg(["mean", "std", "count"])
        .reset_index()
    )


def run_legacy_experiment(cfg: Config, base_dir: Path = Path(".")):
    cfg = cfg.resolved()
    device = check_environment(cfg)
    out, fingerprint = experiment_directory(cfg, base_dir)

    all_history = []
    digital_history_all = []
    digital_comparison = []
    all_results = []
    all_ablation = []
    cdf_for_figure = None

    for fold in range(cfg.train_repeats):
        train_loader, val_loader, test_loader, split_info = build_loaders(cfg, base_dir, fold)
        atomic_torch_save(split_info, out / f"fold{fold}_split.pt")

        # Paper IV-A: FP32 NON-HWA training, THEN transfer the same pretrained
        # checkpoint to both analog methods for controlled comparison.
        digital, digital_history = train_digital_pretrained(
            cfg, fold, train_loader, val_loader, device, out, fingerprint
        )
        hwa, hist_hwa = train_analog_method(
            METHOD_HWA, cfg, fold, digital, train_loader, val_loader, device, out, fingerprint
        )
        dar, hist_dar = train_analog_method(
            METHOD_DAR, cfg, fold, digital, train_loader, val_loader, device, out, fingerprint
        )
        digital_history_all.append(digital_history.assign(fold=fold))
        digital_comparison.extend(compare_digital_and_hwa(
            digital, hwa, dar, test_loader, device, cfg, fold))
        del digital

        all_history.extend([hist_hwa, hist_dar])

        hwa_snap = snapshot_analog_model(hwa)
        dar_snap = snapshot_analog_model(dar)

        del hwa, dar
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for realization in range(cfg.realizations):
            shard = out / f"fold{fold}_realization{realization}.pt"

            if shard.exists():
                pack = torch.load(shard, map_location="cpu", weights_only=False)
                if pack["fingerprint"] != fingerprint:
                    raise ValueError("Retention shard contract mismatch.")
                df = pack["results"]
                cdf_states = pack.get("cdf_states", {})
            else:
                df, cdf_states = retention_realization(
                    hwa_snap,
                    dar_snap,
                    cfg,
                    fold,
                    realization,
                    test_loader,
                    device,
                )
                atomic_torch_save(
                    {
                        "fingerprint": fingerprint,
                        "results": df,
                        "cdf_states": cdf_states,
                    },
                    shard,
                )

            all_results.append(df)

            if fold == 0 and realization == 0 and cdf_states:
                cdf_for_figure = cdf_states

            if cfg.run_bin_ablation:
                all_ablation.append(
                    driftslice_bin_ablation(
                        hwa_snap,
                        cfg,
                        fold,
                        realization,
                        val_loader,
                        device,
                        t_s=5*YEAR,
                    )
                )

            print(
                f"retention fold={fold+1}/{cfg.train_repeats} "
                f"realization={realization+1}/{cfg.realizations}"
            )

    history = pd.concat(all_history, ignore_index=True)
    results = pd.concat(all_results, ignore_index=True)
    ablation = (
        pd.concat(all_ablation, ignore_index=True)
        if all_ablation else pd.DataFrame()
    )

    history.to_csv(out / "training_history.csv", index=False)
    results.to_csv(out / "retention_results.csv", index=False)

    if not ablation.empty:
        ablation.to_csv(out / "driftslice_bin_ablation_validation.csv", index=False)

    summary = summarize_at_anchors(results)
    summary.to_csv(out / "summary.csv", index=False)
    # Important: early-stopped runs MUST NOT be reported as a completed paper protocol.
    if "paper_154_epochs_complete" in history.columns:
        eligibility = history.groupby(["method", "fold"], as_index=False).agg(
            last_epoch=("epoch", "max"),
            paper_154_epochs_complete=("paper_154_epochs_complete", "last"),
            best_val_accuracy=("best_val_accuracy", "max"),
            selected_epoch=("selected_epoch", "last"),
            HWA_trained=("selected_checkpoint_was_HWA_trained", "last"),
        )
        eligibility.to_csv(out / "training_protocol_status.csv", index=False)
        print("\nTraining protocol status:\n", eligibility.to_string(index=False))
    PAPER_REFERENCE.to_csv(out / "paper_reference_values.csv", index=False)

    comparison = pd.DataFrame(digital_comparison)
    comparison.to_csv(out / "digital_vs_hwa_test.csv", index=False)
    plot_digital_hwa(comparison, results, pd.concat(digital_history_all), history, out)
    plot_training(history, out)
    plot_retention(results, out, cfg)

    if cdf_for_figure is not None and len(cdf_for_figure) >= 3:
        plot_weight_cdfs(cdf_for_figure, out)

    if not ablation.empty:
        plot_bin_ablation(ablation, out)

    print("\nPaper reference values (paper's exact digital pretraining recipe is undisclosed):")
    print(PAPER_REFERENCE.to_string(index=False))
    print("\nObserved summary:")
    print(summary.to_string(index=False))

    return {
        "output_dir": out,
        "history": history,
        "results": results,
        "summary": summary,
        "ablation": ablation,
        "paper_reference": PAPER_REFERENCE,
    }


# CELL 20
def preflight(cfg: Config):
    cfg = cfg.resolved()
    device = check_environment(cfg)

    # DAR: zero at t/t0=1 and nonzero finite gradient later.
    w = torch.tensor([[0.2, -0.8, 0.4, -0.1]], device=device, requires_grad=True)
    ref = w.detach().clone()

    with random_stream(123, device):
        p0 = dar_penalty(w, ref, 0, 20.0, cfg)
    assert abs(float(p0.item())) < 1e-12

    with random_stream(124, device):
        p1 = dar_penalty(w, ref, 0, 60.0, cfg)
    grad = torch.autograd.grad(p1, w)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0

    # FP32 -> analog transfer must preserve weights AND bias, with no optimizer step.
    digital_fixture = DigitalMLP().to(device=device, dtype=torch.float32).eval()
    model = transfer_pretrained_digital_to_analog(
        cfg, digital_fixture, 999, device).eval()

    # Regression test for AIHWKit state_dict containing non-Tensor tile metadata.
    # Must pass before the expensive MNIST training is allowed to start.
    import io
    original_state = model.state_dict()
    saved_state = cpu_cloned_state(model)
    assert set(saved_state) == set(original_state), 'State clone lost entries'
    assert hasattr(saved_state, '_metadata') == hasattr(original_state, '_metadata'), (
        'State clone lost PyTorch OrderedDict metadata')
    for key, value in original_state.items():
        if isinstance(value, torch.Tensor):
            assert saved_state[key].device.type == 'cpu', (key, 'not CPU')
            assert saved_state[key].data_ptr() != value.data_ptr(), (
                key, 'checkpoint unexpectedly aliases live weights')
    test_input = torch.rand(8, 1, 28, 28, device=device)
    with torch.inference_mode(), random_stream(812, device):
        before_reload = model(test_input)
    checkpoint_buffer = io.BytesIO()
    torch.save({'model': saved_state}, checkpoint_buffer)
    checkpoint_buffer.seek(0)
    restored_state = torch.load(checkpoint_buffer, map_location='cpu', weights_only=False)
    model.load_state_dict(restored_state['model'], strict=True)
    with torch.inference_mode(), random_stream(812, device):
        after_reload = model(test_input)
    torch.testing.assert_close(before_reload, after_reload, rtol=3e-5, atol=3e-5)
    print('PASS: AIHWKit non-Tensor state_dict checkpoint clone/save/load')

    # Snapshot deployment should reproduce eval-mode forward with the same physical weights.
    snap = snapshot_analog_model(model)
    eff = {k: v["weight"] for k, v in snap.items()}
    dep = SnapshotMLP(snap, eff, cfg).to(device).eval()
    x = torch.rand(8, 1, 28, 28, device=device)

    with torch.inference_mode(), random_stream(77, device):
        y0 = model(x)
    with torch.inference_mode(), random_stream(77, device):
        y1 = dep(x)

    torch.testing.assert_close(y1, y0, atol=3e-5, rtol=3e-4)

    # Absolute 20 s maps to elapsed zero; absolute 40 s includes 20 s drift.
    pcm = PCMPhysicalMatrix(snap["fc1"]["weight"], cfg, 321)
    g0 = pcm.read_conductances(20.0, 555, noiseless=True)
    g20 = pcm.read_conductances(40.0, 555, noiseless=True)
    assert any((a-b).abs().max().item() > 0 for a, b in zip(g20, g0))

    # DriftSlice must remain finite.
    bank = PCMClassReferenceBank(cfg, 1, 777, device)
    gains = bank.gains(20.0, 888)
    assert torch.isfinite(gains).all()

    read = pcm.read_conductances(20.0, 999)
    corr = apply_driftslice(pcm, read, gains, bank.edges)
    w_corr = pcm.logical_weight(corr)
    assert torch.isfinite(w_corr).all()

    # Print source-literal lambda schedule: useful for diagnosing the paper reconstruction.
    schedule = []
    for epoch, t in enumerate(cfg.dar_times_s):
        time_term = t if cfg.dar_time_mode == "literal_seconds" else epoch + 1
        lam = cfg.dar_lambda_init / (epoch+1.0)**1.5 / (1.0 + cfg.dar_tc*time_term)
        schedule.append((epoch+1, t, lam))

    print("PASS: FP32 digital-to-analog weight transfer, DAR gradient, snapshot forward, PCM timing and DriftSlice finite-state.")
    print("DAR reconstruction assumptions:", {
        "eta": cfg.dar_eta,
        "delta_mode": cfg.dar_delta_mode,
        "g_units": cfg.dar_g_units,
        "time_mode": cfg.dar_time_mode,
    })
    print("First DAR-cycle lambda schedule:")
    print(pd.DataFrame(schedule, columns=["epoch_in_cycle", "t_s", "lambda_drift"]).to_string(index=False))




