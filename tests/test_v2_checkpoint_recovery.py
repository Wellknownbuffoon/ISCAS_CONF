"""Native AIHWKit CPU regression checks for the v2 checkpoint recovery cell.

Usage: python tests/test_v2_checkpoint_recovery.py --notebook uploaded.ipynb --out NEW_DIRECTORY
No Drive, downloads, user checkpoints, full run or CUDA execution is performed.
"""
import argparse
import contextlib
import copy
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_notebook_gradients import load_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    with contextlib.redirect_stdout(io.StringIO()):
        r = load_runtime(args.notebook, args.out / "runtime.py")
    source = Path(__file__).resolve().parents[1] / "v2_checkpoint_recovery.py"
    spec = importlib.util.spec_from_file_location("v2_checkpoint_recovery", source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    dev = torch.device("cpu")
    cfg = replace(r.Config(), require_cuda=False, fast_mode=True, strict_154_epochs=False,
                  epochs=2, digital_epochs=1, train_repeats=1, batch_size=2,
                  test_batch_size=2, validation_reads=1)
    gen = torch.Generator().manual_seed(181)
    trainset = TensorDataset(torch.randn(4, 3, 32, 32, generator=gen), torch.tensor([1, 2, 3, 4]))
    valset = TensorDataset(torch.randn(4, 3, 32, 32, generator=gen), torch.tensor([5, 6, 7, 8]))
    split = {"train_indices": torch.arange(4), "val_indices": torch.arange(4, 8),
             "test_indices": torch.arange(4)}

    def loaders(*args, **kwargs):
        train = DataLoader(trainset, batch_size=2, shuffle=True,
                           generator=torch.Generator().manual_seed(182))
        train.split_name = "train"
        val = DataLoader(valset, batch_size=2)
        val.split_name = "val"
        return train, val, val, split

    original_ops = r.ops
    r.ops = lambda exp: SimpleNamespace(loaders=loaders)
    data_hashes = {"synthetic_fixture": "fixed-v1"}
    train_fp = r.compute_fingerprints_resnet(cfg, data_hashes, r.RESNET_SPEC)["train"]
    digital = r.make_digital_resnet32(71).eval()
    digital_state = r.cpu_cloned_state(digital)
    identity = r.state_dict_hash(digital_state)
    fixture = args.out / "fixture" / "fold0"
    train, val, _, _ = loaders()
    with (args.out / "native_training.log").open("w") as log, contextlib.redirect_stdout(log):
        r.train_hwa_resnet(r.HWA, cfg, 0, digital, identity, train, val, dev, fixture, train_fp)
    r.atomic_torch_save({"kind": "digital_best_validation", "train_fp": train_fp, "fold": 0,
                         "model": digital_state, "model_identity": identity}, fixture / "digital_best_validation.pt")
    r.atomic_torch_save({**split, "digest": r.split_digest(split)}, fixture / "split.pt")
    original_best = r.torch_load(fixture / f"{r.HWA}_best_validation.pt")
    original_final = r.torch_load(fixture / f"{r.HWA}_final.pt")

    def experiment(name, best=None, final=None):
        root = args.out / name
        fdir = root / "training" / "fold0"
        fdir.mkdir(parents=True)
        for filename in ("digital_best_validation.pt", "split.pt"):
            shutil.copy2(fixture / filename, fdir / filename)
        r.atomic_torch_save(best if best is not None else original_best, fdir / f"{r.HWA}_best_validation.pt")
        r.atomic_torch_save(final if final is not None else original_final, fdir / f"{r.HWA}_final.pt")
        exp = SimpleNamespace(arch="resnet32", cfg=cfg, device=dev, run_dir=root,
                              train_dir=root / "training", base_dir=root, fp={"train": train_fp},
                              data_hashes=data_hashes)
        return exp, fdir / f"{r.HWA}_best_validation.pt"

    results = {"scope": "Native AIHWKit 1.1.0 synthetic CPU; no user-model/GPU claim"}
    before_fp = r.compute_fingerprints_resnet(cfg, {"fixture": "same"}, r.RESNET_SPEC)
    exp, path = experiment("read_only")
    before = path.read_bytes()
    report = helper.check_v2_checkpoint(vars(r), exp)
    assert report["status"] == "PASS" and path.read_bytes() == before
    assert len(report["validations"]) == 3 and report["same_seed_repeat_span_pp"] == 0
    results["native_full_state_load_and_three_seeded_validations"] = True

    stale = copy.deepcopy(original_best)
    stale["restore_abs_diff_pp"] = 0.2266666667
    stale["restore_reeval_val_accuracy"] = stale["val_accuracy"] + 0.2266666667
    exp, path = experiment("metadata_repair", stale)
    before = path.read_bytes()
    report = helper.check_v2_checkpoint(vars(r), exp, repair=True)
    fixed = r.torch_load(path)
    assert report["status"] == "REPAIRED_STALE_METADATA"
    assert Path(report["backup"]).read_bytes() == before
    assert fixed["restore_abs_diff_pp"] == 0
    assert r.state_dict_hash(fixed["model"]) == original_best["model_identity"]
    for key in set(stale) - {"restore_abs_diff_pp", "restore_reeval_val_accuracy"}:
        assert r.state_dict_hash({key: stale[key]}) == r.state_dict_hash({key: fixed[key]})
    results["stale_metadata_repaired_only_after_validation_with_exact_byte_backup"] = True

    # A real mismatch cannot be repaired by changing the claimed target score.
    mismatch = copy.deepcopy(stale)
    selected = mismatch["selected_best_validation_epoch"]
    mismatch["val_accuracy"] += 10
    mismatch["history"][selected - 1]["val_accuracy"] += 10
    mismatch_final = copy.deepcopy(original_final)
    mismatch_final["history"] = copy.deepcopy(mismatch["history"])
    exp, path = experiment("persistent_mismatch", mismatch, mismatch_final)
    before = path.read_bytes()
    try:
        helper.check_v2_checkpoint(vars(r), exp, repair=True)
    except r.ScientificCheckError as exc:
        assert "mismatch persists" in str(exc)
    else:
        raise AssertionError("A real mismatch was accepted")
    assert path.read_bytes() == before and not list(path.parent.glob("*.bak"))
    results["real_validation_mismatch_stops_without_modifying_checkpoint"] = True

    corrupt = copy.deepcopy(stale)
    key = next(k for k, v in corrupt["model"].items() if torch.is_tensor(v) and v.is_floating_point())
    corrupt["model"][key].add_(0.01)
    exp, path = experiment("identity_mismatch", corrupt)
    before = path.read_bytes()
    try:
        helper.check_v2_checkpoint(vars(r), exp, repair=True)
    except r.ScientificCheckError as exc:
        assert "identity" in str(exc)
    else:
        raise AssertionError("Corrupt model was accepted")
    assert path.read_bytes() == before
    results["corrupt_model_identity_stops_without_modifying_checkpoint"] = True

    wrong_rpu = copy.deepcopy(stale)
    analog_key = next(k for k, v in wrong_rpu["model"].items()
                      if isinstance(v, dict) and "rpu_config" in v)
    wrong_rpu["model"][analog_key]["rpu_config"].forward.out_noise += 0.01
    exp, path = experiment("wrong_rpu", wrong_rpu)
    before = path.read_bytes()
    try:
        helper.check_v2_checkpoint(vars(r), exp, repair=True)
    except r.ScientificCheckError as exc:
        assert "RPU configuration differs" in str(exc)
    else:
        raise AssertionError("Changed RPU configuration was accepted")
    assert path.read_bytes() == before
    results["changed_rpu_configuration_stops_without_modifying_checkpoint"] = True

    exp, path = experiment("wrong_fingerprint", stale)
    before = path.read_bytes()
    try:
        helper.check_v2_checkpoint(vars(r), exp, repair=True, expected_train_prefix="1cb6defff3c3")
    except r.ScientificCheckError as exc:
        assert "fingerprint differs" in str(exc)
    else:
        raise AssertionError("Wrong experiment was accepted")
    assert path.read_bytes() == before
    results["wrong_experiment_stops_without_modifying_checkpoint"] = True
    after_fp = r.compute_fingerprints_resnet(cfg, {"fixture": "same"}, r.RESNET_SPEC)
    assert before_fp == after_fp
    r.ops = original_ops
    results["scientific_fingerprints_unchanged"] = True
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
