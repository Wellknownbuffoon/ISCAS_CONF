# Paste this ENTIRE cell at the bottom of your initialized P4 notebook.
# It reads saved HWA/NSR checkpoints and TRAIN data. No training or retention run.
# No functions/classes are defined, so notebook_code_hash is not extended.
import itertools as _diag_it
import json as _diag_json
import math as _diag_math
import zipfile as _diag_zipfile
from dataclasses import replace as _diag_replace
from pathlib import Path as _diag_Path
import pandas as _diag_pd
import torch as _diag_torch

_diag_exp = globals().get("FULL_EXP")
if _diag_exp is None:
    _diag_exp = open_full_experiment()
if _diag_exp.arch != "resnet32":
    raise RuntimeError("This diagnostic is for the ResNet-32 experiment.")
_diag_cfg, _diag_dev = _diag_exp.cfg, _diag_exp.device
_diag_batches = int(globals().get("NSR_DIAGNOSTIC_BATCHES", 5))
if _diag_batches < 1:
    raise ValueError("NSR_DIAGNOSTIC_BATCHES must be positive.")
_diag_out = _diag_Path(globals().get("NSR_DIAGNOSTIC_OUTPUT", "/content/nsr_checkpoint_diagnostics"))
_diag_out.mkdir(parents=True, exist_ok=True)
_diag_paths = [_diag_exp.train_dir / f"fold{f}" / f"{m}_best_validation.pt"
               for f in range(_diag_cfg.train_repeats) for m in (HWA, NSR)]
for _diag_path in _diag_paths:
    if not _diag_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {_diag_path}")
_diag_rng = rng_snapshot(_diag_dev)
_diag_rows, _diag_bn_rows, _diag_pack_rows, _diag_layer_rows = [], [], [], []
try:
    for _diag_fold in range(_diag_cfg.train_repeats):
        # Independent loaders; preserve existing notebook loader/generator state.
        # Disable data downloads in this local copy of Config only.
        _diag_train, _, _, _ = build_cifar10_loaders(
            _diag_replace(_diag_cfg, allow_download=False),
            _diag_exp.base_dir, _diag_fold)
        _diag_samples = [(x.clone(), y.clone()) for x, y in
                         _diag_it.islice(_diag_train, _diag_batches)]
        if len(_diag_samples) != _diag_batches:
            raise RuntimeError("Not enough training batches for this diagnostic.")
        _diag_parent = None
        for _diag_method in (HWA, NSR):
            _diag_path = _diag_exp.train_dir / f"fold{_diag_fold}" / f"{_diag_method}_best_validation.pt"
            _diag_pack = torch_load(_diag_path)
            verify_pack(_diag_pack, _diag_exp.fp["train"], "hwa_best_validation", _diag_fold, _diag_path)
            if _diag_pack.get("method") != _diag_method or _diag_pack.get("analog_epochs_completed") != _diag_cfg.epochs:
                raise RuntimeError(f"Incomplete or wrong-method checkpoint: {_diag_path}")
            if state_dict_hash(_diag_pack["model"]) != _diag_pack.get("model_identity"):
                raise RuntimeError(f"Checkpoint content hash mismatch: {_diag_path}")
            if not _diag_pack.get("digital_identity"):
                raise RuntimeError("Missing pretrained-parent identity.")
            if _diag_parent is not None and _diag_parent != _diag_pack["digital_identity"]:
                raise RuntimeError("HWA and NSR do not share the same pretrained model.")
            _diag_parent = _diag_pack["digital_identity"]
            _diag_model = load_analog_resnet_from_pack(_diag_cfg, _diag_pack, _diag_dev)
            # Check loaded BatchNorm tensors exactly before any diagnostic forward.
            _diag_loaded = _diag_model.state_dict()
            for _diag_key in batchnorm_state_keys(_diag_model):
                _diag_torch.testing.assert_close(_diag_loaded[_diag_key].cpu(),
                    _diag_pack["model"][_diag_key].cpu(), rtol=0, atol=0)
            for _diag_name, _diag_bn in _diag_model.named_modules():
                if isinstance(_diag_bn, _diag_torch.nn.BatchNorm2d):
                    _diag_finite = all(bool(_diag_torch.isfinite(t).all()) for t in
                        (_diag_bn.running_mean, _diag_bn.running_var, _diag_bn.weight, _diag_bn.bias))
                    if not _diag_finite or bool((_diag_bn.running_var < 0).any()):
                        raise RuntimeError(f"Invalid BatchNorm state: {_diag_name}")
                    _diag_bn_rows.append(dict(fold=_diag_fold, method=_diag_method, layer=_diag_name,
                        loaded_exactly=True, finite=True,
                        running_var_min=float(_diag_bn.running_var.min()),
                        running_var_max=float(_diag_bn.running_var.max()),
                        running_mean_absmax=float(_diag_bn.running_mean.abs().max()),
                        affine_weight_absmax=float(_diag_bn.weight.detach().abs().max()),
                        num_batches_tracked=int(_diag_bn.num_batches_tracked)))
            _diag_pack_rows.append(dict(fold=_diag_fold, method=_diag_method,
                model_identity=_diag_pack["model_identity"], digital_identity=_diag_parent,
                content_hash_valid=True, epochs=_diag_pack["analog_epochs_completed"],
                best_epoch=_diag_pack["selected_best_validation_epoch"],
                saved_val_accuracy=_diag_pack.get("val_accuracy"),
                recorded_restore_diff_pp=_diag_pack.get("restore_abs_diff_pp")))
            _diag_model.train()  # Training-mode diagnostic on a disposable model copy.
            _diag_params = [p for p in _diag_model.parameters() if p.requires_grad]
            _diag_weight_indices = {name: next(i for i, p in enumerate(_diag_params)
                if p is physical_weight_parameter(layer))
                for name, layer in _diag_model.layers().items()}
            for _diag_batch, (_diag_x_cpu, _diag_y_cpu) in enumerate(_diag_samples):
                with random_stream(900_000_000 + _diag_cfg.seed + _diag_fold*1000 + _diag_batch, _diag_dev):
                    _diag_x = cifar_augment(_diag_x_cpu.to(_diag_dev),
                        augment_seed(_diag_cfg, _diag_fold, 1, 0, _diag_batch)).flatten(1)
                    _diag_y = _diag_y_cpu.to(_diag_dev)
                    _diag_ce = _diag_torch.nn.functional.cross_entropy(_diag_model(_diag_x), _diag_y)
                    # Probe the SAME NSR objective at both checkpoints.
                    # HWA did not use this term during its original training.
                    _diag_penalty = nsr_penalty(_diag_model, _diag_cfg)
                    _diag_ce_grads = _diag_torch.autograd.grad(_diag_ce, _diag_params,
                        retain_graph=True, allow_unused=True)
                    _diag_nsr_grads = _diag_torch.autograd.grad(_diag_penalty, _diag_params,
                        allow_unused=True)
                _diag_vectors = []
                for _diag_grads in (_diag_ce_grads, _diag_nsr_grads):
                    _diag_vectors.append(_diag_torch.cat([
                        (_diag_torch.zeros_like(p) if g is None else g).detach().cpu().double().flatten()
                        for p, g in zip(_diag_params, _diag_grads)]))
                _diag_c, _diag_r = _diag_vectors
                _diag_nc, _diag_nr = float(_diag_c.norm()), float(_diag_r.norm())
                _diag_total = float((_diag_c+_diag_r).norm())
                _diag_finite = all(_diag_math.isfinite(v) for v in
                    (float(_diag_ce.detach()), float(_diag_penalty.detach()), _diag_nc, _diag_nr, _diag_total))
                if not _diag_finite:
                    raise RuntimeError("Nonfinite diagnostic loss or gradient.")
                _diag_rows.append(dict(fold=_diag_fold, method=_diag_method, batch=_diag_batch,
                    ce=float(_diag_ce.detach()), weighted_nsr_penalty=float(_diag_penalty.detach()),
                    ce_grad_norm=_diag_nc, nsr_grad_norm=_diag_nr,
                    nsr_to_ce_grad_ratio=_diag_nr/_diag_nc if _diag_nc else float("nan"),
                    gradient_cosine=float(_diag_torch.dot(_diag_c, _diag_r))/(_diag_nc*_diag_nr)
                        if _diag_nc and _diag_nr else float("nan"),
                    hypothetical_combined_grad_norm=_diag_total,
                    would_clip_ce=_diag_nc > _diag_cfg.grad_clip,
                    would_clip_ce_plus_nsr=_diag_total > _diag_cfg.grad_clip,
                    clip_threshold=float(_diag_cfg.grad_clip)))
                for _diag_name, _diag_idx in _diag_weight_indices.items():
                    _diag_wc = _diag_ce_grads[_diag_idx]
                    _diag_wr = _diag_nsr_grads[_diag_idx]
                    _diag_wc = (_diag_torch.zeros_like(_diag_params[_diag_idx]) if _diag_wc is None else _diag_wc).detach().cpu().double().flatten()
                    _diag_wr = (_diag_torch.zeros_like(_diag_params[_diag_idx]) if _diag_wr is None else _diag_wr).detach().cpu().double().flatten()
                    _diag_wcn, _diag_wrn = float(_diag_wc.norm()), float(_diag_wr.norm())
                    _diag_layer_rows.append(dict(fold=_diag_fold, method=_diag_method, batch=_diag_batch,
                        layer=_diag_name, ce_grad_norm=_diag_wcn, nsr_grad_norm=_diag_wrn,
                        nsr_to_ce_grad_ratio=_diag_wrn/_diag_wcn if _diag_wcn else float("nan"),
                        gradient_cosine=float(_diag_torch.dot(_diag_wc, _diag_wr))/(_diag_wcn*_diag_wrn)
                            if _diag_wcn and _diag_wrn else float("nan")))
                print(f"fold{_diag_fold} {_diag_method} batch {_diag_batch+1}/{_diag_batches}: "
                      f"NSR/CE gradient norm={_diag_rows[-1]['nsr_to_ce_grad_ratio']:.3g}; "
                      f"cosine={_diag_rows[-1]['gradient_cosine']:.3f}")
            del _diag_model, _diag_params, _diag_ce_grads, _diag_nsr_grads, _diag_loaded, _diag_pack
            if _diag_dev.type == "cuda":
                _diag_torch.cuda.empty_cache()
finally:
    rng_restore(_diag_rng, _diag_dev)

_diag_table = _diag_pd.DataFrame(_diag_rows)
_diag_table.to_csv(_diag_out / "gradient_batches.csv", index=False)
_diag_pd.DataFrame(_diag_bn_rows).to_csv(_diag_out / "batchnorm_state.csv", index=False)
_diag_pd.DataFrame(_diag_pack_rows).to_csv(_diag_out / "checkpoint_metadata.csv", index=False)
_diag_pd.DataFrame(_diag_layer_rows).to_csv(_diag_out / "gradient_layers.csv", index=False)
_diag_summary = _diag_table.groupby(["fold", "method"])[[
    "ce_grad_norm", "nsr_grad_norm", "nsr_to_ce_grad_ratio", "gradient_cosine",
    "would_clip_ce", "would_clip_ce_plus_nsr"]].mean()
_diag_summary.to_csv(_diag_out / "gradient_summary.csv")
(_diag_out / "scope.json").write_text(_diag_json.dumps(dict(
    scope="Training-mode probes of saved best-validation checkpoints; not historical training replay",
    batches_per_fold=_diag_batches, nsr_lambda=_diag_cfg.nsr_lambda,
    train_fingerprint=_diag_exp.fp["train"], device=str(_diag_dev),
    torch=_diag_torch.__version__,
    limitations=["A negative cosine does not by itself imply a harmful regularizer.",
        "Gradient norms are before clipping and Adam rescaling; they are not update magnitudes.",
        "BN restoration and finite statistics do not establish well-calibrated BN under drift.",
        "This does not reproduce GDC calibration or evaluate retention accuracy."]
), indent=2))
_diag_zip_path = _diag_out.parent / "NSR_checkpoint_diagnostics.zip"
with _diag_zipfile.ZipFile(_diag_zip_path, "w", _diag_zipfile.ZIP_DEFLATED) as _diag_archive:
    for _diag_path in sorted(_diag_out.glob("*.csv")):
        _diag_archive.write(_diag_path, _diag_path.name)
    _diag_archive.write(_diag_out / "scope.json", "scope.json")
print("\n", _diag_summary.to_string())
print("\nDiagnostic ZIP:", _diag_zip_path)
if not globals().get("NSR_DIAGNOSTIC_NO_DOWNLOAD", False):
    from google.colab import files as _diag_colab_files
    _diag_colab_files.download(str(_diag_zip_path))
