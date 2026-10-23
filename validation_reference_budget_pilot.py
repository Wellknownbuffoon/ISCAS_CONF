"""Compare two DriftSlice reference budgets on validation data, using saved models.

No training and no official-test evaluation. The default is one fold, three device
realizations, B=1, unchanged K, and two timestamps. Results are exploratory.
"""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import zipfile

import numpy as np
import pandas as pd


def run_pilot(r, exp=None, folds=(0,), realizations=3, candidate_budget=256, download=True):
    if exp is None:
        exp = r.get("FULL_EXP") or r.get("COORD_EXP")
        if exp is None:
            exp = r["open_full_experiment"](role="worker")
    cfg, device = exp.cfg, exp.device
    if exp.arch != "resnet32" or cfg.fast_mode:
        raise RuntimeError("Use the initialized FULL ResNet-32 notebook, not a smoke experiment.")
    folds = tuple(int(f) for f in folds)
    if not folds or len(set(folds)) != len(folds) or any(f < 0 or f >= cfg.train_repeats for f in folds):
        raise ValueError("Invalid training fold selection.")
    if not 1 <= realizations <= cfg.realizations:
        raise ValueError("Invalid number of programmed-device realizations.")
    baseline_budget = cfg.reference_cell_budget
    if candidate_budget <= baseline_budget or int(candidate_budget) != candidate_budget:
        raise ValueError("Candidate reference-cell budget must be an integer larger than the baseline.")
    budgets = (baseline_budget, int(candidate_budget))
    methods = (r["HWA"], r["NSR"])
    times = np.asarray(r["exact_retention_times"](cfg), dtype=float)
    indices = (0, len(times)-1)
    checkpoints, sources = {}, []
    for fold in folds:
        parent = None
        for method in methods:
            path = exp.train_dir/f"fold{fold}"/f"{method}_best_validation.pt"
            if not path.exists():
                raise FileNotFoundError(f"Completed checkpoint missing: {path}")
            pack = r["torch_load"](path)
            r["verify_pack"](pack, exp.fp["train"], "hwa_best_validation", fold, path)
            if pack.get("method") != method or pack.get("analog_epochs_completed") != cfg.epochs:
                raise RuntimeError(f"Wrong method or incomplete training: {path}")
            if r["state_dict_hash"](pack["model"]) != pack.get("model_identity"):
                raise RuntimeError(f"Content hash mismatch: {path}")
            if not pack.get("digital_identity") or (parent is not None and parent != pack["digital_identity"]):
                raise RuntimeError("HWA and NSR must share the same pretrained model in a fold.")
            parent = pack["digital_identity"]
            checkpoints[fold, method] = pack
            sources.append(dict(fold=fold, method=method, path=str(path),
                model_identity=pack["model_identity"], digital_identity=parent))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = exp.run_dir/"validation_reference_budget_pilot"/stamp
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    total = len(folds)*len(methods)*realizations*len(indices)*len(budgets)
    saved_rng = r["rng_snapshot"](device)
    started = time.monotonic()
    try:
        for fold in folds:
            # Independent loaders; no new split and no change to an existing generator.
            # The native loader may cache official test tensors, but we never evaluate them.
            _, val, _, _ = r["build_cifar10_loaders"](replace(cfg, allow_download=False), exp.base_dir, fold)
            val.split_name = "val"
            r["require_split"](val, "val", "reference-budget validation pilot")
            for method in methods:
                model = r["load_analog_resnet_from_pack"](cfg, checkpoints[fold, method], device)
                snap = r["snapshot_analog_resnet"](model)
                for realization in range(realizations):
                    root = cfg.seed+fold*1_000_000+realization*10_000
                    main = r["build_main_pcm_all"](snap, cfg, root+10)
                    before = r["pcm_state_hash_all"](main)
                    banks = {budget:r["PCMClassReferenceBank"](
                        replace(cfg, reference_cell_budget=budget), 1, root+30, device) for budget in budgets}
                    if any(bank.target.numel() != budget for budget, bank in banks.items()):
                        raise RuntimeError("Reference budget mismatch.")
                    for ti in indices:
                        ts = float(times[ti])
                        _, conductances = r["plain_effective_weights_all"](main, ts, root+100_000+ti*100)
                        for budget, bank in banks.items():
                            corrected, gains = r["driftslice_effective_weights_all"](
                                main, conductances, bank, ts, root+500_000+ti*101)
                            with r["random_stream"](root+700_000+ti*31, device):
                                loss, accuracy = r["evaluate_deployed"](model, corrected, val, device)
                            if not np.isfinite([loss, accuracy]).all() or not 0 <= accuracy <= 100:
                                raise FloatingPointError("Invalid validation loss/accuracy.")
                            rows.append(dict(fold=fold, method=method, realization=realization,
                                t_index=ti, t_s=ts, reference_cell_budget=budget, bins=1,
                                reference_read_repeats=cfg.reference_read_repeats, split="val",
                                accuracy=accuracy, error=100-accuracy, loss=loss, gain=float(gains[0]),
                                main_programming_root=root+10, read_seed=root+100_000+ti*100))
                            pd.DataFrame(rows).to_csv(out/"validation_records.csv", index=False)
                            elapsed = time.monotonic()-started
                            remaining = elapsed/len(rows)*(total-len(rows))
                            print(f"VALIDATION PILOT {len(rows)}/{total}: fold{fold} {method} "
                                  f"R={budget} t={ts:g}s accuracy={accuracy:.3f}%; "
                                  f"rough remaining {remaining/60:.1f} min", flush=True)
                    if before != r["pcm_state_hash_all"](main):
                        raise RuntimeError("Persistent device state changed during the pilot.")
                    del main, banks, corrected, conductances
                del model, snap
                if device.type == "cuda":
                    r["torch"].cuda.empty_cache()
    finally:
        r["rng_restore"](saved_rng, device)
    data = pd.DataFrame(rows)
    foldmeans = data.groupby(["fold", "method", "reference_cell_budget", "t_index", "t_s"]).accuracy.mean().reset_index()
    foldmeans.to_csv(out/"validation_fold_means.csv", index=False)
    summary = foldmeans.groupby(["method", "reference_cell_budget", "t_index", "t_s"]).accuracy.agg(
        mean_accuracy_pct="mean", sd_across_fold_means_pp="std", n_training_folds="count").reset_index()
    summary.to_csv(out/"validation_summary.csv", index=False)
    paired = data.pivot(index=["fold", "method", "realization", "t_index", "t_s"],
                        columns="reference_cell_budget", values="accuracy")
    paired["candidate_minus_baseline_accuracy_pp"] = paired[candidate_budget]-paired[baseline_budget]
    paired.reset_index().to_csv(out/"paired_reference_budget_changes.csv", index=False)
    manifest = dict(status="EXPLORATORY VALIDATION PILOT; not final paper test results",
        target="Investigate whether increasing reference budget improves retained validation accuracy",
        target_accuracy_pct=90, folds=list(folds), realizations_per_fold=realizations,
        budgets=list(budgets), bins=1, reference_read_repeats=cfg.reference_read_repeats,
        times_s=[float(times[i]) for i in indices], original_config=asdict(cfg),
        original_fingerprints=exp.fp, checkpoint_sources=sources,
        new_training_executed=False, official_test_evaluated=False,
        paired_main_devices_and_mvm_seeds=True,
        limitations=["One training fold cannot establish generalization across independent trained models.",
                     "Three device realizations are a screen, not the original 30-realization protocol.",
                     "A validation accuracy of 90% does not guarantee test accuracy of 90%.",
                     "More reference cells have a hardware cost and cannot eliminate main-array device errors."])
    (out/"pilot_scope.json").write_text(json.dumps(manifest, indent=2, default=str))
    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    print("\nValidation means:\n", summary.to_string(index=False))
    print("\nPILOT ONLY:", zip_path)
    if download:
        from google.colab import files
        files.download(str(zip_path))
    return out
