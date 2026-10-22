"""Export existing coordinator results; never execute training, retention or ablation.

Load in a separate namespace and call run_export(dict(globals())) from P0.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ExportError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ExportError(message)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def validate_history(pack, epochs, method, fold):
    h = pd.DataFrame(pack["history"])
    require(len(h) == epochs and np.array_equal(h.epoch, np.arange(1, epochs+1)),
            f"{method} fold{fold}: epoch history incomplete or duplicated")
    require(h.fold.eq(fold).all(), f"{method}: history fold mismatch")
    if method != "FP32-Digital":
        require(h.method.eq(method).all(), f"{method}: history method mismatch")
    val_loss = "val_ce" if method == "FP32-Digital" else "val_loss"
    for field in ("train_accuracy", "val_accuracy", val_loss, "optimizer_steps"):
        require(np.isfinite(h[field]).all(), f"{method}: nonfinite {field}")
    require(h.optimizer_steps.is_monotonic_increasing and (h.optimizer_steps.diff().dropna() > 0).all(),
            f"{method}: optimizer step counts are not increasing")
    require(int(pack["optimizer_steps"]) == int(h.optimizer_steps.iloc[-1]),
            f"{method}: checkpoint/history optimizer steps differ")
    if method != "FP32-Digital":
        require(h.optimizer.eq("AnalogAdam").all(), f"{method}: optimizer is not AnalogAdam")
        require((h.train_extra.eq(0) if method == "PCM-HWA" else np.isfinite(h.train_extra)).all(),
                f"{method}: invalid extra loss")
    selected = int(pack["best_epoch"] if method == "FP32-Digital" else pack["selected_best_validation_epoch"])
    # Digital ResNet keeps the FIRST maximum; HWA additionally breaks ties by loss.
    best = (h.loc[h.val_accuracy.idxmax()] if method == "FP32-Digital" else
            h.sort_values(["val_accuracy", "val_loss", "epoch"], ascending=[False, True, True]).iloc[0])
    require(selected == int(best.epoch), f"{method}: saved best epoch disagrees with validation selection")
    return h, selected


def load_checkpoints(r, exp):
    cfg = exp.cfg
    digital, analog, selections, transfers, identities = [], [], [], [], {}
    needed = []
    for fold in range(cfg.train_repeats):
        folder = exp.train_dir / f"fold{fold}"
        needed.append(folder / "digital_best_validation.pt")
        needed.extend(folder / (f"{m}_analog_checkpoint.pt" if m == r["DIRECT"] else f"{m}_best_validation.pt")
                      for m in exp.models)
    missing = [str(p) for p in needed if not p.is_file()]
    require(not missing, "Training is incomplete; no paper export generated. Missing:\n" + "\n".join(missing))
    for fold in range(cfg.train_repeats):
        folder = exp.train_dir / f"fold{fold}"
        path = folder / "digital_best_validation.pt"
        d = r["torch_load"](path)
        r["verify_pack"](d, exp.fp["train"], "digital_best_validation", fold, path)
        require(d.get("digital_epochs_completed") == cfg.digital_epochs, "Digital epoch count mismatch")
        require(r["state_dict_hash"](d["model"]) == d["model_identity"], f"Content hash mismatch: {path}")
        h, selected = validate_history(d, cfg.digital_epochs, "FP32-Digital", fold)
        digital.append(h)
        selections.append(dict(fold=fold, method="FP32-Digital", completed_epochs=cfg.digital_epochs,
            selected_epoch=selected, best_val_accuracy=d["best_val_accuracy"],
            optimizer_steps=d["optimizer_steps"], checkpoint_identity=d["model_identity"],
            selection_split="validation", checkpoint_path=str(path)))
        parent = d["model_identity"]
        del d
        for method in exp.models:
            direct = method == r["DIRECT"]
            path = folder / (f"{method}_analog_checkpoint.pt" if direct else f"{method}_best_validation.pt")
            p = r["torch_load"](path)
            r["verify_pack"](p, exp.fp["train"], "fp32_direct" if direct else "hwa_best_validation", fold, path)
            require(p.get("method") == method, f"Wrong method: {path}")
            require(r["state_dict_hash"](p["model"]) == p["model_identity"], f"Content hash mismatch: {path}")
            require(p.get("digital_identity") == parent, f"Wrong pretrained parent: {path}")
            identities[fold, method] = p["model_identity"]
            if direct:
                require(p.get("optimizer_steps") == 0, "FP32-Direct has HWA optimizer steps")
                selected, completed, val = 0, 0, np.nan
            else:
                require(p.get("analog_epochs_completed") == cfg.epochs, f"Incomplete HWA epochs: {path}")
                h, selected = validate_history(p, cfg.epochs, method, fold)
                analog.append(h)
                completed, val = cfg.epochs, p["val_accuracy"]
            selections.append(dict(fold=fold, method=method, completed_epochs=completed,
                selected_epoch=selected, best_val_accuracy=val, optimizer_steps=p["optimizer_steps"],
                checkpoint_identity=p["model_identity"], selection_split="none" if direct else "validation",
                checkpoint_path=str(path)))
            transfers.extend(dict(fold=fold, method=method, **item) for item in p.get("transfer_errors", []))
            del p
    return pd.concat(digital, ignore_index=True), pd.concat(analog, ignore_index=True), pd.DataFrame(selections), pd.DataFrame(transfers), identities


def load_bins(r, exp, identities):
    bins, records = {}, []
    for fold in range(exp.cfg.train_repeats):
        path = exp.ret_dir / f"fold{fold}" / "selected_bins.json"
        require(path.is_file(), f"Missing validation selection: {path}")
        rec = json.loads(path.read_text())
        require(rec.get("fold") == fold and rec.get("selection_method") == r["HWA"],
                f"B must be selected on the same HWA validation checkpoint: {path}")
        require(rec.get("selection_model_identity") == identities[fold, r["HWA"]],
                f"Selected B refers to a different HWA checkpoint: {path}")
        require(rec.get("selection") == "validation_only_5yr", f"Selection split not verified: {path}")
        require(rec.get("bins") in exp.cfg.ablation_bins, f"Invalid selected B: {path}")
        bins[fold] = int(rec["bins"])
        records.append(rec)
    return bins, records


def load_retention(r, exp, identities, bins):
    cfg, times = exp.cfg, np.asarray(r["exact_retention_times"](exp.cfg), dtype=float)
    comps = list(r["compensations_for"](cfg))
    expected_paths = {exp.ret_dir / f"fold{f}" / m / f"real{k:02d}.pt": (f, m, k)
        for f in range(cfg.train_repeats) for m in exp.models for k in range(cfg.realizations)}
    missing = [str(p.relative_to(exp.ret_dir)) for p in expected_paths if not p.is_file()]
    require(not missing, f"Retention is incomplete: {len(missing)} missing shards. No inference was run.\n"
            + "\n".join(missing[:40]) + ("\n..." if len(missing) > 40 else ""))
    unexpected = [str(p) for p in exp.ret_dir.glob("fold*/*/real*.pt")
                  if p.parent.name in exp.models and p not in expected_paths]
    require(not unexpected, "Unexpected retention shard filenames:\n" + "\n".join(unexpected))
    frames, sources = [], []
    local_grid = pd.MultiIndex.from_product([comps, range(len(times))], names=["compensation", "t_index"])
    for path, (fold, method, realization) in expected_paths.items():
        pack = r["torch_load"](path)
        require(pack.get("fp") == exp.fp["retention"], f"Foreign fingerprint: {path}")
        require(pack.get("model_identity") == identities[fold, method], f"Foreign model identity: {path}")
        require(pack.get("selected_bins") == bins[fold], f"Foreign selected B: {path}")
        d, meta = pack["results"].copy(), pack["meta"]
        require(meta.get("state_hash_start") == meta.get("state_hash_end") and bool(meta.get("state_hash_start")),
                f"Persistent PCM state check missing/failed: {path}")
        require(meta.get("root_seed") == cfg.seed+fold*1_000_000+realization*10_000,
                f"Wrong realization seed: {path}")
        require(d.fold.eq(fold).all() and d.method.eq(method).all() and d.realization.eq(realization).all(),
                f"Shard path and record identities differ: {path}")
        require(d.split.eq("test").all() and d.selected_bins.eq(bins[fold]).all(),
                f"Wrong split or selected B: {path}")
        index = pd.MultiIndex.from_frame(d[["compensation", "t_index"]])
        require(not index.has_duplicates and len(index) == len(local_grid)
                and len(local_grid.difference(index)) == 0, f"Missing/duplicate retention records: {path}")
        require(np.array_equal(d.t_s, times[d.t_index.to_numpy(dtype=int)]), f"Wrong retention grid: {path}")
        invalid = d.accuracy.isna()
        require((~invalid | d.compensation.eq("GDC-safe")).all(), f"Invalid non-GDC accuracy: {path}")
        if invalid.any():
            require("gdc_valid" in d and d.loc[invalid, "gdc_valid"].eq(False).all()
                    and d.loc[invalid, ["loss", "error"]].isna().all().all(),
                    f"Unexplained invalid calibration records: {path}")
        valid = d.loc[~invalid]
        require(np.isfinite(valid[["accuracy", "error", "loss"]]).all().all()
                and valid.accuracy.between(0, 100).all(), f"Invalid numerical values: {path}")
        require(np.allclose(valid.accuracy+valid.error, 100, rtol=0, atol=1e-8), f"Accuracy/error mismatch: {path}")
        if "gdc_valid" in d:
            require(d.loc[d.compensation.eq("GDC-safe") & ~invalid, "gdc_valid"].eq(True).all(),
                    f"Invalid calibration presented as valid accuracy: {path}")
        frames.append(d)
        sources.append(dict(file=str(path.relative_to(exp.run_dir)), model_identity=pack["model_identity"],
            bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, root_seed=meta["root_seed"],
            persistent_state_hash=meta["state_hash_start"]))
    return pd.concat(frames, ignore_index=True), sources


def training_figure(r, exp, digital, analog, selections, out):
    r["ieee_style"]()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    sources = []
    for axis, field, data, methods, title in [
        (axes[0], "val_accuracy", digital.assign(method="FP32-Digital"), ["FP32-Digital"], "FP32 validation"),
        (axes[1], "val_accuracy", analog, exp.methods, "HWA validation"),
        (axes[2], "train_accuracy", analog, exp.methods, "HWA training")]:
        for method in methods:
            d = data[data.method.eq(method)]
            s = d.groupby("epoch")[field].agg(["mean", "std"]).reset_index()
            style = dict(r["METHOD_STYLE"].get(method, dict(color="#444444", marker="o", ls="-")))
            label = "FP32" if method == "FP32-Digital" else r["PAPER_LABEL"][method]
            r["_band"](axis, s.epoch.to_numpy(), s["mean"].to_numpy(), s["std"].to_numpy(), style, label,
                       every=max(1, len(s)//8))
            sources.append(s.assign(method=method, metric=field, panel=title))
            if field == "val_accuracy":
                for point in selections[selections.method.eq(method)].itertuples():
                    row = d[d.fold.eq(point.fold) & d.epoch.eq(point.selected_epoch)].iloc[0]
                    axis.scatter(point.selected_epoch, row.val_accuracy, marker="v", s=15,
                                 color=style["color"], zorder=5)
        axis.set_title(title)
        axis.set_xlabel("Epoch within stage")
        axis.set_ylabel("Accuracy (%)")
        axis.legend(fontsize=6)
    fig.tight_layout()
    caption = (f"Digital pretraining ({exp.cfg.digital_epochs} epochs) and AnalogAdam HWA fine-tuning "
        f"({exp.cfg.epochs} epochs); mean ± sample SD across {exp.cfg.train_repeats} training repetitions. "
        "Triangles identify the actual saved validation-selected checkpoints. Training accuracy uses each "
        "method's training-time hardware noise and input interventions; validation uses raw inputs.")
    r["save_figure"](fig, out/"fig1_training_validation", pd.concat(sources, ignore_index=True), caption, False)


def optional_diagnostics(r, exp, identities, ret5, bins, out):
    """Only read existing matched HWA/NSR diagnostics. Never visit the ablation directory."""
    models = [r["HWA"], r["NSR"]]
    F, R = exp.cfg.train_repeats, exp.cfg.realizations
    groups = r["ops"](exp).layer_groups
    paths = [(exp.diag_dir/f"fold{f}"/m/f"real{k:02d}.pt", f, m, k)
             for f in range(F) for m in models for k in range(R)]
    missing = [str(p) for p, *_ in paths if not p.exists()]
    if missing:
        return dict(included=False, reason=f"{len(missing)} saved diagnostic shards missing; no diagnostics executed")
    frames = []
    for path, fold, method, real in paths:
        p = r["torch_load"](path)
        require(p.get("fp") == exp.fp["diagnostics"] and p.get("model_identity") == identities[fold, method],
                f"Foreign optional diagnostics: {path}")
        require(p.get("selected_bins") == bins[fold], f"Wrong diagnostic selected B: {path}")
        d = p["results"]
        require(len(d) == len(groups) and set(d.layer) == set(groups) and not d.layer.duplicated().any()
                and d.fold.eq(fold).all() and d.method.eq(method).all() and d.realization.eq(real).all(),
                f"Invalid optional diagnostic record grid: {path}")
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    needed = [key for key, *_ in r["TABLE2_ROWS"] if key != "ref_gain_B1_sd"] + ["ref_gain_rel_error_pct"]
    require(set(needed).issubset(d.columns) and np.isfinite(d[needed]).all().all(),
            "Optional diagnostics have missing/nonfinite mechanism metrics")
    expected = ret5[ret5.compensation.eq("DS-B1")].set_index(["fold", "method", "realization"]).error
    for row in d.itertuples():
        require(abs(row.test_error_B1_5y_pct-expected.loc[row.fold, row.method, row.realization]) <= 0.05,
                "Optional diagnostic accuracy differs from saved retention")
    d.to_csv(out/"mechanism_diagnostics_records.csv", index=False)
    r["make_table2"](d, F, R, out, layers=groups)
    figures = []
    for method in models:
        path = exp.diag_dir/"fold0"/f"distribution_states_{method}.npz"
        if path.exists():
            stem = "fig5_distributions" if method == r["NSR"] else "figS5_distributions_HWA"
            r["fig5_distributions"](path, method, bins[0], out, False, stem, layers=groups)
            figures.append(stem)
    return dict(included=True, records=len(d), distribution_figures=figures,
                distribution_scope="Single saved fold0 realization; not a mechanism significance test")


def run_export(r, exp=None, download=True):
    if exp is None:
        require(r.get("PARALLEL_ROLE") == "coordinator", "Run this export in P0, not a worker.")
        exp = r["open_full_experiment"](role="coordinator-full")
    require(exp.arch == "resnet32", "This export is for the coordinator ResNet/CIFAR-10 run.")
    require(not exp.cfg.fast_mode, "Smoke-test data cannot be used as full paper results.")
    require(set((r["DIRECT"], r["HWA"], r["DAR"], r["ATT"], r["NSR"])).issubset(exp.models),
            "The coordinator must include FP32-Direct and all four primary HWA methods.")
    print("Checking saved training and retention for", exp.models, flush=True)
    digital, analog, selections, transfers, identities = load_checkpoints(r, exp)
    bins, bin_records = load_bins(r, exp, identities)
    ret, source_shards = load_retention(r, exp, identities, bins)
    F, R = exp.cfg.train_repeats, exp.cfg.realizations
    last = len(r["exact_retention_times"](exp.cfg))-1
    ret5 = ret[ret.t_index.eq(last)]
    # New generation folder: never mix with legacy paper or previous exports.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = exp.run_dir/"paper_NO_ABLATION"/f"{exp.fp['retention'][:12]}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    digital.to_csv(out/"training_history_fp32.csv", index=False)
    analog.to_csv(out/"training_history_hwa.csv", index=False)
    selections.to_csv(out/"best_validation_selection.csv", index=False)
    transfers.to_csv(out/"weight_bias_transfer_errors.csv", index=False)
    ret.to_csv(out/"retention_records_test.csv", index=False)
    pd.DataFrame(bin_records).to_csv(out/"driftslice_selected_B_per_fold.csv", index=False)
    summary = r["two_stage_summary"](ret, ["method", "compensation", "t_index", "t_s"], expected_folds=F, expected_reals=R)
    summary.to_csv(out/"retention_summary_two_stage.csv", index=False)
    fold_means = r["fold_means"](ret, ["method", "compensation", "t_index", "t_s"])
    fold_means.to_csv(out/"retention_error_fold_means.csv", index=False)
    accuracy = summary.copy()
    accuracy["mean_accuracy_pct"] = 100-accuracy.pop("mean")
    accuracy = accuracy.rename(columns={"sd": "sd_across_fold_means_pp"})
    accuracy.to_csv(out/"test_accuracy_vs_retention_summary.csv", index=False)
    training_figure(r, exp, digital, analog, selections, out)
    r["fig2_main_retention"](summary, exp.methods, out, False, F=F, R=R,
                            dataset=r["ops"](exp).dataset, n_times=last+1)
    r["fig3_matched"](summary, exp.methods, out, False, F=F, R=R)
    table = r["make_table1"](ret5, exp.models, F, R, bins, out,
        n_test=r["ops"](exp).n_test_full, with_original_gdc=False,
        dataset=r["ops"](exp).dataset, arch_label=r["ops"](exp).label)
    acc_table = table.copy()
    acc_table["mean_accuracy_pct"] = 100-acc_table.mean_error_pct
    acc_table["cell"] = [r["_fmt"](mean, sd) for mean, sd in
                         zip(acc_table.mean_accuracy_pct, acc_table.sd_across_folds_pct)]
    acc_table.to_csv(out/"TableI_five_year_test_accuracy.csv", index=False)
    # Independently re-read CSV source to check every Table-I mean and SD.
    reread = pd.read_csv(out/"retention_records_test.csv")
    expected = r["two_stage_summary"](reread[reread.t_index.eq(last)], ["method", "compensation"], expected_folds=F, expected_reals=R)
    check = table.merge(expected, on=["method", "compensation"], validate="one_to_one")
    require(len(check) == len(table) and np.allclose(check.mean_error_pct, check["mean"], equal_nan=True)
            and np.allclose(check.sd_across_folds_pct, check["sd"], equal_nan=True), "Table-I CSV traceability failed")
    audit = r["claim_audit"](ret5, exp.models, F, None)
    if "p_holm_all_claims" in audit:
        audit["holm_verdict"] = np.where(audit.p_holm_all_claims.isna(), "NOT EVALUABLE",
            np.where(audit.p_holm_all_claims < 0.05,
                np.where(audit.mean_diff_A_minus_B_pp < 0, "SUPPORTED", "CONTRADICTED"), "INCONCLUSIVE"))
    audit.to_csv(out/"reviewer_claim_audit.csv", index=False)
    paired = []
    for mode in r["compensations_for"](exp.cfg):
        d = r["paired_difference"](ret5, {"method": r["NSR"], "compensation": mode},
                                  {"method": r["HWA"], "compensation": mode})
        paired.append(d.assign(compensation=mode, definition="NSR minus HWA test error, percentage points"))
    pd.concat(paired, ignore_index=True).to_csv(out/"NSR_minus_HWA_paired_fold_differences.csv", index=False)
    diagnostics = optional_diagnostics(r, exp, identities, ret5, bins, out)
    rec = dict(config=asdict(exp.cfg), architecture=exp.arch, dataset=r["ops"](exp).dataset,
        fingerprints=exp.fp, source_shards=source_shards, checkpoint_selections=selections.to_dict("records"),
        selected_bins=bins, dataset_hashes=getattr(exp, "data_hashes", {}),
        statistics="Mean across realizations within each training repetition, then mean/sample SD across repetitions",
        programming_bounds="Legacy [0,gmax] cap after programming noise, as in unchanged P0/P1-P4",
        gdc="Adaptive uniform probe, as in unchanged P0/P1-P4; not one-hot calibration",
        export_runtime=dict(torch=r["torch"].__version__, device=str(exp.device)))
    source = globals().get("__source__")
    if source is None and "__file__" in globals():
        source = Path(__file__).read_text()
    rec["exporter_source_sha256"] = hashlib.sha256(source.encode()).hexdigest() if source is not None else None
    write_json(out/"reproducibility_record.json", rec)
    pd.DataFrame([dict(parameter=k, value=json.dumps(v, default=str)) for k, v in asdict(exp.cfg).items()]).to_csv(out/"experiment_configuration.csv", index=False)
    checks = [dict(check=c, status="PASS", scope="saved-data export") for c in [
        "All selected training checkpoints complete and identity hashes verified",
        "Shared pretrained model within each repetition",
        "Full retention grid and shard/checkpoint identities verified",
        "Persistent PCM state metadata verified",
        "HWA validation-only B selection identity verified",
        "Five-year Table I recomputed from exported source CSV"]]
    pd.DataFrame(checks).to_csv(out/"export_integrity_checks.csv", index=False)
    omitted = [dict(output="Figure 4 / reference-budget ablation", reason="Excluded by user request; not read or executed")]
    if not diagnostics["included"]:
        omitted.append(dict(output="Table II mechanism diagnostics / Figure 5 distributions", reason=diagnostics["reason"]))
    elif "fig5_distributions" not in diagnostics.get("distribution_figures", []):
        omitted.append(dict(output="Figure 5 NSR distributions", reason="Saved fold0 NSR distribution NPZ is missing; not reconstructed"))
    status = dict(core_training_retention_complete=True, ablation_included=False, ablation_required=False,
        diagnostics=diagnostics, omitted_outputs=omitted, methods=list(exp.models), training_repetitions=F,
        realizations_per_repetition=R, timestamps=last+1, retention_records=len(ret),
        invalid_calibration_records=int(ret.accuracy.isna().sum()),
        all_retention_accuracies_valid=bool(ret.accuracy.notna().all()),
        scope="Core training/retention results for all selected methods; not a claim that excluded stages completed",
        inference_or_training_executed=False, output_directory=str(out))
    write_json(out/"export_status.json", status)
    (out/"README.txt").write_text(
        "P0 coordinator paper export: no ablation.\n"
        "Figures 1-3, five-year error/accuracy tables, source CSVs, paired claims and configuration records.\n"
        "Figure 2 compares complete pipelines with different compensation; Figure 3 supplies matched controls.\n"
        "Shading is sample SD across training-repetition means, not a confidence interval.\n"
        "Claim-audit verdict uses pointwise 95% intervals; holm_verdict provides the separately adjusted result.\n"
        "Invalid GDC calibration remains NaN/n/a and is never averaged away.\n"
        "No new training, retention, ablation or mechanism diagnostics were run.\n"
        "See export_status.json for omitted optional outputs.\n", encoding="utf-8")
    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(out))
    print("\nCORE PAPER EXPORT COMPLETE:", out)
    print("ZIP:", zip_path)
    for item in omitted:
        print("OMITTED:", item["output"], "—", item["reason"])
    if download:
        from google.colab import files
        files.download(str(zip_path))
    return out
