"""Validate the uploaded 154-epoch v2 ResNet checkpoint without training.

Load this module in a separate namespace, then call check_v2_checkpoint(globals(),
FULL_EXP, repair=True). Only stale restoration metadata can be repaired. A real
validation mismatch, changed weights, history or split always stops the repair.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import time


def check_v2_checkpoint(nb, exp, fold=0, method=None, repair=False,
                        expected_train_prefix=None):
    """Audit saved best weights; repair metadata only after independent checks."""
    torch = nb["torch"]
    method = method or nb["HWA"]
    cfg, dev = exp.cfg, exp.device
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_path = Path(exp.run_dir) / "checks" / f"checkpoint_recheck_{method}_fold{fold}_{stamp}.json"
    report = {"method": method, "fold": fold, "train_fp": exp.fp["train"],
              "environment": nb["environment_identity"](), "repair_requested": bool(repair),
              "training_performed": False, "weights_changed": False, "time_utc": stamp}
    saved_rng = nb["rng_snapshot"](dev)
    model = None

    def require(ok, message):
        if not ok:
            raise nb["ScientificCheckError"](message)

    def save_report():
        nb["atomic_write_text"](report_path, json.dumps(report, indent=2, default=str))

    def state_hash(state):
        return nb["state_dict_hash"](state)

    def load_pack(path, kind):
        p = nb["torch_load"](path)
        nb["verify_pack"](p, exp.fp["train"], kind, fold, path)
        require(p.get("model_identity") == state_hash(p["model"]),
                f"{path.name}: saved model identity does not match its weights.")
        return p

    try:
        require(exp.arch == "resnet32", "This recovery cell is for the uploaded ResNet v2 notebook.")
        current_fp = nb["compute_fingerprints_resnet"](cfg, exp.data_hashes, nb["RESNET_SPEC"])
        require(current_fp["train"] == exp.fp["train"],
                "Current scientific source/environment differs from the opened experiment. Restore the original definitions.")
        if expected_train_prefix:
            require(exp.fp["train"].startswith(expected_train_prefix),
                    "Training fingerprint differs from the failing log. Restore that notebook/config/environment first.")
        fdir = Path(exp.train_dir) / f"fold{fold}"
        best_path = fdir / f"{method}_best_validation.pt"
        final_path = fdir / f"{method}_final.pt"
        require(best_path.exists() and final_path.exists(), "Completed best/final checkpoints are required.")
        original_bytes = best_path.read_bytes()
        original_sha = hashlib.sha256(original_bytes).hexdigest()
        best = load_pack(best_path, "hwa_best_validation")
        final = load_pack(final_path, "hwa_final")
        digital = load_pack(fdir / "digital_best_validation.pt", "digital_best_validation")
        for p in (best, final):
            require(p.get("method") == method and p.get("arch") == "resnet32",
                    "Checkpoint method/architecture differs.")
            require(p.get("digital_identity") == digital["model_identity"],
                    "Checkpoint digital parent identity differs.")
            require(p.get("analog_epochs_completed") == cfg.epochs,
                    "Checkpoint has not completed the configured HWA epochs.")
            require(p.get("optimizer_steps") == cfg.epochs * p["steps_per_epoch"],
                    "Checkpoint optimizer-step count differs.")
        history = best["history"]
        require(history == final["history"], "Best and final checkpoint histories differ.")
        require([r["epoch"] for r in history] == list(range(1, cfg.epochs + 1)),
                "History has missing/duplicated epochs.")
        selected = best["selected_best_validation_epoch"]
        require(selected == nb["best_epoch_from_history"](history),
                "Saved epoch is not the best epoch under the original selection rule.")
        row = next(r for r in history if r["epoch"] == selected)
        require(math.isclose(best["val_accuracy"], row["val_accuracy"], rel_tol=0, abs_tol=1e-10)
                and math.isclose(best["val_loss"], row["val_loss"], rel_tol=0, abs_tol=1e-10),
                "Saved validation metrics differ from the selected history row.")
        report.update(path=str(best_path), original_file_sha256=original_sha,
                      model_identity=best["model_identity"], selected_epoch=selected,
                      historical_val_accuracy=best["val_accuracy"], historical_val_loss=best["val_loss"],
                      historical_gpu=row.get("gpu"),
                      old_restore_accuracy=best.get("restore_reeval_val_accuracy"),
                      old_restore_abs_diff_pp=best.get("restore_abs_diff_pp"),
                      validation_reads=cfg.validation_reads, validation_batch_size=cfg.test_batch_size)

        train, val, test, split = nb["ops"](exp).loaders(
            cfg, exp.base_dir, fold, dev, verify_fast=False)
        del train, test  # Neither training nor test evaluation is performed.
        split_path = fdir / "split.pt"
        require(split_path.exists(), "Original split.pt is missing; cannot verify the validation split.")
        require(nb["torch_load"](split_path)["digest"] == nb["split_digest"](split),
                "Validation split differs from the saved training split.")
        nb["require_split"](val, "val", "HWA best-epoch selection")
        generator = getattr(val, "generator", None)
        generator_state = generator.get_state().clone() if generator is not None else None
        seed = cfg.seed + fold * 100_000 + (selected - 1) * 41
        report["validation_seed"] = seed
        report["gpu"] = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
        report["validations"] = []

        def evaluate_state(state, label):
            nonlocal model
            # Construct afresh, strictly load the complete AIHWKit state (not just
            # logical weights), then restore CPU-mapped tile device bookkeeping.
            model = nb["AnalogResNet32"](cfg, dev)
            nb["load_model_state"](model, copy.deepcopy(state), dev)
            model.eval()
            actual = nb["cpu_cloned_state"](model)
            changed = [k for k in state if state_hash({k: state[k]}) != state_hash({k: actual[k]})]
            report.setdefault("state_comparisons", []).append({"label": label, "changed_keys": changed})
            require(state_hash(actual) == state_hash(state),
                    f"{label}: complete state changed when loaded; see changed_keys in the JSON report.")
            expected_rpu = str(nb["make_hwa_rpu"](cfg))
            tile_info = []
            for name, layer in model.layers().items():
                tile, = list(layer.analog_tiles())
                require(str(tile.rpu_config) == expected_rpu, f"{name}: saved RPU configuration differs.")
                weight = nb["physical_weight_parameter"](layer)
                require(weight.device == dev or (weight.device.type == dev.type and dev.index is None),
                        f"{name}: tile weight is on the wrong device.")
                scales = tile.get_scales().detach()
                require(bool(torch.isfinite(scales).all()) and bool((scales > 0).all()),
                        f"{name}: invalid mapping scales.")
                tile_info.append({"name": name, "min_scale": float(scales.min()),
                                  "input_range": float(tile.input_range.detach())})
            report["tiles"] = tile_info
            if generator_state is not None:
                generator.set_state(generator_state)
            acc, loss = nb["validation_score"](model, val, dev, cfg.validation_reads, seed)
            require(math.isfinite(acc) and math.isfinite(loss), "Nonfinite validation result.")
            after = nb["cpu_cloned_state"](model)
            require(state_hash(actual) == state_hash(after), "Evaluation changed model state.")
            report["validations"].append({"label": label, "accuracy": acc, "loss": loss,
                                           "abs_diff_from_history_pp": abs(acc - best["val_accuracy"])})
            model = None
            return after

        state = evaluate_state(best["model"], "fresh_load_1")
        evaluate_state(best["model"], "fresh_load_2")
        evaluate_state(state, "round_trip")
        values = report["validations"]
        report["same_seed_repeat_span_pp"] = max(v["accuracy"] for v in values) - min(v["accuracy"] for v in values)
        require(report["same_seed_repeat_span_pp"] <= 1e-10,
                "Repeated seeded validation is not reproducible. No checkpoint changes were made.")
        # Keep the existing scientific threshold. Do not average away a failure.
        require(all(v["abs_diff_from_history_pp"] <= 0.05 for v in values),
                "The original validation mismatch persists. No metadata/weights were changed. Send the JSON report.")
        report["status"] = "PASS"
        fresh = values[0]
        old_diff = float(best.get("restore_abs_diff_pp", math.inf))
        if repair and (not math.isfinite(old_diff) or old_diff > 0.05):
            require(best_path.read_bytes() == original_bytes, "Checkpoint changed during this audit; stopped.")
            updated = copy.deepcopy(best)
            updated["restore_reeval_val_accuracy"] = fresh["accuracy"]
            updated["restore_abs_diff_pp"] = fresh["abs_diff_from_history_pp"]
            updated["restoration_recheck"] = {"time_utc": stamp, "report": str(report_path),
                                               "original_sha256": original_sha,
                                               "old_restore_abs_diff_pp": old_diff,
                                               "environment": report["environment"], "gpu": report["gpu"]}
            unchanged = set(best) - {"restore_reeval_val_accuracy", "restore_abs_diff_pp"}
            require(state_hash({k: best[k] for k in unchanged}) == state_hash({k: updated[k] for k in unchanged}),
                    "Repair attempted to change scientific data; stopped.")
            backup = best_path.with_name(best_path.name + f".before_recheck.{original_sha[:12]}.bak")
            if backup.exists():
                require(backup.read_bytes() == original_bytes, "Existing backup has different bytes.")
            else:
                with backup.open("xb") as stream:
                    stream.write(original_bytes)
            nb["atomic_torch_save"](updated, best_path)
            reread = load_pack(best_path, "hwa_best_validation")
            require(reread["model_identity"] == best["model_identity"], "Repaired model identity differs.")
            report.update(status="REPAIRED_STALE_METADATA", backup=str(backup),
                          metadata_only=True, new_restore_abs_diff_pp=updated["restore_abs_diff_pp"])
        elif old_diff > 0.05 or not math.isfinite(old_diff):
            report["status"] = "PASS_BUT_STALE_METADATA_NOT_REPAIRED"
        save_report()
        print(f"{report['status']}: {method} fold{fold}, epoch {selected}; "
              f"validation {fresh['accuracy']:.6f}%, difference {fresh['abs_diff_from_history_pp']:.6f} pp.")
        print(f"Report: {report_path}")
        return report
    except Exception as exc:
        report.update(status="STOPPED", error_type=type(exc).__name__, error=str(exc))
        save_report()
        print(f"STOPPED: {exc}\nReport: {report_path}")
        raise
    finally:
        model = None
        nb["rng_restore"](saved_rng, dev)
