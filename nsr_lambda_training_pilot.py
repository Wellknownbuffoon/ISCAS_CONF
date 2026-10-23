"""Matched, resumable NSR continuation pilot on validation data only.

Run in a separate namespace after initializing the FULL ResNet notebook. This is
a new experiment: it never rewrites the original training/retention checkpoints.
"""
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import pandas as pd
import torch
from aihwkit.optim import AnalogAdam


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _source():
    return globals().get("__source__") or Path(__file__).read_text()


def _retention_validation(r, model, cfg, fold, val, device, realizations):
    """Same physical devices, reference/read seeds and MVM as native retention."""
    r["require_split"](val, "val", "NSR lambda pilot selection")
    before = r["state_dict_hash"](model.state_dict())
    saved_rng = r["rng_snapshot"](device)
    was_training = model.training
    rows = []
    try:
        model.eval()
        snapshot = r["snapshot_analog_resnet"](model)
        times = r["exact_retention_times"](cfg)
        for realization in range(realizations):
            root = cfg.seed + fold * 1_000_000 + realization * 10_000
            main = r["build_main_pcm_all"](snapshot, cfg, root + 10)
            state_hash = r["pcm_state_hash_all"](main)
            bank = r["PCMClassReferenceBank"](cfg, 1, root + 30, device)
            if bank.target.numel() != cfg.reference_cell_budget:
                raise RuntimeError("Wrong physical reference budget")
            for ti in (0, len(times) - 1):
                ts = float(times[ti])
                _, conductances = r["plain_effective_weights_all"](main, ts, root + 100_000 + ti * 100)
                corrected, gains = r["driftslice_effective_weights_all"](
                    main, conductances, bank, ts, root + 500_000 + ti * 101)
                with r["random_stream"](root + 700_000 + ti * 31, device):
                    loss, accuracy = r["evaluate_deployed"](model, corrected, val, device)
                if not np.isfinite([loss, accuracy]).all() or not 0 <= accuracy <= 100:
                    raise FloatingPointError("Nonfinite validation output")
                rows.append(dict(realization=realization, t_index=ti, t_s=ts,
                                 accuracy=accuracy, loss=loss, gain=float(gains[0]), split="val"))
                print(f"  validation fold{fold} device {realization+1}/{realizations} "
                      f"t={ts:g}s: {accuracy:.3f}%", flush=True)
            if state_hash != r["pcm_state_hash_all"](main):
                raise RuntimeError("Persistent programmed devices changed during validation")
            del main, bank, conductances, corrected
    finally:
        model.train(was_training)
        r["rng_restore"](saved_rng, device)
    if before != r["state_dict_hash"](model.state_dict()):
        raise RuntimeError("Validation deployment changed model weights or BatchNorm state")
    return rows


def _score(rows):
    data = pd.DataFrame(rows)
    first, last = int(data.t_index.min()), int(data.t_index.max())
    return dict(accuracy_20s=float(data.loc[data.t_index.eq(first), "accuracy"].mean()),
                accuracy_5y=float(data.loc[data.t_index.eq(last), "accuracy"].mean()))


def _better(candidate, current, baseline, max_drop):
    """Predeclared five-year selection, with a 20-second accuracy guard."""
    if candidate["accuracy_20s"] < baseline["accuracy_20s"] - max_drop - 1e-10:
        return False
    return (candidate["accuracy_5y"], candidate["accuracy_20s"], -candidate["epoch"]) > (
        current["accuracy_5y"], current["accuracy_20s"], -current["epoch"])


def _read_resume(r, folder, fingerprint, source_id, fold, coefficient, batches, epochs):
    found, usable = [], []
    for path in (folder/"training.slotA.pt", folder/"training.slotB.pt"):
        if not path.exists():
            continue
        found.append(path)
        try:
            pack = r["torch_load"](path)
        except Exception as exc:
            print(f"Unreadable interrupted slot {path.name}: {type(exc).__name__}", flush=True)
            continue
        if (pack.get("pilot_fingerprint") != fingerprint or pack.get("source_model_identity") != source_id
                or pack.get("fold") != fold or pack.get("nsr_lambda") != coefficient):
            raise RuntimeError(f"Foreign pilot checkpoint: {path}")
        if (pack.get("kind") != "nsr_lambda_pilot_training"
                or r["state_dict_hash"](pack["model"]) != pack["model_identity"]
                or r["state_dict_hash"](pack["best_model"]) != pack["best_model_identity"]):
            raise RuntimeError(f"Pilot checkpoint content mismatch: {path}")
        epoch = pack["next_epoch"]
        if not 0 <= epoch <= epochs or len(pack["history"]) != epoch or pack["optimizer_steps"] != epoch*batches:
            raise RuntimeError(f"Incomplete pilot epoch/step accounting: {path}")
        usable.append(pack)
    if found and not usable:
        raise RuntimeError(f"No readable pilot checkpoint in {folder}; original checkpoints are untouched")
    return max(usable, key=lambda p: p["next_epoch"]) if usable else None


def _export(r, out, histories, records, summaries):
    for name, rows in (("training_history.csv", histories), ("validation_records.csv", records),
                       ("selected_checkpoints.csv", summaries)):
        r["atomic_write_csv"](pd.DataFrame(rows), out/name)


def run_pilot(r, exp=None, folds=(0,), lambdas=(0.0, 0.1, 1.0), additional_epochs=10,
              realizations=3, learning_rate=1e-4, min_learning_rate=1e-6,
              evaluation_epochs=None, max_initial_drop_pp=0.25, download=True, _after_epoch=None):
    """Train matched continuations, select on retained VALIDATION accuracy, export.

    Default screen: fold0, lambda=0/.1/1, ten additional epochs, evaluate epochs
    0/5/10 at 20 seconds and five years, three fixed device realizations. R=64,
    B=1 and K=4. Epoch zero is an eligible fallback. The official test set is
    never iterated. Rerunning the identical call resumes at an epoch boundary.
    """
    if exp is None:
        exp = r.get("FULL_EXP") or r.get("COORD_EXP")
        if exp is None:
            exp = r["open_full_experiment"](role="worker")
    cfg, device = exp.cfg, exp.device
    if exp.arch != "resnet32" or cfg.fast_mode:
        raise RuntimeError("Initialize the FULL ResNet-32 experiment first")
    if cfg.require_cuda and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("This experiment requires an available CUDA GPU")
    if cfg.reference_cell_budget != 64 or cfg.reference_read_repeats != 4:
        raise ValueError("This pilot fixes the original reference protocol at R=64, K=4")
    if int(additional_epochs) != additional_epochs or additional_epochs < 1:
        raise ValueError("additional_epochs must be a positive integer")
    additional_epochs = int(additional_epochs)
    folds = tuple(folds)
    if (not folds or len(set(folds)) != len(folds)
            or any(int(f) != f or f < 0 or f >= cfg.train_repeats for f in folds)):
        raise ValueError("Invalid folds")
    folds = tuple(map(int, folds))
    lambdas = tuple(float(x) for x in lambdas)
    if (not lambdas or len(set(lambdas)) != len(lambdas) or 0.0 not in lambdas
            or any(not np.isfinite(x) or x < 0 for x in lambdas)):
        raise ValueError("Use distinct nonnegative coefficients including the lambda=0 control")
    if int(realizations) != realizations or not 1 <= realizations <= cfg.realizations:
        raise ValueError("Invalid realization count")
    realizations = int(realizations)
    if not (0 < min_learning_rate <= learning_rate and np.isfinite(learning_rate)
            and np.isfinite(max_initial_drop_pp) and max_initial_drop_pp >= 0):
        raise ValueError("Invalid learning rate or accuracy guard")
    raw_evals = evaluation_epochs if evaluation_epochs is not None else (0, max(1, additional_epochs//2), additional_epochs)
    if any(int(e) != e for e in raw_evals):
        raise ValueError("Evaluation epochs must be integers")
    evals = tuple(sorted(set(map(int, raw_evals))))
    if not evals or evals[0] != 0 or evals[-1] != additional_epochs:
        raise ValueError("Evaluation epochs must include zero and the last epoch, inside the training range")

    sources, checkpoints = [], {}
    for fold in folds:
        path = exp.train_dir/f"fold{fold}"/f"{r['HWA']}_best_validation.pt"
        pack = r["torch_load"](path)
        r["verify_pack"](pack, exp.fp["train"], "hwa_best_validation", fold, path)
        if (pack.get("method") != r["HWA"] or pack.get("analog_epochs_completed") != cfg.epochs
                or not pack.get("digital_identity")
                or r["state_dict_hash"](pack["model"]) != pack.get("model_identity")):
            raise RuntimeError(f"Incomplete or inconsistent HWA starting checkpoint: {path}")
        checkpoints[fold] = pack
        sources.append(dict(fold=fold, path=str(path), model_identity=pack["model_identity"],
                            digital_identity=pack["digital_identity"],
                            original_selected_epoch=pack.get("selected_best_validation_epoch")))
    contract = dict(kind="exploratory_validation_nsr_continuation",
        source_sha256=hashlib.sha256(_source().encode()).hexdigest(),
        original_config=asdict(cfg), original_fingerprints=exp.fp, sources=sources,
        data_hashes=getattr(exp, "data_hashes", {}), folds=folds, lambdas=lambdas,
        additional_epochs=additional_epochs, realizations=realizations,
        learning_rate=learning_rate, min_learning_rate=min_learning_rate,
        scheduler="per-batch cosine decay; fresh AnalogAdam; original manual physical-weight decay",
        evaluation_epochs=evals, selection="largest mean five-year validation accuracy; ties 20s then earliest epoch",
        max_initial_drop_pp=max_initial_drop_pp, initial_checkpoint_eligible=True,
        reference_cell_budget=64, bins=1, reference_read_repeats=4,
        paired_batches_augmentation_and_forward_noise=True,
        official_test_evaluated=False, original_completed_results_modified=False,
        environment=dict(torch=torch.__version__, aihwkit=importlib.metadata.version("aihwkit"),
                         python=platform.python_version(), device=str(device),
                         gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "none"))
    fingerprint = _digest(contract)
    out = exp.run_dir/"nsr_lambda_pilot"/fingerprint[:16]
    out.mkdir(parents=True, exist_ok=True)
    r["atomic_write_text"](out/"pilot_contract.json", json.dumps(dict(contract, pilot_fingerprint=fingerprint), indent=2))
    r["atomic_write_text"](out/"pilot_source.py", _source())
    print(f"NSR VALIDATION PILOT: {out}\nOriginal training and retention files stay in their existing folders.\n"
          f"{len(folds)*len(lambdas)*additional_epochs} continuation epochs; evaluation at {evals}; "
          f"B=1, R=64, K=4; {realizations} programmed devices.\n"
          "These are exploratory validation results, not final paper test results.", flush=True)

    histories, records, summaries = [], [], []
    saved_rng = r["rng_snapshot"](device)
    try:
        for fold in folds:
            train, val, _, _ = r["build_cifar10_loaders"](replace(cfg, allow_download=False), exp.base_dir, fold)
            train.split_name, val.split_name = "train", "val"
            if cfg.fast_gpu_loader and device.type == "cuda":
                train = r["FastGPULoader"](train, device, "train")
                val = r["FastGPULoader"](val, device, "val")
            if train.generator is None or len(train) == 0:
                raise RuntimeError("Expected a nonempty shuffled loader with an independent generator")
            source = checkpoints[fold]
            baseline_path = out/f"baseline_fold{fold}.pt"
            if baseline_path.exists():
                baseline = r["torch_load"](baseline_path)
                if (baseline.get("pilot_fingerprint") != fingerprint
                        or baseline.get("source_model_identity") != source["model_identity"]):
                    raise RuntimeError("Foreign baseline validation cache")
                baseline_rows = baseline["records"]
            else:
                model = r["load_analog_resnet_from_pack"](cfg, source, device)
                baseline_rows = _retention_validation(r, model, cfg, fold, val, device, realizations)
                r["atomic_torch_save"](dict(pilot_fingerprint=fingerprint, source_model_identity=source["model_identity"],
                                             records=baseline_rows), baseline_path)
                del model
            baseline_score = dict(_score(baseline_rows), epoch=0)

            for coefficient in lambdas:
                folder = out/f"fold{fold}"/f"lambda_{coefficient:g}"
                folder.mkdir(parents=True, exist_ok=True)
                local_cfg = replace(cfg, nsr_lambda=coefficient)
                r["seed_all"](cfg.seed + fold)
                train.generator.manual_seed(cfg.seed + fold + 1000)
                model = r["load_analog_resnet_from_pack"](cfg, source, device)
                if r["state_dict_hash"](model.state_dict()) != source["model_identity"]:
                    raise RuntimeError("Loaded HWA state differs from common source")
                weights = [r["physical_weight_parameter"](layer) for layer in model.layers().values()]
                ids = {id(p) for p in weights}
                opt = AnalogAdam([dict(params=weights, weight_decay=0.0),
                    dict(params=[p for p in model.parameters() if id(p) not in ids], weight_decay=0.0)],
                    lr=learning_rate, weight_decay=0.0)
                opt.regroup_param_groups(model)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=additional_epochs*len(train), eta_min=min_learning_rate)
                pack = _read_resume(r, folder, fingerprint, source["model_identity"], fold,
                                    coefficient, len(train), additional_epochs)
                if pack:
                    r["load_model_state"](model, pack["model"], device)
                    opt.load_state_dict(pack["optimizer"])
                    scheduler.load_state_dict(pack["scheduler"])
                    train.generator.set_state(pack["loader_rng"].cpu())
                    r["rng_restore"](pack["rng"], device)
                    history, candidate_rows = pack["history"], pack["validation_records"]
                    best_state, best = pack["best_model"], pack["best_score"]
                    start, steps = pack["next_epoch"], pack["optimizer_steps"]
                    first_change = pack["first_step_max_weight_change"]
                    print(f"RESUME fold{fold} lambda={coefficient:g}: {start}/{additional_epochs} epochs completed", flush=True)
                else:
                    history, candidate_rows = [], [dict(row, epoch=0) for row in baseline_rows]
                    best_state, best = source["model"], dict(baseline_score)
                    start, steps, first_change = 0, 0, None
                for epoch in range(start, additional_epochs):
                    began = time.monotonic()
                    model.train()
                    sums = dict(ce=0.0, extra=0.0, correct=0, samples=0, clipped_batches=0,
                                gradient_norm_sum=0.0)
                    for batch, (x, y) in enumerate(train):
                        x = r["cifar_augment"](x.to(device), r["augment_seed"](cfg, fold, 1, epoch, batch)).flatten(1)
                        y = y.to(device)
                        opt.zero_grad(set_to_none=True)
                        # Same forward random stream for each lambda, even if a regularizer later uses randomness.
                        with r["random_stream"](310_000_000 + cfg.seed + fold*1_000_000 + epoch*10_000 + batch, device):
                            logits = model(x)
                            ce = torch.nn.functional.cross_entropy(logits, y)
                            extra = r["nsr_penalty"](model, local_cfg) if coefficient else torch.zeros((), device=device)
                            loss = ce + extra
                            if not torch.isfinite(loss):
                                raise FloatingPointError(f"Nonfinite loss: fold{fold} lambda={coefficient} epoch={epoch+1} batch={batch}")
                            loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
                        before = [w.detach().clone() for w in weights] if first_change is None else None
                        opt.step()  # AIHWKit tile post-update processing is part of AnalogAdam.step().
                        with torch.no_grad():
                            group_lr = {id(p): float(g["lr"]) for g in opt.param_groups for p in g["params"]}
                            for w in weights:
                                w.mul_(1.0-group_lr.get(id(w), learning_rate)*cfg.weight_decay)
                            for layer in model.layers().values():
                                tile, = list(layer.analog_tiles())
                                tile.input_range.clamp_(min=1e-3)
                        if before is not None:
                            first_change = max(float((w.detach()-old).abs().max()) for w, old in zip(weights, before))
                            if not np.isfinite(first_change) or first_change <= 0:
                                raise RuntimeError("AnalogAdam did not make a finite physical-weight update")
                        scheduler.step()
                        steps += 1
                        n = y.numel()
                        sums["samples"] += n
                        sums["correct"] += int(logits.detach().argmax(1).eq(y).sum())
                        sums["ce"] += float(ce.detach())*n
                        sums["extra"] += float(extra.detach())*n
                        sums["gradient_norm_sum"] += float(norm)
                        sums["clipped_batches"] += int(float(norm) > cfg.grad_clip)
                        if (batch+1) % 50 == 0:
                            print(f"TRAIN fold{fold} lambda={coefficient:g} epoch {epoch+1}/{additional_epochs} "
                                  f"batch {batch+1}/{len(train)}", flush=True)
                    score = None
                    if epoch+1 in evals:
                        print(f"Retention VALIDATION: fold{fold} lambda={coefficient:g} epoch={epoch+1}", flush=True)
                        result = _retention_validation(r, model, cfg, fold, val, device, realizations)
                        candidate_rows.extend(dict(row, epoch=epoch+1) for row in result)
                        score = dict(_score(result), epoch=epoch+1)
                        if _better(score, best, baseline_score, max_initial_drop_pp):
                            best, best_state = score, r["cpu_cloned_state"](model)
                    history.append(dict(fold=fold, nsr_lambda=coefficient, epoch=epoch+1,
                        train_ce=sums["ce"]/sums["samples"], train_extra=sums["extra"]/sums["samples"],
                        train_accuracy=100*sums["correct"]/sums["samples"],
                        preclip_gradient_norm_mean=sums["gradient_norm_sum"]/len(train),
                        clipped_batch_fraction=sums["clipped_batches"]/len(train), optimizer_steps=steps,
                        lr_end=scheduler.get_last_lr()[0], selected_epoch=best["epoch"],
                        validation_20s=score["accuracy_20s"] if score else None,
                        validation_5y=score["accuracy_5y"] if score else None))
                    state = r["cpu_cloned_state"](model)
                    pack = dict(kind="nsr_lambda_pilot_training", pilot_fingerprint=fingerprint,
                        source_model_identity=source["model_identity"], fold=fold, nsr_lambda=coefficient,
                        model=state, model_identity=r["state_dict_hash"](state),
                        optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                        loader_rng=train.generator.get_state(), rng=r["rng_snapshot"](device),
                        next_epoch=epoch+1, optimizer_steps=steps, history=history,
                        validation_records=candidate_rows, best_model=best_state,
                        best_model_identity=r["state_dict_hash"](best_state), best_score=best,
                        first_step_max_weight_change=first_change)
                    r["atomic_torch_save"](pack, folder/("training.slotA.pt" if (epoch+1)%2 else "training.slotB.pt"))
                    r["atomic_write_csv"](pd.DataFrame(history), folder/"history.csv")
                    print(f"DONE fold{fold} lambda={coefficient:g} epoch {epoch+1}/{additional_epochs}: "
                          f"CE={history[-1]['train_ce']:.4f}, extra={history[-1]['train_extra']:.5g}, "
                          f"{time.monotonic()-began:.1f}s including any scheduled validation; "
                          f"selected epoch={best['epoch']}", flush=True)
                    if _after_epoch:
                        _after_epoch(fold, coefficient, epoch+1)
                selected = dict(kind="nsr_lambda_pilot_selected", pilot_fingerprint=fingerprint,
                    source_model_identity=source["model_identity"], digital_identity=source["digital_identity"],
                    fold=fold, nsr_lambda=coefficient, model=best_state,
                    model_identity=r["state_dict_hash"](best_state), selected_epoch=best["epoch"],
                    validation_score=best, continuation_epochs_completed=additional_epochs,
                    optimizer_steps=steps, official_test_evaluated=False)
                r["atomic_torch_save"](selected, folder/"selected_validation.pt")
                summaries.append(dict(fold=fold, nsr_lambda=coefficient, selected_epoch=best["epoch"],
                    accuracy_20s=best["accuracy_20s"], accuracy_5y=best["accuracy_5y"],
                    change_5y_from_start_pp=best["accuracy_5y"]-baseline_score["accuracy_5y"],
                    continuation_epochs_completed=additional_epochs, optimizer_steps=steps,
                    initial_model_identity=source["model_identity"], selected_model_identity=selected["model_identity"],
                    selected_checkpoint=str(folder/"selected_validation.pt")))
                histories.extend(history)
                records.extend(dict(row, fold=fold, nsr_lambda=coefficient) for row in candidate_rows)
                _export(r, out, histories, records, summaries)
                del model, weights, opt, scheduler, pack, best_state, selected
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        r["rng_restore"](saved_rng, device)

    summary = pd.DataFrame(summaries)
    control = summary.loc[summary.nsr_lambda.eq(0), ["fold", "accuracy_5y"]].rename(columns={"accuracy_5y":"control_5y"})
    paired = summary.merge(control, on="fold", validate="many_to_one")
    paired["change_5y_vs_lambda0_pp"] = paired.accuracy_5y-paired.control_5y
    r["atomic_write_csv"](paired, out/"paired_validation_comparison.csv")
    r["atomic_write_text"](out/"status.json", json.dumps(dict(status="COMPLETE", pilot_fingerprint=fingerprint,
        official_test_evaluated=False, scope="EXPLORATORY VALIDATION; not final paper test results"), indent=2))
    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file() and path.suffix in (".csv", ".json", ".py"):
                archive.write(path, str(path.relative_to(out)))
    print("\nValidation-only comparison:\n", paired[["fold", "nsr_lambda", "selected_epoch", "accuracy_20s",
        "accuracy_5y", "change_5y_vs_lambda0_pp"]].to_string(index=False))
    print(f"\nUpload this small diagnostics ZIP: {zip_path}\nSelected model files stay separately on Drive.\n"
          "A positive pilot difference does not guarantee 90% or a final-test improvement.", flush=True)
    if download:
        from google.colab import files
        files.download(str(zip_path))
    return out
