"""Auditable FP32 (30e) -> AnalogAdam HWA (154e) -> PCM retention study.

Adapted from the user's DriftSlice v2.3 and AICAS native notebooks. DriftSlice
reference-bank and persistent device-state PCM logic live in driftslice_core.py.
See AUDIT_REPORT.md for v1.1.0 compatibility, calibration and bounds corrections.

IMPORTANT: DriftSlice's per-device state-dependent gains are an IDEALIZED
weight-reconstruction experiment. It is NOT proof of implementable crossbar
readout without additional hardware and routing/circuit cost modeling.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import driftslice_core as ds

DIRECT = "FP32-Direct"
HWA = ds.METHOD_HWA
DAR = ds.METHOD_DAR
ATT = "PCM-DriftAttention-HWA"
NSR = "PCM-NSR-HWA"
ATT_ONLY = "PCM-AttentionOnly-HWA"
GAIN_ONLY = "PCM-GainOnly-HWA"
ALL_TRAIN_METHODS = (HWA, DAR, ATT, NSR, ATT_ONLY, GAIN_ONLY)


class AttentionGate(nn.Module):
    """AICAS-inspired training-only input gate, as in v2_lib's AttentionGate."""
    def __init__(self, dim=784, cg=0.9):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.cg = cg

    def forward(self, x):
        alpha = torch.softmax(self.proj(x), dim=-1)
        return self.cg * x + (1.0 - self.cg) * alpha, alpha


def training_input(x, method, cfg, gate=None):
    """Training-only learned augmentation; raw inputs define deployment for ALL methods.

    This preserves the existing explicitly documented training-only method identity.
    Gate parameters are saved for reproducibility, not installed in SnapshotMLP.
    """
    if method in (ATT, ATT_ONLY):
        if gate is None:
            raise ValueError('Attention training requires its augmentation gate')
        return gate(x)
    if method == GAIN_ONLY:
        return cfg.attention_input_gain * x, None
    return x, None


def make_hwa_optimizer(model, cfg):
    from aihwkit.optim import AnalogAdam
    weights = [ds.physical_weight_parameter(layer) for layer in model.layers().values()]
    ids = {id(p) for p in weights}
    nonweights = [p for p in model.parameters() if id(p) not in ids]
    opt = AnalogAdam([{'params': weights, 'weight_decay': 0.0},
                      {'params': nonweights, 'weight_decay': 0.0}],
                     lr=cfg.lr, weight_decay=0.0)
    opt.regroup_param_groups(model)
    grouped = [id(p) for g in opt.param_groups for p in g['params']]
    if any(grouped.count(id(p)) != 1 for p in model.parameters()):
        raise RuntimeError('Optimizer lost or duplicated model parameters')
    return opt, weights


@torch.no_grad()
def decay_analog_weights(opt, weights, cfg):
    """Decoupled decay on physical weights only, using each actual group's LR."""
    group_lr = {id(p): float(g['lr']) for g in opt.param_groups for p in g['params']}
    for p in weights:
        p.mul_(1.0 - group_lr[id(p)] * cfg.weight_decay)


def direct_baseline(cfg, digital, seed, device, val_loader=None):
    """Transfer only. No optimizer is constructed or stepped on this branch."""
    model = ds.transfer_pretrained_digital_to_analog(
        cfg, digital, seed, device, val_loader=val_loader)
    ensure_same_pretrained_weights(model, digital)
    return model.eval()


def nsr_penalty(model: ds.AnalogMLP, cfg: ds.Config):
    """Preserve nsr_run.py noise-variance regularizer; not the AICAS DAR loss."""
    L = math.log(5.0 * ds.YEAR / ds.T_REF)
    L2_mean = L ** 2 / 3.0    # logarithmically uniform retention schedule
    t_read = 250e-9
    lt = np.linspace(math.log(ds.T_REF), math.log(5.0 * ds.YEAR), 200)
    R_mean = float(np.mean(lt + math.log(1.0 / (2.0 * t_read))))
    sp0 = (0.26348 / 25.0) ** 2
    losses = []
    for layer in model.layers().values():
        w = ds.physical_weight_parameter(layer)
        g = w.abs() / w.detach().abs().max().clamp_min(1e-12)
        gc = g.clamp_min(1e-7)
        sigma_p = (0.26348 + 1.9650 * g - 1.1731 * g.square()) / 25.0
        sigma_nu = (-0.0125 * gc.log() - 0.0059).clamp(0.008, 0.045)
        q_s = (0.0088 / gc.pow(0.65).clamp_min(1e-3)).clamp(max=0.2)
        variance = sigma_p.square() + g.square() * (
            sigma_nu.square() * L2_mean + q_s.square() * R_mean
        )
        losses.append((variance.sum() + sp0 * g.numel()) / (g.square().sum() + 1e-12))
    return cfg.nsr_lambda * torch.stack(losses).mean()


def validation_score(model: nn.Module, val_loader, device, n_reads: int, seed: int):
    """Independent of held-out test; repeated evaluations reduce noisy selection."""
    scores = []
    for j in range(n_reads):
        with ds.random_stream(seed + j * 173, device):
            scores.append(ds.evaluate_model(model, val_loader, device))
    losses, accuracies = np.asarray(scores, dtype=float).T
    return float(accuracies.mean()), float(losses.mean())


def ensure_same_pretrained_weights(model: ds.AnalogMLP, digital: ds.DigitalMLP):
    """Assert copying both logical analog weights and digital bias before updates."""
    d_state = digital.state_dict()
    for name, layer in model.layers().items():
        w, b = layer.get_weights(apply_weight_scaling=True)
        torch.testing.assert_close(w.cpu(), d_state[name + '.weight'].cpu(), atol=2e-4, rtol=2e-3)
        torch.testing.assert_close(b.cpu(), d_state[name + '.bias'].cpu(), atol=2e-4, rtol=2e-3)


def training_metadata(cfg, fold, method, selected_epoch, trained_epochs):
    return {
        'fold': fold, 'method': method,
        'digital_epochs': cfg.digital_epochs,
        'analog_epochs_completed': trained_epochs,
        'checkpoint_epoch': selected_epoch,
        'optimizer_class': 'aihwkit.optim.AnalogAdam',
        'wd_mode': 'manual_decoupled_analog_weights_only',
        'selection_split': 'validation_only',
        'analog_bias_storage': 'digital_bias=True',
        'retention_model': 'snapshot_device_state_pcm',
        'pcm_conductance_bounds': cfg.pcm_conductance_bounds,
        'input_intervention_semantics': cfg.input_intervention_semantics,
        'deployment_input': 'raw; training augmentation gate/gain is omitted by design',
        'experiment_label': 'SMOKE TEST' if cfg.fast_mode else 'FULL PROTOCOL',
    }


def train_hwa_all(method, cfg, fold, digital, train_loader, val_loader, device, out, fingerprint):
    """Full 154 epochs, checkpoint BEST VAL only after epoch>=1, keep FINAL separately.

    Optimizer type is exactly AnalogAdam; manually decoupled weight decay on
    physical analog weights avoids applying coupled L2 to scales, biases, etc.
    This is a documented adaptation to the user's unstable v1 optimization.
    """
    from aihwkit.optim import AnalogAdam
    if method not in ALL_TRAIN_METHODS:
        raise ValueError(f'Unknown HWA method {method}')
    if not cfg.fast_mode and cfg.epochs != 154:
        raise ValueError('Full HWA requires exactly 154 additional epochs')
    ds.seed_all(cfg.seed + fold)
    train_loader.generator.manual_seed(cfg.seed + fold + 1000)
    model = ds.transfer_pretrained_digital_to_analog(
        cfg, digital, cfg.seed + fold, device, val_loader=val_loader
    )
    ensure_same_pretrained_weights(model, digital)
    gate = AttentionGate(cg=cfg.attention_input_gain).to(device) if method in (ATT, ATT_ONLY) else None
    opt, weights = make_hwa_optimizer(model, cfg)
    assert isinstance(opt, AnalogAdam), 'Wrong optimizer class'
    ensure_same_pretrained_weights(model, digital)  # after optimizer construction, before ANY update
    opt_gate = torch.optim.Adam(gate.parameters(), lr=cfg.attention_lr) if gate else None
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.max_lr, epochs=cfg.epochs,
        steps_per_epoch=len(train_loader), div_factor=cfg.max_lr / cfg.lr,
    )
    initial_refs = {
        name: ds.physical_weight_parameter(layer).detach().clone()
        for name, layer in model.layers().items()
    }
    tag = f'fold{fold}_{method}'
    path = out / (tag + '_training.pt')
    history_path = out / (tag + '_history.csv')
    best_path = out / (tag + '_best_validation.pt')
    final_path = out / (tag + '_final.pt')
    history, start_epoch = [], 0
    best_val_acc, best_val_loss, best_state, best_epoch = -1.0, math.inf, None, -1
    best_gate_state, optimizer_steps, first_update = None, 0, {}
    provenance = model.pretrained_provenance
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved.get('fingerprint') != fingerprint or saved.get('method') != method:
            raise RuntimeError('Incompatible training checkpoint (fingerprint/method)')
        if saved.get('pretrained_provenance') != provenance:
            raise RuntimeError('HWA pretrained initialization provenance mismatch')
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        if gate is not None:
            gate.load_state_dict(saved['gate'])
            opt_gate.load_state_dict(saved['gate_optimizer'])
        history, start_epoch = saved['history'], saved['next_epoch']
        ds.validate_epoch_history(history, cfg.epochs, method, complete=False)
        if start_epoch != len(history):
            raise RuntimeError('HWA resume cursor/history mismatch')
        best_val_acc, best_val_loss = saved['best_val_accuracy'], saved['best_val_loss']
        best_state, best_epoch = saved['best_model'], saved['best_epoch']
        best_gate_state = saved['best_gate']
        optimizer_steps, first_update = saved['optimizer_steps'], saved['first_weight_update']
        initial_refs = {k: v.to(device) for k, v in saved['initial_refs'].items()}
        train_loader.generator.set_state(saved['loader_rng'].cpu())
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type == 'cuda' and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu(), device)

    t_start = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        if gate is not None:
            gate.train()
        sum_ce = sum_extra = sum_loss = 0.0
        n_samples = 0
        for batch, (x, y) in enumerate(train_loader):
            x, y = x.to(device).flatten(1), y.to(device)
            opt.zero_grad(set_to_none=True)
            if opt_gate: opt_gate.zero_grad(set_to_none=True)
            x, alpha = training_input(x, method, cfg, gate)
            logits = model(x)
            ce = F.cross_entropy(logits, y)
            extra = torch.zeros((), device=device)
            if method in (DAR, ATT):
                td = cfg.dar_times_s[epoch % len(cfg.dar_times_s)]
                with ds.random_stream(cfg.seed + fold * 1_000_000 + epoch * 10_000 + batch, device):
                    extra = sum(ds.dar_penalty(ds.physical_weight_parameter(layer),
                                              initial_refs[name], epoch, td, cfg)
                                for name, layer in model.layers().items())
            if method == NSR:
                extra = nsr_penalty(model, cfg)
            if alpha is not None:
                # v2_lib uses scalar mean; keep equivalent regularizer for parity.
                extra = extra + cfg.attention_lambda * ((alpha - alpha.mean()) ** 2).sum(1).mean()
            loss = ce + extra
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{method} nonfinite loss at epoch={epoch+1}, batch={batch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            if gate is not None:
                torch.nn.utils.clip_grad_norm_(gate.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            before_step = [w.detach().clone() for w in weights] if optimizer_steps == 0 else None
            opt.step()
            if before_step is not None:
                first_update = {name: float((w.detach() - old).abs().max())
                                for name, w, old in zip(model.layers(), weights, before_step)}
                if not all(v > 0 for v in first_update.values()):
                    raise RuntimeError('AnalogAdam did not update every physical weight matrix')
            optimizer_steps += 1
            # Decoupled weight decay only on PCM weight tensors, not biases or scales.
            decay_analog_weights(opt, weights, cfg)
            scheduler.step()
            if opt_gate is not None: opt_gate.step()
            with torch.no_grad():
                for layer in model.layers().values():
                    tile, = list(layer.analog_tiles())
                    tile.input_range.clamp_(min=1e-3)
            bs = y.numel()
            n_samples += bs
            sum_ce += float(ce.detach()) * bs
            sum_extra += float(extra.detach()) * bs
            sum_loss += float(loss.detach()) * bs
        # Selection strictly on validation, using RAW input / no training-only gate.
        model.eval()
        val_acc, val_loss = validation_score(
            model, val_loader, device, cfg.validation_reads,
            cfg.seed + fold * 100_000 + epoch * 41,
        )
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc, best_val_loss = val_acc, val_loss
            best_epoch = epoch + 1
            best_state = ds.cpu_cloned_state(model)
            best_gate_state = ds.cpu_cloned_state(gate) if gate else None
        history.append({
            'fold': fold, 'method': method, 'epoch': epoch + 1,
            'train_ce': sum_ce / n_samples, 'train_extra': sum_extra / n_samples,
            'train_loss': sum_loss / n_samples,
            'val_accuracy': val_acc, 'val_loss': val_loss,
            'best_val_accuracy': best_val_acc, 'best_epoch': best_epoch,
            'optimizer': type(opt).__name__,
        })
        pack = {
            'fingerprint': fingerprint, 'method': method,
            'model': ds.cpu_cloned_state(model), 'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict(), 'initial_refs': initial_refs,
            'gate': ds.cpu_cloned_state(gate) if gate else None,
            'gate_optimizer': opt_gate.state_dict() if opt_gate else None,
            'next_epoch': epoch + 1, 'history': history,
            'best_model': best_state, 'best_epoch': best_epoch,
            'best_gate': best_gate_state,
            'pretrained_provenance': provenance,
            'transfer_errors': model.transfer_report,
            'optimizer_steps': optimizer_steps, 'first_weight_update': first_update,
            'best_val_accuracy': best_val_acc, 'best_val_loss': best_val_loss,
            'loader_rng': train_loader.generator.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
        }
        ds.atomic_torch_save(pack, path)
        pd.DataFrame(history).to_csv(history_path, index=False)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == cfg.epochs - 1:
            print(f'[{method}] fold {fold} epoch {epoch+1}/{cfg.epochs}'
                  f' CE={sum_ce/n_samples:.4f} val={val_acc:.2f}% best={best_epoch}'
                  f' time={(time.time()-t_start)/60:.1f}m', flush=True)

    ds.validate_epoch_history(history, cfg.epochs, method)
    if best_state is None or not 1 <= best_epoch <= cfg.epochs:
        raise RuntimeError(f'Incomplete training: {method}, epochs={len(history)}/{cfg.epochs}')
    final_state = ds.cpu_cloned_state(model)
    common = {'pretrained_provenance': provenance, 'transfer_errors': model.transfer_report,
              'optimizer_steps': optimizer_steps, 'first_weight_update': first_update,
              'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
              'optimizer_state_epoch': cfg.epochs, 'history': history,
              'selected_best_validation_epoch': best_epoch}
    ds.atomic_torch_save({
        **common, 'checkpoint_kind': 'final', 'gate': ds.cpu_cloned_state(gate) if gate else None,
        'fingerprint': fingerprint, 'model': final_state,
        **training_metadata(cfg, fold, method, cfg.epochs, len(history)),
    }, final_path)
    ds.atomic_torch_save({
        **common, 'checkpoint_kind': 'best_validation', 'gate': best_gate_state,
        'fingerprint': fingerprint, 'model': best_state,
        'val_accuracy': best_val_acc, 'val_loss': best_val_loss,
        **training_metadata(cfg, fold, method, best_epoch, len(history)),
    }, best_path)
    # Retention uses best checkpoint, but full 154e optimization was completed.
    model.load_state_dict(best_state)
    return model.eval(), pd.DataFrame(history), {
        'method': method, 'fold': fold, 'selected_epoch': best_epoch,
        'completed_epochs': len(history), 'best_val_accuracy': best_val_acc,
        'selection_split': 'validation',
    }


def exact_retention_times(cfg):
    """Exactly N points, including all notebook anchors (original produced >N)."""
    anchors = np.array([20., 60., 1800., 3600., ds.DAY, 5*ds.DAY,
                        30*ds.DAY, 180*ds.DAY, ds.YEAR, 2*ds.YEAR, 3*ds.YEAR, 5*ds.YEAR])
    anchors = np.unique(anchors)
    N = cfg.retention_points
    if N < len(anchors):
        return np.geomspace(ds.T_REF, 5*ds.YEAR, N)  # smoke only
    need = N - len(anchors)
    gaps = np.diff(np.log(anchors))
    shares = need * gaps / gaps.sum()
    allocations = np.floor(shares).astype(int)
    for k in np.argsort(-(shares - allocations))[:need - allocations.sum()]:
        allocations[k] += 1
    sections = [anchors]
    for left, right, n in zip(anchors[:-1], anchors[1:], allocations):
        if n:
            sections.append(np.geomspace(left, right, n + 2)[1:-1])
    result = np.unique(np.concatenate(sections))
    assert len(result) == N and result[0] == ds.T_REF and result[-1] == 5*ds.YEAR
    return result


def select_driftslice_bins_on_validation(hwa_snapshot, cfg, fold, val_loader, device):
    """Select B ONLY using validation samples, never MNIST test labels."""
    candidates = cfg.ablation_bins
    rows = []
    n_realizations = 1 if cfg.fast_mode else cfg.driftslice_selection_realizations
    for r in range(n_realizations):
        root = cfg.seed + fold * 1_000_000 + r * 100_000 + 9_000_000
        main = ds.build_main_pcm(hwa_snapshot, cfg, root)
        weights, conductances = ds.plain_effective_weights(main, 5*ds.YEAR, root+55)
        for b in candidates:
            bank = ds.PCMClassReferenceBank(cfg, int(b), root + 100, device)
            corr, gains = ds.driftslice_effective_weights(main, conductances, bank, 5*ds.YEAR, root+333)
            with ds.random_stream(root + 123_456, device):
                _, acc = ds.evaluate_snapshot(ds.SnapshotMLP(hwa_snapshot, corr, cfg).to(device), val_loader, device)
            rows.append({'fold': fold, 'realization': r, 'bins': int(b),
                         'validation_accuracy': acc, 'validation_error': 100-acc})
    df = pd.DataFrame(rows)
    tab = df.groupby('bins').validation_error.mean().sort_values()
    best_value = float(tab.iloc[0])
    # Deterministic smallest B on a numerical tie.
    best_bins = min(int(b) for b, v in tab.items() if abs(float(v) - best_value) < 1e-10)
    return best_bins, df


def retention_all(snapshot_map: Dict[str, dict], cfg, fold, realization, selected_bins,
                  test_loader, device):
    """Compare ALL methods, same time grid/seed policy, no test-time selection."""
    root = cfg.seed + fold * 1_000_000 + realization * 10_000
    all_times = exact_retention_times(cfg)
    data = []
    # Program one realization independently for each learned model, matched seeds.
    for method, snap in snapshot_map.items():
        main = ds.build_main_pcm(snap, cfg, root + 10)
        bank = ds.PCMClassReferenceBank(cfg, selected_bins, root + 30, device)
        reference, _ = ds.plain_effective_weights(main, ds.T_REF, root + 40)
        for ti, ts in enumerate(all_times):
            ts = float(ts)
            base, conductances = ds.plain_effective_weights(main, ts, root + 100_000 + ti*100)
            variants = {'none': (base, {'fc1': 1., 'fc2': 1.}, {})}
            if cfg.run_gdc_baselines:
                io = ds.make_hwa_rpu(cfg).forward
                for label, calibration in (('GDC-legacy-ones', ds.global_gdc_gain),
                                            ('GDC-safe-onehot', ds.safe_global_gdc_gain)):
                    gains, diagnostics = {}, {}
                    for name in ('fc1', 'fc2'):
                        kwargs = {'return_diagnostics': True} if label == 'GDC-legacy-ones' else {}
                        gains[name], report = calibration(
                            reference[name], base[name], snap[name], io,
                            root + 850_000 + ti*31 + (name == 'fc2'), device, **kwargs)
                        diagnostics.update({f'calibration_{name}_{k}': v for k, v in report.items()})
                    variants[label] = (base, gains, diagnostics)
                variants['GDC-ideal-onehot'] = (base, {
                    name: ds.ideal_global_gdc_gain(reference[name], base[name], snap[name])
                    for name in ('fc1', 'fc2')}, {})
            driftslice, gain_b = ds.driftslice_effective_weights(
                main, conductances, bank, ts, root + 500_000 + ti*101)
            variants['DriftSlice'] = (driftslice, {'fc1': 1., 'fc2': 1.}, {})
            for comp, (weights, gains, diagnostics) in variants.items():
                with ds.random_stream(root + 700_000 + ti*31, device):
                    model = ds.SnapshotMLP(snap, weights, cfg, layer_gains=gains).to(device)
                    loss, accuracy = ds.evaluate_snapshot(model, test_loader, device)
                data.append({
                    'method': method, 'compensation': comp,
                    'fold': fold, 'realization': realization,
                    't_s': ts, 'selected_bins': selected_bins,
                    'loss': loss, 'accuracy': accuracy, 'error': 100.-accuracy,
                    'gdc': comp.startswith('GDC-'), 'type': 'direct' if method == DIRECT else 'hwa',
                    'backend': 'snapshot_device_state_pcm',
                    'pcm_conductance_bounds': cfg.pcm_conductance_bounds,
                    'experiment_label': 'SMOKE TEST' if cfg.fast_mode else 'FULL PROTOCOL',
                    'gain_fc1': gains['fc1'], 'gain_fc2': gains['fc2'],
                    **diagnostics,
                    **({f'class_gain_{i}': float(v) for i, v in enumerate(gain_b.tolist())}
                       if comp=='DriftSlice' else {}),
                })
    return pd.DataFrame(data)


def summarize_retention(df):
    by_fold = df.groupby(['method','compensation','t_s','fold'], as_index=False).error.mean()
    out = by_fold.groupby(['method','compensation','t_s'], as_index=False).error.agg(
        mean_error='mean', sd_across_folds='std', n_training_repeats='count'
    )
    within = df.groupby(['method','compensation','t_s','fold']).error.std().groupby(
        ['method','compensation','t_s']).apply(lambda x: float(np.sqrt(np.nanmean(x.to_numpy()**2)))).rename('rms_hardware_sd').reset_index()
    return out.merge(within, on=['method','compensation','t_s'], how='left')


def plot_all_results(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    fig, ax = plt.subplots(figsize=(11, 6))
    methods = summary.method.unique()
    colors = colormaps['tab10'].resampled(max(3,len(methods)))
    linestyles = {'none': '-', 'GDC-legacy-ones': '--', 'GDC-safe-onehot': '-.',
                 'GDC-ideal-onehot': (0, (1, 1)), 'DriftSlice': ':'}
    for i, method in enumerate(methods):
        for comp in linestyles:
            z = summary[(summary.method==method)&(summary.compensation==comp)].sort_values('t_s')
            if z.empty: continue
            ax.plot(z.t_s, z.mean_error, label=f'{method} / {comp}', ls=linestyles[comp], color=colors(i))
    ax.set_xscale('log')
    ax.set_xlabel('Retention time after programming (s, paper-absolute)')
    ax.set_ylabel('MNIST test classification error (%)')
    smoke = 'SMOKE TEST — ' if 'experiment_label' in summary and (summary.experiment_label == 'SMOKE TEST').all() else ''
    ax.set_title(smoke + 'Matched PCM retention: direct FP32 transfer and HWA methods')
    ax.grid(alpha=.2)
    ax.legend(fontsize=7,ncol=2)
    fig.tight_layout(); fig.savefig(output/'retention_all_methods.png',dpi=170); fig.savefig(output/'retention_all_methods.pdf');plt.close(fig)


def enable_source_provenance():
    """Make all training/resume fingerprints sensitive to both modules."""
    if getattr(ds.scientific_source_hash, '_unified_provenance', False):
        return
    base_hash = ds.scientific_source_hash
    def combined_hash():
        h = hashlib.sha256(base_hash().encode())
        for file in (Path(__file__), Path(ds.__file__)):
            h.update(file.read_bytes())
        return h.hexdigest()
    combined_hash._unified_provenance = True
    ds.scientific_source_hash = combined_hash


def run(cfg: ds.Config, base_dir: Path):
    enable_source_provenance()
    cfg = cfg.resolved()
    print('SMOKE TEST: shortened development protocol' if cfg.fast_mode else 'FULL PROTOCOL', flush=True)
    for method in cfg.methods:
        if method not in ALL_TRAIN_METHODS:
            raise ValueError(f'Unknown method {method!r}')
    device = ds.check_environment(cfg)
    # The original DriftSlice notebook preflight checks native tile state
    # cloning, transfer, time convention, DAR gradients, and snapshot MVM.
    ds.preflight(replace(cfg, fast_mode=True))
    raw_dir = ds.resolve_mnist_raw_dir(cfg, base_dir)
    out, fingerprint = ds.experiment_directory(
        cfg, base_dir, data_identity=ds.mnist_file_hashes(raw_dir))
    histories, digitals, comparisons, selections, bin_ablation, results = [], [], [], [], [], []
    for fold in range(cfg.train_repeats):
        train, val, test, split = ds.build_loaders(cfg, base_dir, fold)
        ds.atomic_torch_save(split, out / f'fold{fold}_split.pt')
        digital, digital_hist = ds.train_digital_pretrained(cfg, fold, train, val, device, out, fingerprint)
        assert len(digital_hist) == cfg.digital_epochs, 'Stale digital pretrain epoch count'
        with ds.random_stream(cfg.seed+fold, device):
            _, digital_test_acc = ds.evaluate_model(digital, test, device)
        direct = direct_baseline(cfg, digital, cfg.seed+fold, device, val_loader=val)
        ds.atomic_torch_save({
            'fingerprint': fingerprint, 'model': ds.cpu_cloned_state(direct),
            'digital_epochs': cfg.digital_epochs, 'analog_epochs': 0,
            'method': DIRECT, 'selection_split': 'validation',
            'optimizer_steps': 0, 'pretrained_provenance': direct.pretrained_provenance,
            'transfer_errors': direct.transfer_report,
            'experiment_label': 'SMOKE TEST' if cfg.fast_mode else 'FULL PROTOCOL',
        }, out / f'fold{fold}_{DIRECT}_analog_checkpoint.pt')
        with ds.random_stream(cfg.seed+fold, device):
            _, direct_test_acc = ds.evaluate_model(direct, test, device)
        comparisons.append({'fold':fold,'method':'FP32-Digital','test_accuracy':digital_test_acc,
                            'trained_epochs':cfg.digital_epochs})
        comparisons.append({'fold':fold,'method':DIRECT,'test_accuracy':direct_test_acc,
                            'trained_epochs':0})
        snapshot_map = {DIRECT: ds.snapshot_analog_model(direct)}
        del direct
        for method in cfg.methods:
            model, hist, metadata = train_hwa_all(method,cfg,fold,digital,train,val,device,out,fingerprint)
            histories.append(hist)
            selections.append(metadata)
            with ds.random_stream(cfg.seed+fold, device):
                _, acc = ds.evaluate_model(model,test,device)
            comparisons.append({'fold':fold,'method':method,'test_accuracy':acc,
                                'trained_epochs':metadata['completed_epochs'],
                                'selected_epoch':metadata['selected_epoch']})
            snapshot_map[method] = ds.snapshot_analog_model(model)
            del model
        del digital
        digital_hist.assign(fold=fold).to_csv(out/f'fold{fold}_digital_history.csv',index=False)
        # Select DriftSlice complexity ONCE per fold using HWA validation only.
        selection_method = HWA if HWA in snapshot_map else cfg.methods[0]
        bs, ablation = select_driftslice_bins_on_validation(
            snapshot_map[selection_method],cfg,fold,val,device)
        bin_ablation.append(ablation)
        (out/f'fold{fold}_selected_bins.json').write_text(json.dumps({
            'fold':fold,'bins':bs,'selection_method':selection_method,
            'selection': 'validation_only_5yr',
        },indent=2))
        for realization in range(cfg.realizations):
            shard = out/f'fold{fold}_realization{realization}_unified.pt'
            if shard.exists():
                pack = torch.load(shard,map_location='cpu',weights_only=False)
                if pack.get('fingerprint')!=fingerprint or pack.get('selected_bins')!=bs:
                    raise RuntimeError('Stale retention shard: fingerprint/selection mismatch')
                df=pack['results']
            else:
                df=retention_all(snapshot_map,cfg,fold,realization,bs,test,device)
                ds.atomic_torch_save({'fingerprint':fingerprint,'selected_bins':bs,'results':df},shard)
            results.append(df)
            print(f'Retention fold {fold+1}/{cfg.train_repeats}, realization {realization+1}/{cfg.realizations}',flush=True)
    history=pd.concat(histories,ignore_index=True)
    retention=pd.concat(results,ignore_index=True)
    summary=summarize_retention(retention)
    summary['experiment_label'] = 'SMOKE TEST' if cfg.fast_mode else 'FULL PROTOCOL'
    history.to_csv(out/'training_history.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'digital_vs_hwa_test.csv',index=False)
    pd.DataFrame(selections).to_csv(out/'best_validation_selection.csv',index=False)
    pd.concat(bin_ablation).to_csv(out/'driftslice_bin_ablation_validation.csv',index=False)
    retention.to_csv(out/'retention_all_methods.csv',index=False)
    calibration_columns = [c for c in retention if c.startswith('calibration_')]
    retention.loc[retention.compensation.isin(['GDC-legacy-ones', 'GDC-safe-onehot']),
                  ['method', 'compensation', 'fold', 'realization', 't_s'] + calibration_columns
                  ].to_csv(out/'gdc_calibration_diagnostics.csv', index=False)
    summary.to_csv(out/'retention_summary.csv',index=False)
    plot_all_results(summary,out)
    print('OUTPUT',out)
    return out


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('smoke','full'),default='smoke')
    p.add_argument('--methods',nargs='+',choices=ALL_TRAIN_METHODS,default=list(ALL_TRAIN_METHODS))
    p.add_argument('--mnist-root',default='data/MNIST/raw')
    p.add_argument('--out',default='.')
    p.add_argument('--run-id',default=None,help='Use an explicit ID to resume; default creates a fresh run')
    p.add_argument('--cpu',action='store_true',help='Allow CPU-only development/smoke')
    p.add_argument('--no-download',action='store_true')
    p.add_argument('--nsr-lambda',type=float,default=0.1)
    return p.parse_args()


def main():
    a=parse_args()
    cfg=ds.Config(
        fast_mode=a.mode=='smoke', require_cuda=not a.cpu,
        methods=tuple(a.methods),data_root=a.mnist_root,
        allow_download=not a.no_download,
        run_id=a.run_id or f'verified_{a.mode}_{time.time_ns()}',
        nsr_lambda=a.nsr_lambda,
    )
    run(cfg,Path(a.out).expanduser().resolve())


if __name__=='__main__':main()
