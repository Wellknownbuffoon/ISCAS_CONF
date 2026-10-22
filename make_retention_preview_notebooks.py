"""Build an optional first-N-realizations preview without altering experiment Config."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONT = '''# RETENTION PREVIEW: first N programmed-device realizations, across EVERY fold.
# The experiment Config and fingerprints stay unchanged. New shards are ordinary
# full-run shards, so the full retention job can reuse them later.
PREVIEW_REALIZATIONS = 1  # use 3 for a larger preview; leave Config.realizations unchanged
RUN_FULL_TRAINING = False
RUN_FULL_RETENTION = False
RUN_REFERENCE_ABLATION = False
RUN_MECHANISM_DIAGNOSTICS = False
GENERATE_PAPER_RESULTS = False

begin_check_session("retention_preview")
require_smoke_pass(OUTPUT_DIR)
FULL_EXP = open_full_experiment()
_pv_exp = FULL_EXP
_pv_cfg, _pv_dev, _pv_ops = _pv_exp.cfg, _pv_exp.device, ops(_pv_exp)
if isinstance(PREVIEW_REALIZATIONS, bool) or not isinstance(PREVIEW_REALIZATIONS, int):
    raise ValueError("PREVIEW_REALIZATIONS must be an integer")
if not 1 <= PREVIEW_REALIZATIONS <= _pv_cfg.realizations:
    raise ValueError(f"PREVIEW_REALIZATIONS must be within 1..{_pv_cfg.realizations}")
_pv_ready, _pv_missing = training_complete(_pv_exp)
if not _pv_ready:
    raise ScientificCheckError("Preview requires completed training: " + ", ".join(_pv_missing[:6]))
_pv_bins_ready, _pv_bins_missing = selected_bins_ready(_pv_exp)
if not _pv_bins_ready:
    raise ScientificCheckError("Run P0's HWA validation-selection cell first; missing " + ", ".join(_pv_bins_missing))

verify_retention_grid(_pv_cfg, "preview")
_pv_total = _pv_cfg.train_repeats * PREVIEW_REALIZATIONS * len(_pv_exp.models)
_pv_done = _pv_skipped = 0
_pv_start = time.time()
for _pv_fold in range(_pv_cfg.train_repeats):
    _, _pv_val, _pv_test, _ = _pv_ops.loaders(_pv_cfg, _pv_exp.base_dir, _pv_fold, _pv_dev, verify_fast=False)
    _pv_snaps = _pv_ops.load_snapshots(_pv_exp, _pv_fold, _pv_val)
    _pv_bins = fold_selected_bins(_pv_exp, _pv_fold, _pv_snaps, _pv_val)
    for _pv_real in range(PREVIEW_REALIZATIONS):
        for _pv_method in _pv_exp.models:
            _pv_path = _pv_exp.ret_dir / f"fold{_pv_fold}" / _pv_method / f"real{_pv_real:02d}.pt"
            if _shard_ok(_pv_path, _pv_exp.fp["retention"], _pv_snaps[_pv_method]["identity"], {"selected_bins": _pv_bins}):
                _pv_skipped += 1
                continue
            _pv_t0 = time.time()
            _pv_df, _pv_meta = _pv_ops.retention(_pv_method, _pv_snaps[_pv_method], _pv_cfg, _pv_fold, _pv_real,
                                               _pv_bins, _pv_test, _pv_dev)
            if _pv_meta["state_hash_start"] != _pv_meta["state_hash_end"]:
                raise ScientificCheckError("Programmed PCM states changed during preview retention")
            atomic_torch_save({"fp": _pv_exp.fp["retention"], "model_identity": _pv_snaps[_pv_method]["identity"],
                               "selected_bins": _pv_bins, "results": _pv_df, "meta": _pv_meta,
                               "gpu": _gpu_name(), "seconds": time.time() - _pv_t0}, _pv_path)
            _pv_done += 1
        print(f"PREVIEW fold {_pv_fold+1}/{_pv_cfg.train_repeats}, realization {_pv_real+1}/{PREVIEW_REALIZATIONS}: "
              f"{_pv_done} new, {_pv_skipped} reused; ETA {_eta(_pv_done, _pv_total-_pv_skipped, _pv_start)}", flush=True)
    del _pv_snaps
    if _pv_dev.type == "cuda":
        torch.cuda.empty_cache()
end_stage_checks(_pv_exp, "retention_preview")

'''

export = (ROOT/'retention_accuracy_export_cell.py').read_text()
export = export.replace('range(_ret_cfg.realizations)', 'range(PREVIEW_REALIZATIONS)')
export = export.replace('expected_reals=_ret_cfg.realizations', 'expected_reals=PREVIEW_REALIZATIONS')
export = export.replace('_ret_identities = {}', '''# Summarize only the requested prefix; retain all other full-run shards on disk.
if len(_ret_df):
    _ret_df = _ret_df[_ret_df['realization'] < PREVIEW_REALIZATIONS].copy()
_ret_meta = [item for item in _ret_meta
             if int(Path(item['file']).stem.removeprefix('real')) < PREVIEW_REALIZATIONS]
_ret_identities = {}''')
export = export.replace("/ 'retention_accuracy' /", "/ 'retention_preview' / f'first_{PREVIEW_REALIZATIONS}' /")
export = export.replace("'complete': _ret_complete,", "'complete': False, 'preview': True, 'preview_complete': _ret_complete,")
export = export.replace("'realizations_per_fold': _ret_cfg.realizations,", "'realizations_per_fold': PREVIEW_REALIZATIONS,\n    'planned_realizations_per_fold': _ret_cfg.realizations,")
export = export.replace('test_accuracy_vs_retention_', 'preview_test_accuracy_vs_retention_')
export = export.replace('accuracy_vs_retention_{_ret_comp}', 'preview_accuracy_vs_retention_{_ret_comp}')
export = export.replace('retention_export_status.json', 'retention_preview_status.json')
export = export.replace('f"{_ret_comp}: mean accuracy and SD across folds"',
                        'f"PREVIEW ({PREVIEW_REALIZATIONS}/{_ret_cfg.realizations} realizations per fold): {_ret_comp}"')
export = export.replace("print(('COMPLETE' if _ret_complete else 'INCOMPLETE') +", "print(('PREVIEW COMPLETE' if _ret_complete else 'PREVIEW INCOMPLETE') +")
export += '\nprint("Preview only: fewer programmed-device samples; full-run completeness is unchanged.")\n'
cell = FRONT + export
compile(cell, 'retention_preview_cell.py', 'exec')
(ROOT/'retention_preview_cell.py').write_text(cell)

for role in ('P1_worker_HWA', 'P2_worker_DAR', 'P3_worker_DriftAttention', 'P4_worker_NSR'):
    n = json.loads((ROOT/f'DriftSlice_ISCAS_{role}_RETENTION_ONLY.ipynb').read_text())
    s = ''.join(n['cells'][3]['source'])
    assert 'RUN_FULL_RETENTION        = True' in s
    n['cells'][3]['source'] = s.replace('RUN_FULL_RETENTION        = True', 'RUN_FULL_RETENTION        = False').splitlines(keepends=True)
    n['cells'][0]['source'] = f'''# {role}: optional retention preview

This copy evaluates the first **one PCM realization per fold** by default, at all
configured retention times and compensation modes. It uses completed training and
existing P0 validation-selected B. It does not retrain or run full ablation/diagnostics.

Stop the original worker cell before running this copy; use the same Drive, run ID,
packages and GPU type. Config, scientific functions, fingerprints, seeds and the
planned 30-realization experiment are unchanged. Existing compatible shards are
reused, and newly calculated shards remain reusable by the full retention run.

The last cell controls preview size: `PREVIEW_REALIZATIONS = 1` (or 3).
It visits every fold, then exports clearly labeled PREVIEW CSVs and PNG/PDF charts
in a separate `retention_preview/first_N/` directory. Preview statistics use only
those N realizations; they do not satisfy the full experiment's completeness criteria.
Fewer device samples give less reliable estimates of programming/drift variability.
Historical cell outputs below are not results of this preview.
'''.splitlines(keepends=True)
    n['cells'][-2]['source'] = ['## Retention preview\n', 'Run the following cell after definitions and the smoke gate.\n']
    n['cells'][-1]['source'] = cell.splitlines(keepends=True)
    for i, c in enumerate(n['cells']):
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), f'{role}/cell{i}', 'exec')
    target = ROOT/f'DriftSlice_ISCAS_{role}_RETENTION_PREVIEW.ipynb'
    target.write_text(json.dumps(n, indent=1, ensure_ascii=False) + '\n')
    print(target.name)
