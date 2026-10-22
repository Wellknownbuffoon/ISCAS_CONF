"""Generate self-contained retention-only copies without changing scientific functions."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
EXPORT = (ROOT/'retention_accuracy_export_cell.py').read_text()
SOURCES = ['DriftSlice_ISCAS_P0_coordinator_FIXED.ipynb',
           'DriftSlice_ISCAS_P1_worker_HWA_FIXED.ipynb',
           'DriftSlice_ISCAS_P2_worker_DAR_FIXED.ipynb',
           'DriftSlice_ISCAS_P3_worker_DriftAttention_FIXED.ipynb',
           'DriftSlice_ISCAS_P4_worker_NSR_FIXED.ipynb']

for name in SOURCES:
    notebook = json.loads((ROOT/name).read_text())
    coordinator = 'P0_' in name
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if 'EXECUTION CONTROL' in source:
            for flag, value in {'RUN_FULL_TRAINING': False, 'RUN_FULL_RETENTION': True,
                                'RUN_REFERENCE_ABLATION': False, 'RUN_MECHANISM_DIAGNOSTICS': False,
                                'GENERATE_PAPER_RESULTS': False}.items():
                source, count = re.subn(rf'^{flag}(\s*=\s*)(True|False)',
                                       rf'{flag}\g<1>{value}', source, flags=re.M)
                assert count == 1, (name, flag)
            source = re.sub(r'^# Each flag controls real execution\..*$',
                            '# Retention-only execution: training, full ablation, diagnostics and full paper generation disabled.',
                            source, flags=re.M)
        if coordinator and '# coordinator: verify all workers' in source:
            source = source.replace('    if coordinator_post_training(COORD_EXP):', '''    # This notebook must never start a new training run. The existing coordinator
    # helper reuses completed pairs only when both final and best files exist.
    _ret_needed = []
    for _ret_f in range(COORD_EXP.cfg.train_repeats):
        _ret_dir = COORD_EXP.train_dir / f'fold{_ret_f}'
        _ret_names = ['digital_final.pt', 'digital_best_validation.pt', f'{DIRECT}_analog_checkpoint.pt']
        _ret_names += [f'{_ret_m}_{_ret_k}.pt' for _ret_m in COORD_EXP.methods
                       for _ret_k in ('final', 'best_validation')]
        _ret_needed += [str(_ret_dir / _ret_n) for _ret_n in _ret_names if not (_ret_dir / _ret_n).exists()]
    if _ret_needed:
        raise ScientificCheckError('Retention-only coordinator needs completed training checkpoints; '
                                   'no training was started. Missing: ' + ', '.join(_ret_needed[:8]))
    if coordinator_post_training(COORD_EXP):''')
        if 'runtime.unassign()' in source:
            source = '# Keep the runtime connected so retention CSV/plot exports remain accessible.\n'
        if 'reviewer_audit.md' in source and source.startswith('pd.set_option('):
            source = ('# Full-paper reviewer audits require stages intentionally excluded here.\n'
                      'print("Retention-only workflow: see the final cell for accuracy CSVs, plots and completeness status.")\n')
        cell['source'] = source.splitlines(keepends=True)
    title = name.removesuffix('_FIXED.ipynb')
    notebook['cells'][0]['source'] = f'''# {title}: retention accuracy only

Full training, reference-budget ablation, mechanism diagnostics and full-paper generation are OFF.
The original Config, trained checkpoints, PCM simulation, compensation modes, retention grid,
three folds and 30 PCM realizations per fold are unchanged. Existing compatible retention shards
are reused. Common scientific functions and the smoke gate are unchanged; an existing smoke pass
is reused when its code, configuration, runtime and GPU match. A fresh small smoke test may run
if that evidence is absent. That test is separate from the full-run stage switches above.

First stop any older ablation cell. Run P0 once after all worker training has finished to verify
checkpoints and select DriftSlice B on validation. Then run the retention-only workers on separate
GPU sessions. Finally rerun P0 to combine their saved retention results into accuracy curves.
Keep the same Drive root, run ID, package versions and Config.

The last cell exports raw accuracy records, fold means, mean/SD CSVs and PNG/PDF plots under
`results/full/<run_id>/retention_accuracy/<retention-fingerprint>/<role>/` on Drive.
Missing or invalid results produce NaN means and exports marked INCOMPLETE, never a full-paper FAIL
for intentionally omitted ablation/diagnostics. Historical outputs below precede this workflow.
'''.splitlines(keepends=True)
    notebook['cells'].extend([
        {'cell_type': 'markdown', 'metadata': {}, 'source': [
            '## Test accuracy versus retention time\n',
            'Read only the saved retention shards and export curves; no ablation or diagnostics are required.\n']},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
         'source': EXPORT.splitlines(keepends=True)},
    ])
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'{name}/cell{i}', 'exec')
    target = ROOT/name.replace('_FIXED.ipynb', '_RETENTION_ONLY.ipynb')
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
    print(target.name)
