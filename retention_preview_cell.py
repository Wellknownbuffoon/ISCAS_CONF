# RETENTION PREVIEW: first N programmed-device realizations, across EVERY fold.
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

# RETENTION-ONLY ACCURACY EXPORT
# This cell only reads saved retention/checkpoint files. It defines no scientific
# functions and does not change training, PCM evaluation or the existing smoke gate.
import matplotlib.pyplot as plt

_ret_exp = globals().get('COORD_EXP') if PARALLEL_ROLE == 'coordinator' else globals().get('FULL_EXP')
if _ret_exp is None:
    _ret_exp = open_full_experiment(role='coordinator-full' if PARALLEL_ROLE == 'coordinator' else None)
_ret_cfg = _ret_exp.cfg
_ret_models = list(_ret_exp.models)
_ret_comps = list(compensations_for(_ret_cfg))
_ret_times = np.asarray(exact_retention_times(_ret_cfg), dtype=float)
_ret_keys = ['fold', 'method', 'compensation', 'realization', 't_index']
_ret_expected = pd.MultiIndex.from_product(
    [range(_ret_cfg.train_repeats), _ret_models, _ret_comps, range(PREVIEW_REALIZATIONS), range(len(_ret_times))],
    names=_ret_keys)
_ret_df, _ret_meta = collect_shards(_ret_exp.ret_dir, _ret_exp.fp['retention'], _ret_models)
# Summarize only the requested prefix; retain all other full-run shards on disk.
if len(_ret_df):
    _ret_df = _ret_df[_ret_df['realization'] < PREVIEW_REALIZATIONS].copy()
_ret_meta = [item for item in _ret_meta
             if int(Path(item['file']).stem.removeprefix('real')) < PREVIEW_REALIZATIONS]
_ret_identities = {}
for _ret_item in _ret_meta:
    _ret_parts = Path(_ret_item['file']).parts
    _ret_fold = int(_ret_parts[0].removeprefix('fold'))
    _ret_method = _ret_parts[1]
    _ret_pair = (_ret_fold, _ret_method)
    if _ret_pair not in _ret_identities:
        _ret_checkpoint = _ret_exp.train_dir / f'fold{_ret_fold}' / (
            f'{DIRECT}_analog_checkpoint.pt' if _ret_method == DIRECT else f'{_ret_method}_best_validation.pt')
        _ret_pack = torch_load(_ret_checkpoint)
        verify_pack(_ret_pack, _ret_exp.fp['train'],
                    'fp32_direct' if _ret_method == DIRECT else 'hwa_best_validation', _ret_fold, _ret_checkpoint)
        if _ret_pack.get('method') != _ret_method:
            raise ScientificCheckError(f'{_ret_checkpoint}: method mismatch')
        _ret_identities[_ret_pair] = _ret_pack['model_identity']
    if _ret_item.get('model_identity') != _ret_identities[_ret_pair]:
        raise ScientificCheckError(f"{_ret_item['file']}: retention model identity differs from the checkpoint")

if len(_ret_df):
    if not set(_ret_keys + ['t_s', 'accuracy', 'split']).issubset(_ret_df.columns):
        raise ScientificCheckError('Retention export: missing required columns')
    if _ret_df.duplicated(_ret_keys).any():
        raise ScientificCheckError('Retention export: duplicate records')
    if not _ret_df['split'].eq('test').all():
        raise ScientificCheckError('Retention export requires official test-set results')
    _ret_index = pd.MultiIndex.from_frame(_ret_df[_ret_keys])
    if not _ret_index.isin(_ret_expected).all():
        raise ScientificCheckError('Retention export: unexpected fold, method, compensation, realization or timestamp index')
    if not np.array_equal(_ret_df['t_s'].to_numpy(), _ret_times[_ret_df['t_index'].to_numpy(dtype=int)]):
        raise ScientificCheckError('Retention export: timestamps differ from the configured grid')
    _ret_values = _ret_df['accuracy'].to_numpy(dtype=float)
    if np.isinf(_ret_values).any() or ((_ret_values < 0) | (_ret_values > 100)).any():
        raise ScientificCheckError('Retention export: invalid accuracy outside [0, 100]')
else:
    _ret_df = pd.DataFrame(columns=_ret_keys + ['t_s', 'accuracy', 'split'])

# Explicit expected grid prevents missing folds/realizations/whole curves from
# producing optimistic averages. NaN accuracy remains NaN; it is never filled.
_ret_grid = _ret_df.set_index(_ret_keys).reindex(_ret_expected).reset_index()
_ret_grid['t_s'] = _ret_times[_ret_grid['t_index'].to_numpy(dtype=int)]
_ret_grid['accuracy'] = pd.to_numeric(_ret_grid['accuracy'])
_ret_summary = two_stage_summary(_ret_grid, ['method', 'compensation', 't_index', 't_s'], value='accuracy',
                                expected_folds=_ret_cfg.train_repeats, expected_reals=PREVIEW_REALIZATIONS)
_ret_summary = _ret_summary.rename(columns={
    'mean': 'test_accuracy_mean_pct', 'sd': 'test_accuracy_sd_across_folds_pp',
    'invalid_records': 'missing_or_invalid_records'})
_ret_summary['complete'] = _ret_summary['missing_or_invalid_records'].eq(0)
_ret_summary['t_days'] = _ret_summary['t_s'] / DAY
_ret_summary['t_years'] = _ret_summary['t_s'] / YEAR
_ret_fold_means = fold_means(_ret_grid, ['method', 'compensation', 't_index', 't_s'], value='accuracy')
_ret_fold_means = _ret_fold_means.rename(columns={'accuracy': 'test_accuracy_mean_over_realizations_pct'})
_ret_complete = bool(len(_ret_df) == len(_ret_expected) and _ret_summary['complete'].all())
_ret_out = _ret_exp.run_dir / 'retention_preview' / f'first_{PREVIEW_REALIZATIONS}' / _ret_exp.fp['retention'][:12] / role_tag()
_ret_out.mkdir(parents=True, exist_ok=True)
atomic_write_csv(_ret_df, _ret_out / 'preview_test_accuracy_vs_retention_raw.csv')
atomic_write_csv(_ret_fold_means, _ret_out / 'preview_test_accuracy_vs_retention_fold_means.csv')
atomic_write_csv(_ret_summary, _ret_out / 'preview_test_accuracy_vs_retention_summary.csv')
atomic_write_text(_ret_out / 'retention_preview_status.json', json.dumps({
    'complete': False, 'preview': True, 'preview_complete': _ret_complete, 'records_present': len(_ret_df), 'records_expected': len(_ret_expected),
    'missing_records': len(_ret_expected) - len(_ret_df),
    'invalid_accuracy_records': int(_ret_df['accuracy'].isna().sum()),
    'models': _ret_models, 'folds': _ret_cfg.train_repeats, 'realizations_per_fold': PREVIEW_REALIZATIONS,
    'planned_realizations_per_fold': _ret_cfg.realizations,
    'aggregation': 'mean across realization means per fold; sample SD across fold means',
    'train_fingerprint': _ret_exp.fp['train'], 'retention_fingerprint': _ret_exp.fp['retention'],
    'source_shards': [_ret_item['file'] for _ret_item in _ret_meta],
    'ablation_required': False, 'mechanism_diagnostics_required': False,
}, indent=2))

for _ret_comp in _ret_comps:
    _ret_fig, _ret_ax = plt.subplots(figsize=(8, 4.8))
    _ret_plotted = False
    for _ret_method in _ret_models:
        _ret_curve = _ret_summary[(_ret_summary['method'] == _ret_method) &
                                  (_ret_summary['compensation'] == _ret_comp)].sort_values('t_s')
        if not _ret_curve['test_accuracy_mean_pct'].notna().any():
            continue
        _ret_x = _ret_curve['t_s'].to_numpy(dtype=float)
        _ret_y = _ret_curve['test_accuracy_mean_pct'].to_numpy(dtype=float)
        _ret_sd = _ret_curve['test_accuracy_sd_across_folds_pp'].to_numpy(dtype=float)
        _ret_ax.plot(_ret_x, _ret_y, marker='o', markersize=3, label=_ret_method)
        _ret_ax.fill_between(_ret_x, _ret_y - _ret_sd, _ret_y + _ret_sd, alpha=.15)
        _ret_plotted = True
    _ret_ax.set_xscale('log')
    _ret_ax.set_xticks([20, 3600, DAY, 30*DAY, YEAR, 5*YEAR])
    _ret_ax.set_xticklabels(['20 s', '1 h', '1 d', '1 mo', '1 y', '5 y'])
    _ret_ax.set_xlabel('Retention time (log scale)')
    _ret_ax.set_ylabel('Test accuracy (%)')
    _ret_ax.set_ylim(0, 100)
    _ret_ax.set_title(f"PREVIEW ({PREVIEW_REALIZATIONS}/{_ret_cfg.realizations} realizations per fold): {_ret_comp}" +
                      ('' if _ret_complete else ' [INCOMPLETE]'))
    _ret_ax.grid(True, alpha=.25)
    if _ret_plotted:
        _ret_ax.legend(fontsize='small')
    else:
        _ret_ax.text(.5, .5, 'Waiting for complete retention results', ha='center', transform=_ret_ax.transAxes)
    _ret_fig.tight_layout()
    _ret_fig.savefig(_ret_out / f'preview_accuracy_vs_retention_{_ret_comp}.png', dpi=180)
    _ret_fig.savefig(_ret_out / f'preview_accuracy_vs_retention_{_ret_comp}.pdf')
    plt.close(_ret_fig)

print(('PREVIEW COMPLETE' if _ret_complete else 'PREVIEW INCOMPLETE') +
      f' retention accuracy export: {len(_ret_df)}/{len(_ret_expected)} records')
print('CSV files and PNG/PDF accuracy curves:', _ret_out)
print('Ablation and mechanism diagnostics are not required for these exports.')

print("Preview only: fewer programmed-device samples; full-run completeness is unchanged.")
