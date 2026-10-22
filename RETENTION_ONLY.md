# Test accuracy versus retention time

Reference-budget ablation and mechanism diagnostics are not required for this
output. The retention-only copies disable full training, full ablation, mechanism
diagnostics and the full-paper generator. They preserve all scientific functions,
the original smoke gate, Config, three folds, 30 PCM realizations per fold, the
12 ResNet retention times, compensation modes and training/retention fingerprints.
They can reuse compatible trained checkpoints and existing retention shards.

## Run order

1. Stop any currently running ablation cell. Completed shards can stay on Drive.
2. Run `DriftSlice_ISCAS_P0_coordinator_RETENTION_ONLY.ipynb` after all workers have
   completed training. It verifies existing completed checkpoints and selects
   DriftSlice B on validation. If final/best checkpoint pairs are missing, it stops
   before invoking the training helper. It never starts new full training.
3. Run the P1–P4 `*_RETENTION_ONLY.ipynb` workers, each in its own GPU session.
   Keep the same Drive root, run ID, configuration and package versions. Their
   existing worker model assignments are preserved. The retention stage skips
   completed compatible shards and computes the remaining evaluations.
4. Rerun the P0 retention-only notebook after the workers finish to combine all
   models into the exported curves. Individual workers also export their own
   assigned models.

Full-stage settings are:

```python
RUN_FULL_TRAINING         = False
RUN_FULL_RETENTION        = True
RUN_REFERENCE_ABLATION    = False
RUN_MECHANISM_DIAGNOSTICS = False
GENERATE_PAPER_RESULTS   = False
```

The existing small smoke-test gate stays enabled. A recorded pass is reusable when
scientific functions, configuration, GPU and environment match. If that evidence
is absent or incompatible, a fresh small smoke test is required; that separate
test still exercises its original synthetic/reduced training and verification
stages. No full ablation or diagnostics stage is launched by these copies.

The original full-paper report requires ablation and diagnostics for its Tables
and reviewer audit. It is disabled here; the retention export does not use those
files or require their completeness. Older full-paper FAIL reports can remain
as historical records without blocking retention exports.

## Files on Drive

The last cell exports into:

```text
/content/drive/MyDrive/ISCAS_DriftSlice/results/full/
  driftslice_iscas_colab_resnet32_cifar10_v2/
    retention_accuracy/<retention-fingerprint>/<role>/
```

- `test_accuracy_vs_retention_raw.csv`: per-fold/realization test accuracies.
- `test_accuracy_vs_retention_fold_means.csv`: mean across realizations per fold.
- `test_accuracy_vs_retention_summary.csv`: mean accuracy and sample SD across
  fold means, with retention time in seconds, days and years.
- `accuracy_vs_retention_<compensation>.png` and `.pdf`: one chart per compensation
  mode, with one curve per assigned model and an SD band.
- `retention_export_status.json`: source shards, completeness and provenance.

Missing folds, realizations, time points or entire curves are explicitly included
as missing in the expected grid. Incomplete means remain NaN; CSV status and plots
are marked INCOMPLETE. Duplicate, foreign, mislabeled split, incorrect-time and
out-of-range records are rejected. Saved checkpoint identities are checked.
The exporter reads checkpoint/shard files and writes only the separate export
directory. It does not rerun model inference or fabricate measurements.

Retention evaluation still requires computation: skipping ablation removes the
reference-budget sweep, while all requested retention evaluations remain.

## Validation

Six CPU tests pass with synthetic saved-shard fixtures. They verify all five
copies preserve scientific function definitions, stage fingerprints, Config,
worker roles and smoke scope; a complete 576-record fixture produces the expected
66% mean and 5-percentage-point SD across folds, plus eight PNG/PDF artifacts.
A missing record or an empty run stays incomplete with NaN means. Corrupt data
is rejected. Checkpoint/shard file bytes remain unchanged. The P0 guard prevents
starting its helper when full training checkpoints are absent. All code cells
compile. No new full training, GPU retention run or actual Drive checkpoint
validation was performed in this workspace.

```bash
python make_retention_only_notebooks.py
python tests/test_retention_only.py --out /tmp/new-retention-test-folder
```

The generated notebooks embed `retention_accuracy_export_cell.py`, so Colab needs
no separate Python-file download. Test results are recorded in
`verification/retention_only_results.json`.
