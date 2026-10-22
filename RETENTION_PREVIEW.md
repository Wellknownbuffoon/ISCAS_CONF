# Faster preview of retention accuracy

The full run evaluates 30 PCM realizations per fold. The preview evaluates the
first **one realization in every fold**, at the same timestamps and compensation
modes. It reuses existing training and retention shards and labels its CSVs and
charts **PREVIEW**. Fewer device samples provide less reliable estimates of
programming/drift variability; a preview is not the completed 30-realization study.

Using the user's initial 32.44-hour P1 estimate as a rough proportional guide:

| Realizations per fold evaluated | Approximate evaluation work/time |
| --- | --- |
| 1 (preview default) | 1/30; about 1.1 hours |
| 3 | 3/30; about 3.2 hours |
| 30 (full protocol) | 30/30; about 32.4 hours |

These are not GPU benchmarks or guarantees. Startup/deployment checks and saved
shard reuse affect the actual remaining time. The first ETA is also uncertain.

## Use it

1. Stop the original running retention cell before starting a preview for the same
   worker. Do not run two writers for the same model/realization concurrently.
2. Open the matching `*_RETENTION_PREVIEW.ipynb` copy. It keeps the same Drive root,
   run ID, worker model assignment, Config and scientific functions as the existing
   corrected workers. Full training, retention, ablation, diagnostics and paper
   stage switches are disabled; the last cell performs the limited preview.
3. Run the notebook from the top. The normal environment/smoke checks remain.
   P0's HWA validation-selected B must already exist. No new full training is run.
   Alternatively, with definitions already loaded in the current worker runtime,
   paste `retention_preview_cell.py` into one new cell and run that cell only.
4. Leave `PREVIEW_REALIZATIONS = 1` for the smallest preview. Use 3 for more samples.
   Do **not** change `Config.realizations`, folds, timestamps or batch sizes.
5. Read the printed export path. Files are under
   `results/full/<run_id>/retention_preview/first_N/<retention-fingerprint>/<role>/`.
   PNG/PDF plots and CSV filenames begin with `preview_`. Status explicitly records
   `preview: true`, the samples used, the planned 30 realizations, and
   `complete: false` for the full study. `preview_complete` refers only to that subset.

P1 has two models: with three folds and one realization the preview needs six
shards (288 accuracy records for 12 times and four compensation modes). P2–P4
each have one model, requiring three shards (144 records) with the same settings.
All folds are included even when the original run stopped during fold zero.

## Preservation of the full run

The preview limits execution indices only. It passes the **unchanged Config** and
original fold/realization indices to the original retention function. It does not
change fingerprints, seeds, inference settings, trained weights or reference B.
Compatible saved shards are skipped. Newly computed shards are ordinary native
retention shards in the original retention directory, so the full run can reuse
them later. Other saved realizations are retained and excluded only from the
separate preview aggregation. The original complete-result exports remain strict
and will still report missing full-run records until all 30 are evaluated.

## Verification

Two CPU regression tests cover all four preview copies and a real AIHWKit 1.1.0
ResNet fixture using synthetic images and an existing two-epoch fixture checkpoint.
Every scientific function, Config value, stage fingerprint and smoke scope matches
the retention-only copy. Native full-grid retention data and PCM metadata for
realization zero match the preview exactly. A second execution reuses its saved
shard without recomputing it. The other realization and training checkpoint bytes
are unchanged. The export uses only the selected prefix and identifies itself as
preview rather than full completion. No Colab GPU timing or actual user Drive
checkpoints were tested here. Synthetic accuracy values are not benchmark results.

```bash
python make_retention_preview_notebooks.py
python tests/test_retention_preview.py --out /tmp/new-preview-tests \
  --hwa-checkpoint /path/to/synthetic/PCM-HWA_best_validation.pt
```

The checkpoint argument enables the real native-equivalence test. It expects the
two-epoch CPU fixture produced by `tests/test_worker_notebooks.py` (train fingerprint
`fixture-fp`); it is not intended to load a real Colab checkpoint as a test fixture.
Results are in `verification/retention_preview_results.json`.
