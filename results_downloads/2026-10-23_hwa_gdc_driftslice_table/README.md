# Corrected Table III: HWA + GDC-safe baseline versus HWA + DriftSlice

Unzip this package, open the extracted folder in MATLAB R2020a or newer, and run:

```matlab
draw_iscas_compensation_table
```

The code reads the bundled full-run fold means, opens the corrected table, and exports PNG, vector PDF, editable MATLAB FIG, CSV and LaTeX files into `compensation_table/`. No additional toolboxes are required. The ready-to-use PNG, PDF, CSV and LaTeX are already included. Use the booktabs and graphicx packages for the LaTeX table.

## Correct comparison

All three rows use the SAME PCM-HWA trained checkpoints:

1. HWA + GDC-safe: baseline.
2. HWA + DriftSlice B=1: proposed compensation with B fixed to one.
3. HWA + DriftSlice selected B: proposed compensation using validation-selected B.

Every improvement is proposed test accuracy minus the HWA + GDC-safe test accuracy at the same retention time. The NSR regularizer is not a changed variable in this table, so its lambda column is omitted. This table measures the contribution of DriftSlice compensation while training is held fixed. It does not measure the additional contribution of NSR training.

## Data and definitions

These are the existing full-run results used in your plots: three training repetitions, each already averaged across 30 device realizations. The script computes the mean of those repetition means. It checks the 12 timestamp anchors, matching folds, finite percent accuracies and duplicate records. It does not retrain or alter any AIHWKit results.

Five-year test accuracies are 84.137222% (GDC-safe), 85.107222% (DriftSlice B=1) and 85.074556% (validation-selected B). Improvements over GDC-safe are +0.970000 pp and +0.937333 pp, respectively. Max-drop = initial 20s accuracy minus the minimum mean accuracy across all 12 retention times.

Final Test before PCM is unavailable in the retention CSVs, so it is shown as an em dash. It is not replaced with initial 20s accuracy. The table retains both DriftSlice variants rather than selecting whichever has the best test result.

The MATLAB source passes syntax parsing and all displayed means/differences were independently recomputed from the full-run fold means. MATLAB is unavailable here, so its graphics/export calls have not been executed. The included PNG/PDF preview was generated with Python from the same verified values. `verification.json` records these checks and limits.
