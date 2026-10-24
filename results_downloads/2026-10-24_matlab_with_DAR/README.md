# Updated MATLAB figures and comparison table with DAR

1. Download and unzip `matlab_figures_and_table_with_DAR.zip`.
2. In MATLAB R2020a or newer, open the extracted folder. Keep the `.m` file and CSVs together.
3. Run:

```matlab
plot_iscas_retention_with_DAR
```

No additional toolboxes are required. Optional folders:

```matlab
plot_iscas_retention_with_DAR('C:\my_data', 'C:\my_figures')
```

## Included outputs

- `figure1_test_error_with_DAR`: panel (a) compares HWA, DAR-HWA and NSR-HWA under the same DriftSlice B=1 compensation; panel (b) compares GDC-safe and DriftSlice with HWA training held fixed.
- `figure2_split_triangle_heatmap_with_DAR`: left panel compares DAR/NSR training against HWA under matched compensation; right panel compares DriftSlice against GDC-safe under each of the three training methods. Upper-right triangles show 20 seconds and lower-left triangles show five years.
- `table_III_all_training_vs_hwa_gdc`: all nine combinations of HWA/DAR-HWA/NSR-HWA training and GDC-safe/DriftSlice B=1/validation-selected DriftSlice. Every improvement column uses **HWA + GDC-safe** as the baseline. Other training rows are complete-pipeline comparisons; matched-training heatmap comparisons isolate compensation.

MATLAB writes PNG, vector PDF, editable FIG and numerical source CSV outputs into `matlab_figures_with_DAR/`, plus CSV and LaTeX versions of Table III. Use the booktabs and graphicx packages in the manuscript preamble for the LaTeX table. Ready-to-use PNG/PDF previews and the table CSV/LaTeX are already included in this package.

## Inputs and identity

The new Drive ZIP contains `PCM-DAR-HWA` only. Its 4,320 test records form the complete 3-fold x 30-realization x 12-timestamp x 4-compensation grid. Its supplied fold means and summary independently matched recomputation from raw records. The original DAR exports are retained under `DAR_original_exports/`.

`source_summary.zip` supplied the existing HWA, NSR-HWA and FP32-Direct data. These were combined with DAR under distinct method names, without replacing previous rows. `source_fold_means.csv` contains all four methods; the MATLAB plots display the three HWA training methods. FP32-Direct remains available in the input CSVs. Neither archive contains DriftAttention results, so no DriftAttention curve is invented.

These CSV archives do not include model/checkpoint fingerprints. The code uses the exported method labels and matching numeric grids; it does not independently authenticate training identities or repair any mislabelled checkpoint/fold files. See `input_verification.json` for archive hashes and validation scope.

## Statistical definitions

The fold-mean input already averages 30 device realizations within each training repetition. MATLAB recomputes the mean and sample SD (N-1 denominator) across three repetition means. Bands are mean +/- 2 SD, not 95% confidence intervals. Timestamps are equally spaced as in the supplied figure. No smoothing, resampling, selected test seeds or test-set hyperparameter selection is performed.

Heatmap and table gains are percentage points, not relative percent changes. The heatmap colors do not indicate statistical significance. Max-drop is initial 20-second mean accuracy minus the minimum mean accuracy over the 12 timestamps. The unavailable pre-PCM `Final Test` values stay NaN/em dashes. No regularization coefficient is inferred from the reference paper.

At five years under DriftSlice B=1, the means are 85.107222% (HWA), 85.319222% (DAR-HWA) and 85.304333% (NSR-HWA). The latter two are close measured means; the table does not establish a statistically significant difference between them.

## Verification and limits

MATLAB syntax was checked with tree-sitter-matlab. All 60 previous plotted curve points, eight previous heatmap comparisons and three previous HWA compensation table rows remain unchanged within 3e-13. The updated outputs contain 72 curve points, 14 heatmap comparisons (28 triangle values) and nine table rows.

MATLAB is unavailable in the execution environment, so the MATLAB graphics/export calls have not been run here. The included PNG/PDF previews were generated with `render_previews.py` from the same verified numerical data. That optional Python script uses numpy, pandas and matplotlib. The expected MATLAB aggregate and heatmap CSVs are included for checking your MATLAB output.
