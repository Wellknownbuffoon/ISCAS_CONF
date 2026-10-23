# MATLAB code for both figures and the requested table

1. Download and unzip `matlab_retention_figures_and_table.zip`.
2. In MATLAB, open the extracted folder (keep the .m file and CSV together).
3. Run:

```matlab
plot_iscas_retention
```

MATLAB R2020a or newer is required. No additional toolboxes are used.

Optional input/output directories:

```matlab
plot_iscas_retention('C:\my_data', 'C:\my_figures')
```

The code opens three windows and writes to `matlab_figures/`:

- `figure1_test_error.png/.pdf/.fig`: the two-panel error-versus-retention plot, with the same 12 equally spaced timestamp positions as your screenshot.
- `figure2_split_triangle_heatmap.png/.pdf/.fig`: upper-right triangle = 20-second accuracy difference; lower-left = five-year difference. Green is improvement and red is degradation.
- `table_III_drift_robustness.png/.pdf/.fig`: a table in the requested style, adapted to your ResNet-32 HWA/NSR experiment with DriftSlice B=1 held fixed.
- `table_III_drift_robustness.csv/.tex`: numerical table and manuscript-ready LaTeX. Use booktabs, multirow and graphicx packages in your LaTeX preamble.
- `matlab_aggregated_source.csv` and `matlab_heatmap_source.csv`: the numerical sources recomputed by MATLAB.

The package also contains `table_III_expected_values.csv` and `table_III_ready_to_use.tex`, computed directly from the uploaded CSVs, so you can inspect/use the table before running MATLAB.

## Data and statistical definitions

The bundled data are the completed three-training-repetition, 30-device-realization results behind your supplied figures. They are not predictions of a new v2 training run. The code uses actual CSV values, not values digitized from the pictures.

`source_fold_means.csv` already averages the 30 device realizations within each repetition. The MATLAB code takes the mean and sample SD (N-1 denominator) across the three repetition means. Error = 100 - accuracy. Shading = mean +/- 2 SD, not a 95% confidence interval. The heatmap displays differences of accuracy means in percentage points; colors do not establish statistical significance. Missing or duplicated records stop the script rather than being averaged away.

For Table III, the comparison is Standard HWA versus NSR-HWA, both with DriftSlice B=1. NSR lambda is the configured 0.1 from the uploaded checkpoint-diagnostic scope. `Max-drop` = initial 20-second accuracy minus the minimum mean accuracy over all 12 timestamps. Improvements at 1h, 1d, 6 months and 5 years = NSR-HWA minus HWA. Both positive and negative differences are retained.

The reference paper's `Time Config`/drift-lambda search is replaced with the applicable compensation and NSR coefficient. No hyperparameter sweep or best-configuration claim is invented. `Final Test` before PCM is unavailable in the supplied retention CSVs, so it is shown as an em dash and stored as NaN. It is not replaced with the 20-second result.

## Checks and limits

The MATLAB source passes tree-sitter-matlab syntax parsing. Independent recomputation from fold means matched all 60 displayed curve points and all 16 heatmap triangle values from the supplied plots (within 3e-13). The table's five-year HWA and NSR accuracies are 85.107222% and 85.304333%, an improvement of 0.197111 percentage points.

MATLAB/Octave is not installed in the execution environment, so the MATLAB graphics/export calls have not been executed here. `verification.json` records that limit. The .fig outputs are editable in MATLAB; the PDFs request vector output and PNGs use 300 dpi.
