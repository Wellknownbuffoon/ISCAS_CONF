# Table III only: MATLAB and your full-run data

Unzip `table_III_matlab_only.zip`, open the extracted folder in MATLAB R2020a or newer, and run:

```matlab
draw_iscas_table_III
```

It opens only the table and writes PNG, vector PDF, editable FIG, CSV and LaTeX outputs to `matlab_table/`. No toolboxes are required. Keep `source_summary.csv` beside the MATLAB file.

This adapts the reference paper's table layout to your actual ResNet-32 results. It does not reproduce the reference paper's ResNet-9s/ResNet-20 experiments. Both rows use the same DriftSlice B=1 compensation; the training comparison is Standard HWA versus NSR-HWA, configured NSR lambda 0.1. Each result averages three training repetitions, with 30 device realizations per repetition.

Five-year accuracy is 85.107222% for HWA and 85.304333% for NSR-HWA. NSR minus HWA is -0.042667 pp at 1 hour, -0.104444 pp at 1 day, +0.034444 pp at 6 months and +0.197111 pp at 5 years. Negative improvements are retained. Max-drop is initial 20s accuracy minus the minimum across all 12 timestamp means.

The reference table's Final Test before PCM cannot be obtained from these retention CSVs. It is shown as an em dash, not invented or replaced by initial 20s accuracy. The experiment does not contain the reference paper's Time Config search; that column is replaced with the actual fixed compensation.

`table_III_ready_to_use.tex` can be included in your manuscript. Use the booktabs and graphicx packages. `table_III_expected_values.csv` supplies the unrounded source values.

The standalone MATLAB source passed syntax parsing; source calculations were independently checked against the full-run CSVs. MATLAB is unavailable in the execution environment, so its graphics/export calls have not been run here. `table_III_preview.png/.pdf` were rendered with Python from the same verified data to show the intended layout.
