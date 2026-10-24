# Corrected DAR benchmarks: no DAR + DriftSlice

Unzip `pure_DAR_benchmark_matlab.zip`, open the extracted folder in MATLAB R2020a or newer, and run:

```matlab
plot_iscas_pure_DAR_benchmark
```

No additional toolboxes are needed. MATLAB saves PNG, vector PDF, editable FIG, CSV and LaTeX outputs in `matlab_pure_DAR_figures/`. Ready-to-use PNG/PDF previews and the table CSV/LaTeX are included.

## DAR definitions, following the two evaluation settings in the paper

The uploaded AICAS 2025 paper, DOI https://doi.org/10.1109/AICAS64808.2025.11173140, Figure 2 (PDF page 3), reports DAR without GDC and with GDC separately. Section IV-B also states that GDC was evaluated across all methods. Neither of those settings uses DriftSlice.

- **Pure DAR** = your `PCM-DAR-HWA` checkpoint evaluated with `compensation=none`. Training remains hardware-aware plus DAR; there is no GDC or DriftSlice during retention evaluation.
- **DAR + GDC-safe** = the same training method evaluated with your GDC-safe implementation, explicitly shown as a separate setting. It is not labelled pure/uncompensated DAR.
- **DAR + DriftSlice** is excluded from every plot, heatmap, table and input/output CSV in this revised package. Those previous hybrid curves must not be presented as the paper's DAR benchmark.

This correction changes which existing result records are plotted. It does not retrain, change the DAR formula, alter weights, or manufacture accuracy results.

## Corrected figures

1. `figure1_pure_DAR_benchmark`: left panel compares HWA, DAR-HWA and NSR-HWA with no compensation for any model; right panel includes pure uncompensated DAR alongside the HWA + GDC-safe baseline and your compensated DriftSlice pipelines. Its full 0-100% error axis is necessary because your pure DAR curve approaches 90% error. This is a whole-pipeline comparison, not an isolated training effect.
2. `figure1b_pure_DAR_comparison`: the right panel saved separately, with DAR clearly labelled and drawn as purple open squares.
3. `figure3_DAR_GDC_safe_separate_benchmark`: a separate, expanded view of DAR + GDC-safe (no DriftSlice), HWA + GDC-safe and your DriftSlice pipelines. This corresponds to the paper's additional with-GDC evaluation setting at the protocol level. Your GDC-safe implementation is not claimed to be numerically identical to the paper's GDC implementation.
4. `figure2_no_DAR_DriftSlice_heatmap`: DAR comparisons exist only in the None and GDC-safe columns. DAR/DriftSlice cells are grey and explicitly excluded; the compensation panel includes only HWA and NSR-HWA.
5. `table_III_no_DAR_DriftSlice`: eight rows; DAR has only None (pure DAR) and GDC-safe rows. Every improvement uses HWA + GDC-safe as the reference. Pre-PCM Final Test values are unavailable and remain em dashes/NaN.

## Your measured results are not the paper's original measurements

Your CSVs give five-year accuracy of **10.064111% for pure DAR without compensation** and **84.463556% for DAR + GDC-safe**. These are your ResNet-32/CIFAR-10 results and must not be replaced by the paper's much higher MNIST results. The paper benchmarks a two-layer MLP/MNIST, five training repetitions and 154 epochs; the current CSVs report three repetitions and do not contain checkpoint fingerprints/epoch metadata. The plots do not claim an exact reproduction of that experiment.

Input DAR records were selected from the uploaded raw export using compensation none/GDC-safe only: 2,160 records, three folds, 30 device realizations, 12 timestamps and two compensation modes. Their means agree with the original summary. HWA/NSR accuracy values are retained unchanged.

Shading is mean +/- 2 sample SD across the three reported training-repetition means, each already averaged over 30 device realizations. It is not a confidence interval. Heatmap colors do not establish statistical significance. Max-drop is initial 20-second mean accuracy minus the minimum over all 12 timestamps. Means and sample SDs are recomputed; missing/duplicate records are rejected.

## Verification limits

The MATLAB source passed syntax parsing. Selected means/SDs and every table metric were independently checked against the provided fold means, and DAR endpoints also against the raw DAR records. A check confirms that no revised CSV or curve definition pairs DAR with DriftSlice. No actual checkpoint/fold identity was authenticated because model fingerprint files were not uploaded.

MATLAB is unavailable here, so MATLAB graphics/export calls were not executed. The included previews were generated with `render_previews.py` from the same verified data. `paper_protocol_reference.json` records the paper passages and PDF hash; `verification.json` records numerical checks and scope.
