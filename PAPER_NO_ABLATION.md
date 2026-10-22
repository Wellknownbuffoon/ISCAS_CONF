# P0 paper results without an ablation stage

Use `DriftSlice_ISCAS_P0_coordinator_PAPER_NO_ABLATION.ipynb`, or copy its entire last code cell into an already initialized P0 coordinator notebook. Execute that cell only. The new notebook retains the existing scientific definitions and configuration; full training, retention, ablation, diagnostics and the old paper launcher are disabled. Automatic runtime disconnect is disabled. Setup remains available for a new runtime.

This is an export of completed P1–P4 ResNet-32/CIFAR-10 results. It does not train, evaluate networks, select new B values, run retention or compute new mechanism diagnostics. It does not read the ablation directory or require ablation shards. It uses the original run, model identities, PCM model and fingerprints.

All selected checkpoints and retention shards must already exist. For the default 3-repetition / 30-realization / 12-timestamp configuration, that means 450 per-model retention shards and 21,600 records across FP32-Direct, HWA, DAR, DriftAttention and NSR. A preview is insufficient. If a worker is incomplete, the export lists missing files and stops without initiating any missing work or calling the result complete.

Output is a new generation folder and adjacent ZIP:

```text
<existing run>/paper_NO_ABLATION/<retention-fingerprint-prefix>_<UTC-generation>/
<existing run>/paper_NO_ABLATION/<retention-fingerprint-prefix>_<UTC-generation>.zip
```

The ZIP downloads automatically in Colab. Existing training, retention, diagnostic and paper files are not overwritten. A fresh export creates a separate generation.

## Core outputs

- Figure 1: FP32 validation and HWA training/validation curves, plus source CSV/caption. Epoch counts come from the actual configuration, not a hardcoded 154-epoch caption. Selection markers use the saved best epochs.
- Figure 2: overall-pipeline retention comparison, plus source CSV/caption. It is explicitly labelled as differing in training and compensation.
- Figure 3: matched-compensation retention comparison for none, B=1, validation-selected B and GDC-safe, plus source CSV/caption.
- Table I: five-year test error as CSV/LaTeX, and an additional accuracy CSV.
- Raw retention records; two-stage error/accuracy summaries; realization-averaged fold means; FP32 and HWA histories; saved checkpoint selections; transfer records; selected B values.
- NSR-minus-HWA paired fold differences, claim audit with pointwise intervals/p-values, and separate Holm-adjusted verdicts.
- Configuration, dataset hashes, stage fingerprints, source checkpoint/shard identities, exporter source hash and export integrity checks.

The export checks the exact fold/method/realization/compensation/time grid, shard/model identity, saved content hashes, shared pretrained initialization, epoch histories, selection metadata and persistent-state metadata. It independently recomputes Table I from its exported CSV. These are export integrity checks, not a claim to have independently rerun original GPU kernels, training, data splitting or all scientific checks.

Both statistics and plot shading preserve the hierarchy: average hardware realizations within each training repetition, then compute mean and sample SD across repetitions. SD is not a confidence interval. An invalid GDC read remains NaN/n/a and cannot be silently averaged away. `core_training_retention_complete` concerns record availability and identity, while `all_retention_accuracies_valid` separately reports numerical validity.

## Optional mechanism outputs

If all saved HWA and NSR diagnostics for every repetition/realization already exist and match the retention/checkpoint identities, the export includes Table II mechanism metrics. Figure 5 is included only when the corresponding saved distribution NPZ is present. Their numbers are not inferred from accuracy CSVs, and no missing states or metrics are invented.

Otherwise those outputs are listed as omitted in `export_status.json`. The core paper export remains available. Figure 4 and all reference-budget ablation data/claims are always omitted by request. Excluded ablation is not a failed export check, and is not marked as having passed.

If the paper uses mechanism claims or resource-budget claims, it still needs the corresponding measurements; omission of those outputs cannot support those claims. The mechanism metrics exported here use the existing ResNet diagnostic definitions and their documented normalizations, not the different Frobenius-normalized MLP metrics in another notebook.

## Validation

Run `python tests/test_paper_no_ablation.py` in the repository environment. Synthetic-record tests exercise all five methods, missing-shard rejection, checkpoint corruption, foreign shard identities, NaN propagation, optional saved diagnostics, source-file preservation and actual figure/table creation. No dataset, training or GPU inference is executed in these tests.
