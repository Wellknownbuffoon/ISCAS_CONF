# Why P2/P3 created worker-DAR and worker-NSR

The exact uploaded notebook settings are:

- P0 coordinator: PARALLEL_ROLE="coordinator", WORKER_MODELS=().
- P2 DAR: PARALLEL_ROLE="worker", WORKER_MODELS=("PCM-DAR-HWA",).
- Uploaded file named P3 DriftAttention: PARALLEL_ROLE="worker", WORKER_MODELS=("PCM-NSR-HWA",).

The P3 filename does not match its active worker setting. role_tag() derives the folder from WORKER_MODELS, not from the notebook filename. This directly explains worker-NSR.

The uploaded P3 also contains saved output identifying worker-DriftAttention and completed 100-epoch DriftAttention training for folds 0/1/2. Saved output may be historical. It does not prove what the current Drive folders contain. The currently loaded FULL_EXP could also be stale after a worker-setting edit: the old export cell chooses its destination using the current role_tag() but can read its models from an earlier FULL_EXP.

## Correct the existing P3 runtime

1. Stop any running stage before changing its worker assignment. In Section 0's EXECUTION CONTROL code cell (zero-based cell 3), use:

```python
PARALLEL_ROLE = "worker"
WORKER_MODELS = ("PCM-DriftAttention-HWA",)
```

2. Run that execution-control cell again. Keep your original configuration, run ID, Drive root and fingerprints.
3. With the existing notebook definitions already loaded, run this new cell to rebuild the experiment for the corrected assignment:

```python
FULL_EXP = open_full_experiment()
print("Configured models:", WORKER_MODELS)
print("Folder tag:", role_tag())
print("Loaded experiment models:", FULL_EXP.models)
assert set(FULL_EXP.models) == set(WORKER_MODELS)
```

It must print worker-DriftAttention and only PCM-DriftAttention-HWA.

4. Run the relevant retention/export cells. Existing compatible DriftAttention checkpoints/results can be reused. Do not rerun training solely to fix this label. The usual HWA validation-selection prerequisite and checkpoint/fold compatibility checks still apply.

## Inspect existing worker-NSR exports before classifying them

Do not rename the folder or its checkpoints to DriftAttention. In the notebook that created the export, run:

```python
print("Last export folder:", _ret_out)
print("Actual methods in its CSV:", pd.read_csv(
    _ret_out / "test_accuracy_vs_retention_raw.csv",
    usecols=["method"],
)["method"].unique().tolist())
```

PCM-NSR-HWA means the exported records are labelled NSR; PCM-DriftAttention-HWA means they are labelled DriftAttention and the role folder was misleading. The existing export additionally checks shard identities against their model checkpoints; do not bypass that verification. Folder names alone cannot prove which training algorithm produced a model.

## Corrected notebook supplied

`DriftSlice_ISCAS_P3_worker_DriftAttention_ROLE_CORRECTED.ipynb` corrects only the worker assignment and adds an export guard that blocks mismatches between WORKER_MODELS and the loaded experiment's models before writing export files. Scientific functions, model/PCM/training parameters and execution flags are unchanged. Historical outputs are cleared to avoid displaying a previous role as a current execution result. No P0 or P2 changes are necessary for this role issue.

Verification used the actual uploaded notebook definitions and native AIHWKit 1.1.0 on CPU. The four scientific stage fingerprints remained equal before/after the role correction under a shared synthetic data identity; role_tag changed from worker-NSR to worker-DriftAttention. The export guard accepted a matching model assignment and rejected a stale NSR experiment before any writes. No Drive files, user checkpoints, full run or GPU training were accessed.
