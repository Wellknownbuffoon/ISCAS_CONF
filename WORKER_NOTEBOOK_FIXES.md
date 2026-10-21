# HWA and NSR worker notebook repairs

The two uploaded workers had incompatible training settings for the coordinator's
current `driftslice_iscas_colab_resnet32_cifar10_v2` run. They also retained the
zero-filter transfer diagnostic that stopped the coordinator.

| Setting | Uploaded workers | Corrected workers and current coordinator |
| --- | --- | --- |
| Training batch | 256 | 64 |
| Evaluation batch | 512 | 128 |
| HWA epochs | 154 | 100 |
| Training folds | 5 | 3 |
| Require original 154-epoch protocol | True | False |

These settings follow the uploaded P0 coordinator and its reported run; they are
not a new choice of experimental hyperparameters. The corrected workers use the
same Config and scientific code as `DriftSlice_ISCAS_P0_coordinator_FIXED.ipynb`.
The only code-cell differences are their execution controls and model assignments:

- `DriftSlice_ISCAS_P1_worker_HWA_FIXED.ipynb`: worker for `PCM-HWA` and `FP32-Direct`.
- `DriftSlice_ISCAS_P4_worker_NSR_FIXED.ipynb`: worker for `PCM-NSR-HWA`.

The transfer diagnostic follows
[AIHWKit 1.1.0's native weight mapping](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/simulator/tiles/periphery.py#L421).
Active rows reach omega; exact-zero rows stay zero with mapping scale 1. The check
derives the expected mapping from the digital target, including the notebook's
existing inactive-filter flush. It also verifies saved FP32-Direct weights, bias,
BatchNorm, scales and identities instead of trusting old report rows in checkpoints.
No new filter pruning or loss changes are introduced. The scalar-conversion warning
in ResNet preflight is removed.

## Running the workers

Run the corrected P0 coordinator first, with the same Drive root, run ID and
AIHWKit/PyTorch versions in every runtime. Wait until it reports that FP32 and
FP32-Direct training finished for all folds. Then open each corrected worker in
its own GPU Colab runtime and run from the top. On an already configured runtime,
`AIHWKIT_INSTALL_MODE="verify_only"` avoids package replacement. Full stage flags
are still enabled, and each worker keeps its original model assignment.

The corrected workers' stage fingerprints match the current corrected coordinator
when configuration, dataset and environment match. Existing matching checkpoints
resume; completed HWA models are loaded without retraining. Checkpoints from the
uploaded workers' old 154-epoch/five-fold protocol remain incompatible and must not
be relabeled to join the coordinator's 100-epoch/three-fold run. No checkpoint
migration, fingerprint bypass or Drive file rewrite was added.

After every worker finishes training, rerun the coordinator's retention cell to
verify all models and select DriftSlice B on validation. Then rerun the workers
for retention and diagnostics. The existing wait messages enforce that ordering.
The reference-budget ablation can run before shared B selection, because it
evaluates the candidate budgets directly. The coordinator generates paper outputs.
Other worker copies must also use the same coordinator configuration and common
code. Embedded notebook outputs are preserved as historical outputs.

## Verification

Sixteen CPU tests passed across the two workers; two CUDA tests were skipped in
this workspace without a GPU. The tests use real AIHWKit 1.1.0 and PyTorch 2.10.0
with synthetic tensors. They exercise:

- Valid zero rows, a fully zero convolution and stale saved transfer reports;
  corrupted weights, NaNs, mapping scales, biases, BatchNorm and identities fail.
- Agreement with the coordinator for all four stage fingerprints, for both
  architectures; all common code cells are identical. The five explicit Config
  differences explain the original workers' incompatible training fingerprints.
- Actual two-epoch ResNet training and AnalogAdam updates for HWA and NSR;
  interruption/resume matches uninterrupted weights and histories exactly.
  The NSR penalty is positive in the fixture (epoch two: 0.0020984).
- Completed checkpoint reuse without retraining or rewriting, preservation of
  verification RNG state, worker model ownership, a read-only coordinator manifest,
  and rejection of foreign training fingerprints.

The coordinator's six CPU regression tests also pass after extending the test
loader to recognize the workers' cell layout. Its CUDA test remains skipped.
All code cells in both workers compile. Full training, GPU performance, retention
accuracy and your actual Drive checkpoints were not evaluated here. JSON
fingerprints use synthetic dataset hashes and this CPU environment, so they are
compatibility-test values, not your Colab run fingerprints.

Run the worker tests with:

```bash
python tests/test_worker_notebooks.py --out /tmp/new-worker-test-folder \
  --original-hwa /path/to/uploaded/DriftSlice_ISCAS_P1_worker_HWA.ipynb \
  --original-nsr /path/to/uploaded/DriftSlice_ISCAS_P4_worker_NSR.ipynb
```

Original-file arguments are optional; they enable the comparison against the
uploaded protocol. Results are in `verification/worker_notebook_fix_results.json`.
