# Coordinator transfer-check repair

The uploaded P0 coordinator stopped at `Weight and bias transfer correctness` even
though logical weight errors were below 6e-8. Its diagnostic required every physical
tile row to have maximum absolute weight `omega=0.8`. The notebook already flushes
inactive FP32 filters to exact zero, so this assertion rejected valid zero rows.

AIHWKit 1.1.0 explicitly uses `alpha = row_max / omega`, replaces zero alpha with 1,
and stores `weights / alpha`. A zero row therefore stays zero with mapping scale 1;
nonzero rows reach omega. See IBM's
[official implementation](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/simulator/tiles/periphery.py#L421).
The installed implementation used for testing matches the cached official source byte for byte.

`DriftSlice_ISCAS_P0_coordinator_FIXED.ipynb` checks both cases against the intended
digital weights (after the notebook's existing inactive-filter flush), and checks
mapping scales as well. MLP diagnostics follow the same native convention. Zero-row
expectations come from the digital target, so a damaged active row cannot pass by
becoming zero. Logical weights, digital biases, BatchNorm and checkpoint identities
remain checked. The unrelated tensor-to-scalar preflight warning is also removed.

Existing FP32-Direct checkpoints contain the old report. Fold verification now loads
their actual tensors and recomputes the report in memory against the FP32 model.
It does not rewrite checkpoints or accept their historical report as evidence.
Both ordinary training and coordinator post-training use this path.

## Resume in Colab

1. Open the corrected notebook on the `verified/pcm-pipeline-audit` branch. Stop the
   old coordinator if it is still running, and restart its session without deleting
   installed packages or Drive files. Execute the corrected notebook from the top.
2. Keep the uploaded notebook's Drive root, ResNet run ID
   `driftslice_iscas_colab_resnet32_cifar10_v2`, configuration and runtime versions.
   All original execution settings are preserved, including enabled full stages.
   On an already configured runtime, `AIHWKIT_INSTALL_MODE="verify_only"` avoids
   package replacement.
3. Allow the fresh smoke test to finish. The notebook-wide smoke hash changes;
   training/retention/ablation/diagnostics fingerprints do not change for the same
   configuration, data and environment. Completed compatible folds are loaded;
   incomplete folds continue through the existing resume mechanism.
4. Expect `tile mapping correct: True (zero rows=..., active rows=...)` instead of
   the invalid universal `tile row-max==omega` assertion. The historical outputs
   embedded in the uploaded notebook are retained and labeled as historical.

For workers, use the same corrected notebook with your existing `PARALLEL_ROLE`
and `WORKER_MODELS` settings. Wait for the coordinator's ready message as before.

## Validation

Six regression tests passed with real AIHWKit 1.1.0 / PyTorch 2.10.0 on CPU; one CUDA
test was skipped because this workspace has no GPU. No full training, datasets or
Drive checkpoints were accessed during validation.

The 34-layer synthetic ResNet fixture includes collapsed filters, an entirely zero
convolution and a zero classifier row (37 zero rows total). The old check reports
error 0.8000000119; the corrected maximum mapping error is 5.96e-8. A saved checkpoint
with the old failing report passes fresh verification with identical file bytes.
Deliberate corruption of zero rows, active rows, mapping scales, bias, BatchNorm,
NaN weights and model identity is rejected. A zero-row MLP passes as well.
All four stage fingerprints match the uploaded notebook for both architectures;
the full notebook smoke hash changes as required. All notebook code cells compile.
Reported fingerprint values in the JSON use synthetic dataset hashes and this CPU
environment; they are compatibility-test values, not your Colab run fingerprints.

Run the checks with:

```bash
python tests/test_coordinator_transfer.py \
  --original-notebook /path/to/uploaded/coordinator.ipynb \
  --out /tmp/new-coordinator-test-folder
```

The original notebook argument is optional; without it the compatibility comparison
is skipped. Results are recorded in `verification/coordinator_transfer_fix_results.json`.
