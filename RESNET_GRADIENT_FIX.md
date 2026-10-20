# ResNet/MLP notebook: gradient-norm overflow fix

Use **`DriftSlice_ISCAS_RESNET_AND_MLP_Full_Experiment_FIXED.ipynb`**. The uploaded original and its embedded outputs were preserved. Only the corrected copy, regression tests, these instructions and execution evidence were added to this repository.

## What was reproduced

The original exception does not distinguish a NaN/Inf gradient from an overflowing **float32 norm reduction on finite gradients**. I reproduced the latter with actual AIHWKit 1.1.0:

| Measurement | Reproduction |
| --- | --- |
| Logical weights in a small AnalogLinear fixture | `1e-18` |
| DAR loss at the epoch-2 time, 60 seconds | `2.9657685445272364e-6`, finite |
| Every gradient entry | Finite |
| Original float32 gradient norm | `inf`; original clipping raises the reported exception |
| Float64 norm | `1.0882664692593338e31`, finite |
| Norm after corrected clipping | `0.8000000083847112` (float32 rounding; target 0.8) |
| Subsequent real AnalogAdam step | Updates weights with finite gradients |

The relevant official source is [AIHWKit v1.1.0 `TorchInferenceTile.set_scales`](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/simulator/tiles/inference_torch.py): its registered weight-gradient hook computes `grad / scales**2`. Small mapping scales can amplify a finite regularizer gradient enough that the intermediate squares in a float32 L2 reduction overflow. The [v1.1.0 optimizer](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/optim/analog_optimizer.py) uses torch Adam for TorchInferenceTile weight parameters. The inspected tag resolves to `809114e582f56e55fa697472fbdaf32708049870`.

Your exact failing Google Drive checkpoint is not accessible from this cloud machine. This reproduces a matching failure mechanism; it is not a claim that your checkpoint's gradients were measured. Actual NaN/Inf gradients remain a distinct failure and are diagnosed explicitly by the corrected code.

## Correction

- `clip_gradients_checked` computes the same global L2 norm and clipping coefficient as the original rule, using float64 temporaries. It also multiplies in float64 before restoring the original gradient dtype, preventing a tiny clipping coefficient from underflowing prematurely. Model weights, forward arithmetic and stored gradients remain FP32.
- The threshold, including the default `grad_clip=0.8`, is unchanged. The helper is used by both FP32 and HWA trainers, for MLP and ResNet, and by the separate attention-gate clipping step.
- Genuine NaN/Inf gradients raise **before any optimizer step**. The message includes method/epoch/batch for HWA, affected parameter names and the smallest mapping scales. No `nan_to_num`, nonfinite-step skipping, loss suppression, learning-rate change or disabled check was introduced.
- AIHWKit state loading replaces `AnalogContext` objects. After a resume, optimizer groups are rebound to the current contexts and checked against all actual model parameters. Physical weight parameters and Adam moments are retained. This is a separately verified resume-integrity correction; it is not asserted to be the source of your nonfinite norm.
- The preflight `float(tensor_requires_grad)` warning now uses `detach().item()`.
- DAR/NSR equations, `TorchInferenceRPUConfig`, ResNet architecture, BatchNorm handling, PCM functions, stochastic seeds, 30 FP32 epochs, 154 additional HWA epochs, five folds, and 30 realizations were preserved. The uploaded ResNet experiment's **12 retention anchors** were preserved; this fix does not change it to the MLP's 75-point grid.
- No AIHWKit package files were changed or packages installed/upgraded. Installation defaults to `verify_only` in the corrected copy, matching your already-working GPU environment.

## Continue your existing training

1. Stop the failed cell and use **Runtime → Restart session** to discard failed-batch gradients. Keep the existing Colab machine/packages and Drive files; do not delete the runtime or checkpoints.
2. Open the **FIXED** notebook, run its cells through the smoke test, and confirm `Finite gradients and stable global clipping` passes on your GPU. The previous smoke PASS is tied to the old code and must not be relabelled as a new PASS.
3. The full-stage flags default to `False`. After the new smoke PASS, enable **only `RUN_FULL_TRAINING=True`** and run the training stage. Keep retention/ablation/paper flags off while validating continuation.
4. The configured source is `driftslice_iscas_colab_resnet32_cifar10_v1`; the destination is the **new** `driftslice_iscas_colab_resnet32_cifar10_gradnorm64_v1`. Both are under the existing `BASE_DIR/results/full` root, so the CIFAR-10 dataset is shared.
5. The importer checks the exact uploaded revision's training-source hash, configuration/data/runtime fingerprint, saved tensors, model identities, pretrained checkpoint identity, and epoch/scheduler cursors. It copies only validated training checkpoints/splits, retains original fingerprints and file SHA-256 lineage in `gradient_fix_import.json`, and never rewrites the old run.

Completed FP32/HWA models are reused. Incomplete DAR resumes its saved next epoch with the saved optimizer, scheduler, RNG and data-generator states; for your reported epoch-1 checkpoint, the continuation is **epoch 2/154**. Later folds still run when they have no checkpoints. No completed stage is silently restarted. An unknown original code revision, changed data/runtime, malformed checkpoint, or different pretrained model causes an explicit refusal instead of a guessed import.

Inherited completed models were trained by the original numerical implementation. That provenance is recorded; they are not claimed to have been retrained with the new clipping reduction. Continuing epochs use the correction. New retention outputs are not mixed with old shards automatically.

If the old checkpoint contains genuinely nonfinite gradients/state rather than the reproduced norm overflow, the new diagnostic stops. Keep its parameter-name/mapping-scale report and the last complete checkpoint; do not set `error_if_nonfinite=False`.

## Verification actually executed

Tests ran with existing **AIHWKit 1.1.0 / PyTorch 2.10.0+cpu** using small synthetic fixtures. No full training or CIFAR-10 benchmark was launched here.

| Regression | Result |
| --- | --- |
| Real AIHWKit scaled DAR gradients reproduce the original error; corrected clipping and AnalogAdam update | PASS |
| Genuine NaN/Inf gradients fail without silently changing other gradients | PASS |
| Ordinary clipping agrees with independent float64 math; extreme coefficient does not zero all gradients | PASS |
| Rebinding after native state load preserves physical weights and Adam moments | PASS |
| MLP and actual 34-layer ResNet DAR interrupted at epoch 1, resumed at epoch 2, identical to uninterrupted corrected runs | PASS |
| Legacy import validates fingerprints, preserves source files, is idempotent and rejects foreign data | PASS |
| Checkpoints generated by the **original uploaded notebook** import correctly; completed FP32/HWA states are reused and DAR continues at epoch 2 | PASS |
| CUDA-specific regression | NOT EXECUTED here; no CUDA device. The notebook executes it during Colab preflight. |

The tests verify the original scientific/model/PCM source cells remain unchanged. All notebook code cells compile, and historical embedded outputs are unchanged. Synthetic fixture accuracy values are test data, not research results.

See [verification/resnet_gradient_fix_results.json](verification/resnet_gradient_fix_results.json) for measurements. Tests are in [tests/test_notebook_gradients.py](tests/test_notebook_gradients.py):

```bash
# Use an already compatible environment; this command installs nothing.
python tests/test_notebook_gradients.py --out /tmp/a-new-regression-output \
  --original-notebook /path/to/DriftSlice_ISCAS_RESNET_AND_MLP_Full_Experiment.ipynb.txt
```

Omitting `--original-notebook` skips the original-checkpoint import regression explicitly. GPU testing is also explicitly skipped when CUDA is unavailable. The test loader executes definition cells only; it does not run setup, mount Drive, download data or launch full-stage cells.
