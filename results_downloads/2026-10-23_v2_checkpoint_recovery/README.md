# Fix the uploaded v2 notebook's checkpoint-restoration failure

This is for `DriftSlice_ISCAS_RESNET_AND_MLP_Full_Experiment_v2_fixed (1).ipynb`, ResNet-32, run `driftslice_iscas_colab_resnet32_cifar10_v2`, five repetitions and 154 HWA epochs. Keep that notebook/configuration.

## What the supplied log actually proves

AIHWKit CUDA and the deployment/physics preflight checks pass. All four fold-0 HWA methods have completed 154 epochs. The notebook then stops in its OWN `verify_training_fold()`, because the PCM-HWA checkpoint contains `restore_abs_diff_pp ≈ 0.226667`, above the existing 0.05 pp threshold. Epoch 150 agrees with recomputation from history. DAR, DriftAttention and NSR report zero restoration difference in this log.

The HWA checkpoint was loaded through the "completed ... not retrained" branch. That branch returns the saved checkpoint, including its old restoration result; it does not perform a fresh validation in this session. Rerunning the same stage alone therefore does not investigate the mismatch.

This does not prove checkpoint corruption or an AIHWKit library bug. The actual A100 checkpoint was not uploaded, so the cause remains unconfirmed.

## Exact steps in Colab

1. Keep your original v2 notebook and runtime. Do not delete checkpoints, change Config, change the run ID, raise the 0.05 pp threshold or disable checks. If the runtime restarted, run setup/definition cells through Section 12 and the smoke test first.
2. Insert ONE new code cell immediately ABOVE Section 13's full-training execution cell (the one beginning `FULL_EXP = None` and containing `run_training_stage(FULL_EXP)`).
3. Copy the complete contents of `section_13_checkpoint_recheck_cell.py` into that new cell and run it. It uses the exact training fingerprint prefix in your supplied log. It does not train or evaluate the test set.
4. If it prints `REPAIRED_STALE_METADATA`, the actual saved model reproduced the historical validation score under the original 0.05 pp requirement on three independent loads. Only then is the stored restoration result updated. An exact-byte backup and a JSON audit record are saved. Model weights, history, selected epoch, optimizer counts, training fingerprint and algorithms remain unchanged.
5. After `REPAIRED_STALE_METADATA` (or `PASS`), rerun the original Section 13 full-training cell. Completed methods are reused; it may train remaining folds according to your unchanged configuration.
6. If it prints `STOPPED`, the checkpoint is not declared repaired. Run `download_diagnostic_report_cell.py` in another cell and share the resulting JSON. This is necessary to distinguish a persistent validation mismatch, changed scientific source/environment, wrong split, loading problem, model identity mismatch or saved RPU-configuration mismatch. Do not continue to paper results with a failed restoration check.
7. There is a second, independent typo in Section 13's full-retention cell. Replace that ENTIRE execution cell with `section_13_retention_entry_fixed.py`, or replace only the undefined name `verifszn̈y_retention_grid` with `verify_retention_grid`. This does not fix the checkpoint failure; it prevents a later NameError.

## AIHWKit 1.1.0 checks

The following official tagged source files were fetched and matched the installed AIHWKit 1.1.0 files byte for byte; hashes are supplied in `aihwkit_1_1_0_source_verification.json`.

- [TorchInferenceTile](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/simulator/tiles/inference_torch.py): supports the TorchInferenceRPUConfig path, applies mapping/periphery, installs the `grad / scale**2` weight hook and exposes post-update clipping.
- [Tile state restoration](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/simulator/tiles/base.py): restores full analog state, resets device bookkeeping to CPU, and creates an AnalogContext. The notebook correctly calls `model.to(device)` after loading a CPU-mapped state and builds the resume optimizer AFTER model-state loading.
- [Torch simulator](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/simulator/tiles/torch_tile.py): weight-modifier noise is training-only; evaluation still has configured forward peripheral noise. Therefore `model.eval()` alone does not make analog validation deterministic. Use the original repeated-read count and seeds.
- [Analog optimizer](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/optim/analog_optimizer.py): handles AnalogContext groups and calls post-update operations.

The supplied recovery uses these existing notebook/AIHWKit paths. It checks complete model-state hashes (including BatchNorm, mapping scales and input ranges), saved RPU configuration, original validation split, selected history row, epoch/optimizer counts, repeated seeded validation and a subsequent state round trip. It preserves global RNG state.

## Separate algorithm issue: ResNet versus MLP

The v2 ResNet trainer computes NSR/DAR gradients on detached tile-weight leaves, then adds them AFTER AIHWKit's scaling hook. The MLP trainer does not. With row scales S and regularizer gradient g_R including lambda:

- MLP/native route: `S**(-2) * (g_CE + g_R)`.
- Uploaded v2 ResNet route: `S**(-2) * g_CE + g_R`.

The scalar NSR formula can match while the optimizer updates differ. This is custom training behavior, not an unsupported library call, and is not the exception in the supplied log. The recovery deliberately does not change it or relabel the completed training as MLP-equivalent. Reverting this route is a separate scientific training change requiring its own source identity; it cannot retroactively change trained weights.

## Verification and limits

Native AIHWKit 1.1.0 CPU tests actually trained a small synthetic two-epoch ResNet HWA fixture. They verified three reproducible validations and full-state loading, exact-byte backup plus metadata-only repair of an injected stale result, and refusal to modify a checkpoint with a real accuracy mismatch, corrupted model identity, changed RPU configuration or wrong experiment fingerprint. Scientific fingerprints stayed unchanged. Results are in `native_cpu_test_results.json`.

These are synthetic CPU regression results, not measurements of your A100 model. No full training, Drive access, GPU checkpoint execution or final accuracy claim was made here.

The loader executes the helper in its own module namespace. It does not replace registered scientific functions or bypass the smoke gate. The original uploaded notebook is not rewritten.
