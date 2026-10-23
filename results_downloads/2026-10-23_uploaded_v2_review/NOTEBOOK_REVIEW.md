# Review of the uploaded v2 notebook

Reviewed file: `DriftSlice_ISCAS_RESNET_AND_MLP_Full_Experiment_v2_fixed (1).ipynb`
SHA256: `fbadb9d9d79aa74e30a8265e8dce4809cec6a80e800c8ba76333f56adf00338d`

## Decision

Do not use this file as a drop-in replacement for the completed three-repetition P1/P4 run. It has a reproducible retention-entry error, materially different ResNet training semantics and settings, and a saved GPU checkpoint-restoration check failure. The original upload, existing research notebooks, user checkpoints and prior results were not modified.

## 1. Retention execution fails with NameError

Section 13's full-retention execution cell (zero-based cell 45, line 5) calls `verifszn̈y_retention_grid`, which is undefined. It is a valid Unicode Python identifier, so parsing succeeds and a smoke test of function definitions can still pass. A stubbed execution of this actual cell reproduced the NameError without running any dataset, smoke, training or retention stage.

The intended call is:

```python
verify_retention_grid(FULL_EXP.cfg, "full")
```

The supplied `retention_entry_typo_fixed.py` contains this cell with only that name corrected. It does not repair the other findings or make the notebook compatible with prior checkpoints.

## 2. ResNet NSR/DAR gradients differ from MLP and completed P4

Section 6b (cell 23, lines 11-47 and 306-322) defines `hook_free_tile_weights`, `TileWeightView`, and `add_tile_penalty_grads`. ResNet computes the regularization loss on detached weight leaves and adds its gradients after AIHWKit's backward hook. The MLP trainer in Section 6 does not use that route. The same bypass applies to DAR and the DAR part of DriftAttention in ResNet.

AIHWKit 1.1.0 `TorchInferenceTile.set_scales()` installs a per-row gradient hook `grad / scales**2`. Let S denote those row scales and let g_R already include the regularizer coefficient. The completed worker/MLP route is:

`g_original = S^(-2) (g_CE + g_R)`.

The uploaded ResNet route is:

`g_v2 = S^(-2) g_CE + g_R`.

The scalar loss can be identical while optimizer input gradients differ. Native AIHWKit CPU probes verified exactly that: the same synthetic NSR scalar produced gradient norms 0.7018589014 through the native hook and 0.0006070581 through the bypass; each native row matched bypass-gradient divided by its row scale squared. These are synthetic probe values, not measurements of the user's saved network.

This may be an intentional new training design, but it is not numerically equivalent to the MLP/completed-worker training route. One global lambda cannot generally restore equivalence because row scales differ and change during optimization. The comments claiming unchanged training should disclose this distinction. The NSR scalar formula and DriftSlice correction formula themselves were not found changed.

## 3. This is a different experiment from the completed full run

Effective ResNet configuration:

| Setting | Uploaded notebook | Completed P1/P4 protocol |
|---|---:|---:|
| HWA epochs |154|100|
| Training repetitions |5|3|
| Train batch size |256|64|
| Evaluation batch size |512|128|
| FP32-to-analog transfer guard |1.0 pp|0.25 pp|
| strict_154_epochs |True|False|

The uploaded file is monolithic: it has no P0/P1/P4 worker ownership controls and trains all four primary HWA methods. Its Drive root is `/content/drive/MyDrive/ISCAS_DriftSlice_resnet`; the completed worker configuration uses `/content/drive/MyDrive/ISCAS_DriftSlice`. A matching-looking run ID does not make these folders or experiments equivalent.

With a common synthetic data identity and the same installed environment, native fingerprint calculation confirms different training fingerprints. The PCM/retention source hash matches the completed worker. Changing a few Config values alone does not undo the different ResNet gradient route or Config-source identity.

## 4. Saved GPU output already records a failed scientific check

Section 13's saved training output (cell 44) ends with:

`ScientificCheckError: ... Best-validation checkpoint restoration: PCM-HWA: selected epoch 150 (recomputed 150) of 154; re-eval |delta accuracy|=2.27e-01 pp`.

The check requires a difference at most 0.05 pp. The selected epoch agrees with history, but the stored re-evaluation differs by about 0.227 pp. This is an observed failing check, not proof that weights were corrupted. Do not simply loosen the threshold to pass it.

The actual GPU checkpoint and runtime were not supplied. Tiny native CPU HWA and NSR training/restore/resume probes passed with zero restoration accuracy difference. Therefore this review has not reproduced or identified the cause of the saved GPU failure. A valid next diagnostic would compare saved/loaded state hashes, tile scales, input ranges, BatchNorm state and repeated seeded validation outputs in that original GPU runtime.

## 5. Gradient-norm overflow remains possible

`clip_grad_norm_or_explain` still calls native FP32 L2 norm reduction and only adds diagnostics after failure. A controlled tensor with finite gradient entries 3e20 reproduced a failure even though every entry was finite; its FP64 norm was finite (4.242640772e20). This does not establish that such overflow caused the saved best-checkpoint failure. A stable FP64 reduction would preserve the global clipping rule, but integrating that fix changes training-source identity and must not be silently mixed into completed runs.

## 6. Every expensive full stage is enabled

Section 0 sets training, retention, reference ablation, diagnostics and paper generation to True, despite its comment saying expensive stages default to False. Running all cells can launch the full five-repetition monolithic experiment. For inspection, set the five flags in `disable_full_stages.py` to False before executing any full-stage cell. Keep the existing P1/P4 workflow for the already completed results.

## Verification performed

- All 27 code cells compile as Python; this does not imply runtime correctness.
- The retention entry NameError was reproduced with stage dependencies stubbed.
- Native AIHWKit 1.1.0 CPU probes confirmed same NSR scalar but different hooked/bypassed gradients.
- Native configuration/fingerprint comparison established incompatibility with the completed worker protocol; PCM/retention code identity matches.
- Actual two-epoch synthetic ResNet HWA and NSR training updated physical weights, restored checkpoints with zero CPU accuracy difference, and resumed bit-identically to uninterrupted references: four optimizer steps each.
- Finite-entry FP32 norm overflow was reproduced.
- The uploaded file's saved GPU failure was read, not claimed reproduced.
- No Colab installation, Drive operation, dataset download, full run or CUDA execution was performed.

Machine-readable results are in `probe_results.json`. `reproduce_notebook_review.py` accepts `--repo`, `--notebook` and a NEW `--out` directory and uses native AIHWKit on CPU. The two supplied snippets address only the retention typo and full-stage execution flags; they do not change scientific training semantics.
