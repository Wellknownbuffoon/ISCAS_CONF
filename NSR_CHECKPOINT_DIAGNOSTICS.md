# Check NSR gradients without retraining

Run `nsr_checkpoint_diagnostic_cell.py` as one new cell at the bottom of an already initialized P4 notebook. It uses the existing full-run configuration and checkpoints. Stop any running training or retention cell first. If Colab has restarted, initialize the existing notebook's installation, Drive, configuration and definition cells first; do not run full training or retention to initialize this diagnostic.

The diagnostic reads the best-validation HWA and NSR checkpoints for every fold and uses five independent training batches per fold. Both methods see the same sampled images, augmentation seeds and forward-noise seeds. It does not use test-set accuracy to choose anything. The native dataset loader may load the existing official test tensors into its cache, but those tensors are not evaluated. Data downloads are disabled in a local copy of the loader configuration.

It checks:

- Checkpoint fingerprint, kind, method, completed epoch count, saved model-content hash and shared pretrained-parent identity.
- Exact BatchNorm restoration from the saved state, finite BatchNorm statistics and nonnegative running variance.
- Cross-entropy and the weighted `nsr_penalty()` gradients separately, across all trainable parameters and each physical-weight layer.
- NSR-to-CE gradient norm ratios, gradient cosine similarity and hypothetical combined-loss clipping.

For the HWA checkpoint, the NSR gradient is a counterfactual probe: the original HWA training did not include NSR. All probes run in training mode on new disposable model copies. BatchNorm updates during these probes affect only those copies. Existing notebook models, loader generators, Config and checkpoint files are not modified. Global Python/NumPy/Torch RNG state is restored. Diagnostic output goes to `/content/nsr_checkpoint_diagnostics/`, and Colab downloads `/content/NSR_checkpoint_diagnostics.zip`.

Files in the ZIP:

- `gradient_batches.csv`: global pre-clipping norms, ratios, cosine similarity and clipping indicators for each probe.
- `gradient_layers.csv`: the same gradient comparisons per physical-weight layer.
- `gradient_summary.csv`: mean probe measurements per method/fold. The clipping columns are fractions of sampled probes.
- `batchnorm_state.csv`: saved-state restoration checks and BatchNorm ranges before the probes.
- `checkpoint_metadata.csv`: checkpoint identities, selected epochs and previously recorded validation restoration differences. This diagnostic does not independently re-evaluate validation accuracy.
- `scope.json`: configuration identity and limitations.

Interpretation:

- A nonzero NSR gradient means that the saved checkpoint has a differentiable NSR objective. It does not prove how much that objective changed historical training.
- A very small NSR-to-CE ratio suggests a small raw contribution in these probes; examine per-layer ratios too. No universal ratio threshold establishes whether a regularizer is useful.
- A negative cosine means opposing local directions. Regularizers often oppose classification gradients; this alone is not evidence of a bug or worse retention.
- A large ratio or frequent hypothetical clipping is a reason to inspect the optimization balance. Adam rescales gradients using its moments, so raw norms are not actual parameter-update magnitudes.
- Exact, finite BatchNorm restoration does not establish that BatchNorm is calibrated under physical drift.

This is not a replay of historical training, a proof of an NSR advantage, a reproduction of GDC calibration, or a retention evaluation. It does not diagnose the cause of the fold-2 GDC degradation by itself. Preserve the current completed experiment. If a new pilot is justified, select its configuration using validation data and record it as a new experiment.

Validation in `NSR_DIAGNOSTIC_VERIFICATION.json` used actual AIHWKit 1.1.0 on CPU with synthetic two-epoch ResNet-32 fixtures. It verified that the predicted combined norm matches a real total-loss backward and native gradient clipping, that file contents and RNG state remain unchanged, and that a corrupted checkpoint is rejected. These fixture measurements are not measurements of the user's completed models. GPU execution has not been tested in this environment.
