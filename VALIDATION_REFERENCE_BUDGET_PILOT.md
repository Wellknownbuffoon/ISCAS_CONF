# Short validation-only reference-budget pilot

This pilot investigates an accuracy target; it does not guarantee 90% retained test accuracy. The completed uploaded NSR result with B=1 is approximately 89.88% at 20 seconds and 85.30% at five years. A larger reference bank reduces uncertainty in calibration estimates but cannot eliminate every main-array device error or guarantee no retention loss.

Run the helper from an initialized P1, P4 or coordinator ResNet notebook. It reads existing HWA and NSR best-validation checkpoints from the shared training folder, verifying their fingerprints, content hashes and shared digital initialization. No training stage or official-test evaluation is executed. It uses the existing training repetition's held-out validation split. The native dataset loader may cache official test tensors, but they are not evaluated.

Default workload:

- Fold 0, chosen in advance.
- Three persistent programmed-device realizations.
- HWA and NSR, both already trained.
- Reference budgets 64 (current control) and 256 (candidate).
- Fixed B=1; unchanged four reads per reference cell, gain bounds and all physical noise settings.
- Two retention times: 20 seconds and five years.
- 24 validation passes in total, each over the existing full validation split.

At a given realization and timestamp, the same main-array programmed state and noisy conductance read are used for both reference budgets. MVM seeds and sample ordering also match. Reference banks have different cell counts and therefore differ; this is the intended resource change. At 256 cells, reference-cell area and the number of reference reads per observation are four times the 64-cell control.

The pilot writes only to a new `<original run>/validation_reference_budget_pilot/<UTC-generation>/` directory and downloads an adjacent ZIP. It never overwrites checkpoints, retention shards or original paper results. Interrupted probes leave partial CSVs in their own folder; they are not treated as complete and are not automatically resumed.

The ZIP contains validation records, realization-averaged fold means, a summary, paired candidate-minus-control changes, and `pilot_scope.json`. One training fold supplies no SD across independent training repetitions (recorded as NaN). Three programmed-device trials do not replace the original thirty-realization full protocol. These files are exploratory validation data and must not be merged into test-result tables.

If 256 cells improves five-year validation accuracy without an unacceptable initial-time loss, confirm on the other training repetitions before running a separately identified final test evaluation. If it does not materially help, increasing this budget alone is unlikely to close the approximately five-point gap to the current target. Any benefit shared by HWA and NSR is a calibration benefit; it does not establish an independent NSR training contribution.

The helper accepts optional `folds`, `realizations` and `candidate_budget` arguments. Keep the default short workload initially. For example, after confirmation is warranted, `folds=(0,1,2)` evaluates all three repetitions; this still remains validation data.

Validation: `tests/test_validation_reference_pilot.py` executes real native AIHWKit 1.1.0 CPU inference on synthetic two-epoch ResNet-32 fixtures. The pilot's 64-cell control matches the original full retention implementation at both timestamps (accuracy, loss and reference gain), checkpoint bytes and Config are preserved, RNG state is restored, and training/test loaders are prohibited. Fixture accuracy values are not estimates for the user's network. GPU execution is not tested in this workspace.
