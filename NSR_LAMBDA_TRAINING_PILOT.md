# Short NSR lambda continuation pilot

Use your initialized P4 ResNet runtime and its existing shared Drive paths. Stop any running training or retention cell before starting this pilot. Add a new code cell at the bottom and paste the contents of [nsr_lambda_pilot_cell.py](nsr_lambda_pilot_cell.py). You do not need to edit `Config`, retrain FP32, rerun full retention or run ablation.

If the runtime was disconnected, initialize the notebook first: in Section 0 set `RUN_FULL_TRAINING`, `RUN_FULL_RETENTION`, `RUN_REFERENCE_ABLATION`, `RUN_MECHANISM_DIAGNOSTICS` and `GENERATE_PAPER_RESULTS` to `False`. Keep your original architecture, run ID, paths and scientific settings. Run setup/Drive and definition cells through Section 11; run the smoke test in Section 12 if required for the new environment. Then run the new pilot cell. The pilot requires the completed original PCM-HWA best-validation checkpoint on Drive, including when launched from P4.

The default pilot is:

| Setting | Value |
|---|---|
| Training repetition | Fold 0 only |
| Starting model for every candidate | Same saved PCM-HWA best-validation model, including BatchNorm and input ranges |
| NSR lambda | 0, 0.1, 1.0 |
| Additional training | 10 epochs per candidate; 30 epochs total |
| Optimizer | Fresh AIHWKit AnalogAdam with native parameter regrouping |
| Learning rate | Per-batch cosine decay from 0.0001 to 0.000001 |
| Other training settings | Original augmentation, batch size, gradient clipping, manual physical-weight decay and tile input-range floor |
| Retention correction | DriftSlice B=1, 64 physical reference cells, four reads per reference cell |
| Validation evaluations | Epochs 0, 5 and 10; 20 seconds and five years; three fixed PCM realizations |
| Checkpoint selection | Highest mean five-year validation accuracy, subject to at most 0.25 percentage points loss in 20-second validation accuracy relative to the common starting model |

Ties prefer higher 20-second accuracy, then the earlier epoch. Epoch zero is an eligible fallback: if fine-tuning does not improve the guarded validation objective, its unchanged HWA starting model remains selected. All ten epochs still execute for each candidate. A selected epoch of zero therefore means that candidate did not supply an improvement under this screen.

Candidates share training order, augmentation seeds, forward-noise seeds, physical-device seed rules, retention timestamps and validation data. Main-array device states are programmed once per validation realization and reused at both timestamps. The NSR equation is unchanged; only its coefficient differs. Lambda zero is the matched extra-training control. The learning-rate schedule is deliberately a new conservative continuation schedule, not a restart of the original 100-epoch OneCycle schedule.

The 20-second guard is applied to compensated validation accuracy, not to clean digital accuracy. There are 42 complete validation passes: six for the shared epoch-zero model, followed by twelve for each candidate at epochs five and ten. The first measured epoch and validation durations indicate the actual runtime; no specific completion time is guaranteed.

This is a **new HWA-warm-start continuation experiment**, not a rerun of the original NSR-from-FP32 training. Compare each selected positive-lambda candidate against the selected lambda-zero continuation, not just against the original HWA checkpoint. The output `paired_validation_comparison.csv` supplies that comparison. A benefit shared by all candidates is attributable to extra training under this protocol, not independently to NSR.

Everything is stored under `<original run>/nsr_lambda_pilot/<new pilot fingerprint>/`. The fingerprint binds the helper source, original fingerprints/configuration, source model identities, screening protocol and environment. Original checkpoints, retention shards and paper CSVs are never overwritten. Each candidate gets two alternating epoch checkpoints with model, optimizer, scheduler, RNG, loader state, selection records and best model. Rerun the identical cell after interruption to resume at the last completed epoch. Completed candidates are skipped. Changed protocol, helper code or GPU environment creates a separate pilot folder.

At completion, Colab downloads a small ZIP containing the contract, helper source, training histories, validation records and comparisons. Selected model files remain on Drive as `fold0/lambda_<value>/selected_validation.pt`. Do not copy these into the original training directory: they have different training and selection protocols and are deliberately marked as pilot checkpoints.

Upload the small ZIP for evaluation before extending the pilot. A promising candidate should be checked on the other training repetitions with the same predeclared settings before a separately identified final test run. The helper supports `folds=(0,1,2)`, but the default single-fold screen is the intended first step with the current deadline. It does not use the official test set for tuning. The native dataset loader may cache official test tensors; it never iterates them in this pilot. These exploratory validation CSVs must not be inserted into final test-result tables. No coefficient can guarantee 90% accuracy.

Validation: three tests in `tests/test_nsr_lambda_pilot.py` pass with native AIHWKit 1.1.0 and Torch 2.10.0 on CPU using synthetic ResNet-32 fixtures. They verify real AnalogAdam updates, zero versus positive NSR loss, matched data order, exact interrupted/resumed model and optimizer histories, completed-run skipping, foreign-checkpoint rejection, checkpoint-selection guards, unchanged original checkpoint/configuration and restored RNG. A separate native comparison verifies that the pilot's validation inference matches the original full retention implementation at both timestamps, including loss, accuracy and gain. Training/resume tests use a deterministic validation substitute to isolate optimizer reproducibility. Synthetic fixture accuracies do not estimate the user's trained model. CUDA training has not been executed in this workspace.
