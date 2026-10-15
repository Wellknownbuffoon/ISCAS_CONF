# AIHWKit PCM pipeline audit and verification

The corrected pipeline passed all 16 lightweight tests and an MNIST **SMOKE TEST** covering all six HWA methods plus FP32-Direct. The historical all-ones GDC calibration saturates on the available checkpoint. A measured one-hot baseline now reports and rejects observed clipping; an ideal linear baseline is separately labelled.

This audit does **not** establish full-protocol scientific performance, native/custom statistical equivalence, or a physically implementable DriftSlice circuit.

## Scope and environment

- Starting repository commit: `22ebeb56e25bbf22173b5278833136cc0057a751`.
- Isolated correction branch: `verified/pcm-pipeline-audit`; no existing results were moved or deleted.
- Inspected all four requested files. No filenames containing `(1)` or conflicting duplicate modules were present.
- Actual runtime: Python **3.12.14**, PyTorch **2.10.0+cpu**, AIHWKit **1.1.0**; CUDA unavailable, zero GPUs. Conda and the requested WSL `/mnt/d/RESEARCH/ISCAS/CODE/data/MNIST/raw` path are unavailable in this attached machine.
- Reused `/workspace/.venvs/iscas-conf` and existing verified MNIST files. **No packages were installed or upgraded during this audit.** The preceding environment-onboarding task had prepared this environment.
- The requested PyTorch 2.6.0 does not satisfy the official v1.1.0 declaration `torch >= 2.9.1` in [pyproject.toml](https://github.com/IBM/aihwkit/blob/v1.1.0/pyproject.toml). Your WSL installation was not inspected, so its actual binary compatibility remains unverified. No changes to it are proposed without approval.
- Full 30+154 training and GPU execution: **NOT EXECUTED**. Only shortened training and small native-API fixtures ran.

The official source was fetched from tag **v1.1.0**, whose Git ref resolved to `809114e582f56e55fa697472fbdaf32708049870`. Its eight directly inspected API source files matched the installed wheel's Python source byte for byte. Source cache: `/workspace/iscas-audit/reference/aihwkit-1.1.0`. AIHWKit package files were not edited.

## A. Version-specific API compatibility

All references below point to **v1.1.0**, not `master`.

| API | Verified behavior | Runtime evidence |
| --- | --- | --- |
| [AnalogLinear](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/nn/modules/linear.py#L17) | Internal constructor initialization precedes explicit `set_weights`; loading replaces weights and bias. The deployed MLP remains 784–250–10 with tanh. | Tests 2, 3, 11; MNIST smoke. |
| [TorchInferenceRPUConfig](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/simulator/configs/configs.py#L263) / [TorchInferenceTile](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/simulator/tiles/inference_torch.py) | Torch simulator weights are differentiable parameters. Input range normalization, output scales, and digital bias belong to the periphery. | Tests 4 and 11 exercise the configured nonideal path. |
| [AnalogAdam and regroup_param_groups](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/optim/analog_optimizer.py) | Actual AnalogAdam combines the analog mixin with torch Adam. Regroup isolates AnalogContext parameters; TorchInferenceTile's physical weight parameters are updated by Adam. The optional model argument is accepted but regroup works on the optimizer's parameters. | Test 4: four groups, every model parameter included once, finite nonzero weight gradients and updates. |
| [Weight transfer / scaling](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/simulator/tiles/periphery.py#L111) | `get_weights(apply_weight_scaling=True)` reconstructs logical weights. Snapshot weights exclude per-column scales, which are applied once at inference. | Tests 2, 3, 11; measured errors below. |
| [Digital bias](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/simulator/tiles/inference_torch.py#L215) | `mapping.digital_bias=True`; bias is outside PCM programming, output scaling, and drift compensation. Snapshot adds it after scale and gain. | Tests 2, 7, 8, 11; native bias unchanged exactly. |
| [PCMLikeNoiseModel](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/inference/noise/pcm.py) | Conductance-dependent programming noise and drift coefficients; programming clips below zero only. Read noise is present at elapsed zero because total read time includes `t_0=20`. | Tests 8–10. |
| [SinglePairConductanceConverter](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/inference/converter/conductance.py#L23) | Maps signed weights to positive/negative conductances with one scale ratio; inverse takes their difference. | Tests 8 and 11. |
| [program_analog_weights / drift_analog_weights](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/nn/modules/base.py#L290) | Require evaluation mode. [Native programming](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/simulator/tiles/inference.py#L162) retains programmed signed weights and drift parameters. Repeated native drift starts from that programmed state. | Test 8 calls both actual methods at multiple times. |
| [GlobalDriftCompensation](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/inference/compensation/drift.py#L21) | Executable `get_readout_tensor` returns **identity/one-hot probes**, despite inherited comments mentioning all ones. Scalar readout is clamped mean absolute output. Baseline is the programmed analog readout. | Test 8 independently reproduces native alpha under perfect IO. |
| [GlobalDriftCompensationWithExactReference](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/inference/compensation/drift.py#L44) | Uses the **unprogrammed target** with an exact linear reference; its current readout still uses the tile. Native readout bypasses input-range and digital output/bias periphery. | Test 8 verifies class, baseline ratio, compensation, and unchanged bias. |

Native compensation measurements use perfect IO to isolate reference/ratio behavior. They are not a claim that native and custom nonideal calibration paths are identical.

## B. Training protocol compliance

| Requirement | Corrected implementation and verification | Full execution |
| --- | --- | --- |
| Exactly 30 FP32 epochs | Full-mode guard; complete ordered history; separate final, validation-selected, and resumable checkpoints. Original earliest maximum-validation-accuracy tie rule is preserved. | NOT EXECUTED; 2 executed in smoke. |
| Which FP32 checkpoint transfers | `digital_pretrained.pt` / `digital_best_validation.pt` contain the same validation-selected state. Selected epoch, completed epochs, state SHA-256, dtype, and selection split are explicit. | Tested in shortened runs. |
| Weights **and biases**, no reset | Numerical transfer checks before optimization and again after optimizer construction; physical initial references and FP32 state identity saved per method. | Tests 2–3 and actual MNIST transfer. |
| Exactly 154 additional HWA epochs | Full-mode guard; ordered history/completion checks; resume cursor must match history. No early-stop path exists in `train_hwa_all`. Changing `strict_154_epochs` cannot bypass the full count. | NOT EXECUTED; 2 executed per method. |
| AnalogAdam and manual decay | Actual imported AnalogAdam; regroup; all parameters included exactly once. Manual decoupled decay uses each group's LR and only the two physical weight matrices. | Test 4; actual first updates recorded for all six methods. |
| Final versus selected HWA | Final `checkpoint_epoch=154`; best `checkpoint_epoch=selected_epoch`; both record the actual selected epoch and `analog_epochs_completed=154`. A controlled unit fixture explicitly selects epoch 1 after completing epoch 2. | Full run NOT EXECUTED; behavior verified with 2 epochs. |
| Optimizer, scheduler, histories | Resumable `_training.pt` stores matching last-step model/optimizer/scheduler/RNG state. Final and best checkpoints also retain history and final optimizer state, labelled `optimizer_state_epoch`; do not interpret that optimizer state as belonging to an earlier selected model epoch. | Interrupted FP32 and attention-HWA runs match uninterrupted CPU runs exactly. |
| FP32-Direct | Same selected FP32 identity; transfers both biases; no HWA optimizer construction; `analog_epochs=0`, `optimizer_steps=0`. Goes through PCM retention. | Tests 7–8 and smoke. |
| Five independent training repetitions | Seeds/splits depend on repetition; full configuration requires 5. | NOT EXECUTED; smoke uses 1. |
| Thirty independent PCM realizations | Independent programming/drift seeds per realization; the same device read is shared across compensation strategies at a given time. | Full 30 NOT EXECUTED; smoke uses 2. |
| Exactly 75 timestamps, 20 s to 5 years | Existing exact-grid construction retained; all twelve anchors present; full-count configuration guard added. | Grid constructed and tested; full evaluation NOT EXECUTED. |
| Validation-only checkpoint/bin choice | Training/selection functions receive validation loaders, not the test loader. Bin selection uses validation accuracy at five years. Held-out test data only measures fixed choices. | Source review, tests, disjoint indices in actual smoke. |
| Provenance / stale checkpoints | Both scientific modules, full config, library versions, and actual MNIST file hashes enter the manifest fingerprint. Pretrained state SHA-256 is checked on reuse. Source/config/data changes cannot silently resume the old run. | Test 16 plus fresh smoke manifest. |

The saved optimizer state in a selected checkpoint is explicitly the **final** optimizer state. Resume from `_training.pt`, where model and optimizer states correspond to the same completed step.

## C. Confirmed issues and minimal corrections

| Location | Root cause / finding | Correction |
| --- | --- | --- |
| `driftslice_core.global_gdc_gain` | Summed all-ones responses can saturate. Input-range clipping and ADC saturation obscure the true decay ratio. The historical name `GDC` could be mistaken for IBM native GDC. | Preserve unchanged measurement semantics as `GDC-legacy-ones`; instrument the actual upstream DAC/ADC path with a local subclass, without editing AIHWKit. Add `GDC-safe-onehot` and `GDC-ideal-onehot`. |
| `PCMPhysicalMatrix`, `PCMClassReferenceBank` | Extra `.clamp(0, g_max)` after programming was absent from the official PCM programming model and also changed reference cells. | Default to upstream nonnegative-only bounds; expose `pcm_conductance_bounds='legacy_gmax_clip'` for the historical custom assumption. Read noise also retains official nonnegative-only bounds. |
| `train_digital_pretrained` | Only the selected FP32 state was saved; final state, selected epoch, step count, and resumable optimization were missing. | Separate final/best/resumable states, complete metadata, validation-selected initialization hash, and epoch-boundary resume. Do not extend the fixed 30 epochs when underfit. |
| `train_hwa_all` | Final metadata incorrectly labelled the final epoch as the best-validation epoch; final/best exported checkpoints lacked initialization identity and standalone optimizer/scheduler evidence. | Distinct `checkpoint_epoch` and `selected_best_validation_epoch`, explicit optimizer-state epoch, transfer diagnostics, step count, first-update measurements, and gate state. |
| Attention and GainOnly | Existing docstrings and validation code explicitly designate a **training-only** gate/intervention. Omitting it in deployment is consistent with that interpretation, but was not sufficiently specified or tested. | Preserve that architecture and training recipe. Add an explicit, validated semantics field and a testable training-input helper; validation and snapshot inference both use raw inputs. Save gate states for audit. No deployable attention architecture is claimed. |
| Resume/completion | A history with duplicates/gaps or a mismatched resume cursor could appear complete by length alone. Data contents were not fingerprinted. | Contiguous epoch/cursor checks, pretrained identity check, actual IDX SHA-256s, tests for source/config/data changes and interrupted continuation. |
| Notebook / CLI | Notebook default launched a full CUDA study; fixed run IDs could accidentally reuse old outputs. | Smoke defaults, existing MNIST path discovery, explicit no-download, fresh run IDs and separate output root. Historical embedded outputs were preserved and labelled as historical. |
| Snapshot evaluation | Nonfinite logits and empty loaders were not explicitly rejected. | Match the existing analog evaluator's failure behavior. |
| `analytic_error_budget.load_legacy_headroom` | `legacy_reference/headroom_original.py` is absent. | Clear actionable FileNotFoundError; no replacement formulas or analytical results. |

The MLP architecture, DAR/NSR equations, regularizer strengths, HWA optimizer recipe, manual decay ordering, validation-only bin criterion, reference-cell budget, and research hypothesis were retained. The old `*_legacy` experiment entry points were not promoted as verified full-protocol paths; use `unified_experiment.run`.

## D. Transfer and optimizer measurements

Measurements below come from the new MNIST **SMOKE TEST**, not a full research experiment. Existing transfer tolerances remain `atol=2e-4, rtol=2e-3`.

| Layer | Maximum absolute weight error | RMS weight error | Maximum/RMS bias error |
| --- | --- | --- | --- |
| fc1 | 3.725290298461914e-9 | 1.614575556585507e-10 | 0 / 0 |
| fc2 | 3.725290298461914e-9 | 1.0259897376130311e-9 | 0 / 0 |

Every HWA method starts from the same selected FP32 state identity. FP32 performed 8 optimizer steps (2 epochs × 4 batches); each HWA method performed 8 additional AnalogAdam steps. First AnalogAdam maximum physical-weight changes were **0.0010000169277191162** (fc1) and **0.0010000020265579224** (fc2), measured before manual decay. Test 4 independently verifies nonzero gradients, exact group membership, and that biases, scales, contexts, and input-range parameters are unaffected by **manual weight decay**.

## E. GDC diagnostics and baseline interpretation

Complete-path instrumentation of the untouched prior cloud smoke HWA checkpoint measured:

| Layer | Input-range clipping | ADC clipping | Noiseless pre-ADC range | ADC bound |
| --- | --- | --- | --- | --- |
| fc1 | 100% | 60% | [-41.28269, 41.30299] | ±10 |
| fc2 | 100% | 10% | [-11.02545, 4.74441] | ±10 |

These refine the initial nominal estimate, which did not account for the preceding input-range clamp. They concern this **smoke checkpoint**, not your inaccessible WSL full checkpoints.

Diagnostics record input-range, DAC and ADC clipping separately, sampled post-noise/pre-clamp ranges, pre-noise signal ranges, readout magnitudes, digital output scales, ADC/DAC bounds/resolutions, and noise settings. Bias is explicitly zeroed for calibration. Input-range and output scaling still affect the historical measurement.

`GDC-safe-onehot` selects a conservative probe amplitude using both reference/current maximum weights and eight output-noise standard deviations of headroom. It retains the configured quantization and noise; it rejects any observed clipping. This amplitude selection has access to simulated weights and is **not a demonstrated hardware calibration algorithm**. One read uses 784 probes for fc1 and 250 for fc2, versus one historical summed probe. Four repeats are used for each reference/current measurement. The new smoke run had **zero observed clipping at all three stages, in both layers, for every safe calibration**. This does not promise zero tail probability for arbitrary future Gaussian-noise draws.

`GDC-ideal-onehot` independently computes the mean absolute, scaled linear one-hot response without DAC/ADC/noise. It is an explicitly ideal reference. In a controlled uniform-decay fixture with `w_now=0.6*w_ref`, the independent expected gain is 1/0.6; the continuous one-hot implementation agrees within 1e-5. With ordinary IO noise/quantization, measured safe gain was 1.66841769; ideal gain was 1.66666651. The clipped historical gain was incorrectly 1.0 in that fixture.

Safe probes can still be limited by ADC resolution and noise floor. Native GDC, safe measured calibration, ideal reference calibration, and DriftSlice have different references, probe counts, and physical assumptions. **No superiority or equal-hardware-cost claim follows from these smoke curves.**

## F. Native versus custom PCM backend

The primary backend is explicitly named `snapshot_device_state_pcm`. It keeps each programmed positive/negative conductance and sampled drift coefficient throughout the retention trajectory, then reconstructs physical weights with the original converter scale. Digital biases remain outside PCM; learned per-column scales and input ranges are restored once in `SnapshotMLP`.

Native AIHWKit `BaseNoiseModel.apply_programming_noise` interleaves programming-noise and drift-coefficient draws by conductance slice, reconstructs signed programmed weights, and retains drift parameters. Native `apply_drift_noise` **reconverts those signed programmed weights** into conductances before reading. That conversion can lose the programmed positive/negative common mode and use a new normalization. The custom backend instead retains the original two device states and original scale. The two paths therefore need not match sample by sample or have exactly the same noisy distribution.

The retained absolute-time convention is **20 s → native elapsed 0 s**, 40 s → elapsed 20 s. The official PCM formula uses `t = elapsed + t_0`, with `t_0=20`; `elapsed=0` still includes read noise when enabled. Each compensation strategy shares the same programmed main devices and the same noisy main-device read at each retention timestamp. Independent calibration/reference noise is additional and explicitly seeded.

Test 8 executes actual native programming/drift and compares:

- **Controlled setting:** programming/read noise disabled, actual native sampled drift coefficients copied to the custom fixture, perfect MVM IO. At absolute 20 s, 40 s, one year, and five years, maximum weight and forward-output differences were **0.0** on CPU. Biases were unchanged exactly.
- **Noisy descriptive setting:** 12 seeds, 8×12 weights, default PCM noise, times 20 s and one year. Mean absolute native/custom weights were 0.40070127 / 0.40018306 at 20 s and 0.20110994 / 0.20154596 at one year. Average paired RMSE was 0.02885471 / 0.03002104. RNG consumption differs, so those RMSEs are **not** a model-error estimate. This small check does not establish or reject statistical equivalence.
- **Native compensation, perfect IO:** actual GlobalDriftCompensation alpha 2.05745149; actual GlobalDriftCompensationWithExactReference alpha 2.05536056. Independently reconstructed native ratios agree within 1e-6. The custom ideal-onehot reference is the programmed/read 20-s state, whereas native ExactReference uses the unprogrammed target; these names are intentionally distinct.

DriftSlice B=1 supplies one common device gain, while B>1 applies class-dependent gains selected using target conductance classes. Tests exercise both and keep **64 reference devices total**, independent of B. Device class knowledge, reference access, and correction routing remain idealized; there is no physical circuit implementation here.

## G. Automated execution results

All tests below actually executed and reported **PASS**. Their full-protocol claims are restricted to configuration/guards; actual full training remains **NOT EXECUTED**.

| Test | Check | Result |
| --- | --- | --- |
| 1 | FP32 epoch configuration, actual shortened optimizer steps and checkpoint selection | PASS |
| 2 | Numerical logical-weight and digital-bias transfer | PASS |
| 3 | Same initialization identity/references; no post-transfer reinitialization | PASS |
| 4 | AnalogAdam class, groups, weight gradients/updates, selective manual decay | PASS |
| 5 | 154-epoch completion guard; malformed resume cursor rejected | PASS |
| 6 | Distinct final/best states and epochs after all shortened epochs complete | PASS |
| 7 | FP32-Direct creates no HWA optimizer and executes custom PCM retention | PASS |
| 8 | Programming changes state, converter inversion, official bounds, actual native APIs | PASS |
| 9 | Drift changes conductances; programmed state and coefficients persist | PASS |
| 10 | 20-second reference convention and analytical drift factor | PASS |
| 11 | Snapshot/analog forward parity, including nonfinite-output rejection | PASS |
| 12 | Attention/GainOnly training-only intervention and raw validation/deployment parity | PASS |
| 13 | Actual clipping diagnostics, safe measured and independent ideal baseline | PASS |
| 14 | B=1 versus B>1 device correction with fixed reference budget | PASS |
| 15 | Exact 75-point full grid, all anchors, five repetitions / 30 realizations | PASS |
| 16 | Interrupted FP32/HWA exact CPU continuation; source/config/data mismatch guards | PASS |

The integration smoke used **1,024 train / 256 validation / 512 held-out test samples**, two FP32 epochs, two HWA epochs per method, one training repetition, two PCM realizations, eight timestamps, and five compensation strategies: **560 finite retention rows**. Train/validation indices were disjoint. All six HWA branches and FP32-Direct executed PCM retention. PNG/PDF outputs validated. Notebook code cells compiled and imports/configuration executed; its training cell was not separately executed, because the same runner was exercised through the CLI. Historical notebook outputs were preserved.

The synthetic selection test deliberately supplies controlled validation scores to test bookkeeping; those scores are unit-test fixtures, not reported MNIST performance.

## H. Existing-results validity

No prior cloud results were edited, and no WSL results were accessible. All 40 files in the previous cloud results match their SHA-256 inventory; original notebook outputs are unchanged. New evidence lives entirely under `/workspace/iscas-audit/results`.

| Existing result | Defensible interpretation / required action |
| --- | --- |
| FP32/HWA weights and training histories | Not invalidated by the GDC issue or changed PCM upper cap: neither modifies historical training. Verify actual epoch completion, weight/bias initialization and selected checkpoint before reuse. Missing metadata is an audit limitation, not proof that training was wrong. |
| Historical final HWA checkpoint metadata | Its prior `selected_best_validation_epoch=154` can describe the final state rather than the selected best state. Recover the actual selected epoch from the training/best checkpoint or history before citing it. |
| Attention and GainOnly results | Can remain interpretable as training-only input interventions with raw deployed inference. They do not support claims about a deployable attention-equipped architecture. If that was the intended scientific claim, it needs a separately specified architecture and new validation. |
| Historical GDC curves / DriftSlice-versus-GDC claims | Preserve as `GDC-legacy-ones` results. Re-evaluate calibration on each real full checkpoint; any saturated comparison cannot establish an advantage over fair global compensation. Cloud smoke clipping is not evidence of the exact clipping rate in all full runs. |
| Historical custom retention with the upper cap | Valid only as the explicitly capped custom assumption. Re-evaluate retention for claims of faithful upstream AIHWKit conductance noise. Saved trained weights can often be reused after provenance audit; retraining is not automatically required. |
| Custom versus native PCM claims | Keep backend names and assumptions explicit. Controlled agreement does not prove equivalence under programming/read noise. Native-comparison results must be reported separately. |
| Device-level DriftSlice gains | Remain idealized simulation results; no routing, circuitry, energy or physical readout feasibility was verified. |
| Analytical error budget | NOT EXECUTED: the original equations file is missing. Restore the authentic dependency before producing analytical conclusions. |

The corrected fingerprint deliberately refuses automatic use of older checkpoints. A separately reviewed checkpoint import can preserve expensive training, but must verify initialization and completion rather than changing old manifests to make them appear compatible. No import or retraining of prior full results was performed.

## I. Evidence and reproduction

- [16-test results and measurements](/workspace/iscas-audit/results/tests-delivery/verification_results.json) and [execution log](/workspace/iscas-audit/results/tests-delivery/unittest.log).
- [MNIST smoke validation](/workspace/iscas-audit/results/smoke_verified.json) and [smoke log](/workspace/iscas-audit/results/mnist-smoke-verified.log).
- [Historical checkpoint calibration](/workspace/iscas-audit/results/historical_calibration.json).
- [Source patch](/workspace/iscas-audit/verified_pipeline.patch).
- [Prior cloud-result SHA-256 inventory](/workspace/iscas-audit/prior_cloud_results_sha256.json).
- Final smoke outputs: `/workspace/iscas-audit/results/mnist-smoke/driftslice_verified_runs/smoke/verified_audit_delivery_smoke`.

Use an already approved, compatible environment. No installation commands are part of this audit. These commands launch **only lightweight tests and smoke training**, and require fresh output names:

```bash
cd /workspace/ISCAS_CONF
source /workspace/.venvs/iscas-conf/bin/activate
export PYTHONPYCACHEPREFIX=/workspace/.cache/pycache
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export XDG_CACHE_HOME=/workspace/.cache
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python tests/test_pipeline.py --out /workspace/iscas-audit/results/my-new-tests
python unified_experiment.py --mode smoke --cpu --no-download \
  --mnist-root /workspace/iscas-environment/data/MNIST/raw \
  --out /workspace/iscas-audit/results/my-new-smoke --run-id smoke_check
python tests/validate_smoke.py \
  /workspace/iscas-audit/results/my-new-smoke/driftslice_verified_runs/smoke/smoke_check \
  --report /workspace/iscas-audit/results/my-new-smoke-validation.json
```

On WSL, activate your existing `ISCAS_CONF` environment, inspect versions/GPU first, substitute your `/mnt/d/.../MNIST/raw` path and a fresh output root, and run smoke only. Omit `--cpu` only when deliberately validating available CUDA support. Dependency changes require your approval; this audit does not authorize or perform them. No full-training command was executed or is needed to reproduce the lightweight evidence.
