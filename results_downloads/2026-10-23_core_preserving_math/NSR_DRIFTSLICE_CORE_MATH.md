# NSR + DriftSlice: core-preserving mathematical and numerical check

## Conclusion

No NSR or DriftSlice equation was changed. The most defensible existing-parameter candidate is B=1, R=256, K=4, with NSR lambda=0.1 and gain bounds [0.5,3] unchanged. Its evidence supports at most a small validation improvement so far, not a prediction of 90% five-year test accuracy. Confirm the remaining training repetitions before deciding whether the hardware cost is worthwhile.

## What stays identical

Freeze the original completed NSR best-validation checkpoint, its physical weights, BatchNorm statistics, input ranges, training settings and noise model. Keep the original target-conductance class rule, ratio-of-means reference gain, differential-branch gain application, gain bounds, and HWA-validation bin-selection rule if evaluating DS-val. The B=1 comparison is a preset and requires no new bin selection. Do not use the later lambda pilot's epoch-zero HWA fallback as an NSR checkpoint.

The previously suggested least-squares gain is excluded: that would change the estimator. Task-aware loss, BatchNorm recalibration, noise reductions, main-array replication and prediction ensembles are also excluded from this strict unchanged-core analysis.

For each conductance class b:

$$a_b(t)=\operatorname{clip}\left(\frac{\overline{G^{ref}_b(20s)}}{\overline{G^{ref}_b(t)}},0.5,3\right),\qquad \widetilde G_i^\pm(t)=a_{b(G_{i,target}^\pm)}(t)G_i^\pm(t).$$

Only the existing parameters B (number of classes), R (total reference cells), and K (reads per reference observation) are considered.

## Why parameter tuning has a limited effect

For a programmed device, ignoring read noise temporarily, the native PCM model has

$$G_i(t)=G_{i,prog}\exp[-\nu_i\ell_t],\qquad \ell_t=\log(t/20s).$$

After class correction, the exact noiseless log-ratio relative to its own programmed state is

$$\log\frac{\widetilde G_i(t)}{G_{i,prog}}=\log a_{b(i)}(t)-\nu_i\ell_t.$$

A shared class gain cannot cancel every independently drawn device exponent. If nu is approximated by a Gaussian with SD sigma, the relative SD of its multiplicative drift factor is

$$c_{device}\approx\sqrt{\exp[(\sigma_\nu\ell_t)^2]-1}.$$

At five years, ell=15.881031. Even at high conductance, where the nominal exponent SD is only 0.008, this gives c_device=12.76%. This is an approximation for device drift variation, not an accuracy bound. Low-conductance devices, read noise, differential branches, programming noise and neural-network nonlinearities are not represented by that single number.

Your 90 recorded five-year B=1 reference gains have mean 2.171653, sample SD 0.044484, and relative SD c_gain=2.05%. The same gains are used across methods, so these are 90 reference realizations, not duplicated independent observations from different methods.

For an illustrative small-error independent model:

$$V_{residual}\approx c_{device}^2+c_{gain}^2+V_{other},\qquad c_{gain}^2(R)\propto 1/R.$$

Changing R=64 to R=256 therefore gives, omitting other errors for illustration:

$$\frac{c_{device}^2+c_{gain}^2/4}{c_{device}^2+c_{gain}^2}-1=-0.01885.$$

That is approximately a 1.89% reduction in this variance proxy, not a 1.89-percentage-point accuracy improvement. This uses the observed reference-gain variation and nominal high-conductance device variation; it is not a fitted model of the network's predictions.

The reference contribution also has approximately separate persistent-device and read terms:

$$\operatorname{Var}(\log a_b)\approx \frac{A_b}{R_b}+\frac{D_b}{R_b K},\qquad R_b=R/B.$$

Increasing K averages read noise but not each reference cell's persistent drift exponent. Increasing R can reduce both components. Increasing B narrows classes but reduces reference cells per class at fixed R; A_b and D_b also differ between classes. Consequently, more classes are not guaranteed to improve accuracy.

For a ratio U/V, the first-order variance also contains covariance:

$$\frac{\operatorname{Var}(U/V)}{(E[U]/E[V])^2}\approx\frac{\operatorname{Var}U}{E[U]^2}+\frac{\operatorname{Var}V}{E[V]^2}-\frac{2\operatorname{Cov}(U,V)}{E[U]E[V]}.$$

The stored initial and aged reference readings come from the same persistent programmed devices, so treating them as independent would be incorrect. The native simulation below preserves that shared state and does not rely on an independence assumption between those readings.

## Actual results supplied by the user

The full three-repetition, 30-realization NSR+B1 test result is 89.8828% at 20s and 85.3043% at five years. The gap to 90% is 4.6957 percentage points. Merely restoring its initial performance would give 89.8828%, still below 90%; this is not a strict upper bound on every possible calibration change.

The existing reference-budget pilot is separate validation data: fold 0 and three programmed-device realizations.

| R (B=1, K=4) | 20s validation accuracy | 5y validation accuracy | Retention loss |
|---|---:|---:|---:|
|64|89.9067%|86.2467%|3.6600 pp|
|256|90.0467%|86.4267%|3.6200 pp|

The five-year gain is +0.1800 pp. Initial accuracy also rises by +0.1400 pp, so the measured reduction in retention loss is only 0.0400 pp. Five-year cross-entropy changes from 0.675114 to 0.677818, slightly worse. Thus the accuracy increase is a small exploratory signal, not robust proof of improvement.

## Native AIHWKit numerical screen

128 independent PCM states per distribution were simulated with 513 synthetic weights each, using two explicitly synthetic distributions (uniform and concentrated near zero). The unchanged P4 programming/drift/read-noise and reference-bank code was used. Both conductance branches were corrected using the existing target-class rule. Main-array states and read seeds were paired across parameter settings. The comparison metric is reconstructed logical-weight MSE relative to target weights at five years, not classification accuracy. It includes the existing P4 programmed-conductance cap and gain bounds.

| B | R | K | refs_per_class | reference_reads_per_observation | mse_change_min_pct | mse_change_max_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 64 | 4 | 64 | 256 | 0.0000 | 0.0000 |
| 1 | 64 | 16 | 64 | 1024 | -0.2685 | -0.2664 |
| 1 | 256 | 4 | 256 | 1024 | -1.5333 | -1.4309 |
| 2 | 256 | 4 | 128 | 1024 | -1.3067 | -0.9472 |
| 4 | 256 | 4 | 64 | 1024 | -0.3743 | 0.4487 |
| 1 | 4096 | 4 | 4096 | 16384 | -1.8506 | -1.6937 |

Negative MSE changes are better. In these model screens, R=256 is a small improvement; K=16 alone is smaller; B=4 is not consistently better. Even R=4096 (64 times the original reference area and reference-read count) has limited benefit. These synthetic distributions cannot establish what B/R/K maximizes accuracy for the trained ResNet or MLP.

## MLP parity and remaining protocol differences

The original uploaded MLP NSR function and the worker NSR function were compared structurally and numerically. NSR loss and physical-weight gradients are bit-identical on the same native AIHWKit MLP. Conductance edges, target-class assignment, reference-read averaging, ratio-of-means gains and branchwise correction formulas also match. Native gain arithmetic and target-class application passed direct checks at several B/R/K settings; programmed conductances and drift exponents remained unchanged by reads.

There is an existing separate protocol difference: the supplied full MLP definition defaults to nonnegative-only programming bounds, while P4 additionally caps programmed conductance at gmax. The saved MLP manifest was not supplied, so its actual chosen bounds policy cannot be confirmed here. This analysis does not change either policy and does not claim complete physical-protocol identity. Core equation identity and full experiment identity are different assertions.

## Recommended next action

Keep the original completed NSR models and lambda=0.1. Keep B=1, K=4 and gain_max=3. If one further validation check fits the deadline and hardware cost is acceptable, confirm R=256 on folds 1 and 2 with the existing helper; do not rerun training. Apply the same resource setting to HWA when assessing NSR's incremental contribution. If the gain does not persist, retain R=64. Only a separately identified final evaluation can measure the chosen configuration's test accuracy.

The companion cell evaluates validation only and writes to a separate pilot directory. It deliberately does not mutate Config or overwrite native training/retention files. Its existing helper was previously tested with native AIHWKit CPU inference and was already exercised in the uploaded GPU reference-budget pilot.

There is no mathematically or empirically justified unchanged-core parameter setting in the supplied evidence that can currently be claimed to deliver >90% at five years.
