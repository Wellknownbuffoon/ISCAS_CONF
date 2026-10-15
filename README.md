# ISCAS_CONF

AIHWKit 1.1.0 MNIST FP32 → AnalogAdam hardware-aware training → PCM retention study.

Start with [the audit report](AUDIT_REPORT.md) for the training protocol, native API comparisons, corrected calibration baselines, measured transfer errors, prior-result limitations, and lightweight reproduction commands. Automated tests are in [tests/test_pipeline.py](tests/test_pipeline.py); [tests/validate_smoke.py](tests/validate_smoke.py) validates a separate smoke run without modifying its artifacts.

The notebook defaults to a **SMOKE TEST** and a fresh output directory. Full mode requires exactly 30 FP32 epochs, 154 additional HWA epochs, five training repetitions, 30 programming realizations, and 75 retention timestamps. Full training and GPU execution were not performed in the CPU audit environment.

Attention/GainOnly remain explicitly training-only interventions. Historical GDC, safe measured one-hot GDC, and ideal one-hot GDC are separately named. DriftSlice is an idealized device-level simulation, not an implemented readout circuit. The optional original analytical dependency is absent; no analytical equations or results have been fabricated.
