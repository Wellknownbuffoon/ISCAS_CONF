# Completed ResNet-compatible notebook

`DriftSlice_ISCAS_COMPLETED_RESNET_REVIEW.ipynb` uses the scientific code of the completed P1/P4 revision. The only code changes relative to P4 are execution controls: all full stages and the smoke runner default to False, and the role is `all`. Scientific definition cells, configuration, stage runners and checkpoint rules are byte-identical to that revision. Saved outputs were cleared and prose was updated to the completed 100-epoch/three-repetition protocol.

It addresses the user's goal of preserving the completed run: it has the correct retention entry, the original native-gradient NSR/DAR route, the completed batch sizes and paths, and no automatically enabled full stages. It does not modify or resume the separately uploaded 154-epoch hook-bypass experiment, and does not claim to fix its saved GPU checkpoint-restoration failure.

No original notebook or user checkpoint was modified. Native AIHWKit 1.1.0 CPU checks verify NSR scalar/gradient parity, scientific code equality, matching training/retention/ablation/diagnostics fingerprints under a common environment/data identity, completed ResNet configuration, all-method dispatch and disabled stage flags. Actual user checkpoint loading and GPU execution were not performed here. Fingerprints still require the original dataset, configuration and runtime when used in Colab.

Open in Colab and keep the default full-stage flags False. This is a review copy, not an automatic paper exporter. Continue using the P0 no-ablation paper-export workflow for existing full-run outputs. Do not use this file's ResNet-oriented configuration to relabel or reproduce the separate original MLP experiment; keep its saved protocol.
