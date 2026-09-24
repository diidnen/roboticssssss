# E3 optional 7D onboarding adapter lineage audit

## Verdict

`DIAGNOSTIC_ONLY_NOT_AUTHORITATIVE`

The separate lane `/home/exouser/FORTE/agent_lanes/Tabero_VTLA_e3_onboarding_20260902` prepared a useful engineering hypothesis for semantic-only onboarding, but it changes the historical data contract and is not admissible as the final E3 π0 pipeline without additional lineage evidence.

Observed changes include:

- a new `E3OptionalTaberoTacFieldInputs` transform instead of authoritative `TaberoTacFieldInputs`;
- omission of `tactile_prefix` when the demonstration lacks marker fields;
- seven-dimensional action normalization statistics while the model and `LiberoForceOutputs` retain a 13D deployment contract;
- `tactile_loss_weight=0`, `padding_loss_weight=0` and a new 5,000-step schedule;
- a new dataset identity and config rather than the existing strict tactile-field conversion path.

Two serving choices are both unproven: new seven-dimensional stats can be incompatible with 13D output unnormalization, while explicit reuse of the old 13D stats would differ from the normalization used during onboarding. A complete policy-construction and 13D round-trip inference test would be the minimum engineering gate, but would still not establish historical-pipeline equivalence.

The need for this exception is removed by the runtime result in `E3_TASK5_ONBOARDING_REPLAY_SMOKE_GATE_RESULT.md`: the project-native `8D assembled -> tactile replay -> 13D 7dpf` path succeeded with real force, marker, RGB, and tactile streams. Therefore final onboarding should use the existing strict converter and exact `TaberoTacFieldDataConfig` semantics. No synthetic, masked, or omitted tactile substitute is needed.

This audit does not delete or modify the diagnostic lane. It only prevents those results from being mislabeled as authoritative E3 evidence.
