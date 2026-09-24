# Task0 Point-WM Model Compatibility

**Decision: `REUSE_EXACT_COMPATIBLE_ROOT_SCALING_DIRECT_FINAL_CHECKPOINTS`.**

The prior Task0 `Base` checkpoint and current formal Direct have identical tensor topology, information, preprocessing, BCE objective, optimizer, epochs, seeds, GT-friction source, absence of visual input, and candidate-force normalization. The only implementation difference is module naming (`command_gru/condition` versus `gru/cond`). Final Direct checkpoints can therefore be reused without changing predictions; grouped-root inner models are newly trained because residual OOF predictions do not exist for these scales.

| Item | Prior Task0 Base | Current Direct | Compatible |
|---|---|---|---|
| Architecture | GRU17→64 + condition54→64 + head128→64→1 | same | yes |
| Inputs | x, μ_GT, candidate F | x, μ_GT, candidate F | yes |
| Preprocessing | per-scale TRAIN-only normalization | same | yes |
| Objective | full-task BCE | full-task BCE | yes |
| Seed / optimizer / epochs | 0/1/2; AdamW; 80 | same | yes |
| Friction / visual | GT; none | GT; none | yes |
| Candidate force | nominal segment then TRAIN normalization | same | yes |

## Old Visual Joint is not Point-WM

Old Visual Joint is incompatible and is not reused: it consumes visual input, receives outcome BCE through the joint model, and does not use the frozen PhysicsOnly→root-OOF residual→expected-utility controller protocol.
