# Authoritative frozen π0 backend

- backend: project-local OpenPI (not LeRobot)
- live server PID at audit: `33931`, port `18881`, mode `control`
- policy config: `pi0_lora_tacfield_tabero`
- checkpoint: `/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999`
- norm stats: checkpoint-local `assets/NathanWu7/tabero`
- OpenPI source commit: `1ed9cf44c05bc63fa3b3dbc0ca83ce9dbc8b7b2e`
- Tabero commit: `80ab3be09ce884f86cfc2037d3af30bc28061426`
- FORTE commit: `7f88d0184c1617ed95e67502da96e60be07b3689`
- `pi0.py` SHA-256: `7959ce59fa17b581bf3fb4f0ba685cc24eb9b6e2410dba5dad84b3da910f5638`
- `tokenizer.py` SHA-256: `965be8b3c393a6811875bbc32da9e01a5d01cc2f87802de801cf7293e049748c`
- server SHA-256: `56851e5d3fdf034eaf8bcdd2350a66187c40a1a4ed86704de85fcdf020797976`
- historical P6G1R1 environment/action adapter SHA-256: `ae21e6d3299aab4bb2b910f27d774f3685dbc8e7ef6d90abe24fadecf780b1d4`
- authoritative B5 client SHA-256: `3b38317efad4ee0b2d9e1bfc4379198bbe0d2eb146665e170070b3ad82ec0a18`
- authoritative B5 launcher SHA-256: `03b9fc9a5cedc285e2439763c6af71f3c94e856358c846cebb1d835a04eb5b44`
- Isaac client Warp provenance: bundled `omni.warp.core-1.8.2+lx64` is first on `PYTHONPATH` with `PYTHONNOUSERSITE=1`, exactly as in the authoritative B5 runner; this prevents the incompatible site `warp-lang 1.16.0` from shadowing Isaac's Warp.
- observations: base and left-wrist RGB, uint8, 224×224; existing tactile/force history transform unchanged
- server action shape: 32D padded, effective first 13D hybrid action
- action execution: 10-step replanning chunks; 50 chunks for long-horizon qualification
- low-level controller: unchanged; robust qualification changes only the two normal force slots to the fixed safe setpoint

The qualification wrapper source-transforms only audited instrumentation/qualification seams in the historical B5 client: dynamic object name, a fixed force-setpoint override, and read-only per-goal/generic-transport/root-state-hash telemetry. It does not change preprocessing, language, action normalization, motion actions, policy weights, checkpoint, environment evaluator, or inference rate.
