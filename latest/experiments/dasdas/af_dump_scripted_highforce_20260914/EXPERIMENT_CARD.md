# Pure scripted dump × multi-force (controlled-motion auxiliary)

Full-task `dump_bin_bigbin.play_once()` with RoboTwin AF **single-finger**
gripper drive limits. No π₀. Within each seed×friction context, only the
commanded force changes.

## Claim

Scripted controlled-motion feasibility / force-causal labels.
**Not** online continuous AF efficacy.

## Design

- Seeds `200002/200003/200004` × μ `.425/.575/.85` = **9 contexts**
- Forces `10/12/15/16/18/20` N (single-finger both grippers) = **54** labels
- Smoke: seed 200002 μ0.85, forces `12/16/20` (12N must succeed)
- Parallel contexts OK (CPU-only; no shared pi0 RNG)

## Force convention

`af_force_limit_n` = single-finger drive limit (RoboTwin AF / forcegrid style).
Not NativeOriginal bilateral F with F/2 per finger.
