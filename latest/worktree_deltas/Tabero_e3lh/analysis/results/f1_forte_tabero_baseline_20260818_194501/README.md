# F1 — FORTE reproduction + Tabero reactive-force baseline

`METHOD_CHANGE = NONE`

This round does **not** implement physical-belief / probing / RL / Exp-Force VLM.

## What was done

1. Cloned official FORTE to `/home/exouser/FORTE` at `7f88d0184c1617ed95e67502da96e60be07b3689` (main, clean).
2. Mapped force estimator, slip detector, and reactive gripper loop **from code**.
3. Hardware audit: this machine has **no** FORTE serial devices and **no** recorded tactile traces.
4. Independent venv software smoke: SVR loads; slip math runs; demos not reproduced.
5. Extracted a Tabero-mapped FORTE-style controller spec (position-increment logic → measured-force servo + slip increment).
6. Prepared Tabero experiment script `scripts/f1_forte_tabero_baseline.py` (scripted arm, cream cheese, friction grid, fixed/oracle/tactile).
7. **Did not launch Isaac** this round: GPU has 2.5 GiB free (see `GPU_BLOCK.txt`).

## FORTE: what could actually be reproduced

```text
YES: import pipeline, load SVR_ckpt.pkl, run synthetic unit tests
NO:  live sensor demo, gripper demo, Franka lift, paper 310-trial table
NO:  "FORTE demo reproduced"
```

## Tabero CSVs

If a file is `NOT_RUN`, it was not fabricated.

Prior round T1 (`analysis/results/t1_tabero_downstream_decision_value_20260818_182357`) already showed a **calibrated opening servo** tracking 2–12 N (measured ~2.2–12.2 N). That is **reference**, not this round's F1 matrix. T1 did not run F×μ or FORTE reactive.

## Files

See sibling `FORTE_*`, `TABERO_*`, `BASELINE_*`, `FINAL_VERDICT.json`.
