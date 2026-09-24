# T1 — Tabero Downstream Decision-Value Qualification

**STATUS:** `T1_POSITIVE_TABERO_PHYSICAL_DECISION_TASK`  
**METHOD_CHANGE:** `NONE` (Tabero source untouched @ `3bda8114`)

Question: with a **neutral** pick-place instruction (no gently/firmly), does hidden **friction** change the **minimum measured grip force** needed for **full downstream success**?

## Answer in one paragraph

Yes, on `libero_object` task 1 (cream cheese → basket), with motion held fixed and only object μ + calibrated squeeze varied. At **4 N**, full pick-place SR is **0% (μ=0.2) / 20% (μ=0.5) / 100% (μ=1.0)**. At **6 N** it is **100% at every μ**. So `F*(μ=0.2)=F*(μ=0.5)=6 N` and `F*(μ=1.0)=4 N`. A **fixed 6 N policy is robust**; success-only VoPI is **0**. The decision value is **gentleness-conditioned**: if the robot always uses ≥6 N it does not need to know friction; if it wants the lowest successful force, it does. This is **not** Tabero π0 success rate.

## Provenance (T1.0)

See `ENV_PROVENANCE.json`. Analysis-only scripts live under `scripts/` in this folder.

## Downstream success (T1.1)

See `DOWNSTREAM_SUCCESS_DEFINITION.md`. Official term is `libero_goals_reached`. After a successful episode the success flag can **stick** into the next episode (object still on the table, `official_success=1`). Reported **full SR** therefore requires **this-episode lift AND basket contact > 0.05 N**. Official definition was not edited.

## Path A vs B (T1.2)

Official OpenPI/JAX was **not** run. Co-resident GPU jobs were **not** killed.

```
POLICY_DOWNSTREAM_REPRODUCTION = BLOCKED_BY_GPU
ORACLE/SCRIPTED_DOWNSTREAM = USED_FOR_T1
```

Scripted Cartesian motion + calibrated force servo. Neutral instruction only. `gently/firmly` unused.

Milk (`libero_object` 7): **NOT_RUN** — cream cheese already produced a force×friction structure.

## Force servo (T1.4–T1.5)

`FORCE_SERVO_CALIBRATION.csv`, `FORCE_SERVO_GATE.json`. Desired 2–12 N → measured ~2.2–12.2 N, ordered and separated. **Not** `T1_BLOCKED_BY_FORCE_CONTROL_RANGE`.

v1 place motion (4 cm insert + servo during transit) never placed; archived as `FORCE_X_FRICTION_FULL_TASK_v1_place_unreliable.csv`. v2 freezes `d_pred` after the hold and drops from 10 cm. Smoke: 2/2 official success at μ=0.5, 8 N (`logs/PLACE_SMOKE_V2.md`).

## Friction (T1.6–T1.7)

See `FRICTION_CONFIG_AUDIT.md`. Object static=dynamic μ ∈ {0.2, 0.5, 1.0}; 0.5 is nominal. Mass / appearance / geometry unchanged.

## Matrix (T1.8)

75 trials, N=5 per cell, `FORCE_X_FRICTION_FULL_TASK.csv` (per episode), `FORCE_X_FRICTION_SUMMARY.csv` (cells). Cameras/GelSight off (`T1_ENABLE_CAMERAS=0`) so Isaac (~3.5 GB) could share the A100.

## Classification notes

- **Action disagreement:** yes (`F*` 6 N vs 4 N).
- **Fixed robust force:** yes (6, 8, 12 N all 100% full SR at every μ).
- **Success-only VoPI:** 0 (belief can pick 6 N).
- **Min-sufficient-force disagreement:** yes.
- T1.10’s intended pattern is present, compressed to one grid step (4 vs 6 N), not 2 vs 8 N.

## Files

| file | |
|---|---|
| `README.md` | this |
| `FINAL_VERDICT.json` | status blob |
| `ENV_PROVENANCE.json` | freeze |
| `DOWNSTREAM_SUCCESS_DEFINITION.md` | official success |
| `DOWNSTREAM_SUCCESS_BASELINE.csv` | official policy NOT_RUN |
| `FORCE_SERVO_CALIBRATION.csv` | T1.5 |
| `FRICTION_CONFIG_AUDIT.md` | T1.6–T1.7 |
| `FORCE_X_FRICTION_FULL_TASK.csv` | 75 episodes |
| `MINIMUM_SUFFICIENT_FORCE.md` | F*(μ) |
| `ORACLE_DECISION_VALUE.json` | VoPI |
| `INTERACTION_OBSERVABILITY.md` | T1.12 |
| `PLACE_CONTROLLER.md` | v2 placer |
| `plots/` | required figures |
| `scripts/t1_oracle_downstream.py` | oracle (eval only) |
