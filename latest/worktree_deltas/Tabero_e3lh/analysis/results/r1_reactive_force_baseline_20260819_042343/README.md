# R1 — Fair Reactive Force Baseline (FORTE-inspired, not official FORTE)

`METHOD_CHANGE = NONE`. Tabero @ `3bda8114` remained clean.

```
THIS IS NOT OFFICIAL FORTE
THIS IS A FORTE-INSPIRED FORCE-SPACE TRANSLATION
start 4 N → GT slip → servo target 6 N → if slip persists → 8 N
```

Official FORTE `cmd_pos -= 0.006` is **not** used (F1R2: 2/10 + ~25 N slam).

## Low-friction gate (μ = 0.20, n = 10, full pick-place)

| method | lift | full | mean F | peak F | updates | notes |
|---|---:|---:|---:|---:|---:|---|
| FIXED_4N | 0.00 | **0.00** | 4.14 | 4.26 | 0 | fails as expected |
| FIXED_6N | 1.00 | **0.90** | 6.32 | 6.54 | 0 | robust force from T1 |
| FIXED_8N | 1.00 | **1.00** | 7.71 | 8.57 | 0 | high-force ref |
| REACTIVE v1 (8 mm GT) | 0.10 | **0.10** | 4.50 | 5.83 | 2.0 | target rises; measured does not |
| REACTIVE v2 (4 mm GT) | 0.80 | **0.60** | 5.88 | 8.15 | 2.0 | always ends at 8 N |

v1 **fails** R1.7 (`≤ 3/10`). One allowed earlier-onset tweak (v2) reaches 6/10 full, 8/10 lift — still **< 0.8 full SR**, still worse than fixed 6 N, and **always climbs to 8 N**.

μ = 0.50 / 1.00 **NOT_RUN** (R1.9 gated on low-friction pass).

## Why v1 fails

At the F1R2 8 mm `rel_z` trigger the object has already left the grasp (`F` 4.2 → 0 in ~150 ms). The servo then closes on air. See `DIAGNOSIS_V1.md`.

## Status

**Primary:** `R1_REACTIVE_RESCUE_FAILED`  
**Secondary:** `R1_SLIP_DETECTION_TOO_LATE`, `R1_DYNAMIC_FORCE_SERVO_FAILED`, `R1_REACTIVE_NO_ADVANTAGE_OVER_FIXED_6N`

Do **not** enter E1 (Exp-Force) or OURS.

## Files

- `REACTIVE_CONTROLLER_SPEC.json`, `ENV_PROVENANCE.json`, `FINAL_VERDICT.json`
- `LOW_FRICTION_RESCUE.csv` (v1 four methods), `LOW_FRICTION_RESCUE_V2.csv`
- `FULL_DOWNSTREAM_RESULTS.csv` (v1), `FORCE_TRAJECTORIES.csv`, `SLIP_AND_UPDATE_EVENTS.csv`
- `ALL_FRICTION_BASELINES.csv` — NOT_RUN
- `plots/`
