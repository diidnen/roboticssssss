# F1-R2 — Tabero Reactive-Force Substrate Execution

Result dir: `analysis/results/f1r2_tabero_reactive_force_20260818_205953/`

`METHOD_CHANGE = NONE`. Tabero core at `3bda8114c07584d3d53ae43fd555ca6606c76274` remained clean. This directory contains analysis scripts and outputs only.

Env: `Isaac-Libero-Franka-Hybrid-Tactile-v0`
Task: `libero_object` 1 — *Pick up the cream cheese and place it in the basket.*
Arm: `ARM_POLICY = SCRIPTED`. Instruction has no force adverb.

## Official call

Auto-gate wrote `F1_FORTE_BASELINE_QUALIFIED_IN_TABERO` because pooled lift SR rose from 0.67 (fixed 4 N) to 0.73 (FORTE position increment).

**Human review rejects that label.** The decision cell is μ=0.20 × 4 N:

| method | n | lift SR | mean F | peak F | updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed 4 N | 10 | 0.00 | 4.12 | 6.54 | 0 |
| FORTE `cmd_pos -= 0.006` | 10 | **0.20** | 5.72 | 9.61 | 9.1 |
| fixed 8 N | 10 | 1.00 | 7.72 | 8.57 | 0 |

The two FORTE “rescues” peaked ~25 N with gripper slammed closed (`d_pred_final = 0`). The other eight stayed at ~4 N despite ~9 increments. Official FORTE is a position increment that is supposed to raise contact force; on Tabero’s 13-D hybrid action the squeeze target fights that close, so force usually does not rise.

Do **not** enter E1 (Exp-Force one-shot) or OURS.

## What did pass

1. **Low-force servo** tracks 1–8 N (measured ~1.20, 2.20, 3.29, 4.23, 6.00, 7.75 N). Far below S1’s ~21 N floor.
2. **Friction changes F\***. Lift τ=0.8: F*(0.20)=6 N, F*(0.50)=4 N, F*(1.00)=4 N. Same object/mass/geometry/trajectory; only object μ overridden after reset.
3. **GT slip** is defined from simulator state (relative z drop, downward relative velocity, xy slip, contact loss, drop). Onset ~10.0–10.05 s on failed 4 N / μ=0.20 lifts.

## What did not pass

- FORTE-style **position increment does not reliably rescue** low friction from a low start.
- Full pick-and-place: `FULL_DOWNSTREAM_NOT_YET_QUALIFIED` (lift-only protocol).
- Tactile slip detector: `NOT_RUN` (completing process skipped explore step logs).

## Files

- `FINAL_VERDICT.json`, `ENV_PROVENANCE.json`, `GPU_GATE.md`
- `LOW_FORCE_SERVO_CALIBRATION.csv`, `FRICTION_RUNTIME_AUDIT.md`
- `FORCE_X_FRICTION_EXPLORATION.csv`, `MINIMUM_SUFFICIENT_FORCE.csv`
- `GT_SLIP_EVENTS.csv`, `FIXED_*_BASELINE.csv`, `FORTE_ORACLE_REACTIVE.csv`
- `plots/` — measured vs desired, success/slip vs F×μ, FORTE episode force, relative motion
- `scripts/f1r2_tabero_reactive_force.py` sha256 `027a059c3d7f8eb973d76b003589b55af8fe0a8ae324c753c1f247e999b7daec`

## Next step

Fix the Tabero mapping so a FORTE close increment actually raises measured squeeze (or run a clearly labeled `FORTE_FORCE_INCREMENT_ABLATION`, not as official FORTE). Re-qualify the reactive baseline on μ=0.20 before any Exp-Force / OURS work.
