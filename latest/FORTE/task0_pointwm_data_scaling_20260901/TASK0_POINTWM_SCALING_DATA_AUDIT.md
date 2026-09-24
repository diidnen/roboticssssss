# Task0 Point-WM Scaling Data Audit

**PASS.** All archived S6/S15/S30/S50 branches contain complete H8 physical telemetry. No branch was removed or repaired. The primary experiment is explicitly a **GT-PHYSICS SCALING DIAGNOSTIC** on the **FROZEN HELD-OUT TASK0 TEST**.

## TRAIN sets (manifest-real counts)

| Scale | Independent roots | Friction contexts | Force cells | Branches | Forces/context | Repeats/force | H8 complete | IE pairs | RGB | π0 features | Labels | State parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S6 | 6 | 18 | 90 | 180 | 5 | 2 | 180 | 144 | 18 | 18 | 180 | 180 |
| S15 | 15 | 27 | 135 | 270 | 5 | 2 | 270 | 216 | 27 | 27 | 270 | 270 |
| S30 | 30 | 42 | 210 | 420 | 5 | 2 | 420 | 336 | 42 | 42 | 420 | 420 |
| S50 | 50 | 62 | 310 | 620 | 5 | 2 | 620 | 496 | 62 | 62 | 620 | 620 |

The counts differ from the rough 60/150/300/500 expectation. S6 contains three friction-conditioned contexts per one of six root families (18 contexts); each later root contributes one frozen friction context. Every context has 5 candidate force cells × 2 repeats, so the real branch counts are 180/270/420/620.

## Nested identity

`S6 ⊂ S15 ⊂ S30 ⊂ S50` is exact by frozen root family identity. `TASK0_POINTWM_SCALING_ROOTS.csv` records the 50 roots in frozen manifest order with first inclusion scale, branch count, friction, and simulator seed.

## H8, nominal motion, IE, and identity

All branches have the required corrected force/velocity columns, finite H8 state targets, nominal π0 command columns, full-task labels, simulator seed, root identity, post-probe snapshot hash, and state-parity pass. IE pairs are adjacent forces within the same context and repeat; counts are printed above.

## FROZEN HELD-OUT TASK0 TEST QA

- Roots: 10 exact; TRAIN overlap: 0
- Force cells / branches: 90 / 450
- Archived grid: 9 forces/root × 5 repeats/cell
- State parity / telemetry / success labels: 450 / 450 / 450
- Visual features / RGB contexts: 10 / 10
- Controller evaluation: 50 real episodes per method/seed (10 selected cells × 5 archived repeats)

TEST is read-only and previously evaluated. No tuning, root/checkpoint choice, or protocol changes are permitted from these outcomes.

## Probe secondary eligibility

Existing legal traces are complete: TRAIN 62/62, TEST 10/10. Secondary Probe-PointWM is RUN; it does not replace the GT-physics primary curve.
