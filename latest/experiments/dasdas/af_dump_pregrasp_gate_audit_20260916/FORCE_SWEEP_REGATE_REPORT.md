# Force-sweep re-analysis under PRE gate

No new rollouts. Query / μ / 32×20 / official 18/19 / Fig.B untouched.

## What can actually be gated

True `PRE_VALID` needs the 20-step PRE window (nx+nx, squeeze, aperture) **before query**. That exists only on the matched-prefix probe (8/10), not inside the force-sweep logs.

| corpus | PRE log | How PRE_INVALID is assigned | BORDERLINE |
|---|---|---|---|
| official v4 32×20 | none | qCR=0 only (F-independent miss detector) | 0 |
| isolation wall 5×4 | none | open gripper ~45 mm + squeeze≈0 + qCR=0 at the shared post-query snapshot | 0 |

Query does not open a living pinch to 45 mm (probe max pair 11.9 mm), so the 45 mm cluster is a grasp-establishment miss, not a force-selection label.

`PRE_UNLOGGED` is **not** `PRE_VALID`. It only means: cannot reject as PRE_INVALID.

Retention on v4 = remainder `contact_ratio >= 0.80`. Full-task = `full_task_success_y` (identical to `success` on all 640 rows). Isolation wall has no separate dump success; retention is the task label.

## Summary counts

### Official 32×20

Gate 前 (all 32):
- N = 32
- all-F-fail = 0
- qCR=0 = 0
- monotonic retention = 12/32
- monotonic full-task = 10/32

After dropping PRE_INVALID:
- N kept = 32 (0 dropped)
- all-F-fail = 0
- qCR=0 = 0
- monotonic retention = 12/32
- monotonic full-task = 10/32

True PRE_VALID on this corpus = **0 logged**.

### Isolation wall F-grid (μ=0.425, F=0.50–1.25)

Gate 前:
- N = 5
- all-F-fail = 2
- qCR=0 = 2
- monotonic retention = 4/5 (includes 2 vacuous all-zero)
- monotonic full-task = 4/5 (same)

PRE_INVALID dropped:
- N kept = 3 PRE_VALID_PROXY
- all-F-fail = 0
- qCR=0 = 0
- monotonic retention = 2/3
- monotonic full-task = 2/3

2/2 all-F-fail and 2/2 qCR=0 are PRE_INVALID open-gripper misses.

## Friction vs F_min (v4, nothing dropped)

Spearman(μ, F_min_retention_stable) = +0.37 (n=12)
Spearman(μ, F_min_success_stable) = +0.75 (n=10)

Both have the **wrong sign** vs low-μ → higher F_min. 8/10 full-task monotonic contexts are seed **200003**, which succeeds from 0.25 N at every μ. That is a seed effect, not Coulomb.

## Residual reasons on the 32 kept v4 contexts

| reason | n |
|---|---|
| none (monotonic retention and full-task) | 10 |
| contact flicker | 14 |
| geometric wedging | 5 |
| downstream dump failure while grasp remains valid | 3 |
| high-force squeeze-induced ejection | 0 |
| controller tracking mismatch | 0 as primary |
| grasp-establishment failure | 0 |

Isolation kept residual: 1/3 high-force squeeze-induced ejection (ep1 retains only at 0.50 N then contact collapses).

## Three answers

1. **How many force-sweep anomalies were “never grasped”?** Isolation grid: **2/2 all-F-fail = PRE_INVALID**. Official 32×20: **0/32**. The Fig.B-era nonmonotonicity is not open-gripper contamination.

2. **Is force-response more reasonable after dropping PRE_INVALID?** Isolation: **yes** (40%→67% retention, all-F-fail gone). Official 32×20: **no change**.

3. **Next step?** **Yes: freeze inner/bottom μ, vary only finger–outer-wall friction.** Do not retune query. Do not auto-resample 32×20.
