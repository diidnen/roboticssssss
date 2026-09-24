# PRE-POUR retention sanity — 2026-09-16

Not paper data. Official collider, deskbin μ=0.30, finger μ only. Query 4 N / 12 mm. No wrist, no pour, no 5-ball success.

## 1. Segment

Frozen before results:

1. PRE_VALID scripted grasp (20-step gate)
2. Official query
3. Realize candidate F (150-step squeeze inner)
4. **Lift:** official `move_by_displacement(left, z=0.08, move_axis=arm)`
5. **Hold:** 100 physics steps (0.40 s) at lift-end pose
6. **Short translation:** 0.08 m toward official pour xyz `[-0.45, -0.05, 1.05]`, **current EE quaternion kept**
7. Stop. Never command pour quat / `pour_actions` / `delay(6)`

Wrist audit: max commanded/observed EE rotation during translation **< 1°** on every seed/μ.

`y_retention = 1` iff lift, hold, and translation have no persistent detach (8 recorded frames ≈ 0.128 s of squeeze<0.05 / non-bilateral) **and** terminal state is bilateral, squeeze>0.05, object still lifted.

Brief flicker allowed. Official dump success unused.

Note: seed 200014 hold frames were not recorded (`scene.step` bypasses the move hook). Hold there is inferred from lift end + translation/final. Extra seeds recorded hold (25 frames).

## 2. Seed 200014 · PRE_VALID all three μ

PRE squeeze 18.8 / 24.1 / 23.7 N (low/mid/high). Opposing ±x pinch.

| F | low | mid | high |
| --- | --- | --- | --- |
| 0.25 | 0 | 0 | **1** |
| 0.50–5.00 | 1 | 1 | 1 |

Bits: low `0` then 19×`1`; mid same; high 20×`1`.

## 3. Retention rates (200014)

| μ | all | F≤1.5 | 1.75–3.5 | F≥3.75 | vs full dump |
| --- | --- | --- | --- | --- | --- |
| low | 19/20 = 0.95 | 5/6 | 8/8 | 6/6 | full dump 2/20 |
| mid | 19/20 = 0.95 | 5/6 | 8/8 | 6/6 | full dump 17/20 |
| high | 20/20 = 1.00 | 6/6 | 8/8 | 6/6 | full dump 4/20 |

## 4. Stable success interval (200014)

- low: **0.50–5.00** (19 pts)
- mid: **0.50–5.00** (19 pts)
- high: **0.25–5.00** (20 pts)

F_min: low 0.50 = mid 0.50 > high 0.25

## 5. Monotonicity (200014)

0 success→failure reversals (full dump had 6). Once on, stays on.

## 6. Multi-seed (PRE_VALID only: 200002, 200010, 200019, 200020)

Same μ / query / grid / motion / label. Not chosen by curve shape.

| seed | pre-pour n/20 L,M,H | full dump n/20 L,M,H | pre-pour reversals |
| --- | --- | --- | --- |
| 200014 | 19, 19, 20 | 2, 17, 4 | 0, 0, 0 |
| 200002 | 14, 13, 14 | 14, 10, 10 | 2, 2, 1 |
| 200010 | 14, 12, 8 | 12, 14, 12 | 0, 3, 4 |
| 200019 | 20, 20, 20 | 19, 19, 20 | 0, 0, 0 |
| 200020 | 14, 12, 12 | 15, 13, 13 | 2, 2, 1 |

Aggregate P(retain | F) mean over 5 seeds:

- low **0.81**
- mid **0.76**
- high **0.74**

0/5 seeds show required F(low) > F(mid) > F(high). 200019 is saturated (all retain). 200010 high is still the worst.

## 7. vs full dump non-monotonicity

On **200014**, yes: chaotic 2/17/4 with reversals becomes a step function; high is no longer the failing μ; the ~6° wrist contact valley never runs.

On **5 seeds**, non-monotonicity is reduced on the previously inverted seed but **not eliminated**. Aggregate ordering is still not Coulomb (low is slightly easiest). Wrist/pour was the main source of the 200014 high-fail spike, not of a universal friction-threshold law.

## 8. MIXED

Script auto-PASS on 200014 alone is overruled.

Wrist/pour removal **helps**: 200014 monotonic, high-fail inversion gone, retain rates up vs dump on that seed.

Geometry / seed dependence **remains**: no stable 3-level force ladder; aggregate even has high ≲ mid ≲ low.

Plots: `plots/seed200014_retention_vs_F.png`, `plots/compare_fulldump_vs_prepour.png`.
