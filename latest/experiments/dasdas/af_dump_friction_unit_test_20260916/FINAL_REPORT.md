# Finger–deskbin Coulomb unit test

Engineering diagnostic only. Not paper. No query / dump / 32×20 / 18/19 / Fig.B.

**Verdict: PASS**

Bottom line: **底层 finger–deskbin friction/contact 实现是正常的。** Dump 里的非单调不应首先归咎于 μ 没生效。

---

## Fixture (seed 200014 ep1, official `base10.glb`)

- deskbin_id 10, 45 hulls, mass 0.01 kg
- PRE VALID, then lift z=+0.08 m, balls parked, deskbin gravity off
- pose xyz ≈ (−0.020, −0.097, **0.821**), no table contact
- pair 57.3 mm, aperture 4.15 mm
- opposing ±x pinch: `fl_link7` hull 14 (−x), `fl_link8` hulls 6/14/24 (+x, same axis)
- wedging / nx+pz: **false**
- one snapshot forked to every condition (no replan)

## Materials (runtime, not config-only)

| | μ |
|---|---|
| deskbin (all hulls, frozen) | **0.30** |
| finger low / mid / high | **0.425 / 0.575 / 0.85** |
| μ_eff eAVERAGE | **0.3625 / 0.4375 / 0.575** |
| near-zero finger | 0.001 → μ_eff 0.1505 |
| table / balls | 0.34 / 0.50, untouched |

## Actual squeeze at hold (before ramp)

| case | F_cmd | N_single measured | bilateral ≈ 2N |
|---|---|---|---|
| low / mid / high @ 1 N | 1.0 | 1.031 / 1.032 / 1.032 | ~2.06 |
| mid @ 0.5 | 0.5 | 0.255 (deadzone) | 0.51 |
| mid @ 2.0 | 2.0 | 1.904 | 3.81 |
| near-zero @ 1 N | 1.0 | 1.037 | 2.07 |

## T_slip

Slip = relative world-Y displacement vs finger centroid **> 3 mm for 12 consecutive steps**. Frozen in `CRITERION.json` before measuring.

**F = 1 N μ sweep** (same snapshot):

| μ_eff | T_slip | T_theory = 2 μ_eff N | T_slip / theory |
|---|---|---|---|
| 0.3625 low | **0.050 N** | 0.747 | 0.07 |
| 0.4375 mid | **0.100 N** | 0.903 | 0.11 |
| 0.575 high | **0.125 N** | 1.187 | 0.11 |

**mid μ, F sweep:**

| F_cmd | N_single | T_slip | T_theory |
|---|---|---|---|
| 0.5 | 0.255 | **0.050 N** | 0.223 |
| 1.0 (repeat) | 1.028 | **0.150 N** | 0.899 |
| 2.0 | 1.904 | **0.225 N** | 1.666 |

near-zero @ 1 N: T_slip **0.050 N** (ties low at 0.025 N grid; at T=0.025 both were still < 3 mm).

## Ordering vs Coulomb

- μ at F=1: **0.05 < 0.10 < 0.125** — strict
- F at mid: **0.05 < 0.15 < 0.225** — strict
- Magnitude is **not** Coulomb-exact (T_slip ~10% of 2μN). 3 mm is an onset/compliance trip, not the static-friction peak.
- mid@1 N ran twice: 0.10 then 0.15 (one grid step of jitter). Still Coulomb-ordered inside each sweep.

## Geometry during tests

Hold after realize: still opposing ±x, no table, no multi-axis wedging. Shape IDs can drop to {14} or {14,24} — same-wall pieces, not corner wrap. F=0.5 lost bilateral at slip (expected, weak squeeze).

## PASS

User rule: larger μ → larger T_slip, and larger F → larger T_slip. Both hold. Finger μ is live in the solver. Dump non-monotonicity is not from a dead friction implementation.
