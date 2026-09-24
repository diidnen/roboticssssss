# Mid vs high first-divergence stage

Engineering only. Same PRE snapshot. Stop after lift. Not paper.

**Classification: LIFT** (first numerical fork). Retention gap itself is **not** produced by T6.

T0 after restore has stale contacts (pose restored, squeeze not). T0 in live tables is ignored; fixture PRE is used instead.

---

## F points

| F | old 200014 mid / high | why |
|---|---|---|
| **1.0** | retain / fail | first typical mismatch |
| 0.75 | retain / retain | low |
| 2.5 | retain / fail | mid |
| 5.0 | retain / retain | high |

## F=1.0 (the diagnostic pair)

| stage | mid sq | high sq | mid mode | high mode | mid hulls | high hulls | pose Δ | pair Δ | slip Δ |
|---|---|---|---|---|---|---|---|---|---|
| T0 PRE (fixture) | 24.1 | 24.1 | bi | bi | 7:6+14 · 8:24 | same snapshot | 0 | 0 | 0 |
| T1 handoff | 8.38 | 8.13 | bi | bi | 6+14 / 14+24 | 6+14 / 14+24 | 1.38 mm | 0.00 | 0.36 mm |
| T2 F applied | 4.36 | 4.21 | bi | bi | 14 / 14 | 14 / 14 | 1.39 mm | 0.01 | 0.38 mm |
| T3 realized | 1.78 | 1.96 | bi | bi | 14 / 14 | 14 / 14+24 | 1.20 mm | 0.01 | 0.27 mm |
| T4 lift start | 1.78 | 1.96 | bi | bi | same as T3 | same as T3 | 1.20 mm | 0.01 | 0.27 mm |
| **T5 lift mid** | **0.23** | **2.34** | bi | bi | 14 / 6 | 14 / 6+14 | **3.25 mm** | 0.15 | **3.04 mm** |
| T6 lift end | 2.28 | 2.18 | bi | bi | 14 / 6+14 | 14 / 14 | 1.95 mm | 0.03 | 0.81 mm |

Both `force_realized_before_motion=true`. Neither dropped. Table leaves in the first lift centimetres for both.

## Other F (after discarding stale T0)

| F | T1 | T3 | T5 | T6 | first obvious |
|---|---|---|---|---|---|
| 0.75 | 8.41 / 8.37 | 1.46 / 1.48 | 0.68 / 1.35 | 1.78 / 1.81 | none |
| 2.5 | 8.34 / 8.30 | 4.63 / 4.61 | pose 2.2 mm | 4.60 / 4.59 | T5 pose only |
| 5.0 | 8.33 / 8.28 | 8.68 / 8.67 | pose 2.9 mm | 8.68 / 8.66 | T5 pose only |

No-query control (F=1.0 / 2.5): realize still matches (~1.93 vs 2.04; ~4.37 vs 4.68). T6 both bilateral. Query history is not required for realize to work.

## Answers

3. First obvious stage: **T5 lift mid** (F=1.0 and the two larger F).
4. Last still-same checkpoint: **T4 lift start** (and T3 realize).
5. First concrete gap at T5 / F=1.0: squeeze 0.23 vs 2.34; both still bilateral; hulls 14/6 vs 14/6+14; slip Δ 3.0 mm; pose 3.3 mm; table already off for both. Mid is the one that dips.
6. Does **not** explain old mid 17/20 vs high 4/20. High never looks worse through T6; the dip heals.
7. **LIFT** as first numerical fork. The retention-causing fork is still after this (pour / remainder).

Auto script said FORCE_REALIZATION from stale T0; overruled.
