# Post-query state divergence (shared PRE snapshot)

Engineering diagnostic only. Not paper. No dump remainder / 32×20 / 18/19 / Fig.B.

**Verdict: MIXED**

Shared PRE + official 4 N / 12 mm world-Y query does **not** send high to a worse handoff than mid. The old 200014 gap (mid post-query squeeze 1.08 vs high 8.83) does **not** reproduce when grasp is frozen first.

---

## Fixture

- seed 200014 ep1, official `base10.glb`, 45 hulls
- deskbin μ = 0.30 frozen; PRE formed at finger μ = 0.575
- PRE VALID, CR=1
- PRE pose xyz = (−0.0078, −0.0985, 0.7424), squeeze **24.05 N**, pair 57.84 mm
- opposing ±x: fl_link7 hulls 6+14, fl_link8 hull 24
- table still in contact (official query protocol; no extra lift)
- one PrefixSnapshot; all four branches restore it, then paint finger μ

Runtime μ confirmed: finger 0.425 / 0.575 / 0.85, deskbin 0.30, μ_eff 0.3625 / 0.4375 / 0.575.

Query duration ≈ 201 physics steps (~0.80 s). No-Query = same duration, force limit 4 N, no arm motion.

Optional lift probe was **not** run: frozen mid-vs-high thresholds did not fire.

## Handoff table

| condition | final squeeze | qCR | pair (mm) | Δxyz (mm) | Δrot (deg) | finger hulls | slip (mm) | mode |
|---|---|---|---|---|---|---|---|---|
| No-Query | 8.36 | 1.00 | 57.93 | −0.35, −0.06, +0.11 | 0.49 | 7: 6+14 ; 8: 24 | 0.48 | bilateral |
| Low | 4.89 | 1.00 | 57.51 | −2.68, +1.27, +0.93 | 1.67 | 7: 6+14 ; 8: 24 | 3.11 | bilateral |
| Mid | 9.38 | 1.00 | 57.50 | −2.79, +1.24, +0.86 | 1.58 | 7: 6+14 ; 8: 14+24 | 2.93 | bilateral |
| High | 8.61 | 1.00 | 57.62 | −2.50, +0.41, +0.78 | 1.55 | 7: 6 ; 8: 14+24 | 2.73 | bilateral |

Normals stay ≈ (−x, +x). No multi-axis wedging. Union hull set is {6,14,24} on every branch.

## Mid vs high (frozen thresholds)

| check | mid | high | fire? |
|---|---|---|---|
| squeeze | 9.38 N | 8.61 N | no (Δ 0.77 < 1 N) |
| object pose | | gap 0.88 mm / 0.33° | no |
| pair | 57.50 mm | 57.62 mm | no |
| rel slip | 2.93 mm | 2.73 mm | no |
| hull union | 6,14,24 | 6,14,24 | no |

High slipped **less** in world-Y (0.49 vs 1.20 mm), as Coulomb predicts. That is not a worse pinch.

## What this does and does not explain

- Low is the only branch that **relaxes** (handoff 4.89 N). Mid and high stay near the No-Query residual (~8.4 N) after the gripper cap drops to 4 N.
- Therefore the previous 8.83 vs 1.08 reading was not “same PRE, query sticks high.” It came from **μ-dependent grasp establishment** (separate PRE per μ), not from query on a shared snapshot.
- high < mid retention (4/20 vs 17/20) is **not** explained by a worse post-query geometry here. Next place to look is downstream force realization / remainder dynamics, or PRE that is itself μ-dependent.

## Verdict

MIXED: low residual squeeze differs; mid ≈ high in pose, hulls, mode, and squeeze. That cannot explain high < mid.
