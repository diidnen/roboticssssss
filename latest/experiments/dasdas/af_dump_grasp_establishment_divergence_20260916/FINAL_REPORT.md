# Grasp-establishment PRE divergence

Engineering only. No query / dump / sweep / 32×20 / 18/19 / Fig.B.

**Verdict: MIXED**

Same initial snapshot. Mid and high end as the same opposing ±x pinch. Low is the only clearly softer PRE. That cannot explain mid 17/20 vs high 4/20.

---

## 1. Same initial state?

Yes. Seed 200014 ep1, official 45-hull id 10. One PrefixSnapshot after `setup_demo`, then restore → paint finger μ → identical `scripted_establish_grasp`.

Initial xyz **(0.0745, −0.1414, 0.7412)** printed on every branch. Deskbin μ 0.30; finger 0.425 / 0.575 / 0.85.

Because initial x > 0, establishment is the official right-to-left transfer (~10 s), then left pinch. All three take that path.

## 2–3. PRE table (200014)

| condition | squeeze | pair (mm) | aperture (mm) | object xyz | Δrot from init | finger hulls | normals | table (Fn) | Δxyz from init (mm) | mode |
|---|---|---|---|---|---|---|---|---|---|---|
| Initial | 0.00 | 137.46 | 44.24 | 0.0745, −0.1414, 0.7412 | 0 | — | — | yes 0.10 | 0 | none |
| Low | **19.13** | 57.21 | 4.11 | −0.0121, −0.0992, 0.7433 | 32.90° | 7: 6+14 · 8: 14+24 | ±x | yes **0.81** | −86.6, +42.1, +2.1 | bilateral |
| Mid | **24.21** | 57.61 | 4.31 | −0.0086, −0.0992, 0.7427 | 32.47° | 7: 6+14 · 8: 14+24 | ±x | yes 2.68 | −83.0, +42.1, +1.5 | bilateral |
| High | **24.26** | 57.86 | 4.43 | −0.0081, −0.0986, 0.7424 | 31.91° | 7: 6+14 · 8: **24** | ±x | yes **3.34** | −82.6, +42.7, +1.1 | bilateral |

All PRE VALID, CR=1. First contact ~3.30 s. First bilateral ~9.8 s (after transfer).

## 4. Mid vs high

| | mid | high | gap | frozen fire? |
|---|---|---|---|---|
| squeeze | 24.21 N | 24.26 N | **0.05 N** | no |
| pose | | | **0.84 mm / 0.87°** | no |
| pair | 57.61 mm | 57.86 mm | **0.25 mm** | no |
| hull union | {6,14,24} | {6,14,24} | same | |
| hull assign | fl8 also on 14 | fl8 only 24 | yes | yes |
| table contact | yes | yes | same | no |
| table Fn | 2.68 N | 3.34 N | high slightly heavier on table | |

High is a bit “stickier” (wider pair, more table load, one fewer fl8 hull). It is not a different grasp.

## 5. Friction-dependent acquisition?

Weakly yes, mostly **low vs mid/high**:
- Low final squeeze 19.1 vs 24.2 N, pair 0.4 mm tighter, ~3.5 mm more −x travel, table Fn 0.81 vs 2.7–3.3.
- Mid/high traces overlap through the transfer; they only split at the last close.

200002 / 200010 / 200019 (PRE_VALID only): same story — pose/squeeze/pair within thresholds; only per-finger hull assignment flickers; `can_explain_retention = false`.

## 6. MIXED

Establishment is not μ-invariant, but mid/high PRE is the same pinch. The retention gap is not planted here.

Next fork, if needed: same PRE → realize → lift → remainder.
