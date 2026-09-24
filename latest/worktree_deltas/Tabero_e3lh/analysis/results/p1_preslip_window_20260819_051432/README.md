# P1 — Pre-Slip Information Window Qualification

**Not a method.** Information audit only. `METHOD_CHANGE = NONE`. Tabero core was not modified.

Question: at **fixed 4 N**, before irreversible slip / contact loss, do observable force / tactile / motion histories distinguish **μ=0.20 future failure** from **μ=0.50 / 1.00 future success**?

## Setup

- Env: `Isaac-Libero-Franka-Hybrid-Tactile-v0`
- Task: `libero_object` 1 — *Pick up the cream cheese and place it in the basket.* No gently/softly/firmly/tightly.
- Force: **4 N** calibrated opening servo (same as T1/R1)
- Frictions: `{0.20, 0.50, 1.00}` object static=dynamic
- Matched seeds: **20** (same seed, same scripted trajectory, same servo; only μ changes) → 60 episodes
- Control: `dt = 0.05 s` (20 Hz)
- Tactile / GelSight: **on**
- GT slip: F1R2 / R1-v1 rule (`rel_z < -8 mm` or `v_rel_z < -0.05` for 2 steps or `rel_xy > 15 mm` or contact loss / drop). **Not retuned.**
- Lift always completed (50 steps) even if the object drops, so matched duration is preserved.

Tabero commit `3bda8114c07584d3d53ae43fd555ca6606c76274`, git clean. Collector/analyzer live only under this results directory.

## Outcomes (lift at 4 N)

| μ    | lift | future fail |
|------|------|-------------|
| 0.20 | 0/20 | 20/20       |
| 0.50 | 20/20 | 0/20       |
| 1.00 | 20/20 | 0/20       |

## Event times (μ=0.20 medians)

| interval | median |
|----------|--------|
| T0 → T_micro | 100 ms (2 steps) |
| T0 → T_slip | 475 ms (9.5 steps) |
| T_slip → T_loss | 250 ms (5 steps) |
| T0 → T_loss | 700 ms (14 steps) |

## What was **not** counted as a window

Hold-phase rank-AUC (500–1000 ms before T_slip, i.e. during close/hold) can look high (e.g. squeeze AUC ~0.98, marker_max ~0.90) because μ=1.00 sits at a slightly different equilibrium and because 0.04 N / 0.4% marker offsets rank-order cleanly when variance is tiny.

Primary comparison is **μ=0.20 vs μ=0.50** (the actual 4 N decision boundary), plus practical mean-gap floors (0.40 N squeeze, 0.20 marker mean, 1 mm rel_z). Tiny hold-phase offsets are **not** a useful window.

Pooled success (0.50+1.00) is in `LEAD_TIME_SEPARABILITY.csv` for completeness; the verdict uses `DECISION_BOUNDARY_SEPARABILITY.csv` / `FRICTION_IDENTIFICATION.csv`.

## Pre-slip signals (μ=0.20 vs μ=0.50)

### Force — not useful before gross slip

Measured squeeze stays ~4.2 N on both 0.20 and 0.50 until **T_slip**. It collapses only as contact is lost. Deployable: **NO** as an early-warning signal.

### Tactile — useful from 300 ms before T_slip

After lift onset, μ=0.20 **unloads** the GelSight markers (mean displacement 4.35 → 3.69 by T_slip−300 ms; 4.16 → 3.13 by T_slip−200 ms). μ=0.50 stays loaded. This is contact-patch unloading during incipient slip, not a large extra shear during hold.

- 500 ms: marker_mean AUC 0.515, Δ≈0 — **no**
- 300 ms: marker_mean AUC **1.000**, d=−5.94, Δ=0.65 — **yes**
- 200 / 100 / 50 ms: still AUC 1.0

Height-map mean tracks the same unloading. Deployable: **YES**.

### Motion — privileged, same 300 ms

`rel_z` is **SIMULATOR_PRIVILEGED_SIGNAL**. At T_slip−300 ms: −2.4 mm (fail) vs −0.54 mm (μ=0.50). AUC 1.0. Tactile is essentially the deployable view of this incipient slip. EEF velocity is the scripted trajectory and does not separate.

## Lead-time table (μ=0.20 vs μ=0.50)

Single-signal ROC-AUC is a **diagnostic probe**, not a learned method.

| lead | steps | squeeze AUC / ΔN | marker_mean AUC / Δ | rel_z AUC / Δmm | practical deployable? |
|------|------:|------------------|---------------------|-----------------|------------------------|
| 1000 ms | 20 | 0.98 / 0.04 | 0.86 / 0.018 | 0.66 / ~0 | no (hold, tiny offset) |
| 750 ms | 15 | 0.99 / 0.04 | 0.86 / 0.018 | ~0.5 / ~0 | no |
| 500 ms | 10 | 0.70 / 0.02 | 0.52 / 0.00 | 0.59 / 0.04 | **no** |
| 300 ms | 6 | 0.70 / 0.01 | **1.00 / 0.65** | **1.00 / 1.8** | **YES (tactile)** |
| 200 ms | 4 | 0.82 / 0.02 | **1.00 / 1.03** | **1.00 / 2.4** | **YES** |
| 100 ms | 2 | 1.00 / 0.05 | **1.00 / 1.61** | **1.00 / 3.3** | YES (already late) |
| 50 ms | 1 | 0.93 / 0.07 | **1.00 / 1.74** | **1.00 / 4.1** | YES (too late for control) |

## Diagnostic hierarchy

- **Q1** 500 ms before slip, single deployable signal obviously split? **No.**
- **Q2** 200 ms? **Yes** (already yes at 300 ms).
- **Q3** Only 50 ms? **No.**
- **Q4** Only at T_slip or later? **No.**

## Potential intervention window (not a rescue experiment)

If a controller acted at the first **practical** deployable separation:

- ~300 ms = **6 control steps** before GT gross slip
- ~550 ms = **11 control steps** before contact loss
- squeeze is still ~4.2 N (contact still loaded)

R1 showed that reacting at the 8 mm GT flag is too late. This window is earlier (~2.4 mm rel_z). Whether 4→6 N can actually be servoed in 6 steps is **not tested here**.

## Verdict

`P1_STRONG_PRESLIP_WINDOW_FOUND`

Meets the written strong bar (≥200 ms, deployable tactile, before gross slip, matched seeds, no outcome leakage). Does **not** meet the preferred 500 ms bar. The cue is **incipient-slip tactile unloading after lift starts**, not a hold-phase physics ID.

- Future 4 N failure predictable before gross slip: **YES** (from ~300 ms, tactile)
- Exact/relative friction before slip: **PARTIAL** (fail vs nearest success yes; μ=0.50 vs 1.00 not by the same primary marker/height-map at 300 ms)
- Online adaptation from current observations: only as a **short, post-lift tactile trigger**, not as a slow belief update during grasp/hold

## Next step implication

Do not default to online physical belief. Hold/pre-lift has no useful 0.20 vs 0.50 signal, so **pre-contact prior and/or explicit probe** remain the way to choose 4 N vs 6 N *before* lift. The 300 ms tactile window is an optional **lift-phase anticipatory bump**, untested.

## Artifacts

- `EVENT_TIMES.csv`, `MATCHED_EPISODES.csv`, `STEP_TRAJECTORIES.csv`
- `FORCE_TRAJECTORIES.csv`, `MOTION_TRAJECTORIES.csv`, `TACTILE_SUMMARIES.csv`
- `LEAD_TIME_SEPARABILITY.csv` (pooled success), `DECISION_BOUNDARY_SEPARABILITY.csv` (0.20 vs 0.50), `FRICTION_IDENTIFICATION.csv`
- `plots/aligned_*.png`, `lead_time_auc.*`, `preslip_window_summary.*`
- `FINAL_VERDICT.json`, `ENV_PROVENANCE.json`
