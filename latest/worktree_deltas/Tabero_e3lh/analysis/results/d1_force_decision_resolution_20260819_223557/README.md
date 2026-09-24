# D1 — Force decision resolution audit

**Status:** `D1_REFINED_DECISION_REDUCES_OVERFORCE_BUT_SR_WEAK`  
**METHOD_CHANGE:** NONE. Probe unchanged (P3 A, +2 mm base-Y).

## Offline (even calib / odd held-out, P3 probe_belief trajectories, N=10×3)

- LOW vs NOT-LOW: `imb_peak` AUC **0.98** (calib), **1.00** (held-out).
- MID vs HIGH: `ftan_hyst` (tangential-force forward/return hysteresis) calib AUC **1.00**, Cohen’s d ≈ −29; held-out AUC **1.00**. Means ≈ 0.002 (μ=0.50) vs ≈ 0.022 (μ=1.00).
- GelSight `mtang_hyst` also mid-vs-high AUC 1.00 (not selected; force hysteresis had larger |d|).
- Refined = Gaussian NB on `{imb_peak, ftan_hyst}`, full-task success table, τ=0.8.

Held-out odd seeds: exact **0.93**, under **0**, over **0.067**, mean F **5.1 N** (current 1D: 0.33 / 0 / 0.67 / 5.7 N). Offline gate **passed**.

## Confirmation (new seeds 20–29, N=10, frozen probe, cameras off / force-only)

| | full SR | mean F | exact | under | over |
| - | ------: | -----: | ----: | ----: | ---: |
| fixed6 | 0.97 | 6.00 | 0.33 | 0 | 0.67 |
| current P3 1D | **1.00** | 5.63 | 0.40 | 0 | 0.60 |
| refined 2D NB | 0.93 | **5.20** | 0.80 | 0.03 | 0.17 |
| oracle | 1.00 | 5.00 | 1.00 | 0 | 0 |

Refined μ=0.20: 9×6 N + **1×5 N (full fail)**. That under-force is from over-confident tiny calib σ on `ftan_hyst` (mid σ≈1e-4) pulling a low-friction episode into the mid class.

New-seed hysteresis means still separate: 0.0012 / 0.0022 / 0.020.

## Interpretation

The probe **does** carry fine 5 N vs 4 N information. Joint NB with empirical σ is **not** a safe mapper. Do not train a network to squeeze this; if anything, next is a hierarchical 6 N veto on imbalance plus a floored-σ rule set on calibration only.
