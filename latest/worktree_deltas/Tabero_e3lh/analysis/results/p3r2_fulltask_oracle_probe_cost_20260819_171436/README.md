# P3-R2 — Full-task force oracle + probe-cost decomposition

**Status:** `P3R2_PROBE_OK_BELIEF_DECISION_WEAK`  
**METHOD_CHANGE:** NONE

Task: `libero_object` 1, cream cheese → basket. No force adverbs.

## Phase A — F*_full (τ=0.8), N=20 matched

F* is **full pick-place**, not lift.

| μ | 4 N | 5 N | 6 N | F*_full | F*_lift |
| - | --: | --: | --: | ------: | ------: |
| 0.20 | 0.00 | 0.00 | 0.80 | **6 N** | 5 N |
| 0.50 | 0.55 | 0.95 | 1.00 | **5 N** | 4 N |
| 1.00 | 1.00 | 1.00 | 1.00 | **4 N** | 4 N |

FIXED_ROBUST = 6 N. Same mapping at τ=0.6 and 0.9 (6 N / μ=0.20 only reaches 0.80 so τ=0.9 still takes the max tested force).

## Phase B — decomposition, N=20

| μ | fixed 6 N | no-probe F* | probe+GT F* | probe+belief |
| - | --------: | ----------: | ----------: | -----------: |
| 0.20 | 0.80 | 0.80 | **0.95** | 0.90 |
| 0.50 | 1.00 | 0.95 | **1.00** | 0.95 |
| 1.00 | 1.00 | 1.00 | **1.00** | 1.00 |
| mean | 0.933 | 0.917 | **0.983** | 0.950 |

Δ_probe = SR(no-probe F*) − SR(probe+GT) = **−0.067** (probe does not tax success).  
Δ_belief = SR(probe+GT) − SR(probe+belief) = **+0.033**.

Belief vs F*_full: exact 38%, under-force 1.7% (1/60, μ=0.50 chose 4 N and failed), over-force 60% (mid mostly 6 N, high mostly 5 N). Mean selected force 5.58 N vs oracle 5.0 N vs robust 6.0 N.

## Do not

No NN / selective probe / VLA / OURS. Tabero core clean.
