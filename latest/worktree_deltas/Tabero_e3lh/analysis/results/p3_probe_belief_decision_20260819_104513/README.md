# P3 — Pre-lift probe + discrete Bayesian force decision

**Status:** `P3_PROBE_SIGNAL_EXISTS_BUT_DECISION_WEAK`  
**METHOD_CHANGE:** NONE (analysis-only scripts in this directory; Tabero checkout unmodified)

Task: `Isaac-Libero-Franka-Hybrid-Tactile-v0` / `libero_object` 1  
Instruction: *Pick up the cream cheese and place it in the basket.* (no gently/firmly)

## What was tested

Always-on **Probe A**: after 4 N grasp/hold, **+2 mm base-Y then return** (1.0 s motion + 0.25 s hold), **before** the 16 cm lift. Same probe for every μ. No GT friction in the controller.

Belief: 3-state discrete Bayes, uniform prior, Gaussian likelihood on `z_imbalance_peak` fit on **even screening seeds**. Decision: min F ∈ {4,6} s.t. Σ b(μ) P(success|F,μ) ≥ 0.8.

Matched eval: μ ∈ {0.20, 0.50, 1.00} × 20 seeds × {fixed4, fixed6, oracle, probe_belief}.

## Full-task SR (honest: this-episode lift AND basket contact > 0.05 N)

| μ | fixed4 | fixed6 | oracle | probe-belief |
| - | -----: | -----: | -----: | -----------: |
| 0.20 | 0.00 | 0.95 | 0.85 | 0.90 |
| 0.50 | 0.55 | 1.00 | 0.60 | 0.90 |
| 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Lift SR: fixed4 0/1/1; all other methods **1.00** at every μ.

## Force selection (probe-belief)

| μ | 4 N | 6 N | mean F |
| - | --: | --: | -----: |
| 0.20 | 0/20 | 20/20 | 6.0 |
| 0.50 | 2/20 | 18/20 | 5.8 |
| 1.00 | 19/20 | 1/20 | 4.1 |

Mean selected force: probe-belief **5.3 N** vs fixed 6 N **6.0 N**.

## Probe cost

Duration 1.25 s, path 4.0 mm, object motion ~1.4 mm, probe failure **0/60**.

## Classification reason

The probe **does** produce a deployable pre-lift signal (force imbalance, low vs non-low AUC 0.98, 0 probe failures). Belief **does** change 4 N vs 6 N (low→6, high→4). Full SR is close to fixed 6 N.

It is **not** strong-qualified because μ=0.50 is not majority-4 N (only 2/20), so the min-sufficient-force goal is only clearly met at μ=1.00. Mid-friction posteriors sit in the τ=0.8 gray zone.

## Do not

No NN, RL, VLA, PLUME, or selective-probe policy was trained. Negative/partial P3 does not authorize OURS.
