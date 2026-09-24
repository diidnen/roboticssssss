# STATUS

COMPLETE — offline forensic ended at the preregistered STOP_BEFORE_COLLECTION gate. No new TRAIN branches, retraining, repeat-enriched DEV evaluation, Probe, TEST, or fresh E2E were run.

# SINGLE SCIENTIFIC GOAL

Determine whether the failed continuous reliability gate is primarily explained by two-repeat label noise, continuous-force coverage, finite-repeat physical stochasticity, or residual Feasibility-only model error.

# CURRENT BACKEND

Feasibility-only is the selected backend. Joint/world-model development is not reopened.

# REAL DEV STOCHASTICITY

Nine of 27 repeated DEV cells (33.3%) are boundary-stochastic (1/5 through 4/5); 18/27 (66.7%) are deterministic-like (0/5 or 5/5). Mean empirical Bernoulli variance is 0.062, below the frozen 0.10 criterion. With only five repeats, these values quantify finite-repeat evidence and do not establish irreducible physical randomness.

Task 1 contains the most mixed cells (6/12, 50%); task 5 has 0/6 mixed cells. MID-friction cells are more mixed (4/6, 66.7%) than LOW-friction cells (5/21, 23.8%), but probability MAE is nearly identical across MID and LOW friction (0.248 vs 0.244), arguing against observed stochasticity as the main error driver.

# HOW MANY DEV CELLS ARE 0/5 ... 5/5?

| Outcome | Cells | Fraction |
|---|---:|---:|
| 0/5 | 5 | 0.185 |
| 1/5 | 3 | 0.111 |
| 2/5 | 2 | 0.074 |
| 3/5 | 1 | 0.037 |
| 4/5 | 3 | 0.111 |
| 5/5 | 13 | 0.481 |

# TWO-REPEAT SUPERVISION NOISE FLOOR

Exact enumeration over every size-2 subsample gives expected MAE **0.080** against observed p5, RMSE 0.153, and p>=0.8 classification mismatch probability 0.063. Current model probability MAE is 0.245, or 3.06x the two-repeat observation floor.

The context-bootstrap model-minus-noise MAE gap is 0.165 (95% CI 0.081 to 0.243). Thus two-repeat label noise is materially too small to explain the current model error.

| Subsample repeats | Exact expected MAE vs p5 | RMSE | Reliability mismatch |
|---:|---:|---:|---:|
| 1 | 0.124 | 0.249 | 0.096 |
| 2 | 0.080 | 0.153 | 0.063 |
| 3 | 0.053 | 0.102 | 0.070 |
| 4 | 0.031 | 0.062 | 0.089 |

# TRAIN TWO-REPEAT QUALITY

The 360 TRAIN cells contain 65 0/2 cells (18.1%), 15 1/2 cells (4.2%), and 280 2/2 cells (77.8%). Only 4.2% are ambiguous 1/2 cells, below the frozen 10% materiality threshold. Ambiguity is not broadly distributed enough for repeat noise to dominate.

# CONTINUOUS FORCE COVERAGE

All task supports are covered by 90 continuous TRAIN forces per task. Observed sampled ranges are task 0 [3.068, 4.991]N, task 1 [4.023, 5.985]N, task 5 [3.006, 4.993]N, and task 6 [3.002, 3.995]N. Maximum pooled within-task force gaps are task 0 0.082N, task 1 0.087N, task 5 0.107N, task 6 0.064N.

DEV error does not rise with sparse force coverage: Spearman(error, nearest-force distance) is -0.223; the sparse-distance quartile has 0.194 lower MAE than the dense quartile. This fails both frozen coverage criteria.

# CURRENT MODEL ERROR TAXONOMY

Among eight valid real frontiers: 6 are ordered but over-force; 2 are ordered but never cross p=0.80; there are no finite under-force decisions and no ordering failures. The inherited under-force rate counts the two missing decisions as safety failures. This pattern is predominantly underconfidence/miscalibration of the learned probability scale, not a failure to order force correctly.

# WHAT DOMINATES THE ERROR?

**MODEL ERROR.** Repeat noise is only 32.7% of model MAE, TRAIN ambiguity is 4.2%, observed mixed DEV cells have mean variance below threshold, and coverage diagnostics do not worsen in sparse regions. The evidence does not justify calling the five-repeat DEV target exact or the process deterministic; it shows only that these data limitations are insufficient to explain the much larger current error.

# TARGETED TRAIN TOP-UP

NOT EXECUTED. The frozen TRAIN-only selection rule would have identified 81 cells (243 branches), but `MODEL_ERROR_DOMINATES` is explicitly ineligible for collection. Executing them would violate the preregistered causal test.

# REPEAT-ENRICHED MODEL

NOT_REACHED. No architecture, optimizer, calibration, checkpoint, or normalization was changed.

# OLD VS NEW GT CONTINUOUS RESULT

NOT_REACHED. The authoritative old Feasibility-only result remains probability MAE 0.245, frontier MAE 0.325N, safety-failure/under-force 2/8, finite decisions 6/8, and monotonicity 9/9.

# GT GATE

NOT_REACHED for a repeat-enriched model. The current authoritative model remains failed; no new DEV evaluation was performed.

# PROBE VS STRICT NO-PROBE

NOT_REACHED because no repeat-enriched backend reached the GT gate.

# QUANTIZATION UNMASKING

NOT_REACHED.

# PRIMARY_CLASSIFICATION

MODEL_ERROR_DOMINATES

# SECONDARY_PROBE_CLASSIFICATION

NOT_REACHED

# WHAT IS NOW PROVEN

Within the current object/task distribution, the current probability error is much larger than the exact empirical two-repeat subsampling noise floor, and neither pooled force coverage nor observed five-repeat stochasticity passes its preregistered materiality criterion. More repeats at the existing forces are therefore not scientifically justified as the next intervention.

# WHAT IS STILL NOT PROVEN

- no claim that five-repeat empirical probabilities are exact or physical stochasticity is irreducible
- no cross-object claim
- no unseen-task claim
- no original TEST
- no fresh E2E
- no when-to-probe policy
- no conclusion about Probe continuous control because GT reliability was not reached

# METHOD IMPLICATION

The candidate pipeline is not yet validated. The next bottleneck is the Feasibility-only context-to-probability mapping—especially probability scale/underconfidence—not evidence quality from two repeats.

# NEXT_METHOD

Audit which context variables or state features are missing, and test probability-scale/representation diagnostics without collecting more force outcomes. Keep reliability-aware force selection conservative; uncertainty lower bounds are a secondary safeguard, not a substitute for fixing the model relation.
