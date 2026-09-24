# Direct × Utility interaction report

## Finding

`PROBLEM_CLASS = MIXED_DIRECT_AND_UTILITY_PROBLEM`

The task6 anomaly is a localized interaction: Direct is severely overconfident on the five failed lowest-force branches, and the current force cost then converts that overconfidence into a low-force choice. Task-specific scaling amplifies the problem, but replacing it with an 8 N global scale rescues only one of the five failures.

## Task6 failure anatomy

Current Utility selects the lowest candidate in all five task6 failures (3.0066–3.0591 N). Direct predicts success probabilities of 0.9394–1.0000 on those failed branches. Every episode has a higher-force archived success, and Fixed-Max succeeds in all 36 episodes. GT friction selects the same failing low branch in all five cases, so the friction estimator is not the primary cause.

The 8 N global scale changes 16/36 task6 decisions and raises mean force from 3.2237 to 3.3238 N, but only one changed decision turns a failure into a success. Task6 SR moves from 31/36 (86.11%) to 32/36 (88.89%), with no collateral failures. Four failures remain because Direct’s near-saturated low-force probability still makes the lower candidate optimal under any positive force cost of the tested magnitude.

Success-Only selects a successful branch in all 36 task6 episodes at mean 3.8655 N. That comparison shows that adequate successful support and a useful probability ordering exist, but the lowest-force probabilities are not calibrated well enough for fine cost-sensitive choice.

## Candidate-level calibration

| Task | Mean Direct p | Archived success | p − empirical | Brier |
|---:|---:|---:|---:|---:|
| 0 | 0.7350 | 0.7833 | -0.0483 | 0.0892 |
| 1 | 0.5938 | 0.6944 | -0.1006 | 0.1030 |
| 5 | 0.7307 | 0.7389 | -0.0082 | 0.0439 |
| 6 | 0.9502 | 0.9722 | -0.0220 | 0.0394 |

Task6’s aggregate calibration gap is mildly conservative, which would hide the failure mechanism. At the lowest candidate rank, mean predicted success is 0.8810 versus empirical 0.8611; more importantly, the five selected failures individually receive 0.9394–1.0000. The error is conditional and decision-local, not a task-wide average bias.

## Operational attribution

- `UTILITY_SCALE_ERROR`: 1/5 current task6 failures (20%) is rescued by the preregistered 8 N global counterfactual.
- `DIRECT_CALIBRATION_ERROR / residual decision error`: 4/5 failures (80%) remain under that global scale while successful higher-force paired branches exist.
- `IDENTIFIER`: GT-friction replacement leaves task6 SR at 86.11% and selects the same failed low branches.
- `CANDIDATE SUPPORT`: not causal for these failures; each episode already contains a successful higher-force branch below 4 N.
- `CONTROLLER / EVALUATOR`: not required to explain the paired pattern. The binary full-task outcomes and same-episode higher-force successes are sufficient for the diagnosis; this analysis does not claim a separate controller-identification experiment.

The 20/80 split is a deterministic counterfactual accounting of these five archived failures, not a population variance decomposition or fresh causal estimate.

## Data-quality limits

The table contains 720 branches, 144 episode tuples, 24 root families, and five candidates per episode. Candidate outcomes and Direct probabilities are paired at fixed task/context/repeat. Current Utility and its choices reproduce exactly. Task6 has 180/180 direct labels and a passing audit. Task1 retains the pre-existing caveat that 140/180 terminal labels were reconstructed, so apparent task1 collateral under the absolute-cost rule should be treated conservatively.

All evidence is already-observed TRAIN/DEV method diagnosis. It cannot be reported as new locked-TEST evidence.

