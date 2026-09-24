# Joint Failure Mechanism Report

## Primary conclusion

`JOINT_FORCE_PREDICTION_GOOD_SELECTION_POOR` is **not supported literally as a force-regression claim**, because the exact Joint has no required-force regression head. The supported mechanism is a prediction–decision mismatch: Joint can achieve low auxiliary trajectory/IE and feasibility training losses, yet its scalar feasibility curve is not optimized directly for the minimum-sufficient-force decision.

At 100% fixed-scene held-out friction DEV, Direct full-task SR is 0.750 and Joint is 0.708. The learning curve is Direct 0.729/0.750/0.667/0.750 versus Joint 0.667/0.708/0.708/0.708 at 25/50/75/100%; Joint does not catch up.

## What the evidence does and does not show

- Force MAE/RMSE/bias: `NOT APPLICABLE`; force is an input, not a predicted output.
- Selection, boundary, ranking, monotonicity, calibration and threshold-crossing diagnostics are in the required CSVs.
- The observed boundary uses the minimum stored force with at least one success among the two repeats; it is not a rho=0.8 frontier.
- Butter is retained; `without_butter` calibration sensitivity is included.
- No gradient-conflict measurement exists in the historical logs.

## Mechanism interpretation

The exact Joint objective gives substantial weight to trajectory and intervention-effect fidelity, but only 0.3 to the feasibility BCE. Its physics head can therefore improve physical prediction without guaranteeing a calibrated, decision-safe minimum-force boundary. In the current 100% audit, Joint has worse boundary-proxy MAE (0.118 vs 0.091 N), more under-force (0.167 vs 0.125), lower exact selection (0.750 vs 0.792), worse all-pair score ordering (0.821 vs 0.906), and worse feasibility NLL (0.700 vs 0.352). Its mean monotonic-violation rate is not worse (0.037 vs 0.046), so monotonicity is not assigned as the primary cause. The evidence supports small boundary/calibration errors being amplified by the first-threshold-crossing rule, not a literal required-force regression failure.

## Classification

Primary: `JOINT_FORCE_PREDICTION_GOOD_SELECTION_POOR` (with the force-regression wording qualified as above). Supported secondary mechanism: `JOINT_THRESHOLD_CALIBRATION_AND_BOUNDARY_BIAS`. Global candidate-score ordering is also weaker, while mean monotonicity is not worse; therefore `JOINT_NONMONOTONICITY` is not claimed as the primary explanation.

TEST remains `TEST NOT OPENED`.
