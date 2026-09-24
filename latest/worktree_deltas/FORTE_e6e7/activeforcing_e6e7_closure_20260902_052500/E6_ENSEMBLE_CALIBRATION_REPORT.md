# E6 Ensemble Calibration Report

STATUS: COMPLETE_NEGATIVE_FOR_DEPLOYMENT_CLOSURE

A matched 3-member physical identifier ensemble was trained on TRAIN roots only. DEV is root-held-out; original TEST rows were not loaded.

DEV MAE=0.1431, RMSE=0.2016, 90% coverage=0.8750, uncertainty-error Spearman=-0.0557.

The interval scale was fit on TRAIN only. Epistemic variance is member spread; aleatoric variance is the mean heteroscedastic member variance. This is a physical belief diagnostic, not evidence that the downstream force decision is calibrated.

See TABLE_E6_ENSEMBLE_CALIBRATION_DEV.csv and PHYSICAL_BELIEF_PREDICTIONS.csv.
