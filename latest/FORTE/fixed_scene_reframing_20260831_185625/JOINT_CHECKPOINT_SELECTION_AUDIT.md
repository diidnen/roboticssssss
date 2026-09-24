# Joint Checkpoint Selection Audit

The frozen implementation trains 80 epochs and uses the final epoch; it does not select a checkpoint by force MAE, DEV feasibility NLL, ranking, or downstream full-task SR. There is no force-regression output and therefore no historical “best force MAE epoch” to recover from these exact logs. This is a potential prediction/decision mismatch, but the data do not support changing the criterion in this audit.
