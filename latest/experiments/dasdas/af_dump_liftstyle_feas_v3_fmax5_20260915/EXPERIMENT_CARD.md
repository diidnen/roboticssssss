# v3 feasibility, force support clipped to [0.25, 5] N

Retrain only. Rows are the existing `af_dump_liftstyle_feas_v3_fork_20260914`
labels with `F <= 5`. No new rollouts. Belief not retrained.

Same seed split: TRAIN 200002 / VAL 200003 / TEST 200010.
VAL is still all-success. Claim stays scripted-remainder auxiliary.
