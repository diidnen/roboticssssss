# Direct Training Audit

Status: `COMPLETE`.

The existing `FeasibilityOnly` implementation from `gnp_style_continuous.py` was trained on 480 TRAIN branches from the fixed-scene common dataset, using H=8 x 71 input, TRAIN-only normalization, BCE feasibility loss, AdamW 8e-4/1e-4, 80 epochs, and seeds 0/1/2. DEV contains 24 held-out friction contexts and was not used for training or checkpoint selection. Online threshold remains 0.5; this audit does not apply rho_frontier.
