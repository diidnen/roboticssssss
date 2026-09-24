# Joint Objective and Decision Contract

## Exact implementation

The fixed-scene Joint is the existing `JointIEFeasibility` implementation from `/home/exouser/FORTE/gnp_style_continuous.py` using `/home/exouser/Tabero/analysis/full_task_feasibility_decoder.py`.

## Outputs and targets

Joint has no required-force regression head. Candidate `force_N` is an input condition encoded in the 54-D condition vector (static force field `force/8.0`). The outputs are (1) an H=8 trajectory head predicting normalized physical state deltas (13 channels), and (2) a feasibility logit predicting the observed full-task success label. The adjacent-force IE target is a difference between trajectory predictions for paired candidate forces.

Thus “Joint force prediction” cannot mean force MAE unless a separate derived frontier metric is named. The 720-row common set has five stored, nonuniform continuous force samples per context; this audit uses those existing candidate cells and does not reinterpret them as a new uniform force grid. The auditable force-related quantities are trajectory loss, IE loss, feasibility probability, force ranking, threshold crossing and selected-force error.

## Objective

`L_joint = L_physics + 1.0 * L_IE + 0.3 * L_feasibility`. `L_physics` is masked Smooth-L1 on H=8 executed trajectory deltas; `L_IE` is masked Smooth-L1 on adjacent-force prediction differences; `L_feasibility` is BCEWithLogits on full-task success.

## Decision

For each candidate force, use the feasibility probability. Select the minimum candidate with `p_success >= 0.5`; use the frozen maximum-force fallback when no candidate passes. This audit does not use `rho_frontier=0.8`.

## Training / normalization

Architecture, AdamW 8e-4/1e-4, 80 epochs, gradient clipping, three seeds, H=8 input and TRAIN-only normalization are unchanged across learning-curve scales. Checkpoints are final-epoch checkpoints; no DEV checkpoint selection is performed.
