# Joint Training Audit

Status: `COMPLETE`.

The existing `JointIEFeasibility` implementation was trained on the same 480 TRAIN branches and 384 adjacent-force IE pairs (48 contexts x 2 repeats x 4 adjacent pairs). It uses the frozen H=8 Physics-GRU initialization, physical trajectory + IE + 0.3 feasibility objective, AdamW 8e-4/1e-4, 80 epochs, and seeds 0/1/2. DEV is held out at complete friction-context level.

Joint remains the historical Direct+Imagination/physics-auxiliary comparison method; it is not visual/probe fusion and is not substituted for Direct online semantics.
