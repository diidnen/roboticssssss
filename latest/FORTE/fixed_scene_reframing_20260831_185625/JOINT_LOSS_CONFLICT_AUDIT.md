# Joint Loss Conflict Audit

Joint feasibility loss, physics loss and IE loss all decrease under the frozen objective at every scale. The available logs contain scalar per-head losses but no per-head gradient vectors, so a gradient-cosine conflict claim is not estimable retrospectively. The correct supported statement is narrower: Joint optimizes physical trajectory/IE fidelity and feasibility BCE simultaneously, while the primary downstream selection boundary is not itself a training loss.

At 100% the final Joint losses are reported in `DIRECT_JOINT_LEARNING_CURVE_TRAINING_DIAGNOSTICS.csv`; the feasibility head is not selected by DEV full-task SR.
