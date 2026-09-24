# Direct Task-Encoding Audit

**`ZERO_SHOT_TASK_TRANSFER_WITH_CURRENT_ENCODING = NOT_IDENTIFIABLE`.**

The frozen Direct receives a 71D nominal tensor split into a 17D step sequence and 54D static condition. The step sequence contains 6 Cartesian command features, a 7D phase one-hot, and a **4D task one-hot**. The condition contains candidate force/8, scalar friction, current 13-channel physical state plus mask, and initial 13-channel state plus mask. It receives **no RGB, frozen π0 visual feature, language embedding, VLA semantic embedding, raw Probe trace, or learned semantic task embedding**.

For a leave-one-task-out B=0 model, the target task's one-hot coordinate is never activated during source training. B=0 is therefore an unseen-coordinate extrapolation diagnostic, not semantic zero-shot new-task transfer. B=10/20/30/60 remain valid root-heldout task-specific adaptation experiments without changing the architecture.

Source: exact input construction in `trajectory_physical_imagination.py` and the frozen 17+54 Direct split in `run_probe_conditioned_wm.py`.
