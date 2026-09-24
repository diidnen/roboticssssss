# Universal Probe Data Audit

**PASS for archived leave-one-task-out Probe evaluation.**

- Population: 144/144 complete fixed-P4B Probe sequences, 48 root families, 36 sequences and 12 roots per task; every sequence is 215 timesteps × 46 legal model features.
- Friction bands: 48 LOW, 48 MID, 48 HIGH; range is [0.2, 1.0] by protocol.
- Assets: task0=`alphabet_soup_1`, task1=`cream_cheese_1`, task5=`tomato_sauce_1`, task6=`butter_1`. Thus held-out task also holds out an object family; this design cannot separate task transfer from object transfer and makes no pure cross-object claim.
- Probe primitive: identical P4B common contact-frame shear for all tasks.
- Inputs include force/contact, gripper/proprioception, commanded probe action, contact state, phase, marker motion, and EEF displacement. GT friction, privileged object pose, and downstream outcome are excluded.
- Learned LOTO Probe is trained on all three source tasks only. No target-task friction label, trace, or estimator fine-tuning is used. Target evaluation is therefore root-disjoint automatically because the entire target task is excluded.
- The force-selection population has a separate complete 72-sequence/24-root archive. Canonical `(task, root_index, root_seed)` identity overlaps the old P5 TRAIN Probe population for 24/24 roots. The historical pooled estimator therefore has 24/24 overlap and cannot establish transfer. In the present LOTO turn, all 6/6 force-selection roots of the target task are excluded from Probe training; source-task overlap remains legal training-side data.

Original pooled root-heldout MAE reference: 0.0655. No root-scaling TEST file or result was discovered or read.
