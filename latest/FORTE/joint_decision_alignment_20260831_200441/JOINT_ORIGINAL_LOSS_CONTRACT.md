# Joint-Original Loss Contract

Executable source: `/home/exouser/FORTE/gnp_style_continuous.py` and `/home/exouser/Tabero/analysis/full_task_feasibility_decoder.py`.

`L_original = 1.0 * L_physics + 1.0 * L_IE + 0.3 * L_feasibility`. `L_physics` is masked Smooth-L1 over normalized H=8 physical trajectory targets from `cf.batch_tensors`; `L_IE` is masked Smooth-L1 over adjacent-force H=8 trajectory differences from `physical_ie_loss`; `L_feasibility` is BCE-with-logits on full-task success labels. Physical and IE reductions divide masked sums by masked counts; the feasibility BCE uses the mean reduction. The physics/IE units are generated only from TRAIN traces and adjacent IE pairs.

Architecture is `JointIEFeasibility`: frozen initialized Physics-GRU trunk plus H=8 trajectory head and feasibility head. Optimizer is AdamW, lr `8e-4`, weight decay `1e-4`, gradient clipping `1.0`, 80 epochs, three seeds. TRAIN-only normalization is fit from the selected TRAIN subset. Checkpoint criterion is frozen final epoch 80; no DEV checkpoint selection.
