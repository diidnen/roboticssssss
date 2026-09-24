# STATUS

CPU-only authoritative task0 collection is unavailable with the frozen implementation. Current validation status: **WAITING_FOR_AUTHORITATIVE_TASK0_HELDOUT_DEV**.

# GPU NON-INTERFERENCE

Did this side-car use or disturb GPU? **NO.** No collector, simulator, model, or CUDA process was launched. Only read-only `nvidia-smi`, `ps`, source, and manifest inspection was performed. `MAIN_GPU_EXPERIMENT_INTERFERED = NO`.

# TASK0 DEV COLLECTION

The frozen task0 DEV manifest contains 2 held-out roots and 90 branches: 9 forces from 3.00 N through 5.00 N, with 5 repeats per force. The authoritative DEV outcome files are not present yet.

The strict CPU microtest was stopped at preflight and no worker was launched. The exact simulator creates the environment on `cuda:0` (`/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py:702`). The frozen visual service is also configured for `cuda:0` and transfers inference inputs to that device (`/home/exouser/FORTE/visual_pi0_server.py:41,72`). With `CUDA_VISIBLE_DEVICES=""`, an authoritative branch cannot execute. Replacing those paths with a new CPU implementation would change the frozen protocol.

# SAME-OBJECT/TASK SCOPE

This is held-out-root evaluation within the same object/task distribution. It is **NOT** a cross-object generalization test. The frozen task0 DEV roots are root06 and root07.

# BASE

Held-out DEV metrics are not available.

# VISUAL RESIDUAL

Held-out DEV metrics are not available.

# FULL VISUAL

Held-out DEV metrics are not available.

# VISUAL JOINT

Held-out DEV metrics are not available.

# DOES VISUAL CONTEXT HELP?

**NOT YET EVALUABLE.** TRAIN ordering cannot answer the held-out question.

# OFFSET OR x × F?

**NOT YET EVALUABLE.** Residual-versus-Full requires authoritative DEV outcomes and aligned DEV visual features.

# DOES JOINT ADD INDEPENDENT VALUE?

**NOT YET EVALUABLE.** Full-versus-Joint requires authoritative DEV outcomes.

# FRONTIER

Not evaluable until the 90 authoritative branches are complete.

# UNDER-FORCE

Not evaluable until the real held-out frontiers exist.

# MONOTONICITY

Not evaluable on held-out DEV yet.

# PROBABILITY QUALITY

Not evaluable on held-out DEV yet.

# TASK0 GT GATE

**NOT YET EVALUABLE.** No gate result was fabricated from TRAIN data.

# FINAL CLASSIFICATION

**CPU_ONLY_TASK0_COLLECTION_UNAVAILABLE**

Operational validation state: **WAITING_FOR_AUTHORITATIVE_TASK0_HELDOUT_DEV**.

# WHAT TASK0 SUPPORTS

At present, task0 supports only the previously observed in-sample TRAIN ordering. It does not yet support a held-out visual, force-interaction, Joint-value, or GT-gate conclusion.

# WHAT TASK0 DOES NOT SUPPORT

- No cross-object claim.
- No unseen-task claim.
- Task0 only.
- Small independent-context population.
- No final multi-task conclusion yet.
- No held-out conclusion until authoritative DEV lands.

# NEXT ACTION

Leave the GPU experiment untouched and continue the existing CPU-only watcher. When the main preregistered pipeline produces `collection_dev/task0`, run frozen-model CPU inference and the preregistered task0 evaluation automatically. Do not replace DEV, retrain models, or tune the protocol.
