# Current feasibility input audit

**Explicit task identity exists: YES.** This conclusion comes from executed source inspection, not paper notation. The exact builder writes `x[:, 6 + (0, 1, 5, 6).index(task)] = 1`. See `CURRENT_FEASIBILITY_FEATURES.csv` for every scalar channel, source, dimension, deployment availability, candidate independence, task dependence, identity status, privileged status, and actual normalization mean/std.

Authoritative lineage: `final_vla_v1` (48 contexts) + `confirmatory_v1` (48) = the completed 96-context main evaluation. `final_continuous_friction_generalization_v1` uses the identical phase-free wrapper SHA256 `456e19e00cc966e80b0dcbfe4af1bd3b3f653221212d43be81df1e8beb1d18cc` and the same three September 6 checkpoints. The pure builders produce bitwise identical tensors on the same saved snapshot and supplied action chunk. Formatting/temporary-variable changes explain different builder AST hashes. No architecture has been modified by this audit.

The effective ensemble contains three `GRU(10,64) + MLP(54,64) + head(128,64,1)` models. The original checkpoints have GRU input width 17; seven constant normalized phase channels are deleted exactly at load. This is a frozen weight migration, not retraining. The 54 conditions are read only at timestep zero; their repetition across eight rows is not extra temporal information.

| Columns | Features | Dimension | Source and meaning | Deployment / privileged status |
|---|---|---:|---|---|
| 0–2 | Relative Cartesian commands | 3 × 8 | First 8 online commands minus command 0 | Available |
| 3–5 | Cartesian command increments | 3 × 8 | Zero then adjacent differences; no division by dt | Available |
| 6–9 | **Explicit task one-hot** | 4 × 8 | IDs 0,1,5,6; repeated at every timestep | Known task metadata; unseen ID raises ValueError |
| 10 | Candidate force / 8 | 1 | Overwritten for each grid candidate | Available; candidate-dependent |
| 11 | Friction quadrature node | 1 | Posterior integration; true friction in training only | Inferred at AF deployment |
| 12–24, 38–50 | Two identical state copies | 13 + 13 | Zero positions (3), object velocity (3), contact force proxies (4), lateral speed (1), finger joints (2) | Velocity and speed read privileged simulator state |
| 25–37, 51–63 | Two all-one masks | 13 + 13 | Constant availability flags, including zero positions | Constant |

## Answers to the six audit questions

1. **Yes: four-way task one-hot.** No learned standalone task embedding was found, although the GRU learns weights on the categorical channels.
2. There is no `c_task` variable in the audited builder. If the paper uses that symbol for this implementation, it must explicitly include the one-hot and the first-eight-command Cartesian summaries, alongside the duplicated decision state. It cannot be described as a generic task-semantic embedding.
3. Task identity is **not only implicit** in actions/state/language. It is explicitly supplied by `plan['task']`. Language and RGB are supplied to the VLA, not directly to feasibility. Only the first three action coordinates reach feasibility; wrist rotation, gripper intent, and future force channels do not. Arm joint configuration, object orientation/geometry, target location, task contact geometry, and later VLA replans are not direct feasibility features.
4. The feature arithmetic is shared, but ID lookup is restricted to `(0,1,5,6)`. Additional task-specific routes include instruction lookup, object/target selection, asset geometry and grasp initialization, sensor bindings, `current_runtime_core.validate_call`, the belief loader's qualified-task guard, and the basket terminal evaluator. The canonical probe motion/force constants are shared. Task-dependent scene configuration is separate from learned input construction.
5. Feasibility normalization is **one pooled TRAIN-only vector**, shared by all seeds/tasks, fitted on 431 rows over samples and timesteps. Original small-variance channels use std=1. Seven deleted phase entries are removed from both mean and std. The belief loader also uses pooled TRAIN normalization. VLA normalization is explicitly loaded from checkpoint `assets/NathanWu7/tabero`; its directory name differs from the current config's dataset name `NathanWu7/tabero_object_25`. Neither directory name proves the exact training task manifest. No task-indexed normalizer is loaded by the authoritative server.
6. No explicit task-conditioned friction prior is fed to primary AF: it uses the pooled probe ensemble posterior. Learned task-dependent feasibility priors are possible through the one-hot. Task-specific object geometry, scene physics/default masses, target regions and grasp initialization are real preprocessing priors. The September 2 historical protocol lists task-specific Fmax values, but **the September 6 model runtime used here has universal Fmax=5 and force support [3,5]**. The no-query-prior baseline is not the primary AF posterior.

## Deployment caveats and compatibility

Object velocity and lateral speed come directly from the captured simulator `root_velocity`; no deployed visual estimator is present in this path. Contact-vector proxies are simulation sensor readbacks. Neither fact should be hidden by calling every channel deployment-observable. The zero position entries have all-one masks despite containing no measured position. Both state blocks are identical, not pre/post state differences.

Unseen task IDs raise `ValueError: tuple.index(x): x not in tuple` in the unmodified builder. The input audit does not remove, zero, average, or relabel the task code. An all-zero unknown code is dimensionally possible but is an untrained encoding outside the four one-hot vertices; aliasing an unseen form to a seen ID would inject an arbitrary task prior. Neither is an established compatibility contract. This limitation does **not** block Fixed-5 VLA-only qualification, which does not need feasibility. Any eventual AF compatibility mapping must be declared and frozen before AF outcomes, with its extrapolative status reported.

## Primary sources

- `/media/volume/newdata/exouser/online_vla_activeforcing_20260907/{final_vla_v1,confirmatory_v1,final_continuous_friction_generalization_v1}/SOURCE_SNAPSHOT/{worker.py,phase_free_feasibility.py,common.py,runtime.py}`.
- `/home/exouser/FORTE/current_fulltask_feasibility_runtime.py` and `train_current_fulltask_feasibility_baseline_20260906.py`.
- `/home/exouser/FORTE/analysis/results/current_fulltask_feasibility_baseline_v1_20260906/TRAINING_PROTOCOL.json`.
- `/home/exouser/FORTE/analysis/results/current_runtime_recovery_v2_20260905/{branch_execution.py,geometry_grasp_initializer.py}`.
- `/home/exouser/FORTE/analysis/results/current_multitask58_loader_candidate_20260905/continuous_belief.py` and `/home/exouser/FORTE/current_contract_belief_features.py`.
- `/home/exouser/FORTE/online_vla_restore_20260907/server.py`.

Reproduce: run `audit_model.py` with `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python` (NumPy and PyTorch required; no simulator launch). `evidence/MODEL_INPUT_VERIFICATION.json` records checkpoint hashes, builder parity and unsupported-ID exceptions.
