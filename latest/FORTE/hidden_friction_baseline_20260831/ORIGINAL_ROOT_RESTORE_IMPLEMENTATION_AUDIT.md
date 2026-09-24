# Original-Root Restore Implementation Audit

Audit time: 2026-08-31T12:45:17Z  
Implementation status: **AUTHORITATIVE CANDIDATE FOUND; CURRENT TASK0 CONTRACT NOT EXECUTED**  
Gate 2: **BLOCKED / NOT RUN because Gate 1 is blocked**

## Strongest existing implementation

The repository already contains a credible state-serialization path; a new snapshot system is not justified.

1. `/home/exouser/Tabero/analysis/p6g1r3_raw_vla_vs_fixed_recipe.py`
   - Root construction: `env.reset(seed=root_seed)` followed by `settle_root_before_hash(...)`.
   - Root capture: `env.scene.get_state(is_relative=True)` and an exact state hash.
   - Independent branch restore: `run_one_arm(...)` calls `env.reset_to(copy.deepcopy(root_state), ..., is_relative=True)` before every arm/policy seed and verifies restore hash parity.
   - Executed evidence: `analysis/results/p6g1r3_raw_vla_vs_fixed_recipe_20260826_123552`; 80 rows and `state_parity_all=true` in its final verdict.

2. `/home/exouser/Tabero/analysis/p7b_gnp_physical_belief_force_planning.py::restore_scene_state_stable`
   - Calls `env.scene.reset_to(copy.deepcopy(state), env_ids, is_relative=True)`.
   - Then runs `env.sim.forward()`, resets observation/action managers, and recomputes `env.obs_buf` with history update.
   - This path was introduced to avoid the camera/sensor reset mutex triggered by repeatedly entering `ManagerBasedEnv.reset_to`.

3. `/home/exouser/Tabero/analysis/p7a_fixed_invocation_force_frontier.py::run_one`
   - Restores one deep-copied root state before each force branch and checks exact state-hash parity.
   - Executed evidence: `analysis/results/p7a_fixed_invocation_force_frontier_20260826_153426`; the recorded audit reports complete branch matrix, state parity, and matched force channels.

## What the saved scene covers

`env.scene.get_state(is_relative=True)` covers IsaacLab scene articulation and rigid-object state, including robot/object generalized state and controller targets represented by the scene. Existing root manifests also explicitly record root seed and object pose; the P5/P6 snapshot helpers separately expose robot joints/gripper and object pose/linear/angular velocity for observable QA.

## What is not yet proven for the requested contract

No recovered executed artifact performs **task0 original R0 → complete 215-step P4-B → restore R0 → compare fresh π0 first action**. Existing exact scene hashes do not by themselves prove equivalence of:

- rendered initial RGB and camera/sensor buffers;
- task state-machine or episode counters outside the scene state;
- Python, NumPy, Torch, environment, or simulator RNG state;
- action/observation manager history after a long probe;
- fresh π0 first action from the restored observation.

The policy itself should be freshly initialized after restore, so no pre-task π0 chunk/history needs serialization.

## Preferred recovery strategy

Use the existing state serialization first because it has executed paired-branch provenance. Compare it against deterministic reconstruction (`reset(seed)`, identical μ/config, identical settle steps) during DEV QA. If scene restore leaves sensor/history contamination but reconstruction reproduces R0, deterministic reconstruction is scientifically acceptable because the root fixture is seed/config-defined.

No new restore code was written in this turn: Gate 1 requires stopping before pipeline implementation or scientific execution.
