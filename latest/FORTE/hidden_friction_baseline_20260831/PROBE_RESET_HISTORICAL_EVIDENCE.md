# Probe / Reset Historical Evidence

## Direct answer

Scene-state restore was historically used. Exact original-root restore after the full probe was not found in the strongest P4-B + FRICTION_GRU execution.

## P5-S0-D and Active Friction Imagination

Implementation: `/home/exouser/Tabero/analysis/p5s0d_fresh_e2e_q2f.py::query_with_captured_s0`.

- `run_probe_episode` first calls `env.reset(seed=...)`, so the P4-B query itself begins from a task root.
- The wrapper captures serialized `env.scene.get_state(is_relative=True)` on the second `_current_contact_frame` call.
- That point is after approach, descend, close, and hold, immediately before shear. It is named `s0` in code but is not the original task root.
- `sq` is captured after the full query.
- `run_arm` restores with `env.reset_to(state, ..., is_relative=True)` and verifies a state hash.
- Fixed/TaskF arms restore pre-shear `s0`; Query-Control/Q2F arms restore post-query `sq`.
- The AFI pilot restores post-query `sq` for imagination and real selected branches.

This proves working serialization, restore, and parity infrastructure. It does not prove `full probe → exact original root → downstream`.

## Strict-preprobe wrapper

Implementation: `/home/exouser/FORTE/continuous_preprobe_collect.py`.

- It intercepts `env.step` and captures scene state after P4-B step 190, the final hold step.
- It lets the probe finish, restores the step-190 snapshot twice, and checks a stable hash.
- The executed directory `/home/exouser/FORTE/continuous_probe_joint_20260830_110712/collection_barehost` contains DEV roots 5106/5107, strict preprobe capture rows, parity records, and completed branches.

This is useful snapshot infrastructure but its restored state is `MID_PROBE_START`, not `ROOT_START`.

## P7-B

P7-B contains scene restore and real π0 continuation, but its clean query uses G2 scripted staging before replaying the P4-B query. The scientific main later had no qualified DEV/TEST query population. It therefore cannot establish the exact current reset contract.

## Required evaluation substitution

To use the recovered sensing method under the clarified experiment, evaluation infrastructure must capture the exact original root before `run_probe_episode`, execute the unchanged probe/estimator/authorized selector, restore that exact root including all required simulator/controller/RNG state, and begin a fresh frozen π0 rollout. This substitution changes evaluation infrastructure only; the recovered sensing method must remain byte-for-byte/semantically frozen.
