# UPS NutAssemblySquare state-parity report

Audit date: 2026-08-29 UTC  
Scope: **INTERVENTION only**

## Status

**NOT RUN — BLOCKED_BY_MISSING_UPS_ARTIFACTS**

The Phase 0 stop condition was reached before creating a paired-rollout harness. The official learned intervention gate, residual policy, identifiable intervention checkpoints, and `scripts/run_ups` runtime are unavailable publicly. State parity alone cannot make a surrogate intervention scientifically faithful, so no simulator state was selected, snapshotted, restored, or branched.

This is not a failed parity test. It is an intentionally unexecuted test.

## What the accessible simulator supports

The pinned public stack provides the basic simulator-state interfaces needed for a future parity harness:

- `robomimic/robomimic/envs/env_robosuite.py::EnvRobosuite.get_state` returns model XML and flattened MuJoCo state.
- `robomimic/robomimic/envs/env_robosuite.py::EnvRobosuite.reset_to` can reload XML and restore flattened state.
- `robosuite/robosuite/utils/binding_utils.py::MjSim.get_state` and `set_state_from_flattened` expose MuJoCo time, qpos, and qvel.

These interfaces make exact-state branching technically plausible, but qpos/qvel alone are insufficient for this experiment.

## State that must be preserved once official UPS is available

| State family | Required parity evidence |
|---|---|
| Simulator | Identical model XML, MuJoCo time, qpos, qvel, and derived observations. |
| Environment | Identical timestep, elapsed time, done state, task instruction/behavior mode, horizon, observable caches, and environment RNG. |
| Robot controller | Identical controller goals, orientation reference, interpolators, gripper goal, and any controller buffers. |
| Base policy | Identical `BasePolicy.act_history`, `prev_img`, `prev_obs`, input normalization, and diffusion-sampler RNG. |
| UPS gate/residual | Identical observation history, classifier/residual hidden state if any, threshold/config, and activation state. Exact fields are **NOT_DETERMINABLE_FROM_CODE** until the private implementation is released. |
| High-level UPS | Identical candidate actions, world-model state, verifier phase, instruction, prediction set, and any frozen external-model transcript. |
| RNG | Python, NumPy, Torch CPU/CUDA, environment, diffusion sampler, and any policy-specific generators. |

## Prospective validation procedure

Once the official runtime and weights are available, parity must be validated before outcomes are observed:

1. Build two fresh instances from the same official environment/controller configuration.
2. Restore the same initial XML/state and replay the identical recorded action prefix to `t0`; prefix replay reconstructs controller state more safely than copying qpos/qvel alone.
3. Clone policy histories and every relevant RNG state.
4. Assert equality of MuJoCo state, observations, policy inputs, instruction/task state, and the next frozen base-policy action.
5. Run a no-intervention replay from both copies to confirm deterministic terminal equivalence.
6. Reject a pair on any mismatch before enabling the branch treatment.

## Parity result

| Check | Result |
|---|---|
| Simulator snapshot/restore | NOT RUN |
| Observation equality | NOT RUN |
| Controller-state equality | NOT RUN |
| Policy-history equality | NOT RUN |
| RNG equality | NOT RUN |
| Next-base-action equality | NOT RUN |
| No-intervention deterministic replay | NOT RUN |
| Valid paired states | 0 |

No parity claim and no paired causal claim can be made from this report.
