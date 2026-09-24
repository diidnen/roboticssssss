# Frozen π0 downstream implementation audit

Final Phase-A classification: **NOT_ESTIMABLE_PHASE0_BLOCKER_ACTIVEFORCING_PROBE_HANDOFF_UNDEFINED**  
Scientific rollouts started: **no**  
Audit completed: 2026-08-31T11:56:55Z

## Answer

A real action-producing π0 server and two partial action-loop implementations were found, but no implementation satisfies the complete formal chain for ActiveForcing. The blocker is not that the server is feature-only: `visual_pi0_server.py` calls the frozen policy's `infer`, returns `actions`, and only then attaches visual diagnostics. The blocker is that the only runtime integration of the frozen friction estimator uses the full scripted P4-B approach/grasp/probe trace and a deterministic scripted downstream controller. Converting it to a probe-only branch from a π0-reached grasp would change the frozen probe/estimator input contract, which is a prohibited scientific method change.

## Frozen π0 identity

- Checkpoint: `/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999`
- Policy config: `pi0_lora_tacfield_tabero` (Pi0Config, not π0.5)
- Neutral task0 prompt: `pick up the alphabet soup and place it in the basket`
- Current action-capable server: `/home/exouser/FORTE/visual_pi0_server.py`, PID 133412 at audit time, port 18881, checkpoint 49999.
- Server behavior: `InstrumentedPolicy.infer()` calls the wrapped policy and returns its 13-D action chunks; the feature tensor is diagnostic metadata, not a replacement for actions.
- No server was killed, restarted, reconfigured, or queried for a rollout during this audit.

## Located action paths

1. `/home/exouser/Tabero/analysis/results/b5_tabero_neutral_20260822_040652/scripts/b5_tabero_neutral_client.py` is the authoritative task0 full π0 E2E loop from episode reset. It executes 13-D π0 actions directly, uses the neutral prompt, replans in frozen chunks, and owns the task0 full-success term.
2. `/home/exouser/Tabero/analysis/p6g1r1_controller_grasp_vla_handoff.py::run_vla_full` is a real π0 continuation loop after a fixed staged handoff. It was built for other P6/P7 protocols, replaces predicted force slots with a fixed force wrapper, and starts a fresh tactile history rather than restoring upstream temporal context.
3. `/home/exouser/Tabero/analysis/p7b_gnp_physical_belief_force_planning.py::branch_from_query` restores only scene state and calls the P6 continuation. Its query first runs fixed G2 staging plus the P4-B approach/descend/close/hold sequence. It is not a π0-reached canonical snapshot branch.
4. `/home/exouser/FORTE/prospective_visual_context_collect.py` uses the P5 scripted staging/downstream path. Its current worker (PID 339876, parent task0 TRAIN collector PID 303716) is not a formal downstream runner and was left untouched.

## Snapshot and policy-state audit

Existing prospective/P5/P7 snapshots use `env.scene.get_state(is_relative=True)` and `reset_to` or `scene.reset_to`. This covers visible scene/robot physics but does not by itself preserve all formal mid-episode state:

- B5 `_OnlineTactileBuffer` deques and marker initialization;
- unexecuted π0 action-chunk cursor/history;
- the server-side JAX policy RNG (`Policy._rng` is split on every inference);
- episode length, task/termination stage, and observation history;
- force-action term internals such as squeeze EMA/controller targets where applicable.

No existing snapshot artifact demonstrates restoration of that complete set. A new serializer/RNG-control path could be an infrastructure correction, but snapshot work cannot advance to formal QA while the frozen ActiveForcing handoff semantics below remain undefined.

## ActiveForcing semantic blocker

- Frozen estimator: `/home/exouser/Tabero/analysis/results/active_friction_imagination_20260828_211106/FRICTION_GRU.pt`, SHA-256 `a6c9d59bfa11c2481b7dca5fec1e41e0b625bbd463ed3fad9e3e4a1cb3eeb6af`.
- `active_friction_imagination.py::load_examples` rejects any source dataset whose probe sequence is not exactly 215 steps.
- The frozen feature manifest includes phases `approach`, `descend`, `close`, `hold`, `probe_out`, `probe_back`, and `probe_hold`.
- `P5S0C_FROZEN_PROBE.json` freezes the implementation as P4-B common contact-frame shear.
- `/home/exouser/Tabero/analysis/active_friction_imagination_e2e.py` is the only found Isaac runtime loading this checkpoint. It calls P5-S0-D/P5-S0-C, records the complete scripted P4-B sequence, and uses `downstream_branch`, a deterministic controller. Its own protocol explicitly records `no_pi0_in_imagination: true` and `deterministic downstream skeleton`.
- Starting from the required π0-reached canonical pre-probe grasp leaves only the short shear/return suffix. Padding, relabeling π0 upstream as P4-B phases, replaying the scripted approach after the π0 grasp, or retraining/accepting variable-length probe-only traces would each change the frozen scientific method.

Therefore the formal runner cannot be implemented faithfully from the frozen specification. Per hard-stop condition 10, DEV microtest, TEST snapshot capture, frontier, and method rollouts were not started.

## Direct backend audit

The frozen task0 backend is the three-seed `VISUAL_INTERCEPT_RESIDUAL` ensemble selected in `/home/exouser/FORTE/task0_gpu_sidecar_20260831_050050/TASK0_GT_GATE.json`. The evaluator uses the raw ensemble sigmoid probability and selects the minimum force with probability at least ρ=0.8. The held-out gate status is `FAIL`, with frontier MAE 0.4 N and under-force rate 0.5; `Probe_status` is `NOT_REACHED`. The calibration intercept/slope in `task0_visual_generalization.py` are reporting diagnostics, not a deployable calibration transform. No threshold or checkpoint was changed.

## Root leakage audit

- Frozen TEST roots: 5174–5179 only.
- Legacy TRAIN 5100–5105 and DEV 5106–5107 remain excluded.
- The active task0 context collector was processing TRAIN seeds 5112–5173 at audit time. It had not entered 5174–5179.
- No TEST snapshot, model query, success outcome, tuning, or selection was performed in this turn.

## Process safety

At audit time GPU processes were PID 133412 (π0 server, 8656 MiB) and PID 339876 (Isaac TRAIN collector, 7636 MiB). Both were left running and unmodified.

## Source hashes

```json
{
  "FRICTION_GRU.pt": "a6c9d59bfa11c2481b7dca5fec1e41e0b625bbd463ed3fad9e3e4a1cb3eeb6af",
  "active_friction_imagination.py": "c06e09b34a27b1c3a1111d60a3be012edf6e9d604008eeb2ef0858a65d532918",
  "active_friction_imagination_e2e.py": "b510af7aaadba91316d91cc219f8a0744824c8ad4babf38ad33a3086c97248ec",
  "b5_serve_policy_with_explicit_norm_stats.py": "a3baac379926856f928b8d5aad5e3285af84c2377158ea1c4bd19440f2907d01",
  "b5_tabero_neutral_client.py": "3b38317efad4ee0b2d9e1bfc4379198bbe0d2eb146665e170070b3ad82ec0a18",
  "p5s0c_model_adjudication.py": "ecf5a57c9bffb313fd3df94bef26d1579754836f26fa154538bff7acf6e7ea9f",
  "p6g1r1_controller_grasp_vla_handoff.py": "ae21e6d3299aab4bb2b910f27d774f3685dbc8e7ef6d90abe24fadecf780b1d4",
  "p7b_gnp_physical_belief_force_planning.py": "2ef853b3d3a4ad4d1642332841dc5deb3daed71a7ca2ae8cb8d588668ec28337",
  "prospective_visual_context_collect.py": "567a34bdae3bca92df7009b39e70a13877652c4a96687dce39708436d3d5e861",
  "task0_visual_generalization.py": "176f0406f697259236eeed01d4eea5cc32e7cef639010416d98e5b082febf8fc",
  "visual_pi0_server.py": "56851e5d3fdf034eaf8bcdd2350a66187c40a1a4ed86704de85fcdf020797976"
}
```
