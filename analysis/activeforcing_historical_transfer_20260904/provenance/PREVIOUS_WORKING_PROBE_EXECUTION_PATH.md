# Previous Working Probe Execution Path

## Selected path

The authoritative executed sensing path is the 2026-08-28 task0 Active Friction Imagination pilot:

1. Entry: `/home/exouser/Tabero/analysis/active_friction_imagination_e2e.py::launch`
2. Isaac worker: `/home/exouser/Tabero/analysis/active_friction_imagination_e2e.py::run_worker`
3. Environment construction:
   - `AppLauncher(headless=True, enable_cameras=True, num_envs=1)`
   - `setup_task_objects(TASK_SUITE, 0)`
   - `parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)`
   - `gym.make(...).unwrapped`
4. Query wrapper: `/home/exouser/Tabero/analysis/p5s0d_fresh_e2e_q2f.py::query_with_captured_s0`
5. Frozen probe import: `/home/exouser/Tabero/analysis/p5s0d_fresh_e2e_q2f.py::import_p4_probe`
6. Root construction and physical query: `/home/exouser/Tabero/analysis/results/p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py::run_probe_episode`
   - `env.reset(seed=seed_idx)`
   - `_apply_friction(env, OBJ_NAME, mu)`
   - `_current_contact_frame(env)`
   - approach 45
   - descend 35
   - close 70
   - hold 40
   - probe_out 10
   - probe_back 10
   - probe_hold 5
   - returns 215 `ProbeStep` records and episode record
7. Trajectory recorder: `active_friction_imagination_e2e.py::write_csv` writes `PROBE_TELEMETRY/<context>.csv`.
8. Runtime feature builder: `/home/exouser/Tabero/analysis/p5s0d_fresh_e2e_q2f.py::OnlineInference.sequence_array`
   - 31 numeric probe channels
   - `eef_dx/eef_dy/eef_dz`
   - seven probe-phase one-hot channels
   - three contact-state one-hot channels
   - two unknown-category flags
   - total 46 channels
9. Normalization: `/home/exouser/Tabero/analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542/P5S0C_NORMALIZATION.json`
   - exact feature order
   - TRAIN-only `dynamic_mean` and `dynamic_std`
10. Estimator architecture: `/home/exouser/Tabero/analysis/active_friction_imagination_e2e.py::FrictionGRU`
    - linear projection 46→16 + ReLU
    - GRU hidden size 16
    - μ and log-σ heads
11. Estimator checkpoint: `/home/exouser/Tabero/analysis/results/active_friction_imagination_20260828_211106/FRICTION_GRU.pt`
    - SHA-256 `a6c9d59bfa11c2481b7dca5fec1e41e0b625bbd463ed3fad9e3e4a1cb3eeb6af`
12. Inference: `/home/exouser/Tabero/analysis/active_friction_imagination_e2e.py::estimator_inference`
    - output `mu_hat`, `sigma_mu`
13. Belief hypotheses: μ̂ and clipped μ̂±1.645σ.
14. Historical decision branches: `/home/exouser/Tabero/analysis/active_friction_imagination_e2e.py::branch`
    - deterministic downstream imagination at candidate forces {3.0,4.0,4.5,5.0}
15. Historical selector: `/home/exouser/Tabero/analysis/active_friction_imagination_e2e.py::choose_from_curves`
    - first force whose mean imagined success is at least 0.9
16. Old real downstream: `p5s0c_paired_boundary_probe_value.py::downstream_branch`
    - restored from post-query `sq`
    - scripted/deterministic; not frozen π0

## Snapshot points in the selected path

`query_with_captured_s0` wraps `_current_contact_frame` and captures `s0` on its second invocation. This is after approach/descend/close/hold and immediately before shear. It captures `sq` after the query. The AFI real arms restore `sq`. Neither state is the exact original root before approach.

## Formal reuse boundary

The reusable, frozen method prefix ends after `estimator_inference` for the explicit friction estimator, or after a separately identified frozen Direct selector if that selector is scientifically authorized. The old `branch`/`downstream_branch` portion is historical diagnostic infrastructure. A formal runner must instead restore the exact original root and invoke the same frozen π0 downstream for every method.
