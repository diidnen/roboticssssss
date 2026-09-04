# Recovered Probe Non-TEST Reproduction

## Result

`PASS` for the recovered P4-B sensing and frozen friction-estimator path.

The already-running authoritative TRAIN collector produced a fresh task0 trajectory at root 5126 with true μ=0.9382535774. The file has 215 data rows and passed through the exact historical runtime feature builder and normalization to a 215×46 tensor. The exact `FRICTION_GRU.pt` checkpoint loaded and returned:

- μ̂ = 0.9045775533
- σμ = 0.0531502068

## Exact components

- Probe producer: `/home/exouser/FORTE/prospective_visual_context_collect.py --worker`
- Probe path: `/home/exouser/FORTE/task0_context_sample_complexity_20260831/collection_train_new/task0/P5S0C_PROBE_TELEMETRY/pv_train_t0_r26_s5126_high_mu0.938254_probe_timesteps.csv`
- Feature builder: `/home/exouser/Tabero/analysis/p5s0d_fresh_e2e_q2f.py::OnlineInference.sequence_array`
- Inference: `/home/exouser/Tabero/analysis/active_friction_imagination_e2e.py::estimator_inference`
- Normalization: `/home/exouser/Tabero/analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542/P5S0C_NORMALIZATION.json`
- Checkpoint: `/home/exouser/Tabero/analysis/results/active_friction_imagination_20260828_211106/FRICTION_GRU.pt`
- Checkpoint SHA-256: `a6c9d59bfa11c2481b7dca5fec1e41e0b625bbd463ed3fad9e3e4a1cb3eeb6af`

## Scope limitation

This was a read-only offline inference over a fresh non-TEST physical trajectory. No retraining, tuning, feature change, normalization change, force-map change, or TEST access occurred. A fresh force decision and downstream branch were not run because the GPU was already occupied by the authoritative collector and π0 feature server; historical force-decision evidence is reported separately and is not presented as fresh reproduction.
