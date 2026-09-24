# P1 simplified runtime collector CPU preflight

Status: `PASS_WRAPPER_FEASIBLE_PENDING_FORCE_CONTRACT`

## Fixed runtime identity

- `libero_10/task5`: `black_book_1 -> desk_caddy_1`
- instruction: `pick up the book and place it in the back compartment of the caddy`
- old P4-B data: not imported or reused

## Feasibility

- Existing OpenPI/B5 client remains the runtime source.
- The existing tactile 13D action seam preserves pose/gripper dimensions and overrides only force slots 7:13.
- Official full-task success remains the existing `success_term.func` result; no evaluator code is replaced.
- `x` is captured from the actual policy input: state, force history, marker motion, and optional image NPZ.
- `z_raw` is captured from runtime physical observables; no hidden friction or guessed learned latent is inserted into `x`.
- A post-reset/post-friction/pre-first-inference state snapshot and digest is saved for matched-root auditing.

## Explicitly not frozen by this preflight

Force range, Fmax, and Utility are null by design. They must be supplied and hashed before outcome collection; no historical outcome is used to fill them.

## Minimum engineering work before launch

1. Fill the placeholders in `P1_SIMPLIFIED_RUNTIME_LAUNCH_TEMPLATE.sh` from an explicit pre-outcome plan.
2. Start the already-locked candidate server only under the existing resource gate.
3. Run one branch per force with the same root seed, then compare preprobe state digests before accepting siblings.
4. Reject any branch missing `branch_result.json`, `P1_BRANCH_PROVENANCE.json`, x/z logs, or locked hashes.

## Artifacts

- `/home/exouser/Tabero/E3_HDF5_PREFLIGHT_CPU/P1_SIMPLIFIED_RUNTIME_COLLECTOR_PREFLIGHT.json`
- `/home/exouser/Tabero/E3_HDF5_PREFLIGHT_CPU/P1_SIMPLIFIED_RUNTIME_LAUNCH_TEMPLATE.sh`
- `/home/exouser/Tabero/E3_HDF5_PREFLIGHT_CPU/p1_transformed_source_compile_only.py` (source only; not executed)
