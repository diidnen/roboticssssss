# New Task-Form Formal Protocol

Frozen: 2026-09-13 UTC  
Status: **PROSPECTIVELY FROZEN; execution waits for the currently running Stage-I GPU job to release the device.**  
Task: `dump_bin_bigbin`  
Scientific claim: using the same ActiveForcing formulation and experimental procedure, ActiveForcing remains useful on a qualitatively different task form across held-out VLA paths and friction settings within one physical root. Zero-shot task-form transfer and unseen-root generalization are not claimed.

## Invariants

- Frozen pi0 checkpoint: current RoboTwin checkpoint 30000 and its recorded source/checkpoint hashes.
- ActiveForcing architecture, 58D belief, phase-free 8x64 feasibility architecture, posterior integration, controller, success criterion, and force measurement remain unchanged.
- Force support remains [.5, 8] N with the existing eight feasibility branches [.5, 1, 2, 3, 4, 5, 6, 8] N and 0.05 N deployment search.
- Utility remains the currently frozen maxF8 definition `p*(8-F)/8 - (1-p)`.
- No outcome-dependent retries, resampling, root replacement, force changes, utility tuning, or method changes.
- Valid failures are retained. Crashes/infrastructure faults are UNKNOWN and rerun only under a recorded technical retry rule.

## Retained single-root feasibility training

Retain the completed root-200002 corpus and frozen `models_v4` ensemble without modification:

| Item | Frozen value |
|---|---|
| Physical training coverage | root 200002 only |
| Downstream feasibility labels | 192 (128 original + 64 maxF8 additions) |
| Deduplicated belief queries | 35 |
| Force branches in each full group | .5, 1, 2, 3, 4, 5, 6, 8 N |
| Existing policy-seed scheme | 30200002 and 40200002, as recorded |
| Model | already selected and frozen `models_v4` three-member ensemble |

The recorded checkpoint locks state that test labels were neither used nor accessed before model lock, and reload/runtime qualification passed. Do not add feasibility roots, relabel data, retrain, reopen checkpoint selection, or use formal-main outcomes to alter the model.

This intentionally does not reproduce the original feasibility corpus's 4/1/1 physical-root split. All training and formal evaluation use physical root 200002. Generalization is assessed only across prospectively held-out VLA paths and friction settings within that root; unseen-root generalization is not claimed.

## Nominal Frozen VLA definition and smoke

The native pi0 output is a 14-dimensional joint-position action for the two-arm RoboTwin interface. Indices 6 and 13 are normalized gripper-position commands; the robot interface maps them through its configured gripper scale to position/closure. They are not Newton-valued force setpoints.

Nominal Frozen VLA therefore:

- uses the frozen pi0 arm and gripper outputs exactly as returned by the normal action path;
- installs no ActiveForcing arbitration, fixed-N override, or force servo;
- starts from the same established-grasp, post-query saved-state handoff as AF and Fixed-8;
- ignores the probe posterior for action selection;
- uses the same full-task predicate and logs bilateral measured squeeze from the existing contact sensors;
- reports commanded force as `N/A`.

Before formal evaluation, run one development smoke on root 200002, friction .575, policy seed 40200002. Admission requires: online policy provenance, no gripper/force override, finite native actions, successful state restoration, valid label computation, and nonempty measured-squeeze telemetry. Task success is not an admission requirement. A technical failure stops the main launch; a valid task failure does not.

## Formal main comparison

The formal population uses root 200002 only. Eight prospectively selected, outcome-blind VLA path seeds are disjoint from training seeds 30200002/40200002, burned diagnostic seeds 50200002/60200002, and all Stage-I paths. Each path receives all three locked frictions and exactly three methods. A targeted repository search before freezing found no prior use of seeds 80200002-80200009.

| Dimension | Frozen value |
|---|---|
| Physical root | 200002 (one root only) |
| Held-out VLA path seeds | 80200002-80200009 (8) |
| Frictions | .425, .575, .85 |
| Contexts | 1 root x 8 paths x 3 frictions = 24 |
| Methods | Nominal Frozen VLA, Fixed-Strong 8 N, ActiveForcing |
| Main rollouts | 24 x 3 = 72 |
| Optional Fixed-Low | Not run in the minimum protocol |

For every `(root, path seed, friction)` context, create one qualified established-grasp/P4 query boundary, save its state, and verify identical physical/post-query metadata across method siblings. Reset pi0 RNG to that context's locked path seed for each sibling. Require identical first VLA action chunk across all three methods. Thereafter every method uses fresh online replanning from its own live observations. Method order is a deterministic per-context permutation frozen before physics.

AF and Fixed-8 use the current force-controller path. Nominal uses the native gripper-position path. Consequently arm-policy and initial physical context are paired, while gripper control is intentionally different. Nominal's commanded force remains N/A; measured contact-conditioned squeeze remains reportable.

## Counts and launch gates

| Stage | Existing reusable | New required |
|---|---:|---:|
| Feasibility labels | 192 | 0 |
| Nominal development smoke | 0 | 1 |
| Formal main rollouts | 0 | 72 |
| Total | 192 completed feasibility rollouts | **73 new physical rollouts** |

Launch order:

1. Wait for the existing Stage-I study to finish and confirm the GPU is no longer occupied by it.
2. Freeze source/config/checkpoint hashes and confirm exact non-exposure of main path seeds 80200002-80200009 against training, diagnostics, and Stage-I.
3. Run the one-context Nominal smoke.
4. Run the 72 formal main branches on root 200002 across the eight locked paths and three frictions without changing the frozen model or protocol from observed outcomes.
5. Produce `NEW_TASK_FORM_FINAL_REPORT.md`; keep Stage-I motion-context evidence in a separate section and denominator, and disclose that the test establishes only same-root held-out-path performance.

The formal main comparison reports full-task success, relevant intermediate success, contact loss/drop, measured squeeze, AF mean commanded force, Fixed-8 commanded force, Nominal commanded force N/A, and the paired AF-vs-Fixed-8 matrix (both success, AF only, Fixed only, both fail). Equal aggregate counts are not described as equivalent reliability.
