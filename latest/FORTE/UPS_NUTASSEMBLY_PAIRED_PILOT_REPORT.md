# Faithful UPS NutAssemblySquare paired pilot

Audit date: 2026-08-29 UTC  
Scope: **INTERVENTION only**  
Final classification: **BLOCKED_BY_MISSING_UPS_ARTIFACTS**

## Executive result

The 40-state paired pilot was **not run**. Faithful official UPS intervention execution is not available from the public artifacts. The missing pieces include the learned gate and residual source, identifiable trained gate/residual checkpoints and configs, and the runtime wrapper that defines their deployment semantics. Reconstructing those pieces from the paper would test a surrogate, which the protocol forbids.

Accordingly, this report contains no empirical UPS collateral result. `CONTINUE_ONLY` is **not estimable**; it must not be reported as zero.

## Phase 0 provenance and live availability check

| Item | Verified state on 2026-08-29 |
|---|---|
| Official repository | `https://github.com/CMU-IntentLab/uncertainty_aware_policy_steering.git` |
| Branch / commit | `main` / `f7bad4d7cd6d5e4350751a958ba9bb146a99ecec` |
| Local vs remote | Local `HEAD` equals fetched `origin/main`; worktree clean. |
| Releases / tags | GitHub release API returned an empty list; no git tags. |
| `diffusion4robotics` | Gitlink `c347ddfdd969a0393f4e4d4f9b918b44bb281ad7`; unauthenticated `git ls-remote` fails by requesting credentials. |
| RoboMimic | Public and checked out at `e10526b9a40c78b41f1e37e60041dc0ec0a5f60f`. |
| robosuite | Public and checked out at `a3f2467375e434aa3c74eeea52e5d592a0cdc4d6`. |
| Public root tree | No `scripts/` directory; `policy/` has only `sim_policy.py`, `wm_predictor.py`, `vlm_video_translator.py`, and `__init__.py`. |
| Diffusion checkpoint Hub repo | Revision `17ad9ee2efb432e71e3d0e1ac730f08a77e9cf57`; one Hydra/config bundle for `data4robotics.models.diffusion_unet.DiffusionUnetAgent` and step 5k–95k checkpoints. No separately identifiable classifier/residual run or checkpoint. |
| World-model checkpoint Hub repo | Revision `6e7ecaa1656dad88e05b0fe2dd8e3ff83dfe18db`; two world-model files. |
| Base-demo link | Not found as either a public HF model or dataset. |
| Residual world-model dataset | Revision `84326dc0ee5836736e875447005f708b9f10f04f`; only `.gitattributes`, and the Hub page labels the dataset empty. |
| Complete public author inventory | Two models (diffusion run and WM checkpoint) and three datasets; no additional public UPS gate or residual repository. |

The official README itself assigns base-policy, intervention-classifier, and residual-policy code to `diffusion4robotics`, and the calibration, correction processing, and overall pipeline to `scripts` ([`_official_ups/README.md:23`](/home/exouser/FORTE/_official_ups/README.md:23)). It invokes the absent `scripts/run_ups` with separate `--classifier_ckpt` and `--residual_ckpt` flags ([`_official_ups/README.md:178`](/home/exouser/FORTE/_official_ups/README.md:178)).

## Required-component audit

| Required scientific component | Public status | Exact blocker | Exact artifact needed |
|---|---|---|---|
| Official NutAssemblySquare environment | **Available as source** | None at Phase 0: pinned RoboMimic and robosuite contain the task. | Matching official environment kwargs/evaluator still needed below. |
| Official base policy | **Incomplete for execution** | Wrapper and base weights/config are public, but Hydra targets `data4robotics.*`, whose implementation is inside the inaccessible submodule. | Pinned `data4robotics` source or a complete, executable exported base policy matching the public weights. |
| Intervention gate/classifier | **Missing** | No public class/function, preprocessing config, normalization, checkpoint, threshold, or call site. | Exact NutAssembly classifier source, frozen checkpoint, normalization, config, threshold, and hash. |
| Residual policy | **Missing** | No public UPS residual source, preprocessing config, frozen checkpoint, action scaling, or clipping code. | Exact NutAssembly residual source, checkpoint, normalization/config, composition/clipping implementation, and hash. |
| Runtime combiner | **Missing** | README requires `scripts/run_ups`; root tree contains no `scripts` directory and no equivalent entry point. | The commit-matched `run_ups` driver and all imports/config files it requires. |
| High-level UPS integration | **Incomplete** | Public fragments do not include the conformal verifier or integrated ACT/ASK/LEARN loop. `WMPredictor` also imports missing `dreamer.prediction` and uses author-local normalization paths. | Complete calibrated verifier/integration source, prompts/configs, checkpoint paths, and deterministic transcript/replay semantics. |
| Official evaluation definition | **Missing** | No UPS evaluation driver establishes horizon, environment/controller kwargs, instruction-mode assignment, seeds, or final instruction-consistent predicate. | Commit-matched evaluation config/script and seed/task manifest. |

The generic `model_based_irl_torch` tree contains classes with words such as “classifier” and “residual,” but no public UPS driver binds them to the documented learned low-level intervention gate or residual policy. They are not evidence that the missing intervention mechanism is executable.

## Checkpoint finding: missing versus merely unnamed

The README says the diffusion Hub repository contains “diffusion and residual checkpoints,” but its only config bundle instantiates a base `DiffusionUnetAgent`; every listed checkpoint belongs to that single step-numbered run. No public artifact identifies which file should satisfy `--classifier_ckpt` or `--residual_ckpt`, and there is no classifier config at all. Therefore the intervention weights are **missing or scientifically unidentifiable**, which is equally blocking for faithful execution. Guessing that a base checkpoint is a residual checkpoint would be an unauthorized surrogate.

## Why the paper is insufficient to reproduce the official treatment

The paper/README provide useful triangulation: image/proprio/base-action inputs, human-control timestep labels, weighted BCE for the gate, delta-action MSE for the residual, and documented per-step gating while constructing combined candidates. They do not publish the trained functions or enough details to reproduce the exact experimental treatment:

- no classifier/residual parameter values or unambiguous checkpoint mapping;
- no learned classifier threshold;
- no exact preprocessing and correction-segment construction code;
- no executable action normalization, residual addition, scaling, or clipping order;
- no definitive live-execution gate cadence, state/history behavior, or control-return semantics;
- no integrated candidate-generation/verifier runtime; and
- no exact downstream horizon, instruction-mode evaluator, or seed schedule.

Training new models from the prose would produce a new implementation, not the official UPS intervention required by this validation experiment. It would also violate the no-training instruction.

## Mandated stop and unexecuted phases

| Phase | Status | Reason |
|---|---|---|
| 0 — faithful artifact availability | **FAILED / BLOCKED** | Required official scientific artifacts are absent, private, or unidentifiable. |
| 1 — reproduction/parity sanity | NOT RUN | There is no complete official UPS policy to activate or compare against base actions. |
| 2 — snapshot/restore validation | NOT RUN | Simulator feasibility was audited previously, but parity must include missing gate/residual/runtime state. |
| 3–5 — branch execution and outcomes | NOT RUN | Running Branch I would require inventing UPS semantics. |
| 6 — `CONTINUE_ONLY` forensics | NOT RUN | No paired trajectories exist. |
| 7 — metrics | NOT ESTIMABLE | No valid paired sample exists. |

No scientific policy source was modified, no threshold was tuned, no model was trained, and no MetaWorld/R20/V1/fixed-24-step substitute was used.

## Paired-pilot results

| Quantity | Result |
|---|---|
| Valid paired states | 0 (experiment not run) |
| `BOTH_SUCCESS` | NOT ESTIMABLE |
| `INTERVENE_ONLY` | NOT ESTIMABLE |
| `CONTINUE_ONLY` | NOT ESTIMABLE |
| `BOTH_FAILURE` | NOT ESTIMABLE |
| `SR_continue` | NOT ESTIMABLE |
| `SR_intervene` | NOT ESTIMABLE |
| `SR_oracle` | NOT ESTIMABLE |
| Collateral rate | NOT ESTIMABLE |
| Rescue rate | NOT ESTIMABLE |
| Net intervention gain | NOT ESTIMABLE |
| Paired success-rate difference | NOT ESTIMABLE |

The header-only [`UPS_NUTASSEMBLY_PAIRED_RESULTS.csv`](/home/exouser/FORTE/UPS_NUTASSEMBLY_PAIRED_RESULTS.csv) records that no manifest rows were generated; it is not a zero-event outcome table.

## Scientific decision

**BLOCKED_BY_MISSING_UPS_ARTIFACTS**

This result neither observes nor rules out UPS collateral risk. The prior MetaWorld collateral cases cannot fill the evidence gap because the fixed-24-step policy replacement was already classified as a **MAJOR_DEVIATION** from documented UPS.

## Exactly one next experiment

After a version-locked official artifact bundle supplies every blocker above, run the preregistered **40-valid-state paired Continue-versus-complete-UPS NutAssemblySquare pilot** in [`UPS_NUTASSEMBLY_PAIRED_PILOT_PROTOCOL.json`](/home/exouser/FORTE/UPS_NUTASSEMBLY_PAIRED_PILOT_PROTOCOL.json), unchanged and without training. Until that bundle exists, do not run a substitute experiment and do not make a method or novelty claim.
