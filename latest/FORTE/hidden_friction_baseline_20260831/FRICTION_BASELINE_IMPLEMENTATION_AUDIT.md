# Hidden-Friction Baseline — Authoritative Implementation Audit

Status: **PHASE-0 BLOCKED; NO BASELINE ROLLOUT WAS STARTED**  
Audit date: 2026-08-31 UTC  
Formal method name: **ActiveForcing**

## Audit conclusion

The current workspace cannot yet produce the requested strictly matched paper table without changing or adding evaluation plumbing. Existing frontier branches are direct full-task outcomes, but their downstream arm motion is a scripted Cartesian pick/transport/place controller, not the frozen π0 downstream policy required by this table. Reusing those rows would violate the central matching condition. The audit therefore freezes task 0, six untouched TEST roots, μ={0.2, 0.5, 1.0}, and the 3.0–5.0 N grid, but stops before scientific execution.

Three further semantic issues must remain explicit. First, the only authoritative runnable “π0 default” in this Tabero environment is the same force-aware Tabero checkpoint and neutral prompt used by `Tabero-Neutral`; the two requested rows are not distinct implementations as currently specified. Second, the frozen simulator reactive baseline identifies itself as **FORTE-inspired, not official FORTE**, and uses privileged simulator GT slip. Third, the current task0 Direct gate failed held-out safety/reliability (`under_force_rate=0.5`, frontier MAE 0.4 N); it is frozen and may be evaluated, but must not be described as already validated.

## π0 checkpoint and server

- Checkpoint: `/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999`
- Policy config: `pi0_lora_tacfield_tabero`
- Base architecture: Pi0Config (π0, not π0.5)
- Authoritative server wrapper: `/home/exouser/Tabero/analysis/results/b5_tabero_neutral_20260822_040652/scripts/b5_serve_policy_with_explicit_norm_stats.py`
- Authoritative client: `/home/exouser/Tabero/analysis/results/b5_tabero_neutral_20260822_040652/scripts/b5_tabero_neutral_client.py`
- Action dimension: 13, slots `[x,y,z,rx,ry,rz,gripper,fLx,fLy,fLz,fRx,fRy,fRz]`
- Frozen downstream runner exists for normal E2E B5 and for selected P6/P7 tasks, but no authoritative runner presently starts all requested baselines from one shared task0 pre-probe snapshot.

### π0 default grip-force semantics

The raw checkpoint predicts gripper position plus left/right 3-D force slots. In the B5 official client these predicted 13-D actions are executed directly. There is no separate locally selected base-π0 checkpoint/adapter with a different “default gripper force” for the Tabero Hybrid-Tactile environment. Consequently, `π0-Default` with a neutral instruction is implementation-identical to `Tabero-Neutral` unless a distinct already-authoritative checkpoint is supplied; creating one here would violate this round’s no-new-model rule.

## Tabero implementation and force-language semantics

- Authoritative model is the same checkpoint above, unmodified at step 49999.
- Neutral prompt for task0: `pick up the alphabet soup and place it in the basket`.
- Dataset converter explicitly maps strong demonstrations to `firmly`/`tightly` and soft demonstrations to `gently`/`softly`.
- Inference `_rewrite_instruction` deterministically adds the selected adverb as a prefix or suffix.
- `Tabero-Neutral`: empty `prompt_adverb` and `prompt_adverbs`; no friction or force word.
- `Tabero-Oracle-Force-Language`: semantically supported, but must be marked **PRIVILEGED / ORACLE LANGUAGE**. The oracle language mapping itself must be frozen from empirical frontiers before its matched run; no per-result wording changes are allowed.

## FORTE implementation

- Official hardware FORTE runtime in this repository estimates force and tactile slip; it does not provide a frozen task0 simulator full-task controller.
- The available simulator baseline is `/home/exouser/Tabero/analysis/results/b4_forte_5task_baseline_20260821_154130/scripts/b4_eval.py` and self-identifies as `FORTE-INSPIRED REACTIVE`, not an official reproduction.
- Initial force: **3.0 N**.
- Force ladder / correction magnitude: **3→4→5→6→8 N**, increments **+1,+1,+1,+2 N**.
- Maximum force: **8.0 N**.
- Slip source: privileged simulator GT gross instability.
- Frozen slip conditions: relative z loss > **0.008 m**, relative z velocity < **−0.05 m/s for 2 steps**, relative xy loss > **0.015 m**, or contact/drop conditions in the frozen code.
- This surrogate may appear only with the qualifier `FORTE-inspired; privileged GT slip`. Calling it official FORTE would be false.

## Force controller and Fixed-Max

- Current calibrated Tabero force servo adjusts gripper opening to track a requested continuous squeeze target.
- Task0 frozen candidate-force range: **3.0–5.0 N**.
- Task0 grid: **0.25 N** increments, 3.0–5.0 N inclusive.
- Fixed-Max / Fixed-Robust: **5.0 N**, frozen from the prior authoritative task0 robust target. It is the maximum of the task0 candidate range and achieved the prior high-force ceiling; it is not selected per friction.

## ActiveForcing probe estimator

- Probe: frozen P4-B common contact-frame shear, nominal displacement cap 2 mm, no retraining in this round.
- ActiveForcing receives only the frozen probe estimator output/belief; GT μ is analysis-only.
- Existing prospective captures use the P4-B scripted grasp/staging path. They are not valid substitutes for a snapshot reached by frozen π0, so none is imported into this TEST manifest.

## ActiveForcing-Direct checkpoint

- Task0 preregistered frozen Direct selection artifact: `VISUAL_INTERCEPT_RESIDUAL`, three-seed ensemble under `/home/exouser/FORTE/task0_visual_context_early_20260831_025000/`.
- Selection record: `/home/exouser/FORTE/task0_gpu_sidecar_20260831_050050/TASK0_GT_GATE.json`.
- Held-out gate status: **FAIL**; Probe stage was **NOT_REACHED** in that gate.
- This is still a direct feasibility predictor, not Joint. No weights, calibration, threshold, or model selection may change during this table.
- Decision target for this table: minimum force with calibrated empirical reliability probability ≥ **ρ=0.8** over the frozen 3.0–5.0 N grid.

## ActiveForcing-NoProbe prior

- Frozen physics prior: **[0.30, 0.56, 0.92]**.
- Aggregation: mean of frozen per-seed TRAIN-isotonic-calibrated feasibility probabilities across the three prior μ values.
- Same context, candidate semantics, backend, and decision rule as Direct; no probe trace, μ-hat, GT μ, post-probe displacement, or outcome input.

## GT-Physics Oracle information

- Receives exact simulator hidden friction μ only.
- Uses the same frozen Direct feasibility ensemble, candidate grid, calibration, and ρ=0.8 decision rule as ActiveForcing-Direct.
- It does not receive the empirical episode outcome or F* label as an input.
- It is **PRIVILEGED** and estimates the force-selection ceiling, not deployability.

## Full-task success evaluator

Required label: direct complete downstream outcome from the pre-probe decision point through transport and placement. A valid success must include stable lift/retention, transport, placement/basket contact, and no drop. Lift-only or immediate no-slip is not success. Pure downstream semantic/trajectory failures after stable grasp must be labeled `VLA_DOWNSTREAM_FAILURE_NOT_GRIP_RELATED`.

## Available snapshots and empirical frontiers

- Prospective TRAIN snapshots exist for roots 5100–5105, but those roots trained the current task0 feasibility models and are TEST-ineligible.
- Roots 5106–5107 participated in DEV/model selection and are also TEST-ineligible.
- Untouched pre-outcome task0 TEST roots 5174–5179 are frozen in the new manifest, but no qualifying frozen-π0-reached canonical pre-probe snapshot has been collected for them.
- Existing empirical frontiers use `/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py::downstream_branch`, whose arm phases are scripted `hold/lift/transit/place/release/settle`. They cannot populate the requested frozen-π0 downstream frontier.

## Joint eligibility

`ActiveForcing-Joint` is excluded from the main table. Current held-out mechanism evidence does not establish independent reliability/value, and task0 Joint is worse than the selected non-Joint direct alternative on the frozen safety-first ordering. If later reported, it belongs only in a secondary/ablation table and must retain the failed-held-out caveat.

## Blocking conditions before Phase 3

1. Collect one frozen-π0-reached canonical pre-probe snapshot for each of roots 5174–5179, with state/RGB hashes and restore parity.
2. Add or identify an authoritative task0 post-snapshot runner that executes the unchanged frozen π0 downstream policy while allowing only force-selection semantics to differ.
3. Resolve the non-independence of `π0-Default` and `Tabero-Neutral` without creating a new model; otherwise report them as the same implementation, not two independent baselines.
4. Decide whether the explicitly non-official, privileged-GT-slip FORTE-inspired surrogate is acceptable under the requested row name. No official claim is permitted.

Until these are resolved, running the 810 frontier episodes or method episodes would create a scientifically mislabeled table. No scientific failure was retried because no baseline rollout was started.
