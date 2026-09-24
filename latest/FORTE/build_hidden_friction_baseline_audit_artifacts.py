#!/usr/bin/env python3
"""Build the Phase-0/1/2 hidden-friction baseline audit package.

This script deliberately does not synthesize rollout results.  It freezes the
eligible task, TEST roots, friction values, metric schemas, and the exact
Phase-0 blockers found in the authoritative implementation audit.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "hidden_friction_baseline_20260831"
OUT.mkdir(parents=True, exist_ok=True)

TASK = 0
OBJECT = "alphabet_soup_1"
INSTRUCTION = "pick up the alphabet soup and place it in the basket"
ROOT_SEEDS = list(range(5174, 5180))
FRICTIONS = [("LOW", 0.2), ("MID", 0.5), ("HIGH", 1.0)]
FORCES = [3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
METHODS = [
    "pi0-Default",
    "Tabero-Neutral",
    "Tabero-Oracle-Force-Language [PRIVILEGED / ORACLE LANGUAGE]",
    "FORTE-Reactive [FORTE-inspired; privileged GT slip surrogate]",
    "Fixed-Max",
    "ActiveForcing-NoProbe",
    "ActiveForcing-Direct",
    "GT-Physics Oracle [PRIVILEGED]",
]


def canonical_json(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_text(name: str, text: str) -> None:
    (OUT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(name: str, obj: object) -> None:
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_csv(name: str, fields: list[str], rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


selection = {
    "schema_version": "1.0",
    "created_utc": "2026-08-31",
    "status": "TASK_FROZEN; ROLLOUT_BLOCKED_AT_PHASE_0",
    "selected_task": {
        "task_id": TASK,
        "suite": "libero_object",
        "object": OBJECT,
        "instruction": INSTRUCTION,
        "table_scope": "post-grasp/pre-probe through direct full-task completion",
    },
    "selection_rule_frozen_before_baseline_execution": [
        "Use an existing task with direct full-task labels and transport/placement.",
        "Require prior frozen-pi0 pick/pre-probe reliability and a high-force completion ceiling.",
        "Require a friction-dependent force frontier.",
        "Do not use ActiveForcing outcomes to select the task.",
    ],
    "preexisting_selection_evidence": {
        "source": "/home/exouser/Tabero/analysis/results/b5_tabero_neutral_20260822_040652/TABERO_NEUTRAL_MAIN_RESULTS.csv",
        "task0": {
            "pick_sr_low_mid_high": [1.0, 1.0, 1.0],
            "full_sr_low_mid_high": [1.0, 1.0, 1.0],
            "canonical_fstar_N_low_mid_high": [5.0, 4.0, 3.0],
            "fixed_robust_sr_low_mid_high": [1.0, 1.0, 1.0],
        },
        "task5_rejected": {
            "reason": "Fails the preregistered stable pi0 reach/pre-probe criterion.",
            "pick_sr_low_mid_high": [0.5, 0.7, 0.7],
            "full_sr_low_mid_high": [0.2, 0.5, 0.1],
        },
    },
    "joint_main_table_eligible": False,
    "joint_reason": "Current held-out mechanism evidence does not establish reliable independent value; Joint is excluded from the selected main controller.",
    "leakage_rule": "Roots 5100-5105 are TRAIN and roots 5106-5107 are DEV/model-selection; neither set is eligible for this TEST table.",
    "selected_test_root_seeds": ROOT_SEEDS,
}
write_json("FRICTION_BASELINE_TASK_SELECTION.json", selection)


case_payload = {
    "schema_version": "1.0",
    "created_utc": "2026-08-31",
    "status": "FROZEN_BEFORE_BASELINE_EXECUTION; SNAPSHOTS_NOT_YET_ELIGIBLE",
    "task_id": TASK,
    "object": OBJECT,
    "root_count": len(ROOT_SEEDS),
    "conditions_per_root": 3,
    "frictions": {"LOW": 0.2, "MID": 0.5, "HIGH": 1.0},
    "friction_source": "Existing authoritative B4/B5 legal task0 range and canonical low/mid/high values; fixed without method outcomes.",
    "force_grid_N": FORCES,
    "frontier_repeat_count": 5,
    "rho": 0.8,
    "frontier_rule": "minimum grid force with >=4/5 direct full-task successes",
    "snapshot_rule": (
        "For each root, frozen pi0 must reach one canonical pre-probe state once. "
        "That exact serialized state and exact RGB pair must be restored for all methods and all three conditions; "
        "only object/gripper material friction may be patched after restore and before the first method action."
    ),
    "cases": [],
}
for index, seed in enumerate(ROOT_SEEDS):
    root_id = f"hf_t0_root{index:02d}_s{seed}"
    for band, mu in FRICTIONS:
        case_payload["cases"].append(
            {
                "case_id": f"{root_id}_{band.lower()}_mu{mu:g}",
                "root_id": root_id,
                "root_index": index,
                "root_seed": seed,
                "source_preoutcome_test_root": f"pv_test_t0_root{seed - 5100:02d}_s{seed}",
                "friction_band": band,
                "mu": mu,
                "canonical_preprobe_snapshot_path": None,
                "canonical_preprobe_snapshot_sha256": None,
                "camera0_rgb_sha256": None,
                "camera1_rgb_sha256": None,
                "same_visual_observation_across_friction": "REQUIRED_NOT_YET_VERIFIED",
                "eligibility": "BLOCKED_MISSING_FROZEN_PI0_PREPROBE_SNAPSHOT",
                "scientific_retry": False,
            }
        )
case_manifest_hash = sha_bytes(canonical_json(case_payload))
case_manifest = dict(case_payload)
case_manifest["content_sha256_without_hash_field"] = case_manifest_hash
write_json("FRICTION_CASE_MANIFEST.json", case_manifest)


audit = f"""# Hidden-Friction Baseline — Authoritative Implementation Audit

Status: **PHASE-0 BLOCKED; NO BASELINE ROLLOUT WAS STARTED**  
Audit date: 2026-08-31 UTC  
Formal method name: **ActiveForcing**

## Audit conclusion

The current workspace cannot yet produce the requested strictly matched paper table without changing or adding evaluation plumbing. Existing frontier branches are direct full-task outcomes, but their downstream arm motion is a scripted Cartesian pick/transport/place controller, not the frozen π0 downstream policy required by this table. Reusing those rows would violate the central matching condition. The audit therefore freezes task 0, six untouched TEST roots, μ={{0.2, 0.5, 1.0}}, and the 3.0–5.0 N grid, but stops before scientific execution.

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
- Neutral prompt for task0: `{INSTRUCTION}`.
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
"""
write_text("FRICTION_BASELINE_IMPLEMENTATION_AUDIT.md", audit)


frontier_rows = []
for case in case_payload["cases"]:
    for force in FORCES:
        frontier_rows.append(
            {
                "root_id": case["root_id"],
                "friction_band": case["friction_band"],
                "friction": case["mu"],
                "force": force,
                "success_count": "",
                "repeat_count": 0,
                "empirical_success_probability": "",
                "Fstar_0.8": "",
                "full_task_evaluator": "frozen_pi0_downstream_direct_full_task",
                "data_status": "NOT_RUN_PHASE0_BLOCKER",
            }
        )
write_csv(
    "FRICTION_EMPIRICAL_FRONTIERS.csv",
    ["root_id", "friction_band", "friction", "force", "success_count", "repeat_count", "empirical_success_probability", "Fstar_0.8", "full_task_evaluator", "data_status"],
    frontier_rows,
)


episode_fields = [
    "episode_id", "method", "privilege", "root_id", "root_seed", "friction_band", "friction", "repeat_index",
    "snapshot_sha256", "camera0_rgb_sha256", "camera1_rgb_sha256", "pi0_checkpoint", "pi0_policy_seed",
    "selected_force_N", "initial_force_N", "mean_force_N", "peak_force_N", "force_time_integral_Ns", "normalized_force_cost",
    "Fstar_0.8_N", "under_force", "excess_force_N", "full_task_success", "grip_related_failure", "failure_code",
    "slip_onset_s", "correction_count", "force_before_correction_N", "force_after_correction_N", "recovery_success", "time_to_correction_s",
    "finite_decision", "scientific_retry", "data_status",
]
episode_rows = []
for method in METHODS:
    privilege = "PRIVILEGED" if "PRIVILEGED" in method else "DEPLOYABLE_OR_CONTROL"
    for case in case_payload["cases"]:
        for repeat in range(1, 6):
            episode_rows.append(
                {
                    "episode_id": f"PLAN_{method.split()[0]}_{case['case_id']}_R{repeat}",
                    "method": method,
                    "privilege": privilege,
                    "root_id": case["root_id"],
                    "root_seed": case["root_seed"],
                    "friction_band": case["friction_band"],
                    "friction": case["mu"],
                    "repeat_index": repeat,
                    "scientific_retry": 0,
                    "data_status": "PLANNED_NOT_RUN_PHASE0_BLOCKER",
                }
            )
write_csv("FRICTION_BASELINE_PER_EPISODE.csv", episode_fields, episode_rows)


main_fields = ["Method", "Low μ SR↑", "Mid μ SR↑", "High μ SR↑", "Avg SR↑", "Mean Force↓", "Peak Force↓", "Excess Force↓", "Grip Failure↓", "Under-force↓", "Force Cost↓", "N", "Data Status"]
write_csv(
    "FRICTION_BASELINE_MAIN_TABLE.csv",
    main_fields,
    [{"Method": method, "N": 0, "Data Status": "NOT_ESTIMABLE_PHASE0_BLOCKER"} for method in METHODS],
)


frontier_methods = [
    "Tabero-Neutral",
    "ActiveForcing-NoProbe",
    "ActiveForcing-Direct",
    "ActiveForcing-Joint [SECONDARY ONLY; INELIGIBLE CURRENT EVIDENCE]",
    "GT-Physics Oracle [PRIVILEGED]",
]
write_csv(
    "FRICTION_BASELINE_FRONTIER_TABLE.csv",
    ["Method", "Frontier MAE↓", "Under-force↓", "Excess Force↓", "Finite Decision↑", "N valid frontiers", "Data Status"],
    [{"Method": method, "N valid frontiers": 0, "Data Status": "NOT_ESTIMABLE_PHASE0_BLOCKER"} for method in frontier_methods],
)


write_csv(
    "FRICTION_BASELINE_FAILURE_TAXONOMY.csv",
    ["episode_id", "method", "root_id", "friction", "failure_code", "grip_related", "evidence", "adjudication_status", "data_status"],
    [],
)


run_status = {
    "status": "BLOCKED_AT_PHASE_0_NO_SCIENTIFIC_ROLLOUTS",
    "baseline_rollouts_started": False,
    "frontier_rollouts_completed": 0,
    "method_rollouts_completed": 0,
    "planned_frontier_rollouts": len(ROOT_SEEDS) * len(FRICTIONS) * len(FORCES) * 5,
    "planned_method_rollouts": len(ROOT_SEEDS) * len(FRICTIONS) * len(METHODS) * 5,
    "task": TASK,
    "roots": ROOT_SEEDS,
    "case_manifest_content_sha256": case_manifest_hash,
    "blockers": [
        "No qualifying frozen-pi0-reached pre-probe snapshots for untouched TEST roots.",
        "No authoritative matched task0 frozen-pi0 downstream runner spanning all requested force methods.",
        "pi0-Default and Tabero-Neutral resolve to the same checkpoint/prompt/execution path.",
        "Available simulator FORTE row is explicitly FORTE-inspired and uses privileged GT slip.",
    ],
}
write_json("RUN_STATUS.json", run_status)


report = f"""# Hidden-Friction Baseline Report

**现在不能诚实回答这七个排名问题：严格 matched rollout 数为 0，因此最稳定方法、Fixed-Max 是否达到类似 success、ActiveForcing 节省的 excess force、相对 FORTE 的 pre-slip 优势、NoProbe 掉点、Tabero-neutral 的自主适应能力，以及距 GT-Physics Oracle 的差距，全部为 `NOT ESTIMABLE`。** 这不是负结果，而是 Phase-0 implementation audit 的停止结论：现有 full-task frontier 使用脚本化 downstream，不是本表要求的 frozen π0 downstream；同时 `π0-Default` 与 `Tabero-Neutral` 当前映射到同一 checkpoint/neutral execution path，所谓 FORTE runner 也由其自身规格明确标为 FORTE-inspired + privileged GT slip。把旧数据填进新表会直接破坏论文的核心 matched claim。

## Technical summary

- Task 已在任何本表结果产生前冻结为 task0 / alphabet soup。选择只使用既有 B5 证据：π0 neutral 在 low/mid/high 的 pick/full-task 均为 1.0，prior frontier 为 5/4/3 N；task5 因 pick rate 0.5/0.7/0.7 被预先排除。
- 六个 untouched TEST root seeds 已冻结为 5174–5179。TRAIN 5100–5105 与 DEV 5106–5107 均禁止进入本表。
- friction 已冻结为 μ={{0.2, 0.5, 1.0}}；task0 force grid 已冻结为 3.0–5.0 N、0.25 N 间隔、每 cell 5 repeats、ρ=0.8。
- Frontier 计划 810 个 rollouts；8 个方法、18 cases、5 paired repeats 计划 720 个 rollouts。两者均未启动。
- Joint 不进入 main table；当前 held-out mechanism evidence 不支持其作为 selected controller。

## Why execution stopped before Phase 3

The requested causal contrast requires one exact pre-probe state and one exact visual observation per root, then friction-only mutation. Existing prospective task0 snapshots either belong to TRAIN/DEV or were reached by scripted P4-B staging rather than frozen π0. Untouched TEST roots have been preregistered but do not yet have eligible snapshots.

The only existing branch generator with dense continuous force labels calls a scripted Cartesian `downstream_branch`. The current frozen π0 runner exists in B5/P6/P7, but there is no authoritative task0 adapter that starts from the new canonical snapshot and supports every requested force method. This is not a normal scientific failure and must not be retried or silently substituted.

## Method readiness

| Method | Current implementation status | Can enter matched table now? |
|---|---|---|
| π0-Default | Raw 13-D action execution from the force-aware Tabero π0 checkpoint under neutral prompt | No; identical to Tabero-Neutral as currently specified |
| Tabero-Neutral | Authoritative B5 checkpoint/client, no force adverb | Semantics ready; matched snapshot runner missing |
| Tabero-Oracle-Force-Language | `gently/softly/firmly/tightly` are supported by converter and inference rewrite | Semantics ready; privileged mapping and matched runner pending |
| FORTE-Reactive | Frozen B4 simulator surrogate, start 3 N, 3→4→5→6→8 N, privileged GT slip | Only if labeled FORTE-inspired, not official FORTE |
| Fixed-Max | Frozen task0 robust target 5 N | Selector ready; matched runner missing |
| ActiveForcing-NoProbe | Prior [0.30,0.56,0.92], same frozen Direct backend | Selector ready; matched runner missing |
| ActiveForcing-Direct | Frozen task0 Direct ensemble; held-out GT gate failed | Runnable as a frozen evaluated method, not a prevalidated controller |
| GT-Physics Oracle | Exact μ into the same frozen Direct backend | Selector ready; privileged; matched runner missing |

## Data-quality and validation status

Overall readiness: **NOT READY FOR CLAIMS**.

- Population coverage: 0/810 frontier episodes and 0/720 method episodes.
- Matching validation: manifest rule frozen, but snapshot/RGB hashes are missing.
- Metric recomputation: not applicable; no outcome data exist.
- Failure taxonomy: schema created, no episodes to adjudicate.
- Existing scripted frontier rows are deliberately excluded rather than mixed with frozen π0 outcomes.
- Main and frontier CSVs contain explicit `NOT_ESTIMABLE_PHASE0_BLOCKER` status, not zero-valued metrics.

## Required next implementation step

Implement or identify one authoritative post-snapshot task0 runner that:

1. lets frozen π0 reach and serialize a canonical pre-probe state for each root 5174–5179;
2. restores exactly that state and identical RGB for μ=0.2/0.5/1.0;
3. changes only force-selection/control semantics while keeping the frozen π0 downstream policy and evaluator fixed;
4. records direct full-task outcomes and the full force/slip/recovery telemetry schema already frozen here;
5. resolves whether `π0-Default` and `Tabero-Neutral` are intentionally the same row or supplies an already-authoritative distinct π0 implementation without training or modification.

Only after those checks pass should the frozen 810-cell frontier run begin. The required paper claims must remain blank until then.

## Further questions

- Is an explicitly labeled `FORTE-inspired (privileged GT-slip)` surrogate acceptable, or is an official tactile FORTE simulator adapter required?
- Should the scientifically redundant π0-Default/Tabero-Neutral rows be merged, or is there an existing distinct frozen π0 checkpoint/adapter not discoverable in the current workspace?
"""
write_text("FRICTION_BASELINE_REPORT.md", report)


html_report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hidden-Friction Baseline Audit</title>
<style>
body{{font-family:Inter,system-ui,sans-serif;max-width:1050px;margin:40px auto;padding:0 24px;color:#17202a;line-height:1.55}}
h1,h2{{color:#102a43}} .blocked{{background:#fff3cd;border-left:5px solid #d39e00;padding:16px 18px}}
code{{background:#f4f6f8;padding:2px 5px;border-radius:4px}} table{{border-collapse:collapse;width:100%;margin:16px 0}}
th,td{{border:1px solid #d9e2ec;padding:8px;text-align:left}} th{{background:#eef4f8}} .muted{{color:#52606d}}
</style></head><body>
<h1>Hidden-Friction Baseline — Phase-0 Audit</h1>
<div class="blocked"><strong>NOT ESTIMABLE:</strong> no strictly matched frozen-π0 downstream rollouts were started. Existing frontiers use scripted downstream motion and cannot support the requested paper claim.</div>
<h2>Frozen design</h2><ul><li>Task 0 / alphabet soup</li><li>TEST roots 5174–5179</li><li>μ = 0.2, 0.5, 1.0</li><li>Force grid 3.0–5.0 N by 0.25 N; 5 repeats; ρ=0.8</li></ul>
<h2>Blocking findings</h2><ol><li>No qualifying frozen-π0-reached canonical pre-probe snapshots exist for these untouched roots.</li><li>No authoritative task0 runner applies all methods from one shared snapshot under the unchanged frozen π0 downstream policy.</li><li><code>π0-Default</code> and <code>Tabero-Neutral</code> currently resolve to the same checkpoint and neutral execution path.</li><li>The available simulator FORTE row is explicitly FORTE-inspired and uses privileged GT slip.</li></ol>
<p class="muted">See FRICTION_BASELINE_IMPLEMENTATION_AUDIT.md and FRICTION_BASELINE_REPORT.md for source paths, semantics, and validation limits.</p>
</body></html>"""
write_text("FRICTION_BASELINE_REPORT.html", html_report)


# Hash all delivered artifacts except the checksum file itself.
artifact_names = [
    "FRICTION_BASELINE_IMPLEMENTATION_AUDIT.md",
    "FRICTION_BASELINE_TASK_SELECTION.json",
    "FRICTION_CASE_MANIFEST.json",
    "FRICTION_EMPIRICAL_FRONTIERS.csv",
    "FRICTION_BASELINE_PER_EPISODE.csv",
    "FRICTION_BASELINE_MAIN_TABLE.csv",
    "FRICTION_BASELINE_FRONTIER_TABLE.csv",
    "FRICTION_BASELINE_FAILURE_TAXONOMY.csv",
    "FRICTION_BASELINE_REPORT.md",
    "FRICTION_BASELINE_REPORT.html",
    "RUN_STATUS.json",
]
lines = [f"{sha_file(OUT / name)}  {name}" for name in artifact_names]
write_text("SHA256SUMS.txt", "\n".join(lines))

print(json.dumps({"out": str(OUT), "status": run_status["status"], "files": artifact_names + ["SHA256SUMS.txt"]}, indent=2))
