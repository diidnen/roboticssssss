#!/usr/bin/env python3
"""Emit the formal hidden-friction Phase-A hard-stop artifacts.

This generator is intentionally non-experimental: it creates provenance,
schemas, and blocker records only.  It never starts Isaac, queries a policy,
or fabricates a success/failure observation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "hidden_friction_baseline_20260831"
SOURCE_PLAN = OUT / "FRICTION_BASELINE_PER_EPISODE.csv"
CASE_MANIFEST = OUT / "FRICTION_CASE_MANIFEST.json"
GIT_COMMIT = "7f88d0184c1617ed95e67502da96e60be07b3689"
CREATED = "2026-08-31T11:56:55Z"
STATUS = "NOT_ESTIMABLE_PHASE0_BLOCKER_ACTIVEFORCING_PROBE_HANDOFF_UNDEFINED"

TASK = 0
ROOTS = [5174, 5175, 5176, 5177, 5178, 5179]
MUS = [0.2, 0.5, 1.0]
FORCES = [3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
CHECKPOINT = "/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999"
PROMPT = "pick up the alphabet soup and place it in the basket"

SOURCE_HASHES = {
    "visual_pi0_server.py": "56851e5d3fdf034eaf8bcdd2350a66187c40a1a4ed86704de85fcdf020797976",
    "prospective_visual_context_collect.py": "567a34bdae3bca92df7009b39e70a13877652c4a96687dce39708436d3d5e861",
    "task0_visual_generalization.py": "176f0406f697259236eeed01d4eea5cc32e7cef639010416d98e5b082febf8fc",
    "active_friction_imagination.py": "c06e09b34a27b1c3a1111d60a3be012edf6e9d604008eeb2ef0858a65d532918",
    "active_friction_imagination_e2e.py": "b510af7aaadba91316d91cc219f8a0744824c8ad4babf38ad33a3086c97248ec",
    "p5s0c_model_adjudication.py": "ecf5a57c9bffb313fd3df94bef26d1579754836f26fa154538bff7acf6e7ea9f",
    "p7b_gnp_physical_belief_force_planning.py": "2ef853b3d3a4ad4d1642332841dc5deb3daed71a7ca2ae8cb8d588668ec28337",
    "p6g1r1_controller_grasp_vla_handoff.py": "ae21e6d3299aab4bb2b910f27d774f3685dbc8e7ef6d90abe24fadecf780b1d4",
    "b5_tabero_neutral_client.py": "3b38317efad4ee0b2d9e1bfc4379198bbe0d2eb146665e170070b3ad82ec0a18",
    "b5_serve_policy_with_explicit_norm_stats.py": "a3baac379926856f928b8d5aad5e3285af84c2377158ea1c4bd19440f2907d01",
    "FRICTION_GRU.pt": "a6c9d59bfa11c2481b7dca5fec1e41e0b625bbd463ed3fad9e3e4a1cb3eeb6af",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(name: str, text: str) -> None:
    (OUT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(name: str, obj: object) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(name: str, fields: list[str], rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalized_method(source: str) -> tuple[str, str, bool, str]:
    if source == "pi0-Default":
        return "π0-Neutral", "legacy_pi0_default_slot", False, ""
    if source == "Tabero-Neutral":
        return "π0-Neutral", "legacy_tabero_neutral_slot", False, ""
    if source.startswith("FORTE-Reactive"):
        return (
            "Privileged Simulator (FORTE-inspired, GT-slip)",
            "legacy_forte_reactive_slot",
            True,
            "ground-truth simulator slip state",
        )
    if source.startswith("Tabero-Oracle"):
        return "Tabero-Oracle-Language", "oracle_language_slot", True, "friction-relevant oracle language"
    if source.startswith("GT-Physics"):
        return "GT-Physics Oracle", "gt_physics_slot", True, "exact simulator friction mu"
    return source, source.lower().replace("-", "_").replace(" ", "_"), False, ""


def build_audit() -> None:
    text = f"""# Frozen π0 downstream implementation audit

Final Phase-A classification: **{STATUS}**  
Scientific rollouts started: **no**  
Audit completed: {CREATED}

## Answer

A real action-producing π0 server and two partial action-loop implementations were found, but no implementation satisfies the complete formal chain for ActiveForcing. The blocker is not that the server is feature-only: `visual_pi0_server.py` calls the frozen policy's `infer`, returns `actions`, and only then attaches visual diagnostics. The blocker is that the only runtime integration of the frozen friction estimator uses the full scripted P4-B approach/grasp/probe trace and a deterministic scripted downstream controller. Converting it to a probe-only branch from a π0-reached grasp would change the frozen probe/estimator input contract, which is a prohibited scientific method change.

## Frozen π0 identity

- Checkpoint: `{CHECKPOINT}`
- Policy config: `pi0_lora_tacfield_tabero` (Pi0Config, not π0.5)
- Neutral task0 prompt: `{PROMPT}`
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

- Frozen estimator: `/home/exouser/Tabero/analysis/results/active_friction_imagination_20260828_211106/FRICTION_GRU.pt`, SHA-256 `{SOURCE_HASHES['FRICTION_GRU.pt']}`.
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
{json.dumps(SOURCE_HASHES, indent=2, sort_keys=True)}
```
"""
    write_text("FROZEN_PI0_DOWNSTREAM_IMPLEMENTATION_AUDIT.md", text)


def build_snapshots() -> None:
    cases = json.loads(CASE_MANIFEST.read_text(encoding="utf-8"))["cases"]
    records = []
    for case in cases:
        records.append({
            "task": TASK,
            "root": int(case["root_seed"]),
            "root_id": case["root_id"],
            "mu": float(case["mu"]),
            "snapshot_path": None,
            "snapshot_sha256": None,
            "physics_state_sha256": None,
            "observation_sha256": None,
            "camera0_rgb_sha256": None,
            "camera1_rgb_sha256": None,
            "pi0_temporal_state_sha256": None,
            "pi0_rng_state_sha256": None,
            "controller_state_sha256": None,
            "canonical_reach_status": "NOT_ATTEMPTED_PHASE_A_HARD_STOP",
            "eligibility": "NOT_ESTIMABLE",
        })
    write_json("CANONICAL_SNAPSHOT_MANIFEST.json", {
        "status": STATUS,
        "planned": 18,
        "valid": 0,
        "failed": 0,
        "not_attempted": 18,
        "interpretation": "Zero valid snapshots means not run, not canonical reach failure.",
        "task": TASK,
        "roots": ROOTS,
        "mu_values": MUS,
        "records": records,
    })
    write_text("CANONICAL_SNAPSHOT_QA.md", f"""# Canonical snapshot QA

Status: **NOT RUN — {STATUS}**

- Planned snapshots: 18 root×μ cases.
- Generated: 0.
- Validated: 0.
- `CANONICAL_REACH_FAILURE`: 0. No upstream TEST rollout was attempted, so zero is not a failure count.
- Restore-equivalence cases executed: 0.
- Image/state/next-action parity checks executed: 0.

Phase D was not entered because Phase A found a prohibited scientific-method interface gap. Reporting snapshot QA as pass or fail would be misleading.
""")


def build_formal_manifest() -> None:
    with SOURCE_PLAN.open(newline="", encoding="utf-8") as fh:
        source_rows = list(csv.DictReader(fh))
    if len(source_rows) != 720:
        raise RuntimeError(f"frozen plan has {len(source_rows)} rows, expected 720")
    combos = Counter((int(r["root_seed"]), float(r["friction"])) for r in source_rows)
    expected = {(root, mu): 40 for root in ROOTS for mu in MUS}
    if combos != expected:
        raise RuntimeError("frozen 720-row plan does not have 40 rows per root×mu")

    entries = []
    method_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for index, row in enumerate(source_rows, 1):
        method, block, privileged, info = normalized_method(row["method"])
        method_counts[method] += 1
        source_counts[row["method"]] += 1
        entries.append({
            "rollout_id": f"formal_plan_{index:04d}",
            "source_episode_id": row["episode_id"],
            "source_method_label": row["method"],
            "method": method,
            "independent_replication_block": block,
            "task": TASK,
            "root": int(row["root_seed"]),
            "root_id": row["root_id"],
            "mu": float(row["friction"]),
            "repeat": int(row["repeat_index"]),
            "force_N": None,
            "snapshot_path": None,
            "snapshot_sha256": None,
            "checkpoint": CHECKPOINT,
            "checkpoint_hash": None,
            "prompt": PROMPT if method != "Tabero-Oracle-Language" else None,
            "pi0_runner_version": None,
            "git_commit": GIT_COMMIT,
            "environment_version": "IsaacLab 5.1 runtime; exact formal runner not instantiated",
            "seed": None,
            "privileged": privileged,
            "privileged_information": info,
            "execution_status": "PLANNED_NOT_RUN_PHASE_A_HARD_STOP",
        })
    write_json("FORMAL_ROLLOUT_MANIFEST.json", {
        "status": STATUS,
        "source_manifest": str(SOURCE_PLAN),
        "source_manifest_sha256": sha(SOURCE_PLAN),
        "planned_rollouts": 720,
        "manifest_entries": len(entries),
        "source_method_counts": dict(sorted(source_counts.items())),
        "scientific_method_counts_after_required_label_normalization": dict(sorted(method_counts.items())),
        "neutral_alias_rule": (
            "The two 90-row legacy neutral slots are implementation-identical and map to one scientific "
            "π0-Neutral table row. If unblocked, they must be 180 fresh independent rollouts, never copied "
            "outcomes; independent_replication_block preserves provenance."
        ),
        "privileged_surrogate_label_rule": "Legacy FORTE-Reactive slot is scientific-facing only as Privileged Simulator (FORTE-inspired, GT-slip).",
        "joint_in_main_table": False,
        "entries": entries,
    })


def build_empty_results() -> None:
    formal_fields = [
        "task", "root", "root_id", "mu", "repeat", "method", "force_N", "selected_force_N",
        "snapshot_path", "snapshot_sha256", "checkpoint", "checkpoint_hash", "prompt", "pi0_runner_version",
        "git_commit", "environment_version", "seed", "start_timestamp", "end_timestamp", "termination_reason",
        "success", "pick_success", "steps_to_termination", "probe_steps", "downstream_pi0_steps", "privileged",
        "privileged_information", "failure_taxonomy", "data_status",
    ]
    write_csv("FORMAL_ROLLOUT_RESULTS.csv", formal_fields, [])

    frontier_rows = []
    for root in ROOTS:
        for mu in MUS:
            for force in FORCES:
                frontier_rows.append({
                    "task": TASK, "root": root, "mu": mu, "force_N": force,
                    "planned_repeats": 5, "completed_repeats": 0, "success_count": "",
                    "empirical_success_probability": "", "Fstar_0.8_N": "",
                    "downstream_policy": "FROZEN_PI0_FORMAL", "data_status": "NOT_RUN_PHASE_A_HARD_STOP",
                })
    write_csv("FRONTIER_RESULTS.csv", [
        "task", "root", "mu", "force_N", "planned_repeats", "completed_repeats", "success_count",
        "empirical_success_probability", "Fstar_0.8_N", "downstream_policy", "data_status",
    ], frontier_rows)

    methods = [
        ("π0-Neutral", False),
        ("Fixed-Max", False),
        ("ActiveForcing-NoProbe", False),
        ("ActiveForcing-Direct", False),
        ("Tabero-Oracle-Language", True),
        ("Privileged Simulator (FORTE-inspired, GT-slip)", True),
        ("GT-Physics Oracle", True),
    ]
    method_rows = [{
        "method": m, "privileged": p, "n": 0, "success_count": "", "full_task_SR": "",
        "SR_mu_0.2": "", "SR_mu_0.5": "", "SR_mu_1.0": "", "mean_force_N": "",
        "peak_force_N": "", "excess_force_N": "", "under_force_rate": "",
        "grip_failure_rate": "", "data_status": "NOT_ESTIMABLE_PHASE_A_HARD_STOP",
    } for m, p in methods]
    write_csv("METHOD_RESULTS.csv", [
        "method", "privileged", "n", "success_count", "full_task_SR", "SR_mu_0.2", "SR_mu_0.5",
        "SR_mu_1.0", "mean_force_N", "peak_force_N", "excess_force_N", "under_force_rate",
        "grip_failure_rate", "data_status",
    ], method_rows)

    write_csv("PAIRED_RESULTS.csv", [
        "comparison", "matched_n", "rescue_count", "collateral_count", "both_success", "both_failure",
        "net_paired_gain", "paired_mean_gain", "paired_median_gain", "paired_test", "p_value", "data_status",
    ], [{
        "comparison": "ActiveForcing-Direct vs π0-Neutral", "matched_n": 0,
        "rescue_count": "", "collateral_count": "", "both_success": "", "both_failure": "",
        "net_paired_gain": "", "paired_mean_gain": "", "paired_median_gain": "",
        "paired_test": "", "p_value": "", "data_status": "NOT_ESTIMABLE_PHASE_A_HARD_STOP",
    }])

    final_rows = []
    for method, privileged in methods:
        final_rows.append({
            "Method": method, "Category": "PRIVILEGED" if privileged else "PRIMARY_OR_CONTROL",
            "Low μ SR↑": "", "Mid μ SR↑": "", "High μ SR↑": "", "Avg SR↑": "",
            "Mean Force↓": "", "Peak Force↓": "", "Excess Force↓": "", "Grip Failure↓": "",
            "N": 0, "Data Status": "NOT_ESTIMABLE_PHASE_A_HARD_STOP",
        })
    write_csv("FRICTION_BASELINE_FINAL_TABLE.csv", [
        "Method", "Category", "Low μ SR↑", "Mid μ SR↑", "High μ SR↑", "Avg SR↑", "Mean Force↓",
        "Peak Force↓", "Excess Force↓", "Grip Failure↓", "N", "Data Status",
    ], final_rows)


def build_report() -> None:
    report = f"""# Hidden-Friction Baseline — Formal frozen π0 run report

## FINAL STATUS

**{STATUS}**. No scientific interpretation of ActiveForcing, π0-Neutral, Fixed-Max, Tabero-Oracle-Language, or the privileged GT-slip simulator is estimable. Phase A found that the frozen ActiveForcing estimator is defined on a complete 215-step scripted P4-B approach/grasp/probe sequence, while the formal experiment requires it to begin after a frozen π0-reached canonical grasp. The only runtime estimator integration also uses scripted downstream. Repairing that mismatch requires changing the scientific method, so the prescribed hard stop was applied before DEV or TEST execution.

## FROZEN PI0 DOWNSTREAM

- Component action runner: **FOUND**, but formal matched integration: **BLOCKED**.
- Task0 E2E action path: `/home/exouser/Tabero/analysis/results/b5_tabero_neutral_20260822_040652/scripts/b5_tabero_neutral_client.py`.
- Mid-task continuation found: `/home/exouser/Tabero/analysis/p6g1r1_controller_grasp_vla_handoff.py::run_vla_full`; it does not restore upstream temporal context and is entered from fixed staging.
- Checkpoint: `{CHECKPOINT}`.
- Prompt: `{PROMPT}` for π0-Neutral.
- The running instrumented server is action-producing; it is not merely a feature service.
- Exact evidence and hashes: `FROZEN_PI0_DOWNSTREAM_IMPLEMENTATION_AUDIT.md`.

## CANONICAL SNAPSHOTS

- Planned: 18.
- Valid: 0.
- Failed: 0.
- Not attempted because of the Phase-A hard stop: 18.

Zero is not interpreted as reach failure. `CANONICAL_SNAPSHOT_MANIFEST.json` and `CANONICAL_SNAPSHOT_QA.md` contain no fabricated snapshot paths or hashes.

## ROLLOUT COUNTS

- Frontier planned/completed: **810 / 0**.
- Method planned/completed: **720 / 0**.
- DEV microtests completed: **0**.
- TEST rollouts started: **no**.

## PRIMARY RESULTS

- π0-Neutral SR: **not estimable**.
- ActiveForcing-Direct SR: **not estimable**.
- Paired gain: **not estimable**.
- Per-μ breakdown: **not estimable**.

No blank metric was converted to zero.

## PRIVILEGED RESULTS

- Tabero-Oracle-Language: **not estimable**.
- Privileged Simulator (FORTE-inspired, GT-slip): **not estimable**.
- GT-Physics Oracle: **not estimable**.

The simulator surrogate is never labeled as an official FORTE reproduction.

## FRONTIER

All 162 root×μ×force cells remain planned with 0/5 completed. No existing scripted frontier row was imported or renamed. Per-μ F* and root heterogeneity are therefore not estimable.

## QA

- Frozen bundle checksum before work: pass for all 11 original files.
- Planned count audit: pass; the source manifest contains exactly 720 rows and 40 rows per each of 18 root×μ cases.
- Frontier arithmetic: pass; 6×3×9×5 = 810.
- Pre-execution root leakage audit: pass; only 5174–5179 are TEST, and the active TRAIN collector was confined to 5112–5173 at audit time.
- Scientific-facing method labels: pass; π0-Default/Tabero-Neutral map to one π0-Neutral row, and the legacy reactive slot maps to `Privileged Simulator (FORTE-inspired, GT-slip)`.
- Downstream semantic match: **not passed / blocker**.
- Snapshot pairing and replay equivalence: not run.
- JSON/CSV parse and SHA-256 checks: recorded after generation.

## SCIENTIFIC VERDICT

**No scientific verdict is permitted.** The experiment is not estimable under the current frozen interfaces. In particular, this run does not establish whether ActiveForcing helps, hurts, matches Fixed-Max, anticipates slip better than the privileged reactive surrogate, or closes the gap to GT physics.

## Required resolution before resumption

The freeze owner must supply an already-authoritative mapping for one of the following without using TEST outcomes:

1. how the frozen 215-step P4-B estimator input is constructed from a π0-reached canonical pre-probe state; or
2. a pre-existing frozen probe-only estimator/checkpoint whose training and normalization match the post-grasp query.

Choosing, padding, relabeling, or retraining that interface here would violate the current prohibition on scientific method changes. Once resolved outside this frozen TEST run, a new amendment must be hashed before any DEV microtest or TEST snapshot capture.
"""
    write_text("FRICTION_BASELINE_FINAL_REPORT.md", report)


def build_status_and_hashes() -> None:
    write_json("RUN_STATUS.json", {
        "status": STATUS,
        "updated_utc": CREATED,
        "baseline_rollouts_started": False,
        "dev_microtests_completed": 0,
        "canonical_snapshots_planned": 18,
        "canonical_snapshots_valid": 0,
        "canonical_snapshots_failed": 0,
        "canonical_snapshots_not_attempted": 18,
        "planned_frontier_rollouts": 810,
        "frontier_rollouts_completed": 0,
        "planned_method_rollouts": 720,
        "method_rollouts_completed": 0,
        "task": TASK,
        "roots": ROOTS,
        "mu_values": MUS,
        "checkpoint": CHECKPOINT,
        "git_commit": GIT_COMMIT,
        "hard_stop_condition": 10,
        "blocker": (
            "Frozen FRICTION_GRU requires the complete 215-step scripted P4-B approach/grasp/probe trace; "
            "no frozen adapter exists for a probe-only branch from a frozen-pi0-reached canonical grasp. "
            "The only runtime integration uses scripted downstream."
        ),
        "zero_is_failure_data": False,
        "existing_gpu_processes_modified": False,
    })

    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    lines = [f"{sha(path)}  {path.name}" for path in files]
    write_text("SHA256SUMS.txt", "\n".join(lines))


def main() -> None:
    if sha(SOURCE_PLAN) != "4f76485bfdabcc76d8a86b1b805179f6b7931c325ca19be2d01e768a363fa8d2":
        raise RuntimeError("source 720-row frozen plan changed")
    if sha(CASE_MANIFEST) != "d4f61ed047e699a9b5e11d5d7298d6e6305ef6b1bb1ee18d0719d113df3d964b":
        raise RuntimeError("frozen case manifest changed")
    build_audit()
    build_snapshots()
    build_formal_manifest()
    build_empty_results()
    build_report()
    build_status_and_hashes()


if __name__ == "__main__":
    main()
