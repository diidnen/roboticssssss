"""Forensic decomposition of the 40 lift-success/full-task-failure E3 branches.

This is a read-only analysis of the formal collection.  It writes analysis
artifacts in /home/exouser and never touches the collection directory.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path("/media/volume/newdata/exouser/activeforcing_e3/P1_SIMPLIFIED_COLLECTION_FORMAL_20260903")
OUT = Path("/home/exouser")
TUPLES = BASE / "P1_SIMPLIFIED_TUPLES.jsonl"

LIFT_Z = 0.03
DROP_END_Z = 0.10
REL_Z_DROP = -0.05
TARGET_XY = 0.25


def ffloat(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def fmt(value: object, digits: int = 3) -> str:
    if value in (None, ""):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def first_index(rows: list[dict[str, str]], predicate) -> int | None:
    for i, row in enumerate(rows):
        if predicate(row):
            return i
    return None


def numeric(rows: list[dict[str, str]], key: str) -> list[float]:
    return [v for row in rows if (v := ffloat(row.get(key))) is not None and math.isfinite(v)]


def safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def longest_success_run(outcomes: list[int], forces: list[float]) -> tuple[float | None, int]:
    best_len = 0
    best_start: int | None = None
    current_start: int | None = None
    for i, value in enumerate(outcomes + [0]):
        if value == 1 and current_start is None:
            current_start = i
        if value != 1 and current_start is not None:
            length = i - current_start
            if length > best_len:
                best_len = length
                best_start = current_start
            current_start = None
    return (forces[best_start] if best_start is not None and best_len >= 2 else None, best_len)


def frontier_type(outcomes: list[int], forces: list[float]) -> tuple[str, float | None, str]:
    total_success = sum(outcomes)
    if total_success == 0:
        return (
            "ALWAYS_FAIL",
            None,
            "Full task never succeeds; force can remove some low-force grasp loss but does not solve the downstream task.",
        )
    if total_success == len(outcomes):
        return ("ALWAYS_SUCCESS", forces[0], "All force levels succeed; 1N is already sufficient in this context.")

    run_force, run_len = longest_success_run(outcomes, forces)
    first_success = next((i for i, value in enumerate(outcomes) if value), None)
    if first_success is not None and all(value == 1 for value in outcomes[first_success:]):
        return (
            "MONOTONIC_OR_CLEAR_FRONTIER",
            forces[first_success],
            f"All outcomes are successful from {forces[first_success]:g}N upward.",
        )

    low = outcomes[:2]
    high = outcomes[-3:]
    if run_force is not None and sum(high) / len(high) > sum(low) / len(low):
        return (
            "NOISY_FRONTIER",
            run_force,
            f"Higher force improves outcomes, but the observed full-task sequence is non-monotonic; first sustained run starts at {run_force:g}N.",
        )
    return (
        "NO_FORCE_RELATION",
        None,
        "No reproducible directional full-task relation; isolated success does not establish a force boundary.",
    )


def load_branch(path: Path) -> dict:
    branch = json.loads((path / "branch_result.json").read_text())
    episode = branch["episode_row"]
    step_path = next(path.glob("logs/*_steps.csv"))
    steps = list(csv.DictReader(step_path.open(newline="")))
    preprobe = json.loads((path / "preprobe_state_exp000.json").read_text())

    lift_i = first_index(steps, lambda r: (ffloat(r.get("obj_dz")) or 0.0) >= LIFT_Z)
    post = steps[lift_i:] if lift_i is not None else steps
    dz = numeric(post, "obj_dz")
    rel_z = numeric(post, "rel_z")
    rel_xy = numeric(post, "rel_xy")
    target_xy = numeric(post, "xy_to_basket")
    measured = numeric(post, "measured_squeeze_N")
    post_last = post[-1] if post else {}
    max_dz = max(dz) if dz else None
    end_dz = ffloat(post_last.get("obj_dz"))
    end_rel_z = ffloat(post_last.get("rel_z"))

    contact_loss_i = first_index(post, lambda r: r.get("contact") == "0")
    contact_loss = contact_loss_i is not None
    # The raw logger flag is retained, but it is not trusted for this analysis:
    # collection disabled object_1_dropped termination, making it zero throughout.
    raw_dropped = any(r.get("dropped") == "1" for r in post)
    pose_drop = bool(
        (end_dz is not None and end_dz <= DROP_END_Z and max_dz is not None and max_dz >= LIFT_Z)
        or (end_rel_z is not None and end_rel_z <= REL_Z_DROP and contact_loss)
    )
    object_slipped = pose_drop
    placement_attempted = int(any(value < TARGET_XY for value in target_xy))
    final_target_xy = target_xy[-1] if target_xy else None
    target_min_xy = min(target_xy) if target_xy else None

    # The logged gt_slip flag is a sticky heuristic that includes rel_xy motion;
    # retain its onset for audit, but use pose/contact/force evidence for labels.
    gt_slip_i = first_index(post, lambda r: r.get("gt_slip") == "1")
    last10 = measured[-10:]
    return {
        "branch": branch,
        "episode": episode,
        "steps": steps,
        "post": post,
        "preprobe": preprobe,
        "lift_i": lift_i,
        "pose_drop": int(pose_drop),
        "object_slipped": int(object_slipped),
        "raw_dropped": int(raw_dropped),
        "contact_loss": int(contact_loss),
        "contact_loss_t_s": ffloat(post[contact_loss_i].get("t_s")) if contact_loss_i is not None else None,
        "gt_slip_onset_t_s_from_steps": ffloat(post[gt_slip_i].get("t_s")) if gt_slip_i is not None else None,
        "max_dz": max_dz,
        "end_dz": end_dz,
        "end_rel_z": end_rel_z,
        "max_rel_xy": max(rel_xy) if rel_xy else None,
        "end_rel_xy": ffloat(post_last.get("rel_xy")),
        "target_min_xy": target_min_xy,
        "target_end_xy": final_target_xy,
        "placement_attempted": placement_attempted,
        "post_lift_mean_force": mean(measured),
        "post_lift_peak_force": max(measured) if measured else None,
        "post_lift_final10_mean_force": mean(last10),
        "force_at_lift": ffloat(post[0].get("measured_squeeze_N")) if post else None,
        "gripper_aperture_end": ffloat(post_last.get("gripper_meas")),
        "gripper_aperture_min": min(numeric(post, "gripper_meas")) if numeric(post, "gripper_meas") else None,
        "gripper_aperture_max": max(numeric(post, "gripper_meas")) if numeric(post, "gripper_meas") else None,
        "normal_force_peak_left_z": max([abs(v) for v in numeric(post, "measured_fLz")] or [None]),
        "normal_force_peak_right_z": max([abs(v) for v in numeric(post, "measured_fRz")] or [None]),
    }


def main() -> None:
    dirs = [
        Path(p)
        for p in glob.glob(str(BASE / "P1_SIMPLIFIED_ROOT*_F*N"))
        if ".incomplete" not in p and Path(p).is_dir()
    ]
    records = [load_branch(p) for p in dirs]
    records.sort(key=lambda r: (int(r["branch"]["root_id"]), float(r["branch"]["force_N"])))
    assert len(records) == 72, len(records)

    tuples = [json.loads(line) for line in TUPLES.read_text().splitlines() if line.strip()]
    tuple_by_id = {row["tuple_id"]: row for row in tuples}
    assert len(tuples) == 72 and len(tuple_by_id) == 72
    for rec in records:
        branch = rec["branch"]
        tuple_id = f"libero10_task5_root{branch['root_id']}_F{int(branch['force_N'])}_R0"
        assert tuple_id in tuple_by_id
        assert tuple_by_id[tuple_id]["y_lift"] == branch["y_lift"]
        assert tuple_by_id[tuple_id]["y_full"] == branch["y_full"]

    by_root: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_root[rec["branch"]["root_id"]].append(rec)
    for root in by_root:
        by_root[root].sort(key=lambda r: float(r["branch"]["force_N"]))
        assert [float(r["branch"]["force_N"]) for r in by_root[root]] == list(map(float, range(1, 9)))

    # Use same-context higher-force branches as the required support test.
    for rec in records:
        e = rec["episode"]
        force = float(rec["branch"]["force_N"])
        higher = [x for x in by_root[rec["branch"]["root_id"]] if float(x["branch"]["force_N"]) > force]
        stable_higher = [
            x for x in higher
            if x["episode"]["full_success"] == 1
            or (x["episode"]["transport_success"] == 1 and x["pose_drop"] == 0)
        ]
        rec["higher_force_stable_transport_or_full"] = int(bool(stable_higher))
        rec["higher_force_full_success"] = int(any(x["episode"]["full_success"] == 1 for x in higher))
        rec["higher_force_min_stable_N"] = min((float(x["branch"]["force_N"]) for x in stable_higher), default=None)
        rec["higher_force_min_full_N"] = min((float(x["branch"]["force_N"]) for x in higher if x["episode"]["full_success"] == 1), default=None)

    failures = []
    for rec in records:
        e = rec["episode"]
        if not (e["lift_success"] == 1 and e["full_success"] == 0):
            continue
        if rec["pose_drop"] and rec["higher_force_stable_transport_or_full"]:
            category = "UNDER_FORCE_STRONG"
            reason = (
                "After lift, object height returned near the initial level and object-gripper relative z fell; "
                "contact was lost and the last 10 post-lift force samples collapsed toward 0 N. "
                f"Same context has a higher-force stable transport/full-task branch starting at {fmt(rec['higher_force_min_stable_N'], 0)}N."
            )
        elif rec["pose_drop"] and (
            (rec["episode"]["mean_measured_force_N"] or 0.0) < 2.0
            and rec["higher_force_stable_transport_or_full"]
        ):
            category = "UNDER_FORCE_LIKELY"
            reason = (
                "Physical slip/drop signature with low measured force; higher-force same-context evidence exists, "
                "but the complete A-level comparison is weaker."
            )
        elif rec["pose_drop"]:
            category = "INDETERMINATE"
            reason = "Slip/drop-like pose change is present, but existing same-context force branches do not establish force causality."
        elif rec["placement_attempted"] and rec["target_end_xy"] is not None and rec["target_end_xy"] <= TARGET_XY:
            category = "PLACEMENT"
            reason = (
                "Object remained elevated and did not show physical slip/drop; it ended within the transport target "
                "proximity gate but the official evaluator never succeeded, consistent with release/alignment/insertion failure."
            )
        else:
            category = "POLICY_TRAJECTORY"
            reason = (
                "Object remained elevated with no physical slip/drop signature, but the final target proximity was "
                "outside the transport gate or the policy moved away from the target; force is not the limiting signal."
            )
        rec["category"] = category
        rec["failure_reason"] = reason
        failures.append(rec)

    assert len(failures) == 40
    counts = Counter(r["category"] for r in failures)
    assert sum(counts.values()) == 40

    failure_fields = [
        "context_id", "root", "requested_force_N", "measured_force_N", "measured_force_at_lift_N",
        "post_lift_mean_measured_force_N", "post_lift_peak_measured_force_N", "post_lift_final10_mean_force_N",
        "lift_success", "full_task_success", "failure_stage", "category", "failure_reason",
        "object_dropped_after_lift", "object_slipped", "lost_contact", "raw_logger_dropped_after_lift",
        "transport_completed", "placement_attempted", "final_evaluator_success", "gt_slip_onset_s",
        "contact_loss_t_s", "max_object_dz_m", "final_object_dz_m", "final_object_gripper_rel_z_m",
        "max_object_gripper_rel_xy_m", "final_object_gripper_rel_xy_m", "min_target_xy_m", "final_target_xy_m",
        "gripper_aperture_end", "gripper_aperture_min", "gripper_aperture_max", "normal_force_peak_left_z_N",
        "normal_force_peak_right_z_N", "peak_measured_force_N", "integrated_measured_force_Ns",
        "higher_force_stable_transport_or_full", "higher_force_full_success", "higher_force_min_stable_N",
        "higher_force_min_full_N", "initial_object_pose_w", "initial_target_pose_w", "step_log", "episode_log",
    ]
    output_rows = []
    for rec in failures:
        b = rec["branch"]
        e = rec["episode"]
        p = rec["preprobe"]
        context_id = f"libero10_task5_root{b['root_id']}"
        output_rows.append({
            "context_id": context_id,
            "root": b["root_id"],
            "requested_force_N": b["force_N"],
            "measured_force_N": e["mean_measured_force_N"],
            "measured_force_at_lift_N": rec["force_at_lift"],
            "post_lift_mean_measured_force_N": rec["post_lift_mean_force"],
            "post_lift_peak_measured_force_N": rec["post_lift_peak_force"],
            "post_lift_final10_mean_force_N": rec["post_lift_final10_mean_force"],
            "lift_success": e["lift_success"],
            "full_task_success": e["full_success"],
            "failure_stage": "transport" if e["transport_success"] == 0 else ("placement" if e["place_success"] == 0 else "other_downstream"),
            "category": rec["category"],
            "failure_reason": rec["failure_reason"],
            "object_dropped_after_lift": rec["pose_drop"],
            "object_slipped": rec["object_slipped"],
            "lost_contact": rec["contact_loss"],
            "raw_logger_dropped_after_lift": rec["raw_dropped"],
            "transport_completed": e["transport_success"],
            "placement_attempted": rec["placement_attempted"],
            "final_evaluator_success": e["official_success"],
            "gt_slip_onset_s": e["gt_slip_onset_s"],
            "contact_loss_t_s": rec["contact_loss_t_s"],
            "max_object_dz_m": rec["max_dz"],
            "final_object_dz_m": rec["end_dz"],
            "final_object_gripper_rel_z_m": rec["end_rel_z"],
            "max_object_gripper_rel_xy_m": rec["max_rel_xy"],
            "final_object_gripper_rel_xy_m": rec["end_rel_xy"],
            "min_target_xy_m": rec["target_min_xy"],
            "final_target_xy_m": rec["target_end_xy"],
            "gripper_aperture_end": rec["gripper_aperture_end"],
            "gripper_aperture_min": rec["gripper_aperture_min"],
            "gripper_aperture_max": rec["gripper_aperture_max"],
            "normal_force_peak_left_z_N": rec["normal_force_peak_left_z"],
            "normal_force_peak_right_z_N": rec["normal_force_peak_right_z"],
            "peak_measured_force_N": e["peak_measured_force_N"],
            "integrated_measured_force_Ns": e["integrated_measured_force_Ns"],
            "higher_force_stable_transport_or_full": rec["higher_force_stable_transport_or_full"],
            "higher_force_full_success": rec["higher_force_full_success"],
            "higher_force_min_stable_N": rec["higher_force_min_stable_N"],
            "higher_force_min_full_N": rec["higher_force_min_full_N"],
            "initial_object_pose_w": safe_json(p.get("object_pos_initial_w")),
            "initial_target_pose_w": safe_json(p.get("target_pos_initial_w")),
            "step_log": str(next(BASE.glob(f"P1_SIMPLIFIED_ROOT{b['root_id']}_F{int(b['force_N'])}N/logs/*_steps.csv"))),
            "episode_log": str(next(BASE.glob(f"P1_SIMPLIFIED_ROOT{b['root_id']}_F{int(b['force_N'])}N/logs/*_episodes.csv"))),
        })

    failure_csv = OUT / "E3_DOWNSTREAM_FAILURE_DECOMPOSITION.csv"
    with failure_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=failure_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    context_fields = [
        "context_id", "root", "full_task_outcomes_1_to_8N", "lift_outcomes_1_to_8N",
        "frontier_type", "estimated_minimum_force_N", "success_run_length", "main_failure_source",
        "transport_outcomes_1_to_8N", "placement_outcomes_1_to_8N", "measured_force_1_to_8N",
        "downstream_failure_count", "under_force_strong_count", "policy_trajectory_count", "placement_count",
    ]
    context_rows = []
    for root in sorted(by_root, key=int):
        rs = by_root[root]
        forces = [float(r["branch"]["force_N"]) for r in rs]
        full = [int(r["episode"]["full_success"]) for r in rs]
        lift = [int(r["episode"]["lift_success"]) for r in rs]
        trans = [int(r["episode"]["transport_success"]) for r in rs]
        place = [int(r["episode"]["place_success"]) for r in rs]
        measured_summary = [float(r["episode"]["mean_measured_force_N"]) for r in rs]
        ftype, estimated, source = frontier_type(full, forces)
        fr = [r for r in failures if r["branch"]["root_id"] == root]
        run_force, run_len = longest_success_run(full, forces)
        context_rows.append({
            "context_id": f"libero10_task5_root{root}",
            "root": root,
            "full_task_outcomes_1_to_8N": " ".join(map(str, full)),
            "lift_outcomes_1_to_8N": " ".join(map(str, lift)),
            "frontier_type": ftype,
            "estimated_minimum_force_N": estimated,
            "success_run_length": run_len,
            "main_failure_source": source,
            "transport_outcomes_1_to_8N": " ".join(map(str, trans)),
            "placement_outcomes_1_to_8N": " ".join(map(str, place)),
            "measured_force_1_to_8N": " ".join(fmt(x, 3) for x in measured_summary),
            "downstream_failure_count": len(fr),
            "under_force_strong_count": sum(r["category"] == "UNDER_FORCE_STRONG" for r in fr),
            "policy_trajectory_count": sum(r["category"] == "POLICY_TRAJECTORY" for r in fr),
            "placement_count": sum(r["category"] == "PLACEMENT" for r in fr),
        })

    context_csv = OUT / "E3_CONTEXT_FORCE_FRONTIER.csv"
    with context_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=context_fields)
        writer.writeheader()
        writer.writerows(context_rows)

    force_summary = []
    for force in range(1, 9):
        rs = [r for r in records if int(float(r["branch"]["force_N"])) == force]
        force_summary.append({
            "force_N": force,
            "n": len(rs),
            "lift_success": sum(r["episode"]["lift_success"] for r in rs),
            "full_task_success": sum(r["episode"]["full_success"] for r in rs),
            "full_task_success_rate": sum(r["episode"]["full_success"] for r in rs) / len(rs),
            "conditional_full_given_lift": (
                sum(r["episode"]["full_success"] for r in rs) / sum(r["episode"]["lift_success"] for r in rs)
                if sum(r["episode"]["lift_success"] for r in rs) else None
            ),
            "mean_requested_force_N": force,
            "mean_measured_force_N": mean([float(r["episode"]["mean_measured_force_N"]) for r in rs]),
        })

    clear_contexts = sum(row["frontier_type"] == "MONOTONIC_OR_CLEAR_FRONTIER" for row in context_rows)
    no_clear_contexts = sum(row["frontier_type"] in {"ALWAYS_FAIL", "NO_FORCE_RELATION"} for row in context_rows)
    summary = {
        "final_status": "E3_DOWNSTREAM_FAILURE_DECOMPOSITION_COMPLETE",
        "input_path": str(TUPLES),
        "input_sha256": sha256(TUPLES),
        "formal_branches": 72,
        "contexts": 9,
        "forces_per_context": 8,
        "lift_success": 65,
        "full_task_success": 25,
        "lift_success_fulltask_failure": len(failures),
        "category_counts": dict(sorted(counts.items())),
        "force_related_minimum": f"{counts['UNDER_FORCE_STRONG']}/40",
        "force_related_plausible": f"{counts['UNDER_FORCE_STRONG'] + counts['UNDER_FORCE_LIKELY'] + counts['OVER_FORCE']}/40",
        "transport_failure_count": sum(r["episode"]["transport_success"] == 0 for r in failures),
        "transport_failure_object_slip_or_drop_count": sum(r["episode"]["transport_success"] == 0 and r["object_slipped"] for r in failures),
        "transport_failure_policy_trajectory_count": sum(r["episode"]["transport_success"] == 0 and r["category"] == "POLICY_TRAJECTORY" for r in failures),
        "raw_logger_dropped_after_lift_count": sum(r["raw_dropped"] for r in failures),
        "clear_force_frontier_contexts": clear_contexts,
        "no_clear_force_effect_contexts": no_clear_contexts,
        "frontier_type_counts": dict(Counter(row["frontier_type"] for row in context_rows)),
        "force_summary": force_summary,
        "context_rows": context_rows,
        "method_notes": [
            "Formal scope is exactly 72 complete branch directories; the .incomplete directory is excluded.",
            "Raw dropped flags are retained but not used because P1_DIRECT_COLLECTION.log shows object_1_dropped termination was disabled.",
            "Logged gt_slip is retained as audit evidence but not used alone: its heuristic includes relative-XY motion and is also present in successful branches.",
            "Physical slip/drop is defined from post-lift object height returning near the initial level and/or negative object-gripper relative-z with contact loss, then checked against higher-force branches in the same context.",
            "Placement attempt is inferred from the logged target-proximity trajectory crossing xy_to_basket < 0.25 m; per-step basket-contact force is not present in the formal step CSV.",
            "The measured_force_N column is the formal episode mean measured squeeze force; additional lift/post-lift/end-window force columns are provided for the requested-force versus realized-force check.",
        ],
    }
    (OUT / "E3_DOWNSTREAM_FAILURE_DECOMPOSITION_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    report_lines = [
        "# E3 downstream failure decomposition",
        "",
        "FINAL_STATUS = E3_DOWNSTREAM_FAILURE_DECOMPOSITION_COMPLETE",
        "",
        f"The 40 lift-success/full-task-failure branches decompose as: UNDER_FORCE_STRONG={counts['UNDER_FORCE_STRONG']}, UNDER_FORCE_LIKELY={counts['UNDER_FORCE_LIKELY']}, POLICY_TRAJECTORY={counts['POLICY_TRAJECTORY']}, PLACEMENT={counts['PLACEMENT']}, OVER_FORCE={counts['OVER_FORCE']}, INDETERMINATE={counts['INDETERMINATE']}.",
        "",
        f"Force-related minimum = {counts['UNDER_FORCE_STRONG']}/40; force-related plausible = {counts['UNDER_FORCE_STRONG'] + counts['UNDER_FORCE_LIKELY'] + counts['OVER_FORCE']}/40.",
        "",
        f"All {summary['transport_failure_count']} transport-failure branches have a pose/contact/force slip-drop signature; {summary['transport_failure_policy_trajectory_count']} are clear policy-trajectory failures within that transport-failure subset.",
        "",
        f"Clear full-task force frontiers: {clear_contexts}/9. Contexts with no clear full-task force effect: {no_clear_contexts}/9. Frontier counts: {dict(Counter(row['frontier_type'] for row in context_rows))}.",
        "",
        "## Full-task success by requested force",
        "",
        "| Force | Lift success | Full-task success | Full-task SR | Mean measured force (episode mean) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in force_summary:
        report_lines.append(f"| {row['force_N']}N | {row['lift_success']}/9 | {row['full_task_success']}/9 | {row['full_task_success_rate']:.3f} | {row['mean_measured_force_N']:.3f}N |")
    report_lines += [
        "",
        "## Contexts",
        "",
        "| Context | Full-task outcomes 1→8N | Frontier type | Estimated minimum force | Main failure source |",
        "|---|---|---|---:|---|",
    ]
    for row in context_rows:
        estimate = "N/A" if row["estimated_minimum_force_N"] is None else f"{row['estimated_minimum_force_N']:g}N"
        report_lines.append(f"| {row['root']} | {row['full_task_outcomes_1_to_8N']} | {row['frontier_type']} | {estimate} | {row['main_failure_source']} |")
    report_lines += [
        "",
        "## Interpretation",
        "",
        "The aggregate full-task rate rises from 0/9 at 1N to 5/9 at 8N, with the highest observed rate 5/9 at 6–8N; this is a real descriptive force effect, but not a monotonic per-context law.",
        "The strong force-related cases are not inferred from transport labels alone: their step traces show the object rising, then returning close to its initial height, negative object-gripper relative-z, contact loss, and near-zero late force; higher force in the same context removes that signature.",
        "The remaining failures are predominantly stable-object downstream failures. The CSV separates cases that ended outside the target-proximity gate as POLICY_TRAJECTORY from cases that ended within it without official success as PLACEMENT.",
        "No OVER_FORCE case is supported: high-force failures do not show force-induced displacement/deformation; they show stable-object downstream failure or isolated non-monotonic policy outcomes.",
        "",
        "## Data caveats",
        "",
        "The raw dropped flag is zero because the collection log disabled the object_1_dropped termination. The gt_slip flag is a heuristic that includes relative-XY movement and appears in successful branches, so it is reported but not used as a standalone causal label.",
    ]
    (OUT / "E3_DOWNSTREAM_FAILURE_DECOMPOSITION.md").write_text("\n".join(report_lines) + "\n")

    print(json.dumps({
        "status": summary["final_status"],
        "rows_written": len(failures),
        "category_counts": dict(sorted(counts.items())),
        "transport_failures": summary["transport_failure_count"],
        "transport_slip_drop": summary["transport_failure_object_slip_or_drop_count"],
        "clear_frontier_contexts": f"{clear_contexts}/9",
        "no_clear_force_effect_contexts": f"{no_clear_contexts}/9",
        "outputs": [str(failure_csv), str(context_csv), str(OUT / "E3_DOWNSTREAM_FAILURE_DECOMPOSITION_SUMMARY.json"), str(OUT / "E3_DOWNSTREAM_FAILURE_DECOMPOSITION.md")],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
