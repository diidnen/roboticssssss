#!/usr/bin/env python3
"""Materialize context-level force supervision artifacts without touching old runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


FORCES = (3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evidence_row(row: dict, *, origin: str) -> dict:
    ev = row.get("evidence") or {}
    dyn = ev.get("dynamic_metrics") or {}
    query = ev.get("query_info") or {}
    task = row.get("task") or ev.get("task_name") or "unknown"
    root = row.get("root_slot", row.get("fresh_slot", "unknown"))
    friction = row.get("friction", ev.get("af_contact_friction"))
    force = row.get("force_n", ev.get("af_force_limit_n"))
    context_id = row.get("context_id") or f"{task}|slot={root}|mu={float(friction):.2f}"
    full_success = row.get("success", ev.get("success"))
    if full_success is not None:
        full_success = bool(full_success)
    contact_ratio = dyn.get("contact_ratio")
    irrecoverable = dyn.get("irrecoverable_failure")
    samples = dyn.get("samples")
    # These are conservative operational labels. Unknown means the existing
    # trace does not expose a release/placement event; we never infer it.
    lift_success = None
    no_lift = None
    pre_release_drop = None
    post_lift_non_drop_failure = None
    retention_success = None
    if irrecoverable is True:
        pre_release_drop = True
        retention_success = False
    elif full_success is True and contact_ratio is not None and float(contact_ratio) >= 0.70:
        lift_success = True
        no_lift = False
        retention_success = True
    elif full_success is False and irrecoverable is False:
        # The trace says the object was not classified as irrecoverably lost;
        # the remaining failure stage is not observable in this task adapter.
        post_lift_non_drop_failure = True
    return {
        "context_id": context_id,
        "root_id": str(row.get("root_id", root)),
        "task": task,
        "friction": "" if friction is None else float(friction),
        "split": row.get("split", "fresh" if origin == "existing_fresh" else "unknown"),
        "force_setpoint": "" if force is None else float(force),
        "measured_contact_squeeze": dyn.get("measured_force_mean_n", query.get("measured_force_mean_n", "")),
        "full_task_success": "" if full_success is None else int(full_success),
        "lift_success": "unknown" if lift_success is None else int(lift_success),
        "pre_release_drop": "unknown" if pre_release_drop is None else int(pre_release_drop),
        "no_lift": "unknown" if no_lift is None else int(no_lift),
        "post_lift_non_drop_failure": "unknown" if post_lift_non_drop_failure is None else int(post_lift_non_drop_failure),
        "placement_failure": "unknown",
        "release_failure": "unknown",
        "retention_success": "unknown" if retention_success is None else int(retention_success),
        "rollout_seed": row.get("actual_seed", row.get("seed", "")),
        "policy_seed": row.get("policy_seed", ""),
        "branch_id": row.get("branch_key", row.get("context_id", "")),
        "source_directory": str(row.get("source_directory", row.get("source", ""))),
        "origin": origin,
        "valid": row.get("valid", True),
        "dynamic_samples": samples if samples is not None else "",
        "contact_ratio": contact_ratio if contact_ratio is not None else "",
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, ctr - half), min(1.0, ctr + half)


def context_key(row: dict) -> tuple[str, float]:
    return row["context_id"], float(row["friction"])


def summarize_contexts(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        if row["force_setpoint"] != "":
            groups[context_key(row)].append(row)
    result = []
    for (context_id, friction), members in sorted(groups.items()):
        valid = [r for r in members if str(r["valid"]).lower() not in {"false", "0"}]
        successes = [r for r in valid if r["full_task_success"] == 1]
        forces = sorted(float(r["force_setpoint"]) for r in valid)
        success_forces = sorted(float(r["force_setpoint"]) for r in successes)
        outcomes = [bool(r["full_task_success"] == 1) for r in sorted(valid, key=lambda x: float(x["force_setpoint"]))]
        transitions = sum(a != b for a, b in zip(outcomes, outcomes[1:]))
        retention_failures = sum(r["retention_success"] == 0 for r in valid)
        post_failures = sum(r["post_lift_non_drop_failure"] == 1 for r in valid)
        result.append({
            "context_id": context_id,
            "friction": friction,
            "root_id": str(members[0]["root_id"]),
            "task": members[0]["task"],
            "split": members[0]["split"],
            "branches_3_5n": len(valid),
            "success_count": len(successes),
            "minimum_successful_tested_force": min(success_forces) if success_forces else "",
            "all_tested_forces_succeeded": int(bool(valid) and len(successes) == len(valid)),
            "all_tested_forces_failed": int(bool(valid) and not successes),
            "observed_transition_count": transitions,
            "pre_release_retention_failures": retention_failures,
            "post_lift_non_retention_failures": post_failures,
            "apparent_type": "EASY" if valid and len(successes) == len(valid) else ("UNRESCUED" if valid and not successes else "TRANSITION"),
            "force_min": min(forces) if forces else "",
            "force_max": max(forces) if forces else "",
        })
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_plots(out: Path, curves: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    informative = defaultdict(list)
    for row in curves:
        if int(row["n"]) > 0:
            informative[row["context_id"]].append(row)
    plot_dir = out / "force_response_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for context_id, members in informative.items():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda r: float(r["force_setpoint"]))
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        for label, key, color in (("full-task", "p_full", "tab:blue"), ("retention", "p_retention", "tab:orange")):
            xs = [float(r["force_setpoint"]) for r in members]
            ys = [float(r[key]) if r[key] != "" else float("nan") for r in members]
            ax.plot(xs, ys, marker="o", label=label, color=color)
        ax.set(xlabel="Force setpoint (N)", ylabel="Empirical success probability", ylim=(-0.05, 1.05))
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        safe = context_id.replace("|", "_").replace("=", "-").replace(".", "_")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{safe}.png", dpi=140)
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeated", type=Path)
    args = ap.parse_args()
    source = args.experiment
    rows = []
    rows.extend(evidence_row(r, origin="existing_development") for r in read_jsonl(source / "branches.jsonl"))
    rows.extend(evidence_row(r, origin="existing_fresh") for r in read_jsonl(source / "fresh_online_branches.jsonl"))
    if args.repeated:
        rows.extend(evidence_row(r, origin="newly_collected_repeat") for r in read_jsonl(args.repeated))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    write_csv(out / "force_context_table.csv", rows, fields)
    summaries = summarize_contexts([r for r in rows if r["origin"] != "existing_fresh"])
    summary_fields = list(summaries[0].keys()) if summaries else []
    write_csv(out / "context_summary.csv", summaries, summary_fields)

    groups: dict[tuple[str, float, float], list[dict]] = defaultdict(list)
    for row in rows:
        if row["origin"] == "existing_fresh" or row["force_setpoint"] == "":
            continue
        groups[(row["context_id"], float(row["friction"]), float(row["force_setpoint"]))].append(row)
    curves = []
    for (context_id, friction, force), members in sorted(groups.items()):
        full = [r for r in members if r["full_task_success"] in (0, 1)]
        ret = [r for r in members if r["retention_success"] in (0, 1)]
        fk, fl, fu = wilson(sum(r["full_task_success"] == 1 for r in full), len(full))
        rk, rl, ru = wilson(sum(r["retention_success"] == 1 for r in ret), len(ret))
        curves.append({
            "context_id": context_id,
            "friction": friction,
            "force_setpoint": force,
            "n": len(full),
            "full_successes": sum(r["full_task_success"] == 1 for r in full),
            "p_full": "" if not full else fk,
            "full_ci_low": "" if not full else fl,
            "full_ci_high": "" if not full else fu,
            "retention_n": len(ret),
            "retention_successes": sum(r["retention_success"] == 1 for r in ret),
            "p_retention": "" if not ret else rk,
            "retention_ci_low": "" if not ret else rl,
            "retention_ci_high": "" if not ret else ru,
        })
    write_csv(out / "repeated_force_curves.csv", curves, list(curves[0].keys()) if curves else [])
    qualification = []
    by_context = defaultdict(list)
    for row in rows:
        if row["origin"] != "existing_fresh" and row["force_setpoint"] != "":
            by_context[row["context_id"]].append(row)
    for context_id, members in sorted(by_context.items()):
        endpoint = {}
        for force in (3.0, 5.0):
            xs = [r for r in members if abs(float(r["force_setpoint"]) - force) < 1e-8 and r["full_task_success"] in (0, 1)]
            k = sum(r["full_task_success"] == 1 for r in xs)
            endpoint[str(force)] = (k, len(xs), k / len(xs) if xs else "")
        p3, p5 = endpoint["3.0"][2], endpoint["5.0"][2]
        if p3 != "" and p5 != "" and p3 >= 2 / 3:
            typ = "EASY"
        elif p3 != "" and p5 != "" and p5 >= 2 / 3 and p3 < p5:
            typ = "TRANSITION"
        elif p5 != "" and p5 < 2 / 3:
            typ = "UNRESCUED"
        else:
            typ = "UNKNOWN"
        qualification.append({"context_id": context_id, "force_3_n": endpoint["3.0"][0], "n_3": endpoint["3.0"][1], "p_full_3": endpoint["3.0"][2], "force_5_n": endpoint["5.0"][0], "n_5": endpoint["5.0"][1], "p_full_5": endpoint["5.0"][2], "qualification_type": typ})
    write_csv(out / "qualification_results.csv", qualification, list(qualification[0].keys()) if qualification else [])
    make_plots(out, curves)
    print(json.dumps({"rows": len(rows), "contexts": len(summaries), "curves": len(curves), "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
