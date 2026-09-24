#!/usr/bin/env python3
"""Analyze the frozen-query structured transport qualification outputs.

The qualification outcome is transport retention: a trajectory that reaches
the basket after losing the object is not a successful full-task branch.
Partial worker directories are retained but excluded from the summary.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def complete_dir(path: Path, task: int) -> bool:
    proto = path / f"M2_STRUCTURED_TASK{task}_PROTOCOL.json"
    branches = path / f"M2_STRUCTURED_TASK{task}_BRANCHES.csv"
    contexts = path / f"M2_STRUCTURED_TASK{task}_CONTEXTS.csv"
    if not (proto.exists() and branches.exists() and contexts.exists()):
        return False
    p = json.loads(proto.read_text())
    b = read_csv(branches)
    c = read_csv(contexts)
    # root7040 was intentionally stopped after its exact 3-context/15-branch
    # pass when a duplicate-root launch was detected.  Treat that directory as
    # recovered only because its complete expected artifact cardinality is
    # present; no incomplete RUNNING directory is promoted by this rule.
    recovered_pass = p.get("status") == "RUNNING" and len(c) == 3 and len(b) == 15 and len(list((path / "telemetry").glob("*.csv"))) == 15
    return (p.get("status") == "COMPLETED" or recovered_pass) and len(c) == 3 and len(b) == 15 and all(int(x["query_valid"]) for x in c)


def f(row, key):
    return float(row[key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", type=int, default=2)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dirs = sorted(a.root.glob("M2_STRUCTURED_TASK*_ROOT*"))
    selected = [d for d in dirs if complete_dir(d, a.task)]
    rows = []
    branch_rows = []
    for d in selected:
        bs = read_csv(d / f"M2_STRUCTURED_TASK{a.task}_BRANCHES.csv")
        for r in bs:
            # Correct the legacy field emitted by the first worker source.
            r["corrected_full_task_success"] = str(int(int(r["lift_success"]) and int(r["transport_retention"]) and int(r["place_success"]) and not int(r["dropped"])))
            branch_rows.append(r)
    if branch_rows:
        groups = {}
        for r in branch_rows:
            groups.setdefault((r["root_seed"], r["mass_band"]), []).append(r)
        for (root_seed, band), rs in sorted(groups.items()):
            ys = [int(r["corrected_full_task_success"]) for r in rs]
            ts = [int(r["transport_retention"]) for r in rs]
            rows.append({"root_seed": root_seed, "mass_band": band, "mass_kg": rs[0]["mass_kg"], "n": len(rs), "lift_sr": np.mean([int(r["lift_success"]) for r in rs]), "transport_sr": np.mean(ts), "full_task_sr": np.mean(ys), "force_min_N": min(f(r, "requested_force_N") for r in rs), "force_max_N": max(f(r, "requested_force_N") for r in rs)})
        by_force = {}
        for r in branch_rows:
            by_force.setdefault((r["mass_band"], r["requested_force_N"]), []).append(r)
        force_rows = []
        for (band, force), rs in sorted(by_force.items()):
            force_rows.append({"mass_band": band, "mass_kg": rs[0]["mass_kg"], "requested_force_N": force, "n": len(rs), "lift_sr": np.mean([int(r["lift_success"]) for r in rs]), "transport_sr": np.mean([int(r["transport_retention"]) for r in rs]), "full_task_sr": np.mean([int(r["corrected_full_task_success"]) for r in rs])})
        fields = list(force_rows[0])
        with (a.out / "MASS_STRUCTURED_FORCE_SUMMARY.csv").open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(force_rows)
        sensitivity = int(len({r["corrected_full_task_success"] for r in branch_rows}) > 1 or len({r["transport_retention"] for r in branch_rows}) > 1)
        mass_sensitivity = int(any(len({x["corrected_full_task_success"] for x in rs}) > 1 for rs in by_force.values()))
    else:
        sensitivity = mass_sensitivity = 0
    summary = {"status": "COMPLETED" if selected else "NO_COMPLETE_OUTPUTS", "selected_directories": [str(x) for x in selected], "n_roots": len(selected), "n_contexts": len(branch_rows) // 5, "n_branches": len(branch_rows), "force_sensitive": sensitivity, "mass_sensitive_at_fixed_force": mass_sensitivity}
    (a.out / "MASS_STRUCTURED_QUALIFICATION_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (a.out / "MASS_STRUCTURED_QUALIFICATION_REPORT.md").open("w", encoding="utf-8") as h:
        h.write("# Structured mass-task qualification\n\n")
        h.write("Outcome is transport retention plus placement; partial or non-completed worker directories are excluded.\n\n")
        h.write(f"- Complete root workers: {len(selected)}\n- Branches: {len(branch_rows)}\n- Force sensitivity: {bool(sensitivity)}\n- Mass sensitivity at fixed requested force: {bool(mass_sensitivity)}\n\n")
        if branch_rows:
            h.write("## Force summary\n\n| Mass band | Mass (kg) | Force (N) | n | Lift SR | Transport SR | Full-task SR |\n|---|---:|---:|---:|---:|---:|---:|\n")
            for r in force_rows:
                h.write(f"| {r['mass_band']} | {float(r['mass_kg']):.2f} | {float(r['requested_force_N']):.1f} | {r['n']} | {r['lift_sr']:.3f} | {r['transport_sr']:.3f} | {r['full_task_sr']:.3f} |\n")
    # Preserve a flat branch artifact for downstream M3 scripts.
    if branch_rows:
        fields = list(branch_rows[0])
        with (a.out / "MASS_STRUCTURED_BRANCHES_CORRECTED.csv").open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(branch_rows)


if __name__ == "__main__":
    main()
