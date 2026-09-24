#!/usr/bin/env python3
"""Generate the registry-complete task-form inventory from checked-in LIBERO configs.

The classification is deliberately conservative.  It records repository evidence and
protocol compatibility; it does not use ActiveForcing outcomes.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path("/home/exouser/Tabero")
OUT = ROOT / "analysis/results/task_form_coverage_20260910"
CONFIG = ROOT / "benchmarks/datasets/libero/config"
FIELDS = [
    "task_id", "task_name", "instruction", "manipulated_object",
    "target/receptacle", "supplied_grasp_available", "native_evaluator_available",
    "frozen_VLA_support", "requires_regrasp", "primary_eligibility",
    "exclusion_or_gate_reason", "lift", "long_transport", "lateral_motion",
    "large_reorientation", "constrained_placement", "insertion_like", "extraction",
    "pulling", "surface_contact", "drawer_interaction", "rack_interaction",
    "release_required", "other", "task_form",
]


def names(task: dict) -> list[str]:
    result = []
    for item in task.get("objects", []):
        result.append(item if isinstance(item, str) else item.get("name", ""))
    return result


def classify(suite: str, task: dict) -> dict:
    instruction = task["language_instruction"].lower()
    goals = task.get("goals", [])
    rel = [g for g in goals if "relationship" in g]
    ops = [g for g in goals if "operation" in g]
    manipulated = list(dict.fromkeys(g["ref_obj"] for g in rel))
    targets = list(dict.fromkeys(g["target"] for g in goals if "target" in g))
    multi_independent = len(manipulated) > 1 or (bool(rel) and bool(ops))
    push = instruction.startswith("push ")
    turn = not rel and any(g.get("operation") in {"turnon", "turnoff"} for g in ops)
    drawer_only = not rel and any(g.get("operation") in {"open", "close"} for g in ops)
    book = manipulated == ["black_book_1"] and targets == ["desk_caddy_1"]
    wine_rack = manipulated == ["wine_bottle_1"] and targets == ["wine_rack_1"]
    bowl_drawer = suite == "libero_spatial" and task["task_id"] == 4

    supplied = bool(rel or drawer_only or turn)
    regrasp = multi_independent
    form = "PICK_TRANSPORT_PLACE"
    if book:
        form = "CONSTRAINED_PLACEMENT_OR_INSERTION"
    elif wine_rack:
        form = "REORIENTATION_HEAVY_RACK_PLACEMENT"
    elif bowl_drawer:
        form = "EXTRACTION_THEN_TRANSPORT_PLACE"
    elif drawer_only:
        form = "EXTRACTION_OR_PULLING"
    elif push:
        form = "SURFACE_CONTACT_MANIPULATION"
    elif turn:
        form = "OTHER_MAINTAINED_CONTACT_FORM"
    elif multi_independent:
        form = "MULTI_PHASE_REGRASP_OUT_OF_SCOPE"

    canonical_book = book and suite == "libero_10"
    primary = "ELIGIBLE_PENDING_5_OF_6_FIXED5" if (canonical_book or wine_rack or bowl_drawer) else "NO"
    if canonical_book or wine_rack or bowl_drawer:
        reason = "Official demonstration supplies replayable grasp construction; frozen online VLA is not yet qualified from that handoff."
    elif book:
        reason = "Exact duplicate of the canonical libero_10/task5 book/caddy candidate; retained in inventory but not double-counted."
    elif regrasp:
        reason = "Multiple independent object/interaction phases require regrasp or grasp switching."
    elif push:
        reason = "Native behavior is non-grasp pushing, so the maintained-grasp intervention contract does not apply."
    elif turn:
        reason = "Knob control uses a different contact/hand-control interface and has no object-side friction intervention."
    elif drawer_only:
        reason = "Maintained handle pull is plausible, but the experiment's frozen object-side friction variable is not defined for this articulation."
    else:
        reason = "Compatible but duplicates the established pick-transport-place family; lower diversity value."

    flags = {
        "lift": bool(rel) and not push,
        "long_transport": bool(rel) and not push,
        "lateral_motion": push or drawer_only or bowl_drawer,
        "large_reorientation": wine_rack,
        "constrained_placement": book or wine_rack or (bool(rel) and any(g.get("relationship") == "in" for g in rel)),
        "insertion_like": book,
        "extraction": bowl_drawer,
        "pulling": drawer_only or bowl_drawer,
        "surface_contact": push or drawer_only or bool(rel),
        "drawer_interaction": drawer_only or bowl_drawer or "drawer" in instruction,
        "rack_interaction": wine_rack,
        "release_required": bool(rel),
        "other": turn or multi_independent,
    }
    slug = re.sub(r"[^a-z0-9]+", "_", instruction).strip("_")
    return {
        "task_id": f"{suite}/task{task['task_id']}",
        "task_name": slug,
        "instruction": task["language_instruction"],
        "manipulated_object": ";".join(manipulated) if manipulated else ";".join(names(task)),
        "target/receptacle": ";".join(targets),
        "supplied_grasp_available": "CONSTRUCTIBLE_FROM_OFFICIAL_DEMONSTRATION" if supplied else "NO",
        "native_evaluator_available": "YES:libero_goals_reached",
        "frozen_VLA_support": "UNQUALIFIED_FROM_SUPPLIED_GRASP",
        "requires_regrasp": "YES" if regrasp else "NO",
        "primary_eligibility": primary,
        "exclusion_or_gate_reason": reason,
        **{key: "1" if value else "0" for key, value in flags.items()},
        "task_form": form,
    }


def main() -> None:
    rows = []
    for path in sorted(CONFIG.glob("*.json")):
        payload = json.loads(path.read_text())
        for task in payload["tasks"]:
            rows.append(classify(path.stem, task))
    rows.sort(key=lambda row: (row["task_id"].split("/")[0], int(row["task_id"].split("task")[1])))
    with (OUT / "ALL_TASK_FORM_CANDIDATES.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    selected = [row for row in rows if row["primary_eligibility"].startswith("ELIGIBLE")]
    matrix_fields = ["task_id", "instruction", "task_form"] + FIELDS[11:24] + ["qualification_status"]
    originals = [{
        "task_id": "libero_object/task0,1,5,6",
        "instruction": "Four original grocery-to-basket tasks",
        "task_form": "PICK_TRANSPORT_PLACE",
        **{name: ("1" if name in {"lift", "long_transport", "surface_contact", "release_required"} else "0") for name in FIELDS[11:24]},
        "qualification_status": "AUTHORITATIVE_FINAL_PAPER_SYSTEM",
    }]
    matrix = originals + [{
        **{key: row[key] for key in matrix_fields if key != "qualification_status"},
        "qualification_status": "PENDING_PREDECLARED_5_OF_6_FIXED5_GATE",
    } for row in selected]
    with (OUT / "TASK_FORM_DIVERSITY_MATRIX.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=matrix_fields)
        writer.writeheader()
        writer.writerows(matrix)

    print(json.dumps({"registry_tasks": len(rows), "primary_candidates": [r["task_id"] for r in selected]}, indent=2))


if __name__ == "__main__":
    main()
