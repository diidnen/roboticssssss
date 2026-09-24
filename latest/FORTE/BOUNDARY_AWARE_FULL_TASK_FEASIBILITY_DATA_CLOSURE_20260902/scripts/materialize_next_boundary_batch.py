#!/usr/bin/env python3
"""Materialize one adaptive 48-branch batch only after force preflight passes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "BOUNDARY_AWARE_FULL_TASK_FEASIBILITY_DATA_CLOSURE_20260902"
NEXT = OUT / "BOUNDARY_NEXT_QUERY_STATE.csv"
CAL = OUT / "MINIMAL_FORCE_INTERFACE_CALIBRATION_PASS.json"
SOURCE = ROOT / "gnp_style_continuous_20260830_125107/GNP_STYLE_CONTINUOUS_TRAINING_PROTOCOL.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration_gate() -> tuple[bool, list[str]]:
    if not CAL.exists():
        return False, ["MINIMAL_FORCE_INTERFACE_CALIBRATION_PASS.json missing"]
    c = json.loads(CAL.read_text())
    reasons = []
    if c.get("status") != "PASS": reasons.append(f"status={c.get('status')}")
    if float(c.get("lowest_stable_force_N", 99)) > 1.0: reasons.append("1.0 N lower guard not stable")
    if float(c.get("tracking_mae_max_N", 99)) > 0.4: reasons.append("tracking MAE > 0.4 N")
    if float(c.get("peak_force_max_N", 99)) > 8.0: reasons.append("peak force > 8 N")
    if not c.get("adjacent_0p25N_distinguishable", False): reasons.append("0.25 N adjacency not distinguishable")
    if int(c.get("runtime_errors", 1)) != 0: reasons.append("runtime errors present")
    return not reasons, reasons


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--round", type=int, required=True); args = ap.parse_args()
    passed, reasons = calibration_gate()
    if not passed:
        print(json.dumps({"status": "REFUSED", "reasons": reasons}, indent=2)); raise SystemExit(75)
    nxt = pd.read_csv(NEXT)
    if len(nxt) != 24 or nxt.context_id.nunique() != 24:
        raise RuntimeError("next-query state is not the frozen 24-context qualification")
    source = json.loads(SOURCE.read_text())
    lookup = {}
    for x in source["train_context_population"]:
        key = (int(x["task"]), int(x["root_index"]), str(x["friction_band"]))
        lookup[key] = x
    contexts, mapping, rows = {}, [], []
    for r in nxt.to_dict("records"):
        key = (int(r["task"]), int(r["root_index"]), str(r["friction_band"]))
        source_ctx = lookup[key]
        if abs(float(source_ctx["friction"]) - float(r["friction"])) > 1e-10:
            raise RuntimeError(f"friction mismatch: {key}")
        cid = str(source_ctx["context_id"]); force = float(r["proposed_force_N"])
        specs = []
        for repeat in (1, 2):
            tag = str(force).replace(".", "p")
            label = f"LOWFORCE_BOUNDARY_Q{args.round}_F{tag}_R{repeat}"
            # The inherited runner adds one to repeat_index in its exported row.
            specs.append({"force_N": force, "repeat_index": repeat - 1, "branch_label": label})
            rows.append({
                "task": int(r["task"]), "collector_context_id": cid, "authoritative_context_id": r["context_id"],
                "root_id": source_ctx["root_id"], "root_index": int(r["root_index"]),
                "friction_band": r["friction_band"], "friction": float(r["friction"]), "force_N": force,
                "repeat": repeat, "search_target": r["search_target"], "acquisition_state": r["acquisition_state"],
                "acquisition_type": "BOUNDARY_SEEKING", "round": args.round, "collection_status": "PENDING_RESOURCE_GATE",
            })
        contexts[cid] = specs
        mapping.append({"collector_context_id": cid, "authoritative_context_id": r["context_id"], "task": int(r["task"]), "root_index": int(r["root_index"]), "friction_band": r["friction_band"]})
    target = {
        "status": "FROZEN_AFTER_FORCE_PREFLIGHT_BEFORE_BOUNDARY_OUTCOMES", "round": args.round,
        "contexts": contexts, "expected_contexts": 24, "expected_branches": 48,
        "context_id_mapping": mapping, "acquisition_type": "BOUNDARY_SEEKING",
        "source_hashes": {str(NEXT): sha(NEXT), str(CAL): sha(CAL), str(SOURCE): sha(SOURCE)},
        "utility_used": False, "task_specific_rule_used": False, "candidate_set_expanded_at_runtime": False,
    }
    target_path = OUT / f"BOUNDARY_ROUND{args.round}_TARGET_MANIFEST.json"
    rows_path = OUT / f"BOUNDARY_ROUND{args.round}_PLANNED_BRANCHES.csv"
    target_path.write_text(json.dumps(target, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(rows_path, index=False)
    print(json.dumps({"status": "MATERIALIZED", "target": str(target_path), "branches": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
