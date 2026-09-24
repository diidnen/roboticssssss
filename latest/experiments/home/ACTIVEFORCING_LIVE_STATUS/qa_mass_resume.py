#!/usr/bin/env python3
"""Independent, read-only QA for the completed Mass formal resume."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


SOURCE = Path("/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M3_TASK2_FORMAL_STRUCTURED_20260902_130700_RESUME")
OLD = Path("/home/exouser/FORTE_massresume/MASS_FORMAL_RECOVERY_20260902_142843/accepted_snapshot")
OUT = SOURCE / "MASS_RESUME_INDEPENDENT_QA.json"
FORCES = {0.5, 1.0, 1.5, 2.5, 4.0}
BANDS = {"LOW": 0.05, "MID": 0.10, "HIGH": 0.20}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finite(value: str) -> bool:
    return math.isfinite(float(value))


def fail(checks, name, detail):
    checks[name] = {"status": "FAIL", "detail": detail}


def main() -> int:
    checks = {}
    context_path = SOURCE / "M3_TASK2_STRUCTURED_FORMAL_CONTEXTS.csv"
    branch_path = SOURCE / "M3_TASK2_STRUCTURED_FORMAL_BRANCHES.csv"
    steps_path = SOURCE / "M3_TASK2_STRUCTURED_FORMAL_QUERY_TIMESTEPS.csv"
    observations_path = SOURCE / "M3_TASK2_STRUCTURED_FORMAL_QUERY_OBSERVATIONS.json"
    protocol_path = SOURCE / "M3_TASK2_STRUCTURED_FORMAL_PROTOCOL.json"
    contexts = read_csv(context_path)
    branches = read_csv(branch_path)
    steps = read_csv(steps_path)
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    checks["protocol_completed"] = {"status": "PASS" if protocol.get("status") == "COMPLETED" else "FAIL", "status_value": protocol.get("status")}
    expected_contexts = {f"mass_structured_{split}_t2_root{root}_{band.lower()}" for split, roots in (("train", (8100, 8101, 8102, 8103)), ("test", (8200, 8201))) for root in roots for band in BANDS}
    actual_contexts = [r["context_id"] for r in contexts]
    checks["context_coverage"] = {"status": "PASS" if len(contexts) == 18 and set(actual_contexts) == expected_contexts and len(set(actual_contexts)) == 18 else "FAIL", "contexts": len(contexts), "expected": 18, "unexpected": sorted(set(actual_contexts) - expected_contexts), "missing": sorted(expected_contexts - set(actual_contexts))}
    checks["observation_coverage"] = {"status": "PASS" if len(observations) == 18 and {str(x["context_id"]) for x in observations} == expected_contexts else "FAIL", "observations": len(observations)}
    checks["query_step_coverage"] = {"status": "PASS" if len(steps) == 18 * 206 and Counter(x["context_id"] for x in steps) == Counter({cid: 206 for cid in expected_contexts}) else "FAIL", "steps": len(steps), "expected": 18 * 206}

    by_context = Counter(r["context_id"] for r in branches)
    keys = [(r["context_id"], float(r["requested_force_N"]), int(r["repeat"])) for r in branches]
    expected_keys = {(cid, force, repeat) for cid in expected_contexts for force in FORCES for repeat in (0, 1)}
    checks["branch_parity_and_duplicates"] = {"status": "PASS" if len(branches) == 180 and len(set(keys)) == 180 and set(keys) == expected_keys and all(by_context[cid] == 10 for cid in expected_contexts) else "FAIL", "branches": len(branches), "unique_keys": len(set(keys)), "expected": 180}

    context_map = {r["context_id"]: r for r in contexts}
    bad_context = []
    for r in contexts:
        if int(r["query_valid"]) != 1 or r["mass_band"] not in BANDS or abs(float(r["mass_kg"]) - BANDS[r["mass_band"]]) > 1e-9 or float(r["friction"]) != 0.5 or int(r["query_history_rows"]) != 206:
            bad_context.append(r["context_id"])
    bad_state = []
    bad_eval = []
    bad_force = []
    bad_telemetry = []
    for r in branches:
        c = context_map.get(r["context_id"])
        expected_success = int(all(int(r[k]) == 1 for k in ("lift_success", "transport_retention", "place_success")) and int(r["dropped"]) == 0)
        if c is None or r["initial_state_hash"] != c["initial_state_hash"] or r["post_query_state_hash"] != c["post_query_state_hash"]:
            bad_state.append(r["context_id"])
        if int(r["full_task_success_y"]) != expected_success:
            bad_eval.append(r["context_id"])
        if not finite(r["measured_force_mean_N"]) or not finite(r["measured_force_peak_N"]) or float(r["requested_force_N"]) not in FORCES:
            bad_force.append(r["context_id"])
        telem = SOURCE / "telemetry" / f"{r['context_id']}_F{float(r['requested_force_N']):g}_R{int(r['repeat'])}.csv"
        if not telem.is_file():
            bad_telemetry.append(str(telem))
        else:
            rows = read_csv(telem)
            if not rows or set(("context_id", "task", "seed", "mass_band", "force", "step", "phase", "measured_force_N", "object_z_delta")) - set(rows[0]):
                bad_telemetry.append(str(telem))
            else:
                if any(x["context_id"] != r["context_id"] or float(x["force"]) != float(r["requested_force_N"]) or not finite(x["measured_force_N"]) for x in rows):
                    bad_telemetry.append(str(telem))
    checks["context_semantics"] = {"status": "PASS" if not bad_context else "FAIL", "bad_contexts": sorted(set(bad_context))}
    checks["state_hash_parity"] = {"status": "PASS" if not bad_state else "FAIL", "bad_rows": len(bad_state)}
    checks["evaluator_semantics"] = {"status": "PASS" if not bad_eval else "FAIL", "bad_rows": len(bad_eval)}
    checks["force_telemetry"] = {"status": "PASS" if not bad_force and not bad_telemetry else "FAIL", "bad_force_rows": len(bad_force), "bad_telemetry_files": len(bad_telemetry)}

    old_contexts = read_csv(OLD / "M3_TASK2_STRUCTURED_FORMAL_CONTEXTS.csv")
    old_branches = read_csv(OLD / "M3_TASK2_STRUCTURED_FORMAL_BRANCHES.csv")
    old_context_ids = {r["context_id"] for r in old_contexts}
    new_old_contexts = {r["context_id"] for r in contexts if r["context_id"] in old_context_ids}
    old_branch_map = {(r["context_id"], r["requested_force_N"], r["repeat"]): r for r in old_branches}
    current_branch_map = {(r["context_id"], r["requested_force_N"], r["repeat"]): r for r in branches}
    compare_fields = [k for k in old_branches[0] if k in current_branch_map[next(iter(current_branch_map))]] if old_branches else []
    drift = []
    for key, old_row in old_branch_map.items():
        cur = current_branch_map.get(key)
        if cur is None or any(cur[k] != old_row[k] for k in compare_fields):
            drift.append(key)
    checks["accepted_prefix_immutability"] = {"status": "PASS" if len(old_contexts) == 10 and len(old_branches) == 100 and len(new_old_contexts) == 10 and not drift else "FAIL", "old_contexts": len(old_contexts), "old_branches": len(old_branches), "drift_rows": len(drift)}
    checks["interruption_spillover"] = {"status": "PASS" if protocol.get("status") == "COMPLETED" and int(protocol.get("contexts", -1)) == 18 and int(protocol.get("branches", -1)) == 180 and not (set(actual_contexts) - expected_contexts) else "FAIL", "collector_status": protocol.get("status"), "protocol_contexts": protocol.get("contexts"), "protocol_branches": protocol.get("branches")}

    passed = all(v["status"] == "PASS" for v in checks.values())
    result = {"status": "PASS" if passed else "FAIL", "qa": "MASS_RESUME_INDEPENDENT_QA", "source": str(SOURCE), "source_hashes": {p.name: sha(p) for p in (context_path, branch_path, steps_path, observations_path, protocol_path)}, "checks": checks}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
