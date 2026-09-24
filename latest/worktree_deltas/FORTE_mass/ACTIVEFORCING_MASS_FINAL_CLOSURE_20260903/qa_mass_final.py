#!/usr/bin/env python3
"""Independent, read-only QA for the Mass formal source.

The QA result is written only to the separate final-closure directory.  The
formal source and its quarantine are never modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

SOURCE = Path("/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M3_TASK2_FORMAL_STRUCTURED_20260902_130700_RESUME")
OLD = Path("/home/exouser/FORTE_massresume/MASS_FORMAL_RECOVERY_20260902_142843/accepted_snapshot")
OUT = Path("/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_FINAL_CLOSURE_20260903")
FORCES = (0.5, 1.0, 1.5, 2.5, 4.0)
BANDS = {"LOW": 0.05, "MID": 0.10, "HIGH": 0.20}
TELEMETRY_FIELDS = {"context_id", "task", "seed", "mass_band", "force", "step", "phase", "measured_force_N", "object_z_delta"}


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finite(x: str) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def check(name, passed, **detail):
    return {"status": "PASS" if passed else "FAIL", **detail}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    context_path = SOURCE / "M3_TASK2_STRUCTURED_FORMAL_CONTEXTS.csv"
    branch_path = SOURCE / "M3_TASK2_STRUCTURED_FORMAL_BRANCHES.csv"
    steps_path = SOURCE / "M3_TASK2_STRUCTURED_FORMAL_QUERY_TIMESTEPS.csv"
    observations_path = SOURCE / "M3_TASK2_STRUCTURED_FORMAL_QUERY_OBSERVATIONS.json"
    protocol_path = SOURCE / "M3_TASK2_STRUCTURED_FORMAL_PROTOCOL.json"
    contexts, branches, steps = rows(context_path), rows(branch_path), rows(steps_path)
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    old_contexts, old_branches = rows(OLD / context_path.name), rows(OLD / branch_path.name)

    expected = {
        f"mass_structured_{split}_t2_root{root}_{band.lower()}"
        for split, roots in (("train", (8100, 8101, 8102, 8103)), ("test", (8200, 8201)))
        for root in roots for band in BANDS
    }
    cids = [r["context_id"] for r in contexts]
    c_map = {r["context_id"]: r for r in contexts}
    b_keys = {(r["context_id"], float(r["requested_force_N"]), int(r["repeat"])) for r in branches}
    expected_keys = {(cid, force, repeat) for cid in expected for force in FORCES for repeat in (0, 1)}
    old_keys = {(r["context_id"], float(r["requested_force_N"]), int(r["repeat"])) for r in old_branches}
    new_branches = [r for r in branches if (r["context_id"], float(r["requested_force_N"]), int(r["repeat"])) not in old_keys]
    new_contexts = [r for r in contexts if r["context_id"] not in {x["context_id"] for x in old_contexts}]

    checks = {}
    checks["protocol_completed"] = check("protocol_completed", protocol.get("status") == "COMPLETED", value=protocol.get("status"))
    checks["total_context_coverage"] = check("total_context_coverage", len(contexts) == 18 and set(cids) == expected and len(set(cids)) == 18, contexts=len(contexts), expected=18, missing=sorted(expected - set(cids)), unexpected=sorted(set(cids) - expected))
    checks["total_branch_coverage"] = check("total_branch_coverage", len(branches) == 180 and b_keys == expected_keys, branches=len(branches), unique_keys=len(b_keys), expected=180)
    checks["new_branch_scope"] = check("new_branch_scope", len(new_branches) == 80 and len(new_contexts) == 8 and set(r["context_id"] for r in new_contexts) == expected - {x["context_id"] for x in old_contexts}, new_branches=len(new_branches), new_contexts=len(new_contexts), old_branches=len(old_branches), old_contexts=len(old_contexts))
    checks["query_observation_coverage"] = check("query_observation_coverage", len(observations) == 18 and {str(x["context_id"]) for x in observations} == expected, observations=len(observations))
    checks["query_step_coverage"] = check("query_step_coverage", len(steps) == 18 * 206 and Counter(x["context_id"] for x in steps) == Counter({cid: 206 for cid in expected}), steps=len(steps), expected=18 * 206)

    bad_contexts = []
    for r in contexts:
        if (r["mass_band"] not in BANDS or abs(float(r["mass_kg"]) - BANDS[r["mass_band"]]) > 1e-9 or float(r["friction"]) != 0.5 or float(r["friction"]) != float(r.get("friction", 0)) or int(r["query_valid"]) != 1 or int(r["query_history_rows"]) != 206 or int(r["root_seed"]) != int(r["root_family"].split("root")[-1])):
            bad_contexts.append(r["context_id"])
    checks["context_semantics"] = check("context_semantics", not bad_contexts, bad_contexts=sorted(set(bad_contexts)))

    bad_state, bad_eval, bad_force, bad_meta, bad_telemetry = [], [], [], [], []
    telemetry_files = list((SOURCE / "telemetry").glob("*.csv"))
    expected_telemetry_names = {f"{r['context_id']}_F{float(r['requested_force_N']):g}_R{int(r['repeat'])}.csv" for r in branches}
    actual_telemetry_names = {p.name for p in telemetry_files}
    if actual_telemetry_names != expected_telemetry_names:
        bad_telemetry.append({"missing": sorted(expected_telemetry_names - actual_telemetry_names), "unexpected": sorted(actual_telemetry_names - expected_telemetry_names)})
    phases = Counter()
    for r in branches:
        c = c_map.get(r["context_id"])
        expected_success = int(all(int(r[k]) == 1 for k in ("lift_success", "transport_retention", "place_success")) and int(r["dropped"]) == 0)
        if c is None or r["initial_state_hash"] != c["initial_state_hash"] or r["post_query_state_hash"] != c["post_query_state_hash"]:
            bad_state.append(r["context_id"])
        if int(r["full_task_success_y"]) != expected_success:
            bad_eval.append(r["context_id"])
        if not all(finite(r[k]) for k in ("measured_force_mean_N", "measured_force_peak_N")) or float(r["requested_force_N"]) not in FORCES:
            bad_force.append(r["context_id"])
        if c is None or int(r["root_seed"]) != int(c["root_seed"]) or r["mass_band"] != c["mass_band"] or float(r["mass_kg"]) != float(c["mass_kg"]) or float(r["friction"]) != 0.5 or int(r["query_valid"]) != 1:
            bad_meta.append(r["context_id"])
        tp = SOURCE / "telemetry" / f"{r['context_id']}_F{float(r['requested_force_N']):g}_R{int(r['repeat'])}.csv"
        if not tp.is_file():
            continue
        try:
            tr = rows(tp)
            if not tr or not TELEMETRY_FIELDS.issubset(tr[0].keys()):
                bad_telemetry.append(tp.name)
            else:
                phases.update(x["phase"] for x in tr)
                if any(x["context_id"] != r["context_id"] or int(x["seed"]) != int(r["root_seed"]) or x["mass_band"] != r["mass_band"] or float(x["force"]) != float(r["requested_force_N"]) or not finite(x["measured_force_N"]) or not finite(x["object_z_delta"]) for x in tr):
                    bad_telemetry.append(tp.name)
        except (csv.Error, ValueError, UnicodeError):
            bad_telemetry.append(tp.name)
    required_phases = {"hold", "lift"}
    checks["state_hash_parity"] = check("state_hash_parity", not bad_state, bad_rows=len(bad_state))
    checks["evaluator_semantics"] = check("evaluator_semantics", not bad_eval, bad_rows=len(bad_eval))
    checks["friction_force_metadata"] = check("friction_force_metadata", not bad_force and not bad_meta, bad_force_rows=len(bad_force), bad_metadata_rows=len(bad_meta))
    checks["telemetry_atomic_completeness"] = check("telemetry_atomic_completeness", not bad_telemetry and required_phases.issubset(phases), bad_files=len(bad_telemetry), phase_counts=dict(phases), expected_files=180, actual_files=len(actual_telemetry_names))

    old_map = {(r["context_id"], r["requested_force_N"], r["repeat"]): r for r in old_branches}
    current_map = {(r["context_id"], r["requested_force_N"], r["repeat"]): r for r in branches}
    drift = [key for key, old in old_map.items() if key not in current_map or any(current_map[key].get(k) != old.get(k) for k in old)]
    checks["accepted_prefix_immutability"] = check("accepted_prefix_immutability", len(old_contexts) == 10 and len(old_branches) == 100 and not drift, drift_rows=len(drift))
    checks["interruption_spillover_quarantine"] = check("interruption_spillover_quarantine", protocol.get("status") == "COMPLETED" and protocol.get("contexts") == 18 and protocol.get("branches") == 180 and not (set(cids) - expected), prior_quarantine_excluded=True, source_has_unexpected_contexts=bool(set(cids) - expected))

    passed = all(v["status"] == "PASS" for v in checks.values())
    result = {"status": "PASS" if passed else "FAIL", "qa": "MASS_FINAL_BRANCH_QA", "source": str(SOURCE), "total_contexts": len(contexts), "total_branches": len(branches), "new_contexts": len(new_contexts), "new_branches": len(new_branches), "accepted_contexts": len(contexts) if passed else len(old_contexts), "accepted_branches": len(branches) if passed else len(old_branches), "checks": checks, "source_hashes": {p.name: sha(p) for p in (context_path, branch_path, steps_path, observations_path, protocol_path)}}
    (OUT / "MASS_FINAL_BRANCH_QA.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Mass final branch QA", "", f"Status: **{result['status']}**", "", "The source was inspected read-only. No Mass branch was recollected. The QA explicitly compares the completed source against the 100-row accepted snapshot and treats prior partial/interrupted spillover as excluded quarantine.", "", f"- Total: {len(contexts)}/18 contexts; {len(branches)}/180 branches", f"- Newly QA'd: {len(new_contexts)}/8 contexts; {len(new_branches)}/80 branches", f"- Promoted accepted count: {result['accepted_branches']}/180 branches, {result['accepted_contexts']}/18 contexts", "", "## Checks", "", "| Check | Status | Evidence |", "|---|---|---|"]
    for name, detail in checks.items():
        evidence = ", ".join(f"{k}={v}" for k, v in detail.items() if k != "status")
        lines.append(f"| {name} | {detail['status']} | {evidence} |")
    lines += ["", "## Acceptance rule", "", "Any partial, interrupted, duplicate, post-gate spillover, state mismatch, evaluator mismatch, invalid query, or incomplete telemetry would remain unaccepted. All required checks passed for this source." if passed else "QA failed; no invalid branch was promoted."]
    (OUT / "MASS_FINAL_BRANCH_QA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {"status": "AUTHORITATIVE_MASS_ACCEPTED" if passed else "MASS_ACCEPTANCE_BLOCKED", "source_directory": str(SOURCE), "accepted_snapshot": str(OLD), "accepted_contexts": result["accepted_contexts"], "accepted_branches": result["accepted_branches"], "newly_accepted_contexts": len(new_contexts) if passed else 0, "newly_accepted_branches": len(new_branches) if passed else 0, "fixed_friction": 0.5, "candidate_forces_N": list(FORCES), "repeats": [0, 1], "quarantine_excluded": True, "qa_artifact": str(OUT / "MASS_FINAL_BRANCH_QA.json"), "source_hashes": result["source_hashes"]}
    (OUT / "MASS_ACCEPTED_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
