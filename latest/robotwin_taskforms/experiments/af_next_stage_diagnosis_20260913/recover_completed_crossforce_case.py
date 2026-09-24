"""Seal completed two-branch evidence after the original six-branch gate rejects it.

The original execution worker correctly completed both requested branches, then its
unchanged V4 qualification wrapper rejected a two-branch result because it expects
the historical six-method TEST.  This recovery only audits and seals already-written
artifacts; it never runs or modifies a trajectory.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "af_dump_original_restore_20260912"
sys.path.insert(0, str(OLD))
from audit_original_collected_group import audit_branch, audit_query
from rootlocal_collection_contract import now, read, sha, write


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recover(case_id: str):
    protocol = read(HERE / "CROSSFORCE_PROTOCOL.json")
    case = next(c for c in protocol["cases"] if c["id"] == case_id)
    output = HERE / ("cross_" + case_id)
    job = output / "job"
    exit_record = read(output / "EXIT.json")
    log = (output / "worker.log").read_text(errors="replace")
    expected_gate = "ValueError: Incomplete paired online TEST" in log
    qualification = read(job / "online_qualification.json")
    branches = sorted(job.glob("branch_*"))
    if len(branches) != 2 or len(qualification["results"]) != 2:
        raise RuntimeError("The two declared branches are not complete")
    seal = read(job / "PREACTION_SELECTION_LOCK.json")
    historical_lock = read(case["historical_lock"])
    state_exact = seal["state_sha256"] == historical_lock["state_sha256"]
    chunk_exact = seal["first_chunk_sha256"] == historical_lock["first_chunk_sha256"]
    outcomes = []
    branch_audits = []
    for index, (branch, force, method) in enumerate(zip(branches, case["forces_N"], case["methods"])):
        result_path = branch / "result.json"
        result = read(result_path)
        if abs(result["force_setpoint_bilateral_n"] - force) > 1e-9:
            raise RuntimeError("Force mismatch")
        qualified_result = qualification["results"][index]["result"]
        if result["trace_sha256"] != qualified_result["trace_sha256"] or result["success"] != qualified_result["success"]:
            raise RuntimeError("Qualification/result mismatch")
        outcome = {
            "method": method,
            "force_N": force,
            "success": int(result["success"]),
            "original_utility": (8.0 - force) / 8.0 if result["success"] else -1.0,
            "actual_mean_force_N": result["measured_mean_squeeze_n"],
            "native_actions": result["native_actions"],
            "result_sha256": file_sha(result_path),
            "trace_sha256": result["trace_sha256"],
        }
        outcomes.append(outcome)
        branch_audits.append(audit_branch(branch))
    historical = read(case["historical_result"])["outcomes"]
    matches = []
    for new in outcomes:
        for old in historical:
            if old["force_N"] == new["force_N"]:
                matches.append(
                    {
                        "force_N": new["force_N"],
                        "old_method": old["method"],
                        "old_success": old["success"],
                        "new_success": new["success"],
                        "old_result_sha256": old["result_sha256"],
                        "new_result_sha256": new["result_sha256"],
                        "result_exact": old["result_sha256"] == new["result_sha256"],
                    }
                )
    query_audit = audit_query(job / "query")
    passed = (
        exit_record["exit_code"] == 1
        and not exit_record["disk_guard_stop"]
        and expected_gate
        and state_exact
        and chunk_exact
        and query_audit["passed"]
        and all(item["passed"] for item in branch_audits)
    )
    audit = {
        "passed": passed,
        "id": case_id,
        "state_exact": state_exact,
        "first_chunk_exact": chunk_exact,
        "outcomes": outcomes,
        "historical_matches": matches,
        "query_audit": query_audit,
        "branch_audits": branch_audits,
        "worker_exit": exit_record,
        "expected_post_execution_gate_rejection": expected_gate,
        "recovery_scope": "Audit/seal already-completed branches only; no simulator execution and no source-result edits.",
        "recovery_script_sha256": sha(__file__),
        "finished_utc": now(),
    }
    write(output / "CROSSFORCE_AUDIT.json", audit)
    if not passed:
        raise RuntimeError("Recovery audit failed")
    print(json.dumps({"passed": passed, "id": case_id, "outcomes": outcomes, "historical_matches": matches}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id")
    args = parser.parse_args()
    recover(args.case_id)
