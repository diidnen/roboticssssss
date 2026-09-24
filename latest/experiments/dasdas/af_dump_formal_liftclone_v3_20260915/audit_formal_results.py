from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OLD = Path(
    "/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/"
    "experiments/af_dump_original_restore_20260912"
)
sys.path.insert(0, str(OLD))
from audit_original_collected_group import audit_branch


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def audit_query(query: Path) -> dict:
    query = Path(query)
    report = read(query / "qualification.json")
    shear = read(query / "DUMP_SHEAR_QUERY.json")
    grasp = read(query / "ESTABLISHED_GRASP.json")
    if not report.get("completed") or report.get("probe_failure"):
        raise ValueError("Lift-clone query did not complete")
    if shear.get("schema_id") != "ROBOTWIN_SHEAR_QUERY_OBSERVABLES_V1":
        raise ValueError("Missing dump shear query")
    if not shear.get("final_bilateral_contact"):
        raise ValueError("Dump shear query lost contact")
    if grasp.get("grasp_force_N") != 12.0 or grasp.get("arm") != "left":
        raise ValueError("Established grasp is not the 12 N left-hand prefix")
    if not (query / "original_squeeze_inner_trace.json").exists():
        raise ValueError("Missing inner-force seed trace")
    return {
        "passed": True,
        "query_kind": "dump_shear_after_established_grasp",
        "not_original_pregrasp_p4": True,
        "contact_ratio": shear.get("contact_ratio"),
        "audit_source_sha256": sha(Path(__file__)),
    }


def audit_context(case: Path) -> dict:
    job = case / "job"
    result = read(job / "FORMAL_CONTEXT_RESULT.json")
    context = result["context"]
    query = audit_query(job / "query")
    if not query["passed"]:
        raise ValueError("Query audit failed")
    lock = read(job / "PREACTION_SELECTION_LOCK.json")
    if result["selection_lock_sha256"] != sha(job / "PREACTION_SELECTION_LOCK.json"):
        raise ValueError("Selection lock changed")
    online = read(job / "online_qualification.json")
    if online["first_chunk_hashes"] != [lock["first_chunk_sha256"]] * len(context["methods"]):
        raise ValueError("Unpaired first chunks")
    if lock.get("handoff_source") != "last gripper command after dump shear query":
        raise ValueError("Handoff is not the lift-clone last-command source")
    branches = []
    for index, method in enumerate(context["methods"]):
        matches = list(job.glob(f"branch_{index}_*"))
        if len(matches) != 1:
            raise ValueError("Missing formal branch")
        branch = matches[0]
        if method != "ActiveForcing":
            raise ValueError("Unexpected method in AF-only lift-clone queue")
        branches.append({"method": method, "audit": audit_branch(branch), "result": read(branch / "result.json")})
    return {
        "passed": True,
        "context": context,
        "query_audit": query,
        "branches": branches,
        "first_chunk_pairing_exact": True,
        "common_handoff_state_sha256": lock["state_sha256"],
        "formal_context_result_sha256": sha(job / "FORMAL_CONTEXT_RESULT.json"),
        "audit_source_sha256": sha(Path(__file__)),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_context(args.case.resolve()), indent=2))
