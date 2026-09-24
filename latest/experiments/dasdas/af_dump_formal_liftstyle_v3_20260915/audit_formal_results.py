from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = Path(
    "/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/"
    "experiments/af_dump_original_restore_20260912"
)
sys.path.insert(0, str(OLD))
from audit_original_collected_group import audit_branch, audit_query


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


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
    branches = []
    for index, method in enumerate(context["methods"]):
        matches = list(job.glob(f"branch_{index}_*"))
        if len(matches) != 1:
            raise ValueError("Missing formal branch")
        branch = matches[0]
        if method != "ActiveForcing":
            raise ValueError("Unexpected method in AF-only lift-style queue")
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
