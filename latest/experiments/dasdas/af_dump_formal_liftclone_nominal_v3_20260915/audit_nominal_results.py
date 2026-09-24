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


def audit_nominal(branch: Path) -> dict:
    result = read(branch / "result.json")
    actions = read(branch / "arbitration.json")
    if not result["completed"] or result["success"] != result["official_final_check"]:
        raise ValueError("Ambiguous nominal outcome")
    if result["force_setpoint_bilateral_n"] is not None or result["commanded_force_N"] is not None:
        raise ValueError("Nominal branch has a commanded force")
    if result["force_controller_installed"] or result["gripper_force_override"]:
        raise ValueError("Nominal branch used force control")
    if not result.get("original_squeeze_inner_disabled_after_query"):
        raise ValueError("Nominal remainder kept the squeeze inner")
    if not result["success"] and result["native_actions"] != result["native_horizon"]:
        raise ValueError("Nominal failure ended before official horizon")
    chunks = sorted(branch.glob("chunk_*.npy"))
    policy = read(branch / "policy_receipts.json")
    if len(chunks) != result["chunks"] or len(chunks) != len(policy):
        raise ValueError("Nominal chunk/receipt mismatch")
    sequences = []
    for path, receipt in zip(chunks, policy):
        array = np.load(path, allow_pickle=False)
        if array.ndim != 2 or array.shape[1] != 14 or not np.isfinite(array).all():
            raise ValueError("Invalid nominal native actions")
        if hashlib.sha256(array.tobytes()).hexdigest() != receipt["actions_sha256"]:
            raise ValueError("Nominal chunk hash mismatch")
        sequences.append(array)
    executed = np.concatenate(sequences)[: len(actions)]
    np.testing.assert_array_equal(executed, np.asarray([row["native_action14"] for row in actions], np.float32))
    for row in actions:
        if row["raw_vla_arm_command"] != row["final_arm_command"] or row["force_controller_installed"] or row["gripper_force_override"]:
            raise ValueError("Nominal action was overridden")
    digest = hashlib.sha256()
    measured = []
    count = 0
    with gzip.open(branch / "physics_trace.jsonl.gz", "rt") as stream:
        for count, line in enumerate(stream, 1):
            encoded = line.rstrip("\n")
            digest.update(encoded.encode())
            row = json.loads(encoded)
            if row["physics_step"] != count or row["original_squeeze_inner"] is not None or not row["native_nominal_gripper"]:
                raise ValueError("Nominal trace provenance mismatch")
            measured.append(float(row["contact"]["measured_squeeze_n"]))
    if count != result["physics_steps"] or digest.hexdigest() != result["trace_sha256"]:
        raise ValueError("Nominal trace count/hash mismatch")
    if float(np.mean(measured)) != result["measured_mean_squeeze_n"] or float(np.max(measured)) != result["measured_max_squeeze_n"]:
        raise ValueError("Nominal measured-force summary mismatch")
    return {
        "passed": True,
        "method": "Nominal Frozen VLA",
        "actions": len(actions),
        "physics_steps": count,
        "native_action_and_gripper_pass_through_exact": True,
        "force_controller_absent": True,
        "trace_sha256": digest.hexdigest(),
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
    if lock.get("commanded_forces_N") != [None]:
        raise ValueError("Nominal selection lock has a commanded force")
    branches = []
    for index, method in enumerate(context["methods"]):
        matches = list(job.glob(f"branch_{index}_*"))
        if len(matches) != 1:
            raise ValueError("Missing formal branch")
        branch = matches[0]
        if method != "Nominal Frozen VLA":
            raise ValueError("Unexpected method in Nominal-only lift-clone queue")
        branches.append({"method": method, "audit": audit_nominal(branch), "result": read(branch / "result.json")})
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
