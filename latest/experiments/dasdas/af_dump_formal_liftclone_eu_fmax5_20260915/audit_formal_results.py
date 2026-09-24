from __future__ import annotations

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


P4_FROZEN_NAMES = (
    "original_raw_rows.json",
    "patch_readbacks.json",
    "qualification.json",
    "original58_engineering.npy",
    "original_probe_summary.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def audit_query(query: Path) -> dict:
    query = Path(query)
    report = read(query / "qualification.json")
    shear = read(query / "DUMP_SHEAR_QUERY.json")
    grasp = read(query / "ESTABLISHED_GRASP.json")
    preserved = read(query / "P4_PRESERVED.json")
    admission = read(query / "ONLINE_QUERY_ADMISSION.json")
    raw = read(query / "original_raw_rows.json")
    if not report.get("completed") or report.get("probe_failure"):
        raise ValueError("Original P4 query did not complete")
    if "original P4" not in str(report.get("scope", "")).lower() and "P4" not in str(report.get("scope", "")):
        raise ValueError("qualification.json is not original P4")
    if report.get("protocol") == "lift_clone_established_grasp_then_dump_shear_query":
        raise ValueError("P4 qualification was overwritten by the dump-shear protocol")
    if not raw or "probe_phase" not in raw[0] or raw[0].get("task_id") != "dump_bin_bigbin":
        raise ValueError("original_raw_rows is not original P4 evidence")
    for name in P4_FROZEN_NAMES:
        expected = preserved["frozen_p4_hashes"][name]
        if sha(query / name) != expected:
            raise ValueError("Frozen P4 artifact changed after preserve: " + name)
    if not admission.get("admitted"):
        raise ValueError("Original P4 query was not admitted")
    if shear.get("schema_id") != "ROBOTWIN_SHEAR_QUERY_OBSERVABLES_V1":
        raise ValueError("Missing dump shear query")
    if not shear.get("final_bilateral_contact"):
        raise ValueError("Dump shear query lost contact")
    if grasp.get("grasp_force_N") != 12.0:
        raise ValueError("Established grasp force cap is not 12 N")
    if grasp.get("arm") not in ("left", "right"):
        raise ValueError("Established grasp arm is missing")
    if grasp.get("skipped_script") is False and grasp.get("arm") != "left":
        raise ValueError("Scripted re-grasp must finish in the left hand")
    if not (query / "original_squeeze_inner_trace.json").exists():
        raise ValueError("Missing inner-force seed trace")
    if not (query / "P4_native_controls.json").exists():
        raise ValueError("P4 native_controls copy missing")
    return {
        "passed": True,
        "query_kind": "original_p4_belief_then_dump_shear_after_12N_grasp",
        "p4_not_overwritten": True,
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
    decision = read(job / "PREACTION_AF_DECISION.json")
    feature = read(job / "PREACTION_FEATURE.json")
    if result["selection_lock_sha256"] != sha(job / "PREACTION_SELECTION_LOCK.json"):
        raise ValueError("Selection lock changed")
    online = read(job / "online_qualification.json")
    if online["first_chunk_hashes"] != [lock["first_chunk_sha256"]] * len(context["methods"]):
        raise ValueError("Unpaired first chunks")
    if lock.get("handoff_source") != "last gripper command after dump shear query":
        raise ValueError("Handoff is not the lift-clone last-command source")
    if lock.get("belief_source") != "frozen dump v4 on original P4 rows":
        raise ValueError("Belief is not frozen dump v4 on original P4")
    if feature.get("source") != "ONLINE_VLA_ACTION_CHUNK":
        raise ValueError("Selection feature is not the live π0 chunk")
    if decision.get("executed_selector") != "expected_utility":
        raise ValueError("Executed selector is not expected_utility")
    if lock.get("decision_executed_selector") != "expected_utility":
        raise ValueError("Sealed executed selector is not expected_utility")
    commanded = float(lock["commanded_forces_N"][0])
    selected = float(decision["selected_force_N"])
    if abs(commanded - selected) > 1e-9:
        raise ValueError("Executed force differs from EU pick")
    if not (0.25 - 1e-9 <= commanded <= 5.0 + 1e-9):
        raise ValueError(f"Executed force outside [0.25, 5]: {commanded}")
    if lock.get("force_clamped_to_0p5"):
        raise ValueError("EU pick was clamped to 0.5 N")
    branches = []
    for index, method in enumerate(context["methods"]):
        matches = list(job.glob(f"branch_{index}_*"))
        if len(matches) != 1:
            raise ValueError("Missing formal branch")
        branch = matches[0]
        if method != "ActiveForcing":
            raise ValueError("Unexpected method in AF-only lift-clone EU queue")
        branches.append({"method": method, "audit": audit_branch(branch), "result": read(branch / "result.json")})
    return {
        "passed": True,
        "context": context,
        "query_audit": query,
        "branches": branches,
        "first_chunk_pairing_exact": True,
        "common_handoff_state_sha256": lock["state_sha256"],
        "formal_context_result_sha256": sha(job / "FORMAL_CONTEXT_RESULT.json"),
        "executed_selector": "expected_utility",
        "commanded_force_N": commanded,
        "audit_source_sha256": sha(Path(__file__)),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_context(args.case.resolve()), indent=2))
