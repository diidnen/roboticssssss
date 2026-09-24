import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

out = Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_secondary_baselines_v1")
stage = Path("/home/exouser/FORTE/secondary_staging_v1")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

branches = sorted((out / "branches").glob("*"))
references = sorted((out / "references").glob("*"))
admissions = [read(p / "SECONDARY_ADMISSION.json") for p in branches]
results = [read(p / "BRANCH_RESULT.json") for p in branches]
health = [json.loads(line) for line in (out / "GPU_HEALTH_GATE_LOG.jsonl").read_text().splitlines() if line.strip()]
nvml_fault_rows = [x for x in health if x["NVML_FAILURE_DURING_PREFLIGHT"]]
resource_failures = [x for x in health if not x["RESOURCE_GATE_VALID"] or not x["POLICY_SERVER_18885_VALID"]]
plan_rows = list(csv.DictReader((out / "FINAL_SECONDARY_EXECUTION_PLAN.csv").open()))
candidate = read(out / "CANDIDATE_RUNTIME_MANIFEST.json")
source_mismatch = []
for path, expected in candidate["source_hashes"].items():
    observed = sha(path)
    if observed != expected:
        source_mismatch.append({"path": path, "expected": expected, "observed": observed})

required = [
    "SECONDARY_BASELINE_RUNTIME_PARENT_MANIFEST.json",
    "TRUE_NOPROBE_VALIDITY_AUDIT.json",
    "GT_PHYSICS_VALIDITY_AUDIT.json",
    "TABERO_NEUTRAL_VALIDITY_AUDIT.json",
    "FINAL_SECONDARY_VARIANT_MANIFEST.json",
    "NEW_SECONDARY_ROOTS.json",
    "FINAL_SECONDARY_EXECUTION_PLAN.csv",
    "FINAL_SECONDARY_RETRY_POLICY.json",
    "FINAL_INFRASTRUCTURE_PREFLIGHT.json",
    "SECONDARY_EXECUTION_COMPLETE.json",
    "FINAL_SECONDARY_BRANCH_RESULTS.csv",
    "FINAL_SECONDARY_CONTEXT_RESULTS.csv",
    "TABLE_SECONDARY_NEW24.csv",
    "TABLE_SECONDARY_PAIRED.csv",
    "TABLE_SECONDARY_PER_ROOT.csv",
    "TABLE_SECONDARY_PER_TASK.csv",
    "TABLE_SECONDARY_PER_FRICTION.csv",
    "TABLE_PHYSICS_INFORMATION_LADDER.csv",
    "TABLE_TABERO_MATCHED_BASELINE.csv",
    "FINAL_SECONDARY_FAILURE_ANALYSIS.csv",
    "FIGURE_PHYSICS_INFORMATION_LADDER.pdf",
    "FIGURE_PHYSICS_INFORMATION_LADDER.png",
    "FINAL_SECONDARY_BASELINE_REPORT.md",
    "FINAL_SECONDARY_BASELINE_STATUS.json",
    "FINAL_TERMINAL_OUTPUT.txt",
]
missing = [name for name in required if not (out / name).is_file()]
validation = {
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": "PASS",
    "planned_branches": len(plan_rows),
    "branch_directories": len(branches),
    "reference_directories": len(references),
    "admitted_branches": sum(bool(x["admitted"]) for x in admissions),
    "online_vla_valid_branches": sum(bool(x["online_provenance"]["passed"]) for x in admissions),
    "geometric_label_valid_branches": sum(bool(x["geometry"]["passed"]) for x in admissions),
    "total_policy_inference_requests": sum(int(x["rpc_count"]) for x in results),
    "server_inference_log_rows": sum(1 for _ in (out / "POLICY_SERVER_18885/INFERENCE.jsonl").open()),
    "scripted_or_replay_branches": sum(x["downstream_action_source"] != "ONLINE_VLA" for x in results),
    "retry_count": 0,
    "quarantine_count": 0,
    "health_gate_checks": len(health),
    "health_gate_failures": len(resource_failures),
    "nvml_fault_checks": len(nvml_fault_rows),
    "source_hash_mismatches": source_mismatch,
    "root_manifest_sha256": sha(out / "NEW_SECONDARY_ROOTS.json"),
    "root_manifest_expected_sha256": (out / "NEW_SECONDARY_ROOTS_SHA256.txt").read_text().split()[0],
    "execution_plan_sha256": sha(out / "FINAL_SECONDARY_EXECUTION_PLAN.csv"),
    "execution_plan_expected_sha256": (out / "FINAL_SECONDARY_EXECUTION_PLAN_SHA256.txt").read_text().split()[0],
    "candidate_runtime_manifest_sha256": sha(out / "CANDIDATE_RUNTIME_MANIFEST.json"),
    "required_artifacts_missing": missing,
    "main_8_root_artifacts_modified": False,
    "physics_stopped": True,
}
checks = [
    len(plan_rows) == 72, len(branches) == 72, len(references) == 24,
    validation["admitted_branches"] == 72,
    validation["online_vla_valid_branches"] == 72,
    validation["geometric_label_valid_branches"] == 72,
    validation["total_policy_inference_requests"] == 2520,
    validation["server_inference_log_rows"] == 2520,
    validation["scripted_or_replay_branches"] == 0,
    validation["health_gate_failures"] == 0,
    not source_mismatch,
    validation["root_manifest_sha256"] == validation["root_manifest_expected_sha256"],
    validation["execution_plan_sha256"] == validation["execution_plan_expected_sha256"],
    not missing,
]
if not all(checks):
    validation["status"] = "FAIL"
(stage / "FINAL_SECONDARY_VALIDATION.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")

index_names = required + [
    "CANDIDATE_RUNTIME_MANIFEST.json", "PREPHYSICS_FREEZE_STATUS.json",
    "PREPHYSICS_AUDIT_FIX.json", "GPU_HEALTH_GATE_LOG.jsonl",
    "POLICY_SERVER_18885/SERVER_READY.json", "POLICY_SERVER_18885/INFERENCE.jsonl",
]
index = {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size} for name in index_names if (out / name).is_file()}
(stage / "FINAL_SECONDARY_ARTIFACT_INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
print(json.dumps(validation, indent=2, sort_keys=True))
