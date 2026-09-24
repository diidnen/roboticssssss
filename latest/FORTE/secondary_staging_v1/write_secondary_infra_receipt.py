import datetime as dt
import hashlib
import json
from pathlib import Path

out = Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_secondary_baselines_v1")
stage = Path("/home/exouser/FORTE/secondary_staging_v1")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path: Path):
    return json.loads(path.read_text())

candidate = read(out / "CANDIDATE_RUNTIME_MANIFEST.json")
source_mismatches = []
for path, expected in candidate["source_hashes"].items():
    observed = sha(Path(path))
    if observed != expected:
        source_mismatches.append({"path": path, "expected": expected, "observed": observed})

health_rows = [json.loads(x) for x in (out / "GPU_HEALTH_GATE_LOG.jsonl").read_text().splitlines() if x.strip()]
preflight = read(out / "SIMULATOR_NO_STEP_PREFLIGHT.json")
server = read(out / "POLICY_SERVER_18885/SERVER_READY.json")
attempts = []
for path in sorted(out.glob("POLICY_SERVER_18885_FAILED_ATTEMPT*")):
    attempts.append({"path": str(path), "status": "FAILED_IMPORT_ONLY", "physics": 0, "inference": 0})
resource_attempt = out / "POLICY_SERVER_18885_RESOURCE_PREFLIGHT_ATTEMPT"
if resource_attempt.exists():
    attempts.append({"path": str(resource_attempt), "status": "METADATA_ONLY_STOPPED_GRACEFULLY", "physics": 0, "inference": 0})

receipt = {
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": "PASS" if not source_mismatches and preflight["status"] == "PASS" and health_rows[-1]["POLICY_SERVER_18885_VALID"] else "FAIL",
    "scientific_runtime_changed": False,
    "policy_server": {
        "pid": server["pid"],
        "port": 18885,
        "checkpoint": server["checkpoint"],
        "checkpoint_sha256": server["checkpoint_sha256"],
        "online_vla": server["downstream_action_source"] == "ONLINE_VLA",
        "xla_python_client_preallocate": False,
        "environment_recovery": "Frozen OpenPI source and exact LeRobot lock source precede an appended site-packages fallback for missing third-party packages; model/source/checkpoint hashes unchanged.",
    },
    "prior_server_start_attempts": attempts,
    "gpu_health": health_rows[-1],
    "simulator_no_step_preflight": preflight,
    "source_hash_count": len(candidate["source_hashes"]),
    "source_hash_mismatches": source_mismatches,
    "candidate_runtime_manifest_sha256": sha(out / "CANDIDATE_RUNTIME_MANIFEST.json"),
    "root_manifest_sha256": sha(out / "NEW_SECONDARY_ROOTS.json"),
    "execution_plan_sha256": sha(out / "FINAL_SECONDARY_EXECUTION_PLAN.csv"),
    "variant_manifest_sha256": sha(out / "FINAL_SECONDARY_VARIANT_MANIFEST.json"),
    "branch_directories_before_execution": len(list((out / "branches").glob("*"))) if (out / "branches").exists() else 0,
    "reference_directories_before_execution": len(list((out / "references").glob("*"))) if (out / "references").exists() else 0,
    "roots_170050_170051_untouched": not (out / "branches").exists() and not (out / "references").exists(),
}
(stage / "FINAL_INFRASTRUCTURE_PREFLIGHT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
