import datetime as dt
import hashlib
import json
from pathlib import Path

stage = Path("/home/exouser/FORTE/secondary_staging_v1")
out = Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_secondary_baselines_v1")
candidate_path = stage / "CANDIDATE_RUNTIME_MANIFEST.json"
status_path = stage / "PREPHYSICS_FREEZE_STATUS.json"
audit_path = out / "SOURCE_SNAPSHOT/audit_secondary_rollout.py"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

candidate = json.loads(candidate_path.read_text())
old_candidate_sha = sha(candidate_path)
old_audit_sha = candidate["source_hashes"][str(audit_path)]
new_audit_sha = sha(audit_path)
candidate["source_hashes"][str(audit_path)] = new_audit_sha
candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
new_candidate_sha = sha(candidate_path)

status = json.loads(status_path.read_text())
status["candidate_runtime_manifest_sha256"] = new_candidate_sha
status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

job_dir = out / "JOBS"
started = sorted(str(p) for p in job_dir.glob("*") if p.is_dir()) if job_dir.exists() else []
receipt = {
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "change_scope": "provenance audit receipt comparison only",
    "reason": "RPC client receipts add client-side fields to the server-signed receipt; compare every server field rather than requiring object equality",
    "scientific_runtime_changed": False,
    "physics_branches_started_before_fix": len(started),
    "started_branch_directories": started,
    "old_audit_sha256": old_audit_sha,
    "new_audit_sha256": new_audit_sha,
    "old_candidate_manifest_sha256": old_candidate_sha,
    "new_candidate_manifest_sha256": new_candidate_sha,
    "root_manifest_sha256": status["root_manifest_sha256"],
    "execution_plan_sha256": status["execution_plan_sha256"],
    "variant_manifest_sha256": status["variant_manifest_sha256"],
}
(stage / "PREPHYSICS_AUDIT_FIX.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
