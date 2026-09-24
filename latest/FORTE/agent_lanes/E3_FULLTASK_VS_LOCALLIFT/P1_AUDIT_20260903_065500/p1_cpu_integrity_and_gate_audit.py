#!/usr/bin/env python3
"""CPU-only P1 audit; never launches or signals an experiment process."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(__file__).resolve().parent
OPENPI = Path("/home/exouser/FORTE/agent_lanes/Tabero_VTLA_e3_onboarding_20260902")
ART = Path("/home/exouser/FORTE_e3/ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000")
LOCK = Path("/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_5DEMO_CANDIDATE_LOCK_20260902/CANDIDATE_LOCK.json")
PROTO = ART / "E3_POST_ONBOARDING_FULLTASK_LOCALLIFT_PROTOCOL.json"
MASTER = Path("/home/exouser/E3_E6_E7_LIVE_STATUS/MASTER_STATUS.json")
FREEZE_SERVER = Path("/home/exouser/FORTE/visual_pi0_server.py")
LANE_SERVER = OPENPI / "server.sh"
NORM = Path("/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_7DPF_ASSETS_20260902_113000/pi0_lora_tacfield_e3_task5_5demo_7dpf/activeforcing_e3_task5_5demo_7dpf/norm_stats.json")
BASE = Path("/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999")

EXPECTED = {
    "openpi_commit": "31049447d685cb36ddaeddda4f1d62fec0bc6392",
    "config_sha256": "296b01a51c985583bab688402808296315bd449ddd2aa343f4f5458ba1d1295e",
    "policy_sha256": "f5eb0161b831c1f4a65b763f24183f5333a3efe54ae0537088a9450fdd54781a",
    "freeze_server_sha256": "56851e5d3fdf034eaf8bcdd2350a66187c40a1a4ed86704de85fcdf020797976",
    "protocol_sha256": "af18dd12cb2a9ceb190c6a375669d923e7d8154fd5844f40335e2a5db823844a",
    "norm_sha256": "492a36faac4809acfe98098f67b56738f89a5600b997f1ea8d72d8bcded8ae32",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_manifest(root: Path) -> tuple[str, int, int, list[str]]:
    h = hashlib.sha256()
    count = 0
    total = 0
    mismatch: list[str] = []
    lock = json.loads(LOCK.read_text())
    expected = {str(x["path"]): x for x in lock["checkpoint_files"]}
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256(path)
        count += 1
        total += size
        h.update(rel.encode())
        h.update(b"\0")
        h.update(str(size).encode())
        h.update(b"\0")
        h.update(digest.encode())
        h.update(b"\n")
        e = expected.get(rel)
        if e is None or e["bytes"] != size or e["sha256"] != digest:
            mismatch.append(rel)
    missing = sorted(set(expected) - {x.relative_to(root).as_posix() for x in root.rglob("*") if x.is_file()})
    mismatch.extend("MISSING:" + x for x in missing)
    return h.hexdigest(), count, total, mismatch


def command(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, text=True, capture_output=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def main() -> int:
    captured = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    lock = json.loads(LOCK.read_text())
    master = json.loads(MASTER.read_text())
    git_rc, git_head, git_err = command(["git", "-C", str(OPENPI), "rev-parse", "HEAD"])
    tree, tree_files, tree_bytes, tree_mismatch = tree_manifest(Path(lock["checkpoint_dir"]))
    norm = json.loads(NORM.read_text())["norm_stats"]
    base_files = {}
    for rel in ("_CHECKPOINT_METADATA", "params/_METADATA", "params/manifest.ocdbt"):
        p = BASE / rel
        base_files[rel] = {"exists": p.is_file(), "sha256": sha256(p) if p.is_file() else None, "bytes": p.stat().st_size if p.is_file() else None}
    smi_rc, smi_out, smi_err = command(["nvidia-smi"])
    protected = master.get("protected_processes_at_audit", {})
    status = {
        "captured_utc": captured,
        "scope": "P1 FullTask vs LocalLift; CPU-only integrity and dynamic admission audit",
        "no_process_mutation": True,
        "candidate": {
            "lock_path": str(LOCK),
            "lock_sha256": sha256(LOCK),
            "status": lock.get("status"),
            "task": lock.get("task"),
            "selected_step": lock.get("selected_step"),
            "checkpoint_dir": lock.get("checkpoint_dir"),
            "checkpoint_tree_recorded_sha256": lock.get("checkpoint_tree_sha256"),
            "checkpoint_tree_recomputed_sha256": tree,
            "checkpoint_files_in_lock": len(lock.get("checkpoint_files", [])),
            "checkpoint_files_recomputed": tree_files,
            "checkpoint_bytes_recomputed": tree_bytes,
            "checkpoint_file_hash_mismatches": tree_mismatch,
            "integrity_pass": bool(tree == lock.get("checkpoint_tree_sha256") and not tree_mismatch),
        },
        "lineage": {
            "openpi_git_rev_parse_rc": git_rc,
            "openpi_commit_observed": git_head,
            "openpi_commit_expected": EXPECTED["openpi_commit"],
            "openpi_commit_pass": git_head == EXPECTED["openpi_commit"],
            "config_observed": sha256(OPENPI / "src/openpi/training/config.py"),
            "config_expected": EXPECTED["config_sha256"],
            "config_pass": sha256(OPENPI / "src/openpi/training/config.py") == EXPECTED["config_sha256"],
            "policy_observed": sha256(OPENPI / "src/openpi/policies/libero_policy.py"),
            "policy_expected": EXPECTED["policy_sha256"],
            "policy_pass": sha256(OPENPI / "src/openpi/policies/libero_policy.py") == EXPECTED["policy_sha256"],
            "protocol_observed": sha256(PROTO),
            "protocol_expected": EXPECTED["protocol_sha256"],
            "protocol_pass": sha256(PROTO) == EXPECTED["protocol_sha256"],
            "freeze_server_observed": sha256(FREEZE_SERVER),
            "freeze_server_expected": EXPECTED["freeze_server_sha256"],
            "freeze_server_pass": sha256(FREEZE_SERVER) == EXPECTED["freeze_server_sha256"],
            "candidate_openpi_server_sh_observed": sha256(LANE_SERVER),
            "candidate_openpi_server_sh_note": "differs from historical freeze server hash; candidate launcher uses isolated OpenPI websocket server, so preserve both identities and do not silently substitute",
            "norm_observed": sha256(NORM),
            "norm_expected": EXPECTED["norm_sha256"],
            "norm_pass": sha256(NORM) == EXPECTED["norm_sha256"],
            "norm_dimensions": {k: len(v["mean"]) for k, v in norm.items()},
            "base_checkpoint_step": 49999,
            "base_checkpoint_files": base_files,
            "base_checkpoint_required_metadata_present": all(x["exists"] for x in base_files.values()),
            "protocol_json": json.loads(PROTO.read_text()),
        },
        "resource_attestation": {
            "source_live_status": str(MASTER),
            "live_status_updated_at": master.get("updated_at"),
            "protected_processes": protected,
            "nvidia_smi_returncode": smi_rc,
            "nvidia_smi_stdout_tail": smi_out[-4000:],
            "nvidia_smi_stderr": smi_err,
            "gpu_launch_authorized": False,
            "blockers": [
                "protected E5 scheduler/worker active",
                "protected Mass worker active",
                "authoritative frozen pi0 server active",
                "no E5 natural-exit evidence and no >=10s silence",
                "candidate status is not final freeze",
            ],
        },
        "protocol_checks": {
            "test_used": False,
            "activeforcing_outcomes_used": False,
            "onboarding_split": lock.get("split"),
            "nominal_dev_roots": [7600, 7601, 7602, 7603, 7604],
            "nominal_dev_force_N": 8,
            "nominal_dev_friction": 0.6,
            "nominal_dev_pass_rule": "official FullTask >=3/5 and query-state reach >=4/5",
            "matched_model_roots": {"train": [7700, 7701, 7702, 7703, 7704, 7705], "heldout": [7800, 7801, 7802]},
            "matched_manifest_status": "NOT_PRESENT; driver must fail closed until branch/context manifests exist",
        },
        "admission": "BLOCKED_PROTECTED_RESOURCE_AND_NO_DYNAMIC_FINAL_GATE",
        "scientific_status": "NOT_YET_ADMISSIBLE",
    }
    (OUT / "P1_CPU_INTEGRITY_AND_GATE_AUDIT.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status["admission"], "candidate_integrity_pass": status["candidate"]["integrity_pass"], "protected_processes": protected, "output": str(OUT / "P1_CPU_INTEGRITY_AND_GATE_AUDIT.json")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
