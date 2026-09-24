"""One retry of official prefix skips. Does not write into the 18/19 tables."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
AF = Path("/media/volume/dasdas/exouser/af_dump_formal_liftclone_eu_fmax5_20260915")
NOM = Path("/media/volume/dasdas/exouser/af_dump_formal_liftclone_eu_fmax5_nominal_20260915")
BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
REPO = BASE / "RoboTwin"
PYTHON = BASE / "venv_robotwin/bin/python3"

sys.path.insert(0, str(AF))
from audit_formal_results import audit_context as audit_af

sys.path.insert(0, str(NOM))
from audit_nominal_results import audit_context as audit_nominal

PREFIX_MARKERS = (
    (
        "Contact-lost probe is not deployment qualified",
        "Original P4 contact-lost; decision prefix not deployment qualified",
    ),
    (
        "Dump shear query lost contact after established grasp",
        "Dump shear query lost contact after established grasp",
    ),
)

AF_SKIP_IDS = [
    "liftclone_eu_fmax5_mu0.575_root200002_ps80200006",
    "liftclone_eu_fmax5_mu0.575_root200002_ps80200009",
]
NOM_SKIP_IDS = [
    "liftclone_eu_fmax5_nominal_mu0.575_root200002_ps80200002",
    "liftclone_eu_fmax5_nominal_mu0.575_root200002_ps80200005",
    "liftclone_eu_fmax5_nominal_mu0.575_root200002_ps80200006",
    "liftclone_eu_fmax5_nominal_mu0.575_root200002_ps80200008",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def server_ready() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 6001), timeout=2):
            return True
    except OSError:
        return False


def verify_official_untouched() -> None:
    for root, expected_selector in ((AF, "expected_utility"), (NOM, "none")):
        plan = root / "plan"
        lock = read(plan / "FREEZE_LOCK.json")
        checks = {
            "protocol_sha256": sha(plan / "PROTOCOL.json"),
            "smoke_context_sha256": sha(plan / "SMOKE_CONTEXT.json"),
            "formal_contexts_sha256": sha(plan / "FORMAL_CONTEXTS.json"),
        }
        if checks != lock:
            raise ValueError("Official freeze lock drifted: " + str(root))
        protocol = read(plan / "PROTOCOL.json")
        for path, digest in protocol["source_hashes"].items():
            if sha(Path(path)) != digest:
                raise ValueError("Official source changed: " + path)
        if protocol.get("executed_selector") != expected_selector:
            raise ValueError("Official selector drifted: " + str(root))
        if not (root / "FINAL_RESULTS.json").exists():
            raise ValueError("Official FINAL_RESULTS missing: " + str(root))


def official_contexts(root: Path) -> dict:
    return {row["id"]: row for row in read(root / "plan/FORMAL_CONTEXTS.json")}


def jobs() -> list[dict]:
    af_ctx = official_contexts(AF)
    nom_ctx = official_contexts(NOM)
    rows = []
    for context_id in AF_SKIP_IDS:
        rows.append({"kind": "ActiveForcing", "root": AF, "infer": AF / "infer_af_context.py", "context": af_ctx[context_id]})
    for context_id in NOM_SKIP_IDS:
        rows.append(
            {
                "kind": "Nominal Frozen VLA",
                "root": NOM,
                "infer": NOM / "infer_nominal_context.py",
                "context": nom_ctx[context_id],
            }
        )
    return rows


def classify_prefix_failure(case: Path) -> str | None:
    if (case / "job/FORMAL_CONTEXT_RESULT.json").exists():
        return None
    log_path = case / "worker.log"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    if "ORIGINAL_ONLINE_CHUNK" in log or "ORIGINAL_ONLINE_BRANCH" in log:
        return None
    for marker, reason in PREFIX_MARKERS:
        if marker in log:
            return reason
    return None


def compact(audit: dict, kind: str) -> dict:
    rows = []
    for branch in audit["branches"]:
        raw = branch["result"]
        rows.append(
            {
                "method": branch["method"],
                "success": int(raw["success"]),
                "commanded_force_N": raw.get("commanded_force_N", raw.get("force_setpoint_bilateral_n")),
                "measured_mean_squeeze_N": raw["measured_mean_squeeze_n"],
                "measured_max_squeeze_N": raw["measured_max_squeeze_n"],
                "native_actions": raw["native_actions"],
                "physics_steps": raw["physics_steps"],
                "trace_sha256": raw["trace_sha256"],
                "raw_audit_passed": branch["audit"]["passed"],
            }
        )
    return {
        "context": audit["context"],
        "kind": kind,
        "fill_in": True,
        "official_18_19_not_modified": True,
        "completed_utc": now(),
        "first_chunk_pairing_exact": audit["first_chunk_pairing_exact"],
        "common_handoff_state_sha256": audit["common_handoff_state_sha256"],
        "formal_context_result_sha256": audit["formal_context_result_sha256"],
        "outcomes": rows,
    }


def run_one(job: dict) -> dict:
    context = job["context"]
    case = HERE / "fill_raw" / context["id"]
    record_path = HERE / "fill_records" / f"{context['id']}.json"
    if record_path.exists():
        return read(record_path)
    skip_path = HERE / "fill_skips" / f"{context['id']}.json"
    if skip_path.exists():
        return read(skip_path)
    if case.exists():
        reason = classify_prefix_failure(case)
        if reason:
            entry = {
                "fill_in": True,
                "prefix_skip": True,
                "still_unqualified_after_one_retry": True,
                "context": context,
                "kind": job["kind"],
                "reason": reason,
                "archived_utc": now(),
            }
            dest = HERE / "fill_prefix_failures" / context["id"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest = HERE / "fill_prefix_failures" / f"{context['id']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            case.rename(dest)
            entry["archive_path"] = str(dest)
            write(skip_path, entry)
            return entry
        raise RuntimeError("Partial fill-in case needs inspection: " + str(case))
    if shutil.disk_usage(HERE).free < int(1.4 * 1024**3):
        raise RuntimeError("Storage gate: less than 1.4 GiB before fill-in context")
    case.mkdir(parents=True)
    training = read(job["root"] / "models_deploy/TRAINING_COMPLETE.json")
    spec = {
        "context": context,
        "models": str(job["root"] / "models_deploy"),
        "belief_manifest_sha256": training["belief_manifest_sha256"],
        "feasibility_manifest_sha256": training["feasibility_manifest_sha256"],
        "fill_in": True,
        "official_18_19_not_modified": True,
    }
    write(case / "SPEC.json", spec)
    command = [
        str(PYTHON),
        str(job["infer"]),
        "--repo",
        str(REPO),
        "--out",
        str(case / "job"),
        "--native-ft",
        "--friction",
        str(context["friction"]),
        "--policy-seed",
        str(context["policy_seed"]),
    ]
    environment = os.environ.copy()
    environment.pop("AF_COLLECTION_CONTEXT", None)
    environment.pop("AF_INFERENCE_CONTEXT", None)
    environment.update(
        AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP="1",
        AF_P4_PHYSICAL_SURFACE_CAMERA="1",
        AF_ORIGINAL_SQUEEZE_INNER="1",
        AF_INFERENCE_CONTEXT=str(case / "SPEC.json"),
        AF_FORMAL_CONTEXT=str(case / "SPEC.json"),
        PYTHONUTF8="1",
        PATH=str(BASE / "runtime_bin") + os.pathsep + environment.get("PATH", "/usr/bin:/bin"),
    )
    write(case / "PROCESS.json", {"command": command, "cwd": str(REPO), "started_utc": now(), "fill_in": True})
    with (case / "worker.log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write(case / "PID.json", {"pid": process.pid})
        disk_stop = False
        while process.poll() is None:
            if shutil.disk_usage(HERE).free < int(0.45 * 1024**3):
                disk_stop = True
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                break
            time.sleep(10)
    write(case / "EXIT.json", {"exit_code": process.returncode, "disk_guard_stop": disk_stop, "finished_utc": now()})
    if process.returncode or not (case / "job/FORMAL_CONTEXT_RESULT.json").exists():
        reason = classify_prefix_failure(case)
        if reason:
            dest = HERE / "fill_prefix_failures" / context["id"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest = HERE / "fill_prefix_failures" / f"{context['id']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            case.rename(dest)
            entry = {
                "fill_in": True,
                "prefix_skip": True,
                "still_unqualified_after_one_retry": True,
                "context": context,
                "kind": job["kind"],
                "reason": reason,
                "archived_utc": now(),
                "archive_path": str(dest),
            }
            write(HERE / "fill_skips" / f"{context['id']}.json", entry)
            return entry
        raise RuntimeError("Fill-in context failed/unknown; retained: " + context["id"])
    audited = audit_af(case) if job["kind"] == "ActiveForcing" else audit_nominal(case)
    write(case / "INDEPENDENT_CONTEXT_AUDIT.json", audited)
    record = compact(audited, job["kind"])
    write(case / "CASE_COMPACT.json", record)
    write(record_path, record)
    kept = HERE / "fill_kept"
    kept.mkdir(parents=True, exist_ok=True)
    shutil.copy2(case / "job/FORMAL_CONTEXT_RESULT.json", kept / f"{context['id']}_FORMAL_CONTEXT_RESULT.json")
    return record


def summarize(records: list[dict], skips: list[dict]) -> dict:
    af_ok = [row for row in records if row.get("kind") == "ActiveForcing"]
    nom_ok = [row for row in records if row.get("kind") == "Nominal Frozen VLA"]
    return {
        "completed": True,
        "fill_in": True,
        "official_paired_18_19_not_modified": True,
        "af_remainder_fail_mu0.425_ps80200008_not_retried": True,
        "claim_boundary": (
            "one retry of official prefix skips only; not written into the "
            "official AF 18/19 vs Nominal 19/19 table"
        ),
        "retried": 6,
        "filled": len(records),
        "still_unqualified": len(skips),
        "af_fill_successes": sum(int(row["outcomes"][0]["success"]) for row in af_ok),
        "af_fill_contexts": len(af_ok),
        "nominal_fill_successes": sum(int(row["outcomes"][0]["success"]) for row in nom_ok),
        "nominal_fill_contexts": len(nom_ok),
        "records": [str(HERE / "fill_records" / f"{row['context']['id']}.json") for row in records],
        "skips": skips,
        "finished_utc": now(),
    }


def main() -> None:
    lock_file = (HERE / "PREFIX_FILL_QUEUE.lock").open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A live prefix fill-in queue already holds the lock") from error
    verify_official_untouched()
    if not server_ready():
        raise RuntimeError("pi0 is not listening on port 6001")
    planned = jobs()
    records = []
    skips = []
    try:
        for job in planned:
            verify_official_untouched()
            write(
                HERE / "STATUS.json",
                {
                    "status": "fill_running",
                    "fill_in": True,
                    "official_18_19_not_modified": True,
                    "current_context": job["context"]["id"],
                    "filled": len(records),
                    "still_unqualified": len(skips),
                    "updated_utc": now(),
                },
            )
            result = run_one(job)
            if result.get("prefix_skip"):
                skips.append(result)
                continue
            records.append(result)
        final = summarize(records, skips)
        write(HERE / "FILL_RESULTS.json", final)
        write(
            HERE / "STATUS.json",
            {
                "status": "complete",
                "fill_in": True,
                "official_18_19_not_modified": True,
                "filled": len(records),
                "still_unqualified": len(skips),
                "current_context": None,
                "updated_utc": now(),
            },
        )
        print(json.dumps({"filled": len(records), "still_unqualified": len(skips)}, indent=2), flush=True)
    except Exception as error:
        write(HERE / "STATUS.json", {"status": "failed", "fill_in": True, "error": repr(error), "updated_utc": now()})
        raise


if __name__ == "__main__":
    main()
