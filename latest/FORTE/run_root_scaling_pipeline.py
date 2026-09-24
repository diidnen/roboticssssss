#!/usr/bin/env python3
"""Safely supervise the long sequential root-scaling pipeline."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "root_scaling_20260831"
PY = Path("/media/volume/newdata/exouser/softvtbench/openpi-venv/bin/python")
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
SERVER = ROOT / "visual_pi0_server.py"
POLICY = Path("/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999")
NORMS = Path("/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/norm_stats/pi0_lora_tacfield_tabero/tabero")
LOG = OUT / "ROOT_SCALING_PIPELINE.log"
STATUS = OUT / "ROOT_SCALING_PIPELINE_STATUS.json"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(stage: str, status: str, **extra) -> None:
    STATUS.write_text(json.dumps({"updated_utc": utc(), "stage": stage, "status": status, **extra},
                                 indent=2, sort_keys=True, default=str) + "\n")


def active_collectors() -> list[dict]:
    me = os.getpid(); rows = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == me: continue
        try: cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except Exception: continue
        if any(x in cmd for x in ("prospective_visual_context_collect.py --worker",
                                  "task0_context_collect.py --split TRAIN",
                                  "root_scaling_collect.py --split")):
            rows.append({"pid": int(p.name), "cmdline": cmd.strip()})
    return rows


def wait_for_existing() -> None:
    while True:
        q = active_collectors()
        if not q: return
        write_status("WAIT_EXISTING_COLLECTOR", "RUNNING", active_collectors=q)
        with LOG.open("a") as f: f.write(f"[{utc()}] waiting for collectors {q}\n")
        time.sleep(30)


def server_ready() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 18881), timeout=2): return True
    except OSError: return False


def ensure_server():
    if server_ready(): return None
    log = (OUT / "visual_pi0_server.log").open("a")
    proc = subprocess.Popen([str(PY), "-u", str(SERVER), "--port", "18881", "--mode", "instrumented",
                             "--policy-dir", str(POLICY), "--norm-stats-dir", str(NORMS)],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(180):
        if proc.poll() is not None: raise RuntimeError(f"visual server exited rc={proc.returncode}")
        if server_ready(): return proc
        time.sleep(1)
    proc.terminate(); raise RuntimeError("visual server did not become ready in 180s")


def run(stage: str, argv: list[str], timeout: int | None = None) -> None:
    write_status(stage, "RUNNING", argv=argv)
    with LOG.open("a") as f:
        f.write(f"[{utc()}] START {stage}: {argv}\n"); f.flush()
        p = subprocess.run(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        f.write(f"[{utc()}] END {stage}: rc={p.returncode}\n"); f.flush()
    if p.returncode != 0:
        write_status(stage, "FAILED_REQUIRES_INSPECTION", argv=argv, returncode=p.returncode)
        raise RuntimeError(f"{stage} failed rc={p.returncode}")


def old_task0_train_ready() -> bool:
    """Legacy gate: S50 completion + independent QA, not the old N80 target."""
    audit_path = OUT / "CURRENT_COLLECTION_PROGRESS_AUDIT.json"
    qa_path = OUT / "S50_REQUIRED_ROOTS_QA.json"
    if not audit_path.exists() or not qa_path.exists():
        return False
    try:
        audit = json.loads(audit_path.read_text())
        qa = json.loads(qa_path.read_text())
        return bool(
            audit.get("all_preregistered_S50_roots_context_complete")
            and audit.get("all_required_branches_complete")
            and audit.get("all_required_parity_checks_pass")
            and audit.get("all_required_context_commits_pass")
            and qa.get("status") == "PASS"
            and qa.get("scientific_population", {}).get("required_root_indices") == list(range(12, 56))
            and qa.get("scientific_population", {}).get("excluded_root_indices") == list(range(56, 74))
        )
    except (OSError, ValueError, TypeError):
        return False


def progress_audit_has_required_roots() -> bool:
    path = OUT / "CURRENT_COLLECTION_PROGRESS_AUDIT.json"
    if not path.exists():
        return False
    try:
        obj = json.loads(path.read_text())
        return bool(obj.get("all_preregistered_S50_roots_context_complete")
                    and obj.get("all_required_branches_complete")
                    and obj.get("all_required_parity_checks_pass")
                    and obj.get("all_required_context_commits_pass"))
    except (OSError, ValueError, TypeError):
        return False


def acceleration_freeze_ready() -> bool:
    """Require the post-collection acceleration decision before touching TEST."""
    freeze_path = OUT / "COLLECTOR_ACCELERATION_FREEZE.json"
    decision_path = OUT / "ACCELERATION_FINAL_DECISION.md"
    if not freeze_path.exists() or not decision_path.exists():
        return False
    try:
        freeze = json.loads(freeze_path.read_text())
        return bool(freeze.get("status") == "FROZEN"
                    and freeze.get("authoritative_implementation") == "sequential_existing_collector"
                    and freeze.get("num_envs") == 1
                    and freeze.get("worker_count") == 1
                    and freeze.get("pi0_action_cache") is False
                    and freeze.get("irreversible_failure_early_stop") is False)
    except (OSError, ValueError, TypeError):
        return False


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    owned_server = None
    try:
        wait_for_existing()
        # Reconcile the inherited collector from authoritative rows before
        # touching TEST. This is an engineering gate, not a scientific gate.
        run("BUILD_CURRENT_COLLECTION_PROGRESS_AUDIT",
            [str(PY), str(ROOT / "build_acceleration_audit_artifacts.py")], 3600)
        if progress_audit_has_required_roots():
            run("QA_S50_REQUIRED_ROOTS",
                [str(PY), str(ROOT / "s50_required_roots_qa.py")], 3600)
        if not acceleration_freeze_ready():
            raise RuntimeError("post-collection acceleration freeze is required before untouched TEST")
        owned_server = ensure_server()
        # Untouched TEST is always collected and committed before any task5
        # TRAIN additions or scaling-model training.
        for task in (0, 5):
            run(f"COLLECT_TASK{task}_TEST", [str(PY), str(ROOT / "root_scaling_collect.py"),
                                              "--split", "TEST", "--task", str(task), "--timeout-s", "43200"], 44000)
            run(f"AUDIT_TASK{task}_TEST", [str(PY), str(ROOT / "audit_root_scaling_collection.py"),
                                            "--split", "TEST", "--task", str(task)], 3600)

        if not old_task0_train_ready():
            if progress_audit_has_required_roots():
                raise RuntimeError("S50 required-roots QA is not PASS; legacy collector is not resumed")
            # Exact infrastructure resume of the already pre-outcome-frozen
            # task0 target. Scientific successes/failures are never retried.
            run("RESUME_INHERITED_TASK0_TRAIN", [str(ISAAC_PY), "-u", str(ROOT / "task0_context_collect.py"),
                                                  "--split", "TRAIN", "--timeout-s", "43200"], 44000)
            run("BUILD_CURRENT_COLLECTION_PROGRESS_AUDIT",
                [str(PY), str(ROOT / "build_acceleration_audit_artifacts.py")], 3600)
            run("QA_S50_REQUIRED_ROOTS",
                [str(PY), str(ROOT / "s50_required_roots_qa.py")], 3600)
        else:
            qa = json.loads((OUT / "S50_REQUIRED_ROOTS_QA.json").read_text())
            included = int(qa["scientific_population"]["included_contexts"])
            collected = int(qa["counts"]["contexts_in_file"])
            excluded_target = len(qa["scientific_population"]["excluded_root_indices"])
            excluded_collected = max(0, collected - included)
            write_status("LEGACY_COLLECTION_EARLY_STOPPED", "PASS",
                         reason="All preregistered S50 scientific roots were already complete. Remaining collector targets belong only to the legacy N80 plan and are preregistered-excluded from the current S6/S15/S30/S50 experiment.",
                         collected_total=collected, included_in_S50=included,
                         excluded_legacy_collected=excluded_collected,
                         excluded_legacy_remaining=excluded_target - excluded_collected,
                         qa=str(OUT / "S50_REQUIRED_ROOTS_QA.json"))
        if not old_task0_train_ready():
            raise RuntimeError("S50 required-roots QA is not PASS after legacy gate")

        run("COLLECT_TASK5_TRAIN", [str(PY), str(ROOT / "root_scaling_collect.py"),
                                     "--split", "TRAIN", "--task", "5", "--timeout-s", "43200"], 44000)
        run("AUDIT_TASK5_TRAIN", [str(PY), str(ROOT / "audit_root_scaling_collection.py"),
                                   "--split", "TRAIN", "--task", "5"], 3600)

        run("TRAIN_72_CHECKPOINTS", [str(PY), str(ROOT / "root_scaling_learning_curve.py"), "train"], 43200)
        run("EVALUATE_UNTOUCHED_TEST", [str(PY), str(ROOT / "root_scaling_learning_curve.py"), "evaluate"], 21600)
        env = os.environ.copy(); env["MPLCONFIGDIR"] = "/tmp/root_scaling_mpl"
        write_status("FINALIZE", "RUNNING")
        with LOG.open("a") as f:
            p = subprocess.run([str(PY), str(ROOT / "root_scaling_finalize.py")], cwd=ROOT,
                               env=env, stdout=f, stderr=subprocess.STDOUT, timeout=3600, check=False)
        if p.returncode != 0: raise RuntimeError(f"FINALIZE failed rc={p.returncode}")
        run("VALIDATE", [str(PY), str(ROOT / "validate_root_scaling_delivery.py")], 3600)
        write_status("COMPLETE", "PASS", report=str(OUT / "ROOT_SCALING_FINAL_REPORT.md"),
                     classification=str(OUT / "ROOT_SCALING_FINAL_CLASSIFICATION.json"))
    except Exception as exc:
        if not STATUS.exists() or json.loads(STATUS.read_text()).get("status") != "FAILED_REQUIRES_INSPECTION":
            write_status("PIPELINE", "FAILED_REQUIRES_INSPECTION", error=repr(exc))
        raise
    finally:
        if owned_server is not None and owned_server.poll() is None:
            owned_server.terminate()
            try: owned_server.wait(timeout=60)
            except subprocess.TimeoutExpired: owned_server.kill(); owned_server.wait()


if __name__ == "__main__": main()
