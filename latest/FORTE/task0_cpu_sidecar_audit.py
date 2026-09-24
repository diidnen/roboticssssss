#!/usr/bin/env python3
"""Read-only CPU-sidecar feasibility and GPU non-interference audit for task0."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
SOURCE = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000")
OUT = ROOT / "task0_cpu_sidecar_20260831_044430"
COLLECTOR = ROOT / "prospective_visual_context_collect.py"
SIM_WORKER = Path("/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py")
VISUAL_SERVER = ROOT / "visual_pi0_server.py"
WATCHER_OUT = ROOT / "task0_visual_generalization_20260831_040609"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(args: list[str]) -> dict:
    p = subprocess.run(args, text=True, capture_output=True, check=False)
    return {"argv": args, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def parse_gpu_snapshot() -> dict:
    gpu = run([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ])
    apps = run([
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ])
    processes = []
    if apps["returncode"] == 0:
        for line in apps["stdout"].splitlines():
            cols = [x.strip() for x in line.split(",", 3)]
            if len(cols) != 4:
                continue
            pid = cols[0]
            ps = run(["ps", "-fp", pid])
            command = ""
            lines = ps["stdout"].splitlines()
            if len(lines) > 1:
                command = lines[-1].split(None, 7)[-1] if len(lines[-1].split(None, 7)) >= 8 else lines[-1]
            processes.append({
                "pid": int(pid),
                "gpu_uuid": cols[1],
                "used_memory_MiB": int(cols[2]),
                "process_name": cols[3],
                "command": command,
            })
    return {
        "captured_at_utc": now(),
        "gpu_query": gpu,
        "compute_process_query": apps,
        "compute_processes": processes,
    }


def write_json(name: str, value: dict) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def line_evidence(path: Path, needles: list[str]) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        for needle in needles:
            if needle in line:
                rows.append({"path": str(path), "line": i, "text": line.strip(), "needle": needle})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = parse_gpu_snapshot()
    baseline.update({
        "audit_type": "read_only_gpu_baseline",
        "main_run_output_directory": str(SOURCE),
        "sidecar_cuda_context_created": False,
        "sidecar_gpu_process_started": False,
        "note": "Snapshot was obtained with nvidia-smi/ps only; no process was modified.",
    })
    write_json("TASK0_CPU_SIDECAR_GPU_BASELINE.json", baseline)

    dev_manifest = json.loads((SOURCE / "PROSPECTIVE_DEV_TARGET_MANIFEST.json").read_text())
    with (SOURCE / "PROSPECTIVE_CONTEXT_MANIFEST.csv").open(newline="") as f:
        context_rows = list(csv.DictReader(f))
    task0_dev = [r for r in context_rows if r["task"] == "0" and r["split"] == "DEV"]
    task0_targets = {k: v for k, v in dev_manifest["contexts"].items() if k.startswith("pv_dev_t0_")}
    expected = sum(len(v) for v in task0_targets.values())

    evidence = []
    evidence += line_evidence(SIM_WORKER, ['parse_env_cfg(ENV_ID, device="cuda:0"'])
    evidence += line_evidence(VISUAL_SERVER, ['pytorch_device="cuda:0"', '.to("cuda:0")', "torch.cuda.manual_seed_all"])
    source_hashes = {
        str(p): sha256(p)
        for p in [
            COLLECTOR,
            SIM_WORKER,
            VISUAL_SERVER,
            SOURCE / "PROSPECTIVE_CONTEXT_MANIFEST.csv",
            SOURCE / "PROSPECTIVE_DEV_TARGET_MANIFEST.json",
            SOURCE / "PROSPECTIVE_DEV_FORCE_PROTOCOL.json",
            SOURCE / "PROSPECTIVE_PROTOCOL_FREEZE.json",
        ]
    }

    microtest = {
        "audited_at_utc": now(),
        "status": "NOT_RUN_SAFETY_STOP_STATIC_PREFLIGHT_PROVES_CUDA_REQUIRED",
        "worker_launched": False,
        "scientific_branch_executed": False,
        "cuda_visible_devices_for_proposed_sidecar": "",
        "sidecar_cuda_context_created": False,
        "CPU_TASK0_DEV_COLLECTION_AVAILABLE": "NO",
        "authoritative_task0_dev_expected_branches": expected,
        "authoritative_task0_dev_contexts": task0_dev,
        "protocol_scope": "same-object/task held-out roots; not cross-object",
        "evidence": evidence,
        "source_hashes": source_hashes,
        "decision_reason": [
            "The exact authoritative Isaac environment is instantiated on cuda:0.",
            "The frozen visual feature service is configured for cuda:0 and transfers inputs to cuda:0.",
            "Strict CUDA disabling therefore cannot execute an authoritative branch.",
            "Changing device/rendering/visual inference would change the frozen implementation and would not be the preregistered protocol.",
            "Launching a failure-only worker would not validate branch semantics and could attempt a forbidden CUDA initialization, so it was not launched.",
        ],
        "safety_action": "WAIT for the existing main prospective pipeline to create collection_dev/task0.",
    }
    write_json("TASK0_CPU_COLLECTION_MICROTEST.json", microtest)

    dev_context = SOURCE / "collection_dev/task0/task0/context.csv"
    dev_branches = SOURCE / "collection_dev/task0/task0/branches.csv"
    dev_visual = SOURCE / "collection_dev/visual_alignment_worker.csv"
    ready = dev_context.exists() and dev_branches.exists() and dev_visual.exists()
    classification = {
        "classified_at_utc": now(),
        "classification": "CPU_ONLY_TASK0_COLLECTION_UNAVAILABLE",
        "current_validation_status": "WAITING_FOR_AUTHORITATIVE_TASK0_HELDOUT_DEV" if not ready else "AUTHORITATIVE_TASK0_HELDOUT_DEV_PRESENT",
        "task0_dev_ready": ready,
        "primary_scientific_result_available": False if not ready else None,
        "does_visual_context_help": "NOT YET EVALUABLE" if not ready else "PENDING EVALUATION",
        "offset_or_x_times_f": "NOT YET EVALUABLE" if not ready else "PENDING EVALUATION",
        "does_joint_add_independent_value": "NOT YET EVALUABLE" if not ready else "PENDING EVALUATION",
        "task0_gt_continuous_gate": "NOT YET EVALUABLE" if not ready else "PENDING EVALUATION",
        "reason": "No authoritative task0 DEV outcomes exist, and exact collection cannot run CPU-only without violating the frozen protocol.",
    }
    write_json("TASK0_FINAL_CLASSIFICATION.json", classification)

    omitted = {
        "recorded_at_utc": now(),
        "principle": "Do not create empty or synthetic metric artifacts that could be mistaken for evaluated results.",
        "omitted_until_authoritative_dev": [
            "TASK0_CPU_DEV_FROZEN_MANIFEST.csv",
            "TASK0_CPU_DEV_FROZEN_MANIFEST.json",
            "TASK0_CPU_DEV_RUN_MANIFEST.csv",
            "TASK0_CPU_DEV_COLLECTION_AUDIT.json",
            "TASK0_REAL_DEV_CURVES.csv",
            "TASK0_REAL_DEV_FRONTIERS.csv",
            "TASK0_DEV_PROBABILITY_METRICS.csv",
            "TASK0_DEV_FORCE_SHAPE_METRICS.csv",
            "TASK0_DEV_FRONTIER_METRICS.csv",
            "TASK0_DEV_CONTEXT_BREAKDOWN.csv",
            "TASK0_RESIDUAL_VS_FULL.csv",
            "TASK0_FULL_VS_JOINT.csv",
            "TASK0_GT_GATE.json",
        ],
        "reason": "CPU collection was not possible and authoritative task0 DEV has not landed.",
    }
    write_json("TASK0_SIDECAR_OMITTED_ARTIFACTS.json", omitted)

    final = parse_gpu_snapshot()
    baseline_pids = {p["pid"] for p in baseline["compute_processes"]}
    final_pids = {p["pid"] for p in final["compute_processes"]}
    final.update({
        "audit_type": "read_only_gpu_final_audit",
        "baseline_captured_at_utc": baseline["captured_at_utc"],
        "baseline_gpu_pids": sorted(baseline_pids),
        "final_gpu_pids": sorted(final_pids),
        "baseline_gpu_pids_still_present": sorted(baseline_pids & final_pids),
        "sidecar_gpu_processes": [],
        "sidecar_cuda_context_created": False,
        "sidecar_commands_launched": [],
        "main_run_processes_modified": False,
        "MAIN_GPU_EXPERIMENT_INTERFERED": "NO",
        "interpretation": "Only nvidia-smi and ps were used. GPU utilization/memory may vary naturally as the independent main pipeline progresses; no sidecar CUDA process or allocation was created.",
    })
    write_json("TASK0_CPU_SIDECAR_GPU_FINAL_AUDIT.json", final)

    report = f"""# STATUS

CPU-only authoritative task0 collection is unavailable with the frozen implementation. Current validation status: **{classification['current_validation_status']}**.

# GPU NON-INTERFERENCE

Did this side-car use or disturb GPU? **NO.** No collector, simulator, model, or CUDA process was launched. Only read-only `nvidia-smi`, `ps`, source, and manifest inspection was performed. `MAIN_GPU_EXPERIMENT_INTERFERED = NO`.

# TASK0 DEV COLLECTION

The frozen task0 DEV manifest contains {len(task0_dev)} held-out roots and {expected} branches: 9 forces from 3.00 N through 5.00 N, with 5 repeats per force. The authoritative DEV outcome files are not present yet.

The strict CPU microtest was stopped at preflight and no worker was launched. The exact simulator creates the environment on `cuda:0` (`{SIM_WORKER}:702`). The frozen visual service is also configured for `cuda:0` and transfers inference inputs to that device (`{VISUAL_SERVER}:41,72`). With `CUDA_VISIBLE_DEVICES=\"\"`, an authoritative branch cannot execute. Replacing those paths with a new CPU implementation would change the frozen protocol.

# SAME-OBJECT/TASK SCOPE

This is held-out-root evaluation within the same object/task distribution. It is **NOT** a cross-object generalization test. The frozen task0 DEV roots are root06 and root07.

# BASE

Held-out DEV metrics are not available.

# VISUAL RESIDUAL

Held-out DEV metrics are not available.

# FULL VISUAL

Held-out DEV metrics are not available.

# VISUAL JOINT

Held-out DEV metrics are not available.

# DOES VISUAL CONTEXT HELP?

**NOT YET EVALUABLE.** TRAIN ordering cannot answer the held-out question.

# OFFSET OR x × F?

**NOT YET EVALUABLE.** Residual-versus-Full requires authoritative DEV outcomes and aligned DEV visual features.

# DOES JOINT ADD INDEPENDENT VALUE?

**NOT YET EVALUABLE.** Full-versus-Joint requires authoritative DEV outcomes.

# FRONTIER

Not evaluable until the 90 authoritative branches are complete.

# UNDER-FORCE

Not evaluable until the real held-out frontiers exist.

# MONOTONICITY

Not evaluable on held-out DEV yet.

# PROBABILITY QUALITY

Not evaluable on held-out DEV yet.

# TASK0 GT GATE

**NOT YET EVALUABLE.** No gate result was fabricated from TRAIN data.

# FINAL CLASSIFICATION

**CPU_ONLY_TASK0_COLLECTION_UNAVAILABLE**

Operational validation state: **WAITING_FOR_AUTHORITATIVE_TASK0_HELDOUT_DEV**.

# WHAT TASK0 SUPPORTS

At present, task0 supports only the previously observed in-sample TRAIN ordering. It does not yet support a held-out visual, force-interaction, Joint-value, or GT-gate conclusion.

# WHAT TASK0 DOES NOT SUPPORT

- No cross-object claim.
- No unseen-task claim.
- Task0 only.
- Small independent-context population.
- No final multi-task conclusion yet.
- No held-out conclusion until authoritative DEV lands.

# NEXT ACTION

Leave the GPU experiment untouched and continue the existing CPU-only watcher. When the main preregistered pipeline produces `collection_dev/task0`, run frozen-model CPU inference and the preregistered task0 evaluation automatically. Do not replace DEV, retrain models, or tune the protocol.
"""
    (OUT / "FINAL_REPORT.md").write_text(report)

    files = sorted(p for p in OUT.iterdir() if p.name != "SHA256SUMS.txt")
    lines = [f"{sha256(p)}  {p.name}" for p in files if p.is_file()]
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(OUT), "classification": classification["classification"], "dev_ready": ready}, indent=2))


if __name__ == "__main__":
    main()
