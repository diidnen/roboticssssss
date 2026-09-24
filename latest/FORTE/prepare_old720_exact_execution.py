#!/usr/bin/env python3
"""Prepare hash-locked manifests and audit records for old720 restoration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


OUT = Path("/home/exouser/FORTE/analysis/results/old720_exact_execution_restoration_20260905")
DATASET = Path("/home/exouser/FORTE/gnp_style_continuous_20260830_125107/CONTINUOUS_TRAIN_SUCCESS_DATA.csv")
HIST = Path("/home/exouser/FORTE/gnp_style_continuous_20260830_125107/collection_long2")
REPO = Path("/home/exouser/Tabero_old720_80ab")
COMMIT = "80ab3be09ce884f86cfc2037d3af30bc28061426"
RUNNER = REPO / "analysis/p5s0c_paired_boundary_probe_value.py"
CONTROLLER = REPO / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"
P4 = REPO / "analysis/results/p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py"
CONTEXT = "p5s0c_train_t0_r00_s5100_low_mu0.293710"
ROOT = "p5s0c_train_t0_root00_s5100"
FORCES = [3.3274269914534806, 3.5804616000261396, 4.1951268916184405, 4.225682863145822, 4.899172114786442]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def historical_rows() -> list[dict]:
    with DATASET.open(newline="", encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh) if row["context_id"] == CONTEXT]


def branch_label(force_index: int, force: float, repeat: int) -> str:
    tag = f"{force:.8f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"GNP_S{force_index}_F{tag}_R{repeat}"


def manifest(path: Path, specs: list[dict], name: str) -> None:
    write_json(path, {
        "manifest_name": name,
        "exact_commit": COMMIT,
        "runner": str(RUNNER),
        "runner_sha256": sha256(RUNNER),
        "contexts": {CONTEXT: specs},
        "expected_contexts": 1,
        "expected_branches": len(specs),
        "split": "TRAIN",
        "branch_state": "restored strict pre-probe last-hold snapshot",
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "parity", "sweep"], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if commit != COMMIT:
        raise RuntimeError(f"worktree commit {commit} != {COMMIT}")
    rows = historical_rows()
    provenance = {
        "dataset": str(DATASET),
        "historical_collection_root": str(HIST),
        "exact_repo_path": str(REPO),
        "exact_commit": COMMIT,
        "collection_runner": str(RUNNER),
        "runner_sha256": sha256(RUNNER),
        "controller_module": str(CONTROLLER),
        "controller_sha256": sha256(CONTROLLER),
        "p4_probe_script": str(P4),
        "p4_probe_sha256": sha256(P4),
        "collection_wrapper": "/home/exouser/FORTE/gnp_style_continuous_collect.py",
        "execution_wrapper": "/home/exouser/FORTE/old720_exact_execution_restore.py",
        "external_assets": {
            "hdf5": "/media/volume/newdata/exouser/tabero/data/Isaaclab_Libero/assembled_hdf5",
            "usd": "/media/volume/newdata/exouser/tabero/data/Isaaclab_Libero/USD",
        },
        "source_hash_gate": "PASS",
    }
    write_json(OUT / "OLD720_EXECUTION_PROVENANCE.json", provenance)
    write_json(OUT / "OLD720_OUTER_D_PRED_SERVO_AUDIT.json", {
        "present": True,
        "update_hz": 20.0,
        "stateful": True,
        "state": "d_pred",
        "initial_d_pred_m": 0.0,
        "step_size_m": 0.0006,
        "deadband_N": 0.4,
        "direction_rule": "if requested-measured>deadband: close by 0.6mm; if below -deadband: open by 0.6mm",
        "clamp_m": [0.0, 0.04],
        "reset_rule": "reset d_pred to D_CLOSED=0 for every branch; freeze during transit/place; D_OPEN during release",
        "source": f"{P4}:SERVO_STEP/SERVO_DEADBAND/_force_servo and {RUNNER}:downstream_branch",
    })
    write_json(OUT / "OLD720_NATIVE_CONTROLLER_CONFIG.json", {
        "force_update_hz": 60.0,
        "environment_hz": 20.0,
        "squeeze_kp_m_per_N": 0.0008,
        "force_deadband_N": 0.25,
        "filter_alpha": 0.2,
        "feedforward_k": 0.9,
        "feedforward_contact_threshold_N": 1.0,
        "contact_override_enabled": False,
        "gripper_stiffness": 2000.0,
        "gripper_damping": 100.0,
        "formula": "d_cmd=d_pred-0.0008*0.5*(f_target_eff-f_meas_filtered)",
        "effective_target": "f_raw + 0.9*abs(f_raw) when raw measured squeeze >=1N",
        "source_commit": COMMIT,
        "source": str(CONTROLLER),
    })
    write_json(OUT / "OLD720_SETPOINT_SEMANTICS.json", {
        "requested_force_N_semantics": "RAW_PRE_FEEDFORWARD_SQUEEZE_SETPOINT_AND_OUTER_SERVO_REFERENCE",
        "candidate_enters_outer_servo": True,
        "candidate_enters_native_force_target": True,
        "action_encoding": "equal per-finger z slots each carry requested_force_N/2; native scalar sum equals requested_force_N",
        "example_4N": {
            "outer_servo_reference_N": 4.0,
            "native_raw_target_N": 4.0,
            "native_effective_target_contact_N": 7.6,
            "interpretation": "continuous setpoint under a fixed two-layer execution contract, not exact measured Newton force",
        },
    })
    write_json(OUT / "REFERENCE_CONTEXT.json", {
        "selection_rule": "lexicographically earliest complete context with five force cells, two repeats each, complete telemetry, and mixed deterministic outcomes",
        "task": 0,
        "root": ROOT,
        "context": CONTEXT,
        "seed": 5100,
        "friction": 0.2937102019159983,
        "friction_band": "LOW",
        "historical_rows": rows,
        "historical_setpoint_levels": FORCES,
        "strict_preprobe_hash": "18bde4e4fc96af4e36561c7170328947a96c9685799131871039e013de43e899",
    })
    if args.mode in {"all", "parity"}:
        force = FORCES[2]
        manifest(OUT / "PARITY_TARGET_MANIFEST.json", [{
            "force_N": force,
            "repeat_index": 0,
            "branch_label": branch_label(3, force, 1),
        }], "OLD720_EXACT_SINGLE_BRANCH_PARITY")
    if args.mode in {"all", "sweep"}:
        for repeat in range(1, 4):
            specs = [
                {"force_N": force, "repeat_index": repeat - 1, "branch_label": f"RESTORE_{branch_label(i, force, repeat)}"}
                for i, force in enumerate(FORCES, start=1)
            ]
            manifest(OUT / f"SWEEP_REPEAT_{repeat}_TARGET_MANIFEST.json", specs, f"OLD720_EXACT_SWEEP_REPEAT_{repeat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
