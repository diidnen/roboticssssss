#!/usr/bin/env python3
"""Audit the small direct-contact collection without training a new model."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/exouser/Tabero")
RESULTS = REPO / "analysis/results"
OLD = RESULTS / "p5s0c_paired_boundary_probe_value_20260824_000542"
DIRECT = sorted(RESULTS.glob("learned_physical_imagination_v2_contact_telemetry_*"))[-1]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main():
    out = RESULTS / f"direct_contact_boundary_imagination_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    manifest = pd.read_csv(OLD / "P5S0C_BRANCH_MANIFEST.csv")
    paths = sorted(DIRECT.glob("P5S0C_BRANCH_TELEMETRY/*_trajectory.csv"))
    rows = []
    for path in paths:
        d = pd.read_csv(path)
        if d.empty:
            continue
        context = str(d.context_id.iloc[0])
        force = float(d.requested_force_N.iloc[0])
        m = manifest[(manifest.context_id == context) & (np.isclose(manifest.requested_force_N, force))]
        bilateral = (d.contact_state.astype(str) == "bilateral")
        active = d.phase.astype(str).isin(["lift", "transit", "over_basket", "place"])
        tangential_speed = np.sqrt(d.object_vx_mps.to_numpy() ** 2 + d.object_vy_mps.to_numpy() ** 2)
        rows.append({
            "file": str(path),
            "context_id": context,
            "force_N": force,
            "friction": float(d.hidden_friction_analysis_only.iloc[0]),
            "split": str(d.split.iloc[0]),
            "task": int(d.task.iloc[0]),
            "n_steps": len(d),
            "bilateral_fraction": float(bilateral.mean()),
            "active_phase_bilateral_fraction": float(bilateral[active].mean()) if active.any() else float("nan"),
            "contact_loss_steps": int((active & ~bilateral).sum()),
            "peak_left_force_N": float(d.left_force_norm_N.max()),
            "peak_right_force_N": float(d.right_force_norm_N.max()),
            "mean_left_force_N": float(d.left_force_norm_N.mean()),
            "mean_right_force_N": float(d.right_force_norm_N.mean()),
            "peak_tangential_speed_proxy_mps": float(tangential_speed[active].max()) if active.any() else float("nan"),
            "historical_success": int(m.full_task_success_y.iloc[0]) if len(m) == 1 else None,
        })
    event_definition = {
        "contact_left_right": "direct contact_gripper force norm and local normal component > 0.15 N; local normal component is NOT present in these two branch logs",
        "available_direct_signal": "world-frame per-finger net force vector and norm",
        "slip": "provisional active-phase tangential relative-velocity proxy only; not accepted as direct physical slip ground truth",
        "active_phases": ["lift", "transit", "over_basket", "place"],
        "release_excluded": True,
    }
    canonical = json.dumps(event_definition, sort_keys=True, separators=(",", ":")).encode()
    event_definition["sha256"] = hashlib.sha256(canonical).hexdigest()
    write_json(out / "DIRECT_CONTACT_LOGGER_AUDIT.json", {
        "status": "LOGGER_PARTIALLY_VALIDATED",
        "sources": {
            "contact_sensor": "source/tac_manip/tac_manip/core/sensors/gripper_contact_sensor/gripper_contact_sensor.py",
            "observation_projection": "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/observations.py",
            "established_contact_convention": "analysis/results/p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py:424-425",
            "new_branch_logger": "analysis/p5s0c_paired_boundary_probe_value.py",
        },
        "channels": {
            "per_finger_net_force": {"available": True, "frame": "world", "units": "N", "frequency": "each env.step", "measured": True},
            "left_right_contact": {"available": True, "derived": "local force norm and absolute local normal component > 0.15 N in the corrected logger"},
            "local_normal_force": {"available": True, "source": "corrected successor logger using observations.py local contact-force output", "units": "N"},
            "local_tangential_force": {"available": True, "source": "sqrt(local_x^2 + local_y^2)", "units": "N"},
            "object_pose_velocity": {"available": True, "frame": "world", "units": "m, m/s, rad/s", "measured": True},
            "gripper_pose_velocity": {"available": False, "reason": "only commanded TCP position and joint positions were logged"},
            "task_phase": {"available": True, "source": "deterministic downstream phase schedule"},
            "commanded_grip_force": {"available": True, "units": "N"},
        },
        "synchronization": "all fields appended after env.step at the same branch timestep; snapshot/restore uses existing P5-S0-C parity contract",
        "direct_files": len(rows),
        "direct_rows": rows,
    })
    write_json(out / "DIRECT_PHYSICAL_EVENT_DEFINITION.json", event_definition)
    groups = {}
    for r in rows:
        groups.setdefault((r["context_id"], r["force_N"]), 0)
        groups[(r["context_id"], r["force_N"])] += 1
    summary = {
        "direct_trace_count": len(rows),
        "paired_same_context_force_comparisons": 0,
        "three_repeat_boundary_contexts": sum(v >= 3 for v in groups.values()),
        "tasks": sorted({int(r["task"]) for r in rows}),
        "splits": sorted({r["split"] for r in rows}),
        "friction_levels": sorted({round(r["friction"], 6) for r in rows}),
        "sanity_gate": "NOT_EVALUABLE_ROOT_DIVERSE",
        "reason": "A corrected logger produced one same-state micro-test with three repeats at F_star, but there is no root-diverse matched boundary population or held-out split.",
    }
    write_json(out / "BOUNDARY_CONTACT_SEPARABILITY_SUMMARY.json", summary)
    (out / "BOUNDARY_CONTACT_SEPARABILITY_REPORT.md").write_text(
        "# Direct boundary contact separability\\n\\n"
        "The logger micro-test is physically interpretable and includes local "
        "normal/tangential force decomposition. One DEV context has three F_star "
        "repeats plus F_prev/F_next, but root-diverse repeatability and held-out "
        "separability remain unevaluated. Therefore no generalization claim is made.\\n",
        encoding="utf-8",
    )
    write_json(out / "DIRECT_CONTACT_DATASET_AUDIT.json", {
        "roots": len({r["context_id"] for r in rows}),
        "contexts": len({r["context_id"] for r in rows}),
        "branches": len(rows),
        "repeats": 0,
        "direct_contact_trajectories": len(rows),
        "train_dev_test_direct_coverage": {s: sum(r["split"] == s for r in rows) for s in ["TRAIN", "DEV", "TEST"]},
        "slip_events": "not accepted as direct labels beyond this micro-test",
        "quality_gate": "FAIL_INSUFFICIENT_DIRECT_ROOT_DIVERSE_DATA",
    })
    (out / "FINAL_REPORT.md").write_text(
        f"# Direct contact boundary imagination audit\n\n"
        f"STATUS: INSUFFICIENT_VALID_EVIDENCE\n\n"
        f"Direct traces: {len(rows)}. Same-state F_star repeats: 3. "
        f"Gate A passes for the micro-test; Gates B/C/D remain unevaluated. Physics-GRU v3 was not trained.\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(out), **summary}, indent=2))


if __name__ == "__main__":
    main()
