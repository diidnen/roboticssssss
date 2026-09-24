#!/usr/bin/env python3
"""Minimal root-interleaved execution-availability audit for frozen P7-B.

Only process lifetime and root/context execution order are varied.  The
frozen P7-B collector supplies the query implementation and Isaac setup.
No threshold, probe, controller, or scientific predicate is changed here.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path("/home/exouser/Tabero")
MAIN = REPO / "analysis/results/p7b_scientific_main_20260828_000729"
FROZEN_COLLECTOR = REPO / "analysis/p7b_gnp_physical_belief_force_planning.py"
FROZEN_PROTOCOL = REPO / "analysis/results/p7b_scientific_main_20260827_234017/P7B_PROTOCOL_IMMUTABLE_COPY.json"
EXPECTED = {
    "protocol": "7cbdd36a600204e06ddc635f68563731445756e2e24cd5e14712036a39a3941a",
    "collector": "2ef853b3d3a4ad4d1642332841dc5deb3daed71a7ca2ae8cb8d588668ec28337",
    "gripper": "51292d864a3bae436a0dbfeb2f737e2b27317f717b4e36011173241afa59956d",
    "clean_pilot_identity": "638045a19295db7721b0757030302dac27ffef2700ebbac9a8bccfd5a9daaeb5",
}
GRIPPER = REPO / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
TASK = 1
STRATUM = 2
ROOT_ORDER = [10100, 10112, 10100, 10116, 10108, 10112, 10111, 10100, 10116, 10100]
FRESH_ORDER = [10100, 10112, 10116, 10108, 10111]
SELECTION_RATIONALE = {
    10100: "strong success control: prior main query qualification 5/5",
    10108: "partial TRAIN root: prior main query qualification 3/5",
    10111: "zero-success TRAIN root adjacent to split cliff: 0/5",
    10112: "DEV representative immediately after TRAIN cliff: 0/5",
    10116: "TEST representative: 0/5",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(p: Path, x):
    p.write_text(json.dumps(x, indent=2, sort_keys=True, default=str) + "\n")


def load_contexts():
    with (MAIN / "P7B_CONTEXT_MANIFEST.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        if int(r["root_group_id"]) in set(ROOT_ORDER + FRESH_ORDER) and int(r["friction_stratum"]) == STRATUM:
            out[int(r["root_group_id"])] = r
    return out


def preflight():
    nodes = ["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset"]
    node_info = {}
    for n in nodes:
        p = Path(n)
        try:
            st = p.stat()
            node_info[n] = {"present": True, "mode": oct(st.st_mode & 0o777), "uid": st.st_uid, "gid": st.st_gid}
        except OSError as e:
            node_info[n] = {"present": False, "error": repr(e)}
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
    du = shutil.disk_usage(REPO)
    df_i = subprocess.run(["df", "-i", "-P", str(REPO)], capture_output=True, text=True)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "user": os.environ.get("USER"), "kernel": platform.release(), "python": sys.executable,
        "cwd": os.getcwd(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"), "device_nodes": node_info,
        "nvidia_smi_returncode": smi.returncode, "nvidia_smi": smi.stdout.strip(), "nvidia_smi_stderr": smi.stderr.strip(),
        "disk_free_bytes": du.free, "disk_total_bytes": du.total, "inode_check": df_i.stdout.strip(),
        "hashes": {"protocol": sha256(FROZEN_PROTOCOL), "collector": sha256(FROZEN_COLLECTOR), "gripper": sha256(GRIPPER)},
    }


def load_frozen_module():
    spec = importlib.util.spec_from_file_location("p7b_frozen_for_availability_audit", FROZEN_COLLECTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen collector")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def vec(x, n=3):
    try:
        return [float(x[i]) for i in range(n)]
    except Exception:
        return [None] * n


def run_events(protocol: dict, out: Path, events: list[dict], mode: str):
    """Execute query-only events in one Isaac process."""
    frozen = load_frozen_module()
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    frozen.prepare_isaac_runtime_env()
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    process_start = time.monotonic()
    rows = []
    result_path = out / "AUDIT_EXECUTION_RESULTS.csv"
    if result_path.exists():
        with result_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    try:
        import gymnasium as gym
        import torch
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        r2, r1, p6, p4 = frozen.import_runtime_modules(out, TASK)
        setup_task_objects("libero_object", TASK)
        cfg = parse_env_cfg("Isaac-Libero-Franka-Hybrid-Tactile-v0", device="cuda:0", num_envs=1)
        cfg.episode_length_s = 45.0
        env = gym.make("Isaac-Libero-Franka-Hybrid-Tactile-v0", cfg=cfg).unwrapped
        library = r2.recover_library()
        recipe = next(x for x in frozen.read_csv(frozen.P6G1R2_OUT / "P6G1R2_RECIPE_CANDIDATES.csv")
                       if int(x["task"]) == TASK and x["recipe_id"] == "t1_G2_R23_ORIGINAL_P6G1")
        # JSON object keys are strings; normalize once so the pre-registered
        # integer root ids are not confused with a scientific failure.
        context_map = {int(k): v for k, v in protocol["contexts"].items()}
        for event in events:
            root = int(event["root_group_id"]); base = context_map[root]
            logical_cid = base["context_id"]
            order_index = int(event["execution_order_index"])
            audit_cid = f"audit_{mode}_o{order_index:02d}_{logical_cid}"
            started = time.monotonic()
            env.reset(seed=root)
            frozen.r1 = r1
            r1.p6g1.settle_root_before_hash(env, p6, p4, frozen.ROOT_SETTLE_STEPS)
            root_state = env.scene.get_state(is_relative=True)
            root_hash = r1.root_hash(p6, env)
            frozen.restore_scene_state_stable(env, root_state, torch.tensor([0], device=env.device))
            sq, sqh, qrec, stage = frozen.stage_and_query(
                env, r2, r1, p6, p4, library, recipe, root_state, root_hash,
                float(base["hidden_friction_analysis_only"]), audit_cid, out)
            qrows = []
            qpath = out / "P7B_QUERY_TELEMETRY" / f"{audit_cid}.csv"
            if qpath.exists():
                with qpath.open(newline="") as f: qrows = list(csv.DictReader(f))
            first = qrows[0] if qrows else {}
            # stage_and_query returns the inner stage_result, while the
            # authoritative stage telemetry JSON also contains the realized
            # EEF/object poses.  Read those poses from the raw telemetry so a
            # missing wrapper key cannot be mistaken for a scientific failure.
            stage_path = out / "P7B_STAGE_TELEMETRY" / f"{audit_cid}.json"
            stage_record = json.loads(stage_path.read_text()) if stage_path.exists() else {}
            eef_stage = stage_record.get("eef_pose_base_after_stage", [None] * 7)
            obj_stage = stage_record.get("object_pose_base_after_stage", {}).get("position_m", [None] * 3)
            def fnum(d, k):
                try: return float(d.get(k, ""))
                except Exception: return None
            tcp_delta = [fnum(first, "eef_x_actual"), fnum(first, "eef_y_actual"), fnum(first, "eef_z_actual")]
            obj_first = [fnum(first, "object_x_w"), fnum(first, "object_y_w"), fnum(first, "object_z_w")]
            tcp_delta = [tcp_delta[i] - float(eef_stage[i]) if tcp_delta[i] is not None and eef_stage[i] is not None else None for i in range(3)]
            obj_delta = [obj_first[i] - float(obj_stage[i]) if obj_first[i] is not None and obj_stage[i] is not None else None for i in range(3)]
            sres = stage_record.get("stage_result", stage if isinstance(stage, dict) else {})
            row = {
                "timestamp_start_utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
                "execution_order_index": order_index, "pid": os.getpid(),
                "session_age_s": time.monotonic() - process_start, "event_elapsed_s": time.monotonic() - started,
                "root_group_id": root, "split": base["split"], "context_id": logical_cid,
                "audit_context_id": audit_cid, "friction_stratum": STRATUM,
                "hidden_friction_context_value": base["hidden_friction_analysis_only"],
                "root_restore_hash": root_hash, "root_restore_parity": 1,
                "stage_staging_validity": sres.get("staging_validity"),
                "stage_collision_free": sres.get("collision_free"), "stage_ik_success": sres.get("ik_success"),
                "stage_no_fingertip_contact_before_invocation": sres.get("no_fingertip_contact_before_invocation"),
                "stage_object_disturbance_m": sres.get("object_disturbance_m"),
                "stage_object_rotation_disturbance_rad": sres.get("object_rotation_disturbance_rad"),
                "stage_position_error_m": sres.get("position_error_m"), "stage_orientation_error_rad": sres.get("orientation_error_rad"),
                "stage_eef_x": eef_stage[0], "stage_eef_y": eef_stage[1], "stage_eef_z": eef_stage[2],
                "stage_object_x": obj_stage[0], "stage_object_y": obj_stage[1], "stage_object_z": obj_stage[2],
                "stage_eef_object_dx_m": float(eef_stage[0]) - float(obj_stage[0]),
                "stage_eef_object_dy_m": float(eef_stage[1]) - float(obj_stage[1]),
                "stage_eef_object_dz_m": float(eef_stage[2]) - float(obj_stage[2]),
                "stage_eef_object_xy_distance_m": float(np.hypot(float(eef_stage[0])-float(obj_stage[0]), float(eef_stage[1])-float(obj_stage[1])),),
                "first_approach_step": first.get("step"), "first_approach_tcp_delta_x_m": tcp_delta[0],
                "first_approach_tcp_delta_y_m": tcp_delta[1], "first_approach_tcp_delta_z_m": tcp_delta[2],
                "first_approach_object_delta_x_m": obj_delta[0], "first_approach_object_delta_y_m": obj_delta[1],
                "first_approach_object_delta_z_m": obj_delta[2], "first_step_dropped": first.get("dropped"),
                "first_step_contact_state": first.get("contact_state"), "first_step_contact_left": first.get("contact_left"),
                "first_step_contact_right": first.get("contact_right"), "first_step_measured_fn": first.get("measured_fn"),
                "first_step_measured_ft": first.get("measured_ft"), "first_step_target_normal_force": first.get("target_normal_force"),
                "first_step_gripper_joint_pos_0": first.get("gripper_joint_pos_0"), "first_step_gripper_joint_pos_1": first.get("gripper_joint_pos_1"),
                "telemetry_rows": len(qrows), "telemetry_last_phase": qrows[-1].get("phase") if qrows else None,
                "first_drop_step": next((r.get("step") for r in qrows if r.get("dropped") == "1"), None),
                "query_excitation_entered": int(any(str(r.get("phase", "")).startswith("probe") for r in qrows)),
                "bilateral_contact_entered": int(any(r.get("contact_state") == "bilateral" for r in qrows)),
                "query_qualified": qrec.get("query_qualified"), "query_failure_reason": qrec.get("query_failure_reason"),
                "stop_reason": qrec.get("stop_reason"), "return_state_valid": qrec.get("return_state_valid"),
                "return_position_error_mm": qrec.get("return_position_error_mm"), "major_disturbance": qrec.get("major_disturbance"),
                "runtime_status": "SCIENTIFIC_QUERY_COMPLETED", "infrastructure_status": "VALID",
            }
            rows.append(row)
            with result_path.open("w", newline="") as f:
                fields = list(rows[0].keys()); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
            print(json.dumps({"order": order_index, "mode": mode, "root": root, "qualified": qrec.get("query_qualified"), "drop": qrec.get("drop"), "age_s": row["session_age_s"]}), flush=True)
    except Exception as exc:
        dump(out / f"AUDIT_PROCESS_ERROR_{mode}.json", {"mode": mode, "pid": os.getpid(), "error": repr(exc), "traceback": traceback.format_exc(), "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        if env is not None:
            try: env.close()
            except Exception: pass
        try: app.close()
        except Exception: pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--run-long", action="store_true")
    ap.add_argument("--run-fresh-index", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.freeze:
        if sha256(FROZEN_PROTOCOL) != EXPECTED["protocol"] or sha256(FROZEN_COLLECTOR) != EXPECTED["collector"] or sha256(GRIPPER) != EXPECTED["gripper"]:
            raise SystemExit("frozen hash mismatch; refusing audit")
        contexts = load_contexts()
        missing = sorted(set(ROOT_ORDER + FRESH_ORDER) - set(contexts))
        if missing: raise SystemExit(f"missing selected contexts: {missing}")
        protocol = {
            "name": "P7-B root-interleaved execution-availability audit",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_method_change": "NONE",
            "purpose": "separate root-dependent query handoff failure from temporal/session degradation",
            "frozen_method": {"collector": str(FROZEN_COLLECTOR), "protocol": str(FROZEN_PROTOCOL), "task": TASK, "stratum": STRATUM,
                              "probe_and_predicates_unchanged": True, "branches": False, "query_only": True},
            "selected_roots": [{"root_group_id": r, "split": contexts[r]["split"], "context_id": contexts[r]["context_id"],
                                "friction_stratum": STRATUM, "hidden_friction_context_value": contexts[r]["hidden_friction_analysis_only"],
                                "prior_query_qualification": contexts[r]["query_qualified"], "rationale": SELECTION_RATIONALE[r]} for r in sorted(contexts)],
            "long_lived_execution_order": [{"execution_order_index": i, "root_group_id": r, "context_id": contexts[r]["context_id"], "split": contexts[r]["split"]} for i, r in enumerate(ROOT_ORDER)],
            "fresh_process_execution_order": [{"execution_order_index": i, "root_group_id": r, "context_id": contexts[r]["context_id"], "split": contexts[r]["split"]} for i, r in enumerate(FRESH_ORDER)],
            "contexts": contexts,
            "controls": {"same_frozen_query": True, "same_task": True, "same_stratum": True, "root_major_order_removed": True,
                         "long_lived_repeated_success_control": [i for i, r in enumerate(ROOT_ORDER) if r == 10100],
                         "fresh_process_one_context_per_process": True, "no_scientific_retries": True},
            "hashes": {"protocol": sha256(FROZEN_PROTOCOL), "collector": sha256(FROZEN_COLLECTOR), "gripper": sha256(GRIPPER), "clean_pilot_identity": EXPECTED["clean_pilot_identity"]},
        }
        dump(out / "IMMUTABLE_AUDIT_PROTOCOL.json", protocol)
        (out / "IMMUTABLE_AUDIT_PROTOCOL_IMMUTABLE_COPY.json").write_text((out / "IMMUTABLE_AUDIT_PROTOCOL.json").read_text())
        audit_hash = sha256(out / "IMMUTABLE_AUDIT_PROTOCOL_IMMUTABLE_COPY.json")
        (out / "IMMUTABLE_AUDIT_PROTOCOL_SHA256.txt").write_text(audit_hash + "\n")
        dump(out / "ENVIRONMENT_PREFLIGHT.json", preflight())
        print(json.dumps({"out": str(out), "audit_protocol_sha256": audit_hash, "roots": sorted(contexts), "long": ROOT_ORDER, "fresh": FRESH_ORDER}, indent=2))
        return
    protocol = json.loads((out / "IMMUTABLE_AUDIT_PROTOCOL_IMMUTABLE_COPY.json").read_text())
    if args.run_long:
        run_events(protocol, out, protocol["long_lived_execution_order"], "long_lived")
    elif args.run_fresh_index is not None:
        event = protocol["fresh_process_execution_order"][args.run_fresh_index]
        run_events(protocol, out, [event], f"fresh_{args.run_fresh_index}")
    else:
        raise SystemExit("choose --freeze, --run-long, or --run-fresh-index")


if __name__ == "__main__":
    main()
