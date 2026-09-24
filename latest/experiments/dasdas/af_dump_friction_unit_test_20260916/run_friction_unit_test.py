#!/usr/bin/env python3
"""Minimal finger–deskbin Coulomb unit test.

No query, dump, VLA, balls-in-band, or official task success.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import sapien

HERE = Path(__file__).resolve().parent
OLD = Path("/media/volume/dasdas/exouser/af_dump_original_geom_finger_friction_sanity_20260916")
sys.path.insert(0, str(OLD))

import run_original_geom_finger_sweep as S
from envs.utils import ArmTag
from force_realize import (
    LiftSqueezeHold,
    capture_stock_drives,
    install_lift_squeeze,
    realize_commanded_force,
    restore_stock_drives,
    uninstall_lift_squeeze,
)
from run_liftstyle_context import PrefixSnapshot, scripted_establish_grasp

CRIT = json.loads((HERE / "CRITERION.json").read_text())
DESK_MU = 0.30
LEVELS = {
    "low": {"mu_finger": 0.425, "mu_eff": 0.3625},
    "mid": {"mu_finger": 0.575, "mu_eff": 0.4375},
    "high": {"mu_finger": 0.85, "mu_eff": 0.575},
    "near_zero": {"mu_finger": 0.001, "mu_eff": 0.1505, "paper": False},
}
SEED = 200014
EPISODE = 1
FIXTURE_FINGER_MU = 0.575
SLIP_DISP = float(CRIT["slip_disp_m"])
SLIP_STREAK = int(CRIT["slip_streak_steps"])
T_STEP = float(CRIT["T_step_N"])
T_MAX = float(CRIT["T_max_N"])
HOLD = int(CRIT["hold_steps_per_T"])
AXIS = 1


def dump(path: Path, value) -> None:
    S.dump_json(path, value)


def desk_body(task):
    entity = task.deskbin.actor if hasattr(task.deskbin, "actor") else task.deskbin
    body = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    if body is None:
        raise RuntimeError("deskbin has no dynamic body")
    return body


def finger_centroid(task) -> np.ndarray:
    worlds = []
    for joint, _, _ in task.robot.left_gripper:
        link = joint.child_link
        pose = None
        for attr in ("get_pose", "pose", "get_entity_pose"):
            val = getattr(link, attr, None)
            try:
                pose = val() if callable(val) else val
                if pose is not None:
                    break
            except Exception:
                continue
        if pose is None or not hasattr(pose, "p"):
            continue
        worlds.append(np.asarray(pose.p, dtype=float))
    if not worlds:
        raise RuntimeError("no finger poses")
    return np.mean(np.stack(worlds, axis=0), axis=0)


def park_balls(task) -> None:
    for sphere in getattr(task, "sphere_lst", []) or []:
        raw = sphere.actor if hasattr(sphere, "actor") else sphere
        try:
            raw.set_pose(sapien.Pose([8.0, 8.0, 8.0], [1, 0, 0, 0]))
        except Exception:
            pass
        body = raw.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if body is not None:
            body.disable_gravity = True
            body.set_linear_velocity([0, 0, 0])
            body.set_angular_velocity([0, 0, 0])


def prepare_fixture_body(task) -> None:
    body = desk_body(task)
    body.disable_gravity = True
    body.set_linear_velocity([0, 0, 0])
    body.set_angular_velocity([0, 0, 0])
    park_balls(task)


def slim_geom(geom: dict) -> dict:
    fingers = {}
    for name, row in (geom.get("per_finger") or {}).items():
        if not row.get("present"):
            continue
        fingers[name] = {
            "n_shapes": row.get("n_shapes"),
            "shape_ids": row.get("shape_ids"),
            "axes": row.get("axes"),
            "fn": row.get("fn"),
            "normals": row.get("normals"),
            "wedging": row.get("multi_hull_wedging"),
        }
    return {
        "shape_ids": geom.get("contact_shape_ids"),
        "pair_dist": geom.get("pair_dist"),
        "aperture": geom.get("aperture"),
        "nl": geom.get("nl"),
        "nr": geom.get("nr"),
        "squeeze": geom.get("squeeze"),
        "obj_xyz": geom.get("obj_xyz"),
        "obj_q": geom.get("obj_q"),
        "obj_z": geom.get("obj_z"),
        "table_mu_eff": geom.get("table_deskbin_mu_eff"),
        "finger_mu_eff": geom.get("finger_deskbin_mu_eff"),
        "wedging": geom.get("multi_hull_wedging_any_finger"),
        "nx_pz": geom.get("split_like_nx_pz"),
        "opposing": bool(
            {"-x", "+x"} <= set(ax for row in fingers.values() for ax in (row.get("axes") or []))
        ),
        "n_multi_hull_fingers": int(sum(1 for row in fingers.values() if int(row.get("n_shapes") or 0) >= 2)),
        "fingers": fingers,
    }


def build_fixture():
    task = S.make_task(SEED, EPISODE)
    stock = capture_stock_drives(task)
    prepared = S.prepare_condition(task, DESK_MU, FIXTURE_FINGER_MU)
    if int(prepared["geometry"]["deskbin_id"]) != 10:
        print(json.dumps({"warn": "deskbin_id_not_10", "id": prepared["geometry"]["deskbin_id"]}), flush=True)
    task.activate_activeforcing_candidate_force()
    scripted_establish_grasp(task)
    window = []
    for _ in range(20):
        task.scene.step()
        window.append(S.pack_state(task))
    gate = S.classify_pre(window)
    print(json.dumps({"event": "PRE", "gate": gate["gate"], "cr": gate["window_cr"]}), flush=True)
    if gate["gate"] != "VALID":
        raise RuntimeError(f"fixture PRE not VALID: {gate}")
    task.move(task.move_by_displacement(arm_tag=ArmTag("left"), z=0.08, move_axis="arm"))
    prepare_fixture_body(task)
    for _ in range(50):
        task.scene.step()
    prepare_fixture_body(task)
    geom = S.contact_geometry(task)
    if geom.get("table_deskbin_mu_eff"):
        print(json.dumps({"warn": "still_table_contact", "mu": geom["table_deskbin_mu_eff"]}), flush=True)
    snapshot = PrefixSnapshot(task)
    fixture = {
        "seed": SEED,
        "episode_id": EPISODE,
        "deskbin_id": prepared["geometry"]["deskbin_id"],
        "n_shapes": prepared["geometry"]["n_shapes"],
        "pre_gate": gate,
        "audit": prepared["audit"],
        "pose": {
            "deskbin_xyz": geom["obj_xyz"],
            "deskbin_q": geom["obj_q"],
            "obj_z": geom["obj_z"],
            "pair_dist": geom["pair_dist"],
            "aperture": geom["aperture"],
        },
        "contact": slim_geom(geom),
        "fixture_finger_mu": FIXTURE_FINGER_MU,
        "deskbin_mu": DESK_MU,
    }
    dump(HERE / "FIXTURE.json", fixture)
    return {
        "task": task,
        "stock": stock,
        "snapshot": snapshot,
        "fixture": fixture,
    }


def realize_and_check(task, stock, force_n: float) -> dict:
    realize = realize_commanded_force(task, force_n, stock)
    window = []
    holder = LiftSqueezeHold(task, force_n, stock)
    task._af_lift_holder = holder
    task._af_lift_inner_active = True
    for _ in range(40):
        task.scene.step()
        contact = task.get_actor_gripper_contact_forces(task.deskbin, "left")
        window.append(
            {
                "single": float(contact["single_finger_normal_force_n"]),
                "bilateral": bool(contact["bilateral_contact"]),
            }
        )
    singles = [row["single"] for row in window]
    mean = float(np.mean(singles))
    std = float(np.std(singles))
    return {
        "realize": realize,
        "measured_single_mean_n": mean,
        "measured_single_std_n": std,
        "measured_bilateral_mean_n": 2.0 * mean,
        "bilateral_frac": float(np.mean([row["bilateral"] for row in window])),
        "stable": bool(mean >= 0.35 * float(force_n) and std <= max(0.35, 0.5 * mean)),
        "force_cmd": float(force_n),
    }


def ramp_slip(task, stock, force_n: float) -> dict:
    body = desk_body(task)
    body.disable_gravity = True
    c0 = finger_centroid(task)
    p0 = np.asarray(task.deskbin.get_pose().p, dtype=float)
    rel0_y = float((p0 - c0)[AXIS])
    geom0 = slim_geom(S.contact_geometry(task))
    trace = []
    slipped = False
    t_slip = None
    streak = 0
    t_values = [round(T_STEP * i, 4) for i in range(int(round(T_MAX / T_STEP)) + 1)]
    for t_cmd in t_values:
        max_s = 0.0
        last_s = 0.0
        last_v = 0.0
        for _ in range(HOLD):
            body.add_force_torque([0.0, float(t_cmd), 0.0], [0.0, 0.0, 0.0], "force")
            task.scene.step()
            c = finger_centroid(task)
            p = np.asarray(task.deskbin.get_pose().p, dtype=float)
            s = float((p - c)[AXIS] - rel0_y)
            v = float(np.asarray(body.get_linear_velocity(), dtype=float)[AXIS])
            last_s = s
            last_v = v
            max_s = max(max_s, abs(s))
            if abs(s) >= SLIP_DISP:
                streak += 1
            else:
                streak = 0
            if streak >= SLIP_STREAK and not slipped:
                slipped = True
                t_slip = float(t_cmd)
                break
        contact = task.get_actor_gripper_contact_forces(task.deskbin, "left")
        trace.append(
            {
                "T": float(t_cmd),
                "s_y": last_s,
                "max_abs_s": max_s,
                "v_y": last_v,
                "nl": float((contact.get("per_finger_normal_force_n") or {}).get(list(contact.get("per_finger_normal_force_n") or {})[:1][0], 0.0))
                if contact.get("per_finger_normal_force_n")
                else float(contact.get("single_finger_normal_force_n") or 0.0),
                "single": float(contact.get("single_finger_normal_force_n") or 0.0),
                "bilateral": bool(contact.get("bilateral_contact")),
                "streak": streak,
            }
        )
        print(json.dumps({"event": "T", "T": t_cmd, "s_mm": 1000.0 * last_s, "slip": slipped}), flush=True)
        if slipped:
            break
    geom1 = slim_geom(S.contact_geometry(task))
    return {
        "slipped": slipped,
        "T_slip": t_slip,
        "T_max_tested": float(trace[-1]["T"]) if trace else 0.0,
        "s_at_end_mm": 1000.0 * float(trace[-1]["s_y"]) if trace else None,
        "geom_before_ramp": geom0,
        "geom_at_end": geom1,
        "mode_change": {
            "shape_ids_changed": geom0.get("shape_ids") != geom1.get("shape_ids"),
            "wedging_appeared": bool(geom1.get("wedging") and not geom0.get("wedging")),
            "lost_opposing": bool(geom0.get("opposing") and not geom1.get("opposing")),
            "lost_bilateral": bool(not geom1.get("fingers")),
        },
        "trace": trace,
    }


def run_case(ctx, label: str, force_n: float) -> dict:
    spec = LEVELS[label]
    ctx["snapshot"].restore()
    prepare_fixture_body(ctx["task"])
    prepared = S.prepare_condition(ctx["task"], DESK_MU, spec["mu_finger"])
    audit = prepared["audit"]
    squeeze = realize_and_check(ctx["task"], ctx["stock"], force_n)
    geom_hold = slim_geom(S.contact_geometry(ctx["task"]))
    n_single = float(squeeze["measured_single_mean_n"])
    theory = 2.0 * float(spec["mu_eff"]) * n_single
    ramp = ramp_slip(ctx["task"], ctx["stock"], force_n)
    uninstall_lift_squeeze(ctx["task"])
    restore_stock_drives(ctx["stock"])
    row = {
        "label": label,
        "mu_finger": spec["mu_finger"],
        "mu_deskbin": DESK_MU,
        "mu_eff": spec["mu_eff"],
        "mu_eff_runtime": audit.get("mu_eff"),
        "finger_mu_runtime": audit.get("finger_mu"),
        "deskbin_mu_runtime": audit.get("deskbin_mu"),
        "table_mu_runtime": audit.get("table_mu"),
        "ball_mu_runtime": audit.get("ball_mu"),
        "F_cmd": float(force_n),
        "squeeze": squeeze,
        "geom_hold": geom_hold,
        "T_theory": theory,
        "T_slip": ramp["T_slip"],
        "slipped": ramp["slipped"],
        "ramp": {k: v for k, v in ramp.items() if k != "trace"},
        "trace": ramp["trace"],
    }
    dump(HERE / "cases" / f"{label}_F{force_n:.2f}.json", row)
    print(
        json.dumps(
            {
                "event": "CASE",
                "label": label,
                "F": force_n,
                "mu_eff": spec["mu_eff"],
                "N": n_single,
                "T_theory": theory,
                "T_slip": ramp["T_slip"],
                "slipped": ramp["slipped"],
                "runtime_mu": {
                    "finger": audit.get("finger_mu"),
                    "deskbin": audit.get("deskbin_mu"),
                    "mu_eff": audit.get("mu_eff"),
                },
            }
        ),
        flush=True,
    )
    return row


def verdict(mu_rows: list[dict], f_rows: list[dict], nz: dict | None) -> dict:
    def t_of(rows, key, val):
        for row in rows:
            if row[key] == val:
                return row["T_slip"]
        return None

    t_low = t_of(mu_rows, "label", "low")
    t_mid = t_of(mu_rows, "label", "mid")
    t_high = t_of(mu_rows, "label", "high")
    by_f = {row["F_cmd"]: row["T_slip"] for row in f_rows}
    mu_ok = all(x is not None for x in (t_low, t_mid, t_high)) and t_low < t_mid < t_high
    mu_weak = all(x is not None for x in (t_low, t_mid, t_high)) and t_low <= t_mid <= t_high and t_low < t_high
    f_vals = [by_f.get(f) for f in (0.5, 1.0, 2.0)]
    f_ok = all(x is not None for x in f_vals) and f_vals[0] < f_vals[1] < f_vals[2]
    f_weak = all(x is not None for x in f_vals) and f_vals[0] <= f_vals[1] <= f_vals[2] and f_vals[0] < f_vals[2]
    wedge = any((row.get("geom_hold") or {}).get("wedging") or (row.get("ramp") or {}).get("mode_change", {}).get("wedging_appeared") for row in mu_rows + f_rows)
    nz_low = None
    if nz and nz.get("T_slip") is not None and t_low is not None:
        nz_low = nz["T_slip"] < t_low
    if mu_ok and f_ok and (nz_low is not False):
        label = "PASS"
        why = "T_slip rises with μ_eff and with normal force. Finger–deskbin friction is doing Coulomb work."
    elif (mu_ok or mu_weak or f_ok or f_weak) and not (t_low is not None and t_high is not None and t_low >= t_high and not f_weak):
        label = "MIXED"
        why = "Trend is Coulomb-like but not a clean strict ladder, or contact geometry interferes."
    else:
        label = "FAIL"
        why = "T_slip does not rise reasonably with μ_eff and/or normal force."
    if wedge and label == "PASS":
        label = "MIXED"
        why = "Ordering looks Coulomb but wedging / multi-axis contacts are present."
    return {
        "verdict": label,
        "why": why,
        "T_slip_mu_F1": {"low": t_low, "mid": t_mid, "high": t_high},
        "mu_strict": mu_ok,
        "mu_weak": mu_weak,
        "T_slip_mid_by_F": by_f,
        "F_strict": f_ok,
        "F_weak": f_weak,
        "near_zero_below_low": nz_low,
        "wedging_seen": wedge,
    }


def main() -> int:
    ctx = build_fixture()
    try:
        mu_rows = [run_case(ctx, name, 1.0) for name in ("low", "mid", "high")]
        f_rows = [run_case(ctx, "mid", f) for f in (0.5, 1.0, 2.0)]
        nz = run_case(ctx, "near_zero", 1.0)
        dec = verdict(mu_rows, f_rows, nz)
        dump(HERE / "MU_SWEEP_F1.json", mu_rows)
        dump(HERE / "F_SWEEP_MID.json", f_rows)
        dump(HERE / "NEAR_ZERO.json", nz)
        dump(HERE / "VERDICT.json", dec)
        print(json.dumps({"done": True, **dec}, indent=2), flush=True)
        return 0
    finally:
        try:
            uninstall_lift_squeeze(ctx["task"])
        except Exception:
            pass
        try:
            ctx["task"].close_env()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
