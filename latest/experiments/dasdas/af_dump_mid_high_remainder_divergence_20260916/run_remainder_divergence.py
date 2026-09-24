#!/usr/bin/env python3
"""Same T6 lift-end snapshot: mid vs high through wrist reorientation / pour only."""
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sapien

HERE = Path(__file__).resolve().parent
GRASP = Path("/media/volume/dasdas/exouser/af_dump_grasp_establishment_divergence_20260916")
OLD = Path("/media/volume/dasdas/exouser/af_dump_original_geom_finger_friction_sanity_20260916")
PLAN = json.loads((HERE / "PLAN.json").read_text())
sys.path[:0] = [str(GRASP), str(OLD)]

import run_grasp_establishment as G
import run_original_geom_finger_sweep as S
from envs.utils import Action, ArmTag
from force_realize import (
    SETTLE_STEPS,
    LiftSqueezeHold,
    capture_stock_drives,
    install_lift_squeeze,
    realize_commanded_force,
    restore_stock_drives,
    uninstall_lift_squeeze,
)
from run_liftstyle_context import PrefixSnapshot, scripted_establish_grasp

SEED = int(PLAN["seed"])
EPISODE = int(PLAN["episode_id"])
DESK_MU = float(PLAN["deskbin_mu_fixed"])
PRE_FINGER = float(PLAN["finger_mu"]["mid"])
QUERY_F = 4.0
QUERY_DIS = 0.012
FORCES = [float(x) for x in PLAN["forces_N"]]
MUS = {"mid": float(PLAN["finger_mu"]["mid"]), "high": float(PLAN["finger_mu"]["high"])}
THR = PLAN["obvious_divergence"]
STAGES = PLAN["stages"]
DT = 0.004
LIFT_Z = 0.08
STRIDE = 4
REFRESH = 2
ARM = ArmTag("left")
WRIST_TGT_Q = [-0.694654, -0.178228, 0.165979, -0.676862]


def dump(path: Path, value) -> None:
    S.dump_json(path, value)


def log(event: dict) -> None:
    print(json.dumps(event), flush=True)


def dropped(state: dict) -> bool:
    return state.get("mode") == "none" or (
        state.get("mode") != "bilateral" and float(state.get("squeeze") or 0.0) < 0.05
    )


def ee_pose(task) -> list[float]:
    return [float(x) for x in task.get_arm_pose(ARM)]


def quat_mul(a, b):
    w1, x1, y1, z1 = [float(x) for x in a]
    w2, x2, y2, z2 = [float(x) for x in b]
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def quat_conj(q):
    return [float(q[0]), -float(q[1]), -float(q[2]), -float(q[3])]


def quat_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rel_obj_in_ee(ee, obj_xyz, obj_q):
    ee_p = np.asarray(ee[:3], dtype=float)
    ee_q = np.asarray(ee[3:7], dtype=float)
    obj_p = np.asarray(obj_xyz, dtype=float)
    r = quat_to_mat(ee_q).T @ (obj_p - ee_p)
    rel_q = quat_mul(quat_conj(ee_q), obj_q)
    return r.tolist(), [float(x) for x in rel_q]


def body_vel(task):
    entity = task.deskbin.actor if hasattr(task.deskbin, "actor") else task.deskbin
    body = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    if body is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return [float(x) for x in body.linear_velocity], [float(x) for x in body.angular_velocity]


def finger_contacts(task) -> list[dict]:
    mapping = S.shape_map(task)
    fingers = {"fl_link7", "fl_link8", "fr_link7", "fr_link8"}
    out = []
    dt = float(getattr(task, "physics_timestep", DT))
    for contact in task.scene.get_contacts():
        name0 = S.body_name(contact.bodies[0])
        name1 = S.body_name(contact.bodies[1])
        if S.DESKBIN not in (name0, name1):
            continue
        other = name1 if name0 == S.DESKBIN else name0
        if other not in fingers:
            continue
        try:
            shapes = list(contact.shapes)
        except Exception:
            shapes = []
        metas = []
        for shape in shapes[:2]:
            try:
                metas.append(mapping.get(S.shape_key(shape)))
            except Exception:
                metas.append(None)
        desk_meta = metas[0] if name0 == S.DESKBIN else (metas[1] if len(metas) > 1 else None)
        hull = None if desk_meta is None else int(desk_meta["index"])
        for point in contact.points:
            try:
                pos = np.asarray(point.position, dtype=float).tolist()
            except Exception:
                pos = None
            normal = np.asarray(point.normal, dtype=float)
            nrm = float(np.linalg.norm(normal)) + 1e-12
            impulse = np.asarray(point.impulse, dtype=float)
            fn = abs(float(np.dot(impulse, normal / nrm))) / max(dt, 1e-9)
            if fn < 1e-6:
                continue
            out.append(
                {
                    "finger": other,
                    "hull": hull,
                    "position": None if pos is None else [float(x) for x in pos],
                    "normal": [float(x) for x in (normal / nrm)],
                    "fn": float(fn),
                }
            )
        if len(out) >= 24:
            break
    return out[:24]


def kinematics(task, t6_ee, prev=None):
    ee = ee_pose(task)
    wrist_deg = G.quat_angle_deg(t6_ee[3:7], ee[3:7])
    target_deg = G.quat_angle_deg(t6_ee[3:7], WRIST_TGT_Q)
    progress = float(np.clip(wrist_deg / max(target_deg, 1e-6), 0.0, 1.5))
    lin, ang = body_vel(task)
    ee_v = [0.0, 0.0, 0.0]
    ee_w_deg = 0.0
    if prev is not None:
        dt = max(STRIDE * DT, 1e-6)
        ee_v = ((np.asarray(ee[:3]) - np.asarray(prev["ee_xyz"])) / dt).tolist()
        ee_w_deg = G.quat_angle_deg(prev["ee_q"], ee[3:7]) / dt
    return {
        "ee_xyz": ee[:3],
        "ee_q": ee[3:7],
        "wrist_angle_deg": wrist_deg,
        "wrist_target_deg": target_deg,
        "wrist_progress": progress,
        "obj_lin_vel": lin,
        "obj_ang_vel": ang,
        "obj_ang_speed": float(np.linalg.norm(ang)),
        "ee_lin_vel": ee_v,
        "ee_lin_speed": float(np.linalg.norm(ee_v)),
        "ee_ang_speed_deg": float(ee_w_deg),
        "wrist_omega_deg": float(ee_w_deg),
    }


def pack(state: dict, extra=None) -> dict:
    row = {
        "squeeze": state.get("squeeze"),
        "nl": state.get("nl"),
        "nr": state.get("nr"),
        "mode": state.get("mode"),
        "pair_dist": state.get("pair_dist"),
        "aperture": state.get("aperture"),
        "obj_xyz": state.get("obj_xyz"),
        "obj_q": state.get("obj_q"),
        "finger_hulls": {
            k: {"shape_ids": v.get("shape_ids"), "axes": v.get("axes"), "n_shapes": v.get("n_shapes")}
            for k, v in (state.get("finger_hulls") or {}).items()
        },
        "all_finger_hulls": state.get("all_finger_hulls"),
        "opposing": state.get("opposing"),
        "table_contact": state.get("table_contact"),
        "table_fn": state.get("table_fn"),
        "delta_xyz": state.get("delta_xyz"),
        "delta_xyz_norm": state.get("delta_xyz_norm"),
        "delta_rot_deg": state.get("delta_rot_deg"),
        "rel_slip_norm": state.get("rel_slip_norm"),
        "rel_slip_y": state.get("rel_slip_y"),
        "rel_slip_z": None if state.get("rel_slip_xyz") is None else state["rel_slip_xyz"][2],
        "rel_obj_ee_xyz": state.get("rel_obj_ee_xyz"),
        "rel_obj_ee_q": state.get("rel_obj_ee_q"),
        "rel_obj_ee_rot_deg": state.get("rel_obj_ee_rot_deg"),
        "contacts": state.get("contacts"),
        "n_hulls_left": None if not state.get("left_hulls") else len(state["left_hulls"]),
        "n_hulls_right": None if not state.get("right_hulls") else len(state["right_hulls"]),
        "dropped": dropped(state),
    }
    if extra:
        row.update(extra)
    return row


def enrich(task, ref, t6_ee, prev_kin=None, with_contacts=False) -> dict:
    st = G.rich_state(task, ref)
    kin = kinematics(task, t6_ee, prev_kin)
    rel_p, rel_q = rel_obj_in_ee(ee_pose(task), st["obj_xyz"], st["obj_q"])
    st["rel_obj_ee_xyz"] = rel_p
    st["rel_obj_ee_q"] = rel_q
    if ref is not None and ref.get("rel_obj_ee_q") is not None:
        st["rel_obj_ee_rot_deg"] = G.quat_angle_deg(ref["rel_obj_ee_q"], rel_q)
    else:
        st["rel_obj_ee_rot_deg"] = 0.0
    if with_contacts:
        st["contacts"] = finger_contacts(task)
    st.update(kin)
    return st


def stage_diff(mid: dict, high: dict) -> dict:
    pose = float(np.linalg.norm(np.asarray(mid["obj_xyz"]) - np.asarray(high["obj_xyz"])))
    rot = G.quat_angle_deg(mid["obj_q"], high["obj_q"])
    pair = abs(float(mid.get("pair_dist") or 0) - float(high.get("pair_dist") or 0))
    squeeze = abs(float(mid.get("squeeze") or 0) - float(high.get("squeeze") or 0))
    slip = abs(float(mid.get("rel_slip_norm") or 0) - float(high.get("rel_slip_norm") or 0))
    hulls = json.dumps(mid.get("finger_hulls"), sort_keys=True) != json.dumps(high.get("finger_hulls"), sort_keys=True)
    flags = {
        "squeeze": squeeze >= float(THR["squeeze_abs_n"]),
        "pose": pose >= float(THR["obj_translation_m"]),
        "rot": rot >= float(THR["obj_rotation_deg"]),
        "pair": pair >= float(THR["pair_dist_m"]),
        "slip": slip >= float(THR["rel_slip_m"]),
        "mode": mid.get("mode") != high.get("mode"),
        "table": bool(mid.get("table_contact")) != bool(high.get("table_contact")),
        "drop": bool(mid.get("dropped")) != bool(high.get("dropped")),
        "hull_assignment": hulls,
    }
    obvious = bool(
        flags["squeeze"]
        or flags["pose"]
        or flags["rot"]
        or flags["pair"]
        or flags["slip"]
        or flags["mode"]
        or flags["table"]
        or flags["drop"]
    )
    first = []
    if flags["mode"]:
        first.append(f"contact mode {mid.get('mode')} vs {high.get('mode')}")
    if flags["drop"]:
        first.append("one side dropped")
    if flags["squeeze"]:
        first.append(f"squeeze Δ={squeeze:.2f} N")
    if flags["pose"]:
        first.append(f"pose Δ={1000 * pose:.2f} mm")
    if flags["rot"]:
        first.append(f"rot Δ={rot:.2f} deg")
    if flags["pair"]:
        first.append(f"pair Δ={1000 * pair:.2f} mm")
    if flags["slip"]:
        first.append(f"slip Δ={1000 * slip:.2f} mm")
    if flags["table"]:
        first.append("table contact mismatch")
    if flags["hull_assignment"] and not obvious:
        first.append("hull assignment only (not obvious)")
    high_worse = bool(
        (high.get("dropped") and not mid.get("dropped"))
        or (mid.get("mode") == "bilateral" and high.get("mode") != "bilateral")
        or (flags["squeeze"] and float(high.get("squeeze") or 0) + 0.5 < float(mid.get("squeeze") or 0))
        or (flags["drop"] and high.get("dropped"))
    )
    return {
        "flags": flags,
        "obvious": obvious,
        "high_worse": high_worse,
        "squeeze_mid": mid.get("squeeze"),
        "squeeze_high": high.get("squeeze"),
        "squeeze_gap": squeeze,
        "mode_mid": mid.get("mode"),
        "mode_high": high.get("mode"),
        "hulls_mid": mid.get("finger_hulls"),
        "hulls_high": high.get("finger_hulls"),
        "pose_gap_m": pose,
        "rot_gap_deg": rot,
        "pair_gap_m": pair,
        "slip_gap_m": slip,
        "wrist_mid": mid.get("wrist_angle_deg"),
        "wrist_high": high.get("wrist_angle_deg"),
        "first_diffs": first,
    }


class RemainderRec:
    def __init__(self, task, ref, t6_ee, force_n):
        self.task = task
        self.ref = ref
        self.t6_ee = t6_ee
        self.force_n = force_n
        self.phase = "R0"
        self.frames = []
        self.n = 0
        self.drop_n = None
        self.unilateral_n = None
        self.large_slip_n = None
        self.hull_switch_n = None
        self.squeeze_collapse_n = None
        self.rel_rot_jump_n = None
        self._last_hull = None
        self._collapse_run = 0
        self._last_rel_rot = 0.0
        self._prev_kin = None
        self._orig = task._after_physics_step

        def hooked():
            self._orig()
            self.n += 1
            if self.n % STRIDE:
                return
            st = enrich(self.task, self.ref, self.t6_ee, self._prev_kin, with_contacts=False)
            last = getattr(self.task, "_af_lift_holder", None)
            last = None if last is None else last.last
            hull_key = json.dumps(st.get("finger_hulls"), sort_keys=True)
            frame = {
                "step": self.n,
                "t_s": self.n * DT,
                "phase": self.phase,
                "squeeze": st["squeeze"],
                "nl": st["nl"],
                "nr": st["nr"],
                "mode": st["mode"],
                "mode_code": {"none": 0, "unilateral": 1, "bilateral": 2}[st["mode"]],
                "pair_dist": st["pair_dist"],
                "aperture": st["aperture"],
                "rel_slip_norm": st.get("rel_slip_norm") or 0.0,
                "rel_slip_y": st.get("rel_slip_y"),
                "rel_slip_z": None if st.get("rel_slip_xyz") is None else st["rel_slip_xyz"][2],
                "delta_xyz_norm": st.get("delta_xyz_norm") or 0.0,
                "delta_rot_deg": st.get("delta_rot_deg") or 0.0,
                "rel_obj_ee_rot_deg": st.get("rel_obj_ee_rot_deg") or 0.0,
                "wrist_angle_deg": st["wrist_angle_deg"],
                "wrist_progress": st["wrist_progress"],
                "wrist_omega_deg": st["wrist_omega_deg"],
                "ee_lin_speed": st["ee_lin_speed"],
                "obj_ang_speed": st["obj_ang_speed"],
                "obj_xyz": st["obj_xyz"],
                "obj_q": st["obj_q"],
                "finger_hulls": {
                    k: {"shape_ids": v.get("shape_ids"), "axes": v.get("axes"), "n_shapes": v.get("n_shapes")}
                    for k, v in (st.get("finger_hulls") or {}).items()
                },
                "hull_key": hull_key,
                "opposing": st["opposing"],
                "table_contact": st["table_contact"],
                "dropped": dropped(st),
                "single": None if last is None else last.get("single_finger_n"),
                "bilateral_n": None if last is None else last.get("bilateral_n"),
                "tracking_err": None
                if last is None
                else abs(float(last.get("single_finger_n") or 0.0) - self.force_n),
            }
            self.frames.append(frame)
            self._prev_kin = {
                "ee_xyz": st["ee_xyz"],
                "ee_q": st["ee_q"],
            }
            if self.unilateral_n is None and st["mode"] == "unilateral":
                self.unilateral_n = self.n
            if self.large_slip_n is None and float(st.get("rel_slip_norm") or 0) >= 0.003:
                self.large_slip_n = self.n
            if self._last_hull is not None and hull_key != self._last_hull and self.hull_switch_n is None:
                self.hull_switch_n = self.n
            self._last_hull = hull_key
            if float(st.get("squeeze") or 0) < 0.3 * max(self.force_n, 0.25):
                self._collapse_run += 1
                if self.squeeze_collapse_n is None and self._collapse_run >= 8:
                    self.squeeze_collapse_n = self.n
            else:
                self._collapse_run = 0
            rel_rot = float(st.get("rel_obj_ee_rot_deg") or 0)
            if self.rel_rot_jump_n is None and abs(rel_rot - self._last_rel_rot) >= 5.0 and self.n > STRIDE * 2:
                self.rel_rot_jump_n = self.n
            self._last_rel_rot = rel_rot
            if self.drop_n is None and dropped(st):
                self.drop_n = self.n

        task._after_physics_step = hooked

    def close(self):
        self.task._after_physics_step = self._orig

    def precursors(self) -> dict:
        return {
            "first_unilateral_s": None if self.unilateral_n is None else self.unilateral_n * DT,
            "first_large_slip_s": None if self.large_slip_n is None else self.large_slip_n * DT,
            "first_hull_switch_s": None if self.hull_switch_n is None else self.hull_switch_n * DT,
            "first_squeeze_collapse_s": None if self.squeeze_collapse_n is None else self.squeeze_collapse_n * DT,
            "first_rel_rot_jump_s": None if self.rel_rot_jump_n is None else self.rel_rot_jump_n * DT,
            "drop_s": None if self.drop_n is None else self.drop_n * DT,
            "n_frames": len(self.frames),
            "n_steps": self.n,
        }


def snapshot_now(task, ref, t6_ee, force_n, extra=None):
    st = enrich(task, ref, t6_ee, with_contacts=True)
    last = getattr(task, "_af_lift_holder", None)
    last = None if last is None else last.last
    packed = pack(
        st,
        {
            "F_cmd": force_n,
            "N_L": st.get("nl"),
            "N_R": st.get("nr"),
            "tracking_err": None if last is None else abs(float(last.get("single_finger_n") or 0.0) - force_n),
            "wrist_angle_deg": st.get("wrist_angle_deg"),
            "wrist_progress": st.get("wrist_progress"),
            "wrist_omega_deg": st.get("wrist_omega_deg"),
            "ee_lin_speed": st.get("ee_lin_speed"),
            "ee_ang_speed_deg": st.get("ee_ang_speed_deg"),
            "obj_ang_speed": st.get("obj_ang_speed"),
            "single": None if last is None else last.get("single_finger_n"),
        },
    )
    if extra:
        packed.update(extra)
    return packed


def nearest_progress(frames, lo, hi, target, phase="wrist"):
    cand = [f for f in frames if f.get("phase") == phase and lo <= float(f.get("wrist_progress") or 0) <= hi]
    if not cand:
        cand = [f for f in frames if f.get("phase") == phase]
    if not cand:
        return None
    return min(cand, key=lambda f: abs(float(f.get("wrist_progress") or 0) - target))


def frame_to_pack(frame, force_n, fallback):
    if frame is None:
        return dict(fallback)
    return {
        "squeeze": frame["squeeze"],
        "nl": frame.get("nl"),
        "nr": frame.get("nr"),
        "mode": frame["mode"],
        "pair_dist": frame.get("pair_dist"),
        "aperture": frame.get("aperture"),
        "obj_xyz": frame.get("obj_xyz") or fallback.get("obj_xyz"),
        "obj_q": frame.get("obj_q") or fallback.get("obj_q"),
        "finger_hulls": frame.get("finger_hulls") or fallback.get("finger_hulls"),
        "all_finger_hulls": fallback.get("all_finger_hulls"),
        "opposing": frame.get("opposing"),
        "table_contact": frame.get("table_contact"),
        "table_fn": fallback.get("table_fn"),
        "delta_xyz_norm": frame.get("delta_xyz_norm"),
        "delta_rot_deg": frame.get("delta_rot_deg"),
        "rel_slip_norm": frame.get("rel_slip_norm"),
        "rel_slip_y": frame.get("rel_slip_y"),
        "rel_slip_z": frame.get("rel_slip_z"),
        "rel_obj_ee_rot_deg": frame.get("rel_obj_ee_rot_deg"),
        "dropped": frame.get("dropped"),
        "F_cmd": force_n,
        "N_L": frame.get("nl"),
        "N_R": frame.get("nr"),
        "tracking_err": frame.get("tracking_err"),
        "wrist_angle_deg": frame.get("wrist_angle_deg"),
        "wrist_progress": frame.get("wrist_progress"),
        "wrist_omega_deg": frame.get("wrist_omega_deg"),
        "ee_lin_speed": frame.get("ee_lin_speed"),
        "obj_ang_speed": frame.get("obj_ang_speed"),
        "from_trace": True,
        "t_s": frame.get("t_s"),
        "phase": frame.get("phase"),
    }


def pick_stages(r0, rec, force_n) -> dict:
    frames = rec.frames
    last = pack_from_last(frames, r0, force_n)
    r1 = nearest_progress(frames, 0.0, 0.08, 0.0, "wrist")
    r2 = nearest_progress(frames, 0.18, 0.35, 0.25, "wrist")
    r3 = nearest_progress(frames, 0.40, 0.60, 0.50, "wrist")
    r4 = nearest_progress(frames, 0.65, 0.85, 0.75, "wrist")
    wrist = [f for f in frames if f.get("phase") == "wrist"]
    r5 = wrist[-1] if wrist else None
    pour1 = [f for f in frames if f.get("phase") == "pour1"]
    r6 = pour1[len(pour1) // 2] if pour1 else None
    if r6 is None:
        rest = [f for f in frames if f.get("phase") in {"shake0", "pour1", "pour2", "delay"}]
        r6 = rest[len(rest) // 2] if rest else None
    r7 = frames[-1] if frames else None
    stages = {
        "R0_T6": r0,
        "R1_wrist_start": frame_to_pack(r1, force_n, r0),
        "R2_wrist_25": frame_to_pack(r2, force_n, r0),
        "R3_wrist_50": frame_to_pack(r3, force_n, r0),
        "R4_wrist_75": frame_to_pack(r4, force_n, last),
        "R5_wrist_max": frame_to_pack(r5, force_n, last),
        "R6_pour_mid": frame_to_pack(r6, force_n, last),
        "R7_pour_end": frame_to_pack(r7, force_n, last),
    }
    if rec.drop_n is not None:
        by_step = {f["step"]: f for f in frames}
        for name, delta in (("Rdrop-2", -2 * STRIDE), ("Rdrop-1", -STRIDE), ("Rdrop", 0), ("Rdrop+1", STRIDE)):
            fr = by_step.get(rec.drop_n + delta) or min(frames, key=lambda f: abs(f["step"] - (rec.drop_n + delta)))
            stages[name] = frame_to_pack(fr, force_n, last)
    return stages


def pack_from_last(frames, r0, force_n):
    if not frames:
        return r0
    return frame_to_pack(frames[-1], force_n, r0)


def run_query(task) -> dict:
    last = None
    for _ in range(3):
        try:
            return task.run_activeforcing_query(query_force_n=QUERY_F, displacement_m=QUERY_DIS)
        except RuntimeError as exc:
            last = repr(exc)
            if "planning failed" not in last:
                raise
    return {"query_failed": True, "error": last, "contact_ratio": 0.0}


def setup_t6(task, stock, pre_snap, force_n: float):
    uninstall_lift_squeeze(task)
    pre_snap.restore()
    G.paint(task, MUS["mid"])
    query = run_query(task)
    realize = realize_commanded_force(task, force_n, stock)
    try:
        task.move(task.move_by_displacement(arm_tag=ARM, z=LIFT_Z, move_axis="arm"))
        lift_err = None
    except Exception as exc:
        lift_err = repr(exc)
    t6_ee = ee_pose(task)
    t6_obj = G.rich_state(task)
    snap = PrefixSnapshot(task)
    return {
        "query_cr": query.get("contact_ratio"),
        "query_failed": bool(query.get("query_failed")),
        "realize": realize,
        "lift_error": lift_err,
        "t6_ee": t6_ee,
        "t6_obj_xyz": t6_obj["obj_xyz"],
        "t6_obj_q": t6_obj["obj_q"],
        "t6_squeeze": t6_obj["squeeze"],
        "t6_mode": t6_obj["mode"],
        "snapshot": snap,
    }


def remainder_script(task, rec):
    arm, acts = task.pour_actions
    rec.phase = "wrist"
    task.move((arm, [acts[0]]))
    rec.phase = "shake0"
    task.move((arm, [acts[1]]))
    rec.phase = "pour1"
    task.move(task.pour_actions)
    rec.phase = "pour2"
    task.move(task.pour_actions)
    rec.phase = "delay"
    task.delay(6)


def run_remainder_branch(task, stock, t6, force_n: float, mu_label: str):
    uninstall_lift_squeeze(task)
    t6["snapshot"].restore()
    audit = G.paint(task, MUS[mu_label])
    holder = LiftSqueezeHold(task, force_n, stock)
    install_lift_squeeze(task, holder)
    task.af_force_limit_n = float(force_n)
    for _ in range(REFRESH):
        task.scene.step()
    t6_ee = t6["t6_ee"]
    r0_live = enrich(task, None, t6_ee, with_contacts=True)
    r0_live["rel_obj_ee_xyz"], r0_live["rel_obj_ee_q"] = rel_obj_in_ee(
        ee_pose(task), r0_live["obj_xyz"], r0_live["obj_q"]
    )
    ref = dict(r0_live)
    r0 = snapshot_now(task, ref, t6_ee, force_n, {"geometry_restored": True, "mu": mu_label})
    rec = RemainderRec(task, ref, t6_ee, force_n)
    try:
        remainder_script(task, rec)
        err = None
    except Exception as exc:
        err = repr(exc)
        log({"event": "remainder_error", "mu": mu_label, "F": force_n, "error": err})
    rec.close()
    stages = pick_stages(r0, rec, force_n)
    uninstall_lift_squeeze(task)
    restore_stock_drives(stock)
    return {
        "mu": mu_label,
        "mu_finger": MUS[mu_label],
        "mu_eff": 0.5 * (MUS[mu_label] + DESK_MU),
        "F": force_n,
        "audit": audit,
        "r0_vs_t6_pose_m": float(
            np.linalg.norm(np.asarray(r0["obj_xyz"]) - np.asarray(t6["t6_obj_xyz"]))
        ),
        "stages": stages,
        "frames": rec.frames,
        "precursors": rec.precursors(),
        "remainder_error": err,
        "retained_R7": bool(
            stages["R7_pour_end"].get("mode") == "bilateral"
            and not stages["R7_pour_end"].get("dropped")
            and float(stages["R7_pour_end"].get("squeeze") or 0) > 0.05
        ),
    }


def pair_table(mid_b, high_b) -> list[dict]:
    rows = []
    for name in STAGES:
        m, h = mid_b["stages"][name], high_b["stages"][name]
        diff = stage_diff(m, h)
        rows.append(
            {
                "stage": name,
                "mid_squeeze": m.get("squeeze"),
                "high_squeeze": h.get("squeeze"),
                "mid_mode": m.get("mode"),
                "high_mode": h.get("mode"),
                "mid_dropped": m.get("dropped"),
                "high_dropped": h.get("dropped"),
                "mid_hulls": m.get("finger_hulls"),
                "high_hulls": h.get("finger_hulls"),
                "mid_wrist_deg": m.get("wrist_angle_deg"),
                "high_wrist_deg": h.get("wrist_angle_deg"),
                "pose_delta_m": diff["pose_gap_m"],
                "pair_delta_m": diff["pair_gap_m"],
                "slip_delta_m": diff["slip_gap_m"],
                "rot_delta_deg": diff["rot_gap_deg"],
                "obvious": diff["obvious"],
                "high_worse": diff["high_worse"],
                "first_diffs": diff["first_diffs"],
            }
        )
    return rows


def first_persistent(rows: list[dict]) -> dict:
    last_same = None
    first_any = None
    persist = None
    for i, row in enumerate(rows):
        if not row["obvious"]:
            last_same = row["stage"]
            continue
        if first_any is None:
            first_any = row
        later = rows[i:]
        r7 = later[-1]
        still = bool(r7["obvious"] or r7.get("high_dropped") or r7.get("mid_dropped"))
        if still:
            persist = row
            break
    return {
        "first_obvious": None if first_any is None else first_any["stage"],
        "first_persistent": None if persist is None else persist["stage"],
        "last_same": last_same if persist is None else STAGES[STAGES.index(persist["stage"]) - 1] if persist["stage"] != STAGES[0] else None,
        "high_worse_at_persist": None if persist is None else persist["high_worse"],
        "persist_first_diffs": None if persist is None else persist["first_diffs"],
        "r7_obvious": rows[-1]["obvious"],
        "r7_high_worse": rows[-1]["high_worse"],
    }


def classify_force(info: dict, rows: list[dict]) -> str:
    if info["first_persistent"] is None:
        return "NO_CLEAR_DIVERGENCE"
    stage = info["first_persistent"]
    diffs = info["persist_first_diffs"] or []
    contactish = any("contact mode" in x or "dropped" in x for x in diffs)
    hull_switch = "hull" in json.dumps(diffs).lower()
    if contactish and (stage != "R0_T6") and (info.get("high_worse_at_persist") or info.get("r7_high_worse")):
        if hull_switch or any("contact mode" in x for x in diffs):
            # keep A/B unless the fork is clearly a contact-mode event without wrist/pour kinematics change
            pass
    if stage in {"R1_wrist_start", "R2_wrist_25", "R3_wrist_50", "R4_wrist_75", "R5_wrist_max"}:
        label = "WRIST_ROTATION"
    elif stage in {"R6_pour_mid", "R7_pour_end"}:
        label = "POUR"
    elif stage == "R0_T6":
        label = "NO_CLEAR_DIVERGENCE"
    else:
        label = "NO_CLEAR_DIVERGENCE"
    if (
        label in {"WRIST_ROTATION", "POUR"}
        and contactish
        and info.get("high_worse_at_persist")
        and any("contact mode" in x or "dropped" in x for x in diffs)
        and not any("pose" in x or "rot Δ" in x or "slip" in x for x in diffs[:1])
    ):
        label = "CONTACT_TRANSITION"
    if not info.get("high_worse_at_persist") and not info.get("r7_high_worse"):
        return "NO_CLEAR_DIVERGENCE"
    return label


def plot_force(force: float, mid_b, high_b) -> None:
    fig_dir = HERE / "plots"
    fig_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(7, 1, figsize=(11, 16), sharex=True)
    specs = [
        ("squeeze", "Bilateral squeeze (N)"),
        ("wrist_angle_deg", "Wrist rotation from T6 (deg)"),
        ("rel_slip_norm", "Relative tangential slip (m)"),
        ("rel_obj_ee_rot_deg", "Object-in-EE relative rotation (deg)"),
        ("mode_code", "Contact mode (0 none / 1 uni / 2 bi)"),
        ("hull_code", "Hull-key code"),
        ("pair_dist", "Finger pair distance (m)"),
    ]
    for ax, (key, title) in zip(axes, specs):
        for packed, color, name in ((mid_b, "#c47b16", "mid 0.575"), (high_b, "#b42318", "high 0.85")):
            frames = packed["frames"]
            t = [f["t_s"] for f in frames]
            if key == "hull_code":
                keys = []
                code = {}
                y = []
                for f in frames:
                    hk = f.get("hull_key") or ""
                    if hk not in code:
                        code[hk] = len(code)
                    y.append(code[hk])
                ax.plot(t, y, color=color, lw=1.4, label=name)
            else:
                ax.plot(t, [f.get(key) if f.get(key) is not None else np.nan for f in frames], color=color, lw=1.4, label=name)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("remainder time (s) from T6")
    fig.suptitle(f"F={force:.2f} N remainder mid vs high, seed {SEED}", y=0.995)
    fig.tight_layout()
    fig.savefig(fig_dir / f"F_{force:.2f}_timeseries.png", dpi=140)
    plt.close(fig)


def run_pure_rotation_control(task, stock, t6, force_n: float) -> dict:
    out = {}
    for mu_label in ("mid", "high"):
        uninstall_lift_squeeze(task)
        t6["snapshot"].restore()
        G.paint(task, MUS[mu_label])
        holder = LiftSqueezeHold(task, force_n, stock)
        install_lift_squeeze(task, holder)
        task.af_force_limit_n = float(force_n)
        for _ in range(REFRESH):
            task.scene.step()
        t6_ee = t6["t6_ee"]
        live = enrich(task, None, t6_ee, with_contacts=True)
        live["rel_obj_ee_xyz"], live["rel_obj_ee_q"] = rel_obj_in_ee(ee_pose(task), live["obj_xyz"], live["obj_q"])
        ref = dict(live)
        rec = RemainderRec(task, ref, t6_ee, force_n)
        rec.phase = "pure_rotation"
        pose = list(ee_pose(task)[:3]) + list(WRIST_TGT_Q)
        try:
            task.move((ARM, [Action(ARM, "move", pose)]))
            err = None
        except Exception as exc:
            err = repr(exc)
        rec.close()
        end = snapshot_now(task, ref, t6_ee, force_n)
        uninstall_lift_squeeze(task)
        restore_stock_drives(stock)
        out[mu_label] = {
            "error": err,
            "end": end,
            "precursors": rec.precursors(),
            "n_frames": len(rec.frames),
            "retained": bool(end.get("mode") == "bilateral" and not end.get("dropped")),
        }
    diff = stage_diff(out["mid"]["end"], out["high"]["end"])
    out["diff"] = diff
    return out


def build_fixture():
    task = S.make_task(SEED, EPISODE)
    stock = capture_stock_drives(task)
    G.paint(task, PRE_FINGER)
    task.activate_activeforcing_candidate_force()
    scripted_establish_grasp(task)
    window = []
    for _ in range(20):
        task.scene.step()
        window.append(S.pack_state(task))
    gate = S.classify_pre(window)
    log({"event": "PRE", "gate": gate["gate"], "cr": gate["window_cr"]})
    if gate["gate"] != "VALID":
        raise RuntimeError(f"PRE not VALID: {gate}")
    pre = G.rich_state(task)
    snapshot = PrefixSnapshot(task)
    dump(
        HERE / "FIXTURE.json",
        {
            "seed": SEED,
            "episode_id": EPISODE,
            "gate": gate,
            "pre": pack(pre),
            "deskbin_mu": DESK_MU,
            "pour_target": list(task.pour_actions[1][0].target_pose),
        },
    )
    return task, stock, snapshot, pre, gate


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    task, stock, pre_snap, pre, gate = build_fixture()
    force_rows = {}
    force_meta = {}
    overall_votes = []
    for force in FORCES:
        log({"event": "setup_T6", "F": force})
        t6 = setup_t6(task, stock, pre_snap, force)
        dump(
            HERE / f"T6_F_{force:.2f}.json",
            {
                "F": force,
                "query_cr": t6["query_cr"],
                "query_failed": t6["query_failed"],
                "realize": t6["realize"],
                "lift_error": t6["lift_error"],
                "t6_ee": t6["t6_ee"],
                "t6_obj_xyz": t6["t6_obj_xyz"],
                "t6_squeeze": t6["t6_squeeze"],
                "t6_mode": t6["t6_mode"],
            },
        )
        branches = {}
        for mu_label in ("mid", "high"):
            log({"event": "remainder", "F": force, "mu": mu_label})
            branches[mu_label] = run_remainder_branch(task, stock, t6, force, mu_label)
            slim = dict(branches[mu_label])
            slim["frames"] = [
                {
                    k: v
                    for k, v in fr.items()
                    if k
                    not in {
                        "obj_xyz",
                        "obj_q",
                        "finger_hulls",
                    }
                }
                for fr in slim["frames"]
            ]
            dump(HERE / f"BRANCH_F_{force:.2f}_{mu_label}.json", slim)
        rows = pair_table(branches["mid"], branches["high"])
        info = first_persistent(rows)
        label = classify_force(info, rows)
        overall_votes.append(label)
        plot_force(force, branches["mid"], branches["high"])
        control = None
        if label == "WRIST_ROTATION" and info.get("high_worse_at_persist"):
            log({"event": "pure_rotation_control", "F": force})
            control = run_pure_rotation_control(task, stock, t6, force)
            dump(HERE / f"CONTROL_pure_rotation_F_{force:.2f}.json", control)
        force_rows[f"{force:.1f}"] = rows
        force_meta[f"{force:.1f}"] = {
            "F": force,
            "t6_squeeze": t6["t6_squeeze"],
            "t6_mode": t6["t6_mode"],
            "realize": t6["realize"],
            "mid_retained": branches["mid"]["retained_R7"],
            "high_retained": branches["high"]["retained_R7"],
            "mid_precursors": branches["mid"]["precursors"],
            "high_precursors": branches["high"]["precursors"],
            "mid_r0_pose_err_m": branches["mid"]["r0_vs_t6_pose_m"],
            "high_r0_pose_err_m": branches["high"]["r0_vs_t6_pose_m"],
            "persist": info,
            "label": label,
            "control_pure_rotation": None
            if control is None
            else {
                "mid_retained": control["mid"]["retained"],
                "high_retained": control["high"]["retained"],
                "obvious": control["diff"]["obvious"],
                "high_worse": control["diff"]["high_worse"],
                "first_diffs": control["diff"]["first_diffs"],
                "squeeze_mid": control["mid"]["end"].get("squeeze"),
                "squeeze_high": control["high"]["end"].get("squeeze"),
                "mode_mid": control["mid"]["end"].get("mode"),
                "mode_high": control["high"]["end"].get("mode"),
            },
            "mid_error": branches["mid"]["remainder_error"],
            "high_error": branches["high"]["remainder_error"],
        }
        dump(HERE / f"TABLE_F_{force:.2f}.json", rows)
        log({"event": "force_done", "F": force, "label": label, "persist": info})

    preferred = None
    for key in ("1.0", "2.5", "5.0"):
        if key in force_meta and force_meta[key]["label"] != "NO_CLEAR_DIVERGENCE":
            preferred = force_meta[key]["label"]
            break
    if preferred is None:
        preferred = "NO_CLEAR_DIVERGENCE"
    summary = {
        "seed": SEED,
        "episode_id": EPISODE,
        "forces_N": FORCES,
        "paper_data": False,
        "classification": preferred,
        "votes": overall_votes,
        "force_meta": force_meta,
        "tables": force_rows,
        "explains_old_17_vs_4": bool(
            preferred != "NO_CLEAR_DIVERGENCE"
            and any(force_meta[k]["mid_retained"] and not force_meta[k]["high_retained"] for k in force_meta)
        ),
    }
    dump(HERE / "SUMMARY.json", summary)
    log({"event": "done", "classification": preferred})
    return summary


if __name__ == "__main__":
    try:
        main()
    except Exception:
        (HERE / "ERROR.txt").write_text(traceback.format_exc())
        raise
