#!/usr/bin/env python3
"""Same PRE snapshot, official 4 N / 12 mm query, three finger μ.

Stops at query handoff. No dump remainder, VLA, force sweep, or task success.
"""
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

HERE = Path(__file__).resolve().parent
OLD = Path("/media/volume/dasdas/exouser/af_dump_original_geom_finger_friction_sanity_20260916")
PLAN = json.loads((HERE / "PLAN.json").read_text())
sys.path.insert(0, str(OLD))

import run_original_geom_finger_sweep as S
from envs.utils import ArmTag
from force_realize import (
    LiftSqueezeHold,
    capture_stock_drives,
    realize_commanded_force,
    restore_stock_drives,
    uninstall_lift_squeeze,
)
from run_liftstyle_context import PrefixSnapshot, scripted_establish_grasp

SEED = int(PLAN["seed"])
EPISODE = int(PLAN["episode_id"])
DESK_MU = float(PLAN["deskbin_mu_fixed"])
PRE_FINGER = float(PLAN["pre_established_at_finger_mu"])
QUERY_F = float(PLAN["query"]["force_n"])
QUERY_DIS = float(PLAN["query"]["displacement_m"])
LEVELS = {
    "low": {"mu_finger": 0.425, "mu_eff": 0.3625},
    "mid": {"mu_finger": 0.575, "mu_eff": 0.4375},
    "high": {"mu_finger": 0.85, "mu_eff": 0.575},
}
THR = PLAN["divergence_thresholds_mid_vs_high"]
STRIDE = 5
DT = 0.004


def dump(path: Path, value) -> None:
    S.dump_json(path, value)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_mul(quat_mul(q, np.array([0.0, v[0], v[1], v[2]])), quat_conj(q))[1:]


def quat_angle_deg(q0, q1) -> float:
    d = abs(float(np.dot(np.asarray(q0, dtype=float), np.asarray(q1, dtype=float))))
    return float(2.0 * math.degrees(math.acos(min(1.0, d))))


def ee_pose(task) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(task.get_arm_pose("left"), dtype=float).reshape(-1)
    return raw[:3].copy(), raw[3:7].copy()


def finger_worlds(task) -> dict:
    out = {}
    for joint, _, _ in task.robot.left_gripper:
        name = str(joint.child_link.get_name())
        pose = None
        for attr in ("get_pose", "pose", "get_entity_pose"):
            val = getattr(joint.child_link, attr, None)
            try:
                pose = val() if callable(val) else val
                if pose is not None:
                    break
            except Exception:
                continue
        if pose is not None and hasattr(pose, "p"):
            out[name] = np.asarray(pose.p, dtype=float).tolist()
    return out


def contact_points(task) -> list[dict]:
    rows = []
    mapping = S.shape_map(task)
    fingers = set(S.FINGER_TOKENS)
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
        shape_id = None
        for shape in shapes:
            try:
                meta = mapping.get(S.shape_key(shape))
            except Exception:
                meta = None
            if meta is not None:
                shape_id = int(meta["index"])
                break
        for point in contact.points:
            pos = np.asarray(point.position, dtype=float)
            nrm = np.asarray(point.normal, dtype=float)
            nrm = nrm / (float(np.linalg.norm(nrm)) + 1e-12)
            rows.append(
                {
                    "finger": other,
                    "shape_id": shape_id,
                    "position": pos.tolist(),
                    "normal": nrm.tolist(),
                }
            )
    return rows


def contact_mode(geom: dict) -> str:
    if geom.get("bilateral"):
        return "bilateral"
    if geom.get("left_present") or geom.get("right_present"):
        return "unilateral"
    return "none"


def hulls_of(geom: dict, side: str) -> list[int]:
    names = {"left": ("fl_link7", "fl_link8"), "right": ("fr_link7", "fr_link8")}[side]
    ids = []
    for name in names:
        row = (geom.get("per_finger") or {}).get(name) or {}
        if row.get("present"):
            ids.extend(row.get("shape_ids") or [])
    return sorted(set(int(x) for x in ids))


def normals_of(geom: dict) -> dict:
    out = {}
    for name, row in (geom.get("per_finger") or {}).items():
        if row.get("present"):
            out[name] = {
                "axes": row.get("axes"),
                "normals": row.get("normals"),
                "shape_ids": row.get("shape_ids"),
                "n_shapes": row.get("n_shapes"),
            }
    return out


def rich_state(task, ref: dict | None = None) -> dict:
    geom = S.contact_geometry(task)
    obj_xyz = np.asarray(geom["obj_xyz"], dtype=float)
    obj_q = np.asarray(geom["obj_q"], dtype=float)
    ee_p, ee_q = ee_pose(task)
    fingers = finger_worlds(task)
    centroid = np.mean(np.array(list(fingers.values()), dtype=float), axis=0) if fingers else ee_p
    left_hulls = hulls_of(geom, "left")
    right_hulls = hulls_of(geom, "right")
    p_rel = quat_rotate(quat_conj(ee_q), obj_xyz - ee_p)
    q_rel = quat_mul(quat_conj(ee_q), obj_q)
    state = {
        "obj_xyz": obj_xyz.tolist(),
        "obj_q": obj_q.tolist(),
        "ee_xyz": ee_p.tolist(),
        "ee_q": ee_q.tolist(),
        "obj_in_ee_xyz": p_rel.tolist(),
        "obj_in_ee_q": q_rel.tolist(),
        "finger_xyz": fingers,
        "finger_centroid": centroid.tolist(),
        "aperture": geom["aperture"],
        "pair_dist": geom["pair_dist"],
        "nl": geom["nl"],
        "nr": geom["nr"],
        "squeeze": geom["squeeze"],
        "bilateral": geom["bilateral"],
        "left_present": geom["left_present"],
        "right_present": geom["right_present"],
        "mode": contact_mode(geom),
        "left_hulls": left_hulls,
        "right_hulls": right_hulls,
        "all_hulls": geom["contact_shape_ids"],
        "n_left_hulls": len(left_hulls),
        "n_right_hulls": len(right_hulls),
        "n_multi_hull_fingers": int(
            sum(1 for row in (geom.get("per_finger") or {}).values() if row.get("present") and int(row.get("n_shapes") or 0) >= 2)
        ),
        "normals": normals_of(geom),
        "contact_points": contact_points(task),
        "opposing": bool(
            {"-x", "+x"}
            <= {
                ax
                for row in (geom.get("per_finger") or {}).values()
                if row.get("present")
                for ax in (row.get("axes") or [])
            }
        ),
        "wedging": geom.get("multi_hull_wedging_any_finger"),
        "table_contact": bool(geom.get("table_deskbin_mu_eff")),
        "finger_mu_eff": geom.get("finger_deskbin_mu_eff"),
    }
    if ref is not None:
        dxyz = obj_xyz - np.asarray(ref["obj_xyz"], dtype=float)
        slip = obj_xyz - centroid
        slip0 = np.asarray(ref["obj_xyz"], dtype=float) - np.asarray(ref["finger_centroid"], dtype=float)
        p_rel0 = np.asarray(ref["obj_in_ee_xyz"], dtype=float)
        state["delta_xyz"] = dxyz.tolist()
        state["delta_xyz_norm"] = float(np.linalg.norm(dxyz))
        state["delta_rot_deg"] = quat_angle_deg(ref["obj_q"], obj_q)
        state["rel_slip_xyz"] = (slip - slip0).tolist()
        state["rel_slip_y"] = float((slip - slip0)[1])
        state["rel_slip_norm"] = float(np.linalg.norm(slip - slip0))
        state["ee_rel_delta_xyz"] = (p_rel - p_rel0).tolist()
        state["ee_rel_delta_norm"] = float(np.linalg.norm(p_rel - p_rel0))
        state["ee_rel_delta_rot_deg"] = quat_angle_deg(ref["obj_in_ee_q"], q_rel)
    return state


def slim_frame(state: dict, step: int) -> dict:
    return {
        "step": step,
        "t_s": step * DT,
        "squeeze": state["squeeze"],
        "nl": state["nl"],
        "nr": state["nr"],
        "bilateral": state["bilateral"],
        "mode": state["mode"],
        "pair_dist": state["pair_dist"],
        "aperture": state["aperture"],
        "obj_xyz": state["obj_xyz"],
        "delta_xyz": state.get("delta_xyz"),
        "delta_rot_deg": state.get("delta_rot_deg"),
        "rel_slip_y": state.get("rel_slip_y"),
        "rel_slip_norm": state.get("rel_slip_norm"),
        "ee_rel_delta_norm": state.get("ee_rel_delta_norm"),
        "ee_rel_delta_rot_deg": state.get("ee_rel_delta_rot_deg"),
        "left_hulls": state["left_hulls"],
        "right_hulls": state["right_hulls"],
        "all_hulls": state["all_hulls"],
        "mode_code": {"none": 0, "unilateral": 1, "bilateral": 2}[state["mode"]],
        "hull_key": ",".join(str(x) for x in state["all_hulls"]),
    }


class StepRecorder:
    def __init__(self, task, ref: dict):
        self.task = task
        self.ref = ref
        self.frames: list[dict] = []
        self.n = 0
        self._orig = task._after_physics_step

        def hooked():
            self._orig()
            self.n += 1
            if self.n % STRIDE == 0:
                self.frames.append(slim_frame(rich_state(task, ref), self.n))

        task._after_physics_step = hooked

    def close(self) -> None:
        self.task._after_physics_step = self._orig


def paint(task, finger_mu: float) -> dict:
    prepared = S.prepare_condition(task, DESK_MU, finger_mu)
    return {
        "finger_mu": prepared["audit"]["finger_mu"],
        "deskbin_mu": prepared["audit"]["deskbin_mu"],
        "mu_eff": prepared["audit"]["mu_eff_average"],
        "n_shapes": prepared["geometry"]["n_shapes"],
        "deskbin_id": prepared["geometry"]["deskbin_id"],
    }


def stability(frames: list[dict]) -> dict:
    if not frames:
        return {"qCR": 0.0, "bilateral_ratio": 0.0, "n_hull_transitions": 0, "n_mode_transitions": 0, "n_slip_events": 0}
    bits = [bool(f["bilateral"]) for f in frames]
    hulls = [tuple(f["all_hulls"] or []) for f in frames]
    modes = [f["mode"] for f in frames]
    slips = [float(f.get("rel_slip_norm") or 0.0) for f in frames]
    hull_tr = int(sum(1 for a, b in zip(hulls, hulls[1:]) if a != b))
    mode_tr = int(sum(1 for a, b in zip(modes, modes[1:]) if a != b))
    slip_ev = 0
    for a, b in zip(slips, slips[1:]):
        if abs(b - a) >= 0.0002:
            slip_ev += 1
    squeezes = [float(f["squeeze"]) for f in frames]
    return {
        "qCR": float(np.mean(bits)),
        "bilateral_ratio": float(np.mean(bits)),
        "n_hull_transitions": hull_tr,
        "n_mode_transitions": mode_tr,
        "n_slip_events": slip_ev,
        "squeeze_peak": float(np.max(squeezes)),
        "squeeze_mean": float(np.mean(squeezes)),
        "squeeze_final": float(squeezes[-1]),
        "slip_peak_m": float(np.max(np.abs(slips))),
        "n_frames": len(frames),
    }


def build_fixture():
    task = S.make_task(SEED, EPISODE)
    stock = capture_stock_drives(task)
    prepared = paint(task, PRE_FINGER)
    task.activate_activeforcing_candidate_force()
    scripted_establish_grasp(task)
    window = []
    for _ in range(20):
        task.scene.step()
        window.append(S.pack_state(task))
    gate = S.classify_pre(window)
    print(json.dumps({"event": "PRE", "gate": gate["gate"], "cr": gate["window_cr"]}), flush=True)
    if gate["gate"] != "VALID":
        raise RuntimeError(f"PRE not VALID: {gate}")
    pre = rich_state(task)
    snapshot = PrefixSnapshot(task)
    fixture = {
        "seed": SEED,
        "episode_id": EPISODE,
        "deskbin_id": prepared["deskbin_id"],
        "n_shapes": prepared["n_shapes"],
        "pre_gate": gate,
        "pre_finger_mu": PRE_FINGER,
        "deskbin_mu": DESK_MU,
        "pre": pre,
        "runtime_mu": prepared,
    }
    dump(HERE / "FIXTURE.json", fixture)
    return {"task": task, "stock": stock, "snapshot": snapshot, "pre": pre, "fixture": fixture}


def run_query(task) -> dict:
    last = None
    for _ in range(3):
        try:
            return task.run_activeforcing_query(query_force_n=QUERY_F, displacement_m=QUERY_DIS)
        except RuntimeError as exc:
            last = repr(exc)
            if "planning failed" not in last:
                raise
    return {"query_failed": True, "error": last, "contact_ratio": 0.0, "samples": 0}


def hold_equal(task, n_steps: int) -> None:
    task.robot.set_gripper_force_limit(QUERY_F, "left")
    for _ in range(int(n_steps)):
        task.scene.step()
        task._after_physics_step()


def summarize_branch(label: str, audit: dict, before: dict, after: dict, frames: list[dict], query: dict | None) -> dict:
    stab = stability(frames)
    return {
        "label": label,
        "runtime_mu": audit,
        "before": before,
        "after": after,
        "query": None
        if query is None
        else {
            "query_failed": bool(query.get("query_failed")),
            "contact_ratio": query.get("contact_ratio"),
            "final_bilateral_contact": query.get("final_bilateral_contact"),
            "measured_force_mean_n": query.get("measured_force_mean_n"),
            "measured_force_p95_n": query.get("measured_force_p95_n"),
            "relative_slip_path_m": query.get("relative_slip_path_m"),
            "relative_offset_max_m": query.get("relative_offset_max_m"),
            "actor_path_m": query.get("actor_path_m"),
            "ee_path_m": query.get("ee_path_m"),
            "actor_return_error_m": query.get("actor_return_error_m"),
            "ee_return_error_m": query.get("ee_return_error_m"),
            "samples": query.get("samples"),
        },
        "stability": stab,
        "n_physics_steps": frames[-1]["step"] if frames else 0,
        "row": {
            "condition": label,
            "final_squeeze": after["squeeze"],
            "qCR": stab["qCR"] if query is not None else stab["bilateral_ratio"],
            "pair_distance": after["pair_dist"],
            "object_delta_xyz": after.get("delta_xyz"),
            "object_delta_rotation_deg": after.get("delta_rot_deg"),
            "left_hull": after["left_hulls"],
            "right_hull": after["right_hulls"],
            "contact_normals": after["normals"],
            "relative_slip_m": after.get("rel_slip_norm"),
            "relative_slip_y_m": after.get("rel_slip_y"),
            "final_contact_mode": after["mode"],
            "aperture": after["aperture"],
            "nl": after["nl"],
            "nr": after["nr"],
            "opposing": after["opposing"],
            "n_multi_hull_fingers": after["n_multi_hull_fingers"],
            "ee_rel_delta_m": after.get("ee_rel_delta_norm"),
            "ee_rel_rot_deg": after.get("ee_rel_delta_rot_deg"),
        },
    }


def compare_mid_high(mid: dict, high: dict) -> dict:
    a, b = mid["after"], high["after"]
    flags = {
        "squeeze_abs": abs(float(a["squeeze"]) - float(b["squeeze"])) >= float(THR["squeeze_abs_n"]),
        "obj_translation": abs(float(a.get("delta_xyz_norm") or 0) - float(b.get("delta_xyz_norm") or 0))
        >= float(THR["obj_translation_m"])
        or float(np.linalg.norm(np.asarray(a["obj_xyz"]) - np.asarray(b["obj_xyz"]))) >= float(THR["obj_translation_m"]),
        "obj_rotation": abs(float(a.get("delta_rot_deg") or 0) - float(b.get("delta_rot_deg") or 0))
        >= float(THR["obj_rotation_deg"])
        or quat_angle_deg(a["obj_q"], b["obj_q"]) >= float(THR["obj_rotation_deg"]),
        "pair_dist": abs(float(a["pair_dist"] or 0) - float(b["pair_dist"] or 0)) >= float(THR["pair_dist_m"]),
        "rel_slip": abs(float(a.get("rel_slip_norm") or 0) - float(b.get("rel_slip_norm") or 0))
        >= float(THR["rel_slip_m"]),
        "hulls": list(a["left_hulls"]) != list(b["left_hulls"]) or list(a["right_hulls"]) != list(b["right_hulls"]),
    }
    pose_gap = float(np.linalg.norm(np.asarray(a["obj_xyz"]) - np.asarray(b["obj_xyz"])))
    rot_gap = quat_angle_deg(a["obj_q"], b["obj_q"])
    high_worse = []
    if float(b["squeeze"]) >= float(a["squeeze"]) + 1.0:
        high_worse.append("high residual squeeze much larger (stuck query load)")
    if flags["hulls"]:
        high_worse.append("contact hulls differ")
    if pose_gap >= float(THR["obj_translation_m"]) or rot_gap >= float(THR["obj_rotation_deg"]):
        high_worse.append("object did not finish at the same pose as mid")
    if abs(float(a.get("rel_slip_norm") or 0) - float(b.get("rel_slip_norm") or 0)) >= float(THR["rel_slip_m"]):
        if float(b.get("rel_slip_norm") or 0) < float(a.get("rel_slip_norm") or 0):
            high_worse.append("high slipped less — query pose not equalized")
        else:
            high_worse.append("high slipped more")
    if abs(float(a["pair_dist"] or 0) - float(b["pair_dist"] or 0)) >= float(THR["pair_dist_m"]):
        high_worse.append("pinch opening / pair distance differs")
    if a["mode"] != b["mode"] and b["mode"] != "bilateral":
        high_worse.append("high lost a clean bilateral pinch")
    if int(b["n_multi_hull_fingers"]) > int(a["n_multi_hull_fingers"]):
        high_worse.append("high has more multi-hull finger contacts")
    if not b.get("opposing") and a.get("opposing"):
        high_worse.append("high lost opposing ±x pinch")
    return {
        "flags": flags,
        "diverged": bool(any(flags.values())),
        "pose_gap_m": pose_gap,
        "rot_gap_deg": rot_gap,
        "squeeze_mid": a["squeeze"],
        "squeeze_high": b["squeeze"],
        "slip_mid": a.get("rel_slip_norm"),
        "slip_high": b.get("rel_slip_norm"),
        "high_worse_reasons": high_worse,
        "can_explain_high_drop": bool(len(high_worse) >= 2),
    }


def run_probe(task, stock, ref: dict) -> dict:
    realize = realize_commanded_force(task, float(PLAN["optional_probe"]["squeeze_n"]), stock)
    holder = LiftSqueezeHold(task, float(PLAN["optional_probe"]["squeeze_n"]), stock)
    task._af_lift_holder = holder
    task._af_lift_inner_active = True
    hold = []
    for _ in range(40):
        task.scene.step()
        task._after_physics_step()
        hold.append(rich_state(task, ref))
    rec = StepRecorder(task, ref)
    try:
        task.move(task.move_by_displacement(arm_tag=ArmTag("left"), z=float(PLAN["optional_probe"]["lift_z_m"]), move_axis="arm"))
    except Exception as exc:
        rec.close()
        uninstall_lift_squeeze(task)
        restore_stock_drives(stock)
        return {"error": repr(exc), "realize": realize}
    rec.close()
    after = rich_state(task, ref)
    uninstall_lift_squeeze(task)
    restore_stock_drives(stock)
    lost = after["mode"] != "bilateral" or float(after["squeeze"]) < 0.05
    return {
        "realize": realize,
        "hold_squeeze_mean": float(np.mean([s["squeeze"] for s in hold])),
        "after_lift": {
            "squeeze": after["squeeze"],
            "mode": after["mode"],
            "pair_dist": after["pair_dist"],
            "hulls": {"left": after["left_hulls"], "right": after["right_hulls"]},
            "delta_xyz": after.get("delta_xyz"),
            "rel_slip_norm": after.get("rel_slip_norm"),
        },
        "lift_stability": stability(rec.frames),
        "lost_or_collapsed": bool(lost),
    }


def plot_series(branches: dict) -> None:
    series = {k: v["frames"] for k, v in branches.items() if v.get("frames")}
    if not series:
        return
    fig_dir = HERE / "plots"
    fig_dir.mkdir(exist_ok=True)
    specs = [
        ("squeeze", "squeeze vs time", "squeeze (N)", lambda f: f["squeeze"]),
        ("slip", "relative tangential displacement vs time", "rel slip Y (mm)", lambda f: 1000.0 * float(f.get("rel_slip_y") or 0.0)),
        ("pair", "pair distance vs time", "pair distance (mm)", lambda f: 1000.0 * float(f.get("pair_dist") or 0.0)),
        ("rot", "object rotation vs PRE vs time", "Δrotation (deg)", lambda f: float(f.get("delta_rot_deg") or 0.0)),
        ("mode", "contact mode / hull count vs time", "mode 0/1/2 ; n_hulls", None),
    ]
    colors = {"no_query": "#7a7a7a", "low": "#3b6ea8", "mid": "#c47b16", "high": "#b42318"}
    for name, title, ylab, fn in specs:
        fig, ax = plt.subplots(figsize=(9.2, 4.2))
        for label, frames in series.items():
            t = [f["t_s"] for f in frames]
            if name == "mode":
                ax.plot(t, [f["mode_code"] for f in frames], label=f"{label} mode", color=colors.get(label), lw=1.8)
                ax.plot(
                    t,
                    [len(f.get("all_hulls") or []) for f in frames],
                    label=f"{label} n_hulls",
                    color=colors.get(label),
                    lw=1.0,
                    ls="--",
                )
            else:
                ax.plot(t, [fn(f) for f in frames], label=label, color=colors.get(label), lw=1.8)
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylab)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}.png", dpi=140)
        plt.close(fig)


def downsample(frames: list[dict], n: int = 80) -> list[dict]:
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idx]


def main() -> None:
    bundle = build_fixture()
    task = bundle["task"]
    stock = bundle["stock"]
    snapshot = bundle["snapshot"]
    pre = bundle["pre"]
    branches = {}
    query_steps = None
    try:
        for label in ("mid", "low", "high"):
            snapshot.restore()
            audit = paint(task, LEVELS[label]["mu_finger"])
            print(json.dumps({"event": "BRANCH", "label": label, "runtime_mu": audit}), flush=True)
            before = rich_state(task, pre)
            rec = StepRecorder(task, pre)
            query = run_query(task)
            rec.close()
            after = rich_state(task, pre)
            packed = summarize_branch(label, audit, before, after, rec.frames, query)
            packed["frames"] = rec.frames
            packed["n_hooked_steps"] = rec.n
            branches[label] = packed
            dump(HERE / "branches" / f"{label}.json", {k: v for k, v in packed.items() if k != "frames"})
            dump(HERE / "branches" / f"{label}_TRACE.json", downsample(rec.frames, 200))
            print(
                json.dumps(
                    {
                        "event": "HANDOFF",
                        "label": label,
                        "squeeze": after["squeeze"],
                        "mode": after["mode"],
                        "hulls": {"L": after["left_hulls"], "R": after["right_hulls"]},
                        "dxyz_mm": None if after.get("delta_xyz") is None else [1000.0 * x for x in after["delta_xyz"]],
                        "drot_deg": after.get("delta_rot_deg"),
                        "slip_y_mm": None if after.get("rel_slip_y") is None else 1000.0 * after["rel_slip_y"],
                        "qCR": packed["stability"]["qCR"],
                        "mu_eff": audit["mu_eff"],
                    }
                ),
                flush=True,
            )
            if query_steps is None:
                query_steps = rec.n

        snapshot.restore()
        audit = paint(task, float(PLAN["no_query"]["finger_mu"]))
        before = rich_state(task, pre)
        rec = StepRecorder(task, pre)
        hold_equal(task, int(query_steps or 800))
        rec.close()
        after = rich_state(task, pre)
        packed = summarize_branch("no_query", audit, before, after, rec.frames, None)
        packed["frames"] = rec.frames
        packed["n_hooked_steps"] = rec.n
        branches["no_query"] = packed
        dump(HERE / "branches" / "no_query.json", {k: v for k, v in packed.items() if k != "frames"})
        dump(HERE / "branches" / "no_query_TRACE.json", downsample(rec.frames, 200))
        print(json.dumps({"event": "HANDOFF", "label": "no_query", "squeeze": after["squeeze"], "mode": after["mode"]}), flush=True)

        cmp = compare_mid_high(branches["mid"], branches["high"])
        low_vs_mid_hulls = branches["low"]["after"]["left_hulls"] != branches["mid"]["after"]["left_hulls"] or branches[
            "low"
        ]["after"]["right_hulls"] != branches["mid"]["after"]["right_hulls"]
        states_differ = bool(
            cmp["diverged"]
            or low_vs_mid_hulls
            or abs(branches["low"]["after"]["squeeze"] - branches["mid"]["after"]["squeeze"]) >= 1.0
        )

        probes = {}
        if cmp["diverged"] and PLAN["optional_probe"]["run_if_diverged"]:
            for label in ("low", "mid", "high"):
                snapshot.restore()
                paint(task, LEVELS[label]["mu_finger"])
                rec = StepRecorder(task, pre)
                run_query(task)
                rec.close()
                probes[label] = run_probe(task, stock, pre)
                print(
                    json.dumps(
                        {
                            "event": "PROBE",
                            "label": label,
                            "lost": probes[label].get("lost_or_collapsed"),
                            "after_squeeze": (probes[label].get("after_lift") or {}).get("squeeze"),
                        }
                    ),
                    flush=True,
                )

        if states_differ and cmp["can_explain_high_drop"]:
            verdict = "PASS"
            why = "Same PRE + same query produced different handoff states; high is a stuck/worse pinch than mid."
        elif states_differ:
            verdict = "MIXED"
            why = "Post-query states differ, but the mid/high gap does not yet clearly explain high < mid retention."
        else:
            verdict = "FAIL"
            why = "low/mid/high post-query states are essentially the same grasp."

        table = [branches[k]["row"] for k in ("no_query", "low", "mid", "high")]
        payload = {
            "verdict": verdict,
            "why": why,
            "mid_vs_high": cmp,
            "states_differ": states_differ,
            "table": table,
            "probes": probes,
            "query_physics_steps": query_steps,
        }
        dump(HERE / "COMPARE.json", table)
        dump(HERE / "MID_VS_HIGH.json", cmp)
        dump(HERE / "PROBES.json", probes)
        dump(HERE / "VERDICT.json", payload)
        plot_series(branches)
        traces = {k: downsample(branches[k]["frames"], 90) for k in ("no_query", "low", "mid", "high")}
        dump(HERE / "TRACES_DOWNSAMPLED.json", traces)
        print(json.dumps({"done": True, "verdict": verdict, "mid_vs_high": cmp}, default=str), flush=True)
    finally:
        try:
            task.close_env()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"fatal": traceback.format_exc()}), flush=True)
        raise
