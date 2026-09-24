"""Per-step real finger–grasp geometry. Collision filter must already be on."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from contact_sampler import (
    DESKBIN,
    FINGER_LINKS,
    TABLE,
    _body_name,
    _finger_state,
    _relative_pose,
    _shape_key,
    deskbin_shape_index,
)
from slab_geometry import build_slab_catalog, local_uv_and_edge, quat_geodesic_deg, quat_to_rpy, rotate_inverse, world_to_object

DEG2 = 2.0
DEG5 = 5.0
DEG10 = 10.0
EDGE_NEAR_M = 0.003
EDGE_DROP_FRAC = 0.5
PINCH_SEP_M = 0.02
STREAK = 3


def _ee_pose(task):
    ee = task.get_arm_pose("left")
    dummy = type("P", (), {})()
    if hasattr(ee, "p"):
        dummy.p = np.asarray(ee.p, dtype=float)
        dummy.q = np.asarray(ee.q, dtype=float)
    else:
        arr = np.asarray(ee, dtype=float).reshape(-1)
        dummy.p, dummy.q = arr[:3], arr[3:7]
    return dummy


def _finger_world(task) -> dict:
    out = {}
    entity = task.robot.left_entity
    for link in entity.get_links():
        name = str(link.get_name())
        if name not in {"fl_link7", "fl_link8"}:
            continue
        pose = link.get_pose()
        out[name] = {"p": [float(x) for x in pose.p], "q": [float(x) for x in pose.q]}
    return out


def _point_forces(point, dt: float):
    impulse = np.asarray(point.impulse, dtype=float)
    normal = np.asarray(point.normal, dtype=float)
    nrm = float(np.linalg.norm(normal)) + 1e-12
    normal = normal / nrm
    fn = abs(float(np.dot(impulse, normal))) / max(dt, 1e-9)
    tangential = impulse - float(np.dot(impulse, normal)) * normal
    ft = float(np.linalg.norm(tangential)) / max(dt, 1e-9)
    sep = float(getattr(point, "separation", 0.0))
    return fn, ft, normal, np.asarray(point.position, dtype=float), sep


class GraspGeomLogger:
    def __init__(self, task, out_dir: Path, commanded_f: float):
        self.task = task
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "frames").mkdir(exist_ok=True)
        self.commanded_f = float(commanded_f)
        self.dt = float(getattr(task, "physics_timestep", 1.0 / 250.0))
        self.shape_map = deskbin_shape_index(task)
        self.slabs = build_slab_catalog(task)
        self.steps = []
        self.events = []
        self.flags = {}
        self.settle_start_rel_q = None
        self.settle_end_rel_q = None
        self.settle_rel_q = None
        self.settle_rel_p = None
        self.settle_obj_z = None
        self.settle_nl = None
        self.settle_nr = None
        self.settle_row = None
        self._saved = set()
        self._uni_streak = 0
        self._both_streak = 0
        self.settle_left_pobj = None
        self.settle_right_pobj = None
        joints = task.robot.left_gripper
        self.finger_names = [joint.child_link.get_name() for joint, _, _ in joints]
        if len(self.finger_names) < 2:
            raise RuntimeError("need two left fingers")
        self.left_name, self.right_name = self.finger_names[0], self.finger_names[1]
        self.settle_edge = {self.left_name: None, self.right_name: None}

    def _phase(self) -> str:
        return str(getattr(self.task, "_af_diag_phase", "unknown"))

    def event(self, name: str, extra: dict | None = None) -> None:
        if name in self.flags:
            return
        i = len(self.steps) - 1
        row = self.steps[i] if i >= 0 else {}
        payload = {"step": int(row.get("i", i)), "t": float(row.get("t", 0.0)), "phase": row.get("phase"), **(extra or {})}
        self.flags[name] = payload
        self.events.append({"event": name, **payload})
        self.save_frame(f"{int(payload['step']):05d}_{name}")

    def save_frame(self, tag: str) -> None:
        if tag in self._saved:
            return
        try:
            self.task.save_camera_rgb(str(self.out_dir / "frames" / f"{tag}.png"), camera_name="head_camera")
            self._saved.add(tag)
        except Exception as exc:
            (self.out_dir / "frames" / f"{tag}.error.txt").write_text(repr(exc))

    def _finger_grasp(self, task) -> dict:
        per = {
            name: {
                "present": False,
                "count": 0,
                "fn": 0.0,
                "ft": 0.0,
                "sep": [],
                "p": [],
                "n": [],
                "slab": None,
                "uv": None,
                "edge": None,
            }
            for name in self.finger_names
        }
        table_fn = 0.0
        table_present = False
        obj = task.deskbin.get_pose()
        for contact in task.scene.get_contacts():
            n0 = _body_name(contact.bodies[0])
            n1 = _body_name(contact.bodies[1])
            if DESKBIN not in (n0, n1):
                continue
            metas = [None, None]
            try:
                shapes = list(contact.shapes)
            except Exception:
                shapes = []
            for index, shape in enumerate(shapes[:2]):
                try:
                    metas[index] = self.shape_map.get(_shape_key(shape))
                except Exception:
                    metas[index] = None
            role = (metas[0] or {}).get("role") if n0 == DESKBIN else (metas[1] or {}).get("role") if n1 == DESKBIN else None
            slab_name = (metas[0] or {}).get("name") if n0 == DESKBIN else (metas[1] or {}).get("name") if n1 == DESKBIN else None
            other = n1 if n0 == DESKBIN else n0
            if other == TABLE and role == "bottom":
                for point in contact.points:
                    fn, _, _, _, _ = _point_forces(point, self.dt)
                    table_fn += fn
                    table_present = table_present or fn > 1e-6
                continue
            if other not in FINGER_LINKS or role != "grasp":
                continue
            bucket = per.get(other)
            if bucket is None:
                continue
            for point in contact.points:
                fn, ft, normal, position, sep = _point_forces(point, self.dt)
                if fn < 1e-6:
                    continue
                bucket["present"] = True
                bucket["count"] += 1
                bucket["fn"] += fn
                bucket["ft"] += ft
                bucket["sep"].append(sep)
                bucket["p"].append(position * fn)
                bucket["n"].append(normal * fn)
                bucket["slab"] = slab_name
        out = {}
        for name, bucket in per.items():
            if bucket["fn"] > 0 and bucket["p"]:
                centroid = np.sum(bucket["p"], axis=0) / bucket["fn"]
                normal = np.sum(bucket["n"], axis=0) / bucket["fn"]
                nrm = float(np.linalg.norm(normal)) + 1e-12
                normal = normal / nrm
                local = world_to_object(obj, centroid)
                n_obj = rotate_inverse(obj.q, normal)
                slab = self.slabs.get(bucket["slab"] or "")
                uv = local_uv_and_edge(local, slab) if slab else None
                out[name] = {
                    "present": True,
                    "count": int(bucket["count"]),
                    "fn": float(bucket["fn"]),
                    "ft": float(bucket["ft"]),
                    "sep": float(np.mean(bucket["sep"])) if bucket["sep"] else None,
                    "p_world": [float(x) for x in centroid],
                    "n_world": [float(x) for x in normal],
                    "p_obj": [float(x) for x in local],
                    "n_obj": [float(x) for x in n_obj],
                    "slab": bucket["slab"],
                    "u": None if uv is None else uv["u"],
                    "v": None if uv is None else uv["v"],
                    "edge": None if uv is None else uv["edge"],
                    "edge_u": None if uv is None else uv["edge_u"],
                    "edge_v": None if uv is None else uv["edge_v"],
                    "u_lo": None if uv is None else uv["u_lo"],
                    "u_hi": None if uv is None else uv["u_hi"],
                    "v_lo": None if uv is None else uv["v_lo"],
                    "v_hi": None if uv is None else uv["v_hi"],
                }
            else:
                out[name] = {
                    "present": False,
                    "count": 0,
                    "fn": 0.0,
                    "ft": 0.0,
                    "sep": None,
                    "p_world": None,
                    "n_world": None,
                    "p_obj": None,
                    "n_obj": None,
                    "slab": None,
                    "u": None,
                    "v": None,
                    "edge": None,
                }
        return out, float(table_fn), bool(table_present)

    def sample(self) -> dict:
        task = self.task
        i = len(self.steps)
        t = i * self.dt
        phase = self._phase()
        gripper = _finger_state(task)
        fingers, table_fn, table_present = self._finger_grasp(task)
        left = fingers[self.left_name]
        right = fingers[self.right_name]
        nl = float(left["fn"])
        nr = float(right["fn"])
        squeeze = 2.0 * min(nl, nr)
        bilateral = bool(left["present"] and right["present"])
        unilateral = bool(left["present"] ^ right["present"])
        pose = task.deskbin.get_pose()
        ee = _ee_pose(task)
        rel_p, rel_ang, rel_q = _relative_pose(pose, ee)
        roll, pitch, yaw = quat_to_rpy(rel_q)
        holder = getattr(task, "_af_lift_holder", None)
        inner_cmd = holder.last.get("inner_aperture_m") if holder is not None and holder.last else None
        if phase == "settle":
            baseline = self.settle_start_rel_q if self.settle_start_rel_q is not None else rel_q
        else:
            baseline = self.settle_end_rel_q or self.settle_start_rel_q or rel_q
        drel = quat_geodesic_deg(baseline, rel_q)
        drel_from_start = 0.0 if self.settle_start_rel_q is None else quat_geodesic_deg(self.settle_start_rel_q, rel_q)
        pair_dist = None
        if left.get("p_world") is not None and right.get("p_world") is not None:
            pair_dist = float(np.linalg.norm(np.asarray(left["p_world"]) - np.asarray(right["p_world"])))
        same_slab = bool(left.get("present") and right.get("present") and left.get("slab") and left.get("slab") == right.get("slab"))
        pinch = bool(same_slab and pair_dist is not None and pair_dist < PINCH_SEP_M)
        fingers_world = _finger_world(task)
        finger_sep = None
        if self.left_name in fingers_world and self.right_name in fingers_world:
            finger_sep = float(
                np.linalg.norm(np.asarray(fingers_world[self.left_name]["p"]) - np.asarray(fingers_world[self.right_name]["p"]))
            )
        left_travel = None
        right_travel = None
        if self.settle_left_pobj is not None and left.get("p_obj") is not None:
            left_travel = float(np.linalg.norm(np.asarray(left["p_obj"]) - np.asarray(self.settle_left_pobj)))
        if self.settle_right_pobj is not None and right.get("p_obj") is not None:
            right_travel = float(np.linalg.norm(np.asarray(right["p_obj"]) - np.asarray(self.settle_right_pobj)))
        row = {
            "i": i,
            "t": t,
            "phase": phase,
            "F": self.commanded_f,
            "nl": nl,
            "nr": nr,
            "squeeze": squeeze,
            "bilateral": bilateral,
            "unilateral": unilateral,
            "aperture": gripper["aperture_m"],
            "qpos": gripper["qpos"],
            "inner_cmd": inner_cmd,
            "obj_p": [float(x) for x in pose.p],
            "obj_q": [float(x) for x in pose.q],
            "ee_p": [float(x) for x in ee.p],
            "ee_q": [float(x) for x in ee.q],
            "rel_p": [float(x) for x in rel_p],
            "rel_ang": float(rel_ang),
            "rel_q": [float(x) for x in rel_q],
            "rel_rpy": [float(roll), float(pitch), float(yaw)],
            "drel_from_settle_deg": float(drel),
            "drel_from_settle_start_deg": float(drel_from_start),
            "left": left,
            "right": right,
            "left_edge": left.get("edge"),
            "right_edge": right.get("edge"),
            "pair_dist": pair_dist,
            "same_slab": same_slab,
            "pinch_same_slab": pinch,
            "finger_sep": finger_sep,
            "finger_world": fingers_world,
            "left_travel": left_travel,
            "right_travel": right_travel,
            "table_bottom_fn": table_fn,
            "table_bottom": table_present,
        }
        self.steps.append(row)
        self._update_events(row)
        if i % 50 == 0:
            self.save_frame(f"{i:05d}_{phase}")
        return row

    def _update_events(self, row: dict) -> None:
        phase = row["phase"]
        if phase == "settle":
            if self.settle_start_rel_q is None:
                self.settle_start_rel_q = row["rel_q"]
                self.save_frame("settle_start")
            self.settle_end_rel_q = row["rel_q"]
            self.settle_rel_q = row["rel_q"]
            self.settle_rel_p = row["rel_p"]
            self.settle_obj_z = row["obj_p"][2]
            self.settle_row = {
                "rel_p": row["rel_p"],
                "rel_q": row["rel_q"],
                "rel_rpy": row["rel_rpy"],
                "left_u": row["left"].get("u"),
                "left_v": row["left"].get("v"),
                "left_edge": row["left"].get("edge"),
                "right_u": row["right"].get("u"),
                "right_v": row["right"].get("v"),
                "right_edge": row["right"].get("edge"),
                "nl": row["nl"],
                "nr": row["nr"],
                "aperture": row["aperture"],
                "drel_from_settle_start_deg": row["drel_from_settle_start_deg"],
            }
            if row["left"]["present"]:
                self.settle_edge[self.left_name] = row["left"]["edge"]
                self.settle_nl = row["nl"]
                self.settle_left_pobj = row["left"].get("p_obj")
            if row["right"]["present"]:
                self.settle_edge[self.right_name] = row["right"]["edge"]
                self.settle_nr = row["nr"]
                self.settle_right_pobj = row["right"].get("p_obj")
        if phase in {"lift-start", "lift"}:
            self.event("T0_lift_start")
        if phase == "wrist-rotation-start":
            self.event("wrist_rotation_start")
        if phase in {"pour-start", "pour"}:
            self.event("pour_start")
        drel = row["drel_from_settle_deg"]
        if drel > DEG2:
            self.event("T1_rel_rot_2deg", {"drel_deg": drel, "phase": phase})
        if drel > DEG5:
            self.event("T1_rel_rot_5deg", {"drel_deg": drel, "phase": phase})
        if drel > DEG10:
            self.event("T1_rel_rot_10deg", {"drel_deg": drel, "phase": phase})
        for side, key, settle_key in (("left", "left_edge", self.left_name), ("right", "right_edge", self.right_name)):
            edge = row[key]
            settle_edge = self.settle_edge.get(settle_key)
            if edge is None:
                continue
            near = float(edge) < EDGE_NEAR_M
            dropped = False
            if settle_edge is not None:
                dropped = float(edge) < EDGE_DROP_FRAC * float(settle_edge) and (float(settle_edge) - float(edge)) > 0.002
            if abs(float(edge)) > 0.2:
                continue
            if near or dropped:
                self.event(
                    "T2_centroid_near_edge",
                    {"side": side, "edge": edge, "settle_edge": settle_edge},
                )
        if (row.get("left_travel") or 0) > 0.005 or (row.get("right_travel") or 0) > 0.005:
            self.event(
                "T2_centroid_travel",
                {"left_travel": row.get("left_travel"), "right_travel": row.get("right_travel")},
            )
        if phase == "settle":
            return
        target = max(self.commanded_f, 1e-6)
        if row["nl"] < 0.5 * target or row["nr"] < 0.5 * target:
            if (self.settle_nl or 0) > 0.2 * target or (self.settle_nr or 0) > 0.2 * target:
                weak = "left" if row["nl"] <= row["nr"] else "right"
                self.event("T3_side_force_half", {"side": weak, "nl": row["nl"], "nr": row["nr"]})
        if row["unilateral"]:
            self._uni_streak += 1
            if self._uni_streak == 1:
                lost = "left" if not row["left"]["present"] else "right"
                self.event("T4_unilateral_flicker", {"lost": lost, "nl": row["nl"], "nr": row["nr"]})
            if self._uni_streak >= STREAK:
                lost = "left" if not row["left"]["present"] else "right"
                self.event("T4_unilateral_loss", {"lost": lost, "nl": row["nl"], "nr": row["nr"], "streak": self._uni_streak})
        else:
            self._uni_streak = 0
        if row["squeeze"] < 1.0:
            self.event("T5_squeeze_lt_1", {"squeeze": row["squeeze"], "nl": row["nl"], "nr": row["nr"]})
        both_gone = (not row["left"]["present"]) and (not row["right"]["present"])
        if both_gone:
            self._both_streak += 1
            if self._both_streak >= STREAK:
                self.event("T6_bilateral_loss", {"nl": row["nl"], "nr": row["nr"], "squeeze": row["squeeze"], "streak": self._both_streak})
        else:
            self._both_streak = 0

    def install(self) -> None:
        if not getattr(self.task, "_af_geom_step_wrapped", False):
            orig = self.task.scene.step

            def stepped():
                result = orig()
                logger = getattr(self.task, "_af_geom_logger", None)
                if logger is not None:
                    logger.sample()
                return result

            self.task.scene.step = stepped
            self.task._af_geom_step_wrapped = True
        self.task._af_geom_logger = self

    def detach(self) -> None:
        self.task._af_geom_logger = None

    def finish(self, retention: int) -> dict:
        if int(retention) == 0:
            last = self.steps[-1] if self.steps else {}
            if "T7_retention_loss" not in self.flags:
                self.flags["T7_retention_loss"] = {"step": last.get("i"), "t": last.get("t"), "phase": last.get("phase")}
                self.events.append({"event": "T7_retention_loss", **self.flags["T7_retention_loss"]})
                self.save_frame("terminal_retention_loss")
        self.save_frame("terminal")
        with (self.out_dir / "steps.jsonl").open("w") as stream:
            for row in self.steps:
                stream.write(json.dumps(row) + "\n")
        keys = ["event", "step", "t", "phase", "drel_deg", "side", "edge", "settle_edge", "lost", "nl", "nr", "squeeze"]
        with (self.out_dir / "events.csv").open("w") as stream:
            stream.write(",".join(keys) + "\n")
            for event in self.events:
                stream.write(",".join(str(event.get(key, "")).replace(",", ";") for key in keys) + "\n")
        order = [event["event"] for event in self.events]
        summary = {
            "commanded_f": self.commanded_f,
            "n_steps": len(self.steps),
            "dt": self.dt,
            "left_finger": self.left_name,
            "right_finger": self.right_name,
            "slabs": self.slabs,
            "flags": self.flags,
            "event_order": order,
            "settle_rel_q": self.settle_rel_q,
            "settle_start_rel_q": self.settle_start_rel_q,
            "settle_end_rel_q": self.settle_end_rel_q,
            "settle_row": self.settle_row,
            "settle_edge": self.settle_edge,
            "frames": sorted(self._saved),
        }
        (self.out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
