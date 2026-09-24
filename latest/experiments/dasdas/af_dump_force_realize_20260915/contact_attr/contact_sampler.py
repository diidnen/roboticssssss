"""Per-physics-step deskbin contact attribution. Does not change the inner loop."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FINGER_LINKS = {"fl_link7", "fl_link8", "fr_link7", "fr_link8"}
DESKBIN = "063_tabletrashbin"
TABLE = "table"
DUSTBIN = "011_dustbin"


def _shape_key(shape) -> tuple:
    vertices = np.asarray(shape.get_vertices(), dtype=float)
    return (int(len(vertices)),) + tuple(np.round(vertices.mean(0), 5))


def deskbin_shape_index(task) -> dict[tuple, dict]:
    actor = task.deskbin
    entity = actor.actor if hasattr(actor, "actor") else actor
    roles = list(getattr(actor, "_af_collision_roles", None) or [])
    names = list(getattr(actor, "_af_collision_names", None) or [])
    mapping = {}
    for component in entity.get_components():
        getter = getattr(component, "get_collision_shapes", None)
        if not callable(getter):
            continue
        shapes = list(getter())
        for index, shape in enumerate(shapes):
            mapping[_shape_key(shape)] = {
                "index": index,
                "role": roles[index] if index < len(roles) else "unknown",
                "name": names[index] if index < len(names) else f"shape{index}",
            }
    return mapping


def _body_name(body) -> str:
    try:
        return str(body.entity.name)
    except Exception:
        return "?"


def _classify_pair(name0: str, name1: str, role0: str | None, role1: str | None) -> tuple[str, bool]:
    names = {name0, name1}
    deskbin_role = role0 if name0 == DESKBIN else role1 if name1 == DESKBIN else None
    other = name1 if name0 == DESKBIN else name0 if name1 == DESKBIN else None
    if other is None:
        return "unrelated", True
    if other in FINGER_LINKS:
        if deskbin_role == "grasp":
            return f"finger-{deskbin_role}:{other}", True
        if deskbin_role in {"inner", "bottom"}:
            return f"finger-{deskbin_role}:{other}", False
        return f"finger-unknown:{other}", False
    if other == TABLE:
        return f"table-{deskbin_role or 'unknown'}", False
    if other == DUSTBIN:
        return f"dustbin-{deskbin_role or 'unknown'}", True
    if "garbage" in other or other.startswith("sphere") or "ball" in other.lower():
        return f"ball-{deskbin_role or 'unknown'}", True
    if other.startswith("fl_link") or other.startswith("fr_link") or "panda" in other:
        return f"robot_nonfinger-{other}", False
    return f"other-{other}-{deskbin_role or 'unknown'}", False


def _quat_rotate_inverse(quat_wxyz, vec):
    w, x, y, z = [float(v) for v in quat_wxyz]
    qvec = np.array([x, y, z], dtype=float)
    uv = np.cross(qvec, vec)
    uuv = np.cross(qvec, uv)
    return np.asarray(vec, dtype=float) + 2 * (w * uv + uuv)


def _relative_pose(object_pose, ee_pose):
    obj_p = np.asarray(object_pose.p, dtype=float)
    ee_p = np.asarray(ee_pose.p, dtype=float)
    ee_q = np.asarray(ee_pose.q, dtype=float)
    rel = _quat_rotate_inverse(ee_q, obj_p - ee_p)
    obj_q = np.asarray(object_pose.q, dtype=float)
    # relative rotation angle via quat multiply ee^{-1} * obj
    w0, x0, y0, z0 = ee_q
    conj = np.array([w0, -x0, -y0, -z0], dtype=float)
    w1, x1, y1, z1 = obj_q
    rel_q = np.array(
        [
            conj[0] * w1 - conj[1] * x1 - conj[2] * y1 - conj[3] * z1,
            conj[0] * x1 + conj[1] * w1 + conj[2] * z1 - conj[3] * y1,
            conj[0] * y1 - conj[1] * z1 + conj[2] * w1 + conj[3] * x1,
            conj[0] * z1 + conj[1] * y1 - conj[2] * x1 + conj[3] * w1,
        ],
        dtype=float,
    )
    angle = 2.0 * float(np.arccos(np.clip(abs(rel_q[0]), 0.0, 1.0)))
    return rel, angle, rel_q


def _finger_state(task, arm: str = "left"):
    joints = getattr(task.robot, f"{arm}_gripper")
    entity = getattr(task.robot, f"{arm}_entity")
    active = list(entity.get_active_joints())
    qpos = np.asarray(entity.qpos, dtype=float)
    qvel = np.asarray(entity.qvel, dtype=float)
    names = []
    pos = []
    vel = []
    target = []
    limits = []
    for joint, _, _ in joints:
        names.append(joint.child_link.get_name())
        index = active.index(joint)
        pos.append(float(qpos[index]))
        vel.append(float(qvel[index]))
        drive = np.asarray(joint.drive_target).reshape(-1)
        target.append(float(drive[0]) if len(drive) else 0.0)
        limits.append(float(joint.force_limit))
    return {
        "finger_names": names,
        "qpos": pos,
        "qvel": vel,
        "drive_target": target,
        "force_limit": limits,
        "aperture_m": float(np.mean(np.abs(pos))) if pos else 0.0,
    }


def _object_velocity(task):
    entity = task.deskbin.actor if hasattr(task.deskbin, "actor") else task.deskbin
    component = entity.find_component_by_type(__import__("sapien").physx.PhysxRigidDynamicComponent)
    if component is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return [float(x) for x in np.asarray(component.linear_velocity)], [
        float(x) for x in np.asarray(component.angular_velocity)
    ]


def sample_contacts(task, shape_map: dict, dt: float) -> list[dict]:
    rows = []
    for contact in task.scene.get_contacts():
        name0 = _body_name(contact.bodies[0])
        name1 = _body_name(contact.bodies[1])
        if DESKBIN not in (name0, name1):
            continue
        shape_meta = [None, None]
        try:
            shapes = list(contact.shapes)
        except Exception:
            shapes = []
        for index, shape in enumerate(shapes[:2]):
            try:
                shape_meta[index] = shape_map.get(_shape_key(shape))
            except Exception:
                shape_meta[index] = None
        role0 = (shape_meta[0] or {}).get("role") if name0 == DESKBIN else None
        role1 = (shape_meta[1] or {}).get("role") if name1 == DESKBIN else None
        name0_shape = (shape_meta[0] or {}).get("name") if name0 == DESKBIN else None
        name1_shape = (shape_meta[1] or {}).get("name") if name1 == DESKBIN else None
        kind, expected = _classify_pair(name0, name1, role0, role1)
        force = 0.0
        support_z = 0.0
        positions = []
        normals = []
        for point in contact.points:
            impulse = np.asarray(point.impulse, dtype=float)
            normal = np.asarray(point.normal, dtype=float)
            fn = abs(float(np.dot(impulse, normal))) / max(dt, 1e-9)
            force += fn
            positions.append(np.asarray(point.position, dtype=float).tolist())
            normals.append(normal.tolist())
            # impulse is on body0; if body0 is deskbin, world z component of force
            if name0 == DESKBIN:
                support_z += float(impulse[2]) / max(dt, 1e-9)
            else:
                support_z -= float(impulse[2]) / max(dt, 1e-9)
        if force < 1e-6:
            continue
        mean_p = np.mean(positions, axis=0).tolist() if positions else [0, 0, 0]
        mean_n = np.mean(normals, axis=0).tolist() if normals else [0, 0, 1]
        rows.append(
            {
                "a": name0,
                "b": name1,
                "sa": name0_shape,
                "sb": name1_shape,
                "role_a": role0,
                "role_b": role1,
                "kind": kind,
                "expected": bool(expected),
                "fn": float(force),
                "support_z": float(support_z),
                "p": [float(x) for x in mean_p],
                "n": [float(x) for x in mean_n],
            }
        )
    return rows


class ContactLogger:
    def __init__(self, task, out_dir: Path, commanded_f: float):
        self.task = task
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.commanded_f = float(commanded_f)
        self.dt = float(getattr(task, "physics_timestep", 1.0 / 250.0))
        self.shape_map = deskbin_shape_index(task)
        self.steps = []
        self.events = []
        self.flags = {
            "first_bilateral_loss": None,
            "first_squeeze_lt_1": None,
            "first_unintended": None,
            "pour_start": None,
            "retention_loss": None,
        }
        self._prev_bilateral = True
        self._prev_unintended = False
        self.frames_dir = self.out_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        self._saved_frame_tags = set()

    def _phase(self) -> str:
        return str(getattr(self.task, "_af_diag_phase", "unknown"))

    def _maybe_event(self, step_i: int, t: float, name: str, extra: dict | None = None):
        if self.flags.get(name) is not None:
            return
        self.flags[name] = {"step": int(step_i), "t": float(t), **(extra or {})}
        self.events.append({"step": int(step_i), "t": float(t), "event": name, **(extra or {})})

    def save_frame(self, tag: str) -> None:
        if tag in self._saved_frame_tags:
            return
        path = self.frames_dir / f"{tag}.png"
        try:
            self.task.save_camera_rgb(str(path), camera_name="head_camera")
            self._saved_frame_tags.add(tag)
        except Exception as exc:
            (self.frames_dir / f"{tag}.error.txt").write_text(repr(exc))

    def sample(self) -> dict:
        task = self.task
        i = len(self.steps)
        t = i * self.dt
        phase = self._phase()
        gripper = _finger_state(task)
        contact = task.get_actor_gripper_contact_forces(task.deskbin, "left")
        per = contact["per_finger_normal_force_n"]
        names = gripper["finger_names"]
        nl = float(per.get(names[0], 0.0)) if names else 0.0
        nr = float(per.get(names[1], 0.0)) if len(names) > 1 else 0.0
        single = float(contact["single_finger_normal_force_n"])
        squeeze = 2.0 * single
        bilateral = bool(contact["bilateral_contact"])
        pose = task.deskbin.get_pose()
        ee = task.get_arm_pose("left")
        ee_pose = type("P", (), {})()
        # get_arm_pose may return 7-vector
        if hasattr(ee, "p"):
            ee_p, ee_q = np.asarray(ee.p), np.asarray(ee.q)
            ee_pose.p, ee_pose.q = ee_p, ee_q
        else:
            ee = np.asarray(ee, dtype=float).reshape(-1)
            ee_pose.p, ee_pose.q = ee[:3], ee[3:7]
        rel_p, rel_ang, rel_q = _relative_pose(pose, ee_pose)
        lin, ang = _object_velocity(task)
        holder = getattr(task, "_af_lift_holder", None)
        inner_cmd = None
        if holder is not None and holder.last:
            inner_cmd = holder.last.get("inner_aperture_m")
        pairs = sample_contacts(task, self.shape_map, self.dt)
        unintended = [row for row in pairs if not row["expected"]]
        unintended_fn = float(sum(row["fn"] for row in unintended))
        support_unint = float(sum(row["support_z"] for row in unintended))
        finger_pairs = [row for row in pairs if row["kind"].startswith("finger-")]
        row = {
            "i": i,
            "t": t,
            "phase": phase,
            "F": self.commanded_f,
            "nl": nl,
            "nr": nr,
            "single": single,
            "squeeze": squeeze,
            "bilateral": bilateral,
            "aperture": gripper["aperture_m"],
            "qpos": gripper["qpos"],
            "qvel": gripper["qvel"],
            "drive_target": gripper["drive_target"],
            "force_limit": gripper["force_limit"],
            "inner_cmd": inner_cmd,
            "obj_p": [float(x) for x in pose.p],
            "obj_q": [float(x) for x in pose.q],
            "obj_v": lin,
            "obj_w": ang,
            "ee_p": [float(x) for x in ee_pose.p],
            "rel_p": [float(x) for x in rel_p],
            "rel_ang": float(rel_ang),
            "rel_q": [float(x) for x in rel_q],
            "unintended_fn": unintended_fn,
            "unintended_support_z": support_unint,
            "n_pairs": len(pairs),
            "pairs": pairs,
        }
        self.steps.append(row)
        if i == 0:
            self._prev_bilateral = bilateral
            self._prev_unintended = bool(unintended)
            if phase == "settle":
                self.save_frame("00000_settle_start")
            if unintended:
                top = max(unintended, key=lambda item: item["fn"])
                self._maybe_event(
                    i,
                    t,
                    "first_unintended",
                    {"kind": top["kind"], "a": top["a"], "b": top["b"], "sa": top["sa"], "sb": top["sb"], "fn": top["fn"]},
                )
                self.save_frame("00000_unintended_onset")
            return row
        if phase == "pour-start" and self.flags["pour_start"] is None:
            self._maybe_event(i, t, "pour_start")
            self.save_frame(f"{i:05d}_pour_start")
        if self._prev_bilateral and not bilateral and self.flags["first_bilateral_loss"] is None:
            self._maybe_event(
                i,
                t,
                "first_bilateral_loss",
                {"squeeze": squeeze, "nl": nl, "nr": nr, "phase": phase},
            )
            self.save_frame(f"{i:05d}_bilateral_loss")
        if squeeze < 1.0 and phase != "settle" and self.flags["first_squeeze_lt_1"] is None:
            self._maybe_event(
                i,
                t,
                "first_squeeze_lt_1",
                {"squeeze": squeeze, "nl": nl, "nr": nr, "phase": phase, "inner_cmd": inner_cmd},
            )
            self.save_frame(f"{i:05d}_squeeze_lt_1")
        if unintended and not self._prev_unintended and self.flags["first_unintended"] is None:
            top = max(unintended, key=lambda item: item["fn"])
            self._maybe_event(
                i,
                t,
                "first_unintended",
                {"kind": top["kind"], "a": top["a"], "b": top["b"], "sa": top["sa"], "sb": top["sb"], "fn": top["fn"]},
            )
            self.save_frame(f"{i:05d}_unintended")
        self._prev_bilateral = bilateral
        self._prev_unintended = bool(unintended)
        return row

    def install(self) -> None:
        if getattr(self.task, "_af_contact_log_installed", False):
            self.task._af_contact_logger = self
            return
        orig = self.task.scene.step

        def stepped():
            result = orig()
            logger = getattr(self.task, "_af_contact_logger", None)
            if logger is not None:
                logger.sample()
            return result

        self.task.scene.step = stepped
        self.task._af_contact_log_installed = True
        self.task._af_contact_logger = self

    def detach(self) -> None:
        self.task._af_contact_logger = None

    def dump(self) -> dict:
        path = self.out_dir / "steps.jsonl"
        with path.open("w") as stream:
            for row in self.steps:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        events_path = self.out_dir / "events.csv"
        keys = ["step", "t", "event", "phase", "squeeze", "nl", "nr", "kind", "a", "b", "sa", "sb", "fn", "inner_cmd"]
        with events_path.open("w") as stream:
            stream.write(",".join(keys) + "\n")
            for event in self.events:
                stream.write(
                    ",".join(str(event.get(key, "")).replace(",", ";") for key in keys) + "\n"
                )
        pair_max: dict[str, dict] = {}
        pair_first: dict[str, dict] = {}
        for row in self.steps:
            for pair in row["pairs"]:
                key = f"{pair['a']}|{pair['b']}|{pair.get('sa')}|{pair.get('sb')}|{pair['kind']}"
                if key not in pair_first:
                    pair_first[key] = {"t": row["t"], "step": row["i"], **pair}
                current = pair_max.get(key)
                if current is None or pair["fn"] > current["fn"]:
                    pair_max[key] = {"t": row["t"], "step": row["i"], **pair}
        table = []
        for key, first in pair_first.items():
            peak = pair_max[key]
            table.append(
                {
                    "geom_a": first["a"],
                    "geom_b": first["b"],
                    "shape_a": first.get("sa"),
                    "shape_b": first.get("sb"),
                    "kind": first["kind"],
                    "expected": first["expected"],
                    "first_t": first["t"],
                    "first_step": first["step"],
                    "max_fn": peak["fn"],
                    "max_fn_t": peak["t"],
                    "max_support_z": peak.get("support_z"),
                }
            )
        table.sort(key=lambda item: (-float(item["max_fn"]), item["first_t"]))
        (self.out_dir / "contact_pairs.json").write_text(json.dumps(table, indent=2) + "\n")
        summary = {
            "commanded_f": self.commanded_f,
            "n_steps": len(self.steps),
            "dt": self.dt,
            "flags": self.flags,
            "n_events": len(self.events),
            "frames": sorted(self._saved_frame_tags),
        }
        (self.out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
