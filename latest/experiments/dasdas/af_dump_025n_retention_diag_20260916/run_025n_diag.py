"""Diagnostic only: why F_cmd=0.25 N retains under grasp-surface isolation.

Does not change query, controller, task criterion, 32x20, official 18/19, or Fig.B.
Near-zero-mu fork is a counterfactual, not paper data.
"""
from __future__ import annotations

import json
import math
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ISO15 = Path("/media/volume/dasdas/exouser/af_dump_grasp_surface_friction_only_20260915")
ISO16 = Path("/media/volume/dasdas/exouser/af_dump_grasp_surface_friction_only_20260916")
V4 = Path("/media/volume/dasdas/exouser/af_dump_liftstyle_feas_v4_relabel_20260915")
ATTR = Path("/media/volume/dasdas/exouser/af_dump_force_realize_20260915/contact_attr")
ROOT = Path("/media/volume/dasdas/exouser/af_dump_force_realize_20260915")
REPO = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/RoboTwin")
sys.path[:0] = [str(ISO16), str(ISO15), str(V4), str(REPO), str(ROOT), str(ATTR), str(HERE)]

import sapien  # noqa: E402

from collision_filter import apply_retention_collision_filters  # noqa: E402
from contact_sampler import DESKBIN, FINGER_LINKS, _body_name, _shape_key, deskbin_shape_index  # noqa: E402
from envs.utils import ArmTag  # noqa: E402
from force_realize import (  # noqa: E402
    LiftSqueezeHold,
    SETTLE_STEPS,
    capture_stock_drives,
    install_lift_squeeze,
    restore_stock_drives,
    uninstall_lift_squeeze,
)
from grasp_geom_logger import GraspGeomLogger, _ee_pose, _finger_state, _point_forces  # noqa: E402
from patch_grasp_surface import (  # noqa: E402
    FIXED_MU,
    apply_grasp_surface_materials,
    contact_material_audit,
    deskbin_shapes,
)
from run_025n_support import aabb_overlap, downsample, hull_audit, json_dump, material_rows  # noqa: E402
from run_isolation_sweep import capture_valid_prefix, pack_state  # noqa: E402
from run_sweep_context import labels_from_metrics  # noqa: E402
from slab_geometry import build_slab_catalog, world_to_object  # noqa: E402

SEED = 200014
MU_NORMAL = 0.425
MU_NEAR_ZERO = 0.001
F_CMD = 0.25
G_FALLBACK = 9.81
CHECK_TIMES = (0.0, 0.1, 0.2, 0.5, 1.0)


def gravity_ms2(task) -> float:
    for obj in (
        getattr(task.scene, "physx_system", None),
        task.scene,
        getattr(task, "scene", None),
    ):
        if obj is None:
            continue
        for name in ("gravity", "get_gravity"):
            val = getattr(obj, name, None)
            if val is None:
                continue
            if callable(val):
                val = val()
            arr = np.asarray(val, dtype=float).reshape(-1)
            if arr.size >= 3:
                return float(np.linalg.norm(arr[:3]))
            if arr.size == 1:
                return abs(float(arr[0]))
    cfg = None
    try:
        import sapien.physx as physx

        cfg = physx.get_scene_config()
    except Exception:
        cfg = None
    if cfg is not None:
        val = getattr(cfg, "gravity", None)
        if val is not None:
            arr = np.asarray(val, dtype=float).reshape(-1)
            if arr.size >= 3:
                return float(np.linalg.norm(arr[:3]))
    return G_FALLBACK


def read_deskbin_mass(task) -> dict:
    entity = task.deskbin.actor if hasattr(task.deskbin, "actor") else task.deskbin
    components = []
    if hasattr(entity, "get_components"):
        components.extend(list(entity.get_components()))
    rows = []
    for component in components:
        if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
            row = {"type": type(component).__name__, "mass_kg": float(component.mass)}
            try:
                row["cm"] = [float(x) for x in np.asarray(component.cmass_local_pose.p, dtype=float)]
            except Exception:
                row["cm"] = None
            try:
                row["inertia"] = [float(x) for x in np.asarray(component.inertia, dtype=float).reshape(-1)[:3]]
            except Exception:
                row["inertia"] = None
            rows.append(row)
    balls = []
    for sphere in getattr(task, "sphere_lst", []) or []:
        raw = sphere.actor if hasattr(sphere, "actor") else sphere
        getter = getattr(raw, "get_components", None) or getattr(raw, "find_component_by_type", None)
        mass = None
        if hasattr(raw, "find_component_by_type"):
            comp = raw.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            if comp is not None:
                mass = float(comp.mass)
        balls.append(mass)
    total = float(sum(row["mass_kg"] for row in rows))
    return {
        "deskbin_components": rows,
        "deskbin_mass_kg": total,
        "ball_masses_kg": balls,
        "n_balls": len(balls),
        "balls_total_kg": float(sum(x for x in balls if x is not None)),
        "af_object_mass_kg": getattr(task, "af_object_mass_kg", None),
        "actor_default_note": "Actor.__init__ calls set_mass(0.01) unless af_object_mass_kg overrides",
    }


def finger_mu_measured(task) -> dict:
    rows = []
    for link in task.robot.left_entity.get_links():
        name = str(link.get_name())
        if name not in {"fl_link7", "fl_link8"}:
            continue
        getter = getattr(link, "get_collision_shapes", None)
        if not callable(getter):
            continue
        for shape in getter():
            mat = shape.get_physical_material()
            rows.append(
                {
                    "link": name,
                    "static_friction": float(mat.static_friction),
                    "dynamic_friction": float(mat.dynamic_friction),
                }
            )
    mus = [r["static_friction"] for r in rows]
    return {
        "shapes": rows,
        "finger_mu": float(np.mean(mus)) if mus else None,
    }


def dump_contacts(task, helper: GraspGeomLogger) -> dict:
    dt = float(getattr(task, "physics_timestep", 1.0 / 250.0))
    shape_map = helper.shape_map
    obj = task.deskbin.get_pose()
    points = []
    per_finger = defaultdict(lambda: {"shapes": set(), "roles": set(), "n_points": 0, "fn": 0.0})
    for contact in task.scene.get_contacts():
        n0 = _body_name(contact.bodies[0])
        n1 = _body_name(contact.bodies[1])
        if DESKBIN not in (n0, n1):
            continue
        other = n1 if n0 == DESKBIN else n0
        if other not in FINGER_LINKS:
            continue
        metas = [None, None]
        try:
            shapes = list(contact.shapes)
        except Exception:
            shapes = []
        for index, shape in enumerate(shapes[:2]):
            try:
                metas[index] = shape_map.get(_shape_key(shape))
            except Exception:
                metas[index] = None
        deskbin_meta = metas[0] if n0 == DESKBIN else metas[1] if n1 == DESKBIN else None
        shape_name = (deskbin_meta or {}).get("name")
        role = (deskbin_meta or {}).get("role")
        for p_i, point in enumerate(contact.points):
            fn, ft, normal, position, sep = _point_forces(point, dt)
            p_obj = world_to_object(obj, position)
            rec = {
                "finger": other,
                "deskbin_shape": shape_name,
                "role": role,
                "point_index": int(p_i),
                "position_world": [float(x) for x in position],
                "position_obj": [float(x) for x in p_obj],
                "normal_world": [float(x) for x in normal],
                "fn": float(fn),
                "ft": float(ft),
                "separation": float(sep),
                "impulse": [float(x) for x in np.asarray(point.impulse, dtype=float)],
            }
            points.append(rec)
            bucket = per_finger[other]
            bucket["n_points"] += 1
            bucket["fn"] += float(fn)
            if shape_name:
                bucket["shapes"].add(str(shape_name))
            if role:
                bucket["roles"].add(str(role))
    fingers = {}
    for name, bucket in per_finger.items():
        fingers[name] = {
            "n_points": bucket["n_points"],
            "fn": bucket["fn"],
            "shapes": sorted(bucket["shapes"]),
            "n_shapes": len(bucket["shapes"]),
            "roles": sorted(bucket["roles"]),
            "multi_hull": len(bucket["shapes"]) >= 2,
        }
    normals = [row["normal_world"] for row in points if row["fn"] > 1e-6]
    opposing = None
    if len(normals) >= 2:
        dots = []
        for i, a in enumerate(normals):
            for b in normals[i + 1 :]:
                dots.append(float(np.dot(a, b)))
        opposing = {
            "min_dot": float(min(dots)),
            "max_dot": float(max(dots)),
            "mean_dot": float(np.mean(dots)),
            "any_not_antiparallel": bool(min(dots) > -0.85) if dots else None,
        }
    return {
        "n_points": len(points),
        "points": points,
        "per_finger": fingers,
        "any_multi_hull_finger": any(v["multi_hull"] for v in fingers.values()),
        "normal_summary": opposing,
        "deskbin_xyz": [float(x) for x in obj.p],
        "deskbin_xyzw": [float(x) for x in obj.q],
    }


def official_squeeze(task) -> dict:
    contact = task.get_actor_gripper_contact_forces(task.deskbin, "left")
    n_min = float(contact["single_finger_normal_force_n"])
    per = {k: float(v) for k, v in (contact.get("per_finger_normal_force_n") or {}).items()}
    return {
        "N_min_official": n_min,
        "s_official": 2.0 * n_min,
        "per_finger_official": per,
        "bilateral_official": bool(contact.get("bilateral_contact")),
    }


class Recorder:
    def __init__(self, task, helper: GraspGeomLogger, out_dir: Path, tag: str):
        self.task = task
        self.helper = helper
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "frames").mkdir(exist_ok=True)
        self.tag = tag
        self.rows = []
        self.keyframes = {}
        self.phase_starts = {}
        self.dt = float(getattr(task, "physics_timestep", 1.0 / 250.0))
        self.t0_cmd = None
        self.drop_t = None
        self.first_bilateral_loss_t = None
        self._stream = (self.out_dir / "steps.jsonl").open("w")

    def _t(self) -> float:
        return len(self.rows) * self.dt

    def save_camera(self, name: str) -> None:
        path = self.out_dir / "frames" / f"{name}.png"
        try:
            self.task.save_camera_rgb(str(path), camera_name="head_camera")
        except Exception as exc:
            (self.out_dir / "frames" / f"{name}.error.txt").write_text(repr(exc))

    def sample(self, keyframe: str | None = None) -> dict:
        task = self.task
        helper = self.helper
        phase = str(getattr(task, "_af_diag_phase", "unknown"))
        if phase not in self.phase_starts:
            self.phase_starts[phase] = {"i": len(self.rows), "t": self._t()}
        packed = pack_state(task, helper)
        official = official_squeeze(task)
        holder = getattr(task, "_af_lift_holder", None)
        inner = dict(holder.last) if holder is not None and holder.last else {}
        fingers, table_fn, _ = helper._finger_grasp(task)
        left = fingers[helper.left_name]
        right = fingers[helper.right_name]
        shapes_l, shapes_r = set(), set()
        contacts = dump_contacts(task, helper) if keyframe else None
        if contacts is None:
            # cheap shape-id set without full dump
            dt = self.dt
            shape_map = helper.shape_map
            for contact in task.scene.get_contacts():
                n0 = _body_name(contact.bodies[0])
                n1 = _body_name(contact.bodies[1])
                if DESKBIN not in (n0, n1):
                    continue
                other = n1 if n0 == DESKBIN else n0
                if other not in FINGER_LINKS:
                    continue
                metas = [None, None]
                try:
                    shapes = list(contact.shapes)
                except Exception:
                    shapes = []
                for index, shape in enumerate(shapes[:2]):
                    try:
                        metas[index] = shape_map.get(_shape_key(shape))
                    except Exception:
                        metas[index] = None
                meta = metas[0] if n0 == DESKBIN else metas[1]
                name = (meta or {}).get("name")
                if name and other == helper.left_name:
                    shapes_l.add(str(name))
                if name and other == helper.right_name:
                    shapes_r.add(str(name))
        else:
            shapes_l = set((contacts["per_finger"].get(helper.left_name) or {}).get("shapes") or [])
            shapes_r = set((contacts["per_finger"].get(helper.right_name) or {}).get("shapes") or [])
        F_cmd = float(getattr(holder, "force_n", F_CMD)) if holder is not None else None
        row = {
            "i": len(self.rows),
            "t": self._t(),
            "phase": phase,
            "F_cmd": F_cmd,
            "force_ref_bilateral": float(getattr(holder, "force_ref", 2.0 * F_CMD)) if holder is not None else None,
            "nl": packed["nl"],
            "nr": packed["nr"],
            "s": packed["squeeze"],
            "N_min": min(packed["nl"], packed["nr"]),
            "s_official": official["s_official"],
            "N_min_official": official["N_min_official"],
            "aperture": packed["aperture"],
            "pair_dist": packed["pair_dist"],
            "obj_z": packed["obj_z"],
            "bilateral": packed["bilateral"],
            "nx_nx": packed["nx_nx"],
            "left_slab": packed["left_slab"],
            "right_slab": packed["right_slab"],
            "left_shapes": sorted(shapes_l),
            "right_shapes": sorted(shapes_r),
            "multi_hull": bool(len(shapes_l) >= 2 or len(shapes_r) >= 2),
            "table_bottom_n": packed["table_bottom_n"],
            "inner_aperture_m": inner.get("inner_aperture_m"),
            "inner_measured_raw": inner.get("bilateral_n"),
            "left_count": int(left.get("count") or 0),
            "right_count": int(right.get("count") or 0),
            "left_n_world": left.get("n_world"),
            "right_n_world": right.get("n_world"),
        }
        self.rows.append(row)
        self._stream.write(json.dumps(row, default=str) + "\n")
        if keyframe:
            self._store_keyframe(keyframe, row, contacts, packed, official)
        if self.drop_t is None and (not row["bilateral"]) and row["obj_z"] < 0.85:
            self.drop_t = row["t"]
        if self.first_bilateral_loss_t is None and len(self.rows) > 5 and not row["bilateral"]:
            self.first_bilateral_loss_t = row["t"]
        return row

    def _store_keyframe(self, keyframe, row, contacts, packed, official) -> None:
        payload = {
            "keyframe": keyframe,
            "row": row,
            "contacts": contacts,
            "pack": packed,
            "official": official,
        }
        self.keyframes[keyframe] = payload
        json_dump(self.out_dir / "keyframes" / f"{keyframe}.json", payload)
        self.save_camera(keyframe)

    def mark_keyframe(self, keyframe: str) -> dict:
        """Dump full contacts for the current physics state without adding a time sample."""
        if not self.rows:
            return self.sample(keyframe=keyframe)
        packed = pack_state(self.task, self.helper)
        official = official_squeeze(self.task)
        contacts = dump_contacts(self.task, self.helper)
        row = dict(self.rows[-1])
        self._store_keyframe(keyframe, row, contacts, packed, official)
        return row

    def close(self) -> None:
        self._stream.close()
        json_dump(self.out_dir / "phase_starts.json", self.phase_starts)
        json_dump(self.out_dir / "drop.json", {"drop_t": self.drop_t, "first_bilateral_loss_t": self.first_bilateral_loss_t})


def install_recorder(task, recorder: Recorder) -> None:
    orig = task.scene.step

    def stepped():
        result = orig()
        rec = getattr(task, "_af_025_recorder", None)
        if rec is not None:
            rec.sample()
        return result

    task.scene.step = stepped
    task._af_025_recorder = recorder
    task._af_025_prev_step = orig


def uninstall_recorder(task) -> None:
    task._af_025_recorder = None
    prev = getattr(task, "_af_025_prev_step", None)
    if prev is not None:
        task.scene.step = prev


def remainder_tagged(task) -> None:
    place = ArmTag("left")
    task._af_diag_phase = "lift"
    task.move(task.move_by_displacement(arm_tag=place, z=0.08, move_axis="arm"))
    task._af_diag_phase = "pour1"
    task.move(task.pour_actions)
    task._af_diag_phase = "pour2"
    task.move(task.pour_actions)
    task._af_diag_phase = "pour3"
    task.move(task.pour_actions)
    task._af_diag_phase = "delay"
    task.delay(6)


def checkpoints(rows: list[dict], t0: float) -> dict:
    out = {}
    for dt in CHECK_TIMES:
        target = t0 + dt
        hit = min(rows, key=lambda r: abs(r["t"] - target)) if rows else None
        out[f"t0_plus_{dt:.1f}s"] = None if hit is None else {
            "t": hit["t"],
            "phase": hit["phase"],
            "s": hit["s"],
            "N_min": hit["N_min"],
            "s_official": hit["s_official"],
            "aperture": hit["aperture"],
            "obj_z": hit["obj_z"],
            "bilateral": hit["bilateral"],
        }
    return out


def phase_stats(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for row in rows:
        by[row["phase"]].append(row)
    out = {}
    for phase, items in by.items():
        s = np.asarray([r["s"] for r in items], dtype=float)
        nmin = np.asarray([r["N_min"] for r in items], dtype=float)
        z = np.asarray([r["obj_z"] for r in items], dtype=float)
        out[phase] = {
            "n": len(items),
            "t_start": items[0]["t"],
            "t_end": items[-1]["t"],
            "s_mean": float(np.mean(s)),
            "s_median": float(np.median(s)),
            "s_p05": float(np.percentile(s, 5)),
            "s_p95": float(np.percentile(s, 95)),
            "N_min_mean": float(np.mean(nmin)),
            "N_min_median": float(np.median(nmin)),
            "bilateral_frac": float(np.mean([bool(r["bilateral"]) for r in items])),
            "nxnx_frac": float(np.mean([bool(r["nx_nx"]) for r in items])),
            "multi_hull_frac": float(np.mean([bool(r["multi_hull"]) for r in items])),
            "z_mean": float(np.mean(z)),
            "z_min": float(np.min(z)),
            "z_max": float(np.max(z)),
            "pair_median": float(np.median([r["pair_dist"] if r["pair_dist"] is not None else math.nan for r in items])),
        }
    return out


def unload_time(rows: list[dict], t0: float, threshold: float) -> float | None:
    for row in rows:
        if row["t"] < t0:
            continue
        if row["s"] <= threshold:
            return float(row["t"] - t0)
    return None


def capacity_block(mass_kg: float, g: float, mu_finger: float, mu_obj: float, N: float) -> dict:
    mu_eff_avg = 0.5 * (mu_finger + mu_obj)
    mu_eff_min = min(mu_finger, mu_obj)
    mu_eff_mul = mu_finger * mu_obj
    mg = mass_kg * g
    f_avg = 2.0 * mu_eff_avg * N
    return {
        "mass_kg": mass_kg,
        "g": g,
        "mg": mg,
        "mu_finger": mu_finger,
        "mu_object": mu_obj,
        "mu_eff_average": mu_eff_avg,
        "mu_eff_min": mu_eff_min,
        "mu_eff_multiply": mu_eff_mul,
        "N_used": N,
        "F_friction_max_average": f_avg,
        "F_friction_max_min": 2.0 * mu_eff_min * N,
        "capacity_over_mg_average": f_avg / mg if mg > 1e-12 else None,
        "enough_average": bool(f_avg >= mg) if mg == mg else None,
    }


def plot_run(rows: list[dict], out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    t = np.asarray([r["t"] for r in rows], dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t, [r["s"] for r in rows], label="s=2*min(NL,NR)", lw=1.2)
    axes[0].plot(t, [r["nl"] for r in rows], label="N_L", lw=0.8, alpha=0.8)
    axes[0].plot(t, [r["nr"] for r in rows], label="N_R", lw=0.8, alpha=0.8)
    axes[0].axhline(0.25, color="k", ls="--", lw=0.8, label="0.25 N")
    axes[0].axhline(0.50, color="k", ls=":", lw=0.8, label="0.50 N = 2*F_cmd")
    axes[0].axhline(1.0, color="0.5", ls=":", lw=0.6, label="1 N")
    axes[0].set_ylabel("force (N)")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(t, [r["aperture"] * 1000 for r in rows], label="aperture mm")
    pair = [np.nan if r["pair_dist"] is None else r["pair_dist"] * 1000 for r in rows]
    axes[1].plot(t, pair, label="pair dist mm")
    axes[1].set_ylabel("mm")
    axes[1].legend(fontsize=8)
    axes[2].plot(t, [r["obj_z"] for r in rows], label="deskbin z")
    axes[2].axhline(1.0, color="k", ls="--", lw=0.8, label="z=1 dump thresh")
    axes[2].set_ylabel("z (m)")
    axes[2].legend(fontsize=8)
    axes[3].plot(t, [int(bool(r["bilateral"])) for r in rows], label="bilateral")
    axes[3].plot(t, [int(bool(r["nx_nx"])) for r in rows], label="nx+nx")
    axes[3].plot(t, [int(bool(r["multi_hull"])) for r in rows], label="multi-hull")
    axes[3].set_ylabel("flag")
    axes[3].set_xlabel("t (s) from recorder start")
    axes[3].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        # phase spans
    phases = []
    for row in rows:
        if not phases or phases[-1][0] != row["phase"]:
            phases.append([row["phase"], row["t"], row["t"]])
        else:
            phases[-1][2] = row["t"]
    colors = {"post_query": "#dddddd", "realize": "#ffe6cc", "lift": "#cce5ff", "pour1": "#d5f5d5", "pour2": "#b7e4b7", "pour3": "#8fd18f", "delay": "#eeeeee"}
    for phase, a, b in phases:
        axes[0].axvspan(a, b, color=colors.get(phase, "#f0f0f0"), alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_contacts(kf: dict, hulls: dict, out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    points = (kf.get("contacts") or {}).get("points") or []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    pairs = [(0, 1, "object X vs Y"), (0, 2, "object X vs Z")]
    for ax, (i, j, lab) in zip(axes, pairs):
        for name, slab in hulls.items():
            if name.startswith("_"):
                continue
            lo = np.asarray(slab["min"], dtype=float)
            size = np.asarray(slab["size"], dtype=float)
            ax.add_patch(
                Rectangle(
                    (lo[i], lo[j]),
                    size[i],
                    size[j],
                    fill=False,
                    lw=1.0,
                    label=name,
                )
            )
        for pt in points:
            p = pt["position_obj"]
            color = "C0" if pt["finger"] == "fl_link7" else "C3"
            ax.scatter(p[i], p[j], c=color, s=18, zorder=5)
            n = pt["normal_world"]
            # normal is world; skip arrow in object frame if mixed
            ax.annotate("", xy=(p[i] + 0.01 * np.sign(n[i] if i < 3 else 0), p[j]), xytext=(p[i], p[j]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.6))
        ax.set_title(lab)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[1].legend(uniq.values(), uniq.keys(), fontsize=7, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def run_fork(task, stock, helper, snapshot, grasp_mu: float, tag: str, out: Path) -> dict:
    snapshot.restore()
    apply_grasp_surface_materials(task, grasp_mu)
    apply_retention_collision_filters(task)
    helper.shape_map = deskbin_shape_index(task)
    helper.slabs = build_slab_catalog(task)
    task.plan_success = True
    task._af_force_trace = []
    task._af_force_trace_step = 0
    task._af_no_contact_samples = 0
    task._af_ever_lifted = False
    run_dir = out / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(task, helper, run_dir, tag)
    helper.commanded_f = F_CMD
    task._af_diag_phase = "post_query"
    post = recorder.sample(keyframe="post_query")
    query_end_squeeze = {
        "s": post["s"],
        "N_min": post["N_min"],
        "s_official": post["s_official"],
        "aperture": post["aperture"],
        "obj_z": post["obj_z"],
        "nl": post["nl"],
        "nr": post["nr"],
        "left_slab": post["left_slab"],
        "right_slab": post["right_slab"],
    }
    holder = LiftSqueezeHold(task, F_CMD, stock)
    install_lift_squeeze(task, holder)
    install_recorder(task, recorder)
    task.af_force_limit_n = float(F_CMD)
    task._af_diag_phase = "realize"
    t0 = recorder._t()
    recorder.t0_cmd = t0
    recorder.sample(keyframe="Fcmd_set")
    want = {24, 49, 124, int(SETTLE_STEPS) - 1}
    for step in range(int(SETTLE_STEPS)):
        task.scene.step()
        if step in want:
            tag = "realize_end" if step == int(SETTLE_STEPS) - 1 else f"realize_step{step+1}"
            recorder.mark_keyframe(tag)
    lift_start_t = recorder._t()
    task._af_force_trace = []
    task._af_force_trace_step = 0
    task._af_no_contact_samples = 0
    task._af_ever_lifted = False
    task.activate_activeforcing_candidate_force()
    restore_stock_drives(stock)
    remainder_error = None
    try:
        remainder_tagged(task)
    except Exception as exc:
        remainder_error = repr(exc)
        traceback.print_exc()
    recorder.mark_keyframe("terminal")
    uninstall_recorder(task)
    uninstall_lift_squeeze(task)
    recorder.close()
    metrics = task.compute_activeforcing_dynamic_metrics()
    labels = labels_from_metrics(task, metrics)
    stats = phase_stats(recorder.rows)
    cps = checkpoints(recorder.rows, t0)
    lift_rows = [r for r in recorder.rows if r["phase"] in {"lift", "pour1", "pour2", "pour3"}]
    high_load = {
        "n": len(lift_rows),
        "s_mean": float(np.mean([r["s"] for r in lift_rows])) if lift_rows else None,
        "s_median": float(np.median([r["s"] for r in lift_rows])) if lift_rows else None,
        "N_min_mean": float(np.mean([r["N_min"] for r in lift_rows])) if lift_rows else None,
        "N_min_median": float(np.median([r["N_min"] for r in lift_rows])) if lift_rows else None,
        "s_p95": float(np.percentile([r["s"] for r in lift_rows], 95)) if lift_rows else None,
    }
    actual_near_025 = None
    if high_load["s_median"] is not None:
        # A if actual bilateral squeeze stays clearly above ~1 N, or N_min >> 0.25
        actual_near_025 = bool(high_load["s_median"] <= 1.0 and high_load["N_min_median"] <= 0.75)
    try:
        plot_run(recorder.rows, run_dir / "timeseries.png", f"{tag}  F_cmd=0.25 N  grasp μ={grasp_mu}")
        hulls = build_slab_catalog(task)
        for name in ("post_query", "realize_end", "terminal"):
            if name in recorder.keyframes:
                plot_contacts(recorder.keyframes[name], hulls, run_dir / f"contacts_{name}.png", f"{tag} {name}")
    except Exception as exc:
        (run_dir / "plot.error.txt").write_text(repr(exc) + "\n" + traceback.format_exc())
    json_dump(
        run_dir / "SUMMARY.json",
        {
            "tag": tag,
            "grasp_mu": grasp_mu,
            "F_cmd": F_CMD,
            "query_end": query_end_squeeze,
            "t0_Fcmd": t0,
            "lift_start_t": lift_start_t,
            "unload_to_s_le_0.75": unload_time(recorder.rows, t0, 0.75),
            "unload_to_s_le_0.50": unload_time(recorder.rows, t0, 0.50),
            "unload_to_s_le_0.375": unload_time(recorder.rows, t0, 0.375),
            "checkpoints": cps,
            "phase_stats": stats,
            "high_load": high_load,
            "actual_squeeze_near_commanded": actual_near_025,
            "labels": labels,
            "metrics": {k: metrics.get(k) for k in ("samples", "contact_ratio", "measured_force_mean_n", "irrecoverable_failure")},
            "drop_t": recorder.drop_t,
            "first_bilateral_loss_t": recorder.first_bilateral_loss_t,
            "remainder_error": remainder_error,
            "n_steps": len(recorder.rows),
            "keyframe_names": list(recorder.keyframes),
            "downsample": downsample(recorder.rows, 0.05),
        },
    )
    return json.loads((run_dir / "SUMMARY.json").read_text())


def classify(normal: dict, zero: dict, mass: dict, cap_cmd: dict, cap_actual: dict, geom: dict) -> dict:
    n_high = normal.get("high_load") or {}
    actual_near = bool(normal.get("actual_squeeze_near_commanded"))
    s_med = n_high.get("s_median")
    n_med = n_high.get("N_min_median")
    controller = bool(s_med is not None and (s_med > 1.0 or (n_med is not None and n_med > 0.75)))
    low_mass = bool(actual_near and cap_actual.get("enough_average"))
    zero_ret = int((zero.get("labels") or {}).get("retention_success") or 0)
    zero_drop = (zero.get("labels") or {}).get("drop")
    zero_cr = (zero.get("labels") or {}).get("contact_ratio")
    geom_flag = bool(geom.get("any_multi_hull") or geom.get("hulls_overlap") or geom.get("normals_not_simple_pinch"))
    zero_still = bool(zero_ret == 1 and (zero_cr or 0) >= 0.8)
    zero_easier = bool(zero_ret == 0 or (zero_cr or 1) < 0.5 or zero_drop == 1)
    bits = []
    if controller:
        bits.append("CONTROLLER / PRELOAD")
    if low_mass or (actual_near and cap_cmd.get("enough_average")):
        bits.append("LOW MASS / FRICTION IS ENOUGH")
    if zero_still and geom_flag:
        bits.append("GEOMETRIC WEDGING / FORM CLOSURE")
    elif zero_still:
        bits.append("GEOMETRIC WEDGING / FORM CLOSURE")
    if len(bits) == 0:
        label = "UNKNOWN"
        reason = "time series did not uniquely match A/B/C"
    elif len(bits) == 1:
        label = bits[0]
        reason = bits[0]
    else:
        label = "MIXED"
        reason = " + ".join(bits)
    return {
        "label": label,
        "reason": reason,
        "controller_preload": controller,
        "low_mass_friction_enough": bool(low_mass or (actual_near and cap_cmd.get("enough_average"))),
        "geometry_supported": bool(zero_still),
        "near_zero_easier_drop": zero_easier,
        "actual_squeeze_near_0.25_command": actual_near,
        "normal_retention": (normal.get("labels") or {}).get("retention_success"),
        "near_zero_retention": zero_ret,
        "near_zero_contact_ratio": zero_cr,
        "s_median_high_load": s_med,
        "N_min_median_high_load": n_med,
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "runs").mkdir(exist_ok=True)
    print(json.dumps({"status": "capturing PRE_VALID", "seed": SEED, "mu": MU_NORMAL}), flush=True)
    try:
        captured = capture_valid_prefix(SEED, MU_NORMAL, episode=1)
    except Exception as exc:
        print(json.dumps({"episode_1_failed": repr(exc), "scanning": True}), flush=True)
        captured = capture_valid_prefix(SEED, MU_NORMAL, episode=None)
    task = captured["task"]
    helper = captured["helper"]
    try:
        mass = read_deskbin_mass(task)
        g = gravity_ms2(task)
        fingers = finger_mu_measured(task)
        mu_finger = float(fingers["finger_mu"] if fingers["finger_mu"] is not None else 0.3)
        hulls = hull_audit(task)
        slabs = build_slab_catalog(task)
        overlap = aabb_overlap(slabs)
        assignment = apply_grasp_surface_materials(task, MU_NORMAL)
        json_dump(
            HERE / "MASS_MU_HULL.json",
            {
                "mass": mass,
                "g": g,
                "fingers": fingers,
                "hulls": hulls,
                "slabs": {k: v for k, v in slabs.items() if not str(k).startswith("_")},
                "aabb_overlap": overlap,
                "assignment_normal": assignment,
                "pre_gate": captured["pre_gate"],
                "episode_id": captured["episode_id"],
                "deskbin_id": captured["deskbin_id"],
                "post_query": captured["post_query"],
                "query": captured["query"],
            },
        )
        N_cmd = F_CMD
        cap_cmd = capacity_block(mass["deskbin_mass_kg"], g, mu_finger, MU_NORMAL, N_cmd)
        # also with balls
        cap_cmd_with_balls = capacity_block(
            mass["deskbin_mass_kg"] + mass["balls_total_kg"], g, mu_finger, MU_NORMAL, N_cmd
        )
        print(json.dumps({"mass_kg": mass["deskbin_mass_kg"], "mg": cap_cmd["mg"], "mu_eff": cap_cmd["mu_eff_average"], "F_fric": cap_cmd["F_friction_max_average"]}), flush=True)

        print(json.dumps({"status": "TEST1 normal mu F=0.25"}), flush=True)
        normal = run_fork(task, captured["stock"], helper, captured["snapshot"], MU_NORMAL, "normal_mu0.425", HERE / "runs")
        print(json.dumps({"status": "TEST3 near-zero mu F=0.25"}), flush=True)
        zero = run_fork(task, captured["stock"], helper, captured["snapshot"], MU_NEAR_ZERO, "nearzero_mu0.001", HERE / "runs")

        n_med = (normal.get("high_load") or {}).get("N_min_median") or N_cmd
        cap_actual = capacity_block(mass["deskbin_mass_kg"], g, mu_finger, MU_NORMAL, float(n_med))
        cap_zero = capacity_block(mass["deskbin_mass_kg"], g, mu_finger, MU_NEAR_ZERO, float((zero.get("high_load") or {}).get("N_min_median") or N_cmd))

        # geometry from normal keyframes
        geom = {
            "hulls_overlap": overlap["any_overlap"],
            "overlap_pairs": overlap["pairs"],
            "hull_thickness": {k: v.get("size") for k, v in hulls["shapes"].items()},
            "any_multi_hull": False,
            "keyframes": {},
            "normals_not_simple_pinch": False,
        }
        for name, path in [
            ("post_query", HERE / "runs/normal_mu0.425/keyframes/post_query.json"),
            ("realize_end", HERE / "runs/normal_mu0.425/keyframes/realize_end.json"),
            ("terminal", HERE / "runs/normal_mu0.425/keyframes/terminal.json"),
        ]:
            if not path.exists():
                continue
            kf = json.loads(path.read_text())
            contacts = kf.get("contacts") or {}
            geom["keyframes"][name] = {
                "per_finger": contacts.get("per_finger"),
                "any_multi_hull_finger": contacts.get("any_multi_hull_finger"),
                "normal_summary": contacts.get("normal_summary"),
                "n_points": contacts.get("n_points"),
            }
            geom["any_multi_hull"] = geom["any_multi_hull"] or bool(contacts.get("any_multi_hull_finger"))
            ns = contacts.get("normal_summary") or {}
            if ns.get("min_dot") is not None and ns["min_dot"] > -0.85:
                geom["normals_not_simple_pinch"] = True

        decision = classify(normal, zero, mass, cap_cmd, cap_actual, geom)
        payload = {
            "seed": SEED,
            "episode_id": captured["episode_id"],
            "deskbin_id": captured["deskbin_id"],
            "pre_gate": captured["pre_gate"],
            "query_contact_ratio": captured["query"].get("contact_ratio"),
            "F_cmd_single_finger_N": F_CMD,
            "force_ref_bilateral_N": 2.0 * F_CMD,
            "convention": {
                "F_cmd": "single-finger force reference",
                "inner_force_ref": "2*F_cmd bilateral",
                "s": "2*min(N_L, N_R) from grasp-surface contacts",
                "N": "min(N_L, N_R); F_friction,max ≈ 2*μ_eff*N",
                "deadzone_note": "OriginalSqueezeInner squeeze_deadzone=0.25 N on position delta; |ΔF| needs ≳0.5 N bilateral to move aperture",
            },
            "TEST1_normal": normal,
            "TEST3_near_zero": zero,
            "TEST2_capacity": {
                "commanded_N_0.25": cap_cmd,
                "commanded_with_balls": cap_cmd_with_balls,
                "actual_high_load_N": cap_actual,
                "near_zero_mu": cap_zero,
            },
            "TEST4_geometry": geom,
            "TEST5_unload": {
                "query_end": normal.get("query_end"),
                "checkpoints": normal.get("checkpoints"),
                "unload_to_s_le_0.75": normal.get("unload_to_s_le_0.75"),
                "unload_to_s_le_0.50": normal.get("unload_to_s_le_0.50"),
                "unload_to_s_le_0.375": normal.get("unload_to_s_le_0.375"),
            },
            "classification": decision,
            "not_paper_data": True,
        }
        json_dump(HERE / "DIAG_RESULT.json", payload)
        print(json.dumps({"classification": decision, "normal_ret": normal.get("labels"), "zero_ret": zero.get("labels")}, indent=2, default=str), flush=True)
    finally:
        try:
            task.close_env()
        except Exception:
            pass


if __name__ == "__main__":
    main()
