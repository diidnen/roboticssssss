"""Grasp-surface-only friction sanity. No 32x20. No query retune. No official overwrite.

Uses existing convex-panel split from 20260915 plus the wall-pinch collision
filter so fingers cannot touch inner/bottom (that leak existed in the 20260915
contact audit).
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ISO = Path("/media/volume/dasdas/exouser/af_dump_grasp_surface_friction_only_20260915")
V4 = Path("/media/volume/dasdas/exouser/af_dump_liftstyle_feas_v4_relabel_20260915")
ATTR = Path("/media/volume/dasdas/exouser/af_dump_force_realize_20260915/contact_attr")
ROOT = Path("/media/volume/dasdas/exouser/af_dump_force_realize_20260915")
REPO = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/RoboTwin")
GATE = json.loads(Path("/media/volume/dasdas/exouser/af_dump_pregrasp_gate_audit_20260916/GATE_DEFINITION.json").read_text())
CUT = GATE["cutoffs"]
sys.path[:0] = [str(ISO), str(V4), str(REPO), str(ROOT), str(ATTR), str(HERE)]

from envs.utils import ArmTag
from patch_grasp_surface import (
    FIXED_MU,
    apply_grasp_surface_materials,
    contact_material_audit,
    deskbin_shapes,
)
from run_liftstyle_context import PrefixSnapshot, scripted_establish_grasp, write
from run_sweep_context import labels_from_metrics, make_task, mu_eff
from force_realize import capture_stock_drives, realize_commanded_force, restore_stock_drives, uninstall_lift_squeeze
from collision_filter import apply_retention_collision_filters
from contact_sampler import sample_contacts
from grasp_geom_logger import GraspGeomLogger, _ee_pose, _finger_state
from slab_geometry import quat_to_rpy
from contact_sampler import _relative_pose

QUERY_F = 4.0
QUERY_DIS = 0.012
OFFICIAL_MU = [0.425, 0.575, 0.85]
OFFICIAL_FORCES = [round(0.25 * i, 2) for i in range(1, 21)]
COMBINE_NOTE = (
    "SAPIEN 3 PhysxMaterial has no combine-mode API. PhysX 5 default is eAVERAGE. "
    "μ_eff_average = 0.5*(μ_finger+μ_grasp). Also report min and multiply."
)


def remainder_phased(task) -> None:
    place = ArmTag("left")
    task.move(task.move_by_displacement(arm_tag=place, z=0.08, move_axis="arm"))
    task.move(task.pour_actions)
    task.move(task.pour_actions)
    task.move(task.pour_actions)
    task.delay(6)


def pack_state(task, helper) -> dict:
    fingers, table_fn, _ = helper._finger_grasp(task)
    left = fingers[helper.left_name]
    right = fingers[helper.right_name]
    gripper = _finger_state(task)
    pose = task.deskbin.get_pose()
    ee = _ee_pose(task)
    rel_p, rel_ang, rel_q = _relative_pose(pose, ee)
    pair = None
    if left.get("p_world") and right.get("p_world"):
        pair = float(np.linalg.norm(np.asarray(left["p_world"]) - np.asarray(right["p_world"])))
    nl, nr = float(left["fn"]), float(right["fn"])
    nx_nx = bool(left.get("slab") == "grasp_nx" and right.get("slab") == "grasp_nx")
    bilateral = bool(left.get("present") and right.get("present"))
    return {
        "left_slab": left.get("slab"),
        "right_slab": right.get("slab"),
        "left_present": bool(left.get("present")),
        "right_present": bool(right.get("present")),
        "nl": nl,
        "nr": nr,
        "squeeze": 2.0 * min(nl, nr),
        "pair_dist": pair,
        "nx_nx": nx_nx,
        "bilateral": bilateral,
        "aperture": float(gripper["aperture_m"]),
        "obj_z": float(pose.p[2]),
        "table_bottom_n": float(table_fn),
        "rel_yaw_deg": float(np.degrees(quat_to_rpy(rel_q)[2])),
        "rel_ang_deg": float(rel_ang),
    }


def classify_pre(window: list[dict]) -> dict:
    if not window:
        return {"gate": "INVALID", "reason": "empty PRE window", "window_cr": 0.0}
    nx = [bool(row["nx_nx"] and row["bilateral"]) for row in window]
    cr = float(np.mean(nx))
    end = window[-1]
    ap = float(end["aperture"] or 0.0)
    sq = float(end["squeeze"] or 0.0)
    endpoint_nx = bool(end["nx_nx"] and end["bilateral"])
    if ap >= CUT["aperture_miss_m"] or (cr <= 1e-9 and sq <= CUT["squeeze_alive_n"] and not endpoint_nx):
        return {
            "gate": "INVALID",
            "reason": f"open/miss ap={ap*1000:.1f} mm squeeze={sq:.3f} CR={cr:.2f}",
            "window_cr": cr,
        }
    if cr >= CUT["window_bilateral_valid_min"] and endpoint_nx and sq > CUT["squeeze_alive_n"] and ap < CUT["aperture_miss_m"]:
        return {"gate": "VALID", "reason": "PRE bilateral nx+nx wall pinch", "window_cr": cr}
    if cr >= CUT["window_bilateral_borderline_min"] and ap < CUT["aperture_miss_m"] and (not endpoint_nx or sq <= CUT["squeeze_alive_n"]):
        return {"gate": "BORDERLINE", "reason": "high-CR PRE with endpoint flicker", "window_cr": cr}
    return {
        "gate": "INVALID",
        "reason": f"weak PRE CR={cr:.2f} nx={endpoint_nx} squeeze={sq:.3f}",
        "window_cr": cr,
    }


def material_snapshot(shape) -> dict:
    mat = shape.get_physical_material()
    return {
        "static_friction": float(mat.static_friction),
        "dynamic_friction": float(mat.dynamic_friction),
        "restitution": float(mat.restitution),
    }


def actor_materials(entity, name: str) -> list[dict]:
    rows = []
    components = []
    if hasattr(entity, "get_components"):
        components.extend(entity.get_components())
    if hasattr(entity, "get_links"):
        components.extend(entity.get_links())
    for component in components:
        getter = getattr(component, "get_collision_shapes", None)
        if not callable(getter):
            continue
        link = getattr(component, "get_name", lambda: name)()
        for shape in getter():
            row = material_snapshot(shape)
            row["link"] = str(link)
            row["actor"] = name
            rows.append(row)
    return rows


def physx_combine_probe() -> dict:
    import sapien.physx as physx

    mat = physx.get_default_material()
    cfg = physx.get_scene_config()
    return {
        "note": COMBINE_NOTE,
        "default_material": {
            "static_friction": float(mat.static_friction),
            "dynamic_friction": float(mat.dynamic_friction),
            "restitution": float(mat.restitution),
            "dir": [x for x in dir(mat) if "combin" in x.lower() or "friction" in x.lower()],
        },
        "scene_config_dir": [x for x in dir(cfg) if "combin" in x.lower() or "friction" in x.lower()],
        "assumed_mode": "PhysX5_eAVERAGE",
    }


def audit_original_and_split(seed: int, friction: float, episode_id: int) -> dict:
    combine = physx_combine_probe()
    # Patched env (this experiment).
    task = make_task(seed, friction, episode_id)
    try:
        assignment = apply_grasp_surface_materials(task, friction)
        filt = apply_retention_collision_filters(task)
        entity, shapes, roles, names = deskbin_shapes(task.deskbin)
        deskbin = [
            {"name": name, "role": role, **material_snapshot(shape)}
            for shape, role, name in zip(shapes, roles, names)
        ]
        fingers = []
        for link in task.robot.left_entity.get_links():
            if str(link.get_name()) not in {"fl_link7", "fl_link8"}:
                continue
            getter = getattr(link, "get_collision_shapes", None)
            if not callable(getter):
                continue
            for shape in getter():
                row = material_snapshot(shape)
                row["link"] = str(link.get_name())
                fingers.append(row)
        table_rows = []
        for actor in task.scene.get_all_actors() if hasattr(task.scene, "get_all_actors") else []:
            pass
        # table is usually named table
        for entity_name, ent in (("table", getattr(task, "table", None)),):
            if ent is None:
                continue
            raw = ent.actor if hasattr(ent, "actor") else ent
            table_rows.extend(actor_materials(raw, entity_name))
        balls = []
        for sphere in task.sphere_lst:
            balls.extend(actor_materials(sphere, "garbage"))
        finger_mu = float(np.mean([r["static_friction"] for r in fingers])) if fingers else None
        table_mu = float(np.mean([r["static_friction"] for r in table_rows])) if table_rows else None
        ball_mu = float(np.mean([r["static_friction"] for r in balls])) if balls else None
        payload = {
            "combine": combine,
            "deskbin_id": int(getattr(task, "deskbin_id", -1)),
            "n_deskbin_shapes": len(shapes),
            "deskbin_shapes": deskbin,
            "assignment": assignment,
            "collision_filter": filt,
            "finger_shapes": fingers,
            "table_shapes": table_rows,
            "ball_shapes": balls[:8],
            "n_ball_shapes": len(balls),
            "finger_mu_measured": finger_mu,
            "table_mu_measured": table_mu,
            "ball_mu_measured": ball_mu,
            "scene_default_mu": 0.5,
            "fixed_inner_bottom_mu": FIXED_MU,
            "object_side_grasp_mu": float(friction),
            "mu_eff": None if finger_mu is None else {
                "average": 0.5 * (finger_mu + float(friction)),
                "min": min(finger_mu, float(friction)),
                "multiply": finger_mu * float(friction),
            },
            "original_asset": {
                "model": "063_tabletrashbin",
                "collision": "assets/objects/063_tabletrashbin/collision/base{id}.glb loaded with add_multiple_convex_collisions_from_file; all hulls then painted by set_actor_contact_friction(af_contact_friction)",
                "visual_unchanged": "assets/objects/063_tabletrashbin/visual/base{id}.glb",
                "after": "runtime create_actor patch: 4 convex grasp slabs (px/nx/pz/nz) variable μ; inner nonconvex fixed 0.3; bottom nonconvex fixed 0.3",
            },
        }
        return payload
    finally:
        try:
            task.close_env()
        except Exception:
            pass


def capture_valid_prefix(seed: int, friction: float, episode: int | None = None):
    last_error = None
    episodes = [int(episode)] if episode is not None else list(range(10))
    for episode_id in episodes:
        task = None
        try:
            task = make_task(seed, friction, episode_id)
            stock = capture_stock_drives(task)
            assignment = apply_grasp_surface_materials(task, friction)
            filt = apply_retention_collision_filters(task)
            task.activate_activeforcing_candidate_force()
            scripted_establish_grasp(task)
            helper = GraspGeomLogger(task, HERE / "_tmp_geom", 0.0)
            window = []
            for _ in range(20):
                task.scene.step()
                window.append(pack_state(task, helper))
            gate = classify_pre(window)
            print(json.dumps({"seed": seed, "mu": friction, "episode_id": episode_id, "gate": gate["gate"], "cr": gate["window_cr"]}), flush=True)
            if gate["gate"] != "VALID":
                try:
                    task.close_env()
                except Exception:
                    pass
                last_error = RuntimeError(gate["reason"])
                continue
            contact_audit = contact_material_audit(task)
            pre_query = PrefixSnapshot(task)
            query = None
            query_error = None
            for _ in range(3):
                try:
                    query = task.run_activeforcing_query(query_force_n=QUERY_F, displacement_m=QUERY_DIS)
                    query_error = None
                    break
                except RuntimeError as exc:
                    query_error = repr(exc)
                    if "planning failed" not in query_error:
                        raise
                    pre_query.restore()
                    apply_grasp_surface_materials(task, friction)
                    apply_retention_collision_filters(task)
            if query is None:
                query = {"contact_ratio": 0.0, "query_failed": True, "error": query_error}
            snapshot = PrefixSnapshot(task)
            post_query = pack_state(task, helper)
            return {
                "task": task,
                "stock": stock,
                "snapshot": snapshot,
                "helper": helper,
                "episode_id": episode_id,
                "deskbin_id": int(getattr(task, "deskbin_id", -1)),
                "assignment": assignment,
                "filter": filt,
                "pre_gate": gate,
                "pre_end": window[-1],
                "pre_window_cr": gate["window_cr"],
                "contact_audit_pre": contact_audit,
                "query": query,
                "post_query": post_query,
            }
        except Exception as exc:
            last_error = exc
            print(json.dumps({"seed": seed, "mu": friction, "episode_id": episode_id, "error": repr(exc)}), flush=True)
            if task is not None:
                try:
                    task.close_env()
                except Exception:
                    pass
            if type(exc).__name__ == "UnStableError" or "UnStableError" in repr(exc):
                break
    raise last_error or RuntimeError("no PRE_VALID episode")


def run_one(task, force: float, friction: float, stock, helper) -> dict:
    apply_grasp_surface_materials(task, friction)
    apply_retention_collision_filters(task)
    task.plan_success = True
    task._af_force_trace = []
    task._af_force_trace_step = 0
    task._af_no_contact_samples = 0
    task._af_ever_lifted = False
    realize = realize_commanded_force(task, force, stock)
    task._af_force_trace = []
    task._af_force_trace_step = 0
    task._af_no_contact_samples = 0
    task._af_ever_lifted = False
    task.activate_activeforcing_candidate_force()
    restore_stock_drives(stock)
    remainder_phased(task)
    uninstall_lift_squeeze(task)
    metrics = task.compute_activeforcing_dynamic_metrics()
    labels = labels_from_metrics(task, metrics)
    end = pack_state(task, helper)
    n_balls, ball_z = 0, []
    for sphere in task.sphere_lst:
        z = float(sphere.get_pose().p[2])
        ball_z.append(z)
        if 0.13 <= z <= 0.25:
            n_balls += 1
    audit_end = contact_material_audit(task)
    garbage_mu = []
    for row in audit_end.get("details", {}).get("garbage", []):
        for shape in row.get("shapes") or []:
            garbage_mu.append({"role": shape.get("role"), "mu": shape.get("static_friction")})
    finger_roles = audit_end.get("finger_grasp_roles", {})
    return {
        "force_N": float(force),
        "retention_success": int(labels["retention_success"]),
        "official_full_task_success": int(labels["official_full_task_success"]),
        "contact_ratio": labels.get("contact_ratio"),
        "drop": labels.get("drop"),
        "irrecoverable_failure": labels.get("irrecoverable_failure"),
        "measured_force_mean_n": labels.get("measured_force_mean_n"),
        "settle_squeeze_mean_n": realize.get("pre_motion_squeeze_mean_n"),
        "force_realized": bool(realize.get("force_realized_before_motion")),
        "n_balls_in_official_band": int(n_balls),
        "ball_z": ball_z,
        "deskbin_z": float(task.deskbin.get_pose().p[2]),
        "pair_dist_end": end.get("pair_dist"),
        "end_nxnx": bool(end.get("nx_nx")),
        "end_aperture_m": end.get("aperture"),
        "end_squeeze": end.get("squeeze"),
        "realize": realize,
        "finger_contact_roles": finger_roles,
        "garbage_contact_materials": garbage_mu,
        "inner_still_fixed": all(abs(float(x["mu"]) - FIXED_MU) < 1e-6 for x in garbage_mu if x.get("role") == "inner") if garbage_mu else None,
    }


def fmin_first(forces, bits) -> float | None:
    for force, ok in zip(forces, bits):
        if int(ok):
            return float(force)
    return None


def fmin_stable(forces, bits) -> float | None:
    seen = 0
    for val in bits:
        if int(val) < seen:
            return None
        seen = max(seen, int(val))
    return fmin_first(forces, bits)


def run_context(seed: int, friction: float, forces: list[float], out: Path, episode: int | None = None) -> dict:
    captured = capture_valid_prefix(seed, friction, episode=episode)
    task = captured["task"]
    outcomes = []
    try:
        for force in forces:
            captured["snapshot"].restore()
            apply_grasp_surface_materials(task, friction)
            apply_retention_collision_filters(task)
            row = run_one(task, force, friction, captured["stock"], captured["helper"])
            row["query_contact_ratio"] = captured["query"].get("contact_ratio")
            print(json.dumps({"seed": seed, "mu": friction, "F": force, "ret": row["retention_success"], "task": row["official_full_task_success"], "realized": row["force_realized"], "balls": row["n_balls_in_official_band"]}), flush=True)
            outcomes.append(row)
    finally:
        try:
            task.close_env()
        except Exception:
            pass
    finger_mu = None
    payload = {
        "seed": int(seed),
        "friction_object_side_grasp": float(friction),
        "friction_inner_bottom_fixed": FIXED_MU,
        "mu_eff_average_if_finger_0.3": mu_eff(friction),
        "episode_id": captured["episode_id"],
        "deskbin_id": captured["deskbin_id"],
        "pre_gate": captured["pre_gate"],
        "pre_end": captured["pre_end"],
        "query_contact_ratio": captured["query"].get("contact_ratio"),
        "query_failed": bool(captured["query"].get("query_failed")),
        "post_query": captured["post_query"],
        "contact_audit_pre": captured["contact_audit_pre"],
        "assignment": captured["assignment"],
        "collision_filter": captured["filter"],
        "outcomes": outcomes,
        "F_min_retention_first": fmin_first(forces, [o["retention_success"] for o in outcomes]),
        "F_min_retention_stable": fmin_stable(forces, [o["retention_success"] for o in outcomes]),
        "F_min_task_first": fmin_first(forces, [o["official_full_task_success"] for o in outcomes]),
        "F_min_task_stable": fmin_stable(forces, [o["official_full_task_success"] for o in outcomes]),
        "n_retention": int(sum(o["retention_success"] for o in outcomes)),
        "n_task": int(sum(o["official_full_task_success"] for o in outcomes)),
        "inner_fixed_on_garbage_contacts": all(
            o.get("inner_still_fixed") in (True, None) for o in outcomes
        ),
    }
    write(out / f"seed{seed}_mu{friction:g}_CONTEXT.json", payload)
    return payload


def summarize(rows: list[dict]) -> dict:
    by_seed: dict = {}
    for row in rows:
        by_seed.setdefault(str(row["seed"]), {})[f"{row['friction_object_side_grasp']:g}"] = {
            "episode_id": row["episode_id"],
            "pre_gate": row["pre_gate"]["gate"],
            "qCR": row["query_contact_ratio"],
            "retention": [int(o["retention_success"]) for o in row["outcomes"]],
            "full_task": [int(o["official_full_task_success"]) for o in row["outcomes"]],
            "n_balls": [int(o["n_balls_in_official_band"]) for o in row["outcomes"]],
            "realized": [int(o["force_realized"]) for o in row["outcomes"]],
            "F_min_retention_first": row["F_min_retention_first"],
            "F_min_retention_stable": row["F_min_retention_stable"],
            "F_min_task_first": row["F_min_task_first"],
            "mean_balls": float(np.mean([o["n_balls_in_official_band"] for o in row["outcomes"]])),
        }
    s014 = by_seed.get("200014", {})
    fmins = []
    for mu in OFFICIAL_MU:
        item = s014.get(f"{mu:g}")
        if item:
            fmins.append(item["F_min_retention_first"])
    coulomb = None
    if len(fmins) == 3 and all(x is not None for x in fmins):
        coulomb = bool(fmins[0] > fmins[1] > fmins[2]) or bool(fmins[0] >= fmins[1] >= fmins[2] and fmins[0] > fmins[2])
    return {"by_seed": by_seed, "coulomb_order_first_crossing_200014": coulomb, "F_min_retention_first_200014": fmins}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--seeds", default="200014")
    parser.add_argument("--mus", default="0.425,0.575,0.85")
    parser.add_argument("--episode", type=int, default=None, help="Pin episode_id; do not scan.")
    parser.add_argument("--tag", default="", help="Optional filename tag.")
    args = parser.parse_args()
    mus = [float(x) for x in args.mus.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    HERE.mkdir(parents=True, exist_ok=True)
    audit = audit_original_and_split(200014, mus[0], 0)
    write(HERE / "ASSET_AUDIT.json", audit)
    print(json.dumps({"audit_finger_mu": audit.get("finger_mu_measured"), "audit_n_shapes": audit.get("n_deskbin_shapes"), "mu_eff": audit.get("mu_eff")}, indent=2), flush=True)
    if args.audit_only:
        return
    rows = []
    for seed in seeds:
        for mu in mus:
            out_id = f"seed{seed}_mu{mu:g}{args.tag}"
            try:
                payload = run_context(seed, mu, OFFICIAL_FORCES, HERE / "sweeps", episode=args.episode)
                rows.append(payload)
                write(HERE / "sweeps" / f"{out_id}_OK.json", {"ok": True})
            except Exception as exc:
                write(
                    HERE / "sweeps" / f"{out_id}_FAIL.json",
                    {"ok": False, "error": repr(exc), "trace": traceback.format_exc()[-4000:]},
                )
                print(json.dumps({"seed": seed, "mu": mu, "failed": repr(exc)}), flush=True)
                if seed == 200003 and ("UnStableError" in repr(exc) or type(exc).__name__ == "UnStableError"):
                    break
    summary = summarize(rows)
    write(HERE / f"SWEEP_SUMMARY{args.tag or ''}.json", summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
