"""Diagnose where commanded squeeze collapses after snapshot restore.

Does not overwrite official 18/19 or the previous isolation sweep.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ISO = Path("/media/volume/dasdas/exouser/af_dump_grasp_surface_friction_only_20260915")
V4 = Path("/media/volume/dasdas/exouser/af_dump_liftstyle_feas_v4_relabel_20260915")
REPO = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/RoboTwin")
sys.path[:0] = [str(HERE), str(ISO), str(V4), str(REPO)]

from patch_grasp_surface import apply_grasp_surface_materials
from run_liftstyle_context import PrefixSnapshot, scripted_establish_grasp
from run_sweep_context import QUERY_FORCE_N, make_task

FORCES = [1.5, 1.75, 4.25, 5.0]
MU = 0.575
SEED = 200014
SETTLE = 80


def squeeze(task) -> dict:
    contact = task.get_actor_gripper_contact_forces(task.deskbin, "left")
    return {
        "squeeze_n": float(contact["single_finger_normal_force_n"]),
        "bilateral": bool(contact["bilateral_contact"]),
        "per_finger": {k: float(v) for k, v in contact["per_finger_normal_force_n"].items()},
        "gripper_val": float(task.robot.get_left_gripper_val()),
        "force_limit": task.robot.get_gripper_force_limit("left"),
        "deskbin_z": float(task.deskbin.get_pose().p[2]),
    }


def settle(task, steps: int) -> list[dict]:
    rows = []
    for _ in range(steps):
        task.scene.step()
        rows.append(squeeze(task))
    return rows


def reclose(task) -> None:
    task.robot.set_gripper(0.0, "left", gripper_eps=0.0)


def set_force(task, force_n: float) -> None:
    task.robot.set_gripper_force_limit(float(force_n), "left")
    task.robot.set_gripper_force_limit(float(force_n), "right")
    task.af_force_limit_n = float(force_n)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    sq = np.asarray([row["squeeze_n"] for row in rows], dtype=float)
    return {
        "squeeze_last": float(sq[-1]),
        "squeeze_mean": float(np.mean(sq)),
        "squeeze_max": float(np.max(sq)),
        "bilateral_last": bool(rows[-1]["bilateral"]),
        "bilateral_frac": float(np.mean([row["bilateral"] for row in rows])),
        "z_last": rows[-1]["deskbin_z"],
        "limit": rows[-1]["force_limit"],
    }


def main() -> None:
    task = make_task(SEED, MU, 0)
    try:
        apply_grasp_surface_materials(task, MU)
        task.activate_activeforcing_candidate_force()
        scripted_establish_grasp(task)
        for _ in range(20):
            task.scene.step()
        query = task.run_activeforcing_query(query_force_n=QUERY_FORCE_N, displacement_m=0.012)
        prefix = squeeze(task)
        snapshot = PrefixSnapshot(task)
        report = {
            "seed": SEED,
            "mu": MU,
            "query_contact_ratio": query.get("contact_ratio"),
            "prefix_after_query": prefix,
            "branches": [],
        }
        for force in FORCES:
            snapshot.restore()
            immediately = squeeze(task)
            snapshot.restore()
            set_force(task, force)
            after_set = squeeze(task)
            settled = settle(task, SETTLE)

            snapshot.restore()
            apply_grasp_surface_materials(task, MU)
            set_force(task, force)
            settled_rebind = settle(task, SETTLE)

            snapshot.restore()
            set_force(task, force)
            reclose(task)
            settled_reclose = settle(task, SETTLE)

            snapshot.restore()
            set_force(task, force)
            reclose(task)
            for _ in range(SETTLE):
                task.scene.step()
            held = squeeze(task)

            report["branches"].append(
                {
                    "force_N": force,
                    "immediately_after_restore": immediately,
                    "after_set_force": after_set,
                    "settle_no_rebind": summarize(settled),
                    "settle_rebind_materials": summarize(settled_rebind),
                    "settle_reclose": summarize(settled_reclose),
                    "held_after_reclose_settle": held,
                }
            )
            print(
                force,
                "imm", round(immediately["squeeze_n"], 3), immediately["bilateral"],
                "set", round(after_set["squeeze_n"], 3),
                "settle", summarize(settled),
                "rebind", summarize(settled_rebind),
                "reclose", summarize(settled_reclose),
                flush=True,
            )
        path = HERE / "DIAGNOSE_FORCE_REALIZE.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print("wrote", path)
    finally:
        task.close_env()


if __name__ == "__main__":
    main()
