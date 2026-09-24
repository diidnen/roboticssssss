"""No-query vs current 4 N shear query. Stop at handoff. No F/μ sweep."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ISO = Path("/media/volume/dasdas/exouser/af_dump_grasp_surface_friction_only_20260915")
V4 = Path("/media/volume/dasdas/exouser/af_dump_liftstyle_feas_v4_relabel_20260915")
REPO = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/RoboTwin")
sys.path[:0] = [str(ISO), str(V4), str(REPO), str(ROOT), str(HERE)]

from patch_grasp_surface import apply_grasp_surface_materials
from run_liftstyle_context import PrefixSnapshot, scripted_establish_grasp, write
from run_sweep_context import make_task
from collision_filter import apply_retention_collision_filters
from contact_sampler import sample_contacts, _finger_state, _relative_pose
from grasp_geom_logger import GraspGeomLogger, _ee_pose
from slab_geometry import quat_geodesic_deg, quat_to_rpy

SEED = 200014
MU = 0.425
QUERY_F = 4.0
QUERY_DIS = 0.012
EPISODES = list(range(10))
OUT = HERE / "query_probe"


def kinds(task, shape_map, dt) -> dict:
    grouped: dict[str, float] = {}
    for pair in sample_contacts(task, shape_map, dt):
        grouped[pair["kind"]] = grouped.get(pair["kind"], 0.0) + float(pair["fn"])
    return grouped


def pack_state(task, helper) -> dict:
    fingers, table_fn, table_present = helper._finger_grasp(task)
    left = fingers[helper.left_name]
    right = fingers[helper.right_name]
    gripper = _finger_state(task)
    pose = task.deskbin.get_pose()
    ee = _ee_pose(task)
    rel_p, rel_ang, rel_q = _relative_pose(pose, ee)
    roll, pitch, yaw = quat_to_rpy(rel_q)
    pair = None
    if left.get("p_world") and right.get("p_world"):
        pair = float(np.linalg.norm(np.asarray(left["p_world"]) - np.asarray(right["p_world"])))
    grouped = kinds(task, helper.shape_map, helper.dt)
    nl, nr = float(left["fn"]), float(right["fn"])
    nx_nx = bool(left.get("slab") == "grasp_nx" and right.get("slab") == "grasp_nx")
    bilateral = bool(left.get("present") and right.get("present"))
    unilateral = bool(left.get("present") ^ right.get("present"))
    both_gone = (not left.get("present")) and (not right.get("present"))
    limits = [float(x) for x in (gripper.get("force_limit") or [])]
    return {
        "left_slab": left.get("slab"),
        "right_slab": right.get("slab"),
        "left_present": bool(left.get("present")),
        "right_present": bool(right.get("present")),
        "left_p_obj": left.get("p_obj"),
        "right_p_obj": right.get("p_obj"),
        "left_n_obj": left.get("n_obj"),
        "right_n_obj": right.get("n_obj"),
        "nl": nl,
        "nr": nr,
        "squeeze": 2.0 * min(nl, nr),
        "pair_dist": pair,
        "nx_nx": nx_nx,
        "bilateral": bilateral,
        "unilateral": unilateral,
        "both_gone": both_gone,
        "alive": bool(nx_nx and bilateral and min(nl, nr) > 0.05),
        "loss": ("none" if nx_nx and bilateral else "unilateral" if unilateral else "bilateral" if both_gone else "slab_change"),
        "aperture": float(gripper["aperture_m"]),
        "force_limit": None if not limits else float(np.min(limits)),
        "drive_target": [float(x) for x in gripper["drive_target"]],
        "rel_p": [float(x) for x in rel_p],
        "rel_ang_deg": float(np.degrees(rel_ang)),
        "rel_q": [float(x) for x in rel_q],
        "rel_rpy_deg": [float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))],
        "obj_p": [float(x) for x in pose.p],
        "obj_z": float(pose.p[2]),
        "finger_inner_n": float(sum(v for k, v in grouped.items() if k.startswith("finger-inner"))),
        "palm_n": float(sum(v for k, v in grouped.items() if k.startswith("robot_nonfinger"))),
        "table_bottom_n": float(table_fn),
        "kinds": {str(k): float(v) for k, v in grouped.items()},
    }


def annotate_from_pre(pre: dict, post: dict) -> dict:
    post = dict(post)
    post["drel_from_pre_deg"] = quat_geodesic_deg(pre["rel_q"], post["rel_q"])
    post["aperture_delta"] = float(post["aperture"] - pre["aperture"])
    post["squeeze_delta"] = float(post["squeeze"] - pre["squeeze"])
    post["obj_disp_m"] = float(np.linalg.norm(np.asarray(post["obj_p"]) - np.asarray(pre["obj_p"])))
    post["rel_p_delta_m"] = float(np.linalg.norm(np.asarray(post["rel_p"]) - np.asarray(pre["rel_p"])))
    for side in ("left", "right"):
        a = pre.get(f"{side}_p_obj")
        b = post.get(f"{side}_p_obj")
        post[f"{side}_travel_m"] = None if not a or not b else float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    return post


def mean_bilateral(task, helper, n: int) -> float:
    hits = 0
    for _ in range(max(int(n), 1)):
        task.scene.step()
        fingers, _, _ = helper._finger_grasp(task)
        left = fingers[helper.left_name]
        right = fingers[helper.right_name]
        hits += int(bool(left.get("present") and right.get("present")))
    return float(hits / max(int(n), 1))


class StepCounter:
    def __init__(self, task):
        self.n = 0
        self._orig = task.scene.step

        def wrapped():
            self.n += 1
            return self._orig()

        task.scene.step = wrapped
        self.task = task

    def detach(self) -> None:
        self.task.scene.step = self._orig


def establish(episode_id: int):
    last = None
    for _ in range(2):
        task = make_task(SEED, MU, int(episode_id))
        apply_retention_collision_filters(task)
        apply_grasp_surface_materials(task, MU)
        task.activate_activeforcing_candidate_force()
        try:
            scripted_establish_grasp(task)
        except Exception as exc:
            last = exc
            try:
                task.close_env()
            except Exception:
                pass
            continue
        apply_retention_collision_filters(task)
        apply_grasp_surface_materials(task, MU)
        return task
    raise last or RuntimeError(f"grasp failed episode {episode_id}")


def run_episode(episode_id: int) -> dict:
    out = OUT / f"ep{episode_id:02d}"
    out.mkdir(parents=True, exist_ok=True)
    row = {"episode_id": int(episode_id), "ok": False}
    task = None
    try:
        task = establish(episode_id)
        row["used_episode_id"] = int(episode_id)
        helper = GraspGeomLogger(task, out, 0.0)
        contact_ratio_pre = mean_bilateral(task, helper, 20)
        pre = pack_state(task, helper)
        helper.save_frame("pre")
        snapshot = PrefixSnapshot(task)
        query = None
        query_error = None
        n_query = 0
        for attempt in range(3):
            counter = StepCounter(task)
            try:
                query = task.run_activeforcing_query(query_force_n=QUERY_F, displacement_m=QUERY_DIS)
                n_query = int(counter.n)
                break
            except Exception as exc:
                query_error = repr(exc)
                if "planning failed" not in query_error or attempt == 2:
                    break
                snapshot.restore()
                apply_retention_collision_filters(task)
                apply_grasp_surface_materials(task, MU)
            finally:
                counter.detach()
        post_query = annotate_from_pre(pre, pack_state(task, helper))
        helper.save_frame("post_query")
        snapshot.restore()
        apply_retention_collision_filters(task)
        apply_grasp_surface_materials(task, MU)
        n_hold = max(n_query, 25)
        hits = 0
        for _ in range(n_hold):
            task.scene.step()
            fingers, _, _ = helper._finger_grasp(task)
            left = fingers[helper.left_name]
            right = fingers[helper.right_name]
            hits += int(bool(left.get("present") and right.get("present")))
        contact_ratio_hold = float(hits / max(n_hold, 1))
        post_hold = annotate_from_pre(pre, pack_state(task, helper))
        helper.save_frame("post_hold")
        qslim = None
        if query is not None:
            qslim = {
                "contact_ratio": query.get("contact_ratio"),
                "final_bilateral_contact": query.get("final_bilateral_contact"),
                "measured_force_mean_n": query.get("measured_force_mean_n"),
                "relative_slip_path_m": query.get("relative_slip_path_m"),
                "relative_offset_max_m": query.get("relative_offset_max_m"),
                "actor_return_error_m": query.get("actor_return_error_m"),
                "ee_return_error_m": query.get("ee_return_error_m"),
                "actor_path_m": query.get("actor_path_m"),
                "samples": query.get("samples"),
            }
        row.update(
            {
                "ok": True,
                "n_query_steps": n_query,
                "n_hold_steps": n_hold,
                "contact_ratio_pre": contact_ratio_pre,
                "contact_ratio_hold": contact_ratio_hold,
                "contact_ratio_query": None if qslim is None else qslim.get("contact_ratio"),
                "pre": pre,
                "hold": post_hold,
                "query": post_query,
                "query_observables": qslim,
                "query_error": query_error,
            }
        )
    except Exception as exc:
        row["ok"] = False
        row["error"] = repr(exc)
        row["traceback"] = traceback.format_exc()
    finally:
        if task is not None:
            try:
                task.close_env()
            except Exception:
                pass
        try:
            write(out / "result.json", row)
        except Exception as exc:
            (out / "result.error.txt").write_text(repr(exc) + "\n" + traceback.format_exc())
    return row


def summarize(rows: list[dict]) -> dict:
    usable = [row for row in rows if row.get("ok") and row.get("pre")]
    n = len(usable)

    def rate(which: str, key: str) -> float | None:
        vals = [bool((row.get(which) or {}).get(key)) for row in usable]
        return None if not vals else 100.0 * float(np.mean(vals))

    def mean_key(which: str, key: str) -> float | None:
        vals = [(row.get(which) or {}).get(key) for row in usable]
        vals = [float(v) for v in vals if v is not None]
        return None if not vals else float(np.mean(vals))

    pre_alive = [bool(row["pre"].get("alive")) for row in usable]
    hold_kills = [
        bool(row["pre"].get("alive")) and not bool(row["hold"].get("alive")) for row in usable
    ]
    query_kills = [
        bool(row["pre"].get("alive")) and not bool(row["query"].get("alive")) for row in usable
    ]
    n_pre_alive = int(sum(pre_alive))
    verdict = "UNCLEAR"
    reason = "not enough usable episodes"
    paired_query_only_kills = int(
        sum(bool(row["hold"].get("alive")) and not bool(row["query"].get("alive")) for row in usable)
    )
    paired_hold_only_kills = int(
        sum(bool(row["query"].get("alive")) and not bool(row["hold"].get("alive")) for row in usable)
    )
    if n >= 5:
        hold_alive = rate("hold", "alive") or 0
        query_alive = rate("query", "alive") or 0
        pre_r = rate("pre", "alive") or 0
        if pre_r >= 70 and hold_alive >= 70 and (hold_alive - query_alive) >= 30:
            verdict = "QUERY_TOO_STRONG"
            reason = "No-query hold keeps the wall pinch; 4 N / 12 mm shear does not"
        elif pre_r < 70 or hold_alive < 70:
            verdict = "WALL_PINCH_UNSTABLE"
            reason = "Even without query the official wall pinch does not stay nx+nx"
        else:
            verdict = "QUERY_NOT_THE_MAIN_SPLIT"
            reason = "Query does not uniquely explain prefix spread versus a timed hold"
    return {
        "seed": SEED,
        "mu": MU,
        "query_force_n": QUERY_F,
        "query_displacement_m": QUERY_DIS,
        "n_ok": n,
        "pre_alive_pct": rate("pre", "alive"),
        "hold_alive_pct": rate("hold", "alive"),
        "query_alive_pct": rate("query", "alive"),
        "pre_nxnx_pct": rate("pre", "nx_nx"),
        "hold_nxnx_pct": rate("hold", "nx_nx"),
        "query_nxnx_pct": rate("query", "nx_nx"),
        "n_pre_alive": n_pre_alive,
        "hold_kill_given_pre_alive": None if not n_pre_alive else 100.0 * float(np.mean(hold_kills)),
        "query_kill_given_pre_alive": None if not n_pre_alive else 100.0 * float(np.mean(query_kills)),
        "mean_drel_hold_deg": mean_key("hold", "drel_from_pre_deg"),
        "mean_drel_query_deg": mean_key("query", "drel_from_pre_deg"),
        "mean_aperture_delta_hold": mean_key("hold", "aperture_delta"),
        "mean_aperture_delta_query": mean_key("query", "aperture_delta"),
        "mean_squeeze_delta_hold": mean_key("hold", "squeeze_delta"),
        "mean_squeeze_delta_query": mean_key("query", "squeeze_delta"),
        "mean_obj_disp_hold_m": mean_key("hold", "obj_disp_m"),
        "mean_obj_disp_query_m": mean_key("query", "obj_disp_m"),
        "paired_query_only_kills": paired_query_only_kills,
        "paired_hold_only_kills": paired_hold_only_kills,
        "verdict": verdict,
        "reason": reason,
    }


def plot_summary(rows: list[dict], summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = [row for row in rows if row.get("ok") and row.get("pre")]
    if not usable:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), dpi=130)
    labels = [str(row.get("used_episode_id", row["episode_id"])) for row in usable]
    x = np.arange(len(usable))
    width = 0.25
    for i, (which, color) in enumerate((("pre", "C2"), ("hold", "C0"), ("query", "C3"))):
        axes[0, 0].bar(x + (i - 1) * width, [int(row[which]["alive"]) for row in usable], width, label=which, color=color)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].set_ylim(-0.05, 1.15)
    axes[0, 0].set_title("alive nx+nx after stage")
    axes[0, 0].set_xlabel("episode")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].bar(
        ["pre", "no-query hold", "4N query"],
        [summary.get("pre_alive_pct") or 0, summary.get("hold_alive_pct") or 0, summary.get("query_alive_pct") or 0],
        color=["C2", "C0", "C3"],
    )
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].set_ylabel("%")
    axes[0, 1].set_title("alive rate across episodes")
    axes[1, 0].plot(labels, [row["hold"].get("drel_from_pre_deg") or 0 for row in usable], marker="o", label="hold")
    axes[1, 0].plot(labels, [row["query"].get("drel_from_pre_deg") or 0 for row in usable], marker="s", label="query")
    axes[1, 0].set_ylabel("deg")
    axes[1, 0].set_title("relative rotation from pre")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(labels, [row["pre"]["squeeze"] for row in usable], marker="o", label="pre squeeze")
    axes[1, 1].plot(labels, [row["hold"]["squeeze"] for row in usable], marker="^", label="hold squeeze")
    axes[1, 1].plot(labels, [row["query"]["squeeze"] for row in usable], marker="s", label="query squeeze")
    axes[1, 1].set_ylabel("N")
    axes[1, 1].set_title("squeeze 2*min(NL,NR)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    fig.suptitle(f"No-query hold vs 4 N / 12 mm query  ·  {summary.get('verdict')}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ep in EPISODES:
        print(json.dumps({"starting_episode": ep}), flush=True)
        row = run_episode(ep)
        slim = {
            "episode": row.get("used_episode_id", ep),
            "ok": row.get("ok"),
            "pre_alive": (row.get("pre") or {}).get("alive"),
            "hold_alive": (row.get("hold") or {}).get("alive"),
            "query_alive": (row.get("query") or {}).get("alive"),
            "pre_squeeze": (row.get("pre") or {}).get("squeeze"),
            "hold_squeeze": (row.get("hold") or {}).get("squeeze"),
            "query_squeeze": (row.get("query") or {}).get("squeeze"),
            "qcr": row.get("contact_ratio_query"),
            "drel_q": (row.get("query") or {}).get("drel_from_pre_deg"),
            "error": row.get("error") or row.get("query_error"),
        }
        print(json.dumps(slim, default=str), flush=True)
        rows.append(row)
        write(OUT / "INDEX.json", {"rows": rows})
    summary = summarize(rows)
    write(OUT / "SUMMARY.json", summary)
    plot_summary(rows, summary, OUT / "noquery_vs_4n.png")
    print(json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
