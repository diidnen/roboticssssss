#!/usr/bin/env python3
"""Same initial snapshot, three finger μ, stop at PRE. No query/dump/sweep."""
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
from run_liftstyle_context import PrefixSnapshot, scripted_establish_grasp

SEED = int(PLAN["seed"])
EPISODE = int(PLAN["episode_id"])
DESK_MU = float(PLAN["deskbin_mu_fixed"])
LEVELS = {
    "low": {"mu_finger": 0.425, "mu_eff": 0.3625},
    "mid": {"mu_finger": 0.575, "mu_eff": 0.4375},
    "high": {"mu_finger": 0.85, "mu_eff": 0.575},
}
THR = PLAN["divergence_thresholds_mid_vs_high"]
STRIDE = 8
DT = 0.004
SETTLE = 20


def dump(path: Path, value) -> None:
    S.dump_json(path, value)


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_mul(a, b):
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


def quat_rotate(q, v):
    return quat_mul(quat_mul(q, np.array([0.0, v[0], v[1], v[2]])), quat_conj(q))[1:]


def quat_angle_deg(q0, q1) -> float:
    d = abs(float(np.dot(np.asarray(q0, dtype=float), np.asarray(q1, dtype=float))))
    return float(2.0 * math.degrees(math.acos(min(1.0, d))))


def finger_worlds(task) -> dict:
    out = {}
    for joint, _, _ in task.robot.left_gripper:
        name = str(joint.child_link.get_name())
        pose = None
        for attr in ("get_entity_pose", "get_pose", "pose"):
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


def table_force(task) -> dict:
    dt = float(getattr(task, "physics_timestep", DT))
    fn = 0.0
    n_pts = 0
    for contact in task.scene.get_contacts():
        name0 = S.body_name(contact.bodies[0])
        name1 = S.body_name(contact.bodies[1])
        if S.DESKBIN not in (name0, name1):
            continue
        other = name1 if name0 == S.DESKBIN else name0
        if other != "table":
            continue
        for point in contact.points:
            impulse = np.asarray(point.impulse, dtype=float)
            normal = np.asarray(point.normal, dtype=float)
            nrm = float(np.linalg.norm(normal)) + 1e-12
            fn += abs(float(np.dot(impulse, normal / nrm))) / max(dt, 1e-9)
            n_pts += 1
    return {"table_contact": bool(fn > 1e-4), "table_fn": float(fn), "table_n_points": int(n_pts)}


def hulls_of(geom, side: str) -> list[int]:
    names = {"left": ("fl_link7", "fl_link8"), "right": ("fr_link7", "fr_link8")}[side]
    ids = []
    for name in names:
        row = (geom.get("per_finger") or {}).get(name) or {}
        if row.get("present"):
            ids.extend(row.get("shape_ids") or [])
    return sorted(set(int(x) for x in ids))


def finger_hull_map(geom) -> dict:
    out = {}
    for name, row in (geom.get("per_finger") or {}).items():
        if row.get("present"):
            out[name] = {
                "shape_ids": row.get("shape_ids"),
                "axes": row.get("axes"),
                "n_shapes": row.get("n_shapes"),
                "normals": row.get("normals"),
                "fn": row.get("fn"),
            }
    return out


def contact_mode(geom) -> str:
    if geom.get("bilateral"):
        return "bilateral"
    if geom.get("left_present") or geom.get("right_present"):
        return "unilateral"
    return "none"


def rich_state(task, ref=None) -> dict:
    geom = S.contact_geometry(task)
    tab = table_force(task)
    obj_xyz = np.asarray(geom["obj_xyz"], dtype=float)
    obj_q = np.asarray(geom["obj_q"], dtype=float)
    fingers = finger_worlds(task)
    centroid = np.mean(np.array(list(fingers.values()), dtype=float), axis=0) if fingers else obj_xyz
    left_hulls = hulls_of(geom, "left")
    right_hulls = hulls_of(geom, "right")
    state = {
        "obj_xyz": obj_xyz.tolist(),
        "obj_q": obj_q.tolist(),
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
        "finger_hulls": finger_hull_map(geom),
        "all_finger_hulls": sorted(set(left_hulls + right_hulls)),
        "n_multi_hull_fingers": int(
            sum(1 for row in (geom.get("per_finger") or {}).values() if row.get("present") and int(row.get("n_shapes") or 0) >= 2)
        ),
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
        "table_contact": tab["table_contact"],
        "table_fn": tab["table_fn"],
        "finger_mu_eff": geom.get("finger_deskbin_mu_eff"),
    }
    if ref is not None:
        dxyz = obj_xyz - np.asarray(ref["obj_xyz"], dtype=float)
        slip = obj_xyz - centroid
        slip0 = np.asarray(ref["obj_xyz"], dtype=float) - np.asarray(ref["finger_centroid"], dtype=float)
        state["delta_xyz"] = dxyz.tolist()
        state["delta_xyz_norm"] = float(np.linalg.norm(dxyz))
        state["delta_rot_deg"] = quat_angle_deg(ref["obj_q"], obj_q)
        state["rel_slip_xyz"] = (slip - slip0).tolist()
        state["rel_slip_y"] = float((slip - slip0)[1])
        state["rel_slip_norm"] = float(np.linalg.norm(slip - slip0))
    return state


def slim(state, step: int) -> dict:
    return {
        "step": step,
        "t_s": step * DT,
        "squeeze": state["squeeze"],
        "nl": state["nl"],
        "nr": state["nr"],
        "pair_dist": state["pair_dist"],
        "aperture": state["aperture"],
        "mode": state["mode"],
        "mode_code": {"none": 0, "unilateral": 1, "bilateral": 2}[state["mode"]],
        "left_hulls": state["left_hulls"],
        "finger_hull_key": "|".join(
            f"{k}:{','.join(str(x) for x in (v.get('shape_ids') or []))}" for k, v in sorted((state.get("finger_hulls") or {}).items())
        ),
        "delta_xyz": state.get("delta_xyz"),
        "delta_xyz_norm": state.get("delta_xyz_norm"),
        "delta_rot_deg": state.get("delta_rot_deg"),
        "rel_slip_y": state.get("rel_slip_y"),
        "rel_slip_norm": state.get("rel_slip_norm"),
        "table_contact": state["table_contact"],
        "table_fn": state["table_fn"],
        "opposing": state["opposing"],
        "bilateral": state["bilateral"],
    }


class Recorder:
    def __init__(self, task, ref):
        self.task = task
        self.ref = ref
        self.frames = []
        self.n = 0
        self.first_contact_step = None
        self.table_leave_step = None
        self._orig = task._after_physics_step

        def hooked():
            self._orig()
            self.n += 1
            if self.n % STRIDE:
                return
            state = rich_state(task, self.ref)
            frame = slim(state, self.n)
            self.frames.append(frame)
            if self.first_contact_step is None and frame["mode"] != "none":
                self.first_contact_step = self.n
            if self.table_leave_step is None and not frame["table_contact"]:
                self.table_leave_step = self.n

        task._after_physics_step = hooked

    def close(self):
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


def row_of(label, state) -> dict:
    return {
        "condition": label,
        "final_squeeze": state.get("squeeze"),
        "pair_distance": state.get("pair_dist"),
        "aperture": state.get("aperture"),
        "object_xyz": state.get("obj_xyz"),
        "object_rotation": state.get("obj_q"),
        "left_hulls": state.get("left_hulls") or (state.get("finger_hulls") or {}).get("fl_link7", {}).get("shape_ids"),
        "right_hulls": (state.get("finger_hulls") or {}).get("fl_link8", {}).get("shape_ids"),
        "finger_hulls": state.get("finger_hulls"),
        "contact_normals": {k: v.get("normals") for k, v in (state.get("finger_hulls") or {}).items()},
        "table_contact": state.get("table_contact"),
        "table_fn": state.get("table_fn"),
        "relative_slip": state.get("rel_slip_norm"),
        "relative_slip_y": state.get("rel_slip_y"),
        "delta_xyz": state.get("delta_xyz"),
        "delta_rot_deg": state.get("delta_rot_deg"),
        "PRE_contact_mode": state.get("mode"),
        "opposing": state.get("opposing"),
        "nl": state.get("nl"),
        "nr": state.get("nr"),
    }


def compare_mid_high(mid, high) -> dict:
    a, b = mid["pre"], high["pre"]
    pose_gap = float(np.linalg.norm(np.asarray(a["obj_xyz"]) - np.asarray(b["obj_xyz"])))
    rot_gap = quat_angle_deg(a["obj_q"], b["obj_q"])
    pair_gap = abs(float(a["pair_dist"] or 0) - float(b["pair_dist"] or 0))
    squeeze_gap = abs(float(a["squeeze"]) - float(b["squeeze"]))
    slip_gap = abs(float(a.get("rel_slip_norm") or 0) - float(b.get("rel_slip_norm") or 0))
    hulls_union = list(a["all_finger_hulls"]) != list(b["all_finger_hulls"])
    hulls_assign = json.dumps(a.get("finger_hulls"), sort_keys=True, default=str) != json.dumps(
        b.get("finger_hulls"), sort_keys=True, default=str
    )
    table_diff = bool(a["table_contact"]) != bool(b["table_contact"])
    flags = {
        "squeeze_abs": squeeze_gap >= float(THR["squeeze_abs_n"]),
        "obj_translation": pose_gap >= float(THR["obj_translation_m"]),
        "obj_rotation": rot_gap >= float(THR["obj_rotation_deg"]),
        "pair_dist": pair_gap >= float(THR["pair_dist_m"]),
        "rel_slip": slip_gap >= float(THR["rel_slip_m"]),
        "hulls": hulls_union or hulls_assign,
        "table_contact": table_diff,
    }
    explain = []
    if flags["hulls"]:
        explain.append("finger hull / contact-point assignment differs")
    if flags["obj_translation"] or flags["obj_rotation"]:
        explain.append("object pose differs")
    if flags["pair_dist"]:
        explain.append("pair distance / pinch opening differs")
    if flags["table_contact"]:
        explain.append("table contact differs")
    if flags["squeeze_abs"] and (flags["hulls"] or flags["obj_translation"] or flags["pair_dist"]):
        explain.append("squeeze differs together with geometry")
    if a.get("opposing") != b.get("opposing"):
        explain.append("opposing pinch status differs")
    if mid["gate"]["gate"] != high["gate"]["gate"]:
        explain.append("PRE gate differs")
    return {
        "flags": flags,
        "diverged": bool(any(flags.values())),
        "squeeze_mid": a["squeeze"],
        "squeeze_high": b["squeeze"],
        "squeeze_gap": squeeze_gap,
        "pose_gap_m": pose_gap,
        "rot_gap_deg": rot_gap,
        "pair_gap_m": pair_gap,
        "slip_gap_m": slip_gap,
        "hulls_union_differ": hulls_union,
        "hulls_assignment_differ": hulls_assign,
        "table_differ": table_diff,
        "explain": explain,
        "can_explain_retention": bool(len(explain) >= 2 and (flags["hulls"] or flags["obj_translation"] or flags["pair_dist"] or flags["table_contact"])),
    }


def downsample(frames, n=120):
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idx]


def plot_seed(seed: int, branches: dict) -> None:
    fig_dir = HERE / "plots" / f"seed{seed}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"low": "#3b6ea8", "mid": "#c47b16", "high": "#b42318"}
    specs = [
        ("squeeze", "squeeze vs time", "squeeze (N)", lambda f: f["squeeze"]),
        ("pair", "pair distance vs time", "pair (mm)", lambda f: 1000.0 * float(f.get("pair_dist") or 0.0)),
        ("trans", "object translation vs initial", "Δxyz (mm)", lambda f: 1000.0 * float(f.get("delta_xyz_norm") or 0.0)),
        ("rot", "object rotation vs initial", "Δrot (deg)", lambda f: float(f.get("delta_rot_deg") or 0.0)),
        ("mode", "contact mode vs time", "mode 0/1/2", lambda f: f["mode_code"]),
        ("table", "table contact force vs time", "table Fn (N)", lambda f: float(f.get("table_fn") or 0.0)),
    ]
    for name, title, ylab, fn in specs:
        fig, ax = plt.subplots(figsize=(9.2, 4.2))
        for label, packed in branches.items():
            frames = packed.get("frames") or []
            if not frames:
                continue
            ax.plot([f["t_s"] for f in frames], [fn(f) for f in frames], label=label, color=colors.get(label), lw=1.7)
            if name == "mode":
                ax.plot(
                    [f["t_s"] for f in frames],
                    [len(f.get("left_hulls") or []) for f in frames],
                    color=colors.get(label),
                    ls="--",
                    lw=1.0,
                    label=f"{label} n_left_hulls",
                )
        ax.set_title(f"seed {seed}: {title}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}.png", dpi=140)
        plt.close(fig)


def run_seed(seed: int, episode_id: int) -> dict:
    task = S.make_task(seed, episode_id)
    try:
        S.freeze_deskbin_material(task, DESK_MU)
        initial = rich_state(task)
        snapshot = PrefixSnapshot(task)
        init_hash = {
            "obj_xyz": initial["obj_xyz"],
            "obj_q": initial["obj_q"],
            "pair_dist": initial["pair_dist"],
            "aperture": initial["aperture"],
        }
        dump(HERE / "seeds" / f"{seed}_INITIAL.json", {"seed": seed, "episode_id": episode_id, "initial": initial})
        branches = {}
        for label in ("low", "mid", "high"):
            snapshot.restore()
            after = rich_state(task)
            if np.linalg.norm(np.asarray(after["obj_xyz"]) - np.asarray(initial["obj_xyz"])) > 1e-9:
                raise RuntimeError(f"restore drifted before paint: {after['obj_xyz']} vs {initial['obj_xyz']}")
            audit = paint(task, LEVELS[label]["mu_finger"])
            print(json.dumps({"event": "BRANCH", "seed": seed, "label": label, "runtime_mu": audit, "init": init_hash}), flush=True)
            task.activate_activeforcing_candidate_force()
            rec = Recorder(task, initial)
            scripted_establish_grasp(task)
            for _ in range(SETTLE):
                task.scene.step()
                task._after_physics_step()
            rec.close()
            window = []
            for _ in range(20):
                task.scene.step()
                window.append(S.pack_state(task))
            gate = S.classify_pre(window)
            pre = rich_state(task, initial)
            packed = {
                "label": label,
                "audit": audit,
                "gate": gate,
                "pre": pre,
                "row": row_of(label, pre),
                "first_contact_step": rec.first_contact_step,
                "first_contact_s": None if rec.first_contact_step is None else rec.first_contact_step * DT,
                "table_leave_step": rec.table_leave_step,
                "table_leave_s": None if rec.table_leave_step is None else rec.table_leave_step * DT,
                "n_steps": rec.n,
                "peak_squeeze": float(max((f["squeeze"] for f in rec.frames), default=0.0)),
                "frames": rec.frames,
            }
            branches[label] = packed
            dump(
                HERE / "seeds" / f"{seed}_{label}.json",
                {k: v for k, v in packed.items() if k != "frames"},
            )
            dump(HERE / "seeds" / f"{seed}_{label}_TRACE.json", downsample(rec.frames, 160))
            print(
                json.dumps(
                    {
                        "event": "PRE",
                        "seed": seed,
                        "label": label,
                        "gate": gate["gate"],
                        "squeeze": pre["squeeze"],
                        "pair_mm": None if pre["pair_dist"] is None else 1000.0 * pre["pair_dist"],
                        "dxyz_mm": None if pre.get("delta_xyz") is None else [1000.0 * x for x in pre["delta_xyz"]],
                        "drot_deg": pre.get("delta_rot_deg"),
                        "hulls": pre.get("finger_hulls"),
                        "table": pre["table_contact"],
                        "table_fn": pre["table_fn"],
                        "first_contact_s": packed["first_contact_s"],
                        "slip_mm": None if pre.get("rel_slip_norm") is None else 1000.0 * pre["rel_slip_norm"],
                        "opposing": pre["opposing"],
                        "mu_eff": audit["mu_eff"],
                    },
                    default=str,
                ),
                flush=True,
            )
        cmp = compare_mid_high(branches["mid"], branches["high"])
        plot_seed(seed, branches)
        table = [row_of("Initial", initial)] + [branches[k]["row"] for k in ("low", "mid", "high")]
        traces = {k: downsample(branches[k]["frames"], 80) for k in ("low", "mid", "high")}
        return {
            "seed": seed,
            "episode_id": episode_id,
            "same_initial": True,
            "initial": init_hash,
            "deskbin_id": branches["mid"]["audit"]["deskbin_id"],
            "n_shapes": branches["mid"]["audit"]["n_shapes"],
            "gates": {k: branches[k]["gate"] for k in branches},
            "table": table,
            "mid_vs_high": cmp,
            "first_contact_s": {k: branches[k]["first_contact_s"] for k in branches},
            "table_leave_s": {k: branches[k]["table_leave_s"] for k in branches},
            "peak_squeeze": {k: branches[k]["peak_squeeze"] for k in branches},
            "traces": traces,
        }
    finally:
        try:
            task.close_env()
        except Exception:
            pass


def verdict_of(cmp: dict, branches_table: list) -> tuple[str, str]:
    if cmp["can_explain_retention"]:
        return "PASS", "PRE mid/high diverge in pose/hulls/pinch/table enough to explain later retention."
    if cmp["diverged"]:
        return "MIXED", "Some PRE differences exist, but they do not clearly explain mid/high retention."
    return "FAIL", "low/mid/high PRE are essentially the same grasp."


def main() -> None:
    primary = run_seed(SEED, EPISODE)
    dump(HERE / "PRIMARY.json", {k: v for k, v in primary.items() if k != "traces"})
    dump(HERE / "COMPARE.json", primary["table"])
    dump(HERE / "MID_VS_HIGH.json", primary["mid_vs_high"])
    dump(HERE / "TRACES_DOWNSAMPLED.json", primary["traces"])
    verdict, why = verdict_of(primary["mid_vs_high"], primary["table"])
    extras = []
    if primary["mid_vs_high"]["diverged"]:
        for seed in PLAN["extra_seeds_if_diverged"]:
            extra = run_seed(int(seed), 1)
            extras.append({k: v for k, v in extra.items() if k != "traces"})
            dump(HERE / "seeds" / f"{seed}_SUMMARY.json", extras[-1])
            print(json.dumps({"event": "EXTRA", "seed": seed, "mid_vs_high": extra["mid_vs_high"]}, default=str), flush=True)
    payload = {
        "verdict": verdict,
        "why": why,
        "primary": {k: v for k, v in primary.items() if k != "traces"},
        "extras": extras,
        "n_extra_diverged": int(sum(1 for e in extras if e["mid_vs_high"]["diverged"])),
    }
    dump(HERE / "VERDICT.json", payload)
    print(json.dumps({"done": True, "verdict": verdict, "mid_vs_high": primary["mid_vs_high"]}, default=str), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"fatal": traceback.format_exc()}), flush=True)
        raise
