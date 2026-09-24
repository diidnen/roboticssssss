#!/usr/bin/env python3
"""Same PRE snapshot: mid vs high through query → realize → lift. No dump."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
GRASP = Path("/media/volume/dasdas/exouser/af_dump_grasp_establishment_divergence_20260916")
OLD = Path("/media/volume/dasdas/exouser/af_dump_original_geom_finger_friction_sanity_20260916")
PLAN = json.loads((HERE / "PLAN.json").read_text())
sys.path[:0] = [str(GRASP), str(OLD)]

import run_grasp_establishment as G
import run_original_geom_finger_sweep as S
from envs.utils import ArmTag
from force_realize import (
    HOLD_WINDOW,
    REALIZE_TOL,
    SETTLE_STEPS,
    LiftSqueezeHold,
    capture_stock_drives,
    install_lift_squeeze,
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
FORCES = [float(x) for x in PLAN["forces_N"]]
MUS = {"mid": 0.575, "high": 0.85}
THR = PLAN["obvious_divergence"]
STAGES = PLAN["stages"]
DT = 0.004
LIFT_Z = 0.08


def dump(path: Path, value) -> None:
    S.dump_json(path, value)


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
    }
    if extra:
        row.update(extra)
    return row


def dropped(state: dict) -> bool:
    return state.get("mode") == "none" or (
        state.get("mode") != "bilateral" and float(state.get("squeeze") or 0.0) < 0.05
    )


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
        "drop": dropped(mid) != dropped(high),
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
        first.append(f"pose Δ={1000*pose:.2f} mm")
    if flags["rot"]:
        first.append(f"rot Δ={rot:.2f} deg")
    if flags["pair"]:
        first.append(f"pair Δ={1000*pair:.2f} mm")
    if flags["slip"]:
        first.append(f"slip Δ={1000*slip:.2f} mm")
    if flags["table"]:
        first.append("table contact mismatch")
    if flags["hull_assignment"] and not obvious:
        first.append("hull assignment only (not obvious)")
    return {
        "flags": flags,
        "obvious": obvious,
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
        "table_mid": mid.get("table_contact"),
        "table_high": high.get("table_contact"),
        "first_diffs": first,
    }


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


def realize_recorded(task, force_n: float, stock, ref: dict):
    holder = LiftSqueezeHold(task, force_n, stock)
    install_lift_squeeze(task, holder)
    task.af_force_limit_n = float(force_n)
    frames = []
    t2 = None
    drop_i = None
    rows = []
    for i in range(int(SETTLE_STEPS)):
        task.scene.step()
        last = dict(holder.last or {})
        rows.append(last)
        if i == 0:
            t2 = pack(G.rich_state(task, ref), {"F_cmd": force_n, "tracking_err": abs(float(last.get("single_finger_n") or 0) - force_n)})
        if drop_i is None and dropped(G.rich_state(task, ref)):
            drop_i = i
        if i % 5 == 0 or i + 1 == SETTLE_STEPS:
            st = G.rich_state(task, ref)
            frames.append(
                {
                    "i": i,
                    "t_s": i * DT,
                    "squeeze": st["squeeze"],
                    "mode": st["mode"],
                    "pair_dist": st["pair_dist"],
                    "rel_slip_norm": st.get("rel_slip_norm"),
                    "delta_xyz_norm": st.get("delta_xyz_norm"),
                    "delta_rot_deg": st.get("delta_rot_deg"),
                    "table_fn": st["table_fn"],
                    "table_contact": st["table_contact"],
                    "single": last.get("single_finger_n"),
                    "bilateral": last.get("bilateral_contact"),
                }
            )
    t3_state = G.rich_state(task, ref)
    window = rows[-HOLD_WINDOW:]
    singles = [float(r.get("single_finger_n") or 0.0) for r in window]
    mean = float(np.mean(singles))
    frac = float(np.mean([bool(r.get("bilateral_contact")) for r in window]))
    rel = abs(mean - float(force_n)) / max(float(force_n), 1e-6)
    info = {
        "commanded_force_N": float(force_n),
        "pre_motion_squeeze_mean_n": mean,
        "pre_motion_rel_error": rel,
        "pre_motion_bilateral_frac": frac,
        "force_realized_before_motion": bool(frac >= 0.8 and rel <= REALIZE_TOL),
        "drop_step": drop_i,
    }
    t3 = pack(t3_state, {"F_cmd": force_n, "tracking_err": abs(mean - force_n), "realize": info})
    return t2, t3, frames, drop_i, info


class LiftRec:
    def __init__(self, task, ref):
        self.task = task
        self.ref = ref
        self.frames = []
        self.n = 0
        self.table_leave = None
        self.drop_n = None
        self.unilateral_n = None
        self._orig = task._after_physics_step

        def hooked():
            self._orig()
            self.n += 1
            if self.n % 4 and self.n > 2:
                return
            st = G.rich_state(task, self.ref)
            self.frames.append(
                {
                    "step": self.n,
                    "t_s": self.n * DT,
                    "squeeze": st["squeeze"],
                    "mode": st["mode"],
                    "pair_dist": st["pair_dist"],
                    "rel_slip_norm": st.get("rel_slip_norm"),
                    "rel_slip_y": st.get("rel_slip_y"),
                    "delta_xyz_norm": st.get("delta_xyz_norm"),
                    "delta_rot_deg": st.get("delta_rot_deg"),
                    "obj_xyz": st["obj_xyz"],
                    "obj_q": st["obj_q"],
                    "finger_hulls": {
                        k: {"shape_ids": v.get("shape_ids"), "axes": v.get("axes"), "n_shapes": v.get("n_shapes")}
                        for k, v in (st.get("finger_hulls") or {}).items()
                    },
                    "opposing": st["opposing"],
                    "obj_z": st["obj_xyz"][2],
                    "table_fn": st["table_fn"],
                    "table_contact": st["table_contact"],
                }
            )
            if self.table_leave is None and not st["table_contact"]:
                self.table_leave = self.n
            if self.unilateral_n is None and st["mode"] == "unilateral":
                self.unilateral_n = self.n
            if self.drop_n is None and dropped(st):
                self.drop_n = self.n

        task._after_physics_step = hooked

    def close(self):
        self.task._after_physics_step = self._orig


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
    print(json.dumps({"event": "PRE", "gate": gate["gate"], "cr": gate["window_cr"]}), flush=True)
    if gate["gate"] != "VALID":
        raise RuntimeError(f"PRE not VALID: {gate}")
    pre = G.rich_state(task)
    snapshot = PrefixSnapshot(task)
    dump(HERE / "FIXTURE.json", {"seed": SEED, "episode_id": EPISODE, "gate": gate, "pre": pack(pre), "deskbin_mu": DESK_MU})
    return task, stock, snapshot, pre, gate


def run_branch(task, stock, snapshot, pre, mu_label: str, force_n: float, do_query: bool):
    uninstall_lift_squeeze(task)
    snapshot.restore()
    audit = G.paint(task, MUS[mu_label])
    t0 = pack(G.rich_state(task, pre), {"F_cmd": None})
    query = None
    if do_query:
        query = run_query(task)
    t1 = pack(G.rich_state(task, pre), {"query_cr": None if query is None else query.get("contact_ratio")})
    t2, t3, realize_frames, drop_r, realize = realize_recorded(task, force_n, stock, pre)
    rec = LiftRec(task, pre)
    t4 = pack(G.rich_state(task, pre))
    try:
        task.move(task.move_by_displacement(arm_tag=ArmTag("left"), z=LIFT_Z, move_axis="arm"))
        lift_err = None
    except Exception as exc:
        lift_err = repr(exc)
    rec.close()
    frames = rec.frames
    t6 = pack(G.rich_state(task, pre))
    if frames:
        mid_i = len(frames) // 2
        fr = frames[mid_i]
        t5 = {
            "squeeze": fr["squeeze"],
            "mode": fr["mode"],
            "pair_dist": fr["pair_dist"],
            "obj_xyz": fr.get("obj_xyz") or t6["obj_xyz"],
            "obj_q": fr.get("obj_q") or t6["obj_q"],
            "finger_hulls": fr.get("finger_hulls") or t6["finger_hulls"],
            "all_finger_hulls": None,
            "opposing": fr.get("opposing"),
            "table_contact": fr["table_contact"],
            "table_fn": fr["table_fn"],
            "delta_xyz_norm": fr.get("delta_xyz_norm"),
            "delta_rot_deg": fr.get("delta_rot_deg"),
            "rel_slip_norm": fr.get("rel_slip_norm"),
            "rel_slip_y": fr.get("rel_slip_y"),
            "nl": None,
            "nr": None,
            "approx_from_trace": True,
        }
    else:
        t5 = pack(G.rich_state(task, pre), {"approx_from_trace": False})
    uninstall_lift_squeeze(task)
    restore_stock_drives(stock)
    stages = {
        "T0_PRE": t0,
        "T1_handoff": t1,
        "T2_F_applied": t2,
        "T3_realized": t3,
        "T4_lift_start": t4,
        "T5_lift_mid": t5,
        "T6_lift_end": t6,
    }
    return {
        "mu": mu_label,
        "mu_finger": MUS[mu_label],
        "mu_eff": 0.5 * (MUS[mu_label] + DESK_MU),
        "F": force_n,
        "do_query": do_query,
        "audit": audit,
        "query": None
        if query is None
        else {"contact_ratio": query.get("contact_ratio"), "query_failed": bool(query.get("query_failed"))},
        "realize": realize,
        "stages": stages,
        "realize_frames": realize_frames,
        "lift_frames": frames,
        "table_leave_s": None if rec.table_leave is None else rec.table_leave * DT,
        "first_unilateral_s": None if rec.unilateral_n is None else rec.unilateral_n * DT,
        "drop_realize_step": drop_r,
        "drop_lift_s": None if rec.drop_n is None else rec.drop_n * DT,
        "lift_error": lift_err,
        "dropped_at_T6": dropped(t6) if isinstance(t6, dict) and t6.get("mode") else False,
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
                "mid_hulls": m.get("finger_hulls"),
                "high_hulls": h.get("finger_hulls"),
                "pose_delta_m": diff["pose_gap_m"],
                "pair_delta_m": diff["pair_gap_m"],
                "slip_delta_m": diff["slip_gap_m"],
                "rot_delta_deg": diff["rot_gap_deg"],
                "table_mid": m.get("table_contact"),
                "table_high": h.get("table_contact"),
                "obvious": diff["obvious"],
                "first_diffs": diff["first_diffs"],
            }
        )
    return rows


def first_obvious(rows: list[dict]) -> tuple[str | None, str | None]:
    last_same = None
    for row in rows:
        if row["obvious"]:
            return row["stage"], last_same
        last_same = row["stage"]
    return None, last_same


def classify(force_rows: dict) -> str:
    votes = []
    for force, rows in force_rows.items():
        stage, _ = first_obvious(rows)
        votes.append(stage)
    if all(v is None for v in votes):
        return "NO_CLEAR_DIVERGENCE"
    # first non-null among the mid-fail forces preferred
    for key in ("1.0", "2.5", "0.75", "5.0"):
        if key in force_rows:
            stage, _ = first_obvious(force_rows[key])
            if stage == "T1_handoff":
                return "QUERY"
            if stage in {"T2_F_applied", "T3_realized"}:
                return "FORCE_REALIZATION"
            if stage in {"T4_lift_start", "T5_lift_mid", "T6_lift_end"}:
                return "LIFT"
    nonempty = [v for v in votes if v]
    if not nonempty:
        return "NO_CLEAR_DIVERGENCE"
    if any(v == "T1_handoff" for v in nonempty):
        return "QUERY"
    if any(v in {"T2_F_applied", "T3_realized"} for v in nonempty):
        return "FORCE_REALIZATION"
    if any(v and v.startswith("T") for v in nonempty):
        return "LIFT"
    return "NO_CLEAR_DIVERGENCE"


def plot_force(force: float, mid_b, high_b) -> None:
    fig_dir = HERE / "plots"
    fig_dir.mkdir(exist_ok=True)
    series = []
    t0 = 0.0
    for label, packed, color in (("mid", mid_b, "#c47b16"), ("high", high_b, "#b42318")):
        r = packed["realize_frames"]
        l = packed["lift_frames"]
        rt = [float(x["t_s"]) for x in r]
        lt = [t0 + (rt[-1] if rt else 0.6) + 0.05 + float(x["t_s"]) for x in l]
        series.append((label, color, r, l, rt, lt))
    specs = [
        ("squeeze", "squeeze (N)", lambda x: x["squeeze"]),
        ("pair", "pair (mm)", lambda x: 1000.0 * float(x.get("pair_dist") or 0)),
        ("mode", "mode none/uni/bi", lambda x: {"none": 0, "unilateral": 1, "bilateral": 2}.get(x.get("mode"), 0)),
        ("slip", "rel slip (mm)", lambda x: 1000.0 * float(x.get("rel_slip_norm") or 0)),
        ("pose", "Δxyz from PRE (mm)", lambda x: 1000.0 * float(x.get("delta_xyz_norm") or 0)),
    ]
    fig, axes = plt.subplots(len(specs), 1, figsize=(9.4, 2.15 * len(specs)), sharex=False)
    for ax, (name, ylab, fn) in zip(axes, specs):
        for label, color, r, l, rt, lt in series:
            if r:
                ax.plot(rt, [fn(x) for x in r], color=color, lw=1.6, label=f"{label} realize")
            if l:
                ax.plot(lt, [fn(x) for x in l], color=color, lw=1.6, ls="--", label=f"{label} lift")
        ax.set_ylabel(ylab)
        ax.set_title(f"F={force:g} N: {name}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    axes[-1].set_xlabel("time (s)  [lift shifted after realize]")
    fig.tight_layout()
    fig.savefig(fig_dir / f"F{force:g}.png", dpi=130)
    plt.close(fig)


def main() -> None:
    task, stock, snapshot, pre, gate = build_fixture()
    results = {}
    try:
        for force in FORCES:
            results[force] = {}
            for mu in ("mid", "high"):
                packed = run_branch(task, stock, snapshot, pre, mu, force, do_query=True)
                results[force][mu] = packed
                dump(HERE / "branches" / f"{mu}_F{force:.2f}.json", {k: v for k, v in packed.items() if k not in {"realize_frames", "lift_frames"}})
                print(
                    json.dumps(
                        {
                            "event": "BRANCH",
                            "mu": mu,
                            "F": force,
                            "T1_sq": packed["stages"]["T1_handoff"]["squeeze"],
                            "T3_sq": packed["stages"]["T3_realized"]["squeeze"],
                            "T3_mode": packed["stages"]["T3_realized"]["mode"],
                            "T6_sq": packed["stages"]["T6_lift_end"]["squeeze"],
                            "T6_mode": packed["stages"]["T6_lift_end"]["mode"],
                            "realized": packed["realize"]["force_realized_before_motion"],
                            "drop_lift_s": packed["drop_lift_s"],
                            "table_leave_s": packed["table_leave_s"],
                        }
                    ),
                    flush=True,
                )
            table = pair_table(results[force]["mid"], results[force]["high"])
            dump(HERE / "tables" / f"F{force:.2f}.json", table)
            stage, last_same = first_obvious(table)
            print(json.dumps({"event": "PAIR", "F": force, "first_obvious": stage, "last_same": last_same, "table": [
                {k: r[k] for k in ("stage", "mid_squeeze", "high_squeeze", "mid_mode", "high_mode", "pose_delta_m", "obvious", "first_diffs")}
                for r in table
            ]}), flush=True)
            plot_force(force, results[force]["mid"], results[force]["high"])

        force_rows = {f"{f:g}": pair_table(results[f]["mid"], results[f]["high"]) for f in FORCES}
        kind = classify(force_rows)
        realize_div = any(
            first_obvious(force_rows[f"{f:g}"])[0] in {"T2_F_applied", "T3_realized"} for f in FORCES
        )
        controls = {}
        if realize_div and PLAN["no_query_control_if_realize_diverges"]:
            for force in (1.0, 2.5):
                controls[force] = {}
                for mu in ("mid", "high"):
                    packed = run_branch(task, stock, snapshot, pre, mu, force, do_query=False)
                    controls[force][mu] = {k: v for k, v in packed.items() if k not in {"realize_frames", "lift_frames"}}
                    print(json.dumps({"event": "NOQUERY", "mu": mu, "F": force, "T3_sq": packed["stages"]["T3_realized"]["squeeze"], "T6_mode": packed["stages"]["T6_lift_end"]["mode"]}), flush=True)
                dump(
                    HERE / "tables" / f"noquery_F{force:.2f}.json",
                    pair_table(
                        {"stages": controls[force]["mid"]["stages"]},
                        {"stages": controls[force]["high"]["stages"]},
                    ),
                )

        payload = {
            "forces": FORCES,
            "classification": kind,
            "tables": force_rows,
            "first_obvious": {f"{f:g}": first_obvious(force_rows[f"{g:g}"])[0] for f in FORCES for g in [f]},
            "last_same": {f"{f:g}": first_obvious(force_rows[f"{g:g}"])[1] for f in FORCES for g in [f]},
            "noquery_ran": bool(controls),
        }
        dump(HERE / "VERDICT.json", payload)
        print(json.dumps({"done": True, "classification": kind, "first_obvious": payload["first_obvious"]}, default=str), flush=True)
    finally:
        try:
            uninstall_lift_squeeze(task)
            task.close_env()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"fatal": traceback.format_exc()}), flush=True)
        raise
