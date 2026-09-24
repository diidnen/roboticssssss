#!/usr/bin/env python3
"""PRE-POUR retention segment: lift + hold + short translation, no wrist/pour."""
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
GRASP = Path("/media/volume/dasdas/exouser/af_dump_grasp_establishment_divergence_20260916")
PLAN = json.loads((HERE / "PLAN.json").read_text())
sys.path[:0] = [str(GRASP), str(OLD)]

import run_grasp_establishment as G
import run_original_geom_finger_sweep as S
from envs.utils import Action, ArmTag
from force_realize import (
    realize_commanded_force,
    restore_stock_drives,
    uninstall_lift_squeeze,
)

SEED0 = int(PLAN["seed_first"])
EPISODE = int(PLAN["episode_id"])
DESK_MU = float(PLAN["deskbin_mu_fixed"])
FORCES = [float(x) for x in PLAN["force_grid_N"]]
LEVELS = PLAN["finger_mu"]
HOLD_STEPS = 100
TRANSLATE_M = 0.08
LIFT_Z = 0.08
DT = 0.004
STRIDE = 4
PERSISTENT = 8
ARM = ArmTag("left")
BINS = PLAN["force_bins"]
FULL_DUMP = PLAN["full_dump_200014_bits_frozen"]
EXTRA_SEEDS = [int(x) for x in PLAN["multi_seed_if_200014_clear_coulomb"]]


def dump(path: Path, value) -> None:
    S.dump_json(path, value)


def log(event: dict) -> None:
    print(json.dumps(event), flush=True)


def quat_angle_deg(q0, q1) -> float:
    return G.quat_angle_deg(q0, q1)


def dropped_state(st: dict) -> bool:
    return st.get("mode") == "none" or (
        st.get("mode") != "bilateral" and float(st.get("squeeze") or 0.0) < 0.05
    )


def ee_pose(task):
    return [float(x) for x in task.get_arm_pose(ARM)]


def short_translate_action(task, distance: float):
    ee = np.asarray(ee_pose(task), dtype=float)
    pour_xyz = np.asarray(task.pour_actions[1][0].target_pose[:3], dtype=float)
    delta = pour_xyz - ee[:3]
    nrm = float(np.linalg.norm(delta))
    if nrm < 1e-9:
        step = np.array([0.0, 0.0, distance], dtype=float)
    else:
        step = delta / nrm * min(float(distance), nrm)
    target = ee.copy()
    target[:3] = ee[:3] + step
    return (ARM, [Action(ARM, "move", target.tolist())]), step.tolist(), nrm


class SegRec:
    def __init__(self, task, ref, lift_q):
        self.task = task
        self.ref = ref
        self.lift_q = lift_q
        self.phase = "pre"
        self.frames = []
        self.n = 0
        self._orig = task._after_physics_step

        def hooked():
            self._orig()
            self.n += 1
            if self.n % STRIDE:
                return
            st = G.rich_state(task, self.ref)
            ee = ee_pose(task)
            self.frames.append(
                {
                    "step": self.n,
                    "t_s": self.n * DT,
                    "phase": self.phase,
                    "squeeze": st["squeeze"],
                    "mode": st["mode"],
                    "pair_dist": st["pair_dist"],
                    "rel_slip_norm": st.get("rel_slip_norm") or 0.0,
                    "delta_xyz_norm": st.get("delta_xyz_norm") or 0.0,
                    "obj_z": st["obj_xyz"][2],
                    "dropped": dropped_state(st),
                    "wrist_deg": quat_angle_deg(self.lift_q, ee[3:7]),
                    "bilateral": st["mode"] == "bilateral",
                }
            )

        task._after_physics_step = hooked

    def close(self):
        self.task._after_physics_step = self._orig


def persistent_in(frames) -> bool:
    run = 0
    for fr in frames:
        if fr.get("dropped"):
            run += 1
            if run >= PERSISTENT:
                return True
        else:
            run = 0
    return False


def first_time(frames, key, pred) -> float | None:
    for fr in frames:
        if pred(fr):
            return float(fr["t_s"])
    return None


def phase_report(frames, name, start_z) -> dict:
    rows = [f for f in frames if f.get("phase") == name]
    last = rows[-1] if rows else None
    return {
        "n": len(rows),
        "ok": None if not rows else (not persistent_in(rows)),
        "end_squeeze": None if last is None else last["squeeze"],
        "end_mode": None if last is None else last["mode"],
        "end_dropped": None if last is None else last["dropped"],
        "end_z": None if last is None else last["obj_z"],
        "max_wrist_deg": 0.0 if not rows else float(max(f["wrist_deg"] for f in rows)),
        "still_lifted": None if last is None else bool(last["obj_z"] > start_z + 0.04),
    }


def score_segment(frames, start_z) -> dict:
    lift = phase_report(frames, "lift", start_z)
    hold = phase_report(frames, "hold", start_z)
    trans = phase_report(frames, "translate", start_z)
    last = frames[-1] if frames else None
    final_ok = bool(
        last is not None
        and last["mode"] == "bilateral"
        and float(last["squeeze"]) > 0.05
        and last["obj_z"] > start_z + 0.04
        and not last["dropped"]
    )
    # Empty hold frames: raw scene.step does not call _after_physics_step.
    # Physics still ran; infer hold from lift end + translation/final.
    hold_ok = True if hold["ok"] is None else bool(hold["ok"])
    lift_ok = False if lift["ok"] is None else bool(lift["ok"])
    trans_ok = False if trans["ok"] is None else bool(trans["ok"])
    y = int(lift_ok and hold_ok and trans_ok and final_ok)
    bi = [int(f["bilateral"]) for f in frames]
    return {
        "y_retention": y,
        "lift_ok": lift_ok,
        "hold_ok": hold_ok,
        "translation_ok": trans_ok,
        "final_ok": final_ok,
        "persistent_detach": persistent_in(frames),
        "drop": int((last is None) or last["dropped"] or not final_ok),
        "contact_ratio_local": float(np.mean(bi)) if bi else 0.0,
        "first_unilateral_s": first_time(frames, "mode", lambda f: f["mode"] == "unilateral"),
        "first_slip_s": first_time(frames, "slip", lambda f: float(f.get("rel_slip_norm") or 0) >= 0.003),
        "first_drop_s": first_time(frames, "drop", lambda f: f["dropped"]),
        "end_squeeze": None if last is None else last["squeeze"],
        "end_mode": None if last is None else last["mode"],
        "end_pair_dist": None if last is None else last["pair_dist"],
        "end_slip": None if last is None else last["rel_slip_norm"],
        "end_z": None if last is None else last["obj_z"],
        "max_wrist_deg": trans["max_wrist_deg"],
        "max_wrist_deg_lift": lift["max_wrist_deg"],
        "lift": lift,
        "hold": hold,
        "translate": trans,
    }


def run_segment(task, force, stock, desk_mu, finger_mu, ref):
    S.prepare_condition(task, desk_mu, finger_mu)
    task.plan_success = True
    realize = realize_commanded_force(task, force, stock)
    task._af_force_trace = []
    task._af_force_trace_step = 0
    task._af_no_contact_samples = 0
    task._af_ever_lifted = False
    task.activate_activeforcing_candidate_force()
    restore_stock_drives(stock)
    start_z = float(task.deskbin.get_pose().p[2])
    lift_q = ee_pose(task)[3:7]
    slip_ref = G.rich_state(task)
    rec = SegRec(task, slip_ref, lift_q)
    err = None
    commanded_step = None
    pour_dist = None
    try:
        rec.phase = "lift"
        task.move(task.move_by_displacement(arm_tag=ARM, z=LIFT_Z, move_axis="arm"))
        rec.lift_q = ee_pose(task)[3:7]
        rec.phase = "hold"
        for _ in range(HOLD_STEPS):
            task.scene.step()
            rec._orig()
            rec.n += 1
            if rec.n % STRIDE:
                continue
            st = G.rich_state(task, rec.ref)
            ee = ee_pose(task)
            rec.frames.append(
                {
                    "step": rec.n,
                    "t_s": rec.n * DT,
                    "phase": rec.phase,
                    "squeeze": st["squeeze"],
                    "mode": st["mode"],
                    "pair_dist": st["pair_dist"],
                    "rel_slip_norm": st.get("rel_slip_norm") or 0.0,
                    "delta_xyz_norm": st.get("delta_xyz_norm") or 0.0,
                    "obj_z": st["obj_xyz"][2],
                    "dropped": dropped_state(st),
                    "wrist_deg": quat_angle_deg(rec.lift_q, ee[3:7]),
                    "bilateral": st["mode"] == "bilateral",
                }
            )
        rec.phase = "translate"
        act, commanded_step, pour_dist = short_translate_action(task, TRANSLATE_M)
        cmd_q = act[1][0].target_pose[3:7]
        if quat_angle_deg(rec.lift_q, cmd_q) > 1.0:
            raise RuntimeError(f"translation commanded wrist rotation {quat_angle_deg(rec.lift_q, cmd_q):.2f} deg")
        task.move(act)
    except Exception as exc:
        err = repr(exc)
        log({"event": "segment_error", "F": force, "error": err})
    rec.close()
    uninstall_lift_squeeze(task)
    scored = score_segment(rec.frames, start_z)
    metrics = task.compute_activeforcing_dynamic_metrics()
    irrec = bool(metrics.get("irrecoverable_failure"))
    return {
        "force_N": float(force),
        "y_retention": scored["y_retention"],
        "lift_ok": scored["lift_ok"],
        "hold_ok": scored["hold_ok"],
        "translation_ok": scored["translation_ok"],
        "final_ok": scored["final_ok"],
        "drop": scored["drop"],
        "persistent_detach": scored["persistent_detach"],
        "irrecoverable": irrec,
        "contact_ratio_local": scored["contact_ratio_local"],
        "contact_ratio_af": float(metrics.get("contact_ratio") or 0.0),
        "settle_squeeze_mean_n": realize.get("pre_motion_squeeze_mean_n"),
        "end_squeeze": scored["end_squeeze"],
        "end_mode": scored["end_mode"],
        "end_pair_dist": scored["end_pair_dist"],
        "end_slip": scored["end_slip"],
        "first_unilateral_s": scored["first_unilateral_s"],
        "first_slip_s": scored["first_slip_s"],
        "first_drop_s": scored["first_drop_s"],
        "max_wrist_deg": scored["max_wrist_deg"],
        "commanded_xyz_step": commanded_step,
        "pour_xyz_remaining_m": pour_dist,
        "error": err,
        "force_realized": bool(realize.get("force_realized_before_motion")),
        "n_frames": len(rec.frames),
        "lift": scored["lift"],
        "hold": scored["hold"],
        "translate": scored["translate"],
    }


def curve_stats(forces, bits) -> dict:
    bits = [int(x) for x in bits]
    reversals = sum(1 for a, b in zip(bits, bits[1:]) if a == 1 and b == 0)
    best = []
    cur = []
    for f, b in zip(forces, bits):
        if b:
            cur.append(float(f))
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    fmin = next((float(f) for f, b in zip(forces, bits) if b), None)
    monotonic = reversals == 0 and (fmin is None or all(bits[i] == 1 for i, f in enumerate(forces) if f >= fmin - 1e-9))
    def bin_rate(lo, hi):
        sel = [b for f, b in zip(forces, bits) if lo - 1e-9 <= f <= hi + 1e-9]
        return None if not sel else float(np.mean(sel))
    return {
        "bits": bits,
        "n_retain": int(sum(bits)),
        "n": len(bits),
        "rate_all": float(np.mean(bits)) if bits else 0.0,
        "rate_lowF": bin_rate(*BINS["low"]),
        "rate_midF": bin_rate(*BINS["mid"]),
        "rate_highF": bin_rate(*BINS["high"]),
        "F_min_first": fmin,
        "longest_success_interval": None if not best else [best[0], best[-1]],
        "longest_success_n": len(best),
        "reversals": reversals,
        "monotonic_once_on": bool(monotonic),
    }


def pre_pack(captured) -> dict:
    geom = captured.get("pre_geom") or {}
    end = captured.get("pre_end") or {}
    per = geom.get("per_finger") or end.get("per_finger") or {}
    return {
        "seed": None,
        "episode_id": captured["episode_id"],
        "PRE_VALID": captured["pre_gate"]["gate"] == "VALID",
        "gate": captured["pre_gate"]["gate"],
        "window_cr": captured["pre_gate"].get("window_cr"),
        "PRE_squeeze": end.get("squeeze") or geom.get("squeeze"),
        "pair_distance": end.get("pair_dist") or geom.get("pair_dist"),
        "contact_hulls": {k: v.get("shape_ids") for k, v in per.items() if v.get("present") or v.get("shape_ids")},
        "contact_normals": {k: v.get("normals") for k, v in per.items() if v.get("present") or v.get("normals")},
        "query_cr": None if captured.get("query") is None else captured["query"].get("contact_ratio"),
    }


def run_level(seed: int, label: str, finger_mu: float) -> dict:
    captured = S.find_and_capture(seed, DESK_MU, finger_mu, label, EPISODE, EPISODE)
    pre = pre_pack(captured)
    pre["seed"] = seed
    pre["mu"] = label
    pre["mu_finger"] = finger_mu
    pre["mu_eff"] = 0.5 * (finger_mu + DESK_MU)
    dump(HERE / f"PRE_{seed}_{label}.json", pre)
    log({"event": "PRE", "seed": seed, "mu": label, "valid": pre["PRE_VALID"], "squeeze": pre["PRE_squeeze"]})
    if not pre["PRE_VALID"]:
        try:
            captured["task"].close_env()
        except Exception:
            pass
        return {"seed": seed, "label": label, "pre": pre, "skipped": True, "reason": "PRE_INVALID"}
    task = captured["task"]
    ref = G.rich_state(task)
    outcomes = []
    try:
        for force in FORCES:
            captured["snapshot"].restore()
            S.prepare_condition(task, DESK_MU, finger_mu)
            row = run_segment(task, force, captured["stock"], DESK_MU, finger_mu, ref)
            outcomes.append(row)
            log(
                {
                    "event": "F",
                    "seed": seed,
                    "mu": label,
                    "F": force,
                    "y": row["y_retention"],
                    "lift": row["lift_ok"],
                    "hold": row["hold_ok"],
                    "trans": row["translation_ok"],
                    "final": row["final_ok"],
                    "CR": round(row["contact_ratio_local"], 3),
                    "wrist": round(row["max_wrist_deg"], 2),
                    "sq": None if row["end_squeeze"] is None else round(row["end_squeeze"], 2),
                }
            )
    finally:
        try:
            task.close_env()
        except Exception:
            pass
    bits = [o["y_retention"] for o in outcomes]
    stats = curve_stats(FORCES, bits)
    payload = {
        "seed": seed,
        "label": label,
        "mu_finger": finger_mu,
        "mu_eff": 0.5 * (finger_mu + DESK_MU),
        "pre": pre,
        "outcomes": outcomes,
        "stats": stats,
        "max_wrist_deg_all_F": float(max((o["max_wrist_deg"] or 0) for o in outcomes) if outcomes else 0),
    }
    dump(HERE / f"SWEEP_{seed}_{label}.json", payload)
    return payload


def coulomb_like(by_mu: dict) -> dict:
    rates = {k: by_mu[k]["stats"]["rate_all"] for k in ("low", "mid", "high")}
    fmins = {k: by_mu[k]["stats"]["F_min_first"] for k in ("low", "mid", "high")}
    rev = {k: by_mu[k]["stats"]["reversals"] for k in ("low", "mid", "high")}
    rate_order = rates["low"] <= rates["mid"] + 1e-12 and rates["mid"] <= rates["high"] + 1e-12
    fmin_order = True
    if None not in fmins.values():
        fmin_order = fmins["low"] + 1e-12 >= fmins["mid"] >= fmins["high"] - 1e-12
    elif fmins["low"] is None and (fmins["mid"] is not None or fmins["high"] is not None):
        fmin_order = False
    full_rev = sum(curve_stats(FORCES, FULL_DUMP[k])["reversals"] for k in ("low", "mid", "high"))
    now_rev = sum(rev.values())
    low_hardest = rates["low"] + 1e-9 < min(rates["mid"], rates["high"])
    high_easiest = rates["high"] + 1e-9 > max(rates["low"], rates["mid"]) or (
        fmins["high"] is not None and fmins["low"] is not None and fmins["high"] < fmins["low"]
    )
    return {
        "rate_low_lt_mid_le_high": bool(rates["low"] < rates["mid"] <= rates["high"] + 1e-9),
        "rate_low_le_mid_le_high": bool(rate_order),
        "fmin_low_ge_mid_ge_high": bool(fmin_order),
        "low_hardest": bool(low_hardest),
        "high_easiest": bool(high_easiest),
        "reversals_now": now_rev,
        "reversals_full_dump": full_rev,
        "fewer_reversals_than_full_dump": now_rev < full_rev,
        "clear_coulomb": bool(high_easiest and fmin_order and now_rev < full_rev),
    }


def classify(by_mu: dict, flag: dict) -> str:
    if flag["clear_coulomb"]:
        return "PASS"
    if flag["low_hardest"] and not (by_mu["mid"]["stats"]["rate_all"] > by_mu["high"]["stats"]["rate_all"] + 0.25):
        return "MIXED"
    if flag["low_hardest"]:
        return "MIXED"
    return "FAIL"


def plot_seed(seed: int, by_mu: dict, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    colors = {"low": "#5b8def", "mid": "#c47b16", "high": "#b42318"}
    for label in ("low", "mid", "high"):
        bits = by_mu[label]["stats"]["bits"]
        ax.step(FORCES, bits, where="mid", color=colors[label], lw=2.0, label=f"{label} μ_eff={by_mu[label]['mu_eff']}")
        ax.scatter(FORCES, bits, color=colors[label], s=18, zorder=3)
    ax.set_xlabel("commanded F (N)")
    ax.set_ylabel("y_retention")
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"seed {seed} pre-pour retention vs F")
    fig.tight_layout()
    (HERE / "plots").mkdir(exist_ok=True)
    fig.savefig(HERE / "plots" / f"{tag}_retention_vs_F.png", dpi=140)
    plt.close(fig)


def plot_vs_fulldump(by_mu: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    colors = {"low": "#5b8def", "mid": "#c47b16", "high": "#b42318"}
    for ax, src, title in (
        (axes[0], {k: FULL_DUMP[k] for k in ("low", "mid", "high")}, "full dump (old)"),
        (axes[1], {k: by_mu[k]["stats"]["bits"] for k in ("low", "mid", "high")}, "pre-pour segment"),
    ):
        for label in ("low", "mid", "high"):
            ax.step(FORCES, src[label], where="mid", color=colors[label], lw=2.0, label=label)
        ax.set_title(title)
        ax.set_xlabel("F (N)")
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("retention")
    axes[1].legend()
    fig.suptitle("seed 200014 · full dump vs pre-pour (no wrist/pour)")
    fig.tight_layout()
    fig.savefig(HERE / "plots" / "compare_fulldump_vs_prepour.png", dpi=140)
    plt.close(fig)


def aggregate(seed_payloads: list[dict]) -> dict:
    by_mu = {"low": [], "mid": [], "high": []}
    for payload in seed_payloads:
        for label in ("low", "mid", "high"):
            by_mu[label].append(payload[label]["stats"]["bits"])
    out = {}
    for label, rows in by_mu.items():
        arr = np.asarray(rows, dtype=float)
        p = arr.mean(0).tolist()
        out[label] = {
            "P_retain": p,
            "rate_all": float(arr.mean()),
            "n_seeds": int(arr.shape[0]),
        }
    order = out["low"]["rate_all"] < out["mid"]["rate_all"] <= out["high"]["rate_all"]
    return {"by_mu": out, "aggregate_low_lt_mid_le_high": bool(order)}


def run_seed(seed: int) -> dict | None:
    by_mu = {}
    for label, finger in LEVELS.items():
        by_mu[label] = run_level(seed, label, float(finger))
        if by_mu[label].get("skipped"):
            log({"event": "seed_skip", "seed": seed, "mu": label})
            dump(HERE / f"SEED_{seed}.json", {"seed": seed, "skipped": True, "by_mu": {k: v.get("pre") for k, v in by_mu.items()}})
            return None
    flag = coulomb_like(by_mu)
    plot_seed(seed, by_mu, f"seed{seed}")
    payload = {
        "seed": seed,
        "low": {"pre": by_mu["low"]["pre"], "stats": by_mu["low"]["stats"], "mu_eff": by_mu["low"]["mu_eff"], "max_wrist": by_mu["low"]["max_wrist_deg_all_F"]},
        "mid": {"pre": by_mu["mid"]["pre"], "stats": by_mu["mid"]["stats"], "mu_eff": by_mu["mid"]["mu_eff"], "max_wrist": by_mu["mid"]["max_wrist_deg_all_F"]},
        "high": {"pre": by_mu["high"]["pre"], "stats": by_mu["high"]["stats"], "mu_eff": by_mu["high"]["mu_eff"], "max_wrist": by_mu["high"]["max_wrist_deg_all_F"]},
        "coulomb": flag,
        "bits": {k: by_mu[k]["stats"]["bits"] for k in ("low", "mid", "high")},
    }
    dump(HERE / f"SEED_{seed}.json", payload)
    return payload


def rescore_level(payload: dict) -> dict:
    for row in payload["outcomes"]:
        hold_n = int((row.get("hold") or {}).get("n") or 0)
        if hold_n == 0:
            row["hold_ok"] = bool(row.get("lift_ok") and row.get("translation_ok") and row.get("final_ok"))
        row["y_retention"] = int(
            bool(row.get("lift_ok")) and bool(row.get("hold_ok")) and bool(row.get("translation_ok")) and bool(row.get("final_ok"))
        )
    bits = [o["y_retention"] for o in payload["outcomes"]]
    payload["stats"] = curve_stats(FORCES, bits)
    return payload


def load_or_run_seed(seed: int, reuse: bool) -> dict | None:
    paths = [HERE / f"SWEEP_{seed}_{label}.json" for label in ("low", "mid", "high")]
    if reuse and all(p.exists() for p in paths):
        by_mu = {}
        for label, path in zip(("low", "mid", "high"), paths):
            by_mu[label] = rescore_level(json.loads(path.read_text()))
            dump(path, by_mu[label])
        flag = coulomb_like(by_mu)
        plot_seed(seed, by_mu, f"seed{seed}")
        payload = {
            "seed": seed,
            "low": {"pre": by_mu["low"]["pre"], "stats": by_mu["low"]["stats"], "mu_eff": by_mu["low"]["mu_eff"], "max_wrist": by_mu["low"]["max_wrist_deg_all_F"]},
            "mid": {"pre": by_mu["mid"]["pre"], "stats": by_mu["mid"]["stats"], "mu_eff": by_mu["mid"]["mu_eff"], "max_wrist": by_mu["mid"]["max_wrist_deg_all_F"]},
            "high": {"pre": by_mu["high"]["pre"], "stats": by_mu["high"]["stats"], "mu_eff": by_mu["high"]["mu_eff"], "max_wrist": by_mu["high"]["max_wrist_deg_all_F"]},
            "coulomb": flag,
            "bits": {k: by_mu[k]["stats"]["bits"] for k in ("low", "mid", "high")},
            "rescored_empty_hold": True,
        }
        dump(HERE / f"SEED_{seed}.json", payload)
        log({"event": "rescored", "seed": seed, "bits": payload["bits"], "flag": flag})
        return payload
    return run_seed(seed)


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    first = load_or_run_seed(SEED0, reuse=True)
    if first is None:
        raise RuntimeError("seed 200014 was not PRE_VALID on all three μ")
    flag = first["coulomb"]
    extras = []
    ran_multi = False
    if flag["clear_coulomb"]:
        ran_multi = True
        for seed in EXTRA_SEEDS:
            log({"event": "multi_seed", "seed": seed})
            row = run_seed(seed)
            if row is not None:
                extras.append(row)
    else:
        log({"event": "skip_multiseed", "reason": "200014 Coulomb recovery not clear", "flag": flag})
    by_mu = {k: {"stats": first[k]["stats"], "mu_eff": first[k]["mu_eff"]} for k in ("low", "mid", "high")}
    plot_vs_fulldump(by_mu)
    verdict = classify(by_mu, flag)
    agg = None if not extras else aggregate([first] + extras)
    summary = {
        "paper_data": False,
        "seed": SEED0,
        "segment": PLAN["segment"],
        "verdict": verdict,
        "coulomb_200014": flag,
        "stats_200014": {k: first[k]["stats"] for k in ("low", "mid", "high")},
        "pre_200014": {k: first[k]["pre"] for k in ("low", "mid", "high")},
        "bits_200014": first["bits"],
        "full_dump_bits": FULL_DUMP,
        "full_dump_stats": {k: curve_stats(FORCES, FULL_DUMP[k]) for k in ("low", "mid", "high")},
        "ran_multiseed": ran_multi,
        "extra_seeds": [r["seed"] for r in extras],
        "aggregate": agg,
        "wrist_audit_deg": {k: first[k]["max_wrist"] for k in ("low", "mid", "high")},
    }
    dump(HERE / "SUMMARY.json", summary)
    dump(HERE / "VERDICT.json", {"verdict": verdict, "coulomb_200014": flag, "ran_multiseed": ran_multi})
    log({"event": "done", "verdict": verdict, "flag": flag})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        (HERE / "ERROR.txt").write_text(traceback.format_exc())
        raise
