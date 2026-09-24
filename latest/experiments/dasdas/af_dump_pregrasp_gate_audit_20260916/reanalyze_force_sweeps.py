"""Re-analyze existing dump force-sweeps under the frozen PRE gate.

No new rollouts. PRE class never uses F-success. v4 has no PRE nx+nx log:
true PRE_VALID cannot be assigned there. Open-gripper / qCR=0 is the only
F-independent miss detector on historical sweeps.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GATE = json.loads((HERE / "GATE_DEFINITION.json").read_text())
CUT = GATE["cutoffs"]
V4 = Path("/media/volume/dasdas/exouser/af_dump_liftstyle_feas_v4_relabel_20260915")
WALL = Path("/media/volume/dasdas/exouser/af_dump_force_realize_20260915/contact_attr/wall_pinch/repeats")
TRAIN = json.loads((V4 / "TRAIN_RESULTS.json").read_text())
RETENTION_CR = 0.80
TRACK_RATIO = 0.40


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def is_nondecreasing(bits: list[int]) -> bool:
    seen = 0
    for val in bits:
        if int(val) < seen:
            return False
        seen = max(seen, int(val))
    return True


def fmin_first(forces: list[float], bits: list[int]) -> float | None:
    for force, ok in zip(forces, bits):
        if int(ok):
            return float(force)
    return None


def fmin_stable(forces: list[float], bits: list[int]) -> float | None:
    if not is_nondecreasing(bits) or not any(int(x) for x in bits):
        return None
    return fmin_first(forces, bits)


def spearman(xs: list[float], ys: list[float]) -> dict:
    if len(xs) < 2:
        return {"n": len(xs), "rho": None}
    rho = float(np.corrcoef(np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys)))[0, 1])
    return {"n": len(xs), "rho": None if np.isnan(rho) else rho}


def residual_reason(
    forces: list[float],
    retention: list[int],
    success: list[int],
    contact: list[float],
    measured: list[float],
) -> dict:
    n = len(forces)
    suc = [int(x) for x in success]
    ret = [int(x) for x in retention]
    cr = [float(x) if x is not None else 0.0 for x in contact]
    meas = [float(x) if x is not None else 0.0 for x in measured]
    track = [meas[i] / forces[i] if forces[i] > 0 else 0.0 for i in range(n)]
    suc_nd = is_nondecreasing(suc)
    ret_nd = is_nondecreasing(ret)
    if suc_nd and ret_nd:
        if not any(suc) and not any(ret):
            return {
                "reason": "all-F-fail",
                "detail": "no retention and no full-task success at any F",
                "secondary": [],
            }
        return {"reason": "none", "detail": "monotonic retention and full-task", "secondary": []}

    suc_ones = [i for i, v in enumerate(suc) if v]
    suc_drop = [i for i in range(1, n) if suc[i] < suc[i - 1]]
    ret_drop = [i for i in range(1, n) if ret[i] < ret[i - 1]]
    interleaved = bool(suc_ones) and any(suc[i] == 0 and i > min(suc_ones) and i < max(suc_ones) for i in range(n))
    fail_hi_cr = [i for i in range(n) if suc[i] == 0 and cr[i] >= RETENTION_CR]
    eject = [
        i
        for i in suc_drop
        if suc_ones
        and i > min(suc_ones)
        and cr[i] < 0.30
        and max(cr[:i] or [0.0]) >= RETENTION_CR
    ]
    flicker = False
    if n >= 4:
        crossings = sum(1 for i in range(1, n) if (cr[i] >= RETENTION_CR) != (cr[i - 1] >= RETENTION_CR))
        flicker = crossings >= 4
    track_fail = sum(
        1
        for i in range(n)
        if suc[i] == 0 and cr[i] >= RETENTION_CR and track[i] < TRACK_RATIO and forces[i] >= 1.0
    )
    later_eject = bool(eject) and (not interleaved) and bool(suc_drop)
    dump_gap = int(sum(ret) - sum(suc)) >= 4 and bool(fail_hi_cr)

    secondary: list[str] = []
    if flicker:
        secondary.append("contact flicker")
    if track_fail >= 3:
        secondary.append("controller tracking mismatch")
    if fail_hi_cr:
        secondary.append("downstream dump failure while grasp remains valid")
    if later_eject:
        secondary.append("high-force squeeze-induced ejection")
    if interleaved and fail_hi_cr:
        secondary.append("geometric wedging")

    if dump_gap:
        reason = "downstream dump failure while grasp remains valid"
        detail = (
            f"retention on {int(sum(ret))}/20 F, full-task only {int(sum(suc))}/20; "
            f"{len(fail_hi_cr)} fails still have contact_ratio>={RETENTION_CR:.1f}"
        )
    elif later_eject and not interleaved:
        reason = "high-force squeeze-induced ejection"
        detail = (
            f"success drops at F={forces[eject[0]]:.2f} N while contact_ratio falls to {cr[eject[0]]:.2f}"
        )
    elif interleaved and fail_hi_cr:
        reason = "geometric wedging"
        detail = "interleaved full-task 0/1 while remainder contact often stays high"
    elif fail_hi_cr and not later_eject:
        reason = "downstream dump failure while grasp remains valid"
        n_fail = len(fail_hi_cr)
        detail = f"{n_fail} F-points fail the dump while contact_ratio>={RETENTION_CR:.1f}"
    elif flicker or (interleaved and suc == ret):
        reason = "contact flicker"
        detail = (
            "remainder contact_ratio crosses the 0.8 retention line repeatedly vs F"
            if flicker
            else "full-task equals retention and both have a mid-F hole"
        )
    elif track_fail >= 3:
        reason = "controller tracking mismatch"
        detail = f"{track_fail} failed high-contact F>=1 N points have measured/commanded < {TRACK_RATIO}"
    else:
        reason = "unknown"
        detail = "nonmonotonic, no single heuristic dominates"
    # keep unique secondary excluding primary
    secondary = [x for x in dict.fromkeys(secondary) if x != reason]
    return {"reason": reason, "detail": detail, "secondary": secondary}


def summarize_subset(rows: list[dict], forces_key: str = "forces") -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    n_all_fail_task = sum(int(r["all_F_fail_task"]) for r in rows)
    n_all_fail_ret = sum(int(r["all_F_fail_retention"]) for r in rows)
    n_qcr0 = sum(int(float(r["query_contact_ratio"] or 0.0) <= 1e-12) for r in rows)
    n_mono_ret = sum(int(r["monotonic_retention"]) for r in rows)
    n_mono_task = sum(int(r["monotonic_full_task"]) for r in rows)
    fmin_ret = [(r["mu"], r["F_min_retention"]) for r in rows if r["F_min_retention_stable"] is not None]
    fmin_task = [(r["mu"], r["F_min_success_stable"]) for r in rows if r["F_min_success_stable"] is not None]
    return {
        "n": n,
        "all_F_fail_retention": n_all_fail_ret,
        "all_F_fail_task": n_all_fail_task,
        "qCR0": n_qcr0,
        "monotonic_retention": n_mono_ret,
        "monotonic_retention_frac": n_mono_ret / n,
        "monotonic_full_task": n_mono_task,
        "monotonic_full_task_frac": n_mono_task / n,
        "spearman_mu_Fmin_retention_stable": spearman([a for a, _ in fmin_ret], [b for _, b in fmin_ret]),
        "spearman_mu_Fmin_success_stable": spearman([a for a, _ in fmin_task], [b for _, b in fmin_task]),
    }


def load_v4() -> list[dict]:
    selected = {row["context_id"]: row for row in TRAIN["selections"]}
    rows = []
    for path in sorted((V4 / "main_records").glob("liftstyle_mu*_seed*.json")):
        rec = json.loads(path.read_text())
        ctx = rec["context"]
        outcomes = rec["outcomes"]
        forces = [float(o["force_N"]) for o in outcomes]
        success = [int(o.get("full_task_success_y", o.get("success", 0))) for o in outcomes]
        success_alias = [int(o.get("success", 0)) for o in outcomes]
        contact = [float(o.get("contact_ratio") or 0.0) for o in outcomes]
        measured = [float(o.get("measured_force_mean_n") or 0.0) for o in outcomes]
        qcr = float(outcomes[0]["query_contact_ratio"])
        retention = [int(c >= RETENTION_CR) for c in contact]
        sid = str(ctx["id"])
        sel = selected.get(sid, {})
        residual = residual_reason(forces, retention, success, contact, measured)
        open_miss = qcr <= 1e-12
        # True PRE nx+nx is not logged. qCR=0 is the only F-independent miss flag.
        if open_miss:
            pre_class = "PRE_INVALID"
            pre_evidence = "qCR=0 (no PRE window in v4; calibrated miss detector)"
        else:
            pre_class = "PRE_UNLOGGED"
            pre_evidence = "v4 has no PRE nx+nx log; qCR>0 so cannot mark PRE_INVALID"
        rows.append(
            {
                "corpus": "v4_32x20",
                "root": sid,
                "seed": int(ctx["seed"]),
                "mu": float(ctx["friction"]),
                "split": sel.get("split") or {200002: "TRAIN", 200003: "TRAIN", 200010: "VAL", 200014: "TEST"}[int(ctx["seed"])],
                "pre_class": pre_class,
                "pre_evidence": pre_evidence,
                "query_contact_ratio": qcr,
                "forces": forces,
                "retention_by_F": retention,
                "full_task_by_F": success,
                "success_equals_full_task": success == success_alias,
                "contact_ratio_by_F": contact,
                "measured_force_mean_n_by_F": measured,
                "monotonic_retention": is_nondecreasing(retention),
                "monotonic_full_task": is_nondecreasing(success),
                "F_min_retention": fmin_first(forces, retention),
                "F_min_success": fmin_first(forces, success),
                "F_min_retention_stable": fmin_stable(forces, retention),
                "F_min_success_stable": fmin_stable(forces, success),
                "all_F_fail_retention": not any(retention),
                "all_F_fail_task": not any(success),
                "all_F_fail": not any(success),
                "selected_force_N": sel.get("selected_force_N"),
                "n_retention": int(sum(retention)),
                "n_success": int(sum(success)),
                "mean_measured_squeeze_proxy_n": float(np.mean(measured)),
                "failure_reason": residual["reason"],
                "failure_detail": residual["detail"],
                "failure_secondary": residual["secondary"],
            }
        )
    return rows


def wall_handoff(repeat: int) -> dict:
    path = WALL / f"r{repeat}_mu0.425_F0.5" / "steps.jsonl"
    settle = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("phase") == "settle":
                settle.append(row)
            if len(settle) >= 20:
                break
    nx = [
        bool((row.get("left") or {}).get("slab") == "grasp_nx" and (row.get("right") or {}).get("slab") == "grasp_nx")
        for row in settle
    ]
    bi = [bool(row.get("bilateral")) for row in settle]
    ap = settle[0].get("aperture") if settle else None
    sq = settle[0].get("squeeze") if settle else None
    return {
        "handoff_aperture_m": ap,
        "handoff_squeeze": sq,
        "handoff_nxnx_window": float(np.mean(nx)) if nx else 0.0,
        "handoff_bilateral_window": float(np.mean(bi)) if bi else 0.0,
    }


def load_wall() -> list[dict]:
    rows = []
    for ctx_path in sorted(WALL.glob("r*_mu0.425_CONTEXT.json")):
        ctx = json.loads(ctx_path.read_text())
        ep = int(ctx["episode_id"])
        hand = wall_handoff(ep)
        outcomes = ctx["outcomes"]
        forces = [float(o["force_N"]) for o in outcomes]
        retention = [int(o.get("retention_success", o.get("retention", 0))) for o in outcomes]
        # Isolation grid has no separate official dump success; retention is the task label.
        success = list(retention)
        contact = [float(o.get("contact_ratio") or 0.0) for o in outcomes]
        measured = [float(o.get("measured_force_mean_n") or 0.0) for o in outcomes]
        squeeze = [float(o.get("settle_squeeze") or 0.0) for o in outcomes]
        qcr = float(ctx.get("query_contact_ratio") or 0.0)
        ap = float(hand["handoff_aperture_m"] or 0.0)
        sq = float(hand["handoff_squeeze"] or 0.0)
        # F-independent miss: open gripper at the shared post-query snapshot.
        # Query does not open living pinches to ~45 mm (probe max pair 12 mm).
        if ap >= CUT["aperture_miss_m"] and sq <= CUT["squeeze_alive_n"] and qcr <= 1e-12:
            pre_class = "PRE_INVALID"
            pre_evidence = f"open gripper {ap*1000:.1f} mm, squeeze≈0, qCR=0 before F sweep"
        elif ap < CUT["aperture_miss_m"] and qcr > 0:
            pre_class = "PRE_VALID_PROXY"
            pre_evidence = "no PRE nx+nx log; closed aperture + qCR>0, not a 45 mm miss"
        else:
            pre_class = "PRE_BORDERLINE"
            pre_evidence = f"mixed handoff: ap={ap*1000:.1f} mm qCR={qcr:.2f} nx-window={hand['handoff_nxnx_window']:.2f}"
        residual = residual_reason(forces, retention, success, contact, measured)
        if pre_class == "PRE_INVALID":
            residual = {
                "reason": "grasp-establishment failure",
                "detail": pre_evidence,
                "secondary": [],
            }
        rows.append(
            {
                "corpus": "wall_fgrid_5x4",
                "root": f"wall_mu0.425_ep{ep}",
                "seed": 200014,
                "mu": 0.425,
                "split": "ISOLATION",
                "pre_class": pre_class,
                "pre_evidence": pre_evidence,
                "query_contact_ratio": qcr,
                "forces": forces,
                "retention_by_F": retention,
                "full_task_by_F": success,
                "success_equals_full_task": True,
                "contact_ratio_by_F": contact,
                "measured_force_mean_n_by_F": measured,
                "settle_squeeze_by_F": squeeze,
                "handoff_aperture_m": ap,
                "handoff_squeeze": sq,
                "monotonic_retention": is_nondecreasing(retention),
                "monotonic_full_task": is_nondecreasing(success),
                "F_min_retention": fmin_first(forces, retention),
                "F_min_success": fmin_first(forces, success),
                "F_min_retention_stable": fmin_stable(forces, retention),
                "F_min_success_stable": fmin_stable(forces, success),
                "all_F_fail_retention": not any(retention),
                "all_F_fail_task": not any(success),
                "all_F_fail": not any(success),
                "selected_force_N": None,
                "n_retention": int(sum(retention)),
                "n_success": int(sum(success)),
                "mean_measured_squeeze_proxy_n": float(np.mean(squeeze)),
                "failure_reason": residual["reason"],
                "failure_detail": residual["detail"],
                "failure_secondary": residual["secondary"],
            }
        )
    return rows


def curve_pct(rows: list[dict], key: str) -> dict[str, float]:
    if not rows:
        return {}
    forces = rows[0]["forces"]
    out = {}
    for i, force in enumerate(forces):
        out[f"{force:g}"] = 100.0 * float(np.mean([r[key][i] for r in rows]))
    return out


def main() -> None:
    v4 = load_v4()
    wall = load_wall()
    assert all(r["success_equals_full_task"] for r in v4)
    v4_invalid = [r for r in v4 if r["pre_class"] == "PRE_INVALID"]
    v4_keep = [r for r in v4 if r["pre_class"] != "PRE_INVALID"]
    wall_invalid = [r for r in wall if r["pre_class"] == "PRE_INVALID"]
    wall_keep = [r for r in wall if r["pre_class"] != "PRE_INVALID"]
    wall_border = [r for r in wall if r["pre_class"] == "PRE_BORDERLINE"]

    v4_all_fail = [r for r in v4 if r["all_F_fail_task"]]
    v4_qcr0 = [r for r in v4 if r["query_contact_ratio"] <= 1e-12]
    wall_all_fail = [r for r in wall if r["all_F_fail_task"]]
    wall_qcr0 = [r for r in wall if r["query_contact_ratio"] <= 1e-12]

    residual_counts = {}
    for r in v4_keep:
        residual_counts[r["failure_reason"]] = residual_counts.get(r["failure_reason"], 0) + 1

    payload = {
        "gate": {
            "note": "True PRE_VALID needs PRE-window nx+nx. Official 32x20 never logged that. PRE_INVALID on sweeps is only the F-independent open-gripper / qCR=0 miss. PRE_UNLOGGED is not PRE_VALID.",
            "retention_rule_v4": "remainder contact_ratio >= 0.80 (same cutoff as isolation retention)",
            "full_task_v4": "full_task_success_y == success on all 640 rows",
            "cutoffs": CUT,
        },
        "contamination": {
            "v4_all_F_fail": len(v4_all_fail),
            "v4_qCR0": len(v4_qcr0),
            "v4_all_F_fail_from_PRE_INVALID": sum(r["pre_class"] == "PRE_INVALID" for r in v4_all_fail),
            "v4_qCR0_from_PRE_INVALID": sum(r["pre_class"] == "PRE_INVALID" for r in v4_qcr0),
            "wall_all_F_fail": len(wall_all_fail),
            "wall_qCR0": len(wall_qcr0),
            "wall_all_F_fail_from_PRE_INVALID": sum(r["pre_class"] == "PRE_INVALID" for r in wall_all_fail),
            "wall_qCR0_from_PRE_INVALID": sum(r["pre_class"] == "PRE_INVALID" for r in wall_qcr0),
        },
        "v4_before": summarize_subset(v4),
        "v4_after_drop_PRE_INVALID": summarize_subset(v4_keep),
        "v4_PRE_INVALID": summarize_subset(v4_invalid),
        "v4_success_pct_by_F_all": curve_pct(v4, "full_task_by_F"),
        "v4_retention_pct_by_F_all": curve_pct(v4, "retention_by_F"),
        "v4_success_pct_by_F_keep": curve_pct(v4_keep, "full_task_by_F"),
        "v4_retention_pct_by_F_keep": curve_pct(v4_keep, "retention_by_F"),
        "wall_before": summarize_subset(wall),
        "wall_after_drop_PRE_INVALID": summarize_subset(wall_keep),
        "wall_PRE_INVALID": summarize_subset(wall_invalid),
        "wall_PRE_BORDERLINE": summarize_subset(wall_border),
        "wall_retention_pct_by_F_all": curve_pct(wall, "retention_by_F"),
        "wall_retention_pct_by_F_keep": curve_pct(wall_keep, "retention_by_F"),
        "v4_residual_counts_keep": residual_counts,
        "questions": {
            "q1_never_grasped_share": {
                "v4": "0/32 all-F-fail and 0/32 qCR=0. Official 32x20 anomalies are not open-gripper misses.",
                "wall": f"{len(wall_all_fail)}/{len(wall)} all-F-fail; {sum(r['pre_class']=='PRE_INVALID' for r in wall_all_fail)}/{len(wall_all_fail) or 1} of those are PRE_INVALID open-gripper misses.",
            },
            "q2_more_reasonable_after_filter": {
                "v4": "No change: nothing to drop. full-task monotonic 10/32, retention monotonic stays whatever the CR>=0.8 curve is.",
                "wall": "Yes on this tiny grid: all-F-fail 2/5 -> 0/3; retention 40% -> 67%.",
            },
            "q3_next_is_outer_wall_mu_isolation": True,
        },
        "rows": v4 + wall,
    }
    write(HERE / "FORCE_SWEEP_REGATE.json", payload)

    table = []
    for r in v4 + wall:
        table.append(
            {
                "root": r["root"],
                "mu": r["mu"],
                "PRE_class": r["pre_class"],
                "monotonic_retention": r["monotonic_retention"],
                "monotonic_full_task": r["monotonic_full_task"],
                "F_min_retention": r["F_min_retention"],
                "F_min_success": r["F_min_success"],
                "all_F_fail": r["all_F_fail"],
                "failure_reason": r["failure_reason"],
            }
        )
    write(HERE / "FORCE_SWEEP_TABLE.json", {"rows": table})
    print(
        json.dumps(
            {
                "v4_n": len(v4),
                "v4_invalid": len(v4_invalid),
                "v4_mono_task": payload["v4_before"]["monotonic_full_task"],
                "v4_mono_ret": payload["v4_before"]["monotonic_retention"],
                "v4_qcr0": payload["v4_before"]["qCR0"],
                "wall_n": len(wall),
                "wall_invalid": len(wall_invalid),
                "wall_keep_mono_ret": payload["wall_after_drop_PRE_INVALID"].get("monotonic_retention"),
                "residuals": residual_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
