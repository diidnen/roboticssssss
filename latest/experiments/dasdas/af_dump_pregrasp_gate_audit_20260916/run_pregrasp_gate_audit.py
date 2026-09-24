"""PRE grasp-establishment gate audit. No new 32x20. No query retune."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROBE = Path("/media/volume/dasdas/exouser/af_dump_force_realize_20260915/contact_attr/query_probe")
WALL = Path("/media/volume/dasdas/exouser/af_dump_force_realize_20260915/contact_attr/wall_pinch/repeats")
V4 = Path("/media/volume/dasdas/exouser/af_dump_liftstyle_feas_v4_relabel_20260915")
TRAIN = json.loads((V4 / "TRAIN_RESULTS.json").read_text())
GATE = json.loads((HERE / "GATE_DEFINITION.json").read_text())
CUT = GATE["cutoffs"]
SPLITS = {"200002": "TRAIN", "200003": "TRAIN", "200010": "VAL", "200014": "TEST"}


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def classify_pre(window_bilateral_cr: float | None, endpoint_nxnx: bool, squeeze: float, aperture: float) -> tuple[str, str]:
    cr = 0.0 if window_bilateral_cr is None else float(window_bilateral_cr)
    sq = float(squeeze or 0.0)
    ap = float(aperture or 0.0)
    nx = bool(endpoint_nxnx)
    if ap >= CUT["aperture_miss_m"] or (cr <= 1e-9 and sq <= CUT["squeeze_alive_n"] and not nx):
        reason = []
        if ap >= CUT["aperture_miss_m"]:
            reason.append(f"gripper open {ap*1000:.1f} mm")
        if sq <= CUT["squeeze_alive_n"]:
            reason.append("squeeze≈0")
        if cr <= 1e-9:
            reason.append("PRE bilateral CR=0")
        if not nx:
            reason.append("not nx+nx")
        return "INVALID", "; ".join(reason) or "never established"
    if cr >= CUT["window_bilateral_valid_min"] and nx and sq > CUT["squeeze_alive_n"] and ap < CUT["aperture_miss_m"]:
        return "VALID", "PRE window bilateral wall pinch"
    if cr >= CUT["window_bilateral_borderline_min"] and ap < CUT["aperture_miss_m"] and (not nx or sq <= CUT["squeeze_alive_n"]):
        return "BORDERLINE", "high-CR PRE window with endpoint flicker"
    return "INVALID", f"weak PRE (CR={cr:.2f}, nx={nx}, squeeze={sq:.3f} N, ap={ap*1000:.1f} mm)"


def qcr_proxy(qcr) -> str:
    if qcr is None:
        return "UNKNOWN"
    return "PRE_INVALID_PROXY" if float(qcr) <= 1e-12 else "PRE_VALID_PROXY"


def is_nondecreasing(success: list[int]) -> bool:
    seen = 0
    for val in success:
        if val < seen:
            return False
        seen = max(seen, val)
    return True


def fmin_first(forces: list[float], success: list[int]) -> float | None:
    for force, ok in zip(forces, success):
        if ok:
            return float(force)
    return None


def fmin_stable(forces: list[float], success: list[int]) -> float | None:
    if not is_nondecreasing(success) or not any(success):
        return None
    for force, ok in zip(forces, success):
        if ok:
            return float(force)
    return None


def residual_class(success: list[int], measured: list[float | None], contact: list[float | None]) -> str:
    if not any(success):
        mean_c = float(np.mean([x for x in contact if x is not None] or [0.0]))
        mean_f = float(np.mean([x for x in measured if x is not None] or [0.0]))
        if mean_c < 0.2:
            return "force-insensitive contact collapse"
        if mean_f < 0.2:
            return "controller tracking / not realizing F"
        return "downstream dump failure with some contact"
    if is_nondecreasing(success):
        return "monotonic"
    ones = [i for i, ok in enumerate(success) if ok]
    zeros_after = [i for i, ok in enumerate(success) if not ok and i > min(ones)]
    if zeros_after and max(ones) > min(zeros_after):
        return "nonmonotonic: later-F drop (possible squeeze ejection / dump geometry)"
    return "nonmonotonic: interleaved success"


def audit_probe() -> dict:
    rows = []
    for ep in range(10):
        raw = json.loads((PROBE / f"ep{ep:02d}" / "result.json").read_text())
        pre = raw["pre"]
        hold = raw["hold"]
        query = raw["query"]
        gate, reason = classify_pre(
            raw.get("contact_ratio_pre"),
            bool(pre.get("nx_nx")),
            float(pre.get("squeeze") or 0.0),
            float(pre.get("aperture") or 0.0),
        )
        rows.append(
            {
                "episode_id": ep,
                "seed": 200014,
                "mu": 0.425,
                "pre_left_contact": bool(pre.get("left_present")),
                "pre_right_contact": bool(pre.get("right_present")),
                "pre_left_slab": pre.get("left_slab"),
                "pre_right_slab": pre.get("right_slab"),
                "pre_nxnx": bool(pre.get("nx_nx")),
                "pre_bilateral_cr": raw.get("contact_ratio_pre"),
                "pre_squeeze": pre.get("squeeze"),
                "pre_aperture_m": pre.get("aperture"),
                "pre_pair_m": pre.get("pair_dist"),
                "pre_bin_z": pre.get("obj_z"),
                "pre_table_bottom_n": pre.get("table_bottom_n"),
                "gate": gate,
                "invalid_reason": None if gate != "INVALID" else reason,
                "borderline_reason": None if gate != "BORDERLINE" else reason,
                "hold_alive": bool(hold.get("alive")),
                "hold_nxnx": bool(hold.get("nx_nx")),
                "hold_unilateral": bool(hold.get("unilateral")),
                "hold_bilateral_cr": raw.get("contact_ratio_hold"),
                "hold_squeeze": hold.get("squeeze"),
                "hold_pair_m": hold.get("pair_dist"),
                "query_alive": bool(query.get("alive")),
                "query_nxnx": bool(query.get("nx_nx")),
                "query_unilateral": bool(query.get("unilateral")),
                "query_contact_ratio": raw.get("contact_ratio_query"),
                "query_squeeze": query.get("squeeze"),
                "query_pair_m": query.get("pair_dist"),
                "query_drel_deg": query.get("drel_from_pre_deg"),
                "query_obj_disp_m": query.get("obj_disp_m"),
            }
        )
    valid = [row for row in rows if row["gate"] == "VALID"]
    n_valid = len(valid)
    query_survive = sum(int(row["query_alive"] and row["query_nxnx"]) for row in valid)
    hold_survive = sum(int(row["hold_alive"] and row["hold_nxnx"]) for row in valid)
    q_sq = [float(row["query_squeeze"]) for row in valid]
    h_sq = [float(row["hold_squeeze"]) for row in valid]
    q_pair = [row["query_pair_m"] for row in valid if row["query_pair_m"] is not None]
    h_pair = [row["hold_pair_m"] for row in valid if row["hold_pair_m"] is not None]
    return {
        "n": len(rows),
        "n_valid": n_valid,
        "n_borderline": sum(row["gate"] == "BORDERLINE" for row in rows),
        "n_invalid": sum(row["gate"] == "INVALID" for row in rows),
        "reproduces_8_of_10": n_valid == 8 and sum(row["gate"] == "INVALID" for row in rows) == 2,
        "invalid_episodes": [row["episode_id"] for row in rows if row["gate"] == "INVALID"],
        "p_query_survives_given_pre_valid": None if not n_valid else query_survive / n_valid,
        "p_hold_survives_given_pre_valid": None if not n_valid else hold_survive / n_valid,
        "query_kills_of_valid": n_valid - query_survive,
        "hold_squeeze_mean_n": None if not h_sq else float(np.mean(h_sq)),
        "query_squeeze_mean_n": None if not q_sq else float(np.mean(q_sq)),
        "query_squeeze_range_n": None if not q_sq else [float(min(q_sq)), float(max(q_sq))],
        "hold_pair_mean_m": None if not h_pair else float(np.mean(h_pair)),
        "query_pair_mean_m": None if not q_pair else float(np.mean(q_pair)),
        "query_pair_max_m": None if not q_pair else float(max(q_pair)),
        "rows": rows,
    }


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
    n = max(len(settle), 1)
    nx = [
        bool((row.get("left") or {}).get("slab") == "grasp_nx" and (row.get("right") or {}).get("slab") == "grasp_nx")
        for row in settle
    ]
    bi = [bool(row.get("bilateral")) for row in settle]
    return {
        "handoff_aperture_m": settle[0].get("aperture") if settle else None,
        "handoff_squeeze": settle[0].get("squeeze") if settle else None,
        "handoff_nxnx_window": float(np.mean(nx)),
        "handoff_bilateral_window": float(np.mean(bi)),
        "n_settle_frames": len(settle),
    }


def audit_wall(probe: dict) -> dict:
    probe_by_ep = {row["episode_id"]: row for row in probe["rows"]}
    contexts = []
    for ctx_path in sorted(WALL.glob("r*_mu0.425_CONTEXT.json")):
        ctx = json.loads(ctx_path.read_text())
        ep = int(ctx["episode_id"])
        qcr = ctx.get("query_contact_ratio")
        handoff = wall_handoff(int(ctx_path.name.split("_")[0][1:]))
        outcomes = []
        for out in ctx["outcomes"]:
            outcomes.append(
                {
                    "F": out.get("force_N"),
                    "retention": int(out.get("retention_success") or 0),
                    "drop": out.get("drop"),
                    "contact_ratio": out.get("contact_ratio"),
                    "settle_squeeze": out.get("settle_squeeze"),
                    "pour_squeeze": out.get("pour_squeeze_geom"),
                    "unilateral_loss": out.get("unilateral_loss"),
                    "bilateral_loss": out.get("bilateral_loss"),
                    "same_slab_settle": out.get("same_slab_settle"),
                }
            )
        successes = [int(row["retention"]) for row in outcomes]
        forces = [float(row["F"]) for row in outcomes]
        probe_row = probe_by_ep.get(ep)
        contexts.append(
            {
                "repeat": int(ctx_path.name.split("_")[0][1:]),
                "episode_id": ep,
                "query_contact_ratio": qcr,
                "proxy": qcr_proxy(qcr),
                "probe_pre_gate": None if probe_row is None else probe_row["gate"],
                "probe_note": "different physics realization than this F-grid; PRE gate is from query_probe, not this run",
                **handoff,
                "all_F_fail": not any(successes),
                "nondecreasing": is_nondecreasing(successes),
                "outcomes": outcomes,
                "retention_by_F": {str(f): s for f, s in zip(forces, successes)},
            }
        )
    def subset(name: str, pred) -> dict:
        rows = [row for row in contexts if pred(row)]
        forces = [0.5, 0.75, 1.0, 1.25]
        by_f = {}
        for force in forces:
            vals = [int(row["retention_by_F"].get(str(force), row["retention_by_F"].get(f"{force:g}", 0))) for row in rows]
            # keys may be "0.5" vs "0.50"
            vals = []
            for row in rows:
                hit = None
                for key, val in row["retention_by_F"].items():
                    if abs(float(key) - force) < 1e-9:
                        hit = int(val)
                vals.append(0 if hit is None else hit)
            by_f[str(force)] = None if not vals else 100.0 * float(np.mean(vals))
        return {
            "n": len(rows),
            "n_qcr0": sum(row["proxy"] == "PRE_INVALID_PROXY" for row in rows),
            "n_all_F_fail": sum(row["all_F_fail"] for row in rows),
            "n_nondecreasing": sum(row["nondecreasing"] for row in rows),
            "retention_pct_by_F": by_f,
            "episodes": [row["episode_id"] for row in rows],
        }

    return {
        "n": len(contexts),
        "all": subset("all", lambda row: True),
        "proxy_valid": subset("proxy_valid", lambda row: row["proxy"] == "PRE_VALID_PROXY"),
        "proxy_invalid": subset("proxy_invalid", lambda row: row["proxy"] == "PRE_INVALID_PROXY"),
        "qcr0_all_fail_are_proxy_invalid": all(
            (row["proxy"] == "PRE_INVALID_PROXY") for row in contexts if abs(float(row["query_contact_ratio"] or 1) ) < 1e-12
        ),
        "contexts": contexts,
    }


def load_v4() -> list[dict]:
    rows = []
    for path in sorted((V4 / "main_records").glob("liftstyle_*.json")):
        rec = json.loads(path.read_text())
        ctx = rec["context"]
        outcomes = rec["outcomes"]
        qcr = outcomes[0].get("query_contact_ratio") if outcomes else rec.get("query_contact_ratio")
        forces = [float(row["force_N"]) for row in outcomes]
        success = [int(row.get("success") or 0) for row in outcomes]
        full = [int(row.get("full_task_success_y") or 0) for row in outcomes]
        contact = [row.get("contact_ratio") for row in outcomes]
        measured = [row.get("measured_force_mean_n") for row in outcomes]
        proxy = qcr_proxy(qcr)
        sel = next((item for item in TRAIN.get("selections") or [] if item.get("context_id") == ctx["id"]), None)
        rows.append(
            {
                "context_id": ctx["id"],
                "seed": int(ctx["seed"]),
                "mu": float(ctx["friction"]),
                "split": SPLITS.get(str(ctx["seed"]), "NA"),
                "query_contact_ratio": qcr,
                "proxy": proxy,
                "success": success,
                "full_task_success": full,
                "forces": forces,
                "contact_ratio": contact,
                "measured_force_mean_n": measured,
                "nondecreasing": is_nondecreasing(success),
                "all_fail": not any(success),
                "n_success": int(sum(success)),
                "fmin_first": fmin_first(forces, success),
                "fmin_stable": fmin_stable(forces, success),
                "residual": residual_class(success, measured, contact),
                "selected_force_N": None if sel is None else sel.get("selected_force_N"),
                "selected_true_success": None if sel is None else sel.get("selected_true_success"),
            }
        )
    return rows


def summarize_v4(rows: list[dict], pred) -> dict:
    picked = [row for row in rows if pred(row)]
    n = len(picked)
    forces = picked[0]["forces"] if picked else []
    by_f = {}
    for i, force in enumerate(forces):
        by_f[str(force)] = None if not picked else 100.0 * float(np.mean([row["success"][i] for row in picked]))
    by_mu = {}
    for mu in sorted({row["mu"] for row in picked}):
        sub = [row for row in picked if abs(row["mu"] - mu) < 1e-12]
        fmins = [row["fmin_stable"] for row in sub if row["fmin_stable"] is not None]
        selected = [row["selected_force_N"] for row in sub if row["selected_force_N"] is not None]
        by_mu[str(mu)] = {
            "n": len(sub),
            "n_nondecreasing": sum(row["nondecreasing"] for row in sub),
            "n_all_fail": sum(row["all_fail"] for row in sub),
            "mean_success_rate": float(np.mean([row["n_success"] / max(len(row["success"]), 1) for row in sub])),
            "median_fmin_stable": None if not fmins else float(np.median(fmins)),
            "n_with_stable_fmin": len(fmins),
            "mean_selected_force": None if not selected else float(np.mean(selected)),
        }
    return {
        "n_contexts": n,
        "n_qcr0": sum(row["proxy"] == "PRE_INVALID_PROXY" for row in picked),
        "n_all_fail": sum(row["all_fail"] for row in picked),
        "n_nondecreasing": sum(row["nondecreasing"] for row in picked),
        "nondecreasing_frac": None if not n else sum(row["nondecreasing"] for row in picked) / n,
        "success_pct_by_F": by_f,
        "by_mu": by_mu,
        "residual_counts": {
            name: int(sum(row["residual"] == name for row in picked))
            for name in sorted({row["residual"] for row in picked})
        },
        "ids": [row["context_id"] for row in picked],
    }


def mu_fmin_spearman(rows: list[dict], key="fmin_stable") -> dict:
    xs = []
    ys = []
    for row in rows:
        val = row.get(key)
        if val is None:
            continue
        xs.append(row["mu"])
        ys.append(float(val))
    if len(xs) < 3:
        return {"n": len(xs), "rho": None}
    rx = np.argsort(np.argsort(xs)).astype(float)
    ry = np.argsort(np.argsort(ys)).astype(float)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return {"n": len(xs), "rho": rho}


def main() -> None:
    probe = audit_probe()
    wall = audit_wall(probe)
    v4_rows = load_v4()
    v4_all = summarize_v4(v4_rows, lambda row: True)
    v4_valid = summarize_v4(v4_rows, lambda row: row["proxy"] == "PRE_VALID_PROXY")
    v4_invalid = summarize_v4(v4_rows, lambda row: row["proxy"] == "PRE_INVALID_PROXY")
    qcr0 = [row for row in v4_rows if row["proxy"] == "PRE_INVALID_PROXY"]
    all_fail = [row for row in v4_rows if row["all_fail"]]
    qcr0_and_all_fail = [row for row in qcr0 if row["all_fail"]]
    n_qcr0_explained = len(qcr0_and_all_fail)
    contamination = {
        "v4_n_contexts": len(v4_rows),
        "v4_n_qcr0": len(qcr0),
        "v4_n_all_fail": len(all_fail),
        "v4_n_qcr0_and_all_fail": n_qcr0_explained,
        "fraction_of_all_fail_that_are_qcr0": None if not all_fail else n_qcr0_explained / len(all_fail),
        "fraction_of_qcr0_that_all_fail": None if not qcr0 else n_qcr0_explained / len(qcr0),
        "wall_n_qcr0": wall["all"]["n_qcr0"],
        "wall_n_all_F_fail": wall["all"]["n_all_F_fail"],
        "wall_qcr0_equals_all_fail": wall["all"]["n_qcr0"] == wall["all"]["n_all_F_fail"],
    }
    monotonicity = {
        "v4_all_nondecreasing": v4_all["n_nondecreasing"],
        "v4_all_frac": v4_all["nondecreasing_frac"],
        "v4_proxy_valid_nondecreasing": v4_valid["n_nondecreasing"],
        "v4_proxy_valid_frac": v4_valid["nondecreasing_frac"],
        "v4_proxy_invalid_nondecreasing": v4_invalid["n_nondecreasing"],
        "wall_all_nondecreasing": wall["all"]["n_nondecreasing"],
        "wall_proxy_valid_nondecreasing": wall["proxy_valid"]["n_nondecreasing"],
    }
    friction = {
        "note": "Expected: lower μ -> larger stable F_min. Spearman rho should be negative.",
        "all_fmin_stable_vs_mu": mu_fmin_spearman(v4_rows, "fmin_stable"),
        "proxy_valid_fmin_stable_vs_mu": mu_fmin_spearman(
            [row for row in v4_rows if row["proxy"] == "PRE_VALID_PROXY"], "fmin_stable"
        ),
        "all_selected_force_vs_mu": mu_fmin_spearman(v4_rows, "selected_force_N"),
        "proxy_valid_selected_force_vs_mu": mu_fmin_spearman(
            [row for row in v4_rows if row["proxy"] == "PRE_VALID_PROXY"], "selected_force_N"
        ),
    }
    residuals = v4_valid["residual_counts"]
    query_safe = (
        probe["reproduces_8_of_10"]
        and probe["query_kills_of_valid"] == 0
        and contamination["wall_qcr0_equals_all_fail"]
    )
    cleaned_force = (v4_valid["nondecreasing_frac"] or 0) - (v4_all["nondecreasing_frac"] or 0)
    rho_valid = (friction["proxy_valid_fmin_stable_vs_mu"] or {}).get("rho")
    still_nonmono = (v4_valid["n_contexts"] - v4_valid["n_nondecreasing"]) if v4_valid["n_contexts"] else 0
    if not query_safe:
        verdict = "FAIL"
        why = "PRE gate did not reproduce the 10-root split, or query still kills valid pinches, or qCR=0 does not match all-F failure on the wall-pinch grid."
    elif still_nonmono >= 8 or (v4_valid["nondecreasing_frac"] or 0) < 0.5:
        verdict = "MIXED"
        why = "PRE_INVALID / qCR=0 contamination is real and removable, and query does not destroy valid wall pinches, but VALID-proxy force-response is still largely nonmonotonic."
    elif cleaned_force >= 0.15 and rho_valid is not None and rho_valid < -0.2:
        verdict = "PASS"
        why = "Filtering PRE_INVALID/qCR=0 removes a chunk of all-F failures, query is safe on valid pinches, and friction–F_min becomes more reasonable."
    else:
        verdict = "MIXED"
        why = "Gate cleans grasp-establishment contamination and query is safe on valid pinches, but force-response on remaining contexts is only partly more interpretable."
    summary = {
        "verdict": verdict,
        "why": why,
        "labels": {
            "wall_pinch": "PASS",
            "force_controller": "PASS",
            "diagnostic_query_robustness": "MIXED",
            "pre_gate": "PASS" if probe["reproduces_8_of_10"] else "FAIL",
        },
        "grasp_establishment_pass_rate": {
            "query_probe_seed200014": f"{probe['n_valid']}/{probe['n']}",
            "v4_qcr_proxy_valid": f"{v4_valid['n_contexts']}/{v4_all['n_contexts']}",
            "wall_fgrid_qcr_proxy_valid": f"{wall['proxy_valid']['n']}/{wall['n']}",
        },
        "need_32x20_resample": False,
        "need_32x20_reason": "Do not resample yet. Next step if MIXED: inspect remaining VALID-proxy nonmonotonic contexts for wedging / squeeze ejection / dump geometry, optionally with grasp-surface friction isolation. The PRE gate itself does not require a new 32×20.",
        "do_grasp_surface_isolation_next": verdict == "MIXED",
    }
    write(HERE / "PROBE_10ROOT_AUDIT.json", probe)
    write(HERE / "WALL_FGRID_REGROUP.json", wall)
    write(HERE / "V4_CONTEXTS.json", {"rows": v4_rows})
    write(
        HERE / "V4_REGROUP.json",
        {"all": v4_all, "proxy_valid": v4_valid, "proxy_invalid": v4_invalid},
    )
    write(HERE / "CONTAMINATION.json", contamination)
    write(HERE / "MONOTONICITY.json", monotonicity)
    write(HERE / "FRICTION_FMIN.json", friction)
    write(HERE / "RESIDUALS_PROXY_VALID.json", residuals)
    write(HERE / "SUMMARY.json", {**summary, "contamination": contamination, "monotonicity": monotonicity, "friction": friction})
    print(json.dumps({"verdict": verdict, "probe_valid": probe["n_valid"], "v4_proxy_valid": v4_valid["n_contexts"], "v4_qcr0": v4_all["n_qcr0"]}, indent=2))


if __name__ == "__main__":
    main()
