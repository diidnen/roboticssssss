#!/usr/bin/env python3
"""D2: aggregate confirmation + write FINAL_VERDICT.json and README."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
D1 = Path("/home/exouser/Tabero/analysis/results/d1_force_decision_resolution_20260819_223557")
FSTAR = {0.2: 6.0, 0.5: 5.0, 1.0: 4.0}


def _f(x, d=np.nan):
    try:
        if x in ("", "None", None):
            return d
        return float(x)
    except Exception:
        return d


def load_confirm():
    p = OUT / "FULLTASK_CONFIRMATION.csv"
    if not p.exists() or p.stat().st_size == 0:
        return []
    return list(csv.DictReader(p.open()))


def force_metrics(rows, method):
    xs = [r for r in rows if r["method"] == method]
    if not xs:
        return {}
    chosen = [_f(r["chosen_force"]) for r in xs]
    under = over = exact = 0
    for r in xs:
        mu = _f(r["friction"])
        fs = FSTAR[mu]
        cf = _f(r["chosen_force"])
        under += int(cf < fs - 1e-6)
        over += int(cf > fs + 1e-6)
        exact += int(abs(cf - fs) < 1e-6)
    n = len(xs)
    return {
        "n": n,
        "full_sr": float(np.mean([int(r["full_task_success"]) for r in xs])),
        "pick_sr": float(np.mean([int(r["pick_success"]) for r in xs])),
        "lift_sr": float(np.mean([int(r["lift_success"]) for r in xs])),
        "place_sr": float(np.mean([int(r["place_success"]) for r in xs])),
        "mean_chosen_F": float(np.mean(chosen)),
        "mean_squeeze": float(np.mean([_f(r["mean_force"]) for r in xs])),
        "peak_squeeze": float(np.mean([_f(r["peak_force"]) for r in xs])),
        "integrated_force": float(np.mean([_f(r["integrated_force"]) for r in xs])),
        "exact_rate": exact / n,
        "under_rate": under / n,
        "over_rate": over / n,
        "under_n": under,
        "over_n": over,
        "exact_n": exact,
    }


def selection_by_mu(rows, method):
    xs = [r for r in rows if r["method"] == method]
    out = {}
    for mu in (0.2, 0.5, 1.0):
        sub = [r for r in xs if abs(_f(r["friction"]) - mu) < 1e-6]
        counts = {4: 0, 5: 0, 6: 0}
        for r in sub:
            counts[int(_f(r["chosen_force"]))] = counts.get(int(_f(r["chosen_force"])), 0) + 1
        out[str(mu)] = counts
    return out


def classify(offline, confirm, hier):
    if not confirm or hier.get("n", 0) < 30:
        return "D2_INCOMPLETE"
    if hier.get("under_rate", 1) > 0.02 or hier.get("under_n", 99) > 2:
        return "D2_HIERARCHICAL_RULE_NOT_STABLE"
    fixed6 = confirm.get("fixed6", {})
    oracle = confirm.get("oracle", {})
    sr_gap = fixed6.get("full_sr", 0) - hier.get("full_sr", 0)
    force_ok = hier.get("mean_chosen_F", 99) <= 5.25
    over_ok = hier.get("over_rate", 1) <= 0.20
    sr_ok = sr_gap <= 0.08
    stage1_ok = offline.get("heldout_hierarchical", {}).get("low_miss_rate", 1) == 0
    if stage1_ok and hier.get("under_rate", 1) <= 0.02 and force_ok and sr_ok and over_ok:
        return "D2_HIERARCHICAL_FORCE_DECISION_QUALIFIED"
    if hier.get("under_rate", 0) == 0 and hier.get("mean_chosen_F", 99) < fixed6.get("mean_chosen_F", 0) and sr_gap > 0.08:
        return "D2_FORCE_SELECTION_GOOD_BUT_DOWNSTREAM_SR_GAP"
    # coarse: stage1 stable but stage2 not
    sel = selection_by_mu(load_confirm(), "hierarchical")
    mid_err = sel.get("0.5", {}).get(4, 0)
    high_err = sel.get("1.0", {}).get(5, 0) + sel.get("1.0", {}).get(6, 0)
    if stage1_ok and hier.get("under_rate", 0) == 0 and (mid_err > 3 or high_err > 3):
        return "D2_COARSE_6_VS_NOT6_ONLY"
    return "D2_PARTIAL"


def main():
    rows = load_confirm()
    th = json.loads((OUT / "CALIBRATION_THRESHOLDS.json").read_text())
    offline = json.loads((OUT / "OFFLINE_SUMMARY.json").read_text())

    # FORCE_SELECTIONS.csv
    if rows:
        fs_rows = []
        for r in rows:
            mu = _f(r["friction"])
            fs = FSTAR[mu]
            cf = _f(r["chosen_force"])
            fs_rows.append({
                "trial_id": r["trial_id"], "method": r["method"], "seed_idx": r["seed_idx"],
                "gt_mu": mu, "chosen_F": cf, "fstar": fs,
                "exact": int(abs(cf - fs) < 1e-6),
                "under": int(cf < fs - 1e-6), "over": int(cf > fs + 1e-6),
                "decision_rule": r.get("decision_rule", ""),
                "z_imbalance_peak": r.get("z_imbalance_peak", ""),
                "z_ftan_hyst": r.get("z_ftan_hyst", ""),
                "full_task_success": r.get("full_task_success", ""),
            })
        with (OUT / "FORCE_SELECTIONS.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fs_rows[0].keys()))
            w.writeheader()
            w.writerows(fs_rows)

    methods = ["fixed6", "current_p3", "hierarchical", "oracle"]
    confirm = {m: force_metrics(rows, m) for m in methods}
    hier = confirm.get("hierarchical", {})
    fixed6 = confirm.get("fixed6", {})
    oracle = confirm.get("oracle", {})
    p3 = confirm.get("current_p3", {})

    status = classify(offline, confirm, hier)
    seeds_per = len({r["seed_idx"] for r in rows if r["method"] == "hierarchical"}) if rows else 0

    verdict = {
        "status": status,
        "method_change": "NONE",
        "probe_changed": False,
        "probe": "base-Y +2mm return",
        "target_force_mapping_N": {"0.20": 6, "0.50": 5, "1.00": 4},
        "stage1_signal": th["stage1_signal"],
        "stage1_threshold": th["theta_low"],
        "stage1_low_miss_rate": offline["heldout_hierarchical"]["low_miss_rate"],
        "stage2_signal": th["stage2_signal"],
        "stage2_threshold": th["theta_mid_high"],
        "offline_exact_accuracy": offline["heldout_hierarchical"]["exact"],
        "offline_under_force_rate": offline["heldout_hierarchical"]["under"],
        "offline_over_force_rate": offline["heldout_hierarchical"]["over"],
        "offline_mean_selected_force_N": offline["heldout_hierarchical"]["mean_F"],
        "confirmation_seed_count_per_friction": seeds_per,
        "confirmation_complete": len(rows) >= 240,
        "fixed6_full_sr": fixed6.get("full_sr"),
        "current_p3_full_sr": p3.get("full_sr"),
        "hierarchical_full_sr": hier.get("full_sr"),
        "oracle_full_sr": oracle.get("full_sr"),
        "hierarchical_mean_selected_force_N": hier.get("mean_chosen_F"),
        "oracle_mean_selected_force_N": 5.0,
        "hierarchical_matches_fixed6_success": (
            abs((hier.get("full_sr") or 0) - (fixed6.get("full_sr") or 0)) <= 0.08
            if hier and fixed6 else None
        ),
        "hierarchical_reduces_force": (
            (hier.get("mean_chosen_F") or 99) < (fixed6.get("mean_chosen_F") or 0)
            if hier and fixed6 else None
        ),
        "thresholds_tuned_on_confirmation": False,
        "primary_evidence": [
            "Stage-1 imb_peak veto: held-out low_miss=0, confirmation under-force priority",
            "Hierarchical rule avoids joint NB σ over-confidence on ftan_hyst",
            f"Offline held-out exact={offline['heldout_hierarchical']['exact']:.2f}, mean F={offline['heldout_hierarchical']['mean_F']:.2f}N",
        ],
        "limitations": [
            "Single task/object; thresholds calibrated on seeds 0-18 only",
            "Stage-2 μ=0.50 may over-veto to 6N (conservative Stage-1)",
        ],
        "confirmation_metrics": confirm,
        "selection_by_friction_hierarchical": selection_by_mu(rows, "hierarchical") if rows else {},
    }

    if rows:
        verdict["primary_evidence"].append(
            f"Confirmation N={seeds_per}/μ: hier SR={hier.get('full_sr', float('nan')):.3f}, mean F={hier.get('mean_chosen_F', float('nan')):.2f}N"
        )

    (OUT / "FINAL_VERDICT.json").write_text(json.dumps(verdict, indent=2) + "\n")

    readme = [
        "# D2 — Hierarchical Force Decision Qualification",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"**Status:** `{status}`",
        "",
        "## Rule",
        (OUT / "HIERARCHICAL_RULE_SPEC.md").read_text(),
        "",
        "## Offline held-out (odd seeds)",
        f"- exact: {offline['heldout_hierarchical']['exact']:.3f}",
        f"- under: {offline['heldout_hierarchical']['under']:.3f}",
        f"- over: {offline['heldout_hierarchical']['over']:.3f}",
        f"- mean F: {offline['heldout_hierarchical']['mean_F']:.2f} N",
        f"- gate passed: {offline.get('gate_passed')}",
        "",
        "## Confirmation (seeds 30–49)",
    ]
    if confirm.get("hierarchical"):
        for m in methods:
            c = confirm[m]
            readme += [
                f"### {m}",
                f"- full SR: {c.get('full_sr', float('nan')):.3f}",
                f"- mean chosen F: {c.get('mean_chosen_F', float('nan')):.2f} N",
                f"- under/over: {c.get('under_rate', float('nan')):.3f} / {c.get('over_rate', float('nan')):.3f}",
                "",
            ]
    else:
        readme.append("_Confirmation pending or incomplete._")
    (OUT / "README.md").write_text("\n".join(readme) + "\n")
    print(json.dumps({"status": status, "confirm_rows": len(rows), "seeds_per_mu": seeds_per}, indent=2))


if __name__ == "__main__":
    main()
