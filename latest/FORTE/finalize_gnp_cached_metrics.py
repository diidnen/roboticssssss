#!/usr/bin/env python3
"""Finalize cached DEV metrics without re-reading or re-running frozen DEV."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

import gnp_style_continuous as g


OUT = Path("/home/exouser/FORTE/gnp_style_continuous_20260830_125107")


def pyrow(row: pd.Series) -> dict:
    return {k: v.item() if hasattr(v, "item") else v for k, v in row.to_dict().items()}


def main() -> None:
    decisions_df = pd.read_csv(OUT / "CONTINUOUS_DEV_FRONTIER_METRICS.csv")
    metrics_df = pd.read_csv(OUT / "CONTINUOUS_DEV_PROBABILITY_METRICS.csv")
    per_backend = {
        b: [pyrow(r) for _, r in decisions_df[decisions_df.backend == b].iterrows()]
        for b in ["FEASIBILITY_ONLY", "JOINT"]
    }

    # Correct only cached summary semantics.  No DEV source or checkpoint is read.
    metric_rows = []
    primary = {}
    for _, row in metrics_df.iterrows():
        d = pyrow(row)
        dec = per_backend[str(d["backend"])]
        valid = [r for r in dec if str(r["real_frontier_status"]) == "VALID_FINE_FRONTIER"]
        finite = [r for r in valid if math.isfinite(float(r["selected_force_N"]))]
        finite_under = [float(r["under_force_magnitude_N"]) for r in finite if int(r["under_force"]) == 1]
        d["valid_force_decision_contexts"] = len(finite)
        d["mean_under_force_magnitude_N"] = (
            sum(finite_under) / len(finite_under) if finite_under else math.nan
        )
        metric_rows.append(d)
        if str(d["probability_semantics"]) == "RAW_ENSEMBLE_PRIMARY":
            primary[str(d["backend"])] = d
    g.write_csv(OUT / "CONTINUOUS_DEV_PROBABILITY_METRICS.csv", metric_rows)

    valid_cids = [
        str(r["context_id"])
        for r in per_backend["JOINT"]
        if str(r["real_frontier_status"]) == "VALID_FINE_FRONTIER"
    ]
    all_cids = [str(r["context_id"]) for r in per_backend["JOINT"]]
    boots = [
        g.paired_bootstrap(per_backend["JOINT"], per_backend["FEASIBILITY_ONLY"], key, cids)
        for key, cids in [
            ("under_force", valid_cids),
            ("frontier_abs_error_N", valid_cids),
            ("probability_MAE", all_cids),
            ("excess_force_N", valid_cids),
        ]
    ]

    fm, jm = primary["FEASIBILITY_ONLY"], primary["JOINT"]
    joint_eval = {
        "under_force_safely_nonworse": jm["under_force_rate"] <= fm["under_force_rate"],
        "frontier_MAE_gain_N": fm["frontier_MAE_N"] - jm["frontier_MAE_N"],
        "frontier_margin_pass": fm["frontier_MAE_N"] - jm["frontier_MAE_N"] >= 0.05 - 1e-12,
        "probability_MAE_relative_gain": (fm["probability_MAE"] - jm["probability_MAE"]) / fm["probability_MAE"],
        "probability_margin_pass": (fm["probability_MAE"] - jm["probability_MAE"]) / fm["probability_MAE"] >= 0.10 - 1e-12,
        "nonmonotonicity_pass": jm["systematic_nonmonotonic_context_rate"] <= 0.10
        and jm["systematic_nonmonotonic_context_rate"] <= fm["systematic_nonmonotonic_context_rate"],
    }
    scientific_choice = "FEASIBILITY-ONLY"
    selected = "FEASIBILITY_ONLY"
    physical_status = "NO"
    classification = "CONTINUOUS_FEASIBILITY_STILL_NOT_VALIDATED"
    coverage = 8 / 9
    gates = {b: g.gate_for(primary[b], coverage) for b in primary}

    prior = json.loads((OUT / "SELECTED_CONTINUOUS_BACKEND.json").read_text())
    selected_payload = {
        "selected_backend": selected,
        "scientific_choice": scientific_choice,
        "physical_auxiliary_independent_advantage": physical_status,
        "PRIMARY_CLASSIFICATION": classification,
        "raw_probability_primary": True,
        "rho": g.RHO,
        "query_grid_step_N": g.GRID_STEP,
        "metrics": primary,
        "gates": gates,
        "joint_win_evaluation": joint_eval,
        "paired_bootstrap": boots,
        "DEV_input_sha256": prior["DEV_input_sha256"],
        "posthoc_summary_correction": {
            "DEV_reloaded": False,
            "checkpoints_rerun": False,
            "reason": "finite-pair bootstrap and non-comparable old off-grid/all-cell MAE classification correction",
        },
    }
    g.write_json(OUT / "SELECTED_CONTINUOUS_BACKEND.json", selected_payload)

    comparison = []
    for b in ["FEASIBILITY_ONLY", "JOINT"]:
        comparison.append(
            {
                **primary[b],
                "GT_gate_pass": gates[b]["pass"],
                "selected": int(b == selected),
                "scientific_choice": scientific_choice,
                "physical_auxiliary_status": physical_status,
            }
        )
    comparison.extend({"backend": "PAIRED_BOOTSTRAP", **x} for x in boots)
    g.write_csv(OUT / "CONTINUOUS_JOINT_VS_FEAS.csv", comparison)

    train_df = pd.read_csv(OUT / "CONTINUOUS_TRAIN_SUCCESS_DATA.csv")
    train_summary = {
        "total": 288 + len(train_df),
        "successes": 216 + int(train_df.full_task_success_y.sum()),
        "failures": 72 + len(train_df) - int(train_df.full_task_success_y.sum()),
    }
    g.render_final_report(
        OUT,
        classification,
        scientific_choice,
        physical_status,
        selected,
        primary,
        gates,
        joint_eval,
        None,
        train_summary,
        coverage,
        boots,
    )

    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    (OUT / "SHA256SUMS.txt").write_text(
        "".join(f"{g.sha256(p)}  {p.name}\n" for p in files), encoding="utf-8"
    )
    print(json.dumps({"status": "CORRECTED_FROM_CACHED_DEV_OUTPUTS", "classification": classification, "boots": boots}, indent=2))


if __name__ == "__main__":
    main()
