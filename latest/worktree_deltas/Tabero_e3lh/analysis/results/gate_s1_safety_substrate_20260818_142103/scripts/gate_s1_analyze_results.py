#!/usr/bin/env python3
"""Post-process Gate S1 sweep CSVs into verdict artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(os.environ.get("GATE_S1_OUT", Path(__file__).resolve().parents[1]))
PLOTS = OUT / "representative_plots"
PLOTS.mkdir(exist_ok=True)


def load_summaries() -> pd.DataFrame:
    path = OUT / "FORCE_CALIBRATION.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def pareto_dominates(a: tuple, b: tuple) -> bool:
    """a dominates b if a >= b on all objectives and strictly > on one."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def best_force_per_phi(df: pd.DataFrame, phi_col: str, phi_val: float) -> dict:
    sub = df[df[phi_col] == phi_val].copy()
    agg = (
        sub.groupby("target_squeeze_N")
        .agg(
            grasp_stable_rate=("grasp_stable", "mean"),
            slip_rate=("slip", "mean"),
            drop_rate=("dropped", "mean"),
            mean_squeeze=("hold_mean_squeeze_N", "mean"),
            mean_contact=("hold_contact_ratio", "mean"),
            obj_disp_xy=("obj_disp_xy_m", "mean"),
        )
        .reset_index()
    )
    # Objectives: maximize grasp_stable, minimize squeeze (gentleness telemetry)
    best_grasp = agg.loc[agg["grasp_stable_rate"].idxmax()]
    # Among max grasp rate, pick lowest squeeze
    top = agg[agg["grasp_stable_rate"] >= agg["grasp_stable_rate"].max() - 1e-9]
    best_pareto = top.loc[top["mean_squeeze"].idxmin()]
    return {
        "agg": agg,
        "F_star_grasp": float(best_grasp["target_squeeze_N"]),
        "F_star_pareto": float(best_pareto["target_squeeze_N"]),
        "grasp_rate_at_F_star": float(best_pareto["grasp_stable_rate"]),
        "squeeze_at_F_star": float(best_pareto["mean_squeeze"]),
    }


def compute_vopi(df: pd.DataFrame, phi_col: str, phi_values: list[float]) -> dict | None:
    per_phi = {}
    for phi in phi_values:
        info = best_force_per_phi(df, phi_col, phi)
        per_phi[phi] = info
    # Q(F, phi) = (grasp_stable_rate, -mean_squeeze) aggregated over seeds
    forces = sorted(df["target_squeeze_N"].unique())
    q = {}
    for phi in phi_values:
        sub = df[df[phi_col] == phi]
        for f in forces:
            rows = sub[sub["target_squeeze_N"] == f]
            q[(f, phi)] = (
                float(rows["grasp_stable"].mean()),
                -float(rows["hold_mean_squeeze_N"].mean()),
            )
    # belief: uniform over phi
    p = 1.0 / len(phi_values)
    v_belief = max(
        sum(p * q[(f, phi)][0] for phi in phi_values) for f in forces
    )
    v_oracle = sum(p * max(q[(f, phi)][0] for f in forces) for phi in phi_values)
    return {
        "V_belief": v_belief,
        "V_oracle": v_oracle,
        "VoPI": v_oracle - v_belief,
        "per_phi_F_star": {str(k): v["F_star_pareto"] for k, v in per_phi.items()},
    }


def main():
    df = load_summaries()
    df_nom = df[df["mass_scale"] == 1.0].copy()

    # Calibration plot
    cal = (
        df_nom.groupby("target_squeeze_N")
        .agg(
            mean_meas=("hold_mean_squeeze_N", "mean"),
            std_meas=("hold_mean_squeeze_N", "std"),
            contact=("hold_contact_ratio", "mean"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        cal["target_squeeze_N"],
        cal["mean_meas"],
        yerr=cal["std_meas"],
        fmt="o-",
        capsize=3,
        label="measured squeeze (hold)",
    )
    ax.plot(cal["target_squeeze_N"], cal["target_squeeze_N"], "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("target squeeze N")
    ax.set_ylabel("measured squeeze N")
    ax.set_title("Force calibration (mass_scale=1.0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "force_calibration.png", dpi=150)
    plt.close(fig)

    # Monotonicity check
    targets = cal["target_squeeze_N"].values
    means = cal["mean_meas"].values
    mono = bool(np.all(np.diff(means) >= -0.5))  # allow small noise

    phi_values = sorted(df["mass_scale"].unique())
    physics_results = {}
    action_disagreement = False
    f_stars = {}
    fixed_robust = None

    if len(phi_values) > 1:
        for phi in phi_values:
            physics_results[str(phi)] = best_force_per_phi(df, "mass_scale", phi)
            f_stars[phi] = physics_results[str(phi)]["F_star_pareto"]
        unique_f = set(f_stars.values())
        action_disagreement = len(unique_f) > 1
        # fixed robust: one force with grasp_stable_rate >= best-0.05 across all phi
        forces = sorted(df["target_squeeze_N"].unique())
        for f in forces:
            ok = True
            for phi in phi_values:
                sub = df[(df["mass_scale"] == phi) & (df["target_squeeze_N"] == f)]
                if sub["grasp_stable"].mean() < df[df["mass_scale"] == phi]["grasp_stable"].max() - 0.05:
                    ok = False
                    break
            if ok:
                fixed_robust = f
                break

        # Physics heatmap
        pivot = df.pivot_table(
            index="mass_scale",
            columns="target_squeeze_N",
            values="grasp_stable",
            aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(8, 3))
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"m={i:g}" for i in pivot.index])
        ax.set_xlabel("target force N")
        ax.set_ylabel("mass scale")
        ax.set_title("grasp_stable rate")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(PLOTS / "grasp_stable_heatmap.png", dpi=150)
        plt.close(fig)

    vopi = compute_vopi(df, "mass_scale", phi_values) if len(phi_values) > 1 else None

    # Belief update signal: do force/tactile stats separate mass scales at same target force?
    belief_signal = {}
    if "marker_motion_mean_hold" in df.columns:
        for f in sorted(df["target_squeeze_N"].unique()):
            sub = df[df["target_squeeze_N"] == f]
            belief_signal[str(f)] = {
                "squeeze_std_across_phi": float(sub.groupby("mass_scale")["hold_mean_squeeze_N"].mean().std()),
                "marker_std_across_phi": float(sub.groupby("mass_scale")["marker_motion_mean_hold"].mean().std()),
                "disp_std_across_phi": float(sub.groupby("mass_scale")["obj_disp_xy_m"].mean().std()),
            }

    analysis = {
        "force_calibration_monotonic": mono,
        "calibration_table": cal.to_dict(orient="records"),
        "physics_results": {
            k: {
                "F_star_pareto": v["F_star_pareto"],
                "grasp_rate": v["grasp_rate_at_F_star"],
                "squeeze": v["squeeze_at_F_star"],
            }
            for k, v in physics_results.items()
        },
        "action_disagreement": action_disagreement,
        "F_star_by_phi": {str(k): v for k, v in f_stars.items()},
        "fixed_robust_force": fixed_robust,
        "vopi": vopi,
        "belief_update_signal": belief_signal,
    }
    (OUT / "ANALYSIS_SUMMARY.json").write_text(json.dumps(analysis, indent=2))
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
