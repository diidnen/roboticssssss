#!/usr/bin/env python3
"""Layout-only regeneration of the two continuous-friction multipanel figures.

This consumes frozen CSV/JSON outputs and changes no metric or physical result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_COLORS = {0: "#2563EB", 1: "#D97706", 5: "#DB2777", 6: "#4D7C0F"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def anchor_map(plan: dict) -> dict[int, tuple[float, float, float]]:
    return {
        int(row["task"]): (
            float(row["low_anchor"]),
            float(row["mid_anchor"]),
            float(row["high_anchor"]),
        )
        for row in plan["tasks"]
    }


def belief_figure(out: Path, anchors: dict[int, tuple[float, float, float]]) -> None:
    belief = pd.read_csv(out / "TABLE_CONTINUOUS_FRICTION_BELIEF.csv")
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.5))
    for ax, task in zip(axes.flat, [0, 1, 5, 6]):
        data = belief[belief["task"] == task].sort_values(["mu_test", "root"])
        color = TASK_COLORS[task]
        lo = min(data["mu_test"].min(), data["interval_90_low"].min())
        hi = max(data["mu_test"].max(), data["interval_90_high"].max())
        pad = 0.04 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#334155", lw=1.2, label="y = x")
        yerr = np.vstack([
            data["posterior_mean"].to_numpy() - data["interval_90_low"].to_numpy(),
            data["interval_90_high"].to_numpy() - data["posterior_mean"].to_numpy(),
        ])
        ax.errorbar(
            data["mu_test"], data["posterior_mean"], yerr=yerr,
            fmt="o", ms=4.5, color=color, ecolor=color, alpha=0.82,
            capsize=2, lw=1.0, label="posterior mean ± 90% interval",
        )
        for index, anchor in enumerate(anchors[task]):
            ax.axvline(
                anchor, color="#64748B", ls=":", lw=0.9, alpha=0.8,
                label="canonical anchors" if index == 0 else None,
            )
        ax.set_title(f"Task {task}", fontsize=11)
        ax.set_xlabel("True unseen object-side μ")
        ax.set_ylabel("Posterior mean")
        ax.grid(alpha=0.22)
        ax.set_xlim(lo - pad, hi + pad)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Continuous-friction belief on 48 unseen contexts", fontsize=14, y=0.975)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=3, frameon=False, fontsize=8.5)
    fig.subplots_adjust(top=0.83, bottom=0.08, left=0.09, right=0.98, hspace=0.40, wspace=0.25)
    fig.savefig(out / "FIGURE_CONTINUOUS_FRICTION_BELIEF.pdf", bbox_inches="tight")
    plt.close(fig)


def force_figure(out: Path) -> None:
    trace = pd.read_csv(out / "TABLE_CONTINUOUS_FRICTION_FULL_TRACE.csv")
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.5), sharey=True)
    style = {
        "ACTIVEFORCING": ("ActiveForcing", "#2563EB", "o", "-"),
        "GT_PHYSICS": ("GT-Physics", "#D97706", "s", "--"),
    }
    for ax, task in zip(axes.flat, [0, 1, 5, 6]):
        task_data = trace[trace["task"] == task]
        for method, (label, color, marker, ls) in style.items():
            data = task_data[task_data["method"] == method]
            grouped = data.groupby("mu_test")["selected_force"].agg(["mean", "min", "max"]).reset_index()
            ax.fill_between(grouped["mu_test"], grouped["min"], grouped["max"], color=color, alpha=0.10)
            ax.plot(grouped["mu_test"], grouped["mean"], ls=ls, marker=marker, ms=4.5, color=color, lw=1.7, label=label)
        ax.axhline(4.0, color="#334155", ls=":", lw=1.3, label="Fixed-4")
        ax.set_title(f"Task {task}", fontsize=11)
        ax.set_xlabel("True unseen object-side μ")
        ax.set_ylabel("Selected force (N)")
        ax.set_ylim(2.9, 5.1)
        ax.grid(alpha=0.22)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Force selection on unseen continuous friction", fontsize=14, y=0.975)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(top=0.83, bottom=0.08, left=0.09, right=0.98, hspace=0.40, wspace=0.20)
    fig.savefig(out / "FIGURE_CONTINUOUS_FRICTION_FORCE.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plan = json.loads((args.out / "FINAL_UNSEEN_FRICTION_PLAN.json").read_text())
    belief_figure(args.out, anchor_map(plan))
    force_figure(args.out)


if __name__ == "__main__":
    main()
