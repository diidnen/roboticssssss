#!/usr/bin/env python3
"""Reproducible static M1 figure from reviewed P4-B episode summaries."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500")
DATA = ROOT / "M1_TASK0_P4B/M1_TASK0_P4B_EPISODES.csv"
OUT = ROOT / "PAPER_MASS_FIGURES/M1_P4B_RESPONSE_BY_MASS.png"


def main():
    with DATA.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    panels = [
        ("f_meas_mean", "Mean measured squeeze (N)"),
        ("normal_force_mean", "Mean normal force (N)"),
        ("rho_impulse", "Normalized shear impulse"),
        ("marker_mean", "Mean marker motion"),
    ]
    bands = [("LOW", 0.05), ("MID", 0.10), ("HIGH", 0.20)]
    roots = sorted({r["seed_idx"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    for ax, (key, ylabel) in zip(axes.flat, panels):
        for root in roots:
            sub = [r for r in rows if r["seed_idx"] == root]
            sub = sorted(sub, key=lambda r: float(r["mass_kg"]))
            ax.plot([float(r["mass_kg"]) for r in sub], [float(r[key]) for r in sub], marker="o", color="#2f5d8c", alpha=0.62, linewidth=0.9)
            for r in sub:
                if r["split"] == "HELDOUT":
                    ax.scatter(float(r["mass_kg"]), float(r[key]), facecolors="white", edgecolors="#2f5d8c", s=40, zorder=3)
        ax.set_title(ylabel, color="#222222")
        ax.set_xlabel("Mass (kg)")
        ax.set_xticks([m for _, m in bands], [b for b, _ in bands])
        ax.grid(True, color="#d9dee5", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("P4-B query response across fixed mass bands", fontsize=14, color="#222222")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.16, wspace=0.28, hspace=0.38)
    fig.text(0.5, 0.035, "Filled markers: development roots (n=6); open markers: heldout roots (n=2). Fixed friction=0.5.", ha="center", fontsize=9, color="#555555")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
