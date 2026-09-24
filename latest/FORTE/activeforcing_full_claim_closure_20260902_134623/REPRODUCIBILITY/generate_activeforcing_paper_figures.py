#!/usr/bin/env python3
"""Create reproducible static figures from the assembled Direct evidence bundle."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def make_figures(bundle: Path) -> None:
    import matplotlib.pyplot as plt

    fig_dir = bundle / "PAPER_FIGURES"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})
    blue, gold, charcoal, light = "#245A9C", "#B78319", "#30343B", "#D9E4F1"

    main = [r for r in read_csv(bundle / "TABLE_MAIN_FORCE_ADAPTATION.csv")
            if r.get("scope") == "72 contexts × 2 repeats"]
    labels = [r["method"] for r in main]
    sr = [100 * (f(r, "full_task_success_rate") or 0) for r in main]
    under = [100 * (f(r, "under_force_rate") or 0) for r in main]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    x = list(range(len(labels)))
    ax[0].bar(x, sr, color=blue, width=0.65)
    ax[0].set_title("720-branch full-task success")
    ax[0].set_ylabel("Success rate (%)")
    ax[0].set_ylim(0, 105)
    ax[1].bar(x, under, color=gold, width=0.65)
    ax[1].set_title("720-branch under-force rate")
    ax[1].set_ylabel("Rate (%)")
    ax[1].set_ylim(0, max(8, max(under) * 1.35))
    for a in ax:
        a.set_xticks(x, labels, rotation=25, ha="right")
        a.set_xlabel("Physical input variant")
    fig.suptitle("ActiveForcing-Direct paired hidden-physics benchmark", color=charcoal)
    fig.savefig(fig_dir / "FIGURE_MAIN_720_BENCHMARK.png", dpi=220)
    fig.savefig(fig_dir / "FIGURE_MAIN_720_BENCHMARK.pdf")
    plt.close(fig)

    ident = read_csv(bundle / "TABLE_PHYSICAL_IDENTIFICATION.csv")
    labels = [f"{r['method']}\n({r['split'].replace('_', ' ').title()})" for r in ident]
    mae = [f(r, "friction_MAE") or 0 for r in ident]
    fig, ax = plt.subplots(figsize=(8.3, 4.5), constrained_layout=True)
    ax.bar(range(len(labels)), mae, color=blue, width=0.62)
    ax.set_title("Friction identification error")
    ax.set_ylabel("MAE (friction units)")
    ax.set_xlabel("Estimator and evaluation split")
    ax.set_xticks(range(len(labels)), labels, rotation=18, ha="right")
    ax.set_ylim(0, max(mae) * 1.25)
    fig.savefig(fig_dir / "FIGURE_PHYSICAL_IDENTIFICATION_MAE.png", dpi=220)
    fig.savefig(fig_dir / "FIGURE_PHYSICAL_IDENTIFICATION_MAE.pdf")
    plt.close(fig)

    locked = []
    for task in (0, 1, 5, 6):
        p = bundle / "E5_LOCKED_TEST" / f"task{task}" / "E5_UTILITY_ROLLOUTS.csv"
        if p.exists():
            for r in read_csv(p):
                r["task"] = str(task)
                locked.append(r)
    methods = [
        ("FROZEN_PI0_NATIVE_DEFAULT", "Default"),
        ("FIXED_MAX", "Fixed-Max"),
        ("NO_QUERY_TRAINING_PRIOR_UTILITY", "NoQuery"),
        ("ACTIVEFORCING_1Q_UTILITY", "Direct"),
        ("GT_PHYSICS_DIRECT_UTILITY", "GT physics"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    width = 0.15
    for j, (source, label) in enumerate(methods):
        vals = []
        for task in (0, 1, 5, 6):
            q = [r for r in locked if r.get("task") == str(task) and r.get("method") == source]
            vals.append(100 * (f(q[0], "full_task_success_y") if q else 0))
        ax.bar([i + (j - 2) * width for i in range(4)], vals, width=width,
               label=label, color=[charcoal, gold, light, blue, "#7A8F5B"][j],
               edgecolor=charcoal, linewidth=0.35)
    ax.set_title("Fresh reset-to-end locked subset: task success")
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Task (one root3/LOW tuple per method)")
    ax.set_xticks(range(4), ["task0", "task1", "task5", "task6"])
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(fig_dir / "FIGURE_LOCKED_E5_TASK_SUCCESS.png", dpi=220, bbox_inches="tight")
    fig.savefig(fig_dir / "FIGURE_LOCKED_E5_TASK_SUCCESS.pdf", bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "generator": "generate_activeforcing_paper_figures.py",
        "figures": [
            {"file": "FIGURE_MAIN_720_BENCHMARK.png", "source": "TABLE_MAIN_FORCE_ADAPTATION.csv", "grain": "72 contexts x 2 repeats"},
            {"file": "FIGURE_PHYSICAL_IDENTIFICATION_MAE.png", "source": "TABLE_PHYSICAL_IDENTIFICATION.csv", "grain": "reported estimator/split rows"},
            {"file": "FIGURE_LOCKED_E5_TASK_SUCCESS.png", "source": "E5_LOCKED_TEST/task*/E5_UTILITY_ROLLOUTS.csv", "grain": "one root3/LOW tuple per task and method"},
        ],
        "caveat": "The locked E5 figure is a small held-out subset, not a broad balanced benchmark; all expanded-method claims remain separately classified.",
    }
    (fig_dir / "FIGURE_METADATA.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (fig_dir / "FIGURE_STATUS.md").write_text(
        "# Paper figures\n\n"
        "Three reproducible static figures are generated from the assembled CSV/rollout evidence. "
        "The locked E5 panel is explicitly a one-root-per-task held-out subset; it is not a broad balanced claim.\n\n"
        "- `FIGURE_MAIN_720_BENCHMARK`: paired 720-branch success and under-force comparison.\n"
        "- `FIGURE_PHYSICAL_IDENTIFICATION_MAE`: reported friction MAE by estimator/split.\n"
        "- `FIGURE_LOCKED_E5_TASK_SUCCESS`: current Direct fresh reset-to-end locked subset.\n",
        encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: generate_activeforcing_paper_figures.py BUNDLE")
    make_figures(Path(sys.argv[1]).resolve())
