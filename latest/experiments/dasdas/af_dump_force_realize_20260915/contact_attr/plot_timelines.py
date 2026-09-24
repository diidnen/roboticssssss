"""6-panel contact timelines from steps.jsonl."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_steps(directory: Path) -> list[dict]:
    rows = []
    with (directory / "steps.jsonl").open() as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mark(axis, flags: dict, mapping: dict[str, str]) -> None:
    ymin, ymax = axis.get_ylim()
    for key, label in mapping.items():
        event = flags.get(key)
        if not event or event.get("t") is None:
            continue
        axis.axvline(float(event["t"]), color="0.4", linewidth=0.8, linestyle="--")
        axis.text(float(event["t"]), ymax, label, rotation=90, va="top", ha="right", fontsize=7)


def plot_timeline(directory: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    steps = load_steps(directory)
    summary = json.loads((directory / "SUMMARY.json").read_text()) if (directory / "SUMMARY.json").exists() else {}
    flags = summary.get("flags") or {}
    t = np.asarray([row["t"] for row in steps], dtype=float)
    fig, axes = plt.subplots(6, 1, figsize=(11, 14), dpi=130, sharex=True)
    axes[0].plot(t, [row["F"] for row in steps], label="F_cmd single-finger", color="0.3")
    axes[0].plot(t, [row["squeeze"] for row in steps], label="squeeze=2*min(NL,NR)", color="C0")
    axes[0].set_ylabel("N")
    axes[0].set_title("commanded F vs realized squeeze")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(t, [row["nl"] for row in steps], label="NL")
    axes[1].plot(t, [row["nr"] for row in steps], label="NR")
    axes[1].set_ylabel("N")
    axes[1].set_title("per-finger target-object normal force")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(t, [row["aperture"] for row in steps], label="mean |qpos|")
    cmds = [row["inner_cmd"] if row["inner_cmd"] is not None else np.nan for row in steps]
    axes[2].plot(t, cmds, label="inner aperture cmd", linestyle="--")
    axes[2].set_ylabel("m")
    axes[2].set_title("gripper aperture / inner command")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[3].plot(t, [1.0 if row["bilateral"] else 0.0 for row in steps], drawstyle="steps-post")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_ylabel("0/1")
    axes[3].set_title("bilateral finger-target contact")
    series = defaultdict(lambda: np.zeros(len(steps)))
    for index, row in enumerate(steps):
        grouped = defaultdict(float)
        for pair in row["pairs"]:
            if pair["kind"].startswith("finger-grasp"):
                continue
            grouped[pair["kind"]] += pair["fn"]
        for kind, value in grouped.items():
            series[kind][index] = value
    for kind, values in sorted(series.items(), key=lambda item: -float(np.max(item[1]))):
        if float(np.max(values)) < 0.05:
            continue
        axes[4].plot(t, values, label=kind)
    axes[4].set_ylabel("N")
    axes[4].set_title("non-finger-grasp deskbin contacts (grouped)")
    if series:
        axes[4].legend(loc="upper right", fontsize=7, ncol=2)
    axes[5].plot(t, [row["rel_p"][0] for row in steps], label="rel x")
    axes[5].plot(t, [row["rel_p"][1] for row in steps], label="rel y")
    axes[5].plot(t, [row["rel_p"][2] for row in steps], label="rel z")
    axes[5].plot(t, [row["rel_ang"] for row in steps], label="rel angle (rad)")
    axes[5].set_ylabel("m / rad")
    axes[5].set_title("object pose relative to gripper")
    axes[5].legend(loc="upper right", fontsize=8)
    axes[5].set_xlabel("t (s)")
    marks = {
        "pour_start": "pour",
        "first_bilateral_loss": "bilat loss",
        "first_squeeze_lt_1": "sq<1N",
        "first_unintended": "unint.",
        "retention_loss": "ret. loss",
    }
    for axis in axes:
        axis.grid(True, alpha=0.3)
        _mark(axis, flags, marks)
    fig.tight_layout()
    path = directory / "timeline.png"
    fig.savefig(path)
    plt.close(fig)
    return path
