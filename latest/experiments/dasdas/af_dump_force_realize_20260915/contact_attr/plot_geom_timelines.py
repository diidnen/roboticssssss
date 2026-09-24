"""7-panel grasp-geometry timelines and overlays."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_steps(directory: Path) -> list[dict]:
    rows = []
    path = Path(directory) / "steps.jsonl"
    if not path.exists():
        return rows
    with path.open() as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _marks(axis, flags: dict, names: dict[str, str]) -> None:
    ymax = axis.get_ylim()[1]
    for key, label in names.items():
        event = flags.get(key)
        if not event or event.get("t") is None:
            continue
        axis.axvline(float(event["t"]), color="0.55", linewidth=0.7, linestyle="--")
        axis.text(float(event["t"]), ymax, label, rotation=90, va="top", ha="right", fontsize=6)


MARKS = {
    "T0_lift_start": "lift",
    "wrist_rotation_start": "wrist",
    "pour_start": "pour",
    "T4_unilateral_loss": "uni",
    "T5_squeeze_lt_1": "sq<1",
    "T6_bilateral_loss": "bilat",
    "T7_retention_loss": "ret",
    "T2_centroid_near_edge": "edge",
}


def plot_seven_panel(directory: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    steps = load_steps(directory)
    if not steps:
        return None
    flags = json.loads((directory / "SUMMARY.json").read_text()).get("flags") or {}
    t = np.asarray([row["t"] for row in steps], dtype=float)
    fig, axes = plt.subplots(7, 1, figsize=(11, 16), dpi=120, sharex=True)
    axes[0].plot(t, [row["F"] for row in steps], color="0.35", label="F_cmd")
    axes[0].plot(t, [row["squeeze"] for row in steps], label="squeeze=2*min(NL,NR)")
    axes[0].set_ylabel("N")
    axes[0].set_title("commanded F vs realized squeeze")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[1].plot(t, [row["nl"] for row in steps], label="NL fl_link7")
    axes[1].plot(t, [row["nr"] for row in steps], label="NR fl_link8")
    axes[1].set_ylabel("N")
    axes[1].set_title("grasp-slab normal force")
    axes[1].legend(fontsize=7, loc="upper right")
    axes[2].plot(t, [row["aperture"] for row in steps], label="aperture")
    axes[2].plot(t, [row["inner_cmd"] if row["inner_cmd"] is not None else np.nan for row in steps], linestyle="--", label="inner cmd")
    axes[2].set_ylabel("m")
    axes[2].set_title("gripper aperture")
    axes[2].legend(fontsize=7, loc="upper right")
    axes[3].plot(t, [row["drel_from_settle_deg"] for row in steps], label="geodesic from settle-end (deg)")
    axes[3].plot(t, [row.get("drel_from_settle_start_deg", 0.0) for row in steps], label="geodesic from settle-start (deg)", linestyle="--")
    rpy = np.asarray([row["rel_rpy"] for row in steps], dtype=float)
    axes[3].plot(t, np.degrees(rpy[:, 0]), label="roll deg", alpha=0.7)
    axes[3].plot(t, np.degrees(rpy[:, 1]), label="pitch deg", alpha=0.7)
    axes[3].plot(t, np.degrees(rpy[:, 2]), label="yaw deg", alpha=0.7)
    axes[3].set_ylabel("deg")
    axes[3].set_title("object-relative-to-gripper rotation")
    axes[3].legend(fontsize=7, loc="upper right", ncol=2)
    axes[4].plot(t, [row["left"]["u"] if row["left"]["u"] is not None else np.nan for row in steps], label="left u")
    axes[4].plot(t, [row["left"]["v"] if row["left"]["v"] is not None else np.nan for row in steps], label="left v")
    axes[4].plot(t, [row["right"]["u"] if row["right"]["u"] is not None else np.nan for row in steps], label="right u")
    axes[4].plot(t, [row["right"]["v"] if row["right"]["v"] is not None else np.nan for row in steps], label="right v")
    axes[4].set_ylabel("object m")
    axes[4].set_title("grasp-slab contact centroid local u/v")
    axes[4].legend(fontsize=7, loc="upper right", ncol=2)
    axes[5].plot(t, [row["left_edge"] if row["left_edge"] is not None else np.nan for row in steps], label="left edge dist")
    axes[5].plot(t, [row["right_edge"] if row["right_edge"] is not None else np.nan for row in steps], label="right edge dist")
    axes[5].axhline(0.0, color="0.5", linewidth=0.6)
    axes[5].set_ylabel("m")
    axes[5].set_title("distance to grasp-slab in-plane edge (negative = outside)")
    axes[5].legend(fontsize=7, loc="upper right")
    axes[6].plot(t, [1 if row["left"]["present"] else 0 for row in steps], drawstyle="steps-post", label="left grasp")
    axes[6].plot(t, [1 if row["right"]["present"] else 0 for row in steps], drawstyle="steps-post", label="right grasp")
    axes[6].plot(t, [1 if row["bilateral"] else 0 for row in steps], drawstyle="steps-post", label="bilateral")
    axes[6].plot(t, [row.get("pair_dist") if row.get("pair_dist") is not None else np.nan for row in steps], label="contact pair dist m")
    axes[6].plot(t, [row.get("finger_sep") if row.get("finger_sep") is not None else np.nan for row in steps], label="finger link sep m", linestyle="--")
    axes[6].plot(t, [row["table_bottom_fn"] for row in steps], label="table-bottom Fn", alpha=0.7)
    axes[6].set_ylabel("0/1 or N")
    axes[6].set_title("contact validity and table-bottom force")
    axes[6].legend(fontsize=7, loc="upper right")
    axes[6].set_xlabel("t (s)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        _marks(axis, flags, MARKS)
    fig.tight_layout()
    path = directory / "timeline7.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_slab_maps(directory: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    steps = load_steps(directory)
    if not steps:
        return None
    flags = json.loads((directory / "SUMMARY.json").read_text()).get("flags") or {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=130)
    for axis, side, title in ((axes[0], "left", "left fl_link7"), (axes[1], "right", "right fl_link8")):
        us, vs, fns = [], [], []
        for row in steps:
            item = row[side]
            if item.get("u") is None:
                continue
            us.append(item["u"])
            vs.append(item["v"])
            fns.append(item["fn"])
        if us:
            axis.scatter(us, vs, c=fns, s=8, cmap="viridis", alpha=0.8)
            item = next((row[side] for row in steps if row[side].get("u_lo") is not None), None)
            if item:
                axis.plot(
                    [item["u_lo"], item["u_hi"], item["u_hi"], item["u_lo"], item["u_lo"]],
                    [item["v_lo"], item["v_lo"], item["v_hi"], item["v_hi"], item["v_lo"]],
                    color="0.2",
                    linewidth=1,
                )
        for key, color in (("T0_lift_start", "C3"), ("T2_centroid_near_edge", "C1"), ("T4_unilateral_loss", "C0")):
            event = flags.get(key)
            if not event:
                continue
            step_i = event.get("step")
            if step_i is None or step_i >= len(steps):
                continue
            item = steps[int(step_i)][side]
            if item.get("u") is None:
                continue
            axis.scatter([item["u"]], [item["v"]], s=40, facecolors="none", edgecolors=color, label=key)
        axis.set_title(f"{title} contact on slab (object u,v)")
        axis.set_xlabel("u (m)")
        axis.set_ylabel("v (m)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=6)
    fig.tight_layout()
    path = directory / "slab_map.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _series(directory: Path):
    steps = load_steps(directory)
    flags = {}
    summary = Path(directory) / "SUMMARY.json"
    if summary.exists():
        flags = json.loads(summary.read_text()).get("flags") or {}
    return steps, flags


def plot_overlay_425(root: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = Path(root) / "mu0.425_F0.75"
    b = Path(root) / "mu0.425_F1"
    if not (a / "steps.jsonl").exists() or not (b / "steps.jsonl").exists():
        return None

    def _lab(path: Path, fallback: str) -> str:
        result = path / "result.json"
        if result.exists():
            payload = json.loads(result.read_text())
            return f"{fallback} ret={payload.get('retention_success')}"
        return fallback

    sa, fa = _series(a)
    sb, fb = _series(b)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), dpi=130, sharex=False)
    for steps, flags, label, color in (
        (sa, fa, _lab(a, "F=0.75"), "C0"),
        (sb, fb, _lab(b, "F=1.00"), "C3"),
    ):
        t = [row["t"] for row in steps]
        axes[0].plot(
            t,
            [row.get("drel_from_settle_start_deg", row["drel_from_settle_deg"]) for row in steps],
            label=label,
            color=color,
        )
        axes[1].plot(t, [row["left_edge"] if row["left_edge"] is not None else np.nan for row in steps], color=color, linestyle="-")
        axes[1].plot(t, [row["right_edge"] if row["right_edge"] is not None else np.nan for row in steps], color=color, linestyle="--")
        axes[2].plot(t, [row["nl"] for row in steps], color=color, linestyle="-")
        axes[2].plot(t, [row["nr"] for row in steps], color=color, linestyle="--")
        axes[3].plot(t, [row["aperture"] for row in steps], label=label, color=color)
        _marks(axes[0], flags, {"T0_lift_start": "lift", "T4_unilateral_loss": "uni", "T6_bilateral_loss": "bilat"})
    axes[0].set_title("μ=0.425 overlay: relative rotation from settle-start")
    axes[0].set_ylabel("deg")
    axes[1].set_title("edge distance (solid left, dashed right)")
    axes[1].set_ylabel("m")
    axes[2].set_title("NL (solid) / NR (dashed)")
    axes[2].set_ylabel("N")
    axes[3].set_title("aperture")
    axes[3].set_ylabel("m")
    axes[3].set_xlabel("t (s)")
    axes[0].legend(fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    path = Path(root) / "overlay_mu0.425_F0.75_vs_F1.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_overlay_575(root: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = [
        (Path(root) / "mu0.575_F1", "F=1.00", "C0"),
        (Path(root) / "mu0.575_F1.25", "F=1.25", "C2"),
        (Path(root) / "mu0.575_F1.5", "F=1.50", "C3"),
    ]
    if not all((path / "steps.jsonl").exists() for path, _, _ in files):
        return None
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), dpi=130, sharex=False)
    for path, label, color in files:
        steps, flags = _series(path)
        t = [row["t"] for row in steps]
        axes[0].plot(
            t,
            [row.get("drel_from_settle_start_deg", row["drel_from_settle_deg"]) for row in steps],
            label=label,
            color=color,
        )
        axes[1].plot(t, [row["left_edge"] if row["left_edge"] is not None else np.nan for row in steps], color=color, linestyle="-")
        axes[1].plot(t, [row["right_edge"] if row["right_edge"] is not None else np.nan for row in steps], color=color, linestyle="--")
        axes[2].plot(t, [row["nl"] for row in steps], color=color, linestyle="-")
        axes[2].plot(t, [row["nr"] for row in steps], color=color, linestyle="--")
        axes[3].plot(t, [row["aperture"] for row in steps], label=label, color=color)
        _marks(axes[0], flags, {"T0_lift_start": "lift", "T4_unilateral_loss": "uni"})
    axes[0].set_title("μ=0.575 overlay: relative rotation from settle-start")
    axes[0].set_ylabel("deg")
    axes[1].set_title("edge distance (solid left, dashed right)")
    axes[1].set_ylabel("m")
    axes[2].set_title("NL (solid) / NR (dashed)")
    axes[2].set_ylabel("N")
    axes[3].set_title("aperture")
    axes[3].set_ylabel("m")
    axes[3].set_xlabel("t (s)")
    axes[0].legend(fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    path = Path(root) / "overlay_mu0.575_F1_1.25_1.5.png"
    fig.savefig(path)
    plt.close(fig)
    return path
