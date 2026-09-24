"""Wall-pinch operating-range plots and tables."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "wall_pinch"


def load_metrics(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("**/metrics.json")):
        if path.parent == root:
            continue
        row = json.loads(path.read_text())
        row["_dir"] = str(path.parent)
        rows.append(row)
    return rows


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return None if not xs else float(np.mean(xs))


def _min(xs):
    xs = [float(x) for x in xs if x is not None]
    return None if not xs else float(np.min(xs))


def _max(xs):
    xs = [float(x) for x in xs if x is not None]
    return None if not xs else float(np.max(xs))


def group_table(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[(float(row["friction"]), float(row["force_N"]))].append(row)
    table = []
    for (mu, force), items in sorted(buckets.items()):
        table.append(
            {
                "mu": mu,
                "F": force,
                "n": len(items),
                "retention_pct": 100.0 * np.mean([int(x.get("retention_success") or 0) for x in items]),
                "settle_squeeze": _mean([x.get("settle_squeeze") for x in items]),
                "lift_squeeze": _mean([x.get("lift_squeeze") for x in items]),
                "wrist_squeeze": _mean([x.get("wrist_squeeze") for x in items]),
                "pour_squeeze": _mean([x.get("pour_squeeze_geom") if x.get("pour_squeeze_geom") is not None else x.get("measured_force_mean_n") for x in items]),
                "max_relative_rotation": _mean([x.get("max_relative_rotation_deg") for x in items]),
                "min_aperture": _mean([x.get("min_aperture") for x in items]),
                "unilateral_loss_pct": 100.0 * np.mean([1.0 if x.get("unilateral_loss") else 0.0 for x in items]),
                "bilateral_loss_pct": 100.0 * np.mean([1.0 if x.get("bilateral_loss") else 0.0 for x in items]),
                "min_edge_margin": _mean([x.get("min_edge_margin") for x in items]),
                "force_realized_pct": 100.0 * np.mean([1.0 if x.get("force_realized") else 0.0 for x in items]),
                "same_nx_pct": 100.0 * np.mean([1.0 if x.get("left_slab_settle") == "grasp_nx" and x.get("right_slab_settle") == "grasp_nx" else 0.0 for x in items]),
            }
        )
    return table


def matched_f075_vs_f100(rows: list[dict], mu: float = 0.425) -> dict:
    by = defaultdict(dict)
    for row in rows:
        if abs(float(row["friction"]) - mu) > 1e-9:
            continue
        by[int(row.get("episode_id", row.get("repeat", -1)))][float(row["force_N"])] = row
    pairs = []
    for rid, fmap in sorted(by.items()):
        a, b = fmap.get(0.75), fmap.get(1.00)
        if not a or not b:
            continue
        pairs.append(
            {
                "repeat": rid,
                "ret_075": int(a.get("retention_success") or 0),
                "ret_100": int(b.get("retention_success") or 0),
                "pour_075": a.get("pour_squeeze_geom"),
                "pour_100": b.get("pour_squeeze_geom"),
                "rot_075": a.get("max_relative_rotation_deg"),
                "rot_100": b.get("max_relative_rotation_deg"),
                "ap_075": a.get("min_aperture"),
                "ap_100": b.get("min_aperture"),
                "uni_075": bool(a.get("unilateral_loss")),
                "uni_100": bool(b.get("unilateral_loss")),
            }
        )
    n = len(pairs)
    if not n:
        return {"n": 0}
    pour_sign = []
    for p in pairs:
        if p["pour_075"] is None or p["pour_100"] is None:
            continue
        pour_sign.append(np.sign(float(p["pour_075"]) - float(p["pour_100"])))
    rot_sign = []
    for p in pairs:
        if p["rot_075"] is None or p["rot_100"] is None:
            continue
        rot_sign.append(np.sign(float(p["rot_100"]) - float(p["rot_075"])))
    ap_sign = []
    for p in pairs:
        if p["ap_075"] is None or p["ap_100"] is None:
            continue
        ap_sign.append(np.sign(float(p["ap_075"]) - float(p["ap_100"])))
    return {
        "n": n,
        "pairs": pairs,
        "retention_075_rate": float(np.mean([p["ret_075"] for p in pairs])),
        "retention_100_rate": float(np.mean([p["ret_100"] for p in pairs])),
        "frac_075_higher_pour": None if not pour_sign else float(np.mean([s > 0 for s in pour_sign])),
        "frac_100_higher_rotation": None if not rot_sign else float(np.mean([s > 0 for s in rot_sign])),
        "frac_100_smaller_aperture": None if not ap_sign else float(np.mean([s > 0 for s in ap_sign])),
        "frac_100_earlier_or_more_uni": float(np.mean([p["uni_100"] and not p["uni_075"] or p["uni_100"] for p in pairs])),
    }


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


def plot_overlay_mu425(root: Path, repeat: int, dest: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {0.50: "C0", 0.75: "C2", 1.00: "C3", 1.25: "C1"}
    series = []
    for force in (0.50, 0.75, 1.00, 1.25):
        directory = root / f"r{repeat}_mu0.425_F{force:g}"
        if not (directory / "steps.jsonl").exists():
            continue
        steps = load_steps(directory)
        flags = {}
        if (directory / "SUMMARY.json").exists():
            flags = json.loads((directory / "SUMMARY.json").read_text()).get("flags") or {}
        series.append((force, steps, flags))
    if len(series) < 2:
        return None
    fig, axes = plt.subplots(6, 1, figsize=(12, 14.5), dpi=130, sharex=True)

    def marks(axis, flags):
        ymax = axis.get_ylim()[1]
        for key, label in (
            ("T0_lift_start", "lift"),
            ("wrist_rotation_start", "wrist"),
            ("pour_start", "pour"),
            ("T4_unilateral_loss", "uni"),
            ("T6_bilateral_loss", "bilat"),
        ):
            event = flags.get(key)
            if not event or event.get("t") is None:
                continue
            axis.axvline(float(event["t"]), color="0.6", linewidth=0.6, linestyle="--")
            axis.text(float(event["t"]), ymax, label, rotation=90, va="top", ha="right", fontsize=6, color="0.35")

    for force, steps, flags in series:
        t = [row["t"] for row in steps]
        color = colors[force]
        label = f"F={force:.2f}"
        axes[0].plot(t, [row["F"] for row in steps], color=color, linestyle=":", alpha=0.7)
        axes[0].plot(t, [row["squeeze"] for row in steps], color=color, label=label)
        axes[1].plot(t, [row.get("drel_from_settle_start_deg") or 0.0 for row in steps], color=color, label=label)
        axes[2].plot(t, [row["aperture"] for row in steps], color=color, label=label)
        axes[3].plot(t, [row["nl"] for row in steps], color=color, linestyle="-")
        axes[3].plot(t, [row["nr"] for row in steps], color=color, linestyle="--")
        axes[4].plot(
            t,
            [row.get("left", {}).get("edge_v") if row.get("left", {}).get("edge_v") is not None and abs(row.get("left", {}).get("edge_v")) < 0.2 else np.nan for row in steps],
            color=color,
            linestyle="-",
        )
        axes[4].plot(
            t,
            [row.get("right", {}).get("edge_v") if row.get("right", {}).get("edge_v") is not None and abs(row.get("right", {}).get("edge_v")) < 0.2 else np.nan for row in steps],
            color=color,
            linestyle="--",
        )
        axes[5].plot(t, [1 if row.get("left", {}).get("present") else 0 for row in steps], color=color, linestyle="-", drawstyle="steps-post")
        axes[5].plot(t, [1 if row.get("right", {}).get("present") else 0 for row in steps], color=color, linestyle="--", drawstyle="steps-post")
        axes[5].plot(t, [1 if row.get("bilateral") else 0 for row in steps], color=color, linestyle=":", drawstyle="steps-post", alpha=0.7)
        marks(axes[0], flags)
    axes[0].set_ylabel("N")
    axes[0].set_title("μ=0.425 matched prefix: F_cmd (dotted) vs realized squeeze=2*min(NL,NR)")
    axes[0].legend(fontsize=8, loc="upper right", ncol=4)
    axes[1].set_ylabel("deg")
    axes[1].set_title("relative object-gripper rotation from settle-start")
    axes[2].set_ylabel("m")
    axes[2].set_title("gripper aperture")
    axes[3].set_ylabel("N")
    axes[3].set_title("NL (solid) / NR (dashed)")
    axes[4].set_ylabel("m")
    axes[4].set_title("grasp_nx in-plane edge margin edge_v (solid left, dashed right)")
    axes[5].set_ylabel("0/1")
    axes[5].set_title("contact validity: left solid / right dashed / bilateral dotted")
    axes[5].set_xlabel("t (s)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)
    return dest


def repeatability_ok(table_425: list[dict], pair: dict) -> tuple[bool, str]:
    if pair.get("n", 0) < 4:
        return False, "fewer than 4 matched 0.75 vs 1.00 pairs"
    # chaotic: retention rates near 0.4-0.6 at every F with no structure
    rates = [row["retention_pct"] for row in table_425]
    if all(20 < r < 80 for r in rates) and max(rates) - min(rates) < 25:
        return False, "retention chaotic across F with no operating range"
    # same-F: if n>=4 and retention is 0 or 100, that's stable; mixed 2/5 vs 3/5 at every F is noisy
    noisy = 0
    for row in table_425:
        p = row["retention_pct"] / 100.0
        if 0.2 < p < 0.8:
            noisy += 1
    if noisy >= 3:
        return False, "same-F retention mixed at most force levels"
    return True, "same-F retention clustered; F=0.75 vs 1.00 comparable across matched prefixes"


def main() -> None:
    root = OUT / "repeats"
    rows = load_metrics(root)
    table = group_table(rows)
    pair = matched_f075_vs_f100(rows)
    table_425 = [row for row in table if abs(row["mu"] - 0.425) < 1e-9]
    ok, reason = repeatability_ok(table_425, pair) if table_425 else (False, "no mu=0.425 rows")
    payload = {
        "n_runs": len(rows),
        "table": table,
        "f075_vs_f100": pair,
        "repeatability_pass": ok,
        "repeatability_reason": reason,
    }
    (OUT / "OPERATING_TABLE.json").write_text(json.dumps(payload, indent=2) + "\n")
    # overlay: prefer repeat 0, else first available
    repeats = sorted({int(str(Path(row["_dir"]).name).split("_")[0][1:]) for row in rows if abs(float(row["friction"]) - 0.425) < 1e-9})
    overlays = []
    for rid in repeats:
        overlay = plot_overlay_mu425(root, rid, OUT / f"overlay_mu0.425_r{rid}.png")
        if overlay:
            overlays.append(str(overlay))
    payload["overlays"] = overlays
    valid = [row for row in rows if row.get("left_slab_settle") == "grasp_nx" and row.get("right_slab_settle") == "grasp_nx"]
    payload["table_valid_prefix"] = group_table(valid)
    payload["n_valid_runs"] = len(valid)
    payload["invalid_prefix_episodes"] = sorted({int(row.get("episode_id")) for row in rows if row.get("left_slab_settle") != "grasp_nx"})
    (OUT / "OPERATING_TABLE.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"n": len(rows), "n_valid": len(valid), "repeatability_pass": ok, "reason": reason, "overlays": overlays}, default=str), flush=True)


if __name__ == "__main__":
    main()
