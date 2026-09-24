"""Attribution tables from filtered grasp-geometry canaries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plot_geom_timelines import load_steps, plot_overlay_425, plot_overlay_575, plot_seven_panel, plot_slab_maps

ROOT = Path(__file__).resolve().parent / "geom_runs"
PHASE_LIFT = {"lift-start", "lift"}
PHASE_POUR = {"wrist-rotation-start", "pour-start", "pour", "terminal"}


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _min(xs):
    xs = [float(x) for x in xs if x is not None]
    return float(np.min(xs)) if xs else None


def summarize_dir(directory: Path) -> dict:
    steps = load_steps(directory)
    summary = json.loads((directory / "SUMMARY.json").read_text()) if (directory / "SUMMARY.json").exists() else {}
    result = json.loads((directory / "result.json").read_text()) if (directory / "result.json").exists() else {}
    flags = summary.get("flags") or {}
    settle = [row for row in steps if row["phase"] == "settle"]
    lift = [row for row in steps if row["phase"] in PHASE_LIFT]
    pour = [row for row in steps if row["phase"] in PHASE_POUR]
    settle_end = settle[-1] if settle else (steps[0] if steps else {})
    motion = [row for row in steps if row["phase"] != "settle"]

    def phase_squeeze(rows):
        return _mean([row["squeeze"] for row in rows[-25:]]) if rows else None

    left_edges = [row["left_edge"] for row in motion if row.get("left_edge") is not None]
    right_edges = [row["right_edge"] for row in motion if row.get("right_edge") is not None]
    max_drel = _max([row.get("drel_from_settle_start_deg") for row in steps])
    max_drel_motion = _max([row.get("drel_from_settle_deg") for row in motion])
    return {
        "dir": directory.name,
        "mu": result.get("friction"),
        "F_cmd": result.get("force_N"),
        "retention": result.get("retention_success"),
        "contact_ratio": result.get("contact_ratio"),
        "settle_squeeze": phase_squeeze(settle),
        "lift_squeeze": phase_squeeze(lift),
        "pour_squeeze": phase_squeeze(pour),
        "settle_nl": settle_end.get("nl"),
        "settle_nr": settle_end.get("nr"),
        "settle_aperture": settle_end.get("aperture"),
        "settle_rel_p": settle_end.get("rel_p"),
        "settle_rel_rpy_deg": None if not settle_end else [float(np.degrees(x)) for x in settle_end.get("rel_rpy") or [0, 0, 0]],
        "settle_left_uv": [settle_end.get("left", {}).get("u"), settle_end.get("left", {}).get("v")] if settle_end else None,
        "settle_right_uv": [settle_end.get("right", {}).get("u"), settle_end.get("right", {}).get("v")] if settle_end else None,
        "settle_left_edge": settle_end.get("left_edge"),
        "settle_right_edge": settle_end.get("right_edge"),
        "settle_left_n": settle_end.get("left", {}).get("n_world") if settle_end else None,
        "settle_right_n": settle_end.get("right", {}).get("n_world") if settle_end else None,
        "settle_drel_from_start_deg": settle_end.get("drel_from_settle_start_deg"),
        "max_relative_angle_from_start_deg": max_drel,
        "max_relative_angle_from_settle_end_deg": max_drel_motion,
        "min_left_edge_distance": _min(left_edges),
        "min_right_edge_distance": _min(right_edges),
        "unilateral_loss": flags.get("T4_unilateral_loss"),
        "bilateral_loss": flags.get("T6_bilateral_loss"),
        "event_order": summary.get("event_order") or result.get("event_order"),
        "flags": {key: {"t": val.get("t"), "phase": val.get("phase"), **{k: val[k] for k in val if k not in {"step"}}} for key, val in flags.items()},
        "table_bottom_settle": settle_end.get("table_bottom_fn"),
        "table_bottom_gone_t": next((row["t"] for row in steps if row["phase"] != "settle" and not row.get("table_bottom")), None),
        "realize": result.get("realize"),
    }


def _max(xs):
    xs = [float(x) for x in xs if x is not None]
    return float(np.max(xs)) if xs else None


def compare_pair(a: dict, b: dict) -> dict:
    def delta(key):
        va, vb = a.get(key), b.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            return {"a": va, "b": vb, "b_minus_a": float(vb) - float(va)}
        return {"a": va, "b": vb}

    return {
        "settle_rel_p": delta("settle_rel_p"),
        "settle_rel_rpy_deg": {"a": a.get("settle_rel_rpy_deg"), "b": b.get("settle_rel_rpy_deg")},
        "settle_drel_from_start_deg": delta("settle_drel_from_start_deg"),
        "settle_left_uv": {"a": a.get("settle_left_uv"), "b": b.get("settle_left_uv")},
        "settle_right_uv": {"a": a.get("settle_right_uv"), "b": b.get("settle_right_uv")},
        "settle_left_edge": delta("settle_left_edge"),
        "settle_right_edge": delta("settle_right_edge"),
        "settle_aperture": delta("settle_aperture"),
        "max_relative_angle_from_start_deg": delta("max_relative_angle_from_start_deg"),
        "min_left_edge_distance": delta("min_left_edge_distance"),
        "min_right_edge_distance": delta("min_right_edge_distance"),
        "event_order_a": a.get("event_order"),
        "event_order_b": b.get("event_order"),
        "unilateral_a": a.get("unilateral_loss"),
        "unilateral_b": b.get("unilateral_loss"),
    }


def matching_keyframes(fail_dir: Path, ok_dir: Path) -> dict:
    fail_sum = json.loads((fail_dir / "SUMMARY.json").read_text())
    ok_steps = load_steps(ok_dir)
    wanted = [
        "settle_end",
        "T0_lift_start",
        "T1_rel_rot_2deg",
        "T2_centroid_near_edge",
        "T4_unilateral_loss",
        "T6_bilateral_loss",
        "T7_retention_loss",
    ]
    rows = []
    flags = fail_sum.get("flags") or {}
    fail_steps = load_steps(fail_dir)
    for name in wanted:
        event = flags.get(name)
        t = None
        if name == "settle_end" and fail_steps:
            settle = [row for row in fail_steps if row["phase"] == "settle"]
            t = settle[-1]["t"] if settle else 0.0
        elif event:
            t = event.get("t")
        if t is None:
            continue
        nearest = min(ok_steps, key=lambda row: abs(row["t"] - t)) if ok_steps else None
        fail_row = min(fail_steps, key=lambda row: abs(row["t"] - t)) if fail_steps else None
        rows.append(
            {
                "event": name,
                "t": t,
                "fail": None
                if fail_row is None
                else {
                    "phase": fail_row["phase"],
                    "nl": fail_row["nl"],
                    "nr": fail_row["nr"],
                    "aperture": fail_row["aperture"],
                    "drel_from_start": fail_row.get("drel_from_settle_start_deg"),
                    "left_edge": fail_row.get("left_edge"),
                    "right_edge": fail_row.get("right_edge"),
                    "left_uv": [fail_row["left"].get("u"), fail_row["left"].get("v")],
                    "right_uv": [fail_row["right"].get("u"), fail_row["right"].get("v")],
                    "left_n": fail_row["left"].get("n_world"),
                    "right_n": fail_row["right"].get("n_world"),
                },
                "success_at_same_t": None
                if nearest is None
                else {
                    "phase": nearest["phase"],
                    "nl": nearest["nl"],
                    "nr": nearest["nr"],
                    "aperture": nearest["aperture"],
                    "drel_from_start": nearest.get("drel_from_settle_start_deg"),
                    "left_edge": nearest.get("left_edge"),
                    "right_edge": nearest.get("right_edge"),
                    "left_uv": [nearest["left"].get("u"), nearest["left"].get("v")],
                    "right_uv": [nearest["right"].get("u"), nearest["right"].get("v")],
                },
            }
        )
    return {"fail_dir": fail_dir.name, "ok_dir": ok_dir.name, "rows": rows}


def main(root: Path = ROOT) -> dict:
    dirs = sorted(path for path in root.glob("mu*_F*") if path.is_dir() and (path / "steps.jsonl").exists())
    rows = [summarize_dir(path) for path in dirs]
    by = {(row["mu"], row["F_cmd"]): row for row in rows}
    pair = None
    keys = None
    if (0.425, 0.75) in by and (0.425, 1.0) in by:
        pair = compare_pair(by[(0.425, 0.75)], by[(0.425, 1.0)])
        keys = matching_keyframes(root / "mu0.425_F1", root / "mu0.425_F0.75")
    trio = [by[key] for key in ((0.575, 1.0), (0.575, 1.25), (0.575, 1.5)) if key in by]
    ref = [by[key] for key in ((0.85, 0.5), (0.85, 1.0)) if key in by]
    for path in dirs:
        plot_seven_panel(path)
        plot_slab_maps(path)
    plot_overlay_425(root)
    plot_overlay_575(root)
    payload = {
        "rows": rows,
        "mu0425_F075_vs_F100": pair,
        "mu0425_matching_keyframes": keys,
        "mu0575_trio": [
            {
                "F_cmd": row["F_cmd"],
                "retention": row["retention"],
                "settle_squeeze": row["settle_squeeze"],
                "lift_squeeze": row["lift_squeeze"],
                "pour_squeeze": row["pour_squeeze"],
                "settle_drel_from_start_deg": row["settle_drel_from_start_deg"],
                "max_relative_angle_from_start_deg": row["max_relative_angle_from_start_deg"],
                "min_left_edge_distance": row["min_left_edge_distance"],
                "min_right_edge_distance": row["min_right_edge_distance"],
                "settle_aperture": row["settle_aperture"],
                "unilateral_loss": row["unilateral_loss"],
                "event_order": row["event_order"],
            }
            for row in trio
        ],
        "mu085_reference": [
            {
                "F_cmd": row["F_cmd"],
                "retention": row["retention"],
                "max_relative_angle_from_start_deg": row["max_relative_angle_from_start_deg"],
                "min_left_edge_distance": row["min_left_edge_distance"],
                "min_right_edge_distance": row["min_right_edge_distance"],
                "unilateral_loss": row["unilateral_loss"],
                "bilateral_loss": row["bilateral_loss"],
                "event_order": row["event_order"],
            }
            for row in ref
        ],
        "tiny_table": [
            {
                "mu": row["mu"],
                "F_cmd": row["F_cmd"],
                "settle_squeeze": row["settle_squeeze"],
                "lift_squeeze": row["lift_squeeze"],
                "pour_squeeze": row["pour_squeeze"],
                "max_relative_angle": row["max_relative_angle_from_start_deg"],
                "min_left_edge_distance": row["min_left_edge_distance"],
                "min_right_edge_distance": row["min_right_edge_distance"],
                "unilateral_loss": 0 if row["unilateral_loss"] is None else 1,
                "bilateral_loss": 0 if row["bilateral_loss"] is None else 1,
                "retention": row["retention"],
            }
            for row in rows
        ],
    }
    (root / "COMPARE.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    main()
