"""Summarize force-realize one-seed sweep. Does not overwrite official records."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "main_raw"
FORCES = [round(0.25 * i, 2) for i in range(1, 21)]
FRICS = [0.425, 0.575, 0.85]


def load_rows() -> list[dict]:
    rows = []
    for result in RAW.glob("*/job/FORCE_REALIZE_CONTEXT_RESULT.json"):
        payload = json.loads(result.read_text())
        context = payload["context"]
        for outcome in payload["outcomes"]:
            detail = {}
            for path in result.parent.glob("branch_*N/result.json"):
                item = json.loads(path.read_text())
                if abs(float(item.get("force_setpoint_single_finger_N", -1)) - float(outcome["force_N"])) < 1e-9:
                    detail = item
                    break
            commanded = float(outcome["force_N"])
            measured = outcome.get("measured_force_mean_n") or detail.get("measured_force_mean_n")
            pre = outcome.get("pre_motion_squeeze_mean_n") or detail.get("realize_pre_motion_squeeze_mean_n")
            rel = None
            if measured is not None:
                rel = abs(float(measured) - commanded) / max(commanded, 1e-6)
            rows.append(
                {
                    "seed": int(context["seed"]),
                    "mu_object": float(context["friction"]),
                    "mu_eff": float(context.get("mu_eff_average") or 0.5 * (0.3 + context["friction"])),
                    "force_N": commanded,
                    "retention": int(outcome.get("retention_success") or 0),
                    "official": int(outcome.get("official_full_task_success") or 0),
                    "drop": int(detail.get("drop") or outcome.get("drop") or 0),
                    "contact_ratio": outcome.get("contact_ratio"),
                    "squeeze_mean": measured,
                    "pre_motion_squeeze_mean": pre,
                    "force_realized_before_motion": bool(
                        outcome.get("force_realized_before_motion")
                        or detail.get("realize_force_realized_before_motion")
                    ),
                    "remainder_force_tracked": bool(
                        outcome.get("remainder_force_tracked") or detail.get("remainder_force_tracked")
                    ),
                    "squeeze_rel_error": rel,
                    "n_balls": outcome.get("n_balls_in_official_band") or detail.get("n_balls_in_official_band"),
                    "deskbin_z": outcome.get("deskbin_z") or detail.get("deskbin_z"),
                    "unstable": bool(outcome.get("unstable_layout") or detail.get("unstable_layout")),
                }
            )
    return rows


def fmin_reliable(pairs: list[tuple[float, int]]) -> float | None:
    pairs = sorted(pairs)
    found = None
    for force, success in pairs:
        if all(flag == 1 for _force, flag in pairs if _force >= force):
            found = force
            break
    return found


def monotonic_nondecreasing(flags: list[int]) -> bool:
    best = 0
    for flag in flags:
        if flag < best:
            return False
        best = max(best, flag)
    return True


def main() -> None:
    rows = load_rows()
    table = []
    verdicts = {}
    for seed in sorted({row["seed"] for row in rows}):
        verdicts[seed] = {}
        for mu in FRICS:
            subset = [row for row in rows if row["seed"] == seed and abs(row["mu_object"] - mu) < 1e-9]
            subset = sorted(subset, key=lambda row: row["force_N"])
            if not subset:
                continue
            ret = [int(row["retention"]) for row in subset]
            off = [int(row["official"]) for row in subset]
            fmin = fmin_reliable([(row["force_N"], int(row["retention"])) for row in subset])
            rels = [row["squeeze_rel_error"] for row in subset if row["squeeze_rel_error"] is not None]
            verdicts[seed][str(mu)] = {
                "n": len(subset),
                "retention": ret,
                "official": off,
                "F_min_retention": fmin,
                "retention_monotonic_nondecreasing": monotonic_nondecreasing(ret),
                "official_monotonic_nondecreasing": monotonic_nondecreasing(off),
                "squeeze_mean": [row["squeeze_mean"] for row in subset],
                "pre_motion_squeeze_mean": [row["pre_motion_squeeze_mean"] for row in subset],
                "n_pre_motion_realized": int(sum(1 for row in subset if row["force_realized_before_motion"])),
                "n_remainder_tracked": int(sum(1 for row in subset if row["remainder_force_tracked"])),
                "median_remainder_rel_error": (sorted(rels)[len(rels) // 2] if rels else None),
                "n_balls": [row["n_balls"] for row in subset],
                "drop": [row["drop"] for row in subset],
            }
            table.extend(subset)
    (HERE / "SWEEP_ROWS.json").write_text(json.dumps(table, indent=2) + "\n")
    (HERE / "SWEEP_SUMMARY.json").write_text(json.dumps(verdicts, indent=2, sort_keys=True) + "\n")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("no matplotlib", exc)
        return
    if 200014 not in verdicts:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=140)
    for mu, data in verdicts[200014].items():
        forces = FORCES[: len(data["retention"])]
        axes[0].plot(forces, data["retention"], marker="o", label=f"μ={mu}")
        axes[1].plot(forces, data["official"], marker="o", label=f"μ={mu}")
        axes[2].plot(forces, data["squeeze_mean"], marker="o", label=f"μ={mu} remainder")
        axes[2].plot(forces, data["pre_motion_squeeze_mean"], marker="x", linestyle="--", label=f"μ={mu} pre")
    axes[2].plot(FORCES, FORCES, color="gray", linewidth=1, label="F_cmd")
    axes[0].set_title("retention")
    axes[1].set_title("official dump")
    axes[2].set_title("measured squeeze")
    for axis in axes:
        axis.set_xlabel("F (N)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(HERE / "FIG_200014_force_realize.png")
    print("wrote", HERE / "FIG_200014_force_realize.png")
    print(json.dumps(verdicts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
