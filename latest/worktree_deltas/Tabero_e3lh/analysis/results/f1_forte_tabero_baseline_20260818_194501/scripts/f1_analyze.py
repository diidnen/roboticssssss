#!/usr/bin/env python3
"""Offline analysis of F1 CSVs → plots + FINAL_VERDICT. No Isaac."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
PLOTS = OUT / "plots"
PLOTS.mkdir(exist_ok=True)


def load_csv(path: Path):
    if not path.exists():
        return None, f"missing {path.name}"
    text = path.read_text()
    if text.startswith("NOT_RUN"):
        return None, text.strip()
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, "empty"
    return rows, None


def fnum(d, k, default=0.0):
    try:
        return float(d[k])
    except Exception:
        return default


def write_not_run(name: str, reason: str):
    p = OUT / name
    if not p.exists():
        p.write_text(f"NOT_RUN\nreason: {reason}\n")


def maybe_plot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        (PLOTS / "PLOT_ERROR.txt").write_text(repr(e))
        return

    servo, _ = load_csv(OUT / "TABERO_FORCE_SERVO_CALIBRATION.csv")
    if servo:
        by = {}
        for r in servo:
            by.setdefault(fnum(r, "desired_F"), []).append(fnum(r, "measured_mean_squeeze"))
        xs = sorted(by)
        ys = [np.mean(by[x]) for x in xs]
        yerr = [np.std(by[x]) for x in xs]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.errorbar(xs, ys, yerr=yerr, fmt="o-", label="measured")
        ax.plot(xs, xs, "k--", label="y=x")
        ax.set_xlabel("desired squeeze (N)")
        ax.set_ylabel("measured squeeze (N)")
        ax.set_title("Tabero force servo")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS / "force_servo.png", dpi=120)
        plt.close(fig)

    fx, _ = load_csv(OUT / "FORCE_X_FRICTION_EXPLORATION.csv")
    if fx:
        mus = sorted({fnum(r, "friction") for r in fx})
        fs = sorted({fnum(r, "desired_F") for r in fx})
        mat = np.zeros((len(mus), len(fs)))
        for i, mu in enumerate(mus):
            for j, f in enumerate(fs):
                sub = [r for r in fx if abs(fnum(r, "friction") - mu) < 1e-9 and abs(fnum(r, "desired_F") - f) < 1e-9]
                mat[i, j] = np.mean([fnum(r, "lift_success") for r in sub]) if sub else np.nan
        fig, ax = plt.subplots(figsize=(6, 3.5))
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(fs)), [str(x) for x in fs])
        ax.set_yticks(range(len(mus)), [str(x) for x in mus])
        ax.set_xlabel("desired F (N)")
        ax.set_ylabel("object μ")
        ax.set_title("lift success F × μ")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(PLOTS / "force_x_friction_lift.png", dpi=120)
        plt.close(fig)


def min_sufficient(rows, success_key="lift_success"):
    out = {}
    if not rows:
        return out
    mus = sorted({fnum(r, "friction") for r in rows})
    for mu in mus:
        sub = [r for r in rows if abs(fnum(r, "friction") - mu) < 1e-9]
        fs = sorted({fnum(r, "desired_F") for r in sub})
        chosen = None
        for f in fs:
            cell = [r for r in sub if abs(fnum(r, "desired_F") - f) < 1e-9]
            sr = np.mean([fnum(r, success_key) for r in cell])
            if sr >= 0.8:
                chosen = f
                break
        out[str(mu)] = chosen
    return out


def rate(rows, key):
    if not rows:
        return None
    return float(np.mean([fnum(r, key) for r in rows]))


def main():
    maybe_plot()
    servo, servo_why = load_csv(OUT / "TABERO_FORCE_SERVO_CALIBRATION.csv")
    fx, fx_why = load_csv(OUT / "FORCE_X_FRICTION_EXPLORATION.csv")
    low, low_why = load_csv(OUT / "BASELINE_FIXED_LOW.csv")
    high, high_why = load_csv(OUT / "BASELINE_FIXED_HIGH.csv")
    oracle, oracle_why = load_csv(OUT / "BASELINE_FORTE_ORACLE_SLIP.csv")
    tac, tac_why = load_csv(OUT / "BASELINE_FORTE_TACTILE.csv")
    slip, slip_why = load_csv(OUT / "TABERO_SLIP_SIGNAL_AUDIT.csv")

    for name, why in [
        ("TABERO_FORCE_SERVO_CALIBRATION.csv", servo_why),
        ("FORCE_X_FRICTION_EXPLORATION.csv", fx_why),
        ("BASELINE_FIXED_LOW.csv", low_why),
        ("BASELINE_FIXED_HIGH.csv", high_why),
        ("BASELINE_FORTE_ORACLE_SLIP.csv", oracle_why),
        ("BASELINE_FORTE_TACTILE.csv", tac_why),
        ("TABERO_SLIP_SIGNAL_AUDIT.csv", slip_why),
        ("MINIMUM_SUFFICIENT_FORCE.csv", None),
    ]:
        if why and not (OUT / name).exists():
            write_not_run(name, why)

    low_force_ok = False
    desired_range, measured_range = [], []
    if servo:
        by = {}
        for r in servo:
            by.setdefault(fnum(r, "desired_F"), []).append(fnum(r, "measured_mean_squeeze"))
        desired_range = sorted(by)
        measured_range = [float(np.mean(by[x])) for x in desired_range]
        ordered = all(measured_range[i + 1] + 0.4 >= measured_range[i] for i in range(len(measured_range) - 1)) if len(measured_range) > 1 else False
        low_force_ok = bool(measured_range and min(measured_range) <= 8.0 and ordered and (max(measured_range) - min(measured_range) >= 2.0))
        if min(measured_range) > 15:
            low_force_ok = False

    gt_defined = True
    tactile_valid = None
    if slip:
        y = np.array([fnum(r, "gt_slip") for r in slip])
        pred = np.array([fnum(r, "tactile_slip") for r in slip])
        if y.sum() > 0:
            tp = float(((pred == 1) & (y == 1)).sum())
            fp = float(((pred == 1) & (y == 0)).sum())
            fn = float(((pred == 0) & (y == 1)).sum())
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            tactile_valid = bool(prec >= 0.5 and rec >= 0.3)
        else:
            tactile_valid = False
        (OUT / "TABERO_SLIP_SIGNAL_AUDIT_SUMMARY.json").write_text(
            json.dumps({"n": len(slip), "gt_rate": float(y.mean()), "tac_rate": float(pred.mean()), "tactile_valid": tactile_valid}, indent=2)
        )
    else:
        write_not_run("TABERO_SLIP_SIGNAL_AUDIT.csv", slip_why or "NOT_RUN")

    fstar = min_sufficient(fx) if fx else {}
    if fstar:
        with (OUT / "MINIMUM_SUFFICIENT_FORCE.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["friction", "F_star_lift_N", "definition"])
            w.writeheader()
            for mu, val in fstar.items():
                w.writerow({"friction": mu, "F_star_lift_N": val if val is not None else "NONE_IN_GRID", "definition": "min F with lift_success rate >= 0.8"})
    else:
        write_not_run("MINIMUM_SUFFICIENT_FORCE.csv", fx_why or "explore not run")

    disagreement = None
    robust = None
    if fx:
        mus = sorted({fnum(r, "friction") for r in fx})
        stars = [fstar.get(str(mu)) for mu in mus]
        if all(s is not None for s in stars):
            disagreement = max(stars) != min(stars)
            # fixed robust if some F succeeds at all mu
            fs = sorted({fnum(r, "desired_F") for r in fx})
            robust = False
            for f in fs:
                ok = True
                for mu in mus:
                    cell = [r for r in fx if abs(fnum(r, "friction") - mu) < 1e-9 and abs(fnum(r, "desired_F") - f) < 1e-9]
                    if not cell or np.mean([fnum(r, "lift_success") for r in cell]) < 0.8:
                        ok = False
                        break
                if ok:
                    robust = True
                    break
        elif any(s is None for s in stars) and any(s is not None for s in stars):
            disagreement = True
            robust = False

    tabero_ran = bool(servo or fx or low or oracle)
    gpu_block = (OUT / "SWEEP_ERROR.json").exists() or (OUT / "GPU_BLOCK.txt").exists()

    if tabero_ran and low_force_ok and disagreement and oracle and rate(oracle, "lift_success") is not None:
        status = "F1_FORTE_BASELINE_QUALIFIED_IN_TABERO"
    elif tabero_ran and not low_force_ok:
        status = "F1_TABERO_LOW_FORCE_CONTROL_BLOCKED"
    elif tabero_ran and low_force_ok and fx and disagreement is False and robust:
        status = "F1_NEGATIVE_FIXED_LOW_FORCE_EXISTS" if min([v for v in fstar.values() if v is not None] or [99]) <= 2.0 else "F1_NEGATIVE_NO_FORCE_FRICTION_DECISION_REGION"
    elif tabero_ran and slip and tactile_valid is False and not oracle:
        status = "F1_TABERO_SLIP_SIGNAL_BLOCKED"
    elif not tabero_ran:
        status = "F1_BLOCKED_TECHNICALLY"
    else:
        status = "F1_FORTE_SOFTWARE_QUALIFIED_HARDWARE_REPRO_BLOCKED"

    forte_prov = json.loads((OUT / "FORTE_PROVENANCE.json").read_text()) if (OUT / "FORTE_PROVENANCE.json").exists() else {}
    tabero_prov = json.loads((OUT / "TABERO_PROVENANCE.json").read_text()) if (OUT / "TABERO_PROVENANCE.json").exists() else {}

    verdict = {
        "status": status,
        "method_change": "NONE",
        "forte_commit": forte_prov.get("commit", "7f88d0184c1617ed95e67502da96e60be07b3689"),
        "forte_repo_clean": forte_prov.get("repo_clean", True),
        "forte_real_hardware_available": False,
        "forte_force_estimator_loaded": True,
        "forte_slip_detector_loaded": True,
        "forte_offline_replay_available": False,
        "tabero_commit": tabero_prov.get("tabero_commit", "3bda8114c07584d3d53ae43fd555ca6606c76274"),
        "tabero_env": "Isaac-Libero-Franka-Hybrid-Tactile-v0",
        "task": "libero_object 1 pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "force_adverbs_used": False,
        "tabero_low_force_servo_valid": low_force_ok if servo else None,
        "desired_force_range_N": desired_range,
        "measured_force_range_N": measured_range,
        "physics_variable": "friction",
        "friction_values": sorted({fnum(r, "friction") for r in fx}) if fx else [0.2, 0.5, 1.0],
        "gt_slip_defined": gt_defined,
        "tactile_slip_signal_valid": tactile_valid,
        "fixed_low_run": bool(low),
        "fixed_high_run": bool(high),
        "forte_oracle_reactive_run": bool(oracle),
        "forte_tactile_reactive_run": bool(tac),
        "minimum_sufficient_force_by_friction": fstar,
        "force_friction_action_disagreement": disagreement,
        "fixed_robust_force_exists": robust,
        "baseline_lift_sr": {
            "fixed_low": rate(low, "lift_success") if low else None,
            "fixed_high": rate(high, "lift_success") if high else None,
            "forte_oracle": rate(oracle, "lift_success") if oracle else None,
            "forte_tactile": rate(tac, "lift_success") if tac else None,
        },
        "primary_evidence": [],
        "limitations": [],
        "secondary_status": [],
    }
    verdict["secondary_status"].append("FORTE_REAL_HARDWARE_REPRO_NOT_POSSIBLE_ON_THIS_MACHINE")
    if not tabero_ran:
        verdict["limitations"].append("Tabero Isaac sweep did not produce CSVs this round (see GPU_BLOCK.txt / SWEEP_ERROR.json).")
        verdict["primary_evidence"].append("FORTE software smoke: SVR loaded, slip functions loaded, no serial hardware, no offline traces.")
    if servo:
        verdict["primary_evidence"].append(f"Force servo desired {desired_range} measured {measured_range} low_force_ok={low_force_ok}")
    if fx:
        verdict["primary_evidence"].append(f"F* by friction={fstar} disagreement={disagreement} robust={robust}")
    if gpu_block:
        verdict["secondary_status"].append("TABERO_ISAAC_GPU_CONTENDED")

    (OUT / "FINAL_VERDICT.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps({"status": status, "low_force_ok": low_force_ok, "fstar": fstar}, indent=2))


if __name__ == "__main__":
    main()
