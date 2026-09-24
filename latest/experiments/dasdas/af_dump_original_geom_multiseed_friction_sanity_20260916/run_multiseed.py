#!/usr/bin/env python3
"""Multi-seed original-geom finger-μ retention sanity.

Frozen protocol from af_dump_original_geom_finger_friction_sanity_20260916.
Does not recapture 32×20, overwrite 18/19, or Fig.B.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = Path("/media/volume/dasdas/exouser/af_dump_original_geom_finger_friction_sanity_20260916")
sys.path.insert(0, str(OLD))

import run_original_geom_finger_sweep as S

DESK_MU = 0.30
LEVELS = {
    "low": {"mu_finger": 0.425, "mu_eff": 0.3625, "mu_deskbin_fixed": DESK_MU},
    "mid": {"mu_finger": 0.575, "mu_eff": 0.4375, "mu_deskbin_fixed": DESK_MU},
    "high": {"mu_finger": 0.85, "mu_eff": 0.575, "mu_deskbin_fixed": DESK_MU},
}
LEVEL_ORDER = ["low", "mid", "high"]
CANDIDATES = [200002, 200010, 200019, 200020, 200021, 200022, 200023, 200024, 200025, 200026, 200027, 200028, 200029, 200030]
EXCLUDED = [200003, 200004, 200005, 200006, 200007, 200008, 200009, 200011, 200012, 200013, 200015, 200016, 200017, 200018]
EPISODES = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]
TARGET_NEW = 4
LOW_F = [round(0.25 * i, 2) for i in range(1, 7)]
MID_F = [round(0.25 * i, 2) for i in range(7, 15)]
HIGH_F = [round(0.25 * i, 2) for i in range(15, 21)]
DETAIL_F = {0.25, 0.75, 2.5}


def dump(path: Path, value) -> None:
    S.dump_json(path, value)


def slim_finger(geom: dict | None) -> dict | None:
    if not geom:
        return None
    fingers = {}
    for name, row in (geom.get("per_finger") or {}).items():
        if not row.get("present"):
            continue
        fingers[name] = {
            "n_shapes": row.get("n_shapes"),
            "shape_ids": row.get("shape_ids"),
            "axes": row.get("axes"),
            "fn": row.get("fn"),
            "wedging": row.get("multi_hull_wedging"),
            "normals": row.get("normals"),
        }
    return {
        "shape_ids": geom.get("contact_shape_ids"),
        "pair_dist": geom.get("pair_dist"),
        "aperture": geom.get("aperture"),
        "squeeze": geom.get("squeeze"),
        "nl": geom.get("nl"),
        "nr": geom.get("nr"),
        "obj_xyz": geom.get("obj_xyz"),
        "obj_z": geom.get("obj_z"),
        "wedging": geom.get("multi_hull_wedging_any_finger"),
        "nx_pz": geom.get("split_like_nx_pz"),
        "nx_nz_pz": geom.get("split_like_nx_nz_pz"),
        "mu_eff_finger": geom.get("finger_deskbin_mu_eff"),
        "mu_eff_ball": geom.get("ball_deskbin_mu_eff"),
        "mu_eff_table": geom.get("table_deskbin_mu_eff"),
        "fingers": fingers,
    }


def env_ok(seed: int, episode_id: int = 0) -> dict:
    task = None
    try:
        task = S.make_task(seed, episode_id)
        geom = S.assert_official_geometry(task)
        return {
            "ok": True,
            "seed": seed,
            "episode_id": episode_id,
            "deskbin_id": geom["deskbin_id"],
            "n_shapes": geom["n_shapes"],
            "mass_kg": geom["mass_kg"],
        }
    except Exception as exc:
        return {
            "ok": False,
            "seed": seed,
            "episode_id": episode_id,
            "error": repr(exc),
            "unstable": type(exc).__name__ == "UnStableError" or "UnStableError" in repr(exc),
        }
    finally:
        if task is not None:
            try:
                task.close_env()
            except Exception:
                pass


def pre_only(seed: int, episode_id: int, label: str) -> dict:
    spec = LEVELS[label]
    task = None
    try:
        task = S.make_task(seed, episode_id)
        S.prepare_condition(task, DESK_MU, spec["mu_finger"])
        geom0 = S.assert_official_geometry(task)
        task.activate_activeforcing_candidate_force()
        S.scripted_establish_grasp(task)
        window = []
        for _ in range(20):
            task.scene.step()
            window.append(S.pack_state(task))
        gate = S.classify_pre(window)
        pre_geom = S.contact_geometry(task)
        audit = S.materials_audit(task)
        payload = {
            "ok": True,
            "seed": seed,
            "episode_id": episode_id,
            "label": label,
            "deskbin_id": geom0["deskbin_id"],
            "n_shapes": geom0["n_shapes"],
            "mu_finger": spec["mu_finger"],
            "mu_eff": spec["mu_eff"],
            "gate": gate["gate"],
            "pre_gate": gate,
            "pre_end": window[-1] if window else {},
            "pre_geom": slim_finger(pre_geom),
            "audit": {
                "finger_mu": audit["finger_mu"],
                "deskbin_mu": audit["deskbin_mu"],
                "table_mu": audit["table_mu"],
                "ball_mu": audit["ball_mu"],
                "mu_eff": audit["mu_eff_average"],
            },
        }
        print(json.dumps({"event": "PRE", "seed": seed, "episode_id": episode_id, "label": label, "gate": gate["gate"], "cr": gate["window_cr"]}), flush=True)
        return payload
    except Exception as exc:
        print(json.dumps({"event": "PRE_ERROR", "seed": seed, "episode_id": episode_id, "label": label, "error": repr(exc)}), flush=True)
        return {
            "ok": False,
            "seed": seed,
            "episode_id": episode_id,
            "label": label,
            "gate": "ERROR",
            "error": repr(exc),
            "unstable": type(exc).__name__ == "UnStableError" or "UnStableError" in repr(exc),
        }
    finally:
        if task is not None:
            try:
                task.close_env()
            except Exception:
                pass


def screen_seed(seed: int) -> dict:
    rows = []
    for episode_id in EPISODES:
        by = {}
        failed = False
        for label in LEVEL_ORDER:
            rec = pre_only(seed, episode_id, label)
            by[label] = rec
            rows.append(rec)
            if rec.get("unstable"):
                return {
                    "seed": seed,
                    "paired": False,
                    "reason": "UnStableError",
                    "episode_id": None,
                    "by_level": by,
                    "attempts": rows,
                }
            if rec.get("gate") != "VALID":
                failed = True
                break
        if not failed and all(by[k].get("gate") == "VALID" for k in LEVEL_ORDER):
            return {
                "seed": seed,
                "paired": True,
                "reason": "PRE_VALID at low/mid/high on first eligible episode",
                "episode_id": episode_id,
                "deskbin_id": by["mid"]["deskbin_id"],
                "n_shapes": by["mid"]["n_shapes"],
                "by_level": {k: {"gate": by[k]["gate"], "pre_gate": by[k]["pre_gate"], "pre_geom": by[k]["pre_geom"], "audit": by[k]["audit"], "deskbin_id": by[k]["deskbin_id"]} for k in LEVEL_ORDER},
                "attempts": rows,
            }
    return {
        "seed": seed,
        "paired": False,
        "reason": "no episode with PRE_VALID at all three μ",
        "episode_id": None,
        "by_level": {},
        "attempts": rows,
    }


def curve_stats(outcomes: list[dict]) -> dict:
    forces = [float(o["force_N"]) for o in outcomes]
    bits = [int(o["retention_success"]) for o in outcomes]
    by = {float(o["force_N"]): int(o["retention_success"]) for o in outcomes}

    def rate(region):
        vals = [by[f] for f in region if f in by]
        return None if not vals else float(np.mean(vals))

    longest = 0
    cur = 0
    start = None
    best = None
    for f, b in zip(forces, bits):
        if b:
            if cur == 0:
                start = f
            cur += 1
            if cur > longest:
                longest = cur
                best = {"n": cur, "F_start": start, "F_end": f}
        else:
            cur = 0
    reversals = 0
    seen_one = False
    for prev, nxt in zip(bits, bits[1:]):
        if prev == 1:
            seen_one = True
        if seen_one and prev != nxt:
            reversals += 1
    monotone = True
    peak = 0
    for b in bits:
        if b < peak:
            monotone = False
            break
        peak = max(peak, b)
    return {
        "n_retention": int(sum(bits)),
        "n_points": len(bits),
        "rate_all": float(np.mean(bits)) if bits else 0.0,
        "rate_lowF": rate(LOW_F),
        "rate_midF": rate(MID_F),
        "rate_highF": rate(HIGH_F),
        "first_success_F": S.fmin_first(forces, bits),
        "longest_success_interval": best or {"n": 0, "F_start": None, "F_end": None},
        "n_reversals": reversals,
        "monotone_nondecreasing": monotone,
        "bits": bits,
        "forces": forces,
    }


def compare_pair(a: float, b: float, n: int = 20) -> str:
    if a is None or b is None:
        return "?"
    da = a * n
    db = b * n
    if da <= db - 2:
        return "<"
    if db <= da - 2:
        return ">"
    return "~="


def ordering_of(stats: dict) -> dict:
    rl, rm, rh = stats["low"]["rate_all"], stats["mid"]["rate_all"], stats["high"]["rate_all"]
    return {
        "low_vs_mid": compare_pair(rl, rm),
        "mid_vs_high": compare_pair(rm, rh),
        "low_vs_high": compare_pair(rl, rh),
        "pattern": f"low {compare_pair(rl, rm)} mid {compare_pair(rm, rh)} high",
        "high_much_worse_than_mid": bool(stats["high"]["n_retention"] <= stats["mid"]["n_retention"] - 5),
    }


def run_level(seed: int, episode_id: int, label: str) -> dict:
    spec = LEVELS[label]
    out = HERE / "sweeps" / f"seed{seed}_{label}_CONTEXT.json"
    captured = S.find_and_capture(seed, DESK_MU, spec["mu_finger"], label, 1, episode_id)
    task = captured["task"]
    outcomes = []
    try:
        for force in S.OFFICIAL_FORCES:
            captured["snapshot"].restore()
            S.prepare_condition(task, DESK_MU, spec["mu_finger"])
            detail = float(force) in DETAIL_F
            row = S.run_one(task, force, captured["stock"], DESK_MU, spec["mu_finger"], geom_f025=detail)
            outcomes.append(row)
            print(
                json.dumps(
                    {
                        "event": "F",
                        "seed": seed,
                        "label": label,
                        "F": force,
                        "ret": row["retention_success"],
                        "task": row["official_full_task_success"],
                        "CR": row["contact_ratio"],
                        "squeeze": row["settle_squeeze_mean_n"],
                        "qCR": (captured["query"] or {}).get("contact_ratio"),
                    }
                ),
                flush=True,
            )
        stats = curve_stats(outcomes)
        context = {
            "label": label,
            "seed": seed,
            "episode_id": captured["episode_id"],
            "deskbin_id": captured["deskbin_id"],
            "n_shapes": captured["n_shapes"],
            "mu_finger": spec["mu_finger"],
            "mu_deskbin_fixed": DESK_MU,
            "mu_eff": spec["mu_eff"],
            "audit": captured["audit"],
            "pre_gate": captured["pre_gate"],
            "pre_end": captured["pre_end"],
            "pre_geom": slim_finger(captured["pre_geom"]),
            "query": {
                "contact_ratio": (captured["query"] or {}).get("contact_ratio"),
                "query_failed": bool((captured["query"] or {}).get("query_failed")),
                "error": (captured["query"] or {}).get("error"),
            },
            "post_query": captured["post_query"],
            "post_query_geom": slim_finger(captured["post_query_geom"]),
            "outcomes": outcomes,
            "stats": stats,
        }
        dump(out, context)
        return context
    finally:
        try:
            task.close_env()
        except Exception:
            pass


def ingest_200014() -> dict:
    by = {}
    for label in LEVEL_ORDER:
        raw = json.loads((OLD / "sweeps" / f"{label}_CONTEXT.json").read_text())
        outcomes = raw["outcomes"]
        by[label] = {
            "label": label,
            "seed": 200014,
            "episode_id": raw["episode_id"],
            "deskbin_id": raw["deskbin_id"],
            "n_shapes": raw["n_shapes"],
            "mu_finger": raw["mu_finger"],
            "mu_deskbin_fixed": raw["mu_deskbin_fixed"],
            "mu_eff": raw["mu_eff"],
            "audit": raw.get("audit"),
            "pre_gate": raw["pre_gate"],
            "pre_end": raw.get("pre_end"),
            "pre_geom": slim_finger(raw.get("pre_geom")),
            "query": raw.get("query"),
            "post_query": raw.get("post_query"),
            "post_query_geom": slim_finger(raw.get("post_query_geom")),
            "outcomes": outcomes,
            "stats": curve_stats(outcomes),
            "reused": True,
        }
        dump(HERE / "sweeps" / f"seed200014_{label}_CONTEXT.json", by[label])
    return pack_seed(200014, 1, by, reused=True)


def pack_seed(seed: int, episode_id: int, by: dict, reused: bool = False) -> dict:
    stats = {k: by[k]["stats"] for k in LEVEL_ORDER}
    order = ordering_of(stats)
    return {
        "seed": seed,
        "episode_id": episode_id,
        "deskbin_id": by["mid"].get("deskbin_id"),
        "reused": reused,
        "pre": {k: by[k]["pre_gate"]["gate"] for k in LEVEL_ORDER},
        "qCR": {k: (by[k].get("query") or {}).get("contact_ratio") for k in LEVEL_ORDER},
        "stats": stats,
        "ordering": order,
        "retention_bits": {k: stats[k]["bits"] for k in LEVEL_ORDER},
        "n_retention": {k: stats[k]["n_retention"] for k in LEVEL_ORDER},
        "pre_geom": {k: by[k].get("pre_geom") for k in LEVEL_ORDER},
        "post_query_geom": {k: by[k].get("post_query_geom") for k in LEVEL_ORDER},
    }


def classify_anomaly(seed_pack: dict, by: dict) -> dict | None:
    if not seed_pack["ordering"]["high_much_worse_than_mid"]:
        return None
    notes = []
    high_pre = seed_pack["pre_geom"].get("high") or {}
    mid_pre = seed_pack["pre_geom"].get("mid") or {}
    if high_pre.get("wedging") or high_pre.get("nx_pz") or high_pre.get("nx_nz_pz"):
        notes.append("high PRE already multi-axis / split-like wrapping")
    if mid_pre.get("wedging") or mid_pre.get("nx_pz"):
        notes.append("mid PRE also wrapped")
    hp = high_pre.get("pair_dist")
    mp = mid_pre.get("pair_dist")
    if hp is not None and mp is not None and abs(hp - mp) > 0.005:
        notes.append(f"PRE pair distance shifted high={hp:.4f} vs mid={mp:.4f}")
    ha = high_pre.get("aperture")
    ma = mid_pre.get("aperture")
    if ha is not None and ma is not None and abs(ha - ma) > 0.002:
        notes.append(f"PRE aperture shifted high={ha:.4f} vs mid={ma:.4f}")
    hxyz = high_pre.get("obj_xyz")
    mxyz = mid_pre.get("obj_xyz")
    if hxyz and mxyz:
        delta = float(np.linalg.norm(np.asarray(hxyz) - np.asarray(mxyz)))
        if delta > 0.01:
            notes.append(f"PRE deskbin pose shifted {delta:.4f} m")
    high_q = seed_pack["post_query_geom"].get("high") or {}
    mid_q = seed_pack["post_query_geom"].get("mid") or {}
    if high_q.get("squeeze") is not None and mid_q.get("squeeze") is not None:
        if abs(float(high_q["squeeze"]) - float(mid_q["squeeze"])) > 2.0:
            notes.append(f"post-query squeeze high={high_q['squeeze']:.2f} vs mid={mid_q['squeeze']:.2f}")
    # flicker: high has isolated 1s
    hbits = seed_pack["retention_bits"]["high"]
    isolated = sum(1 for i, b in enumerate(hbits) if b and (i == 0 or not hbits[i - 1]) and (i == len(hbits) - 1 or not hbits[i + 1]))
    if isolated >= 2:
        notes.append(f"high retention is flicker-like ({isolated} isolated successes)")
    # squeeze-induced ejection: high realize squeeze ok but CR collapses
    high_outs = by["high"]["outcomes"]
    mid_outs = {float(o["force_N"]): o for o in by["mid"]["outcomes"]}
    eject = 0
    for o in high_outs:
        m = mid_outs.get(float(o["force_N"]))
        if m and int(m["retention_success"]) == 1 and int(o["retention_success"]) == 0:
            sq = float(o.get("settle_squeeze_mean_n") or 0.0)
            if sq >= 0.5 * float(o["force_N"]) and float(o.get("contact_ratio") or 0) < 0.3:
                eject += 1
    if eject >= 3:
        notes.append(f"high often realizes squeeze then loses contact ({eject} mid-success F points)")
    hf = high_pre.get("fingers") or {}
    n_multi = sum(1 for v in hf.values() if int(v.get("n_shapes") or 0) >= 2)
    if n_multi:
        notes.append(f"high PRE: {n_multi} finger(s) touch 2+ hulls")
    if not notes:
        notes.append("no clear PRE geometry shift; likely contact-mode / downstream sticking or flicker")
    return {
        "seed": seed_pack["seed"],
        "n_mid": seed_pack["n_retention"]["mid"],
        "n_high": seed_pack["n_retention"]["high"],
        "notes": notes,
        "high_pre": high_pre,
        "mid_pre": mid_pre,
        "high_post_query": high_q,
        "mid_post_query": mid_q,
    }


def aggregate(seeds: list[dict]) -> dict:
    forces = S.OFFICIAL_FORCES
    p = {k: [] for k in LEVEL_ORDER}
    for f in forces:
        for k in LEVEL_ORDER:
            vals = []
            for seed in seeds:
                bits = dict(zip(seed["stats"][k]["forces"], seed["stats"][k]["bits"]))
                vals.append(bits[f])
            p[k].append(float(np.mean(vals)))
    rates = {
        k: {
            "all": float(np.mean([s["stats"][k]["rate_all"] for s in seeds])),
            "lowF": float(np.mean([s["stats"][k]["rate_lowF"] for s in seeds])),
            "midF": float(np.mean([s["stats"][k]["rate_midF"] for s in seeds])),
            "highF": float(np.mean([s["stats"][k]["rate_highF"] for s in seeds])),
        }
        for k in LEVEL_ORDER
    }
    n_high_worse = sum(1 for s in seeds if s["ordering"]["high_much_worse_than_mid"])
    n_low_hardest = sum(1 for s in seeds if s["ordering"]["low_vs_mid"] == "<" and s["ordering"]["low_vs_high"] == "<")
    n_full_ladder = sum(1 for s in seeds if s["ordering"]["low_vs_mid"] == "<" and s["ordering"]["mid_vs_high"] == "<")
    low_lt_mid = rates["low"]["all"] < rates["mid"]["all"] - 0.05
    mid_lt_high = rates["mid"]["all"] < rates["high"]["all"] - 0.05
    high_lt_mid = rates["high"]["all"] < rates["mid"]["all"] - 0.05
    if low_lt_mid and mid_lt_high:
        verdict = "PASS"
        why = "Across PRE_VALID seeds, aggregate retention is low < mid < high. 200014 high<mid looks like a seed effect."
    elif n_low_hardest >= max(2, len(seeds) - 1) and not mid_lt_high:
        verdict = "MIXED"
        why = "Low friction is consistently hardest, but mid/high ordering is unstable across seeds."
    elif not low_lt_mid:
        verdict = "FAIL"
        why = "low / mid / high do not separate stably; dump grasp is not usable as friction-to-force evidence."
    else:
        verdict = "MIXED"
        why = "Low is harder on aggregate, but high does not sit above mid. Geometry/seed effects remain large."
    if n_high_worse >= max(2, (len(seeds) + 1) // 2) and verdict == "PASS":
        verdict = "MIXED"
        why = "Mean ladder appeared but high << mid still recurs on multiple seeds."
    return {
        "n_seeds": len(seeds),
        "forces": forces,
        "p_retain": p,
        "mean_rates": rates,
        "n_low_hardest": n_low_hardest,
        "n_full_ladder": n_full_ladder,
        "n_high_much_worse_than_mid": n_high_worse,
        "low_lt_mid": low_lt_mid,
        "mid_lt_high": mid_lt_high,
        "high_lt_mid": high_lt_mid,
        "verdict": verdict,
        "why": why,
    }


def plot_all(seeds: list[dict], agg: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"low": "#b42318", "mid": "#b54708", "high": "#027a48"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    F = agg["forces"]
    for k in LEVEL_ORDER:
        axes[0].plot(F, agg["p_retain"][k], marker="o", color=colors[k], label=k, linewidth=2)
    axes[0].set_title("Aggregate P(retain | F, μ_eff)")
    axes[0].set_xlabel("F_cmd (N)")
    axes[0].set_ylabel("P(retain)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    names = [str(s["seed"]) for s in seeds]
    x = np.arange(len(names))
    w = 0.25
    for i, k in enumerate(LEVEL_ORDER):
        axes[1].bar(x + (i - 1) * w, [s["n_retention"][k] for s in seeds], width=w, color=colors[k], label=k)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("n retain / 20")
    axes[1].set_title("Per-seed retention count")
    axes[1].set_ylim(0, 20)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()
    fig.suptitle(f"Official collider, finger μ only · {agg['verdict']}")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for s in seeds:
        for k in LEVEL_ORDER:
            ax.step(F, s["retention_bits"][k], where="mid", color=colors[k], alpha=0.25, linewidth=1)
    for k in LEVEL_ORDER:
        ax.plot(F, agg["p_retain"][k], color=colors[k], linewidth=2.5, label=f"{k} aggregate")
    ax.set_xlabel("F_cmd (N)")
    ax.set_ylabel("retain")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Seed curves (faint) and aggregate P(retain)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out.with_name("seed_and_aggregate_retention.png"), dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["screen", "sweep", "all"], default="all")
    parser.add_argument("--target-new", type=int, default=TARGET_NEW)
    args = parser.parse_args()
    dump(HERE / "EXCLUSIONS.json", {"excluded": EXCLUDED, "candidates": CANDIDATES})

    screen_path = HERE / "PRE_SCREEN.json"
    if args.phase in {"screen", "all"}:
        screens = []
        paired = []
        for seed in CANDIDATES:
            if len(paired) >= args.target_new:
                break
            probe = env_ok(seed, 0)
            print(json.dumps({"event": "ENV", **probe}), flush=True)
            dump(HERE / "screen" / f"env_{seed}.json", probe)
            if not probe["ok"]:
                screens.append({"seed": seed, "paired": False, "reason": "env init failed", "env": probe})
                continue
            rec = screen_seed(seed)
            rec["env"] = probe
            dump(HERE / "screen" / f"pre_{seed}.json", rec)
            screens.append(rec)
            print(json.dumps({"event": "SCREEN", "seed": seed, "paired": rec["paired"], "episode_id": rec.get("episode_id")}), flush=True)
            if rec["paired"]:
                paired.append(rec)
        dump(screen_path, {"screens": screens, "paired": [{"seed": p["seed"], "episode_id": p["episode_id"], "deskbin_id": p["deskbin_id"]} for p in paired]})
    else:
        payload = json.loads(screen_path.read_text())
        paired = [s for s in payload["screens"] if s.get("paired")]

    if args.phase == "screen":
        return 0

    seeds = []
    reused = ingest_200014()
    seeds.append(reused)
    dump(HERE / "seeds" / "seed200014.json", reused)

    for rec in paired:
        by = {}
        all_valid = True
        for label in LEVEL_ORDER:
            ctx = run_level(rec["seed"], rec["episode_id"], label)
            by[label] = ctx
            if ctx["pre_gate"]["gate"] != "VALID":
                all_valid = False
        if not all_valid:
            print(json.dumps({"event": "DROP_UNPAIRED", "seed": rec["seed"], "pre": {k: by[k]["pre_gate"]["gate"] for k in LEVEL_ORDER}}), flush=True)
            dump(HERE / "seeds" / f"seed{rec['seed']}_UNPAIRED.json", {k: by[k]["pre_gate"] for k in LEVEL_ORDER})
            continue
        packed = pack_seed(rec["seed"], rec["episode_id"], by)
        dump(HERE / "seeds" / f"seed{rec['seed']}.json", packed)
        seeds.append(packed)

    anomalies = []
    for packed in seeds:
        by = {}
        for label in LEVEL_ORDER:
            by[label] = json.loads((HERE / "sweeps" / f"seed{packed['seed']}_{label}_CONTEXT.json").read_text())
        anom = classify_anomaly(packed, by)
        if anom:
            anomalies.append(anom)
    dump(HERE / "ANOMALIES.json", anomalies)
    agg = aggregate(seeds)
    dump(HERE / "AGGREGATE.json", agg)
    dump(HERE / "SEEDS.json", seeds)
    plot_all(seeds, agg, HERE / "aggregate_retention.png")
    dump(HERE / "VERDICT.json", {"verdict": agg["verdict"], "why": agg["why"], "n_seeds": agg["n_seeds"]})
    print(json.dumps({"done": True, "verdict": agg["verdict"], "why": agg["why"], "n_seeds": agg["n_seeds"], "orderings": [s["ordering"]["pattern"] for s in seeds]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
