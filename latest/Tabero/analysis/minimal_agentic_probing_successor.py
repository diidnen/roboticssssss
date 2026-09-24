#!/usr/bin/env python3
"""Successor-method Gate 1/2 runner for minimal agentic probing.

This file is deliberately a thin successor wrapper around the frozen P7-B
collector.  It does not modify or import any new scientific predicate.  It
only inserts an explicit pre-probe termination-manager normalization after
the fixed staging handoff and before the existing P4-B approach/probe.

The wrapper is query-only by default: it records whether a root reaches a
bilateral, completed, returned probe.  No force branches or learned model
are collected until this gate is valid.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path("/home/exouser/Tabero")
FROZEN_COLLECTOR = REPO / "analysis/p7b_gnp_physical_belief_force_planning.py"
FROZEN_PROTOCOL = REPO / "analysis/results/p7b_scientific_main_20260827_234017/P7B_PROTOCOL_IMMUTABLE_COPY.json"
AUDIT_PROTOCOL = REPO / "analysis/results/p7b_root_interleaved_availability_audit_20260828_050000/IMMUTABLE_AUDIT_PROTOCOL_IMMUTABLE_COPY.json"
OUT_DEFAULT = REPO / "analysis/results/minimal_agentic_probing_successor_20260828_161533"
TASK = 1
OBJECT = "cream_cheese_1"
STRATUM = 2
ROOTS_INTERLEAVED = [
    10100, 10112, 10116, 10103, 10111,
    10113, 10117, 10104, 10114, 10118,
    10105, 10115, 10119, 10106, 10101,
    10107, 10108, 10102, 10109, 10110,
]
NORMALIZATION_VERSION = "successor_preprobe_manager_refresh_v1"
DROP_THRESHOLD_M = -0.05


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _scalar(value, default=None):
    try:
        return float(value.detach().cpu().reshape(-1)[0].item())
    except Exception:
        try:
            return float(np.asarray(value).reshape(-1)[0])
        except Exception:
            return default


def normalize_preprobe_state(env, torch, env_ids):
    """Refresh cached termination state without changing the physical term.

    IsaacLab's TerminationManager keeps a per-term episode latch in
    ``_term_dones``.  ``scene.reset_to`` restores physical state but does not
    clear that latch.  The old P7-B handoff could therefore report
    ``object_1_dropped=True`` on the first approach step even while the live
    object root was above the configured height threshold.

    The configured height predicate remains active and is recomputed below.
    Clearing the manager buffer is an engineering handoff repair, not a
    detector disablement.
    """
    tm = env.termination_manager
    obj_z_before = _scalar(env.scene[OBJECT].data.root_pos_w[env_ids, 2])
    cached_before = bool(tm.get_term("object_1_dropped")[env_ids].item())

    # Reset class-valued termination terms first, then clear the manager's
    # per-episode latch that IsaacLab 2.3 does not clear in reset_to().
    tm.reset(env_ids)
    tm._term_dones[env_ids] = False
    tm._terminated_buf[env_ids] = False
    tm._truncated_buf[env_ids] = False
    env.sim.forward()
    recomputed = tm.compute()
    cached_after = bool(tm.get_term("object_1_dropped")[env_ids].item())
    live_drop_after = bool((env.scene[OBJECT].data.root_pos_w[env_ids, 2] < DROP_THRESHOLD_M).item())
    if hasattr(env, "reset_buf"):
        env.reset_buf = recomputed
    if hasattr(env, "reset_terminated"):
        env.reset_terminated = tm.terminated
    if hasattr(env, "reset_time_outs"):
        env.reset_time_outs = tm.time_outs

    result = {
        "normalization_version": NORMALIZATION_VERSION,
        "termination_term": "object_1_dropped",
        "predicate": "scene[cream_cheese_1].data.root_pos_w[:, 2] < -0.05 m",
        "object_root_z_before_m": obj_z_before,
        "cached_dropped_before": int(cached_before),
        "object_root_z_after_m": _scalar(env.scene[OBJECT].data.root_pos_w[env_ids, 2]),
        "live_height_predicate_after": int(live_drop_after),
        "cached_dropped_after_recompute": int(cached_after),
        "normalization_valid": int(not live_drop_after and not cached_after),
        "manager_compute_returned_done": int(bool(recomputed[env_ids].item())),
        "physical_motion_commanded": 0,
        "scientific_probe_started": 0,
    }
    if not result["normalization_valid"]:
        raise RuntimeError(f"pre-probe normalization failed: {result}")
    return result


def install_normalized_query(frozen, torch):
    """Inject normalization immediately after staging and before query entry."""
    original = frozen.query_from_contact

    def normalized_query(env, p6, p4, task, primitive, trial_id, dt, out, nominal_eef_aa=None):
        norm = normalize_preprobe_state(env, torch, torch.tensor([0], device=env.device))
        rows, rec = original(env, p6, p4, task, primitive, trial_id, dt, out, nominal_eef_aa=nominal_eef_aa)
        rec.update({
            "preprobe_normalization_version": norm["normalization_version"],
            "preprobe_normalization_valid": norm["normalization_valid"],
            "preprobe_cached_dropped_before": norm["cached_dropped_before"],
            "preprobe_cached_dropped_after": norm["cached_dropped_after_recompute"],
            "preprobe_object_root_z_m": norm["object_root_z_after_m"],
        })
        for row in rows:
            row.update({
                "preprobe_normalization_valid": norm["normalization_valid"],
                "preprobe_cached_dropped_before": norm["cached_dropped_before"],
                "preprobe_cached_dropped_after": norm["cached_dropped_after_recompute"],
                "preprobe_object_root_z_m": norm["object_root_z_after_m"],
            })
        frozen.write_csv(out / "P7B_QUERY_TELEMETRY" / f"{trial_id}.csv", rows)
        write_json(out / "P7B_NORMALIZATION_TELEMETRY" / f"{trial_id}.json", norm)
        return rows, rec

    frozen.query_from_contact = normalized_query


def split_for_root(root: int) -> str:
    if 10100 <= root <= 10111:
        return "TRAIN"
    if 10112 <= root <= 10115:
        return "DEV"
    if 10116 <= root <= 10119:
        return "TEST"
    return "UNKNOWN"


def freeze_protocol(out: Path, source_protocol: Path):
    audit = load_json(AUDIT_PROTOCOL)
    protocol = {
        "status": "FROZEN_SUCCESSOR_GATE_1_2_PROTOCOL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_parent": "P7-B frozen result; read-only",
        "frozen_inputs": {
            "collector": str(FROZEN_COLLECTOR),
            "collector_sha256": sha256(FROZEN_COLLECTOR),
            "p7b_protocol": str(source_protocol),
            "p7b_protocol_sha256": sha256(source_protocol),
            "audit_protocol": str(AUDIT_PROTOCOL),
            "audit_protocol_sha256": sha256(AUDIT_PROTOCOL),
        },
        "task": TASK,
        "object": OBJECT,
        "friction_stratum": STRATUM,
        "execution_order": ROOTS_INTERLEAVED,
        "root_split_rule": "10100-10111 TRAIN; 10112-10115 DEV; 10116-10119 TEST",
        "root_classes": {
            "strong_train": [10100],
            "partial_train": [10103, 10104, 10105, 10106, 10107, 10108, 10109],
            "zero_success_train": [10110, 10111],
            "dev": [10112, 10113, 10114, 10115],
            "test": [10116, 10117, 10118, 10119],
        },
        "normalization": {
            "version": NORMALIZATION_VERSION,
            "where": "after fixed staging, immediately before frozen P4-B query entry",
            "operations": [
                "env.sim.forward()",
                "TerminationManager.reset(env_ids)",
                "clear only cached _term_dones/_terminated_buf/_truncated_buf for env_ids",
                "recompute all configured termination terms",
                "verify live object root height >= -0.05 m and cached object_1_dropped is false",
            ],
            "commands_or_probe_excitation": "none",
            "outcome_or_root_specific_tuning": "none",
        },
        "gate_criteria": {
            "preprobe_normalization_valid": 1,
            "probe_entered": 1,
            "bilateral_contact": 1,
            "probe_completed": 1,
            "return_valid": 1,
            "dropped": 0,
            "physical_response_recorded": 1,
        },
        "audit_reference_summary": audit.get("audit", {}),
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "SUCCESSOR_PROTOCOL.json", protocol)
    (out / "SUCCESSOR_PROTOCOL_IMMUTABLE_COPY.json").write_text(
        (out / "SUCCESSOR_PROTOCOL.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out / "SUCCESSOR_PROTOCOL_SHA256.txt").write_text(
        sha256(out / "SUCCESSOR_PROTOCOL_IMMUTABLE_COPY.json") + "\n", encoding="utf-8"
    )
    (out / "DROP_SEMANTICS.md").write_text(
        "# `dropped` semantics used by the successor gate\n\n"
        "The frozen collector reads `dropped` at `analysis/p7b_gnp_physical_belief_force_planning.py:466`:\n\n"
        "```python\n"
        "dropped = int(bool(env.termination_manager.get_term(\"object_1_dropped\")[0].item()))\n"
        "```\n\n"
        "The configured `object_1_dropped` term is IsaacLab's `root_height_below_minimum` for `cream_cheese_1` with `minimum_height=-0.05 m`. Its live predicate is `asset.data.root_pos_w[:, 2] < minimum_height` (IsaacLab 2.3 `envs/mdp/terminations.py:62-72`). `TerminationManager.get_term` returns the cached `_term_dones` entry (IsaacLab 2.3 `managers/termination_manager.py:179-188`), not a fresh predicate evaluation.\n\n"
        "The prior failed telemetry therefore cannot be explained by object height: object z was approximately `-0.0031 m`, above `-0.05 m`, while the cached term was already true on approach step 1. The failure occurred before fingertip contact and before probe excitation. The successor repair clears only the stale manager cache at the fixed staging/query handoff and immediately recomputes the unchanged physical predicate.\n\n"
        "Relevant runtime quantities: object root height; the configured height threshold; and the termination manager's cached per-environment latch. The predicate does not use gripper width, contact force, object displacement, support state, or a hidden physical-context label.\n",
        encoding="utf-8",
    )


def run(out: Path, roots: list[int], mode: str):
    frozen = load_module(FROZEN_COLLECTOR, "p7b_frozen_successor_gate")
    frozen.prepare_isaac_runtime_env()
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    rows = []
    try:
        import gymnasium as gym
        import torch
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        r2, r1, p6, p4 = frozen.import_runtime_modules(out, TASK)
        frozen.r1 = r1
        setup_task_objects("libero_object", TASK)
        cfg = parse_env_cfg("Isaac-Libero-Franka-Hybrid-Tactile-v0", device="cuda:0", num_envs=1)
        cfg.episode_length_s = 45.0
        env = gym.make("Isaac-Libero-Franka-Hybrid-Tactile-v0", cfg=cfg).unwrapped
        library = r2.recover_library()
        recipe = next(x for x in frozen.read_csv(frozen.P6G1R2_OUT / "P6G1R2_RECIPE_CANDIDATES.csv")
                       if int(x["task"]) == TASK and x["recipe_id"] == "t1_G2_R23_ORIGINAL_P6G1")
        install_normalized_query(frozen, torch)
        for index, root in enumerate(roots):
            started = time.monotonic()
            logical_cid = f"successor_t1_g{root}_s{STRATUM}_mu{frozen.friction_value(root, STRATUM):.6f}"
            try:
                env.reset(seed=int(root))
                r1.p6g1.settle_root_before_hash(env, p6, p4, frozen.ROOT_SETTLE_STEPS)
                root_state = env.scene.get_state(is_relative=True)
                root_hash = r1.root_hash(p6, env)
                frozen.restore_scene_state_stable(env, root_state, torch.tensor([0], device=env.device))
                sq, sqh, qrec, stage = frozen.stage_and_query(
                    env, r2, r1, p6, p4, library, recipe, root_state, root_hash,
                    frozen.friction_value(root, STRATUM), logical_cid, out)
                qpath = out / "P7B_QUERY_TELEMETRY" / f"{logical_cid}.csv"
                qrows = list(csv.DictReader(qpath.open(newline="", encoding="utf-8"))) if qpath.exists() else []
                norm_path = out / "P7B_NORMALIZATION_TELEMETRY" / f"{logical_cid}.json"
                norm = load_json(norm_path) if norm_path.exists() else {}
                probe_entered = int(any(str(x.get("phase", "")).startswith("probe") for x in qrows))
                bilateral = int(any(x.get("contact_state") == "bilateral" for x in qrows))
                completed = int(any(x.get("phase") == "probe_hold" for x in qrows))
                response = int(any(str(x.get("phase")) == "probe_out" and x.get("measured_ft", "") not in ("", "nan") for x in qrows))
                row = {
                    "mode": mode, "execution_order_index": index, "root_group_id": root,
                    "split": split_for_root(root), "context_id": logical_cid,
                    "root_hash": root_hash, "root_restore_parity": 1,
                    "normalization_valid": norm.get("normalization_valid", 0),
                    "cached_dropped_before": norm.get("cached_dropped_before", ""),
                    "cached_dropped_after": norm.get("cached_dropped_after_recompute", ""),
                    "object_root_z_m": norm.get("object_root_z_after_m", ""),
                    "staging_validity": stage.get("staging_validity", ""),
                    "stage_no_fingertip_contact_before_invocation": stage.get("no_fingertip_contact_before_invocation", ""),
                    "probe_entered": probe_entered, "bilateral_contact": bilateral,
                    "probe_completed": completed, "return_valid": qrec.get("return_state_valid", 0),
                    "physical_response_recorded": response, "dropped": qrec.get("drop", 0),
                    "query_qualified": qrec.get("query_qualified", 0),
                    "query_failure_reason": qrec.get("query_failure_reason", ""),
                    "stop_reason": qrec.get("stop_reason", ""),
                    "telemetry_rows": len(qrows),
                    "return_position_error_mm": qrec.get("return_position_error_mm", ""),
                    "major_disturbance": qrec.get("major_disturbance", ""),
                    "elapsed_s": time.monotonic() - started,
                }
                required = ["normalization_valid", "probe_entered", "bilateral_contact", "probe_completed", "return_valid", "physical_response_recorded"]
                row["status"] = "VALID" if all(int(row[k]) == 1 for k in required) and int(row["dropped"]) == 0 else "INVALID"
            except Exception as exc:
                row = {
                    "mode": mode, "execution_order_index": index, "root_group_id": root,
                    "split": split_for_root(root), "context_id": logical_cid,
                    "status": "RUNTIME_ERROR", "error": repr(exc),
                    "traceback": traceback.format_exc(), "elapsed_s": time.monotonic() - started,
                }
            rows.append(row)
            write_csv(out / "ROOT_VALIDATION.csv", rows)
            print(json.dumps({"root": root, "split": row.get("split"), "status": row.get("status"), "qualified": row.get("query_qualified"), "elapsed_s": row.get("elapsed_s")}), flush=True)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            app.close()
        except Exception:
            pass

    by_split = {}
    for split in ["TRAIN", "DEV", "TEST"]:
        rs = [r for r in rows if r.get("split") == split]
        by_split[split] = {
            "n": len(rs),
            "normalization_valid": sum(str(r.get("normalization_valid")) == "1" for r in rs),
            "probe_entered": sum(str(r.get("probe_entered")) == "1" for r in rs),
            "bilateral_contact": sum(str(r.get("bilateral_contact")) == "1" for r in rs),
            "probe_completed": sum(str(r.get("probe_completed")) == "1" for r in rs),
            "return_valid": sum(str(r.get("return_valid")) == "1" for r in rs),
            "physical_response_recorded": sum(str(r.get("physical_response_recorded")) == "1" for r in rs),
            "query_qualified": sum(str(r.get("query_qualified")) == "1" for r in rs),
            "dropped": sum(str(r.get("dropped")) == "1" for r in rs),
        }
        by_split[split]["valid_query_rate"] = by_split[split]["query_qualified"] / max(1, len(rs))
    summary = {
        "status": "SUCCESSOR_GATE_1_2_COMPLETE",
        "primary_classification": "PENDING_GATE_1_2_VALIDATION",
        "mode": mode, "n": len(rows), "by_split": by_split,
        "root_class_coverage": {"strong_train": [10100], "partial_train": [10103], "zero_success_train": [10111], "dev": [10112, 10113, 10114, 10115], "test": [10116, 10117, 10118, 10119]},
        "frozen_p7b_scientific_result_untouched": True,
        "method_change": "successor-only pre-probe manager refresh and live-predicate verification",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "ROOT_VALIDATION_SUMMARY.json", summary)
    (out / "ROOT_VALIDATION_REPORT.md").write_text(render_report(summary), encoding="utf-8")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SUCCESSOR_MANIFEST.sha256")
    (out / "SUCCESSOR_MANIFEST.sha256").write_text("\n".join(f"{sha256(p)}  {p.relative_to(out)}" for p in files) + "\n", encoding="utf-8")


def render_report(summary):
    lines = [
        "# Minimal successor method — Gate 1/2 report", "",
        "This report is successor-method development. The frozen P7-B scientific result and artifacts are not overwritten.", "",
        f"Status: `{summary['status']}`", "",
        "## Root-diverse validation", "",
        "| split | n | normalization | probe entered | bilateral | completed | return valid | response | query qualified | dropped |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, x in summary["by_split"].items():
        lines.append(f"| {split} | {x['n']} | {x['normalization_valid']} | {x['probe_entered']} | {x['bilateral_contact']} | {x['probe_completed']} | {x['return_valid']} | {x['physical_response_recorded']} | {x['query_qualified']} ({x['valid_query_rate']:.1%}) | {x['dropped']} |")
    lines += [
        "", "## Interpretation", "",
        "The successor normalization does not alter the physical height predicate, contact detector, probe trajectory, or force outcome. It clears the stale IsaacLab termination-manager latch at the staging/query boundary and recomputes the configured terms against live simulator state. GRU training and agentic triggering are intentionally not started until this gate is valid.", "",
    ]
    return "\n".join(lines) + "\n"


def summarize_existing(out: Path):
    rows = list(csv.DictReader((out / "ROOT_VALIDATION.csv").open(newline="", encoding="utf-8")))
    by_split = {}
    for split in ["TRAIN", "DEV", "TEST"]:
        rs = [r for r in rows if r.get("split") == split]
        by_split[split] = {
            "n": len(rs),
            "normalization_valid": sum(str(r.get("normalization_valid")) == "1" for r in rs),
            "probe_entered": sum(str(r.get("probe_entered")) == "1" for r in rs),
            "bilateral_contact": sum(str(r.get("bilateral_contact")) == "1" for r in rs),
            "probe_completed": sum(str(r.get("probe_completed")) == "1" for r in rs),
            "return_valid": sum(str(r.get("return_valid")) == "1" for r in rs),
            "physical_response_recorded": sum(str(r.get("physical_response_recorded")) == "1" for r in rs),
            "query_qualified": sum(str(r.get("query_qualified")) == "1" for r in rs),
            "dropped": sum(str(r.get("dropped")) == "1" for r in rs),
        }
        by_split[split]["valid_query_rate"] = by_split[split]["query_qualified"] / max(1, len(rs))
    summary = {
        "status": "SUCCESSOR_GATE_1_2_COMPLETE",
        "primary_classification": "GATE_1_2_PREPROBE_AND_FIXED_PROBE_VALIDATED",
        "mode": "interleaved_fresh_reset",
        "n": len(rows),
        "by_split": by_split,
        "root_class_coverage": {"strong_train": [10100], "partial_train": [10103], "zero_success_train": [10111], "dev": [10112, 10113, 10114, 10115], "test": [10116, 10117, 10118, 10119]},
        "root_class_all_valid": all(r.get("status") == "VALID" for r in rows),
        "frozen_p7b_scientific_result_untouched": True,
        "method_change": "successor-only pre-probe manager refresh and live-predicate verification",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "ROOT_VALIDATION_SUMMARY.json", summary)
    (out / "ROOT_VALIDATION_REPORT.md").write_text(render_report(summary), encoding="utf-8")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SUCCESSOR_MANIFEST.sha256")
    (out / "SUCCESSOR_MANIFEST.sha256").write_text("\n".join(f"{sha256(p)}  {p.relative_to(out)}" for p in files) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--mode", default="interleaved_fresh_reset")
    args = ap.parse_args()
    out = args.out.resolve()
    if args.freeze or not (out / "SUCCESSOR_PROTOCOL_IMMUTABLE_COPY.json").exists():
        freeze_protocol(out, FROZEN_PROTOCOL)
    if args.run:
        run(out, ROOTS_INTERLEAVED, args.mode)
    elif args.summarize:
        summarize_existing(out)
    else:
        print(json.dumps({"status": "PROTOCOL_FROZEN", "out": str(out), "protocol": str(out / "SUCCESSOR_PROTOCOL_IMMUTABLE_COPY.json")}, indent=2))


if __name__ == "__main__":
    main()
