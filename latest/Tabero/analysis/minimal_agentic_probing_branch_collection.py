#!/usr/bin/env python3
"""Collect minimal successor force branches after a validated fixed probe.

The query sequence is the successor Gate 1/2 sequence.  Each force branch is
restored from that same post-probe scene state, so the branch labels are
matched within root and physical context.  This script intentionally collects
one fixed friction stratum and one policy repeat per root.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path("/home/exouser/Tabero")
SUCCESSOR = REPO / "analysis/minimal_agentic_probing_successor.py"
OUT_DEFAULT = REPO / "analysis/results/minimal_agentic_probing_successor_20260828_161533"
FORCES = [5.0, 6.0, 7.0, 8.0]
REPEAT = 0


def load_successor():
    spec = importlib.util.spec_from_file_location("successor_gate", SUCCESSOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load successor gate")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    successor = load_successor()
    frozen = successor.load_module(successor.FROZEN_COLLECTOR, "p7b_frozen_successor_branches")
    frozen.prepare_isaac_runtime_env()
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    server = None
    contexts = []
    branches = []
    parity = []
    try:
        import gymnasium as gym
        import torch
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        from openpi_client import websocket_client_policy

        r2, r1, p6, p4 = frozen.import_runtime_modules(out, successor.TASK)
        frozen.r1 = r1
        setup_task_objects("libero_object", successor.TASK)
        cfg = parse_env_cfg("Isaac-Libero-Franka-Hybrid-Tactile-v0", device="cuda:0", num_envs=1)
        cfg.episode_length_s = 45.0
        env = gym.make("Isaac-Libero-Franka-Hybrid-Tactile-v0", cfg=cfg).unwrapped
        library = r2.recover_library()
        recipe = next(x for x in frozen.read_csv(frozen.P6G1R2_OUT / "P6G1R2_RECIPE_CANDIDATES.csv")
                       if int(x["task"]) == successor.TASK and x["recipe_id"] == "t1_G2_R23_ORIGINAL_P6G1")
        successor.install_normalized_query(frozen, torch)
        server = frozen.start_server(out, 8770)
        deadline = time.time() + 900
        while not frozen._tcp_open("127.0.0.1", 8770):
            if server is not None and server.poll() is not None:
                raise RuntimeError("policy server exited")
            if time.time() > deadline:
                raise RuntimeError("policy server start timeout")
            time.sleep(2)
        client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8770)
        for index, root in enumerate(successor.ROOTS_INTERLEAVED):
            mu = frozen.friction_value(root, successor.STRATUM)
            cid = f"successor_t1_g{root}_s{successor.STRATUM}_mu{mu:.6f}"
            started = time.monotonic()
            try:
                env.reset(seed=int(root))
                r1.p6g1.settle_root_before_hash(env, p6, p4, frozen.ROOT_SETTLE_STEPS)
                root_state = env.scene.get_state(is_relative=True)
                root_hash = r1.root_hash(p6, env)
                frozen.restore_scene_state_stable(env, root_state, torch.tensor([0], device=env.device))
                sq, sqh, qrec, stage = frozen.stage_and_query(
                    env, r2, r1, p6, p4, library, recipe, root_state, root_hash,
                    mu, cid, out)
                c = {
                    "context_id": cid, "root_group_id": root, "root_seed": root,
                    "split": successor.split_for_root(root), "task": successor.TASK,
                    "friction_stratum": successor.STRATUM, "hidden_friction_analysis_only": mu,
                    "probe_qualified": qrec.get("query_qualified", 0),
                    "query_qualified": qrec.get("query_qualified", 0),
                    "drop": qrec.get("drop", 0), "contact_retained": qrec.get("contact_retained", 0),
                    "post_query_state_hash": sqh, "root_state_hash": root_hash,
                    "query_telemetry_path": str(out / "P7B_QUERY_TELEMETRY" / f"{cid}.csv"),
                    "normalization_telemetry_path": str(out / "P7B_NORMALIZATION_TELEMETRY" / f"{cid}.json"),
                    "state_snapshot_path": str(out / "SUCCESSOR_STATE_SNAPSHOTS" / f"{cid}.pt"),
                }
                contexts.append(c)
                if int(qrec.get("query_qualified", 0)):
                    Path(c["state_snapshot_path"]).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(sq, c["state_snapshot_path"])
                    context_for_branch = {"context_id": cid, "root_group_id": root, "root_seed": root,
                                          "split": c["split"], "post_query_state_hash": sqh,
                                          "primitive": r2.make_recipe_primitive(library, recipe)}
                    for force in FORCES:
                        bid = f"{cid}_F{force:g}_R{REPEAT}"
                        br = frozen.branch_from_query(env, r1, p6, p4, client, sq, sqh, context_for_branch,
                                                      force, REPEAT, recipe, out)
                        br.update({"successor_context_id": cid, "successor_normalization": successor.NORMALIZATION_VERSION,
                                   "label_source": "SUCCESSOR_FIXED_PROBE_SINGLE_POST_QUERY_STATE_BRANCH"})
                        branches.append(br)
                        parity.append({"branch_id": bid, "context_id": cid, "root_state_hash": root_hash,
                                       "post_query_state_hash": sqh, "restore_state_hash": br.get("restore_hash", ""),
                                       "parity_pass": br.get("state_parity", 0), "query_state_parity": br.get("query_state_parity", 0)})
                        write_csv(out / "SUCCESSOR_BRANCH_MANIFEST.csv", branches)
                        write_csv(out / "SUCCESSOR_STATE_PARITY.csv", parity)
                        print(json.dumps({"root": root, "force": force, "full_task_success": br.get("full_task_success_y"),
                                          "chunk_exhausted": br.get("vla_chunk_budget_exhausted"), "elapsed_s": time.monotonic() - started}), flush=True)
                else:
                    c["branch_status"] = "SKIPPED_QUERY_INVALID"
                write_csv(out / "SUCCESSOR_CONTEXT_MANIFEST.csv", contexts)
            except Exception as exc:
                contexts.append({"context_id": cid, "root_group_id": root, "root_seed": root,
                                 "split": successor.split_for_root(root), "query_qualified": 0,
                                 "branch_status": "RUNTIME_ERROR", "error": repr(exc),
                                 "traceback": traceback.format_exc()})
                write_csv(out / "SUCCESSOR_CONTEXT_MANIFEST.csv", contexts)
                print(json.dumps({"root": root, "status": "RUNTIME_ERROR", "error": repr(exc)}), flush=True)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if server is not None and server.poll() is None:
            try:
                server.terminate()
            except Exception:
                pass
        try:
            app.close()
        except Exception:
            pass
    write_json(out / "SUCCESSOR_BRANCH_COLLECTION_SUMMARY.json", {
        "status": "COMPLETE", "contexts": len(contexts), "branches": len(branches),
        "qualified_contexts": sum(int(x.get("query_qualified", 0)) == 1 for x in contexts),
        "forces": FORCES, "repeat": REPEAT, "root_split": "by root; no random frame split",
        "label_source": "same root/context/probe post-state restored before each force branch",
    })


def summarize_existing(out: Path):
    contexts = read_csv(out / "SUCCESSOR_CONTEXT_MANIFEST.csv")
    branches = read_csv(out / "SUCCESSOR_BRANCH_MANIFEST.csv")
    write_json(out / "SUCCESSOR_BRANCH_COLLECTION_SUMMARY.json", {
        "status": "INTERRUPTED_BY_DOWNSTREAM_CHUNK_CENSORING",
        "contexts_written": len(contexts),
        "branches_written": len(branches),
        "full_task_successes": sum(str(x.get("full_task_success_y")) == "1" for x in branches),
        "chunk_budget_exhausted": sum(str(x.get("vla_chunk_budget_exhausted")) == "1" for x in branches),
        "runtime_errors": sum(str(x.get("error", "")) not in {"", "vla_chunk_budget_exhausted"} for x in branches),
        "complete_force_vectors": "none; collection was stopped after the first root's continuation hit the frozen VLA chunk-budget path",
        "scientific_use": "not sufficient for GRU held-out force-outcome evaluation",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize_existing(args.out.resolve())
    else:
        run(args.out.resolve())


if __name__ == "__main__":
    main()
