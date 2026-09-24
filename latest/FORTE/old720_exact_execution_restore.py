#!/usr/bin/env python3
"""Run the old720 branch contract from the hash-locked historical sources.

This file is deliberately only an execution wrapper.  It does not implement or
modify a controller.  The P5 runner, P4-B probe, task configuration, and
ForcePositionAction are loaded from the detached 80ab3be worktree.  The only
behavioral wrapper is the original strict-preprobe snapshot/restore contract
used by gnp_style_continuous_collect.py during old720 collection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


EXACT_COMMIT = "80ab3be09ce884f86cfc2037d3af30bc28061426"
EXACT_REPO = Path("/home/exouser/Tabero_old720_80ab")
EXACT_RUNNER = EXACT_REPO / "analysis/p5s0c_paired_boundary_probe_value.py"
EXACT_P4 = EXACT_REPO / "analysis/results/p4_contact_conditioned_probe_20260822_184213"
EXACT_P4_COLLECT = EXACT_P4 / "scripts/p4_collect_probe.py"
EXPECTED_RUNNER_SHA256 = "0c029e02544e7d22fbba6f253e29584db6cf796dfabf471e66f157ac66b283b2"
EXPECTED_P4_SHA256 = "a1566334f9f79ad9d8491612f10d086386049295314f6d1fd62512be1867bca9"
EXPECTED_CONTROLLER_SHA256 = "22382587995d505821ae5189afc6b7f11a2f27b7ed22d62015ee6ba7c845ddee"
CONTROLLER = EXACT_REPO / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"
PREPROBE_STEP = 45 + 35 + 70 + 40


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_sources() -> None:
    expected = {
        EXACT_RUNNER: EXPECTED_RUNNER_SHA256,
        EXACT_P4_COLLECT: EXPECTED_P4_SHA256,
        CONTROLLER: EXPECTED_CONTROLLER_SHA256,
    }
    errors = [f"{path}: {sha256(path)} != {digest}" for path, digest in expected.items() if sha256(path) != digest]
    if errors:
        raise RuntimeError("OLD720_SOURCE_HASH_MISMATCH\n" + "\n".join(errors))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def append_capture_row(out: Path, row: dict) -> None:
    path = out / f"task{row['task']}" / "strict_preprobe_capture.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def worker() -> int:
    verify_sources()
    import torch

    p5 = load_module("old720_exact_p5", EXACT_RUNNER)
    # P5/P4 stored absolute paths to the then-current checkout.  Redirect those
    # source locations to the detached copy of that exact commit.  Dataset USD
    # and HDF5 assets remain the same external immutable stores used originally.
    p5.REPO = EXACT_REPO
    p5.RESULTS_ROOT = EXACT_REPO / "analysis/results"
    p5.P4 = EXACT_P4
    p5.P4_COLLECT = EXACT_P4_COLLECT
    p5.OPENPI_CLIENT_SRC = EXACT_REPO / "benchmarks/openpi/openpi-client/src"
    original_import = p5.import_p4_probe

    def strict_import(task_id: int):
        p4 = original_import(task_id)
        p4.TABERO = EXACT_REPO
        os.chdir(EXACT_REPO)
        original_run = p4.run_probe_episode

        def strict_run(env, *, seed_idx: int, mu: float, trial_id: str, dt: float):
            import inspect

            term = env.action_manager.get_term("arm_action")
            cfg = term.cfg
            runtime_path = Path(os.environ["P5S0C_OUT"]) / f"task{task_id}" / "runtime_source_and_config.json"
            runtime_path.parent.mkdir(parents=True, exist_ok=True)
            runtime_path.write_text(json.dumps({
                "tac_manip_module": inspect.getfile(sys.modules["tac_manip"]),
                "force_position_action_module": inspect.getfile(type(term)),
                "p4_module": str(Path(inspect.getfile(p4)).resolve()),
                "p5_module": str(EXACT_RUNNER),
                "physics_dt_s": float(env.cfg.sim.dt),
                "decimation": int(env.cfg.decimation),
                "environment_dt_s": float(dt),
                "squeeze_kp": float(cfg.squeeze_kp),
                "squeeze_deadzone": float(cfg.squeeze_deadzone),
                "meas_force_filter_alpha": float(cfg.meas_force_filter_alpha),
                "squeeze_ff_k_load_z": float(cfg.squeeze_ff_k_load_z),
                "squeeze_ff_contact_threshold": float(cfg.squeeze_ff_contact_threshold),
                "target_contact_squeeze_enabled": bool(cfg.target_contact_squeeze_enabled),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            base_step = env.step
            counter = {"n": 0, "snapshot": None}

            def capture_step(action):
                result = base_step(action)
                counter["n"] += 1
                if counter["n"] == PREPROBE_STEP:
                    counter["snapshot"] = env.scene.get_state(is_relative=True)
                return result

            env.step = capture_step
            try:
                rows, record = original_run(env, seed_idx=seed_idx, mu=mu, trial_id=trial_id, dt=dt)
            finally:
                env.step = base_step
            if counter["snapshot"] is None:
                raise RuntimeError(f"STRICT_PREPROBE_CAPTURE_MISSING:{trial_id}:steps={counter['n']}")
            postprobe_hash = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))
            env.reset_to(counter["snapshot"], torch.tensor([0], device=env.device), is_relative=True)
            preprobe_hash = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))
            env.reset_to(counter["snapshot"], torch.tensor([0], device=env.device), is_relative=True)
            preprobe_hash_2 = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))
            if preprobe_hash != preprobe_hash_2:
                raise RuntimeError(f"STRICT_PREPROBE_RESTORE_UNSTABLE:{trial_id}")
            append_capture_row(Path(os.environ["P5S0C_OUT"]), {
                "task": int(task_id),
                "context_id": trial_id,
                "seed": int(seed_idx),
                "friction": float(mu),
                "capture_step": PREPROBE_STEP,
                "last_hold_step": int(max(r.step for r in rows if str(r.probe_phase) == "hold")),
                "probe_out_present": int(any(str(r.probe_phase) == "probe_out" for r in rows)),
                "preprobe_state_hash": preprobe_hash,
                "second_restore_hash": preprobe_hash_2,
                "postprobe_state_hash_forbidden": postprobe_hash,
                "preprobe_restore_stable": 1,
                "branch_snapshot_semantics": "STRICT_PREPROBE_LAST_HOLD",
            })
            return rows, record

        p4.run_probe_episode = strict_run
        return p4

    p5.import_p4_probe = strict_import
    return int(p5.worker_main())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if not args.worker:
        raise SystemExit("This wrapper must be launched with --worker inside Isaac Python")
    return worker()


if __name__ == "__main__":
    raise SystemExit(main())
