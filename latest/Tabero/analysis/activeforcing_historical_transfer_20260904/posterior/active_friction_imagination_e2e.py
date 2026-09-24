#!/usr/bin/env python3
"""Small Isaac pilot for the active-friction imagination chain.

The pilot reuses the P5-S0-D/P5-S0-C query, snapshot restore, deterministic
downstream controller, friction setter, and success predicate.  Only the
new explicit friction estimator and posterior-aware imagination wrapper are
new.  It is deliberately a separate worker protocol and never modifies old
P5-S0 result directories.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

REPO = Path("/home/exouser/Tabero")
RESULTS_ROOT = REPO / "analysis/results"
ESTIMATOR_MODULE = REPO / "analysis/active_friction_imagination.py"
P5D_MODULE = REPO / "analysis/p5s0d_fresh_e2e_q2f.py"
C_ARTIFACT = RESULTS_ROOT / "p5s0c_paired_boundary_probe_value_20260824_000542"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
WARP_CORE = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64")
OPENPI_CLIENT_SRC = REPO / "benchmarks/openpi/openpi-client/src"
OUT = Path(os.environ.get("AFI_OUT", RESULTS_ROOT / f"active_friction_imagination_e2e_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"))

# Representative fresh pilot: one root with all three hidden friction strata
# for task 0.  This is a gate pilot, not a broad claim over all tasks.
TASK = 0
ROOT_SEED = 8100
FRICTION_SAMPLER_SEED = 20260828_1
FRICTION_BANDS = {"LOW": (0.20, 0.30), "MID": (0.45, 0.60), "HIGH": (0.90, 1.00)}
FORCES_BY_TASK = {0: [3.0, 4.0, 4.5, 5.0], 1: [4.0, 4.5, 5.0, 5.5, 6.0], 5: [3.0, 4.0, 4.5, 5.0], 6: [3.0, 3.5, 4.0]}
PILOT_FORCES = FORCES_BY_TASK[TASK]
ROBUST_FORCE = {0: 5.0, 1: 6.0, 5: 5.0, 6: 4.0}


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


class FrictionGRU(nn.Module):
    """Worker-local copy of the tiny architecture; avoids pandas in Isaac."""

    def __init__(self, input_dim: int, projection_dim: int = 16, hidden_dim: int = 16):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, projection_dim), nn.ReLU())
        self.gru = nn.GRU(projection_dim, hidden_dim, batch_first=True)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.projection(x)
        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h = h[-1]
        return self.mu_head(h).squeeze(1), self.log_sigma_head(h).squeeze(1).clamp(-5.0, 1.5)


def pilot_plans() -> list[dict[str, Any]]:
    rng = np.random.default_rng(FRICTION_SAMPLER_SEED)
    return [
        {
            "task": TASK, "root_seed": ROOT_SEED, "root_id": f"afi_pilot_t{TASK}_root00_s{ROOT_SEED}",
            "band": band, "friction": float(rng.uniform(*bounds)),
            "context_id": f"afi_pilot_t{TASK}_root00_s{ROOT_SEED}_{band.lower()}",
        }
        for band, bounds in FRICTION_BANDS.items()
    ]


def estimator_inference(est, online, telemetry_path: Path) -> tuple[float, float]:
    norm = json.loads((C_ARTIFACT / "P5S0C_NORMALIZATION.json").read_text())
    # P5-S0-D's worker-safe OnlineInference.sequence_array is the exact
    # feature construction used by the frozen deployable representation.
    arr = online.sequence_array(str(telemetry_path)).astype(np.float32)
    arr = (arr - np.asarray(norm["dynamic_mean"], dtype=np.float32)) / np.asarray(norm["dynamic_std"], dtype=np.float32)
    with torch.no_grad():
        x = torch.tensor(arr[None], dtype=torch.float32)
        lengths = torch.tensor([arr.shape[0]], dtype=torch.long)
        mu, log_sigma = est(x, lengths)
    return float(mu.item()), float(torch.exp(log_sigma).item())


def restore(env, state):
    env.reset_to(state, torch.tensor([0], device=env.device), is_relative=True)


def branch(p5c, env, p4, plan, force: float, mu: float, label: str, state, ref_hash: str, dt: float, imagined: bool) -> dict[str, Any]:
    restore(env, state)
    p4._apply_friction(env, p5c.TASK_OBJECTS[int(plan["task"])], float(mu))
    telemetry = OUT / ("IMAGINATION_TELEMETRY" if imagined else "REAL_TELEMETRY") / f"{plan['context_id']}_{label}_F{force:g}_mu{mu:.4f}.csv"
    logger = p5c.StageLogger(OUT, int(plan["task"]), int(plan["root_seed"]), dt=dt, context_id=plan["context_id"], split="PILOT", friction=float(plan["friction"]))
    rec = p5c.downstream_branch(
        env, p4, task_id=int(plan["task"]), force=float(force), branch_label=label,
        context_id=plan["context_id"], split="PILOT", seed=int(plan["root_seed"]),
        friction=float(plan["friction"]), dt=dt, logger=logger, telemetry_path=telemetry,
        label_source="AFI_IMAGINATION" if imagined else "AFI_REAL_EXECUTION",
    )
    rec.update({"plan_context_id": plan["context_id"], "band": plan["band"], "imagined_mu": float(mu), "imagined": int(imagined), "restore_reference_hash": ref_hash})
    return rec


def choose_from_curves(curves: dict[float, list[int]], threshold: float = 0.9) -> float:
    for force in PILOT_FORCES:
        if float(np.mean(curves[force])) >= threshold:
            return float(force)
    return float(max(PILOT_FORCES))


def run_worker() -> int:
    from isaaclab.app import AppLauncher

    # Isaac/Omniverse imports must occur after SimulationApp/AppLauncher has
    # been instantiated.  This mirrors the authoritative P5-S0-D worker
    # contract and avoids a false runner failure before any physics executes.
    app_launcher = AppLauncher(headless=True, enable_cameras=True, num_envs=1)
    app = app_launcher.app
    import gymnasium as gym
    import tac_manip.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from tac_manip.utils.task_configs import setup_task_objects

    p5d = import_module(P5D_MODULE, "afi_p5s0d")
    p5d.OUT = OUT
    p5c = p5d.P5C_DATA
    ckpt_path = Path(os.environ["AFI_ESTIMATOR_CKPT"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    est = FrictionGRU(int(ckpt["input_dim"]), int(ckpt["projection_dim"]), int(ckpt["hidden_dim"]))
    est.load_state_dict(ckpt["state_dict"]); est.eval()
    online = p5d.OnlineInference("cpu")

    env = None
    try:
        p4 = p5d.import_p4_probe(TASK)
        setup_task_objects(p5d.TASK_SUITE, TASK)
        env_cfg = parse_env_cfg(p5d.ENV_ID, device="cuda:0", num_envs=1)
        env_cfg.episode_length_s = 45.0
        env = gym.make(p5d.ENV_ID, cfg=env_cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        all_rows: list[dict[str, Any]] = []
        query_rows: list[dict[str, Any]] = []
        action_rows: list[dict[str, Any]] = []
        for plan in pilot_plans():
            s0, h0, sq, hq, steps, probe_rec = p5d.query_with_captured_s0(env, p4, seed=int(plan["root_seed"]), mu=float(plan["friction"]), trial_id=plan["context_id"], dt=dt)
            probe_path = OUT / "PROBE_TELEMETRY" / f"{plan['context_id']}.csv"
            write_csv(probe_path, [vars(s) for s in steps])
            mu_hat, sigma = estimator_inference(est, online, probe_path)
            low = float(np.clip(mu_hat - 1.645 * sigma, 0.20, 1.00))
            high = float(np.clip(mu_hat + 1.645 * sigma, 0.20, 1.00))
            hypotheses = sorted({float(np.clip(mu_hat, 0.20, 1.00)), low, high})
            query_rows.append({**plan, "probe_path": str(probe_path), "probe_steps": len(steps), "probe_qualified": 1, "mu_hat": mu_hat, "sigma_mu": sigma, "mu_hypotheses": json.dumps(hypotheses), "query_start_hash": h0, "post_query_hash": hq})

            gt_curves: dict[float, list[int]] = {}
            est_curves: dict[float, list[int]] = {f: [] for f in PILOT_FORCES}
            for force in PILOT_FORCES:
                gt_rec = branch(p5c, env, p4, plan, force, float(plan["friction"]), "GT_IMAGINATION", sq, hq, dt, True)
                gt_curves[force] = [int(gt_rec["full_task_success_y"])]
                for hm in hypotheses:
                    er = branch(p5c, env, p4, plan, force, hm, "EST_IMAGINATION", sq, hq, dt, True)
                    est_curves[force].append(int(er["full_task_success_y"]))
            gt_force = choose_from_curves(gt_curves)
            est_force = choose_from_curves(est_curves)
            # Execute both selections from the same restored post-query state;
            # the environment retains the true hidden friction only for these
            # real branches.
            gt_real = branch(p5c, env, p4, plan, gt_force, float(plan["friction"]), "GT_SELECTED_REAL", sq, hq, dt, False)
            est_real = branch(p5c, env, p4, plan, est_force, float(plan["friction"]), "EST_SELECTED_REAL", sq, hq, dt, False)
            fixed_real = branch(p5c, env, p4, plan, ROBUST_FORCE[TASK], float(plan["friction"]), "NO_PHYSICS_FIXED_REAL", sq, hq, dt, False)
            action_rows.append({**plan, "mu_hat": mu_hat, "sigma_mu": sigma, "gt_selected_force_N": gt_force, "estimated_selected_force_N": est_force, "no_physics_force_N": ROBUST_FORCE[TASK], "gt_curve": json.dumps(gt_curves, sort_keys=True), "estimated_curve": json.dumps(est_curves, sort_keys=True), "gt_real_success": gt_real["full_task_success_y"], "estimated_real_success": est_real["full_task_success_y"], "no_physics_real_success": fixed_real["full_task_success_y"], "force_match": int(gt_force == est_force), "estimated_under_force": int(est_real["full_task_success_y"] == 0 and est_force < gt_force), "state_hash": hq})
            all_rows.extend([gt_real, est_real, fixed_real])
            write_csv(OUT / "QUERY_RESULTS.csv", query_rows)
            write_csv(OUT / "ACTION_RESULTS.csv", action_rows)
            write_csv(OUT / "REAL_BRANCH_RESULTS.csv", all_rows)
        write_json(OUT / "PILOT_RESULT.json", {"status": "PASS", "contexts": len(action_rows), "real_success_estimated": float(np.mean([int(r["estimated_real_success"]) for r in action_rows])), "force_match": float(np.mean([int(r["force_match"]) for r in action_rows])), "checkpoint": str(ckpt_path)})
        return 0
    except Exception as exc:
        write_json(OUT / "ERROR.json", {"error": repr(exc), "trace": traceback.format_exc()})
        return 1
    finally:
        if env is not None:
            try: env.close()
            except Exception: pass
        try: app.close()
        except Exception: pass


def launch(out: Path, ckpt: Path) -> int:
    env = os.environ.copy()
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(REPO), str(OPENPI_CLIENT_SRC)]), "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "TABERO_ROOT": str(REPO), "HDF5_TRAJ_SOURCE_DIR": str(REPO / "benchmarks/datasets/libero/assembled_hdf5"), "LIBERO_CONFIG_DIR": str(REPO / "benchmarks/datasets/libero/config"), "LIBERO_ASSETS_DATA_DIR": str(REPO / "benchmarks/datasets/libero/USD"), "AFI_OUT": str(out), "AFI_ESTIMATOR_CKPT": str(ckpt), "AFI_WORKER": "1", "P5S0D_WORKER": "1"})
    log = out / "worker.log"
    with log.open("w", encoding="utf-8") as fh:
        proc = __import__("subprocess").Popen([str(ISAAC_PY), "-u", str(Path(__file__).resolve()), "--worker"], cwd=REPO, env=env, stdout=fh, stderr=__import__("subprocess").STDOUT)
        returncode = proc.wait(timeout=21600)
    return int(returncode)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    global OUT
    OUT = args.out or OUT
    if args.worker:
        # The parent launcher creates and records the result namespace before
        # spawning Isaac.  The worker must reuse that directory, not attempt
        # to recreate it.
        OUT = Path(os.environ.get("AFI_OUT", str(OUT)))
        return run_worker()
    OUT.mkdir(parents=True, exist_ok=False)
    ckpt = args.checkpoint or max(RESULTS_ROOT.glob("active_friction_imagination_*/FRICTION_GRU.pt"), key=lambda p: p.stat().st_mtime)
    write_json(OUT / "PROTOCOL.json", {"name": "Active friction imagination v1 fresh pilot", "task": TASK, "root_seed": ROOT_SEED, "friction_only": True, "force_grid_N": PILOT_FORCES, "hypotheses": "mu_hat and clipped mu_hat +/- 1.645 sigma", "selection": "minimum force with simulated success mean >= 0.9", "real_branches": ["GT_SELECTED_REAL", "EST_SELECTED_REAL", "NO_PHYSICS_FIXED_REAL"], "estimator_checkpoint": str(ckpt), "no_pi0_in_imagination": True, "semantic_context": "frozen Pi0Config task instruction; deterministic downstream skeleton after standardized post-query state"})
    return launch(OUT, ckpt)


if __name__ == "__main__":
    raise SystemExit(main())
