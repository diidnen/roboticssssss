#!/usr/bin/env python3
"""Pre-TEST context audit and two-method task0 formal runner.

This file is intentionally a thin runtime adapter.  It reuses the frozen
P4-B probe, the existing IsaacLab scene restore, the frozen P6G1R1/π0 loop,
and the frozen GNP FEASIBILITY_ONLY checkpoints.  It does not train, tune,
or read TEST outcomes while building a Direct decision.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

FORTE = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
OUT = FORTE / "hidden_friction_baseline_20260831"
FORMAL_OUT = OUT / "TWO_METHOD_FORMAL_RUNTIME_RERUN1"
P6_PATH = TABERO / "analysis/p6g1r1_controller_grasp_vla_handoff.py"
GNP_PATH = FORTE / "gnp_style_continuous.py"
P5D_PATH = TABERO / "analysis/p5s0d_fresh_e2e_q2f.py"
AFI_PATH = TABERO / "analysis/active_friction_imagination.py"
FRICTION_CKPT = TABERO / "analysis/results/active_friction_imagination_20260828_211106/FRICTION_GRU.pt"
GNP_OUT = FORTE / "gnp_style_continuous_20260830_125107"
TASK = 0
OBJECT = "alphabet_soup_1"
PROMPT = "pick up the alphabet soup and place it in the basket"
DEV_ROOT = 5106
DEV_MU = 0.2
GRID = [3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]
TEST_ROOTS = [5174, 5175, 5176, 5177, 5178, 5179]
MUS = [0.2, 0.5, 1.0]
REPEATS = 10
PI0_HOST = "127.0.0.1"
PI0_PORT = 18881
SETTLE_STEPS = 20


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(x: Any) -> Any:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def stable_hash(x: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(x), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def max_abs(a: Any, b: Any) -> float:
    if isinstance(a, dict) and isinstance(b, dict):
        return max((max_abs(a.get(k), b.get(k)) for k in set(a) | set(b)), default=0.0)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max((max_abs(x, y) for x, y in zip(a, b)), default=0.0)
    try:
        x, y = float(a), float(b)
        if not (np.isfinite(x) and np.isfinite(y)):
            return 0.0 if (not np.isfinite(x) and not np.isfinite(y)) else float("inf")
        return abs(x - y)
    except Exception:
        return 0.0 if a == b else float("inf")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader()
        w.writerows(rows)


def probe_dicts(rows: list[Any]) -> list[dict[str, Any]]:
    return [jsonable(vars(x) if hasattr(x, "__dict__") else x) for x in rows]


def strict_state_from_probe(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, int, float]:
    hold = [r for r in rows if str(r.get("probe_phase", "")) == "hold"]
    if not hold or not any(str(r.get("probe_phase", "")) == "probe_out" for r in rows):
        raise RuntimeError("P4-B probe did not contain frozen pre-shear hold and probe_out")
    r = hold[-1]
    opening = float(r.get("gripper_opening", r.get("gripper_opening_m", 0.0)))
    state = np.zeros(13, np.float32)
    mask = np.zeros(13, np.float32)
    state[11:13] = [opening, -opening]
    mask[:6] = 1.0
    mask[11:13] = 1.0
    return state, mask, int(float(r.get("step", 0))), opening


def write_query_skeleton(path: Path, eef_pose: np.ndarray, task: int) -> None:
    """Write the frozen first-H branch_hold query skeleton.

    Direct's frozen decoder consumes the first H=8 rows of the nominal
    full-task command trace.  P5-S0-C's branch_hold is constant, so the
    online skeleton is exactly eight copies of the post-probe EEF command;
    no task outcome or hidden friction is needed.
    """
    rows = [{"step": i + 1, "phase": "branch_hold", "cmd_x": float(eef_pose[0]),
             "cmd_y": float(eef_pose[1]), "cmd_z": float(eef_pose[2])} for i in range(8)]
    write_csv(path, rows)


class FeasibilityOnly(nn.Module):
    """Exact architecture of full_task_feasibility_decoder.FeasibilityOnly."""
    def __init__(self):
        super().__init__()
        self.command_gru = nn.GRU(17, 64, batch_first=True)
        self.condition = nn.Sequential(nn.Linear(54, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, step, cond):
        _, h = self.command_gru(step)
        z = torch.cat([h[-1], self.condition(cond)], dim=-1)
        return self.head(z).squeeze(-1)


class FrictionGRU(nn.Module):
    """Exact frozen active_friction_imagination estimator architecture."""
    def __init__(self, input_dim: int, projection_dim: int = 16, hidden_dim: int = 16):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, projection_dim), nn.ReLU())
        self.gru = nn.GRU(projection_dim, hidden_dim, batch_first=True)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        z = self.projection(x)
        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h = h[-1]
        return self.mu_head(h).squeeze(1), self.log_sigma_head(h).squeeze(1).clamp(-5.0, 1.5)


PROBE_NUMERIC_COLS = [
    "t_s", "force_target", "measured_squeeze", "target_normal_force", "measured_fn", "measured_ft",
    "ft_over_fn", "left_fx", "left_fy", "left_fz", "right_fx", "right_fy", "right_fz",
    "force_imbalance", "force_imbalance_ratio", "gripper_opening", "contact_normal_x", "contact_normal_y",
    "contact_normal_z", "contact_tangent_x", "contact_tangent_y", "contact_tangent_z",
    "commanded_tangent_increment_mm", "accumulated_displacement_mm", "marker_motion", "marker_tangential",
    "marker_velocity", "marker_loading_unloading", "contact_left", "contact_right", "tactile_ok",
]


def probe_feature_contract():
    q = json.loads((TABERO / "analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542/P5S0C_NORMALIZATION.json").read_text())
    return list(q["dynamic_feature_names"]), list(q["phase_categories_from_train"]), list(q["contact_state_categories_from_train"])


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key, "")
        return default if v in (None, "") else float(v)
    except Exception:
        return default


def online_sequence_array(path: Path, feature_names: list[str], phases: list[str], states: list[str]) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        return np.zeros((1, len(feature_names)), np.float32)
    ex0 = _float(raw[0], "eef_x"); ey0 = _float(raw[0], "eef_y"); ez0 = _float(raw[0], "eef_z")
    rows = []
    for rr in raw:
        feat = {col: _float(rr, col) for col in PROBE_NUMERIC_COLS}
        feat["eef_dx"] = _float(rr, "eef_x") - ex0
        feat["eef_dy"] = _float(rr, "eef_y") - ey0
        feat["eef_dz"] = _float(rr, "eef_z") - ez0
        phase = str(rr.get("probe_phase", "NA")); state = str(rr.get("contact_state", "NA"))
        for p in phases: feat[f"phase={p}"] = float(phase == p)
        for s in states: feat[f"contact_state={s}"] = float(state == s)
        feat["probe_phase_unknown"] = float(phase not in phases)
        feat["contact_state_unknown"] = float(state not in states)
        rows.append([float(feat.get(name, 0.0)) for name in feature_names])
    return np.asarray(rows, np.float32)


def nominal_segment(cmd: np.ndarray, task: int, force: float, mu: float, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cmd = np.asarray(cmd, np.float64)
    cr = cmd - cmd[0]
    cd = np.vstack([np.zeros((1, 3)), np.diff(cmd, axis=0)])
    phases = np.zeros((len(cmd), 7), np.float64); phases[:, 0] = 1.0  # branch_hold
    tasks = np.zeros((len(cmd), 4), np.float64); tasks[:, [0, 1, 2, 3][0 if task == 0 else 1 if task == 1 else 2 if task == 5 else 3]] = 1.0
    static = np.repeat([[force / 8.0, mu]], len(cmd), axis=0)
    state_all = np.repeat(np.asarray(state, np.float64)[None], len(cmd), axis=0)
    mask_all = np.repeat(np.asarray(mask, np.float64)[None], len(cmd), axis=0)
    init = np.repeat(np.asarray(state, np.float64)[None], len(cmd), axis=0)
    init_mask = np.repeat(np.asarray(mask, np.float64)[None], len(cmd), axis=0)
    return np.concatenate([cr, cd, phases, tasks, static, state_all, mask_all, init, init_mask], axis=1).astype(np.float32)


class DirectRuntime:
    def __init__(self):
        norm_q = json.loads((GNP_OUT / "GNP_STYLE_TRAIN_NORMALIZATION.json").read_text())
        self.gnp_norm = tuple(np.asarray(norm_q[k], np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"])
        self.models = []
        for seed in [0, 1, 2]:
            ck = torch.load(GNP_OUT / f"GNP_STYLE_CONTINUOUS_FEAS_seed{seed}.pt", map_location="cpu", weights_only=False)
            model = FeasibilityOnly()
            model.load_state_dict(ck["state_dict"])
            model.eval()
            self.models.append(model)
        ck = torch.load(FRICTION_CKPT, map_location="cpu", weights_only=False)
        self.estimator = FrictionGRU(int(ck["input_dim"]), int(ck["projection_dim"]), int(ck["hidden_dim"]))
        self.estimator.load_state_dict(ck["state_dict"])
        self.estimator.eval()
        self.probe_feature_names, self.probe_phases, self.probe_states = probe_feature_contract()

    def estimate_mu(self, telemetry_path: Path) -> tuple[float, float]:
        norm_path = TABERO / "analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542/P5S0C_NORMALIZATION.json"
        q = json.loads(norm_path.read_text())
        arr = online_sequence_array(telemetry_path, self.probe_feature_names, self.probe_phases, self.probe_states)
        arr = (arr - np.asarray(q["dynamic_mean"], np.float32)) / np.asarray(q["dynamic_std"], np.float32)
        with torch.no_grad():
            mu, log_sigma = self.estimator(torch.tensor(arr[None]), torch.tensor([arr.shape[0]], dtype=torch.long))
        return float(mu.item()), float(torch.exp(log_sigma).item())

    def score(self, task: int, force: float, mu: float, state: np.ndarray, mask: np.ndarray, query_path: Path) -> dict[str, Any]:
        d = list(csv.DictReader(query_path.open(newline="", encoding="utf-8")))
        cmd = np.asarray([[float(r["cmd_x"]), float(r["cmd_y"]), float(r["cmd_z"])] for r in d], np.float64)
        nominal = nominal_segment(cmd, task, force, mu, state, mask)
        x = (nominal - self.gnp_norm[0]) / self.gnp_norm[1]
        step = torch.tensor(x[None, :, :17], dtype=torch.float32)
        cond = torch.tensor(x[None, 0, 17:], dtype=torch.float32)
        seed_rows = []
        with torch.no_grad():
            for seed, model in zip([0, 1, 2], self.models):
                logit = float(model(step, cond).item())
                p = float(1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50))))
                seed_rows.append({"seed": seed, "logit": logit, "raw_probability": p})
        return {"candidate_force_N": float(force), "raw_probability": float(np.mean([r["raw_probability"] for r in seed_rows])),
                "seed_scores": seed_rows, "input_shape": list(x.shape), "mu_input": float(mu),
                "normalization": "GNP_STYLE_TRAIN_NORMALIZATION.json"}

    def decide(self, task: int, mu: float, state: np.ndarray, mask: np.ndarray, query_path: Path) -> tuple[list[dict[str, Any]], float, bool]:
        scores = [self.score(task, f, mu, state, mask, query_path) for f in GRID]
        passing = [r for r in scores if r["raw_probability"] >= 0.5]
        return scores, float(passing[0]["candidate_force_N"] if passing else GRID[-1]), not bool(passing)


def configure_task(p6r1) -> None:
    p6r1.OBJECTS[TASK] = OBJECT
    p6r1.INSTRUCTIONS[TASK] = PROMPT
    p6r1.p6g1.TASK_OBJECTS[TASK] = OBJECT
    p6r1.p6g1.TASK_NAMES[TASK] = "alphabet soup"
    p6r1.p6g1.TASK_INSTRUCTIONS[TASK] = PROMPT


def capture_r0(env, p6, p4, p6r1, seed: int, mu: float, torch_mod) -> Any:
    env.reset(seed=seed)
    p6r1.p6g1.settle_root_before_hash(env, p6, p4, SETTLE_STEPS)
    p4._apply_friction(env, OBJECT, mu)
    return copy.deepcopy(env.scene.get_state(is_relative=True))


def run_probe_no_reset(env, p4, root_state, seed: int, mu: float, dt: float, trial_id: str, torch_mod):
    env.reset_to(copy.deepcopy(root_state), torch_mod.tensor([0], device=env.device), is_relative=True)
    original_reset = env.reset
    env.reset = lambda *_args, **_kwargs: (env.observation_manager.compute(), {})
    try:
        rows, rec = p4.run_probe_episode(env, seed_idx=seed, mu=mu, trial_id=trial_id, dt=dt)
    finally:
        env.reset = original_reset
    return rows, rec


def policy_summary(env, p6r1, client) -> dict[str, Any]:
    obs = env.observation_manager.compute()
    tactile = p6r1.p6g1.OnlineTactileBuffer(tactile_output_type="tactile_rgb")
    element = p6r1.p6g1.build_policy_observation(env, obs, PROMPT, tactile)
    result = client.infer(element)
    action = np.asarray(result["actions"], np.float32)
    return {"initial_rgb_hash": hashlib.sha256(np.ascontiguousarray(element["image"]).tobytes()).hexdigest(),
            "initial_wrist_rgb_hash": hashlib.sha256(np.ascontiguousarray(element["wrist_image"]).tobytes()).hexdigest(),
            "first_pi0_action": action[0].tolist(), "action_shape": list(action.shape),
            "controller_initialized_state": {"tactile_buffer": "fresh", "left_frames": 0, "right_frames": 0,
                                               "force_history": 0, "marker_history": 0}}


def execute_pi0(env, p6r1, p6, p4, client, root_state, force: float, method: str, seed: int, mu: float, dt: float, out: Path, repeat: int = 0) -> dict[str, Any]:
    import torch as torch_mod
    env.reset_to(copy.deepcopy(root_state), torch_mod.tensor([0], device=env.device), is_relative=True)
    # Preserve the existing friction setter across scene restores. This is
    # part of the frozen environment/root contract, not a method input.
    p4._apply_friction(env, OBJECT, float(mu))
    p6r1.FORCE_N = float(force)
    p6r1.FORCE_HALF_N = float(force) / 2.0
    tid = f"{method}_t0_s{seed}_mu{mu:g}_r{repeat}"
    meta = {"trial_id": tid, "task": TASK, "root_seed": seed, "arm": method,
            "policy_repeat": "", "policy_repeat_seed": f"{method}_{seed}_{mu:g}",
            "state_parity": 1, "staging_success": 1, "handoff_ready": 1}
    raw = p6r1.run_vla_full(env, p6, p4, client, TASK, None, meta, dt, out)
    return {"method": method, "root_seed": seed, "mu": mu, "requested_force_N": force,
            **raw, "force_telemetry_path": str(raw.get("force_telemetry_path", "")),
            "full_task_success": int(raw.get("full_task_success_y", 0))}


def run_dev_context() -> int:
    from isaaclab.app import AppLauncher
    p6r1 = load(P6_PATH, "two_method_dev_p6r1")
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    result: dict[str, Any] = {"status": "RUNNING", "test_roots_accessed": [], "dev_root": DEV_ROOT, "dev_mu": DEV_MU,
                               "threshold": 0.5, "candidates_N": GRID}
    try:
        from openpi_client import websocket_client_policy
        env, p6, p4 = p6r1.import_env_modules(TASK)
        configure_task(p6r1)
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        client = websocket_client_policy.WebsocketClientPolicy(PI0_HOST, PI0_PORT)
        torch_mod = torch
        root_state = capture_r0(env, p6, p4, p6r1, DEV_ROOT, DEV_MU, torch_mod)
        r0_hash = stable_hash(root_state)
        direct = DirectRuntime()
        rows, rec = run_probe_no_reset(env, p4, root_state, DEV_ROOT, DEV_MU, dt, "hf_dev_online_context_s5106", torch_mod)
        raw = probe_dicts(rows)
        probe_path = OUT / "DEV_ONLINE_CONTEXT" / "P4B_PROBE_TELEMETRY.csv"
        write_csv(probe_path, raw)
        state, mask, hold_step, opening = strict_state_from_probe(raw)
        eef = env.observation_manager.compute()["policy"]["eef_pose"][0].detach().cpu().numpy()
        query_path = OUT / "DEV_ONLINE_CONTEXT" / "DIRECT_QUERY_SKELETON.csv"
        write_query_skeleton(query_path, eef[:3], TASK)
        mu_hat, sigma = direct.estimate_mu(probe_path)
        scores, selected, fallback = direct.decide(TASK, mu_hat, state, mask, query_path)
        restored = False
        env.reset_to(copy.deepcopy(root_state), torch_mod.tensor([0], device=env.device), is_relative=True)
        restored_hash = stable_hash(env.scene.get_state(is_relative=True))
        restore_diff = max_abs(jsonable(root_state), jsonable(env.scene.get_state(is_relative=True)))
        p6r1.FORCE_N = selected; p6r1.FORCE_HALF_N = selected / 2.0
        exec_out = execute_pi0(env, p6r1, p6, p4, client, root_state, selected, "ACTIVEFORCING_DIRECT_DEV_ONLINE", DEV_ROOT, DEV_MU, dt, OUT / "DEV_ONLINE_CONTEXT" / "PI0_TELEMETRY")
        input_shape = scores[0]["input_shape"]
        ok = (len(rows) >= 1 and len(scores) == 9 and restore_diff <= 1e-5 and exec_out.get("episode_steps", 0) > 0)
        result.update({"status": "PASS" if ok else "BLOCKED_BY_DIRECT_TEST_CONTEXT_BUILDER",
                       "gate": "Gate 4 corrected online context", "probe_rows": len(rows), "probe_record": rec,
                       "probe_path": str(probe_path), "query_skeleton": str(query_path), "query_input_shape": input_shape,
                       "mu_hat": mu_hat, "sigma_mu": sigma, "strict_preprobe_hold_step": hold_step, "opening": opening,
                       "candidate_scores": scores, "selected_force_N": selected, "fallback_used": fallback,
                       "root_restore": {"r0_hash": r0_hash, "restored_hash": restored_hash, "max_abs_diff": restore_diff},
                       "execution": exec_out,
                       "test_status": "TEST_NOT_OPENED"})
        write_json(OUT / "DEV_CONTEXT_REBUILD_EQUIVALENCE.json", result)
        (OUT / "DEV_CONTEXT_REBUILD_EQUIVALENCE.md").write_text(
            "# DEV Context Rebuild Equivalence\n\n"
            f"Status: **{result['status']}**\n\n"
            "The formal TEST context is generated online from raw P4-B telemetry, "
            "the frozen probe feature builder/estimator, and an eight-row branch_hold "
            "query skeleton derived from the post-probe EEF command.\n\n"
            f"- raw probe rows: `{len(rows)}`\n- online μ̂: `{mu_hat:.8f}`; σ: `{sigma:.8f}`\n"
            f"- Direct input shape: `{input_shape}`\n- restore max abs diff: `{restore_diff:.8g}`\n"
            "- historical raw-probe rebuild is recorded separately in the CPU audit artifact.\n"
            "- TEST roots accessed: `none`\n",
            encoding="utf-8")
        return 0 if ok else 2
    except Exception as exc:
        result.update({"status": "BLOCKED_BY_DIRECT_TEST_CONTEXT_BUILDER", "error": repr(exc), "trace": traceback.format_exc(), "test_status": "TEST_NOT_OPENED"})
        write_json(OUT / "DEV_CONTEXT_REBUILD_EQUIVALENCE.json", result)
        (OUT / "DEV_CONTEXT_REBUILD_EQUIVALENCE.md").write_text("# DEV Context Rebuild Equivalence\n\nStatus: **BLOCKED_BY_DIRECT_TEST_CONTEXT_BUILDER**\n\n" + repr(exc) + "\n", encoding="utf-8")
        return 3
    finally:
        if env is not None:
            try: env.close()
            except Exception: pass
        try: app.close()
        except Exception: pass


def formal_plan() -> list[dict[str, Any]]:
    return [{"root_seed": root, "mu": mu, "repeat": rep, "task": TASK,
             "case_id": f"t0_s{root}_mu{mu:g}_r{rep}"}
            for root in TEST_ROOTS for mu in MUS for rep in range(1, REPEATS + 1)]


def run_formal() -> int:
    gate = json.loads((OUT / "DEV_CONTEXT_REBUILD_EQUIVALENCE.json").read_text())
    if gate.get("status") != "PASS":
        raise RuntimeError("TEST sealed: corrected DEV context rebuild equivalence is not PASS")
    FORMAL_OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STATUS.json", {
        "status": "FORMAL_TEST_RUNNING", "scope": "two-method matched comparison",
        "methods": ["pi0-Neutral", "ActiveForcing-Direct"], "task": 0,
        "test_roots": TEST_ROOTS, "mu_values": MUS, "repeats_per_case": REPEATS,
        "planned_rollouts": len(TEST_ROOTS) * len(MUS) * REPEATS * 2,
        "frontier_rollouts_completed": 0, "privileged_rollouts_completed": 0,
        "test_simulator_accessed": True, "test_entry_gate": "FORMAL_TEST_ENTRY_PASS",
        "direct_threshold": 0.5, "frontier_rho": 0.8, "afi_eta": 0.9,
        "test_context_generated_online": True,
    })
    from isaaclab.app import AppLauncher
    p6r1 = load(P6_PATH, "two_method_formal_p6r1")
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    try:
        from openpi_client import websocket_client_policy
        env, p6, p4 = p6r1.import_env_modules(TASK)
        configure_task(p6r1)
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        client = websocket_client_policy.WebsocketClientPolicy(PI0_HOST, PI0_PORT)
        direct = DirectRuntime()
        for plan in formal_plan():
            root, mu, rep, cid = int(plan["root_seed"]), float(plan["mu"]), int(plan["repeat"]), plan["case_id"]
            r0 = capture_r0(env, p6, p4, p6r1, root, mu, torch)
            r0_hash = stable_hash(r0)
            probe_rows, probe_rec = run_probe_no_reset(env, p4, r0, root, mu, dt, cid, torch)
            raw = probe_dicts(probe_rows)
            probe_valid = (int(probe_rec.get("probe_failure", 0)) == 0 and int(probe_rec.get("contact_lost_probe", 0)) == 0
                           and int(probe_rec.get("dropped", 0)) == 0 and len(raw) >= 1)
            if not probe_valid:
                raise RuntimeError(f"invalid frozen P4-B probe for {cid}: {probe_rec}")
            probe_path = FORMAL_OUT / "DIRECT" / "probe" / f"{cid}.csv"
            write_csv(probe_path, raw)
            state, mask, hold_step, opening = strict_state_from_probe(raw)
            eef = env.observation_manager.compute()["policy"]["eef_pose"][0].detach().cpu().numpy()
            query_path = FORMAL_OUT / "DIRECT" / "query_context" / f"{cid}.csv"
            write_query_skeleton(query_path, eef[:3], TASK)
            mu_hat, sigma = direct.estimate_mu(probe_path)
            scores, selected, fallback = direct.decide(TASK, mu_hat, state, mask, query_path)
            decisions.append({"case_id": cid, "root_seed": root, "mu": mu, "repeat": rep, "probe_rows": len(raw),
                              "mu_hat": mu_hat, "sigma_mu": sigma, "threshold": 0.5,
                              "candidate_scores": scores, "selected_force_N": selected, "fallback_used": fallback,
                              "probe_path": str(probe_path), "query_context_path": str(query_path),
                              "r0_hash": r0_hash, "strict_preprobe_hold_step": hold_step, "opening": opening})
            direct_out = execute_pi0(env, p6r1, p6, p4, client, r0, selected, "ACTIVEFORCING_DIRECT", root, mu, dt, FORMAL_OUT / "DIRECT", rep)
            direct_out.update({"case_id": cid, "repeat": rep, "root_restore_hash": r0_hash, "privileged": False,
                               "probe_used": True, "mu_hat": mu_hat, "sigma_mu": sigma,
                               "selected_force_N": selected, "fallback_used": fallback, "probe_rows": len(raw)})
            rows.append(direct_out)
            # Matched Neutral starts from the same saved R0 after the Direct
            # branch; it performs no probe and uses default zero-force π0.
            neutral = execute_pi0(env, p6r1, p6, p4, client, r0, 0.0, "PI0_NEUTRAL", root, mu, dt, FORMAL_OUT / "PI0_NEUTRAL", rep)
            neutral.update({"case_id": cid, "repeat": rep, "root_restore_hash": r0_hash, "privileged": False, "probe_used": False})
            rows.append(neutral)
            print(json.dumps({"completed_cases": len(decisions), "total_cases": len(TEST_ROOTS) * len(MUS) * REPEATS,
                              "case_id": cid, "selected_force_N": selected, "neutral_success": neutral.get("full_task_success"),
                              "direct_success": direct_out.get("full_task_success")}), flush=True)
        write_csv(FORMAL_OUT / "TWO_METHOD_FORMAL_RESULTS.csv", rows)
        write_csv(FORMAL_OUT / "DIRECT_TEST_DECISIONS.csv", [{k: (json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in decisions])
        write_json(FORMAL_OUT / "TWO_METHOD_FORMAL_MANIFEST.json", {
            "status": "COMPLETED", "scope": "PRETEST_METHOD_SCOPE_REDUCTION", "methods": ["π0-Neutral", "ActiveForcing-Direct"],
            "task": TASK, "test_roots": TEST_ROOTS, "mu_values": MUS, "repeats_per_root_mu": REPEATS,
            "planned_cases": len(TEST_ROOTS) * len(MUS) * REPEATS, "planned_rollouts": len(rows), "test_simulator_accessed": True,
            "direct_threshold": 0.5, "frontier_rho": 0.8, "afi_eta": 0.9, "afi_role": "ablation_only",
            "no_privileged_methods_run": True, "gate_status": gate,
        })
        return 0
    except Exception as exc:
        write_csv(FORMAL_OUT / "TWO_METHOD_FORMAL_RESULTS.csv", rows)
        write_csv(FORMAL_OUT / "DIRECT_TEST_DECISIONS.csv", [{k: (json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in decisions])
        write_json(FORMAL_OUT / "TWO_METHOD_FORMAL_ERROR.json", {"status": "FORMAL_PARTIAL_INFRASTRUCTURE_BLOCKER", "error": repr(exc), "trace": traceback.format_exc(), "completed_rollouts": len(rows), "completed_decisions": len(decisions), "test_roots_accessed": sorted({int(r["root_seed"]) for r in formal_plan()[:len(decisions)]})})
        return 3
    finally:
        if env is not None:
            try: env.close()
            except Exception: pass
        try: app.close()
        except Exception: pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev-context", "formal"], required=True)
    args = ap.parse_args()
    return run_dev_context() if args.mode == "dev-context" else run_formal()


if __name__ == "__main__":
    raise SystemExit(main())
