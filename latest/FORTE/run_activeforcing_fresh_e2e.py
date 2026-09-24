#!/usr/bin/env python3
"""Current frozen ActiveForcing-Direct fresh E2E runner.

This is an engineering adapter around the existing frozen P6G1R1 π0 loop.  It
does not train, tune, alter nominal motion generation, or change the Direct
selector.  Each root/friction tuple is reset before every arm; query arms
share a single P4-B query and exact root restore, while NoQuery-Prior never
executes the physical query.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

FORTE = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
RESULTS = TABERO / "analysis/results"
P6R1_PATH = TABERO / "analysis/p6g1r1_controller_grasp_vla_handoff.py"
TWO_METHOD_PATH = FORTE / "run_two_method_formal.py"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
WARP_CORE = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64")
OPENPI_CLIENT = TABERO / "benchmarks/openpi/openpi-client/src"

TASKS = [0, 1, 5, 6]
ACTIVE_TASKS = list(TASKS)
OBJECTS = {0: "alphabet_soup_1", 1: "cream_cheese_1", 5: "tomato_sauce_1", 6: "butter_1"}
INSTRUCTIONS = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
}
BANDS = {"LOW": (0.20, 0.30), "MID": (0.45, 0.60), "HIGH": (0.90, 1.00)}
ROOTS_PER_TASK = {"smoke": 1, "full": 5}
ROOT_BASE = {"smoke": 7100, "full": 7200}
FRICTION_SEED = 2026082405
PRIOR_MU = 0.60
GRID = [3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]
FIXED_MAX = {0: 5.0, 1: 5.0, 5: 5.0, 6: 4.0}
DEFAULT_FORCE = 0.0
SETTLE_STEPS = 20

ARM_NAMES = ["FROZEN_VLA_DEFAULT", "FIXED_MAX", "NO_QUERY_PRIOR", "ACTIVEFORCING_DIRECT", "GT_PHYSICS_DIRECT"]
FIELDS = [
    "phase", "task", "object", "root_id", "root_seed", "friction_band", "friction",
    "method", "query_state_reached", "query_valid", "query_duration_s", "query_rows",
    "mu_hat", "sigma_mu", "prior_mu", "selected_force_N", "selection_fallback",
    "full_task_success_y", "pick_success", "lift_success", "transport_retention",
    "place_success", "dropped", "lost_in_transit", "under_force", "excess_force",
    "failure_stage", "failure_reason", "measured_force_mean_N", "measured_force_peak_N",
    "trajectory_path", "query_path", "decision_path", "root_state_hash", "restore_hash",
    "restore_parity", "episode_steps", "error", "started_utc", "ended_utc",
]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields or ["status"], extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)


def append_csv(path: Path, row: dict, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if not exists:
            wr.writeheader()
        wr.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def plan(phase: str) -> list[dict]:
    n = ROOTS_PER_TASK[phase]
    rng = np.random.default_rng(FRICTION_SEED + (0 if phase == "smoke" else 100))
    rows = []
    for task in ACTIVE_TASKS:
        for ri in range(n):
            seed = ROOT_BASE[phase] + ri
            root_id = f"activeforcing_{phase}_t{task}_root{ri:02d}_s{seed}"
            for band, (lo, hi) in BANDS.items():
                mu = float(rng.uniform(lo, hi))
                rows.append({"phase": phase, "task": task, "object": OBJECTS[task], "root_index": ri,
                             "root_seed": seed, "root_id": root_id, "friction_band": band,
                             "friction": mu, "tuple_id": f"{root_id}_{band.lower()}_mu{mu:.6f}"})
    return rows


def configure_task(p6r1, task: int) -> None:
    obj = OBJECTS[task]
    prompt = INSTRUCTIONS[task]
    p6r1.OBJECTS[task] = obj
    p6r1.INSTRUCTIONS[task] = prompt
    p6r1.p6g1.TASK_OBJECTS[task] = obj
    p6r1.p6g1.TASK_NAMES[task] = obj.removesuffix("_1").replace("_", " ")
    p6r1.p6g1.TASK_INSTRUCTIONS[task] = prompt


def capture_root(env, p6r1, p6, p4, task: int, seed: int, mu: float):
    import torch
    env.reset(seed=int(seed))
    p6r1.p6g1.settle_root_before_hash(env, p6, p4, SETTLE_STEPS)
    p4._apply_friction(env, OBJECTS[task], float(mu))
    state = copy.deepcopy(env.scene.get_state(is_relative=True))
    return state, stable_hash(state)


def restore(env, state):
    import torch
    env.reset_to(copy.deepcopy(state), torch.tensor([0], device=env.device), is_relative=True)
    restored = copy.deepcopy(env.scene.get_state(is_relative=True))
    return restored, stable_hash(restored)


def write_probe(path: Path, rows: list[Any]) -> None:
    write_csv(path, [jsonable(vars(x) if hasattr(x, "__dict__") else x) for x in rows])


def eef_and_query(p6r1, env, task: int, path: Path) -> None:
    obs = env.observation_manager.compute()
    eef = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    rows = [{"step": i + 1, "phase": "branch_hold", "cmd_x": float(eef[0]), "cmd_y": float(eef[1]), "cmd_z": float(eef[2])} for i in range(8)]
    write_csv(path, rows)


def query_quality(rows: list[dict], rec: dict) -> tuple[int, int]:
    reached = int(bool(rows) and any(str(r.get("probe_phase", "")) == "probe_out" for r in rows))
    valid = int(reached and int(rec.get("probe_failure", 0)) == 0 and int(rec.get("contact_lost_probe", 0)) == 0 and int(rec.get("dropped", 0)) == 0)
    return reached, valid


def as_int(row: dict, key: str) -> int:
    try:
        return int(float(row.get(key, 0)))
    except Exception:
        return 0


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def run_arm(p6r1, env, p6, p4, client, task: int, item: dict, state, root_hash: str, force: float, method: str, out: Path, query_info: dict) -> dict:
    import torch
    restored, restore_hash = restore(env, state)
    parity = int(restore_hash == root_hash)
    p4._apply_friction(env, OBJECTS[task], float(item["friction"]))
    p6r1.FORCE_N = float(force)
    p6r1.FORCE_HALF_N = float(force) / 2.0
    p6r1.p6g1.FIXED_FORCE_N = float(force)
    p6r1.p6g1.FORCE_SLOT_HALF_N = float(force) / 2.0
    trial = f"{item['tuple_id']}_{method}"
    started = now()
    try:
        raw = p6r1.run_vla_full(
            env, p6, p4, client, task, None,
            {"trial_id": trial, "task": task, "root_seed": item["root_seed"], "arm": method,
             "policy_repeat": 0, "policy_repeat_seed": f"{method}_{item['root_seed']}",
             "state_parity": parity, "staging_success": 1, "handoff_ready": 1},
            float(env.cfg.sim.dt) * int(env.cfg.decimation), out,
        )
    except Exception as exc:
        raw = {"full_task_success_y": 0, "failure_stage": "OTHER", "error": repr(exc), "episode_steps": 0}
    ended = now()
    requested = float(force)
    failed = as_int(raw, "full_task_success_y") == 0
    stage = str(raw.get("failure_stage", ""))
    under = int(failed and stage in {"drop", "transport", "placement"} and requested < FIXED_MAX[task])
    excess = float(max(requested - 4.0, 0.0)) if requested > 0 else 0.0
    return {
        "phase": item["phase"], "task": task, "object": OBJECTS[task], "root_id": item["root_id"],
        "root_seed": item["root_seed"], "friction_band": item["friction_band"], "friction": item["friction"],
        "method": method, **query_info, "selected_force_N": requested,
        "selection_fallback": query_info.get("selection_fallback", 0),
        "full_task_success_y": as_int(raw, "full_task_success_y"), "pick_success": as_int(raw, "pick_success"),
        "lift_success": as_int(raw, "stable_lift"), "transport_retention": as_int(raw, "transport_grasp_retained"),
        "place_success": as_int(raw, "placement_success"), "dropped": as_int(raw, "drop"),
        "lost_in_transit": int(stage == "transport"), "under_force": under, "excess_force": excess,
        "failure_stage": stage, "failure_reason": str(raw.get("error", "")),
        "measured_force_mean_N": raw.get("mean_measured_force_N", ""), "measured_force_peak_N": raw.get("peak_measured_force_N", ""),
        "trajectory_path": str(raw.get("contact_telemetry_path", "")), "query_path": query_info.get("query_path", ""),
        "decision_path": query_info.get("decision_path", ""), "root_state_hash": root_hash,
        "restore_hash": restore_hash, "restore_parity": parity, "episode_steps": raw.get("episode_steps", 0),
        "error": str(raw.get("error", "")), "started_utc": started, "ended_utc": ended,
    }


def run_phase(out: Path, phase: str) -> dict:
    from isaaclab.app import AppLauncher
    import torch
    import gymnasium as gym  # noqa: F401
    from openpi_client import websocket_client_policy

    p6r1 = load(P6R1_PATH, f"activeforcing_p6r1_{os.getpid()}")
    two = load(TWO_METHOD_PATH, f"activeforcing_two_{os.getpid()}")
    # The historical adapter's 22-chunk cap is a runtime guard, not part of
    # the scientific method.  Keep the smoke run reproducible at its recorded
    # default, while allowing a later fresh run to use a larger end-to-end
    # budget after qualification.
    if phase == "full":
        p6r1.VLA_MAX_CHUNKS = int(os.environ.get("AF_VLA_MAX_CHUNKS", "50"))
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    rows: list[dict] = []
    queries: list[dict] = []
    decisions: list[dict] = []
    worker_errors: list[dict] = []
    try:
        client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 18881)
        for task in ACTIVE_TASKS:
            env = None
            try:
                configure_task(p6r1, task)
                env, p6, p4 = p6r1.import_env_modules(task)
                dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
                direct = two.DirectRuntime()
                for item in [x for x in plan(phase) if int(x["task"]) == task]:
                    state, root_hash = capture_root(env, p6r1, p6, p4, task, item["root_seed"], item["friction"])
                    restored, restored_hash = restore(env, state)
                    query_path = out / "query" / f"{item['tuple_id']}.csv"
                    probe_path = out / "probe" / f"{item['tuple_id']}.csv"
                    decision_dir = out / "decisions"
                    # Root-only skeleton for NoQuery-Prior. No physical query is executed.
                    eef_and_query(p6r1, env, task, query_path)
                    prior_scores, prior_force, prior_fallback = direct.decide(task, PRIOR_MU, np.zeros(13, np.float32), np.zeros(13, np.float32), query_path)
                    prior_decision_path = decision_dir / f"{item['tuple_id']}_NO_QUERY_PRIOR.json"
                    write_json(prior_decision_path, {"method": "NO_QUERY_PRIOR", "task": task, "prior_mu": PRIOR_MU, "scores": prior_scores, "selected_force_N": prior_force, "fallback": prior_fallback})
                    qinfo = {"query_state_reached": 0, "query_valid": 0, "query_duration_s": "", "query_rows": 0, "mu_hat": "", "sigma_mu": "", "prior_mu": PRIOR_MU, "query_path": "", "decision_path": str(prior_decision_path), "selection_fallback": int(prior_fallback)}
                    decisions.append({"tuple_id": item["tuple_id"], "task": task, "method": "NO_QUERY_PRIOR", "selected_force_N": prior_force, "fallback": prior_fallback, "scores": prior_scores})
                    # One shared current P4-B query for ActiveForcing and the GT-physics arm.
                    # The authoritative helper returns (probe_rows, probe_record).
                    # Capture the post-query simulator state separately for provenance;
                    # this adapter must not invent an additional return contract.
                    probe_steps, probe_rec = two.run_probe_no_reset(env, p4, state, int(item["root_seed"]), float(item["friction"]), dt, item["tuple_id"], torch)
                    post_query = copy.deepcopy(env.scene.get_state(is_relative=True))
                    raw_probe = two.probe_dicts(probe_steps)
                    write_probe(probe_path, probe_steps)
                    query_reached, query_valid = query_quality(raw_probe, probe_rec)
                    eef_and_query(p6r1, env, task, query_path)
                    strict_state, strict_mask, hold_step, opening = two.strict_state_from_probe(raw_probe)
                    mu_hat, sigma_mu = direct.estimate_mu(probe_path)
                    scores, selected, fallback = direct.decide(task, mu_hat, strict_state, strict_mask, query_path)
                    direct_path = decision_dir / f"{item['tuple_id']}_ACTIVEFORCING_DIRECT.json"
                    write_json(direct_path, {"method": "ACTIVEFORCING_DIRECT", "task": task, "friction_band": item["friction_band"], "mu_hat": mu_hat, "sigma_mu": sigma_mu, "threshold": 0.5, "scores": scores, "selected_force_N": selected, "fallback": fallback, "query_path": str(probe_path), "hold_step": hold_step, "opening": opening})
                    gt_path = decision_dir / f"{item['tuple_id']}_GT_PHYSICS_DIRECT.json"
                    gt_scores, gt_selected, gt_fallback = direct.decide(task, float(item["friction"]), strict_state, strict_mask, query_path)
                    write_json(gt_path, {"method": "GT_PHYSICS_DIRECT", "task": task, "friction": item["friction"], "threshold": 0.5, "scores": gt_scores, "selected_force_N": gt_selected, "fallback": gt_fallback})
                    queries.append({**item, "query_state_reached": query_reached, "query_valid": query_valid, "query_rows": len(raw_probe), "query_duration_s": probe_rec.get("probe_duration_s", ""), "probe_path": str(probe_path), "mu_hat": mu_hat, "sigma_mu": sigma_mu, "root_hash": root_hash, "post_query_hash": stable_hash(post_query), "restored_hash": restored_hash, "restore_parity": int(restored_hash == root_hash)})
                    decisions.extend([
                        {"tuple_id": item["tuple_id"], "task": task, "method": "ACTIVEFORCING_DIRECT", "selected_force_N": selected, "fallback": fallback, "scores": scores},
                        {"tuple_id": item["tuple_id"], "task": task, "method": "GT_PHYSICS_DIRECT", "selected_force_N": gt_selected, "fallback": gt_fallback, "scores": gt_scores},
                    ])
                    active_info = {"query_state_reached": query_reached, "query_valid": query_valid, "query_duration_s": probe_rec.get("probe_duration_s", ""), "query_rows": len(raw_probe), "mu_hat": mu_hat, "sigma_mu": sigma_mu, "prior_mu": PRIOR_MU, "query_path": str(probe_path), "decision_path": str(direct_path), "selection_fallback": int(fallback)}
                    gt_info = {**active_info, "mu_hat": float(item["friction"]), "sigma_mu": 0.0, "decision_path": str(gt_path), "selection_fallback": int(gt_fallback)}
                    arms = [("FROZEN_VLA_DEFAULT", DEFAULT_FORCE, {"query_state_reached": 0, "query_valid": 0, "query_duration_s": "", "query_rows": 0, "mu_hat": "", "sigma_mu": "", "prior_mu": PRIOR_MU, "query_path": "", "decision_path": "", "selection_fallback": 0}), ("FIXED_MAX", FIXED_MAX[task], {"query_state_reached": 0, "query_valid": 0, "query_duration_s": "", "query_rows": 0, "mu_hat": "", "sigma_mu": "", "prior_mu": PRIOR_MU, "query_path": "", "decision_path": "", "selection_fallback": 0}), ("NO_QUERY_PRIOR", prior_force, qinfo), ("ACTIVEFORCING_DIRECT", selected, active_info), ("GT_PHYSICS_DIRECT", gt_selected, gt_info)]
                    for method, force, info in arms:
                        rows.append(run_arm(p6r1, env, p6, p4, client, task, item, state, root_hash, force, method, out, info))
                        append_csv(out / "E5_ROLLOUTS.csv", rows[-1], FIELDS)
                    print(json.dumps({"phase": phase, "tuple_id": item["tuple_id"], "task": task, "selected_direct": selected, "selected_gt": gt_selected, "completed_arms": len(arms)}, sort_keys=True), flush=True)
            except Exception as exc:
                worker_errors.append({"phase": phase, "task": task, "error": repr(exc), "trace": traceback.format_exc()})
                write_json(out / "errors" / f"task{task}.json", worker_errors[-1])
            finally:
                if env is not None:
                    try: env.close()
                    except Exception: pass
        write_csv(out / "E5_QUERY_RESULTS.csv", queries)
        write_csv(out / "E5_DECISIONS.jsonl.csv", [{"tuple_id": x.get("tuple_id", ""), "task": x.get("task", ""), "method": x.get("method", ""), "selected_force_N": x.get("selected_force_N", ""), "fallback": x.get("fallback", ""), "scores": json.dumps(x.get("scores", []), sort_keys=True)} for x in decisions])
        write_json(out / f"{phase.upper()}_STATUS.json", {"status": "PASS" if not worker_errors else "PARTIAL", "phase": phase, "tasks": ACTIVE_TASKS, "planned_tuples": len(plan(phase)), "completed_rollouts": len(rows), "expected_rollouts": len(plan(phase)) * len(ARM_NAMES), "query_rows": len(queries), "worker_errors": worker_errors})
        return {"rows": rows, "queries": queries, "decisions": decisions, "errors": worker_errors}
    finally:
        try: app.close()
        except Exception: pass


def aggregate(out: Path, phase: str, payload: dict) -> None:
    rows = payload["rows"]
    by_method = {}
    for method in ARM_NAMES:
        g = [r for r in rows if r.get("method") == method]
        by_method[method] = {"n": len(g), "full_task_success_rate": float(np.mean([as_int(x, "full_task_success_y") for x in g])) if g else None, "query_state_reach_rate": float(np.mean([as_int(x, "query_state_reached") for x in g])) if g else None, "query_valid_rate": float(np.mean([as_int(x, "query_valid") for x in g])) if g else None, "conditional_downstream_success": float(np.mean([as_int(x, "full_task_success_y") for x in g if as_int(x, "query_valid")])) if any(as_int(x, "query_valid") for x in g) else None, "mean_selected_force_N": float(np.mean([as_float(x, "selected_force_N") for x in g])) if g else None, "under_force_rate": float(np.mean([as_int(x, "under_force") for x in g])) if g else None, "mean_measured_force_N": float(np.mean([as_float(x, "measured_force_mean_N") for x in g])) if g else None}
    write_json(out / f"{phase.upper()}_SUMMARY.json", {"phase": phase, "methods": by_method, "root_cluster_unit": "root_id", "bootstrap": "reserved for final aggregation", "data_semantics": {"default_force": DEFAULT_FORCE, "no_query_prior_mu": PRIOR_MU, "gt_physics_is_privileged": True}})
    lines = [f"# ActiveForcing current Direct fresh E2E — {phase}", "", "`METHOD_CHANGE = NONE`", "", f"- Rollouts: `{len(rows)}` / `{len(plan(phase)) * len(ARM_NAMES)}`", "- Paired tuple: task × root × friction band; each arm restores the same root.", "- Query arms use one P4-B query and exact root restore; NoQuery-Prior does not execute P4-B.", "- Frozen Direct threshold: `0.5`; candidate grid: `" + ", ".join(f"{x:.2f}" for x in GRID) + " N`.", "", "| method | n | E2E SR | query reach | query valid | conditional downstream | mean force N | under-force |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, s in by_method.items():
        f = lambda x: "NA" if x is None else f"{x:.3f}"
        lines.append(f"| {method} | {s['n']} | {f(s['full_task_success_rate'])} | {f(s['query_state_reach_rate'])} | {f(s['query_valid_rate'])} | {f(s['conditional_downstream_success'])} | {f(s['mean_selected_force_N'])} | {f(s['under_force_rate'])} |")
    lines += ["", "## Semantic caveats", "", f"`FROZEN_VLA_DEFAULT` is executed as the frozen neutral zero-force override (`{DEFAULT_FORCE:.1f} N`) because an exact native-slot default is not present in the archived 720 branch grid; this is preserved as a data-gap label, not a nearest-force substitution.", "`GT_PHYSICS_DIRECT` receives simulator friction as a privileged oracle and is not deployable.", "`excess_force` is a descriptive force-cost field (`max(selected−4.0,0)`) and is not used for selection."]
    (out / f"{phase.upper()}_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "full"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", type=int, action="append", choices=TASKS, help="run only selected task(s) in a fresh worker")
    args = ap.parse_args()
    global ACTIVE_TASKS
    ACTIVE_TASKS = args.task if args.task else list(TASKS)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if not (out / "RUN_PROTOCOL.json").exists():
        write_json(out / "RUN_PROTOCOL.json", {"experiment": "ACTIVEFORCING_FINAL_E5", "phase": args.phase, "method_change": "NONE", "tasks": TASKS, "arms": ARM_NAMES, "grid_N": GRID, "prior_mu": PRIOR_MU, "default_force_N": DEFAULT_FORCE, "fixed_max_by_task": FIXED_MAX, "roots": plan(args.phase), "runner_sha256_at_start": sha256_file(Path(__file__).resolve()), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=FORTE, text=True).strip(), "start_utc": now()})
    payload = run_phase(out, args.phase)
    aggregate(out, args.phase, payload)
    return 0 if not payload["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
