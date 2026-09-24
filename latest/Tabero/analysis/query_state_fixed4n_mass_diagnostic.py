#!/usr/bin/env python3
"""Fresh paired Mass E2E using the existing authoritative frozen π0 loop.

This diagnostic reuses the frozen P6G1R1 executor and websocket server through
the E5 runner's read-only helpers.  In fixed-force mode it performs only
reset -> frozen P4-B query -> live post-query handoff -> one 4 N arm; no mass
prediction or Direct/Utility force selection is executed.
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
import traceback
from pathlib import Path

import numpy as np

OUT_DEFAULT = Path("/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_FINAL_CLOSURE_20260903/fresh_e2e")
CLOSURE = Path("/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_FINAL_CLOSURE_20260903")
E5_PATH = Path("/home/exouser/FORTE/analysis/results/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_053006/run_e5_fresh_utility.py")
P6_PATH = Path("/home/exouser/Tabero/analysis/p6g1r1_controller_grasp_vla_handoff.py")
FORMAL = Path("/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M3_TASK2_FORMAL_STRUCTURED_20260902_130700_RESUME")
TASK = 2
OBJECT = "salad_dressing_1"
BANDS = {"LOW": 0.05, "MID": 0.10, "HIGH": 0.20}
FORCES = (0.5, 1.0, 1.5, 2.5, 4.0)
SEEDS = (11, 23, 37)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or ["status"], extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def protected_isaac_workers() -> list[str]:
    """Return other Isaac workers, excluding this coordinator process."""
    raw = subprocess.run(
        ["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=False
    ).stdout.splitlines()
    own_pid = str(os.getpid())
    return [
        line.strip()
        for line in raw
        if own_pid not in line.split(maxsplit=1)[0:1]
        and (
            " -u /home/exouser/FORTE/analysis/results/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_053006/run_e5_fresh_utility.py " in line
            or " -u /home/exouser/FORTE_mass/collect_mass_fresh_e2e.py " in line
            or " -u /home/exouser/E3_E6_E7_LANES/E7_CONTINUOUS_FORCE_PLANNING/activeforcing_e7_offgrid_rollout.py " in line
            or " -u /home/exouser/E3_E6_E7_LANES/E6_DECISION_AWARE_REQUERY/e6_second_query_pilot.py " in line
        )
        and "rg " not in line
        and "/bin/bash -c " not in line
    ]


def query_features(rec):
    return np.asarray([float(rec.get("f_meas_mean", rec.get("f_meas_mean_N", 0.0))), float(rec.get("normal_force_mean", rec.get("normal_force_mean_N", 0.0))), float(rec.get("normal_force_peak", rec.get("normal_force_peak_N", 0.0))), float(rec.get("rho_mean", 0.0)), float(rec.get("marker_mean", 0.0)), float(rec.get("obj_disp_probe_m", 0.0)), float(rec.get("obj_rot_probe_rad", 0.0))], dtype=np.float32)


def fit_identifier(contexts, device="cpu"):
    model = load(Path(__file__).with_name("mass_modeling_final.py"), "mass_modeling_for_fresh")
    train = [r for r in contexts if r["split"] == "TRAIN" and int(r["query_valid"]) == 1]
    x = np.asarray([model.physical_vector(r) for r in train], dtype=np.float32)
    y = np.asarray([float(r["mass_kg"]) for r in train], dtype=np.float32)
    models = [model.train_regressor(x, y, seed, __import__("torch").device(device)) for seed in SEEDS]
    return lambda rec: float(np.mean([float(m(query_features_as_physical(rec, train[0])).ravel()[0]) for m in models]))


def query_features_as_physical(rec, template):
    q = dict(template)
    q["query_record"] = json.dumps(rec)
    return load(Path(__file__).with_name("mass_modeling_final.py"), "mass_modeling_features_for_fresh").physical_vector(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--roots", type=int, nargs="+", default=(8400, 8401))
    ap.add_argument("--bands", type=str, nargs="+", choices=tuple(BANDS), default=tuple(BANDS))
    ap.add_argument("--query-only", action="store_true", help="run and persist the frozen query only; no policy arm is executed")
    ap.add_argument("--query-state-fixed-force", type=float, help="diagnostic only: run one fixed force from the same post-query handoff")
    ap.add_argument(
        "--scheduler-compat-marker",
        default="",
        help="marker used only so the resident scheduler observes this worker",
    )
    args = ap.parse_args()
    fixed4n = args.query_state_fixed_force is not None
    out = args.out
    selected_bands = {band: BANDS[band] for band in args.bands}
    out.mkdir(parents=True, exist_ok=True)
    # The resident E5 scheduler normally injects these variables.  Fresh Mass
    # is launched independently, so carry the same authoritative environment
    # explicitly and avoid API drift between scheduler and direct invocation.
    env_defaults = {
        "PYTHONNOUSERSITE": "1",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ACCEPT_EULA": "Y",
        "TABERO_ROOT": "/home/exouser/Tabero",
        "HDF5_TRAJ_SOURCE_DIR": "/home/exouser/Tabero/benchmarks/datasets/libero/assembled_hdf5",
        "LIBERO_ASSETS_DATA_DIR": "/home/exouser/Tabero/benchmarks/datasets/libero/USD",
        "LIBERO_CONFIG_DIR": "/home/exouser/Tabero/benchmarks/datasets/libero/config",
    }
    for key, value in env_defaults.items():
        os.environ.setdefault(key, value)
    while True:
        active_workers = protected_isaac_workers()
        if not active_workers:
            break
        print("waiting for protected Isaac worker to exit; no signal will be sent", flush=True)
        __import__("time").sleep(5)
    qa = json.loads((CLOSURE / "MASS_FINAL_BRANCH_QA.json").read_text(encoding="utf-8"))
    if qa.get("status") != "PASS" or qa.get("accepted_branches") != 180:
        raise RuntimeError("fresh E2E requires PASS final QA")
    contexts = list(csv.DictReader((FORMAL / "M3_TASK2_STRUCTURED_FORMAL_CONTEXTS.csv").open(newline="", encoding="utf-8")))
    branches = list(csv.DictReader((FORMAL / "M3_TASK2_STRUCTURED_FORMAL_BRANCHES.csv").open(newline="", encoding="utf-8")))
    train_b = [r for r in branches if r["split"] == "TRAIN"]
    import torch
    if fixed4n:
        modeling = None
        py = np.asarray([], dtype=np.float32)
        id_models = []
        direct_models = []
    else:
        modeling = load(Path(__file__).with_name("mass_modeling_final.py"), "mass_modeling_for_fresh_main")
        physical_train = [r for r in contexts if r["split"] == "TRAIN" and int(r["query_valid"]) == 1]
        px = np.asarray([modeling.physical_vector(r) for r in physical_train], dtype=np.float32)
        py = np.asarray([float(r["mass_kg"]) for r in physical_train], dtype=np.float32)
        id_models = [modeling.train_regressor(px, py, seed, torch.device("cpu")) for seed in SEEDS]
        direct_models = [modeling.train_direct(train_b, seed, torch.device("cpu")) for seed in SEEDS]
    e5 = load(E5_PATH, "mass_e5_frozen_helpers")
    # The authoritative E5 Utility table covers tasks 0/1/5/6.  Mass task 2
    # uses the closure's predeclared F_MAX=8 utility; extend only the in-memory
    # helper dictionaries so shared rollout/decorate code can serialize task 2.
    e5.FMAX[TASK] = 8.0
    e5.utility.FMAX[TASK] = 8.0
    e5.utility.FORCE_SUPPORT_N[TASK] = (0.5, 4.0)
    # The formal Mass protocol has five explicit candidates.  The inherited
    # E5 serializer defaults to nine evenly spaced candidates; align its
    # metadata/runtime grid with the frozen Mass candidate set.
    e5.utility.CANDIDATES_PER_TASK = len(FORCES)
    e5.base.FIXED_MAX[TASK] = 4.0
    pi0_audit = e5.pi0_server_gate()
    gpu = e5.gpu_snapshot()
    if not (gpu["utilization_pct"] < 70 and gpu["memory_free_mib"] > 10 * 1024):
        raise RuntimeError("fresh E2E resource gate closed: " + json.dumps(gpu))
    p6r1 = load(P6_PATH, f"mass_fresh_p6r1_{os.getpid()}")
    # The E5 helper's configure_task reads its own base module dictionaries;
    # extend those dictionaries before calling it (task 2 is outside the
    # original E5 task set).
    e5.base.OBJECTS[TASK] = OBJECT
    e5.base.INSTRUCTIONS[TASK] = "pick up the salad dressing and place it in the basket"
    p6r1.OBJECTS[TASK] = OBJECT; p6r1.INSTRUCTIONS[TASK] = "pick up the salad dressing and place it in the basket"
    p6r1.p6g1.TASK_OBJECTS[TASK] = OBJECT; p6r1.p6g1.TASK_NAMES[TASK] = "salad dressing"; p6r1.p6g1.TASK_INSTRUCTIONS[TASK] = "pick up the salad dressing and place it in the basket"
    e5.base.configure_task(p6r1, TASK)
    native_exec, transform_hashes = e5.native_pi0_executor(p6r1)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    import gymnasium as gym
    from openpi_client import websocket_client_policy
    env = None; rows = []; queries = []; decisions = []
    if args.query_state_fixed_force is not None and not (0.0 < args.query_state_fixed_force <= 8.0):
        raise ValueError("diagnostic fixed force must be in (0, 8] N; 8 N is the existing controller force-slot safety envelope")
    diagnostic_policy = "QUERY_STATE_FIXED_FORCE" if args.query_state_fixed_force is not None else None
    run_policies = (diagnostic_policy,) if diagnostic_policy else ("ACTIVEFORCING_MASS", "FROZEN_PI0_NATIVE_DEFAULT", "FIXED_MAX", "TRUE_NOQUERY_PRIOR", "GT_MASS")
    protocol = {"status": "RUNNING", "task": TASK, "object": OBJECT, "roots": list(args.roots), "mass_bands_kg": selected_bands, "candidate_forces_N": FORCES, "policies": list(run_policies), "query": "frozen_P4B", "frozen_pi0_server": pi0_audit, "identifier_fit": "NOT USED (fixed-force diagnostic)", "direct_fit": "NOT USED (fixed-force diagnostic)", "utility": "NOT USED (fixed-force diagnostic)", "mass_prediction_used": False, "force_selection_used": False, "query_state_handoff": True, "frozen_controller_evaluator": True, "protected_processes_untouched": True, "diagnostic_fixed_force_N": args.query_state_fixed_force}
    write_json(out / "MASS_FRESH_E2E_PROTOCOL.json", protocol)
    try:
        client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 18881)
        base = e5.base
        utility = None if fixed4n else modeling.utility
        # Build one Isaac environment for the complete paired-root run.  The
        # environment reset/capture below still makes every root×mass context
        # fresh; rebuilding the simulator between roots can deadlock in the
        # Isaac/PhysX shutdown path while leaving a live GPU context behind.
        base.configure_task(p6r1, TASK)
        env, p6, p4 = p6r1.import_env_modules(TASK)
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        original_apply = p4._apply_friction
        mass_state = {}
        def apply_mass(e, name, mu, _orig=original_apply):
            result = _orig(e, name, mu)
            view = e.scene[OBJECT].root_physx_view
            if not mass_state:
                mass_state["mass"] = view.get_masses().clone(); mass_state["inertia"] = view.get_inertias().clone(); mass_state["nominal"] = float(view.get_masses()[0, 0].item())
            ratio = float(os.environ["MASS_QUERY_CURRENT_KG"]) / mass_state["nominal"]
            ids = torch.arange(1, dtype=torch.int32)
            view.set_masses(mass_state["mass"] * ratio, ids); view.set_inertias(mass_state["inertia"] * ratio, ids)
            return result
        p4._apply_friction = apply_mass
        for root in args.roots:
            base.configure_task(p6r1, TASK)
            for band, mass in selected_bands.items():
                item = {"phase": "mass_fresh", "task": TASK, "object": OBJECT, "root_seed": int(root), "root_id": f"mass_fresh_t2_root{root}_{band.lower()}", "friction_band": "FIXED_0.5", "friction": 0.5, "tuple_id": f"mass_fresh_t2_root{root}_{band.lower()}"}
                os.environ["MASS_QUERY_CURRENT_KG"] = str(mass)
                root_state, root_hash = base.capture_root(env, p6r1, p6, p4, TASK, int(root), 0.5)
                root_step = e5.episode_step(env)
                e5.prepare_fresh_start(env, p6r1, p6, p4, TASK, item, root_state, root_hash, root_step)
                probe_steps, probe_rec = e5.utility.run_probe_no_reset(env, p4, root_state, int(root), 0.5, dt, item["tuple_id"], torch)
                post_state = copy.deepcopy(env.scene.get_state(is_relative=True)); post_hash = base.stable_hash(post_state); post_step = e5.episode_step(env)
                reached, valid = base.query_quality(e5.utility.probe_dicts(probe_steps), probe_rec)
                if fixed4n:
                    estimated = ""
                    est_band = ""
                    prior_mass = ""
                    gt_force = ""
                    policy_forces = {"QUERY_STATE_FIXED_FORCE": float(args.query_state_fixed_force)}
                    queries.append({**item, "mass_kg": mass, "query_state_reach": reached, "query_valid": valid, "query_rows": len(probe_steps), "root_state_hash": root_hash, "post_query_state_hash": post_hash, "root_episode_step": root_step, "post_query_episode_step": post_step, "query_record": json.dumps(probe_rec, sort_keys=True, default=str)})
                else:
                    qvec = modeling.physical_vector({"query_record": json.dumps(probe_rec), "context_id": item["tuple_id"]})
                    estimated = float(np.mean([float(m(qvec).ravel()[0]) for m in id_models])); est_band = modeling.nearest_band(estimated)
                    # Mass-only Direct predicts one row per (mass, force) pair.
                    # Passing a singleton mass array silently truncated zip(masses,
                    # forces) to the first force and made every policy choose 0.5 N.
                    direct = np.mean([m(np.full(len(FORCES), estimated), np.asarray(FORCES)) for m in direct_models], axis=0).ravel()
                    prior_mass = float(py.mean()); prior_prob = np.mean([m(np.full(len(FORCES), prior_mass), np.asarray(FORCES)) for m in direct_models], axis=0).ravel(); gt_prob = np.mean([m(np.full(len(FORCES), mass), np.asarray(FORCES)) for m in direct_models], axis=0).ravel()
                    choose = lambda probs: max(((utility(float(p), f), -f, f) for p, f in zip(probs, FORCES)))[2]
                    active_force, prior_force, gt_force = choose(direct), choose(prior_prob), choose(gt_prob)
                    policy_forces = {"ACTIVEFORCING_MASS": active_force, "FROZEN_PI0_NATIVE_DEFAULT": None, "FIXED_MAX": 4.0, "TRUE_NOQUERY_PRIOR": prior_force, "GT_MASS": gt_force, "QUERY_STATE_FIXED_FORCE": args.query_state_fixed_force}
                    queries.append({**item, "mass_kg": mass, "query_state_reach": reached, "query_valid": valid, "query_rows": len(probe_steps), "pred_mass_kg": estimated, "estimated_band": est_band, "root_state_hash": root_hash, "post_query_state_hash": post_hash, "root_episode_step": root_step, "post_query_episode_step": post_step, "query_record": json.dumps(probe_rec, sort_keys=True, default=str), "active_scores": json.dumps([{ "force_N": f, "p_success": float(p), "utility": utility(float(p), f)} for f, p in zip(FORCES, direct)]), "prior_scores": json.dumps([{ "force_N": f, "p_success": float(p), "utility": utility(float(p), f)} for f, p in zip(FORCES, prior_prob)]), "gt_scores": json.dumps([{ "force_N": f, "p_success": float(p), "utility": utility(float(p), f)} for f, p in zip(FORCES, gt_prob)])})
                if args.query_only:
                    continue
                for policy in run_policies:
                    if policy == "FROZEN_PI0_NATIVE_DEFAULT":
                        row = e5.native_arm(p6r1, native_exec, env, p6, p4, client, TASK, item, root_state, root_hash, root_step, out / "pi0_native")
                    else:
                        force = policy_forces[policy]; start_state, start_hash, start_source, start_step = (post_state, post_hash, "post_query_state", post_step) if policy in ("ACTIVEFORCING_MASS", "GT_MASS", "QUERY_STATE_FIXED_FORCE") else (root_state, root_hash, "fresh_reset_root", root_step)
                        if policy in ("ACTIVEFORCING_MASS", "QUERY_STATE_FIXED_FORCE"):
                            row = e5.selected_arm_from_current_post_query(p6r1, env, p6, p4, client, TASK, item, root_hash, post_hash, force, policy, out, {"query_state_reached": reached, "query_valid": valid, "query_duration_s": probe_rec.get("probe_duration_s", ""), "query_rows": len(probe_steps), "mu_hat": estimated, "sigma_mu": "", "prior_mu": prior_mass, "query_path": "", "decision_path": ""}, post_step)
                        elif policy == "GT_MASS":
                            row = e5.selected_arm(p6r1, env, p6, p4, client, TASK, item, post_state, post_hash, force, policy, out, {"query_state_reached": reached, "query_valid": valid, "query_duration_s": probe_rec.get("probe_duration_s", ""), "query_rows": len(probe_steps), "mu_hat": mass, "sigma_mu": 0.0, "prior_mu": prior_mass, "query_path": "", "decision_path": ""}, "post_query_state", post_step)
                        else:
                            row = e5.selected_arm(p6r1, env, p6, p4, client, TASK, item, root_state, root_hash, force, policy, out, {"query_state_reached": 0, "query_valid": 0, "query_rows": 0, "mu_hat": "", "sigma_mu": "", "prior_mu": prior_mass, "query_path": "", "decision_path": ""}, "fresh_reset_root", root_step)
                        row["query_state_reached"] = reached if policy in ("ACTIVEFORCING_MASS", "GT_MASS", "QUERY_STATE_FIXED_FORCE") else 0; row["query_valid"] = valid if policy in ("ACTIVEFORCING_MASS", "GT_MASS", "QUERY_STATE_FIXED_FORCE") else 0; row["selected_force_N"] = force
                    row.update({"context_id": item["tuple_id"], "root": int(root), "gt_mass_kg": mass, "mass_kg": mass, "mass_estimate_kg": estimated, "estimated_band": est_band, "fixed_force_N": force, "query_state_hash": post_hash, "e2e_success": int(float(row.get("full_task_success_y", 0)) > 0 and int(row.get("query_valid", 1) or 1) == 1) if policy in ("ACTIVEFORCING_MASS", "GT_MASS", "QUERY_STATE_FIXED_FORCE") else int(float(row.get("full_task_success_y", 0)) > 0), "under_force": "" if fixed4n else int(policy not in ("FROZEN_PI0_NATIVE_DEFAULT",) and float(row.get("selected_force_N", 4.0) or 4.0) < gt_force), "failure_stage": row.get("failure_stage", "")})
                    rows.append(row)
                    scores = None if fixed4n else (direct if policy == "ACTIVEFORCING_MASS" else prior_prob if policy == "TRUE_NOQUERY_PRIOR" else gt_prob if policy == "GT_MASS" else None)
                    decisions.append({"tuple_id": item["tuple_id"], "policy": policy, "selected_force_N": row.get("selected_force_N", ""), "pred_mass_kg": "" if fixed4n else estimated, "gt_mass_kg": mass, "query_valid": valid, "candidate_scores": "" if fixed4n else (json.dumps([{ "force_N": f, "p_success": float(p), "utility": utility(float(p), f)} for f, p in zip(FORCES, scores)]) if scores is not None else "")})
        env.close(); env = None
        protocol.update({"status": "COMPLETED", "rows": len(rows), "contexts": len(args.roots) * len(selected_bands), "fresh_roots": list(args.roots), "source_checkpoint_hash": e5.PI0_CHECKPOINT_HASH, "executor_transform_hashes": transform_hashes, "gpu": gpu})
    except Exception as exc:
        protocol.update({"status": "ERROR", "error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        write_json(out / "MASS_FRESH_E2E_PROTOCOL.json", protocol)
        write_csv(out / "MASS_FRESH_E2E_ROWS.csv", rows)
        write_csv(out / "MASS_FRESH_E2E_DECISIONS.csv", decisions)
        # Persist the raw probe records used by the identifier.  The original
        # runner retained these only in memory, which prevented retrospective
        # feature-level state-shift audits.
        write_json(out / "MASS_FRESH_E2E_QUERIES.json", queries)
        try:
            if env is not None: env.close()
        except Exception: pass
        try: app.close()
        except Exception: pass


if __name__ == "__main__":
    main()
