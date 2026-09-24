#!/usr/bin/env python3
"""DEPRECATED pre-handoff diagnostic; retained only for provenance.

This runner predates the post-grasp force-adaptation scope.  It starts from a
pre-probe state and changes force slots before a semantic VLA-to-force
handoff, so it must not be used for the formal ActiveForcing question.  It is
kept readable for provenance, but is blocked by default.  A replacement
post-grasp runner must start from a frozen-VLA-created bilateral handoff
snapshot and branch only ``F_des``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ANALYSIS = Path("/home/exouser/Tabero/analysis")
sys.path.insert(0, str(ANALYSIS))
import target_object_force_calibration_20260904 as cal  # noqa: E402

REPO = Path("/home/exouser/Tabero")
E3_ROOT = Path("/media/volume/newdata/exouser/activeforcing_e3/P1_SIMPLIFIED_COLLECTION_FORMAL_20260903")
SELECTOR = E3_ROOT / "P1_SIMPLIFIED_MATCHED_DIRECT/P1_SIMPLIFIED_MATCHED_HELDOUT_DECISIONS.csv"
OUT = Path(os.environ.get("AF_FIXEDMAX_OUT", str(REPO / "E3_E6_E7_LANES/ACTIVEFORCING_VS_FIXEDMAX_PHYSICAL_20260904")))
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"
TASK_SUITE = "libero_10"
TASK_ID = 5
OBJECT = "black_book_1"
TARGET = "desk_caddy_1"
ROOTS = [7800, 7801, 7802]
ACTIVE_METHOD = "FULLTASK_DIRECT"
ACTIVE_FORCE_DEFAULT = 1.0
MAX_FORCE = 8.0
DT = 1.0 / 20.0
REPLAY_STEPS_PER_CHUNK = 10
CONTACT_EPS = 0.15
FRICTION = 0.6
POST_GRASP_SCOPE_REQUIRED = True


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_selected() -> dict[int, float]:
    out: dict[int, float] = {}
    with SELECTOR.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("arm") != ACTIVE_METHOD:
                continue
            root = int(row["root_id"])
            if root in ROOTS:
                out[root] = float(row["selected_force_N"])
    missing = sorted(set(ROOTS) - set(out))
    if missing:
        raise RuntimeError(f"missing frozen selector decisions: {missing}")
    return out


def load_trace(root: int, source_force: int = 1) -> list[np.ndarray]:
    p = E3_ROOT / f"P1_SIMPLIFIED_ROOT{root}_F{source_force}N/raw_policy/b5_t5_mu0.6_exp000_action_chunks.npz"
    if not p.is_file():
        raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=False)
    keys = sorted(k for k in z.files if k.startswith("chunk_"))
    if not keys:
        raise RuntimeError(f"no action chunks in {p}")
    return [np.asarray(z[k], dtype=np.float32)[i]
            for k in keys for i in range(min(REPLAY_STEPS_PER_CHUNK, len(z[k])))]


def load_preprobe(root: int) -> tuple[Any, dict[str, Any], str]:
    d = E3_ROOT / f"P1_SIMPLIFIED_ROOT{root}_F1N"
    meta = json.loads((d / "preprobe_state_exp000.json").read_text(encoding="utf-8"))
    state_path = d / "preprobe_state_exp000.pt"
    if int(meta["root_id"]) != root or int(meta["episode_index"]) not in (6, 7, 8):
        raise RuntimeError(f"unexpected preprobe metadata for root {root}: {meta}")
    import torch
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    return state, meta, cal.state_hash(state)


def move_state_to_device(value: Any, device: Any) -> Any:
    """Move a saved Isaac scene state recursively without changing its values."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {k: move_state_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_state_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_state_to_device(v, device) for v in value)
    return value


def phase_for(row: dict[str, Any], start_z: float, snapshot_eef: np.ndarray, prior: str) -> str:
    dz = float(row["object_height_m"]) - start_z
    eef = np.asarray(json.loads(row["eef_pos_w"]), dtype=float)
    moved = float(np.linalg.norm(eef[:2] - snapshot_eef[:2])) > 0.03
    if prior == "PRE_CONTACT":
        return "GRASP" if row["bilateral_object_contact"] else "PRE_CONTACT"
    if prior == "GRASP" and dz > 0.01:
        return "LIFT"
    if prior in ("GRASP", "LIFT") and dz > 0.03 and moved:
        return "TRANSPORT"
    return prior


def current_term(env: Any, name: str) -> bool:
    """Evaluate a registered termination function now, not its latched log."""
    manager = env.termination_manager
    idx = manager._term_name_to_term_idx[name]
    cfg = manager._term_cfgs[idx]
    return bool(cfg.func(env, **cfg.params)[0].item())


def apply_context_friction(env: Any, mu: float) -> dict[str, float]:
    """Reapply formal E3 object friction because scene snapshots omit materials."""
    view = env.scene[OBJECT].root_physx_view
    mats = view.get_material_properties().clone()
    mats[..., 0] = float(mu)
    mats[..., 1] = float(mu)
    torch = __import__("torch")
    # Isaac PhysX material API expects the body-index tensor on CPU even
    # though the material tensor itself is CUDA-resident.
    ids = torch.arange(mats.shape[0], dtype=torch.int32, device="cpu")
    view.set_material_properties(mats, ids)
    got = view.get_material_properties().detach().cpu().numpy().reshape(-1, 3)
    return {"static_mean": float(got[:, 0].mean()), "dynamic_mean": float(got[:, 1].mean())}


def finite_values(rows: list[dict[str, Any]], key: str, *, bilateral: bool = True) -> np.ndarray:
    vals = []
    for r in rows:
        if bilateral and not int(r["bilateral_object_contact"]):
            continue
        x = float(r[key])
        if np.isfinite(x):
            vals.append(x)
    return np.asarray(vals, dtype=float)


def metric(rows: list[dict[str, Any]], phase: str | None = None) -> dict[str, float]:
    use = [r for r in rows if phase is None or r["phase"] == phase]
    x = finite_values(use, "object_bilateral_squeeze_N")
    if not len(x):
        return {"mean_N": np.nan, "median_N": np.nan, "peak_N": np.nan,
                "integral_Ns": 0.0, "n": 0, "bilateral_fraction": 0.0}
    return {"mean_N": float(np.mean(x)), "median_N": float(np.median(x)),
            "peak_N": float(np.max(x)), "integral_Ns": float(np.sum(x) * DT),
            "n": int(len(x)),
            "bilateral_fraction": float(np.mean([int(r["bilateral_object_contact"]) for r in use])) if use else 0.0}


def run_branch(env: Any, torch: Any, root: int, method: str, force: float,
               state: Any, state_hash: str, trace: list[np.ndarray], trace_source_force: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = move_state_to_device(state, env.device)
    env.reset(seed=root)
    env.reset_to(state, torch.tensor([0], device=env.device), is_relative=True)
    # Match formal E3 ordering exactly: the saved state is restored first;
    # hidden object friction is then applied before the first action.
    material = apply_context_friction(env, FRICTION)
    restored_hash = cal.state_hash(env.scene.get_state(is_relative=True))
    obs = env.observation_manager.compute()
    start_z = float(cal.npv(env.scene[OBJECT].data.root_pos_w)[0, 2])
    reset_eef = cal.npv(obs["policy"]["eef_pose"])[0, :3].astype(float)
    rows: list[dict[str, Any]] = []
    phase = "PRE_CONTACT"
    success_latched = False
    lift_latched = False
    transport_latched = False
    place_latched = False
    for i, raw in enumerate(trace):
        action = cal.set_force(raw, force)
        obs, _, term, trunc, _ = env.step(torch.from_numpy(action).reshape(1, -1).to(env.device))
        r = cal.snapshot_row(env, obs, i + 1, force, phase)
        next_phase = phase_for(r, start_z, reset_eef, phase)
        r["phase"] = next_phase
        r.update({"root_id": root, "context_id": f"libero10_task5_root{root}",
                  "method": method, "selected_force_command_N": force,
                  "source_trace": str(E3_ROOT / f"P1_SIMPLIFIED_ROOT{root}_F{trace_source_force}N/raw_policy/b5_t5_mu0.6_exp000_action_chunks.npz"),
                  "preprobe_state_hash": state_hash, "restored_state_hash": restored_hash,
                  "reset_state_parity": int(restored_hash == state_hash),
                  "friction_static_applied": material["static_mean"], "friction_dynamic_applied": material["dynamic_mean"],
                  "source_action_index": i})
        rows.append(r)
        phase = next_phase
        dz = float(r["object_height_m"]) - start_z
        lift_latched = lift_latched or dz >= 0.03
        transport_latched = transport_latched or (lift_latched and phase == "TRANSPORT")
        # Match the formal collector: a terminal failure/timeout stops first;
        # the registered success function is then evaluated at the current
        # state.  get_term() is a latched episode log in IsaacLab, so it must
        # not be used as the live success predicate here.
        if bool(term[0].item()) or bool(trunc[0].item()):
            break
        official = current_term(env, "success")
        success_latched = success_latched or official
        place_latched = place_latched or official
        if success_latched:
            break
    all_contact = metric(rows)
    grasp = metric(rows, "GRASP")
    lift = metric(rows, "LIFT")
    transport = metric(rows, "TRANSPORT")
    phases = {p: sum(1 for r in rows if r["phase"] == p) for p in ("PRE_CONTACT", "GRASP", "LIFT", "TRANSPORT")}
    # Keep failure stage descriptive without redefining the evaluator.
    if success_latched:
        failure_stage = "none"
    elif not lift_latched:
        failure_stage = "lift"
    elif not transport_latched:
        failure_stage = "transport"
    else:
        failure_stage = "placement_or_evaluator"
    summary = {
        "context_id": f"libero10_task5_root{root}", "root_id": root,
        "task": "libero_10/task5", "object": OBJECT, "target": TARGET,
        "method": method, "selected_force_command_N": force,
        "friction_static_applied": material["static_mean"], "friction_dynamic_applied": material["dynamic_mean"],
        "full_task_success": int(success_latched), "lift_success": int(lift_latched),
        "transport_success_observed": int(transport_latched),
        "place_success_observed": int(place_latched), "failure_stage": failure_stage,
        "episode_steps": len(rows), "reset_state_parity": int(restored_hash == state_hash),
        "grasp_mean_physical_squeeze_N": grasp["mean_N"], "grasp_median_physical_squeeze_N": grasp["median_N"],
        "grasp_peak_physical_squeeze_N": grasp["peak_N"], "grasp_force_integral_Ns": grasp["integral_Ns"],
        "lift_mean_physical_squeeze_N": lift["mean_N"], "lift_median_physical_squeeze_N": lift["median_N"],
        "lift_peak_physical_squeeze_N": lift["peak_N"], "lift_force_integral_Ns": lift["integral_Ns"],
        "lift_bilateral_contact_fraction": lift["bilateral_fraction"],
        "transport_mean_physical_squeeze_N": transport["mean_N"], "transport_median_physical_squeeze_N": transport["median_N"],
        "transport_peak_physical_squeeze_N": transport["peak_N"], "transport_force_integral_Ns": transport["integral_Ns"],
        "transport_bilateral_contact_fraction": transport["bilateral_fraction"],
        "full_contact_conditioned_mean_physical_squeeze_N": all_contact["mean_N"],
        "full_contact_conditioned_median_physical_squeeze_N": all_contact["median_N"],
        "full_contact_conditioned_peak_physical_squeeze_N": all_contact["peak_N"],
        "force_time_integral_Ns": all_contact["integral_Ns"],
        "bilateral_contact_steps": all_contact["n"],
        "phase_steps_json": json.dumps(phases, sort_keys=True),
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def main() -> int:
    if POST_GRASP_SCOPE_REQUIRED and os.environ.get("ALLOW_LEGACY_PRE_HANDOFF_DIAGNOSTIC") != "1":
        raise RuntimeError(
            "Blocked: this legacy runner changes force before a semantic "
            "post-grasp handoff. Use a post-grasp snapshot runner; this "
            "diagnostic is not valid evidence for ActiveForcing."
        )
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {OUT}")
    selected = load_selected()
    states = {}
    traces = {}
    metadata = {}
    for root in ROOTS:
        states[root], metadata[root], state_hash = load_preprobe(root)
        metadata[root]["loaded_state_hash"] = state_hash
        traces[root] = {
            1: load_trace(root, 1),
            8: load_trace(root, 8),
        }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "PAIRING_INPUTS.json").write_text(json.dumps({
        "read_only": True, "formal_data_modified": False,
        "selector_artifact": str(SELECTOR), "selector_sha256": sha256(SELECTOR),
        "active_method": ACTIVE_METHOD, "active_selected_force_N": selected,
        "fixed_max_command_N": MAX_FORCE, "roots": ROOTS,
        "same_trace_source": "Default mode uses each root's formal F1 raw_policy action chunks; formal_force_trace mode uses the corresponding formal F1/F8 trace per method. In both modes the first 10 rows/chunk are used, exactly as validated E3 consumer.",
        "only_changed_action": "force slots 7:13; slots 9 and 12 set to force/2, all other action dimensions preserved",
        "formal_friction": FRICTION,
        "target_sensor": "contact_grasp_black_book_1.data.force_matrix_w",
        "physical_squeeze_definition": "2*min(abs(left finger target-object force), abs(right finger target-object force)); only bilateral target-object contact steps enter physical metrics",
        "dt_s": DT, "preprobe_metadata": metadata,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    os.environ.update({
        "HDF5_TRAJ_SOURCE_DIR": "/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_SOURCE_20260902_103300/assembled_hdf5",
        "LIBERO_CONFIG_DIR": str(REPO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(REPO / "benchmarks/datasets/libero/USD"),
        "TASK_SUITE": TASK_SUITE, "TASK_ID": str(TASK_ID),
        "ENABLE_CLOSED_LOOP_FORCE_CONTROLLER": "0",
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y",
    })
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    try:
        import gymnasium as gym
        import torch
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        setup_task_objects(TASK_SUITE, TASK_ID)
        cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 30.0
        env = gym.make(ENV_ID, cfg=cfg).unwrapped
        summaries: list[dict[str, Any]] = []
        telemetry: list[dict[str, Any]] = []
        for root in ROOTS:
            state, _, state_hash = load_preprobe(root)
            trace_mode = os.environ.get("AF_TRACE_MODE", "same_f1_trace")
            for method, force in (("ActiveForcing", selected[root]), ("Fixed-Max", MAX_FORCE)):
                trace_source_force = 1 if trace_mode == "same_f1_trace" else (1 if method == "ActiveForcing" else 8)
                trace = traces[root][trace_source_force]
                print(f"RUN root={root} method={method} force={force}", flush=True)
                summary, rows = run_branch(env, torch, root, method, force, state, state_hash, trace, trace_source_force)
                summary["trace_mode"] = trace_mode
                summary["trace_source_force_N"] = trace_source_force
                summaries.append(summary)
                telemetry.extend(rows)
                (OUT / f"root{root}_{method.replace('-', '').replace(' ', '_')}_timeline.json").write_text(
                    json.dumps(rows, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
                print(json.dumps(summary, sort_keys=True, default=str), flush=True)
        write_csv(OUT / "PAIRED_SUMMARY.csv", summaries)
        write_csv(OUT / "TARGET_OBJECT_FORCE_TIMESERIES.csv", telemetry)
        (OUT / "RUN_COMPLETE.json").write_text(json.dumps({
            "status": "COMPLETE", "n_contexts": len(ROOTS), "n_rollouts": len(summaries),
            "expected_rollouts": 2 * len(ROOTS), "trace_mode": os.environ.get("AF_TRACE_MODE", "same_f1_trace"), "summary": str((OUT / "PAIRED_SUMMARY.csv").resolve()),
            "telemetry": str((OUT / "TARGET_OBJECT_FORCE_TIMESERIES.csv").resolve()),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os._exit(0)
    finally:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
