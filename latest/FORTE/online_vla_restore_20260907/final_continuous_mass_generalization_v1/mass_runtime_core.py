"""Current multi-task probe/decision core with one MASS-specific intervention.

This is a minimal fork of the frozen current core.  Probe, feature, snapshot,
controller and callback ordering are unchanged.  The only new behavior wraps
the existing object-friction setter so that, after every reset, object mass is
set to ``plan['mass_kg']`` and the nominal inertia tensor is scaled by the same
ratio.  Mass/inertia are metadata only and never belief inputs.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import re
import sys
import traceback

import numpy as np

ROOT = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
P5 = TABERO / "analysis/p5s0c_paired_boundary_probe_value.py"
SHARED = ROOT / "analysis/results/current_runtime_recovery_v2_20260905"
for path in (SHARED, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

VERSION = "SHARED_CURRENT_MULTITASK_MASS_PROBE_DECISION_CORE_V1"
INTERVENTION = "TOTAL_OBJECT_MASS_WITH_PROPORTIONAL_INERTIA_V1"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(clean(value), handle, indent=2, allow_nan=False)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def validate_call(plan, job, runtime_manifest_sha256):
    if not isinstance(runtime_manifest_sha256, str) or re.fullmatch("[0-9a-f]{64}", runtime_manifest_sha256) is None:
        raise ValueError("Driver must pass SHA256 of frozen runtime manifest")
    required = {"task", "root", "id", "object", "target", "mu", "mass_kg"}
    if required - set(plan):
        raise ValueError("Incomplete explicit MASS context")
    if plan["task"] not in (0, 1, 5, 6):
        raise ValueError("Unsupported task")
    if not np.isfinite(plan["mu"]) or abs(float(plan["mu"]) - 0.5) > 1e-12:
        raise ValueError("MASS protocol requires fixed object friction 0.5")
    if not np.isfinite(plan["mass_kg"]) or float(plan["mass_kg"]) <= 0:
        raise ValueError("Invalid requested object mass")
    protected = ("STARTED.json", "RESULT.json", "ERROR.json", "RAW_PROBE.csv", "DECISION_STATE.pt")
    if any((Path(job) / name).exists() for name in protected):
        raise FileExistsError("Output already contains evidence")


def install_mass_intervention(p4, requested_mass_kg):
    """Set mass after the probe's unchanged friction application on each reset."""
    original = p4._apply_friction
    state = {}

    def apply(env, name, mu):
        applied = original(env, name, mu)
        import torch

        view = env.scene[name].root_physx_view
        if not state:
            state["nominal_mass"] = view.get_masses().clone()
            state["nominal_inertia"] = view.get_inertias().clone()
            state["nominal_total_mass_kg"] = float(state["nominal_mass"].sum().item())
        nominal = state["nominal_total_mass_kg"]
        if not np.isfinite(nominal) or nominal <= 0:
            raise RuntimeError("Invalid nominal object mass")
        ratio = float(requested_mass_kg) / nominal
        ids = torch.arange(state["nominal_mass"].shape[0], dtype=torch.int32)
        view.set_masses(state["nominal_mass"] * ratio, ids)
        view.set_inertias(state["nominal_inertia"] * ratio, ids)
        got_mass = view.get_masses().clone()
        got_inertia = view.get_inertias().clone()
        total = float(got_mass.sum().item())
        if abs(total - float(requested_mass_kg)) > 1e-7:
            raise RuntimeError(f"Mass readback mismatch: requested={requested_mass_kg} actual={total}")
        state.update(
            ratio=ratio,
            requested_mass_kg=float(requested_mass_kg),
            mass_readback=got_mass,
            inertia_readback=got_inertia,
        )
        return applied

    p4._apply_friction = apply
    return state


def invoke_decision(callback, admitted, env, p4, p5, plan, job, primitive_step):
    if callback is None:
        return {"called": False, "reason": "NO_CALLBACK"}
    if not admitted:
        return {"called": False, "reason": "PROBE_NOT_ADMITTED"}
    required = ("RAW_PROBE.csv", "CONTACT_PATCH_READBACK.json", "RAW_FEATURES.npy", "DECISION_STATE.pt", "DECISION_METADATA.json", "RESULT.json")
    if any(not (Path(job) / name).is_file() for name in required):
        raise RuntimeError("Decision callback requires completed immutable preaction evidence")
    result = json.loads((Path(job) / "RESULT.json").read_text())
    decision = json.loads((Path(job) / "DECISION_METADATA.json").read_text())
    if result.get("candidate_actions_executed") != 0 or decision.get("candidate_actions_already_executed") != 0:
        raise RuntimeError("Candidate action already occurred")
    if not result.get("probe_qualified"):
        raise RuntimeError("Saved probe admission failed")
    env.step = primitive_step
    return {"called": True, "result": callback(env, p4, p5, plan, Path(job))}


def run_probe(plan, job, measurement_hooks=None, on_decision=None, *, runtime_manifest_sha256=None):
    validate_call(plan, job, runtime_manifest_sha256)
    cid = plan["id"]
    job = Path(job)
    write(job / "STARTED.json", {"utc": now(), "plan": plan, "candidate_actions": 0, "runtime_manifest_sha256": runtime_manifest_sha256})
    import torch

    env = None
    records = []
    phase = "PROBE"
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        from activeforcing_preact_action_probe_pilot_20260905 import patch_data
        from activeforcing_probe_friction_contract import FrictionProbeBudget
        from current_contract_belief_features import ProbeEvidence
        from activeforcing_execution_snapshot import capture

        p5 = module("current_mass_task_sensors", P5)
        p5.OUT = job
        p5.P4_COLLECT = ROOT / "activeforcing_current_probe.py"
        p4 = p5.import_p4_probe(plan["task"])
        assert p4.OBJ_NAME == plan["object"]
        mass_state = install_mass_intervention(p4, plan["mass_kg"])
        setup_task_objects("libero_object", plan["task"])
        cfg = parse_env_cfg(p4.ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 45.0
        getattr(cfg.scene, "contact_grasp_" + p4.OBJ_NAME).max_contact_data_count_per_prim = 128
        if measurement_hooks is not None and hasattr(measurement_hooks, "configure"):
            receipt = measurement_hooks.configure(cfg, plan)
            if receipt is not None:
                write(job / "MEASUREMENT_CONFIG.json", receipt)
        write(job / "RUNTIME_CONFIG.json", {
            "task": plan["task"], "object": p4.OBJ_NAME, "target": plan["target"],
            "actions": str(cfg.actions), "seed": cfg.seed, "physics_dt": cfg.sim.dt,
            "decimation": cfg.decimation, "mass_intervention": INTERVENTION,
        })
        env = gym.make(p4.ENV_ID, cfg=cfg).unwrapped
        env.reset(seed=plan["root"])
        if measurement_hooks is not None and hasattr(measurement_hooks, "bind"):
            receipt = measurement_hooks.bind(env, plan)
            if receipt is not None:
                write(job / "MEASUREMENT_BINDINGS.json", receipt)
        original = env.step

        def step(action):
            if phase != "PROBE":
                raise RuntimeError("Candidate execution forbidden in probe-only stage")
            result = original(action)
            obj = env.scene[p4.OBJ_NAME]
            records.append({
                "step": len(records) + 1,
                "action": clean(action),
                "policy_local_normal_projection": clean(result[0]["policy"]["gripper_net_force"]),
                "eef_pose": clean(result[0]["policy"]["eef_pose"]),
                "finger_joints": clean(env.scene["robot"].data.joint_pos[0, -2:]),
                "aperture_m": float(env.scene["robot"].data.joint_pos[0, -2:].sum()),
                "object_position": clean(obj.data.root_pos_w),
                "object_velocity": clean(obj.data.root_lin_vel_w),
                "object_quaternion": clean(obj.data.root_quat_w),
                "target_object_force": p5.target_object_force_snapshot(env, p4, p4.OBJ_NAME),
                "patches": clean(measurement_hooks.patches(env, plan, float(cfg.sim.dt)) if measurement_hooks is not None and hasattr(measurement_hooks, "patches") else patch_data(env.scene["contact_grasp_" + p4.OBJ_NAME], float(cfg.sim.dt))),
                "controller_debug": clean(p4._dbg(env)),
            })
            return result

        env.step = step
        from geometry_grasp_initializer import install

        install(p4, plan["task"], job)
        rows, rec = p4.run_probe_episode(
            env, seed_idx=plan["root"], mu=plan["mu"], trial_id=cid, dt=0.05,
            termination_signal=FrictionProbeBudget(records, p4._quat_apply_np),
        )
        phase = "DECISION_STATE"
        raw = [vars(row) for row in rows]
        with (job / "RAW_PROBE.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
            writer.writeheader()
            writer.writerows(raw)
        write(job / "CONTACT_PATCH_READBACK.json", records)
        evidence = ProbeEvidence()
        x = evidence.rows(raw, records)
        np.testing.assert_array_equal(x, evidence.load(job))
        np.save(job / "RAW_FEATURES.npy", x)
        obj = env.scene[p4.OBJ_NAME]
        target = env.scene[plan["target"]]
        materials = obj.root_physx_view.get_material_properties().cpu().numpy().reshape(-1, 3)
        mass_readback = obj.root_physx_view.get_masses().cpu().numpy()
        inertia_readback = obj.root_physx_view.get_inertias().cpu().numpy()
        if not np.allclose(materials[:, :2], plan["mu"], rtol=0, atol=1e-6):
            raise RuntimeError("Fixed friction readback mismatch")
        if abs(float(mass_readback.sum()) - float(plan["mass_kg"])) > 1e-7:
            raise RuntimeError("Final mass readback mismatch")
        write(job / "MASS_INTERVENTION_READBACK.json", {
            "contract": INTERVENTION,
            "requested_total_mass_kg": plan["mass_kg"],
            "actual_mass_kg": mass_readback,
            "actual_total_mass_kg": float(mass_readback.sum()),
            "nominal_total_mass_kg": mass_state["nominal_total_mass_kg"],
            "mass_ratio": mass_state["ratio"],
            "nominal_inertia": mass_state["nominal_inertia"],
            "actual_inertia": inertia_readback,
            "inertia_scaled_by_mass_ratio": bool(np.allclose(inertia_readback, clean(mass_state["nominal_inertia"] * mass_state["ratio"]), rtol=0, atol=1e-8)),
            "object_static_dynamic_friction": materials[:, :2],
            "geometry_or_appearance_modified": False,
        })
        reasons = [key for key in ("probe_failure", "contact_lost_probe", "dropped", "major_disturbance") if rec.get(key, 0)]
        from current_contract_physical_belief import verify_decision_prefix
        try:
            verify_decision_prefix(raw)
        except ValueError as exc:
            reasons.append(str(exc))
        write(job / "DECISION_METADATA.json", {
            "plan": plan, "step": len(raw), "object_pose": clean(obj.data.root_state_w[0, :7]),
            "target_pose": clean(target.data.root_state_w[0, :7]), "eef_pose": records[-1]["eef_pose"],
            "last_gripper_command": records[-1]["action"][0][6], "aperture_feature_m": records[-1]["aperture_m"],
            "finger_joints": records[-1]["finger_joints"], "contact": records[-1]["target_object_force"],
            "material_properties": materials.tolist(), "object_mass_kg_analysis_only": mass_readback.tolist(),
            "object_inertia_analysis_only": inertia_readback.tolist(), "candidate_actions_already_executed": 0,
            "snapshot": str(job / "DECISION_STATE.pt"), "mass_intervention": INTERVENTION,
        })
        state = capture(env)
        state.update(candidate_actions_already_executed=0, observation_step=len(raw), context=plan)
        torch.save(state, job / "DECISION_STATE.pt")
        write(job / "RESULT.json", {
            "runtime_manifest_sha256": runtime_manifest_sha256, "core_source_sha256": sha(Path(__file__)),
            "plan": plan, "record": rec, "steps": len(raw),
            "outward_steps": sum(row["probe_phase"] == "probe_out" for row in raw),
            "probe_qualified": not reasons, "exclusions": reasons, "feature_dim": 58,
            "train_runtime_feature_parity": True, "max_feature_diff": 0.0, "candidate_actions_executed": 0,
            "task_branches": 0, "full_task_labels": 0, "actual_material": materials.tolist(),
            "actual_total_mass_kg": float(mass_readback.sum()), "mass_intervention": INTERVENTION,
            "source_hashes": {name: sha(job / name) for name in ("RAW_PROBE.csv", "CONTACT_PATCH_READBACK.json", "RAW_FEATURES.npy", "DECISION_STATE.pt", "MASS_INTERVENTION_READBACK.json")},
        })
        callback = invoke_decision(on_decision, not reasons, env, p4, p5, plan, job, original)
        if on_decision is not None:
            write(job / "DECISION_CALLBACK_RESULT.json", callback)
        return 0
    except Exception as exc:
        write(job / "ERROR.json", {"error": repr(exc), "traceback": traceback.format_exc(), "observed_steps": len(records)})
        if records:
            write(job / "PARTIAL_READBACK.json", records)
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            env.close()
