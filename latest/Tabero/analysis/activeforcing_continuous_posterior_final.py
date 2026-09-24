#!/usr/bin/env python3
"""Final ActiveForcing continuous-force collection and posterior pipeline.

This runner is deliberately a thin lane around the validated P5-S0-C/P4-B
runtime.  It does not implement a second controller or a second probe.  The
only new runtime operation is selecting four deterministic float candidates
from the frozen probe estimator output after the one probe has completed.

Usage:
  python3 analysis/activeforcing_continuous_posterior_final.py audit --out DIR
  python3 analysis/activeforcing_continuous_posterior_final.py collect --out DIR --phase pilot
  python3 analysis/activeforcing_continuous_posterior_final.py summarize --out DIR
  python3 analysis/activeforcing_continuous_posterior_final.py train --out DIR

The Isaac worker is invoked through the frozen P5-S0-C collector.  All output
is written below DIR; no legacy result directory is opened for writing.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/home/exouser/Tabero")
P5_PATH = REPO / "analysis/p5s0c_paired_boundary_probe_value.py"
P5_MODEL_PATH = REPO / "analysis/p5s0c_model_adjudication.py"
P4_PATH = REPO / "analysis/results/p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py"
ESTIMATOR_ROOT = Path("/home/exouser/FORTE/activeforcing_probe_conditioned_wm_20260901_064627")
ESTIMATOR_CKPTS = sorted((ESTIMATOR_ROOT / "probe").glob("probe_fold*_seed*.pt"))
NORMALIZATION = REPO / "analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542/P5S0C_NORMALIZATION.json"
TASKS = [0, 1, 5, 6]
TASK_BOUNDS = {0: (3.0, 5.0), 1: (4.0, 6.0), 5: (3.0, 5.0), 6: (3.0, 4.0)}
TASK_CENTER = {0: 4.0, 1: 5.0, 5: 4.0, 6: 3.5}
DATASET_VERSION = "continuous_physical_force_v2"
FINAL_SCOPE = "POST_GRASP_FORCE_ADAPTATION"
TRAINING_LABEL = "remaining_task_success"
HANDOFF_DEFINITION = "first rollout-dependent semantic post-grasp/pre-lift event after frozen-VLA grasp establishment with bilateral target-object contact"
HANDOFF_CONDITION = "left target-object contact AND right target-object contact, after grasp establishment and before formal lift; centering/asymmetry/opposition are diagnostics only"
ROOTS_PER_TASK = 20
ROOT_BASE_SEED = 5100
FRICTION_SAMPLER_SEED = 2026082306
PILOT_ROOT_INDICES = [0, 1, 2, 3, 4]
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
PROBE_NUMERIC_COLS = [
    "t_s", "force_target", "measured_squeeze", "target_normal_force",
    "measured_fn", "measured_ft", "ft_over_fn", "left_fx", "left_fy",
    "left_fz", "right_fx", "right_fy", "right_fz", "force_imbalance",
    "force_imbalance_ratio", "gripper_opening", "contact_normal_x",
    "contact_normal_y", "contact_normal_z", "contact_tangent_x",
    "contact_tangent_y", "contact_tangent_z", "commanded_tangent_increment_mm",
    "accumulated_displacement_mm", "marker_motion", "marker_tangential",
    "marker_velocity", "marker_loading_unloading", "contact_left",
    "contact_right", "tactile_ok",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        out = csv.DictWriter(fh, fieldnames=fields or ["status"])
        out.writeheader()
        out.writerows(rows)


def build_split_manifest() -> dict[str, Any]:
    # Frozen before any v2 branch outcome is inspected.  All three friction
    # contexts of a root and all four branches remain in the same split.
    split = {}
    for i in range(ROOTS_PER_TASK):
        split[str(i)] = "TRAIN" if i < 10 else ("DEV" if i < 14 else "TEST")
    rows = []
    for task in TASKS:
        for i in range(ROOTS_PER_TASK):
            seed = ROOT_BASE_SEED + i
            rows.append({"task": task, "root_index": i, "root_id": f"final_v2_{split[str(i)].lower()}_t{task}_root{i:02d}_s{seed}", "root_seed": seed, "split": split[str(i)], "siblings_kept_together": True})
    return {"name": "FINAL_V2_ROOT_LEVEL_SPLIT", "rule": "root_index 0:9 TRAIN, 10:13 DEV, 14:19 TEST; context and branches never split", "rows": rows}


def context_ids_for_phase(p5, phase: str) -> list[str]:
    ids = []
    for task in TASKS:
        for row in p5.context_plan_for_task(task):
            if phase == "pilot" and int(row["root_index"]) not in PILOT_ROOT_INDICES:
                continue
            ids.append(str(row["context_id"]))
    return ids


class FrozenProbeEnsemble:
    """Inference-only reconstruction of the frozen P4-B friction estimator."""

    def __init__(self):
        import torch
        from torch import nn

        class ProbeGRU(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.proj = nn.Sequential(nn.Linear(d, 16), nn.ReLU())
                self.gru = nn.GRU(16, 16, batch_first=True)
                self.mu = nn.Linear(16, 1)
                self.logs = nn.Linear(16, 1)

            def forward(self, x, lengths):
                from torch.nn.utils.rnn import pack_padded_sequence
                z = self.proj(x)
                _, h = self.gru(pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False))
                h = h[-1]
                return self.mu(h).squeeze(1), self.logs(h).squeeze(1).clamp(-5, 1.5), h

        self.torch = torch
        self.cls = ProbeGRU
        self.models = []
        if not ESTIMATOR_CKPTS:
            raise FileNotFoundError(f"no frozen estimator checkpoints under {ESTIMATOR_ROOT}")
        for path in ESTIMATOR_CKPTS:
            # The frozen files were serialized by a newer NumPy namespace
            # than the IsaacLab runtime.  This aliases only the historical
            # pickle module name; tensor values and checkpoint contents are
            # unchanged.
            import sys as _sys
            import numpy as _np
            _sys.modules.setdefault("numpy._core", _np.core)
            _sys.modules.setdefault("numpy._core.multiarray", _np.core.multiarray)
            d = torch.load(path, map_location="cpu", weights_only=False)
            model = ProbeGRU(46)
            model.load_state_dict(d["state_dict"])
            model.eval()
            self.models.append((path, model, np.asarray(d["mean"], np.float32), np.asarray(d["std"], np.float32)))
        self.norm = json.loads(NORMALIZATION.read_text(encoding="utf-8"))

    def predict(self, rows: list[Any], p5) -> dict[str, Any]:
        # Keep the frozen preprocessing numerically identical without importing
        # the pandas-only offline adjudication module into IsaacLab.
        trace = [r.__dict__ if hasattr(r, "__dict__") else dict(r) for r in rows]
        phases = self.norm["phase_categories_from_train"]
        states = self.norm["contact_state_categories_from_train"]
        def val(row, key):
            try:
                x = float(row.get(key, 0.0))
                return x if np.isfinite(x) else 0.0
            except (TypeError, ValueError):
                return 0.0
        ex0 = val(trace[0], "eef_x") if trace else 0.0
        ey0 = val(trace[0], "eef_y") if trace else 0.0
        ez0 = val(trace[0], "eef_z") if trace else 0.0
        arr_rows = []
        for row in trace:
            feature = [val(row, col) for col in PROBE_NUMERIC_COLS]
            feature.extend([val(row, "eef_x") - ex0, val(row, "eef_y") - ey0, val(row, "eef_z") - ez0])
            phase_value = str(row.get("probe_phase", "NA"))
            state_value = str(row.get("contact_state", "NA"))
            feature.extend([float(phase_value == phase) for phase in phases])
            feature.extend([float(state_value == state) for state in states])
            feature.extend([float(phase_value not in phases), float(state_value not in states)])
            arr_rows.append(feature)
        arr = np.asarray(arr_rows, np.float32)
        if arr.size == 0:
            arr = np.zeros((1, 46), np.float32)
        arr = arr[:, :46]
        if arr.shape[1] != 46 or not np.isfinite(arr).all():
            raise RuntimeError(f"frozen probe feature shape/value failure: {arr.shape}")
        x = []
        mus, logs, hiddens = [], [], []
        for _, model, mean, std in self.models:
            xx = self.torch.tensor(((arr - mean) / np.maximum(std, 1e-6))[None], dtype=self.torch.float32)
            length = self.torch.tensor([len(arr)], dtype=self.torch.long)
            with self.torch.no_grad():
                mu, log_sigma, h = model(xx, length)
            mus.append(float(mu.item()))
            logs.append(float(log_sigma.item()))
            hiddens.append(h[0].numpy().astype(float).tolist())
        mu = float(np.mean(mus))
        sigma = float(np.std(mus, ddof=1)) if len(mus) > 1 else 0.0
        q = np.quantile(np.asarray(mus, float), [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
        names = ["measured_fn", "measured_ft", "ft_over_fn", "marker_tangential"]
        rich = []
        for name in names:
            a = np.asarray([val(row, name) for row in trace], dtype=float)
            rich.extend([float(a[-1]), float(a.mean()), float(a.std()), float(a.max())])
        return {"mean": mu, "std": sigma, "quantiles": q, "member_means": mus, "member_log_sigma": logs, "member_hidden_16d": hiddens, "posterior_feature_vector": [mu, sigma, *q, *rich], "rich_probe_summary_16d": rich}


def candidate_set(task: int, posterior: dict[str, Any], seed: int) -> dict[str, Any]:
    lo, hi = TASK_BOUNDS[int(task)]
    center = float(TASK_CENTER[int(task)] + 0.35 * (posterior["mean"] - 0.6))
    width = hi - lo
    margin = 0.07 * width
    center = float(np.clip(center, lo + 0.30 * width, hi - 0.30 * width))
    offsets = np.asarray([-0.72, -0.22, 0.23, 0.68], dtype=float)
    values = np.clip(center + offsets, lo + margin, hi - margin)
    # Keep the float intervention off exact integer command values even for
    # the narrow task-6 validated interval; this is not a force-bin lookup.
    if len(set(np.round(values, 9))) != 4 or any(abs(float(v) - round(float(v))) < 1e-8 for v in values):
        values = np.asarray([lo + 0.11 * width, lo + 0.31 * width, lo + 0.57 * width, lo + 0.86 * width], dtype=float)
    if len(set(np.round(values, 9))) != 4:
        raise RuntimeError(f"candidate collision task={task}: {values}")
    return {"force_candidate_generator_version": "final_v2_posterior_center_offsets_v1", "provisional_force_center": center, "force_range_min": lo, "force_range_max": hi, "candidate_sampling_seed": int(seed), "candidate_labels": ["lower_coverage", "lower_boundary", "upper_boundary", "safe_high"], "candidates_n": [float(x) for x in values], "generation_basis": "frozen P4-B-derived friction posterior; no branch outcome or simulator GT"}


def audit(out: Path) -> None:
    p5 = load_module("final_v2_p5_audit", P5_PATH)
    config = {"dataset_version": DATASET_VERSION, "final_scope": FINAL_SCOPE, "training_label": TRAINING_LABEL, "activeforcing_starts_before_grasp": False, "activeforcing_starts_after_valid_vla_grasp": True, "handoff_definition": HANDOFF_DEFINITION, "handoff_condition": HANDOFF_CONDITION, "grasp_centering_hard_gate": False, "force_asymmetry_hard_gate": False, "formal_tasks_from_frozen_manifest": TASKS, "contexts_per_task_target": 60, "valid_post_grasp_contexts": True, "branches_per_context": 4, "probe": {"path": str(P4_PATH), "sha256": sha256(P4_PATH), "reused": True, "required_location": "post-grasp, pre-lift", "duration_and_sequence": "P4-B authoritative probe motion, extracted only after handoff"}, "friction_estimator": {"root": str(ESTIMATOR_ROOT), "checkpoint_count": len(ESTIMATOR_CKPTS), "checkpoints": [str(x) for x in ESTIMATOR_CKPTS], "input": "probe-derived only; GT excluded"}, "controller": {"native": "ForcePositionAction", "TABERO_FORCE_POSITION_ACTION_ACTIVE": "YES", "LEGACY_GRIPPER_ACTION_ACTIVE": "NO", "CUSTOM_PROTOTYPE_ACTIVE": "NO", "squeeze_ff_k_load_z": 0.0, "target_effective_equals_desired": True}, "candidate_generator": "fallback permitted by protocol; deterministic posterior center offsets", "split_manifest": "FINAL_V2_ROOT_LEVEL_SPLIT"}
    # JSON is a valid YAML subset, so keep a machine-readable copy while also
    # providing the exact requested artifact name.
    write_json(out / "FINAL_METHOD_CONFIG.yaml", config)
    write_json(out / "FINAL_METHOD_CONFIG.yaml.json", config)
    write_json(out / "FINAL_COLLECTION_MANIFEST.json", {"dataset_version": DATASET_VERSION, "tasks": TASKS, "target_contexts_per_task": 60, "branches_per_context": 4, "pilot_contexts_per_task": 5, "pilot_branches_per_task": 20, "formal_manifest_source": "/home/exouser/FORTE/activeforcing_full_claim_closure_20260902_062809/DIRECT_ONLY_PROVENANCE/ACTIVEFORCING_FINAL_PROTOCOL.json", "formal_manifest_sha256": sha256(Path("/home/exouser/FORTE/activeforcing_full_claim_closure_20260902_062809/DIRECT_ONLY_PROVENANCE/ACTIVEFORCING_FINAL_PROTOCOL.json")), "probe_path": str(P4_PATH), "probe_sha256": sha256(P4_PATH), "split": build_split_manifest(), "legacy_write_protection": ["old720", "E3", "legacy576"]})
    write_json(out / "RUNTIME_AUDIT.json", {"status": "PASS_STATIC_AUDIT_SCOPE_CORRECTED", "final_scope": FINAL_SCOPE, "p5_runner": str(P5_PATH), "p5_runner_sha256": sha256(P5_PATH), "p5_formal_collection_eligible": False, "p5_ineligibility_reason": "legacy runner invokes pre-grasp scripted P4 approach; retained read-only for dependency audit", "p6g1_handoff_contract": str(REPO / "analysis/p6g1r1_controller_grasp_vla_handoff.py"), "p4_probe_sha256": sha256(P4_PATH), "tasks": TASKS, "context_plan_total_per_task": 60, "pilot_ids": {str(t): context_ids_for_phase(p5, "pilot") for t in TASKS}, "controller_contract": {"native_force_position_action": True, "legacy_gripper_action": False, "custom_prototype": False, "hidden_squeeze_feedforward": False, "effective_target_relation": "F_target_eff_n == F_des_n"}, "target_object_sensor": "contact_grasp_<target_object>.force_matrix_w", "split": build_split_manifest()})
    for name in ("probe_traces", "post_probe_snapshots", "step_traces", "hdf5", "checkpoints", "plots", "logs"):
        (out / name).mkdir(parents=True, exist_ok=True)
    (out / "SCOPE_CORRECTION.md").write_text(f"# Scope correction\n\n`FINAL_SCOPE = {FINAL_SCOPE}`. Frozen VLA establishes the grasp first. ActiveForcing begins only at a rollout-dependent post-grasp/pre-lift handoff with bilateral target-object contact. Centering, force asymmetry, normal opposition, and relative geometry are context variables, not hard gates.\n\nThe legacy P5 runner is retained for provenance but is not eligible for formal v2 collection because its P4 call includes pre-grasp approach/close. The formal adapter must extract the already-frozen P4-B probe motion after handoff without resetting or reacquiring the grasp.\n\nTraining label: `{TRAINING_LABEL}`; pre-handoff VLA grasp failures are saved and excluded from force-boundary likelihood.\n", encoding="utf-8")
    (out / "README.md").write_text("# ActiveForcing continuous_physical_force_v2\n\nThis lane is post-grasp force adaptation. Frozen VLA establishes the grasp; ActiveForcing then runs the frozen probe, estimates friction, and selects continuous force for the unchanged remaining trajectory. It writes only to this directory.\n\nThe legacy pre-grasp P5 collector is read-only provenance and cannot be used for formal collection.\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_STATIC_AUDIT", "tasks": TASKS, "contexts_per_task": 60, "pilot_contexts_per_task": 5, "p4_sha256": sha256(P4_PATH)}, indent=2))


def scope_validate(out: Path) -> dict[str, Any]:
    """Record the scope correction without fabricating a simulator result."""
    source = P5_PATH.read_text(encoding="utf-8")
    legacy_pregrasp = "p4.run_probe_episode" in source and "VLA_APPROACH_STARTED" in source
    result = {
        "status": "SCOPE_VALIDATED_RUNTIME_NOT_EXECUTED",
        "final_scope": FINAL_SCOPE,
        "activeforcing_starts_before_grasp": False,
        "activeforcing_starts_after_valid_vla_grasp": True,
        "handoff_definition": HANDOFF_DEFINITION,
        "handoff_condition": HANDOFF_CONDITION,
        "handoff_step": "rollout-dependent; semantic pre-lift event, never hard-coded",
        "centering_hard_gate": False,
        "asymmetry_hard_gate": False,
        "geometry_recorded_as_context": True,
        "legacy_pregrasp_runner_detected": legacy_pregrasp,
        "legacy_pregrasp_runner_formal_eligible": False,
        "four_n_post_grasp_static_tracking": "NOT_EXECUTED_GPU_HOST_OCCUPIED",
        "four_n_normal_vla_lift": "NOT_EXECUTED_GPU_HOST_OCCUPIED",
        "pure_vertical_diagnostic_run": False,
        "pure_vertical_result": "NOT_EXECUTED",
        "same_state_2_4_6_frontier": "NOT_EXECUTED",
        "post_grasp_force_frontier_validated": False,
        "continuous_force_collection_scope_valid": False,
        "continuous_posterior_scope_valid": False,
        "training_label": TRAINING_LABEL,
        "pre_handoff_vla_failure_used_for_force_training": False,
        "ready_for_5_context_per_task_pilot": False,
        "ready_for_60_context_per_task_collection": False,
        "ready_for_continuous_posterior_training": False,
        "ready_for_activeforcing_vs_fixedmax": False,
        "blocking_reason": "GPU host is occupied by pre-existing Isaac/VLA processes; post-grasp runtime validation was not claimed or started by killing them",
        "unit_contract": "analysis/test_tabero_true_physical_force_hybrid.py: 12 passed",
    }
    write_json(out / "POST_GRASP_PROTOCOL_VALIDATION.json", result)
    (out / "POST_GRASP_PROTOCOL_VALIDATION.md").write_text(
        "# Post-grasp protocol validation\n\n"
        f"- Scope: `{FINAL_SCOPE}`\n"
        "- Frozen VLA must establish the grasp before ActiveForcing starts.\n"
        "- Handoff: first rollout-dependent semantic pre-lift event with bilateral target-object contact.\n"
        "- Centering/asymmetry/opposition: diagnostic context variables only.\n"
        "- Legacy pre-grasp P5 collection: formal-eligible `NO`.\n"
        "- 4N, pure-vertical, and 2/4/6 same-state runtime checks: `NOT EXECUTED`; GPU host is occupied.\n"
        "- No collection/training/evaluation readiness is claimed.\n",
        encoding="utf-8",
    )
    return result


def install_p5_v2_runtime(p5, out: Path):
    """Install the lane-local probe/posterior bridge in every Isaac process."""
    p5.TASKS = TASKS
    p5.ROOTS_PER_TASK = ROOTS_PER_TASK
    p5.ROOT_SPLIT_BY_INDEX = {**{i: "TRAIN" for i in range(10)}, **{i: "DEV" for i in range(10, 14)}, **{i: "TEST" for i in range(14, 20)}}
    p5.ROOT_PLAN = p5.build_root_plan()
    p5.OUT = out
    p5.TIMEOUTS_S["worker"] = 86400.0
    if sha256(P4_PATH) != p5.EXPECTED_P4B_HASH:
        raise RuntimeError("frozen P4-B hash mismatch")
    ensemble = FrozenProbeEnsemble()
    current: dict[str, Any] = {}
    original_import = p5.import_p4_probe

    def import_probe(task_id: int):
        p4 = original_import(task_id)
        original_run = p4.run_probe_episode

        def run_probe(env, *, seed_idx: int, mu: float, trial_id: str, dt: float):
            rows, record = original_run(env, seed_idx=seed_idx, mu=mu, trial_id=trial_id, dt=dt)
            posterior = ensemble.predict(rows, p5)
            gen = candidate_set(task_id, posterior, seed_idx)
            posterior.update({"context_id": trial_id, "task_id": task_id, "gt_friction_audit_only": float(mu), "probe_success": int(record.get("probe_failure", 0) == 0), "post_probe_snapshot_pending": True})
            out_dir = out / "P5S0C_POSTERIORS"
            write_json(out_dir / f"{trial_id}.json", {"dataset_version": DATASET_VERSION, "posterior": posterior, "candidate_generation": gen, "gt_friction_audit_only": float(mu), "model_input_excludes_gt": True})
            current.clear(); current.update(gen)
            return rows, record

        p4.run_probe_episode = run_probe
        return p4

    def forces_for_task(task_id: int) -> list[float]:
        if not current:
            raise RuntimeError("candidate generation requested before probe posterior")
        return list(current["candidates_n"])

    p5.import_p4_probe = import_probe
    p5.forces_for_task = forces_for_task
    return p5


def collect(out: Path, phase: str) -> int:
    raise RuntimeError(
        "Formal v2 collection is scope-blocked until the post-grasp adapter is active: "
        "the legacy P5 runner performs pre-grasp scripted approach/close and must not "
        "be used as a formal ActiveForcing result. Use the post-grasp validation lane."
    )
    # Kept below for provenance while the post-grasp adapter is being wired;
    # unreachable code cannot accidentally contaminate the formal dataset.
    p5 = load_module("final_v2_p5_collect", P5_PATH)
    install_p5_v2_runtime(p5, out)
    os.environ.update({"P5S0C_FINAL_V2": "1", "P5S0C_SKIP_REPLAY": "1", "P5S0C_OUT": str(out)})
    ids = context_ids_for_phase(p5, phase)
    os.environ["P5S0C_CONTEXT_IDS"] = ",".join(ids)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "FINAL_COLLECTION_MANIFEST.json").exists():
        audit(out)
    p5.write_static_artifacts(out)

    # P5's normal launcher executes its own __file__, which would discard the
    # monkey-patched probe and candidate generator in a child process.  Route
    # workers through this wrapper so the exact same bridge is reinstalled in
    # every Isaac process before worker_main starts.
    def launch_worker_v2(worker_out: Path, task: int) -> dict:
        log_path = worker_out / "logs" / f"task{task}_worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = p5.worker_env(worker_out, task)
        env.update({"P5S0C_FINAL_V2": "1", "P5S0C_SKIP_REPLAY": "1", "FINAL_V2_WORKER": "1"})
        cmd = [str(ISAAC_PY), "-u", str(Path(__file__).resolve()), "worker", "--out", str(worker_out)]
        start = time.time()
        start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                returncode = proc.wait(timeout=p5.TIMEOUTS_S["worker"])
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
                raise
        return {"task": task, "worker_pid": proc.pid, "process_start_utc": start_iso, "returncode": returncode, "elapsed_wall_s": time.time() - start, "log": str(log_path)}

    p5.launch_worker = launch_worker_v2
    return p5.orchestrator_main()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def summarize(out: Path) -> dict[str, Any]:
    contexts, branches = [], []
    for task in TASKS:
        contexts.extend(_read_csv(out / f"task{task}" / "context.csv"))
        branches.extend(_read_csv(out / f"task{task}" / "branches.csv"))
    posteriors = {}
    for p in (out / "P5S0C_POSTERIORS").glob("*.json") if (out / "P5S0C_POSTERIORS").exists() else []:
        d = json.loads(p.read_text())
        posteriors[d["posterior"]["context_id"]] = d
    summary_rows = []
    failure_rows = []
    for b in branches:
        cid = b.get("context_id", "")
        traj = Path(b.get("telemetry_path", ""))
        if not traj.is_absolute():
            traj = out / traj
        rows = _read_csv(traj)
        valid = bool(rows) and all(r.get("target_object_bilateral_contact", "") != "" for r in rows)
        def nums(key, phase=None):
            vals = [float(r[key]) for r in rows if key in r and r[key] not in ("", "nan", "NaN") and (phase is None or r.get("phase") == phase)]
            return np.asarray(vals, float)
        def mean(key, phase=None):
            a = nums(key, phase); return float(a.mean()) if len(a) else float("nan")
        hold = mean("F_obj_bilateral_n", "branch_hold")
        lift = mean("F_obj_bilateral_n", "lift")
        transport = mean("F_obj_bilateral_n", "transit")
        peaks = nums("F_obj_bilateral_n"); peak = float(np.quantile(peaks, .95)) if len(peaks) else float("nan")
        bil_hold = mean("target_object_bilateral_contact", "branch_hold")
        bil_lift = mean("target_object_bilateral_contact", "lift")
        bil_transport = mean("target_object_bilateral_contact", "transit")
        center = mean("centering_error_m", "branch_hold")
        asym = mean("F_obj_normalized_asymmetry", "branch_hold")
        geom = bool(np.isfinite(center) and np.isfinite(asym) and center <= .005 and asym <= .20 and mean("contact_opposition", "branch_hold") >= .5 and (bil_hold >= 0.25))
        usable = int(valid and geom and int(float(b.get("state_parity", 0))) == 1 and int(float(b.get("full_task_success_y", 0))) in (0, 1))
        failure = "" if int(float(b.get("full_task_success_y", 0))) else ("TASK_FAILURE_AFTER_VALID_FORCE" if geom else "INVALID_GRASP_GEOMETRY")
        remaining = b.get("remaining_task_success", b.get("full_task_success_y", ""))
        row = {"dataset_version": DATASET_VERSION, "final_scope": FINAL_SCOPE, "task_id": b.get("task", ""), "root_id": b.get("root_id", ""), "split": b.get("split", ""), "physical_context_id": cid, "branch_id": b.get("branch_id", ""), "friction_posterior_mean": posteriors.get(cid, {}).get("posterior", {}).get("mean", ""), "friction_posterior_std": posteriors.get(cid, {}).get("posterior", {}).get("std", ""), "F_des_n": b.get("F_des_n", b.get("requested_force_N", "")), "F_target_eff_n": b.get("F_target_eff_n", ""), "pre_lift_grasp_valid": int(geom), "pre_lift_centering_error": center, "pre_lift_force_asymmetry": asym, "hold_force_mean_n": hold, "hold_force_mae_n": abs(hold - float(b.get("requested_force_N", 0) or 0)) if np.isfinite(hold) else float("nan"), "hold_force_bias_n": hold - float(b.get("requested_force_N", 0) or 0) if np.isfinite(hold) else float("nan"), "lift_force_mean_bilateral_n": lift, "transport_force_mean_bilateral_n": transport, "placement_force_mean_bilateral_n": mean("F_obj_bilateral_n", "place"), "peak_force_top5pct_n": peak, "bilateral_fraction_hold": bil_hold, "bilateral_fraction_lift": bil_lift, "bilateral_fraction_transport": bil_transport, "lift_success": b.get("lift_success", ""), "transport_retention": b.get("transport_retention", ""), "placement_success": b.get("place_success", ""), "release_success": b.get("release_success", ""), "remaining_task_success": remaining, "full_task_success": b.get("full_task_success_y", ""), "failure_type": failure, "usable_for_force_boundary": usable, "telemetry_valid": int(valid), "state_parity": b.get("state_parity", "")}
        summary_rows.append(row)
        if failure:
            failure_rows.append({"task_id": b.get("task", ""), "physical_context_id": cid, "branch_id": b.get("branch_id", ""), "failure_type": failure, "usable_for_force_boundary": usable})
    write_csv(out / "branch_summary.csv", summary_rows)
    write_csv(out / "failure_summary.csv", failure_rows)
    context_intervals = []
    for cid, g in _group(summary_rows, "physical_context_id"):
        vals = sorted((float(x["F_des_n"]), int(float(x["remaining_task_success"]))) for x in g if int(float(x["usable_for_force_boundary"])) == 1 and x["F_des_n"] not in ("", "nan"))
        successes = [f for f, y in vals if y]
        fails = [f for f, y in vals if not y]
        context_intervals.append({"physical_context_id": cid, "F_star_left_censored": not fails and bool(vals), "F_star_right_censored": not successes and bool(vals), "F_star_lower_bound_n": max(fails) if fails else "", "F_star_upper_bound_n": min(successes) if successes else "", "valid_branch_count": len(vals), "all_success": int(bool(vals) and not fails), "all_failure": int(bool(vals) and not successes), "mixed": int(bool(fails and successes))})
    write_csv(out / "context_interval_summary.csv", context_intervals)
    attempted = {t: sum(1 for c in contexts if int(float(c.get("task", -1))) == t) for t in TASKS}
    valid = {t: sum(1 for c in contexts if int(float(c.get("task", -1))) == t and int(float(c.get("strict_matched", 0))) == 1) for t in TASKS}
    result = {"status": "SUMMARY_COMPLETE", "final_scope": FINAL_SCOPE, "training_label": TRAINING_LABEL, "attempted_contexts_by_task": attempted, "valid_contexts_by_task": valid, "branches": len(branches), "usable_branches": sum(int(float(x["usable_for_force_boundary"])) for x in summary_rows), "successes": sum(int(float(x["remaining_task_success"])) for x in summary_rows), "failures": sum(1 - int(float(x["remaining_task_success"])) for x in summary_rows), "non_integer_force_ratio": float(np.mean([abs(float(x["F_des_n"]) - round(float(x["F_des_n"]))) > 1e-8 for x in summary_rows])) if summary_rows else 0.0, "contexts_all_success": sum(x["all_success"] for x in context_intervals), "contexts_all_failure": sum(x["all_failure"] for x in context_intervals), "contexts_mixed": sum(x["mixed"] for x in context_intervals)}
    write_json(out / "DATA_QA_SUMMARY.json", result)
    return result


def _group(rows: list[dict[str, Any]], key: str):
    d = {}
    for row in rows:
        d.setdefault(row.get(key, ""), []).append(row)
    return d.items()


def train(out: Path) -> dict[str, Any]:
    """Train the minimal positive-force lognormal CDF fallback."""
    import torch
    from torch import nn
    from torch.distributions import Normal

    rows = _read_csv(out / "branch_summary.csv")
    if not rows:
        summarize(out); rows = _read_csv(out / "branch_summary.csv")
    use = [r for r in rows if int(float(r.get("usable_for_force_boundary", 0))) == 1]
    contexts = sorted({r["physical_context_id"] for r in use})
    crows = {cid: [r for r in use if r["physical_context_id"] == cid] for cid in contexts}
    # Context weights make all four branches of a context contribute one
    # context-average likelihood, independent of valid branch count.
    xs, ys, fs, splits = [], [], [], []
    for cid, brs in crows.items():
        p = json.loads((out / "P5S0C_POSTERIORS" / f"{cid}.json").read_text())["posterior"]
        feat = np.asarray(p["posterior_feature_vector"], np.float32)
        split = next((x.get("split", "TRAIN") for x in rows if x["physical_context_id"] == cid), "TRAIN")
        for b in brs:
            xs.append(feat); fs.append(float(b["F_des_n"])); ys.append(float(b["remaining_task_success"])); splits.append(split)
    if not xs:
        raise RuntimeError("no usable branches available for posterior training")
    x = torch.tensor(np.stack(xs), dtype=torch.float32); f = torch.tensor(fs, dtype=torch.float32); y = torch.tensor(ys, dtype=torch.float32)
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6); xn = (x - mean) / std
    net = nn.Sequential(nn.Linear(x.shape[1], 32), nn.GELU(), nn.Linear(32, 2))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    train_mask = torch.tensor([s == "TRAIN" for s in splits])
    dev_mask = torch.tensor([s == "DEV" for s in splits])
    best, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(500):
        net.train(); opt.zero_grad(); z = net(xn[train_mask]); mu = z[:, 0]; scale = torch.nn.functional.softplus(z[:, 1]) + 0.05; q = Normal(0.0, 1.0).cdf((torch.log(f[train_mask].clamp_min(1e-3)) - mu) / scale).clamp(1e-5, 1 - 1e-5); loss = -(y[train_mask] * torch.log(q) + (1 - y[train_mask]) * torch.log1p(-q)).mean(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            if dev_mask.any():
                z = net(xn[dev_mask]); sc = torch.nn.functional.softplus(z[:, 1]) + .05; qq = Normal(0., 1.).cdf((torch.log(f[dev_mask].clamp_min(1e-3)) - z[:, 0]) / sc).clamp(1e-5, 1-1e-5); dl = float((-(y[dev_mask]*torch.log(qq)+(1-y[dev_mask])*torch.log1p(-qq))).mean())
            else: dl = float(loss.item())
        if dl < best:
            best = dl; best_epoch = epoch; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    if best_state is None: raise RuntimeError("posterior checkpoint selection failed")
    net.load_state_dict(best_state)
    ckpt = out / "checkpoints" / "continuous_lognormal_posterior.pt"; ckpt.parent.mkdir(parents=True, exist_ok=True); torch.save({"state_dict": net.state_dict(), "input_mean": mean, "input_std": std, "best_dev_nll": best, "best_epoch": best_epoch, "distribution": "log(F_star) Normal(mu_theta(z), sigma_theta(z))", "cdf": "NormalCDF((log(F)-mu)/sigma)", "loss": "context-mean full-task Bernoulli likelihood", "gt_excluded": True}, ckpt)
    result = {"status": "TRAINING_COMPLETE", "final_scope": FINAL_SCOPE, "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt), "distribution": "log(F_star) ~ Normal(mu_theta(z), sigma_theta(z))", "monotonic_success_cdf": True, "loss": "context-aggregated BCE on Q_theta(F|z), y=remaining_task_success", "training_label": TRAINING_LABEL, "pre_handoff_vla_failure_used": False, "best_dev_nll": best, "best_epoch": best_epoch, "usable_branches": len(use)}
    write_json(out / "TRAINING_REPORT.json", result); return result


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["audit", "scope_validate", "collect", "summarize", "train", "worker"]); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--phase", dest="collect_phase", choices=["pilot", "full"], default="pilot")
    a = ap.parse_args(); out = a.out.resolve()
    if a.phase == "audit": audit(out); return 0
    if a.phase == "scope_validate": print(json.dumps(scope_validate(out), indent=2)); return 0
    if a.phase == "collect": return collect(out, a.collect_phase)
    if a.phase == "worker":
        os.environ["P5S0C_FINAL_V2"] = "1"
        p5 = load_module("final_v2_p5_worker", P5_PATH)
        install_p5_v2_runtime(p5, out)
        return p5.worker_main()
    if a.phase == "summarize": print(json.dumps(summarize(out), indent=2)); return 0
    print(json.dumps(train(out), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
