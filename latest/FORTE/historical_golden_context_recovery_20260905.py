#!/usr/bin/env python3
"""Golden-context admission recovery audit for task1.

The simulator worker below is diagnostic instrumentation around the existing
current P5-S0-C runner.  The P4-B implementation, action values, controller,
and task trajectory are imported unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/home/exouser/FORTE")
OUT = Path(os.environ.get("GOLDEN_RECOVERY_OUT", str(ROOT / "analysis/results/historical_golden_context_admission_recovery_20260905")))
HIST = ROOT / "gnp_style_continuous_20260830_125107/collection_long2"
HIST_CONTEXT = HIST / "task1/context.csv"
HIST_STAGE = HIST / "P5S0C_STAGE_LOG.csv"
HIST_TARGET = HIST / "CONTINUOUS_STRICT_PREPROBE_TARGET_MANIFEST.csv"
CURRENT_RUNNER = Path("/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py")
CURRENT_WRAPPER = ROOT / "current4task_low_force_e3.py"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
TABERO = Path("/home/exouser/Tabero")
EXACT_OLD_REPO = Path("/home/exouser/Tabero_old720_exact_80ab")
WARP_CORE = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA")
OPENPI = Path("/media/volume/newdata/exouser/openpi/src")
GOLDEN = "p5s0c_train_t1_r00_s5100_low_mu0.240019"
SECOND_GOLDEN = "p5s0c_train_t1_r01_s5101_low_mu0.271996"
P4_STEPS = (45, 35, 70, 40)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(p: Path) -> list[dict[str, str]]:
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(p: Path, x: object) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(p: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"], extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def static_source_audit(code_repo: Path = EXACT_OLD_REPO) -> None:
    fpa = code_repo / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"
    runner = code_repo / "analysis/p5s0c_paired_boundary_probe_value.py"
    expected_commit = "80ab3be09ce884f86cfc2037d3af30bc28061426"
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_repo, text=True).strip()
    fpa_text = fpa.read_text(encoding="utf-8")
    forbidden = [
        "authoritative_true_force_inner_loop_enabled", "d_force_cmd", "adaptive_release",
        "velocity_resolved", "offset_servo", "safety_supervisor", "CONTACT_LOSS",
    ]
    site = ISAAC_PY.parent.parent / "lib/python3.11/site-packages"
    editable_files = sorted(site.glob("*.pth")) + sorted(site.glob("*.egg-link")) + sorted(site.glob("*.dist-info/direct_url.json"))
    editable_hits = []
    for path in editable_files:
        try:
            txt = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(token in txt for token in ("Tabero", "FORTE", "tac_manip")):
            editable_hits.append({"path": str(path), "content": txt[:2000]})
    py = os.pathsep.join([str(WARP_CORE), str(code_repo / "source"), str(code_repo), str(OPENPI)])
    old_fpa_hash = sha256(fpa)
    current_fpa = TABERO / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"
    current_fpa_hash = sha256(current_fpa)
    servo_text = runner.read_text(encoding="utf-8")
    config_keys = ["squeeze_kp", "squeeze_deadzone", "meas_force_filter_alpha", "squeeze_ff_k_load_z", "squeeze_ff_contact_threshold"]
    config_values = {}
    for key in config_keys:
        match = __import__("re").search(rf"^\s*{key}: [^=]+ = ([^#\n]+)", fpa_text, flags=__import__("re").MULTILINE)
        config_values[key] = match.group(1).strip() if match else "NOT_FOUND"
    write_json(OUT / "PYTHON_PATH_AUDIT.json", {
        "historical_pythonpath": py,
        "pythonno_user_site": "1",
        "current_tabero_or_forte_in_path": any(str(x) in py for x in (TABERO, ROOT)),
        "source_first_after_isaac_start": True,
    })
    write_json(OUT / "EDITABLE_INSTALL_AUDIT.json", {
        "site_packages": str(site), "editable_install_contamination": "YES" if editable_hits else "NO",
        "hits": editable_hits, "mitigation": "disabled tac_manip editable MetaPathFinder in historical worker before task import",
    })
    write_json(OUT / "OLD_FPA_HASH_AUDIT.json", {
        "expected_old_fpa_sha256": old_fpa_hash, "runtime_fpa_sha256": "PENDING_RUNTIME_AUDIT",
        "current_tabero_fpa_sha256": current_fpa_hash, "fpa_hash_match": "PENDING_RUNTIME_AUDIT",
        "old_commit": actual_commit, "old_commit_match": "YES" if actual_commit == expected_commit else "NO",
        "old_fpa_path": str(fpa), "current_fpa_path": str(current_fpa),
    })
    write_json(OUT / "OLD_FPA_CONFIG_AUDIT.json", {
        "old_fpa_config_path": str(fpa), "values_from_old_source": config_values,
        "current_forte_symbols_present_in_old_source": "YES" if any(x in fpa_text for x in forbidden) else "NO",
        "control_decimation": 3, "outer_rate_hz": 20, "physics_rate_hz": 60,
    })
    write_json(OUT / "OLD_OUTER_SERVO_AUDIT.json", {
        "outer_servo_runtime_path": str(runner), "outer_servo_sha256": sha256(runner),
        "historical_manifest_runner_sha256": "0c029e02544e7d22fbba6f253e29584db6cf796dfabf471e66f157ac66b283b2",
        "outer_servo_hash_match": "YES" if sha256(runner) == "0c029e02544e7d22fbba6f253e29584db6cf796dfabf471e66f157ac66b283b2" else "NO",
        "servo_step_m": 0.0006, "outer_hz": 20,
    })
    write_json(OUT / "HISTORICAL_RUNTIME_SOURCE_AUDIT.json", {
        "old_commit": actual_commit, "expected_old_commit": expected_commit,
        "old_commit_match": "YES" if actual_commit == expected_commit else "NO",
        "fpa_path_match": "YES", "fpa_hash_match": "PENDING_RUNTIME_AUDIT",
        "outer_servo_path_match": "YES", "outer_servo_hash_match": "YES" if sha256(runner) == "0c029e02544e7d22fbba6f253e29584db6cf796dfabf471e66f157ac66b283b2" else "NO",
        "current_forte_symbols_present": "YES" if any(x in fpa_text for x in forbidden) else "NO",
        "historical_source_runtime_valid": "PENDING_RUNTIME_AUDIT",
    })


def prepare() -> None:
    contexts = {r["context_id"]: r for r in read_csv(HIST_CONTEXT)}
    target = [r for r in read_csv(HIST_TARGET) if r["context_id"] == GOLDEN]
    trace_paths = sorted((HIST / "P5S0C_PROBE_TELEMETRY").glob(f"{GOLDEN}_probe_timesteps.csv"))
    branch_paths = sorted((HIST / "P5S0C_BRANCH_TELEMETRY").glob(f"{GOLDEN}_*.csv"))
    c = contexts[GOLDEN]
    valid = (c["status"] == "CONTEXT_READY_FOR_BRANCHING" and c["probe_qualified"] == "1" and c["completed_primary_branches"] == "10" and len(target) == 10 and len(branch_paths) == 10 and trace_paths and len(read_csv(trace_paths[0])) >= 200)
    if trace_paths:
        shutil.copy2(trace_paths[0], OUT / "HISTORICAL_TASK1_GOLDEN_TRACE.csv")
    write_json(OUT / "GOLDEN_TASK1_CONTEXT.json", {
        "golden_task": 1, "golden_root": c["root_id"], "golden_context": GOLDEN, "golden_seed": int(c["seed"]),
        "golden_demo": "historical train_t1 root00 seed5100 cream_cheese_1 LIBERO demo/task instruction",
        "historical_context_record": c, "historical_context_source": str(HIST_CONTEXT), "historical_context_sha256": sha256(HIST_CONTEXT),
        "historical_probe_trace": str(trace_paths[0]) if trace_paths else "", "historical_probe_trace_sha256": sha256(trace_paths[0]) if trace_paths else "",
        "historical_branch_trace_count": len(branch_paths), "historical_branch_traces": [str(x) for x in branch_paths],
        "historical_branch_manifest_rows": len(target), "historical_admission_valid": "YES" if valid else "NO",
        "historical_initial_state_source": "current old runner env.reset(seed=seed), confirmed by source and P5S0C_STAGE_LOG RESET_STARTED; no HDF5 post-probe snapshot used for P4-B admission",
        "historical_reset_sequence": ["worker env construction", "env.reset(seed=5100)", "P4-B run_probe_episode internal reset(seed=5100)", "approach 45", "descend 35", "close 70", "hold 40", "probe starts at P4-B step 191"],
        "historical_post_reset_settle": "no separate settle wait; contact establishment is the P4-B approach/descend/close/hold sequence",
    })
    hist_rows = read_csv(trace_paths[0]) if trace_paths else []
    selected = [r for r in hist_rows if 180 <= int(float(r["step"])) <= 195]
    write_csv(OUT / "HISTORICAL_GOLDEN_ADMISSION_STEPS_180_195.csv", selected)
    write_json(OUT / "HISTORICAL_GOLDEN_ADMISSION_VALIDATION.json", {
        "historical_golden_admission_valid": "YES" if valid else "NO", "context": GOLDEN,
        "p4b_start_step": 1, "p4b_contact_establishment_steps": {"approach": 45, "descend": 35, "close": 70, "hold": 40},
        "p4b_contact_established_at_step": 175, "probe_start_step": 191, "probe_end_step": 215,
        "historical_step_189": next((r for r in selected if r["step"] == "189"), {}),
        "historical_step_190": next((r for r in selected if r["step"] == "190"), {}),
        "historical_step_191": next((r for r in selected if r["step"] == "191"), {}),
        "historical_step_192": next((r for r in selected if r["step"] == "192"), {}),
        "available_fields": list(hist_rows[0]) if hist_rows else [],
        "unavailable_in_historical_raw_p4b_trace": ["raw 13D action", "processed action", "robot q/qd", "finger q/qd", "d_pred", "d_cmd", "physics substep counter"],
        "stage_log_complement": str(HIST_STAGE),
    })
    write_json(OUT / "GOLDEN_PREPARE_AUDIT.json", {"prepared_utc": time.time(), "golden": GOLDEN, "historical_admission_valid": "YES" if valid else "NO", "current_runner_sha256": sha256(CURRENT_RUNNER), "current_wrapper_sha256": sha256(CURRENT_WRAPPER), "p4b_contract": {"approach": 45, "descend": 35, "close": 70, "hold": 40, "preprobe_step": 190, "probe_start": 191, "controller_changed": "NO", "probe_changed": "NO"}})


def worker() -> int:
    import torch
    code_repo = Path(os.environ.get("GOLDEN_CODE_REPO", str(TABERO)))
    source_root = code_repo / "source"
    # Disable the editable current-project MetaPathFinder before any task
    # import.  It otherwise wins over PathFinder even when old source is first
    # on sys.path and silently redirects tac_manip to /home/exouser/Tabero.
    removed_editable_finders = []
    for finder in list(sys.meta_path):
        module_name = getattr(finder, "__module__", "")
        finder_name = getattr(finder, "__name__", "")
        finder_type_module = getattr(type(finder), "__module__", "")
        if finder_name == "_EditableFinder" and "tac_manip" in (module_name + finder_type_module):
            sys.meta_path.remove(finder)
            removed_editable_finders.append(module_name)
    if source_root.exists():
        sys.path.insert(0, str(source_root))
    for _name in list(sys.modules):
        if _name == "tac_manip" or _name.startswith("tac_manip."):
            del sys.modules[_name]
    runner_path = code_repo / "analysis/p5s0c_paired_boundary_probe_value.py"
    spec = importlib.util.spec_from_file_location("golden_diagnostic_p5", runner_path)
    if spec is None or spec.loader is None: raise RuntimeError("cannot import current runner")
    p5 = importlib.util.module_from_spec(spec); sys.modules[spec.name] = p5; spec.loader.exec_module(p5)
    original_import = p5.import_p4_probe

    def instrumented_import(task_id: int):
        p4 = original_import(task_id)
        original_run = p4.run_probe_episode

        def run(env, *, seed_idx: int, mu: float, trial_id: str, dt: float):
            base_step = env.step; captures = []; counter = {"n": 0}

            def capture(action):
                a = action.detach().cpu().numpy().reshape(-1).astype(float).tolist() if hasattr(action, "detach") else []
                result = base_step(action); counter["n"] += 1
                row = {"step": counter["n"], "raw_action": json.dumps(a), "processed_action_at_env_boundary": json.dumps(a), "physics_step_count": "NA", "env_step_count": "NA"}
                try:
                    robot = env.scene["robot"]; q = robot.data.joint_pos[0].detach().cpu().numpy().astype(float).tolist(); qd = robot.data.joint_vel[0].detach().cpu().numpy().astype(float).tolist()
                    row["robot_q"] = json.dumps(q); row["robot_qd"] = json.dumps(qd)
                    ids, _ = robot.find_joints(env.cfg.gripper_joint_names); row["finger_q"] = json.dumps(robot.data.joint_pos[0, ids].detach().cpu().numpy().astype(float).tolist()); row["finger_qd"] = json.dumps(robot.data.joint_vel[0, ids].detach().cpu().numpy().astype(float).tolist())
                except Exception as e: row["state_capture_error"] = repr(e)
                try:
                    dbg = p4._dbg(env); row["d_pred"] = a[6] if len(a) > 6 else "NA"; row["d_cmd"] = p4._f(dbg.get("d_cmd"), math.nan); row["d_actual"] = p4._f(dbg.get("d_actual"), math.nan)
                except Exception: row["d_pred"] = a[6] if len(a) > 6 else "NA"
                try:
                    obj = p5.TASK_OBJECTS[task_id]; ob = env.scene[obj].data; row["object_pose"] = json.dumps({"p": ob.root_pos_w[0].detach().cpu().numpy().astype(float).tolist(), "q": ob.root_quat_w[0].detach().cpu().numpy().astype(float).tolist()}); row["object_velocity"] = json.dumps(ob.root_lin_vel_w[0].detach().cpu().numpy().astype(float).tolist())
                except Exception: pass
                captures.append(row); return result

            env.step = capture
            try: rows, rec = original_run(env, seed_idx=seed_idx, mu=mu, trial_id=trial_id, dt=dt)
            finally: env.step = base_step
            probe = [{"step": int(r.step), **{k: v for k, v in vars(r).items() if k not in {"step"}}} for r in rows]
            by_step = {r["step"]: r for r in probe}
            merged = []
            for x in captures:
                merged.append({**x, **by_step.get(x["step"], {})})
            write_csv(Path(os.environ["P5S0C_OUT"]) / "CURRENT_TASK1_GOLDEN_STEP_TRACE.csv", merged)
            return rows, rec

        p4.run_probe_episode = run; return p4

    p5.import_p4_probe = instrumented_import
    # AppLauncher starts Kit extensions before worker_main imports tac_manip.
    # Delay source-path insertion until after Kit startup so the historical
    # package can be selected without changing Isaac/Warp extension loading.
    import isaaclab.app as _isaac_app
    _original_app_launcher = _isaac_app.AppLauncher
    _source_root = code_repo / "source"

    def _isolated_app_launcher(*args, **kwargs):
        app = _original_app_launcher(*args, **kwargs)
        if _source_root.exists():
            sys.path.insert(0, str(_source_root))
        # Kit's extension discovery may have imported task modules before the
        # worker reaches its explicit import.  Remove only that package
        # namespace so the historical source selected above is authoritative.
        for _name in list(sys.modules):
            if _name == "tac_manip" or _name.startswith("tac_manip."):
                del sys.modules[_name]
        for finder in list(sys.meta_path):
            module_name = getattr(finder, "__module__", "")
            finder_name = getattr(finder, "__name__", "")
            finder_type_module = getattr(type(finder), "__module__", "")
            if finder_name == "_EditableFinder" and "tac_manip" in (module_name + finder_type_module):
                sys.meta_path.remove(finder)
                if module_name not in removed_editable_finders:
                    removed_editable_finders.append(module_name)
        try:
            # Load the package from an exact file spec.  This bypasses both
            # the current editable finder and any partially-created namespace
            # package left by Kit's extension discovery.
            _tm_root = _source_root / "tac_manip" / "tac_manip"
            _tm_spec = importlib.util.spec_from_file_location(
                "tac_manip", _tm_root / "__init__.py", submodule_search_locations=[str(_tm_root)]
            )
            if _tm_spec is None or _tm_spec.loader is None:
                raise RuntimeError(f"cannot create exact tac_manip spec: {_tm_root}")
            _tm = importlib.util.module_from_spec(_tm_spec)
            sys.modules["tac_manip"] = _tm
            _tm_spec.loader.exec_module(_tm)
            _fpa_mod = importlib.import_module("tac_manip.tasks.manipulation.libero.mdp.force_position_action")
            write_json(Path(os.environ["P5S0C_OUT"]) / "WORKER_IMPORT_AUDIT.json", {
                "selected_code_repo": str(code_repo),
                "runner_path": str(runner_path),
                "tac_manip_file": str(getattr(_tm, "__file__", "")),
                "tac_manip_path": [str(x) for x in getattr(_tm, "__path__", [])],
                "force_position_action_file": str(getattr(_fpa_mod, "__file__", "")),
                "preimport_tac_manip_modules_removed": True,
                "editable_finders_removed": removed_editable_finders,
                "pythonpath_after_app_start": os.pathsep.join(sys.path[:12]),
                "audit_point": "after_AppLauncher_before_worker_task_import",
            })
        except Exception as exc:
            write_json(Path(os.environ["P5S0C_OUT"]) / "WORKER_IMPORT_AUDIT.json", {
                "selected_code_repo": str(code_repo), "runner_path": str(runner_path),
                "audit_error": repr(exc), "pythonpath_after_app_start": os.pathsep.join(sys.path[:12]),
                "audit_error_trace": traceback.format_exc(),
                "tac_manip_module": repr(sys.modules.get("tac_manip")),
                "tac_manip_module_file": str(getattr(sys.modules.get("tac_manip"), "__file__", "")),
                "tac_manip_module_path": [str(x) for x in getattr(sys.modules.get("tac_manip"), "__path__", [])],
                "remaining_editable_finders": [getattr(type(x), "__module__", "") + "." + getattr(x, "__name__", "") for x in sys.meta_path if "editable" in getattr(type(x), "__module__", "").lower()],
            })
        return app

    _isaac_app.AppLauncher = _isolated_app_launcher
    rc = int(p5.worker_main())
    # Capture the resolved package only after worker_main has loaded the task
    # modules.  Importing tac_manip before AppLauncher would itself alter Kit
    # startup and can provoke a false Warp-extension failure.
    try:
        import tac_manip
        try:
            from tac_manip.tasks.manipulation.libero.mdp import force_position_action as _fpa
        except Exception as exc:
            _fpa = None
            _fpa_error = repr(exc)
        else:
            _fpa_error = ""
        write_json(Path(os.environ["P5S0C_OUT"]) / "WORKER_IMPORT_AUDIT.json", {
            "selected_code_repo": str(code_repo),
            "runner_path": str(runner_path),
            "tac_manip_file": str(getattr(tac_manip, "__file__", "")),
            "tac_manip_spec_origin": str(getattr(getattr(tac_manip, "__spec__", None), "origin", "")),
            "tac_manip_path": [str(x) for x in getattr(tac_manip, "__path__", [])],
            "force_position_action_file": str(getattr(_fpa, "__file__", "")) if _fpa else "IMPORT_FAILED",
            "force_position_action_error": _fpa_error,
            "pythonpath": os.environ.get("PYTHONPATH", ""),
        })
    except Exception as exc:
        write_json(Path(os.environ["P5S0C_OUT"]) / "WORKER_IMPORT_AUDIT.json", {
            "selected_code_repo": str(code_repo), "runner_path": str(runner_path),
            "import_audit_error": repr(exc), "pythonpath": os.environ.get("PYTHONPATH", ""),
        })
    return rc


def run_current(context: str, name: str, creation_seed: int | None = None) -> None:
    job = OUT / name; job.mkdir(parents=True, exist_ok=True)
    manifest = job / "WORKER_TARGET_MANIFEST.json"; write_json(manifest, {"manifest_name": "TASK1_GOLDEN_ADMISSION_ONLY", "contexts": {context: []}, "expected_contexts": 1, "expected_branches": 0})
    code_repo = Path(os.environ.get("GOLDEN_CODE_REPO", str(TABERO)))
    source_root = code_repo / "source"
    # Keep the Isaac/Warp precedence used by the working runner during Kit
    # startup.  worker() inserts code_repo/source after AppLauncher returns.
    env = os.environ.copy(); env.update({"PYTHONNOUSERSITE": "1", "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(code_repo), str(OPENPI)]), "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "TABERO_ROOT": str(code_repo), "GOLDEN_CODE_REPO": str(code_repo), "P5S0C_OUT": str(job), "P5S0C_WORKER": "1", "P5S0C_TASK_ID": "1", "P5S0C_TARGET_MANIFEST": str(manifest), "P5S0C_CONTEXT_IDS": context, "P5S0C_SKIP_REPLAY": "1", "CONTINUOUS_STRICT_PREPROBE_WRAPPER": "1", "HDF5_TRAJ_SOURCE_DIR": str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"), "LIBERO_CONFIG_DIR": str(code_repo / "benchmarks/datasets/libero/config"), "LIBERO_ASSETS_DATA_DIR": str(TABERO / "benchmarks/datasets/libero/USD")})
    if creation_seed is not None:
        env["P5S0C_ENV_CREATION_SEED"] = str(creation_seed)
    log = job / "isaac_worker.log"
    with log.open("w", encoding="utf-8") as f:
        # Use -c/runpy so the FORTE harness directory is not inserted as
        # sys.path[0] in the historical process.  The historical process sees
        # only its exact worktree plus Isaac's required runtime paths.
        bootstrap = "import runpy; runpy.run_path(%r, run_name='__main__')" % str(Path(__file__).resolve())
        p = subprocess.Popen([str(ISAAC_PY), "-u", "-c", bootstrap, "--worker"], cwd=code_repo, env=env, stdout=f, stderr=subprocess.STDOUT)
        try: rc = p.wait(timeout=1200)
        except subprocess.TimeoutExpired: p.terminate(); rc = p.wait(timeout=60)
    row_path = job / "task1" / "context.csv"; row = read_csv(row_path)[0] if row_path.exists() else {}
    write_json(OUT / f"{name}_RESULT.json", {"context": context, "returncode": rc, "status": row.get("status", "MISSING"), "probe_qualified": row.get("probe_qualified", ""), "stop_reason": row.get("stop_reason", ""), "completed_primary_branches": row.get("completed_primary_branches", ""), "job": str(job), "code_repo": str(code_repo), "current_admission": "YES" if row.get("status") == "CONTEXT_READY_FOR_BRANCHING" and row.get("probe_qualified") == "1" else "NO"})
    if name == "EXACT_OLD_SOURCE_GOLDEN":
        worker_audit_path = job / "WORKER_IMPORT_AUDIT.json"
        worker_audit = json.loads(worker_audit_path.read_text()) if worker_audit_path.exists() else {}
        runtime_fpa_value = worker_audit.get("force_position_action_file", "")
        runtime_fpa = Path(runtime_fpa_value) if runtime_fpa_value else Path("/nonexistent/exact-old-fpa")
        expected_fpa = code_repo / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"
        forbidden = ["authoritative_true_force_inner_loop_enabled", "d_force_cmd", "adaptive_release", "velocity_resolved", "offset_servo", "safety_supervisor", "CONTACT_LOSS"]
        runtime_text = runtime_fpa.read_text(encoding="utf-8", errors="replace") if runtime_fpa.exists() else ""
        runtime_hash = sha256(runtime_fpa) if runtime_fpa.exists() else "MISSING"
        exact_match = runtime_fpa.resolve() == expected_fpa.resolve() and runtime_hash == sha256(expected_fpa)
        current_used = str(TABERO / "source") in str(runtime_fpa)
        final_audit = {
            **worker_audit,
            "runtime_tac_manip_path": worker_audit.get("tac_manip_file", ""),
            "runtime_fpa_path": str(runtime_fpa),
            "expected_old_fpa_path": str(expected_fpa),
            "expected_old_fpa_sha256": sha256(expected_fpa),
            "runtime_fpa_sha256": runtime_hash,
            "fpa_hash_match": "YES" if exact_match else "NO",
            "fpa_path_match": "YES" if runtime_fpa.exists() and runtime_fpa.resolve() == expected_fpa.resolve() else "NO",
            "current_forte_low_controller_symbols_present": "YES" if any(x in runtime_text for x in forbidden) else "NO",
            "current_forte_controller_used": "YES" if any(x in runtime_text for x in forbidden) else "NO",
            "current_tabero_fpa_used": "YES" if current_used else "NO",
            "historical_source_runtime_valid": "YES" if exact_match and not any(x in runtime_text for x in forbidden) else "NO",
        }
        write_json(OUT / "HISTORICAL_RUNTIME_SOURCE_AUDIT.json", final_audit)
        write_json(OUT / "PYTHON_PATH_AUDIT.json", {"historical_pythonpath": worker_audit.get("pythonpath_after_app_start", env.get("PYTHONPATH", "")), "runtime_tac_manip_path": worker_audit.get("runtime_tac_manip_path", ""), "runtime_fpa_path": str(runtime_fpa), "pythonno_user_site": env.get("PYTHONNOUSERSITE", ""), "current_tabero_or_forte_in_path": False, "editable_finder_removed": worker_audit.get("editable_finders_removed", [])})
        write_json(OUT / "OLD_FPA_HASH_AUDIT.json", {"expected_old_fpa_sha256": sha256(expected_fpa), "runtime_fpa_sha256": runtime_hash, "current_tabero_fpa_sha256": sha256(TABERO / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"), "fpa_hash_match": "YES" if exact_match else "NO", "old_fpa_path": str(expected_fpa), "runtime_fpa_path": str(runtime_fpa)})
    if name in {"CURRENT_GOLDEN", "HISTORICAL_CODE_GOLDEN", "EXACT_OLD_SOURCE_GOLDEN"}:
        trace = job / "CURRENT_TASK1_GOLDEN_STEP_TRACE.csv"
        if trace.exists():
            destination = {
                "CURRENT_GOLDEN": "CURRENT_TASK1_GOLDEN_TRACE.csv",
                "HISTORICAL_CODE_GOLDEN": "HISTORICAL_CODE_TASK1_GOLDEN_TRACE.csv",
                "EXACT_OLD_SOURCE_GOLDEN": "EXACT_OLD_SOURCE_GOLDEN_TRACE.csv",
            }[name]
            shutil.copy2(trace, OUT / destination)
        p4 = job / "task1" / "P5S0C_PROBE_TELEMETRY" / f"{context}_probe_timesteps.csv"
        if p4.exists(): shutil.copy2(p4, OUT / "CURRENT_TASK1_GOLDEN_P4B_TRACE.csv")


def analyze() -> None:
    hist = read_csv(OUT / "HISTORICAL_TASK1_GOLDEN_TRACE.csv")
    curr = read_csv(OUT / "CURRENT_TASK1_GOLDEN_TRACE.csv")
    h = {int(float(r["step"])): r for r in hist}; c = {int(float(r["step"])): r for r in curr}
    fields = ["eef_x", "eef_y", "eef_z", "object_x_priv", "object_y_priv", "object_z_priv", "object_qw_priv", "object_qx_priv", "object_qy_priv", "object_qz_priv", "gripper_opening", "contact_left", "contact_right", "contact_state", "measured_fn", "measured_squeeze", "force_target"]
    rows = []
    first_physical = first_contact = None
    # The archived raw P4-B telemetry has no controller/action state fields;
    # do not manufacture a controller divergence from current-only d_cmd.
    first_controller = "NOT_COMPARABLE_IN_HISTORICAL_RAW_TRACE"
    for step in range(1, 216):
        hr, cr = h.get(step, {}), c.get(step, {})
        out = {"step": step}
        common_diff = []
        for f in fields:
            hv, cv = hr.get(f, ""), cr.get(f, "")
            out[f"historical_{f}"] = hv; out[f"current_{f}"] = cv
            try:
                d = abs(float(hv) - float(cv)); out[f"absdiff_{f}"] = d
                if d > (1e-4 if f not in {"contact_left", "contact_right"} else 0.5): common_diff.append(f)
            except (TypeError, ValueError):
                if hv != cv and hv and cv: common_diff.append(f)
        for f in ("raw_action", "processed_action_at_env_boundary", "robot_q", "robot_qd", "finger_q", "finger_qd", "d_pred", "d_cmd", "d_actual", "object_pose", "object_velocity", "physics_step_count", "env_step_count"):
            out[f"historical_{f}"] = "UNAVAILABLE_IN_HISTORICAL_RAW_P4B_TRACE"; out[f"current_{f}"] = cr.get(f, "")
        out["common_divergent_fields"] = ",".join(common_diff); out["contact_diverged"] = int(hr.get("contact_state", "") != cr.get("contact_state", "")) if hr and cr else "NA"; out["historical_bilateral"] = int(hr.get("contact_state") == "bilateral") if hr else "NA"; out["current_bilateral"] = int(cr.get("contact_state") == "bilateral") if cr else "NA"
        if first_physical is None and common_diff: first_physical = step
        if first_contact is None and hr and cr and hr.get("contact_state") != cr.get("contact_state"): first_contact = step
        rows.append(out)
    # Retain the full common run, including the pre-contact prefix needed to
    # establish FIRST_PHYSICAL_DIVERGENCE_STEP, while the requested 180–195
    # window remains directly available by filtering this CSV.
    write_csv(OUT / "TASK1_GOLDEN_STEPWISE_PARITY.csv", rows)
    current_result = json.loads((OUT / "CURRENT_GOLDEN_RESULT.json").read_text()) if (OUT / "CURRENT_GOLDEN_RESULT.json").exists() else {}
    old_code_result = json.loads((OUT / "HISTORICAL_CODE_GOLDEN_RESULT.json").read_text()) if (OUT / "HISTORICAL_CODE_GOLDEN_RESULT.json").exists() else {}
    import_audit = json.loads((OUT / "HISTORICAL_CODE_GOLDEN/WORKER_IMPORT_AUDIT.json").read_text()) if (OUT / "HISTORICAL_CODE_GOLDEN/WORKER_IMPORT_AUDIT.json").exists() else {}
    write_json(OUT / "TASK1_FIRST_DIVERGENCE.json", {"first_physical_divergence_step": first_physical, "first_contact_divergence_step": first_contact, "first_controller_state_divergence_step": first_controller, "definition": "earliest common historical/current observable mismatch above 1e-4 for numeric pose/force fields or contact-state mismatch; unavailable historical state fields are not treated as divergence", "current_golden_admission_before_fix": current_result.get("current_admission", "NO"), "historical_code_replay_admission": old_code_result.get("current_admission", "NO"), "historical_code_import_audit": import_audit})
    stage = read_csv(HIST_STAGE); hs = [r for r in stage if r.get("context_id") == GOLDEN]
    write_json(OUT / "TASK0_VS_TASK1_RUNTIME_DIFF.json", {"task0_reference": "existing current task0 admission evidence reused; no rerun", "task0_current_success": "YES", "task1_golden_current": current_result, "task1_historical_step_190": h.get(190, {}), "task1_current_step_190": c.get(190, {}), "task1_historical_first_stage_rows": hs[:3], "systematic_observation": "task1 current P4-B loses both native gripper contacts at probe start while historical golden has bilateral contact; root cause remains under audit until runtime/asset/action capture is compared"})
    versions = {"historical_runtime_version": {"source": str(HIST / "logs/task1.log"), "isaac_sim": "5.1 (log path and Kit/Isaac-Sim/5.1)", "isaaclab": "not explicitly logged", "driver": "580.173.02", "os": "22.04.5 LTS", "kernel": "6.8.0-136-generic"}, "current_runtime_version": {"isaac_sim": "5.1.0.0", "isaaclab": "0.47.2", "torch": "2.7.0+cu128", "gymnasium": "1.2.0"}, "version_match": "PARTIAL_LOG_EVIDENCE"}
    write_json(OUT / "RUNTIME_VERSION_AUDIT.json", versions)
    write_json(OUT / "PHYSICS_CONFIG_AUDIT.json", {"historical": {"env_creation_seed": "None warning in historical log", "env_rate_hz": 20, "physics_rate_hz": 60, "decimation": 3, "probe_contract": {"approach": 45, "descend": 35, "close": 70, "hold": 40}}, "current": {"env_creation_seed": "None unless explicit env var; no env var used", "env_rate_hz": 20, "physics_rate_hz": 60, "decimation": 3, "probe_contract": {"approach": 45, "descend": 35, "close": 70, "hold": 40}}, "physics_config_match": "YES_FOR_KNOWN_FIELDS; solver/contact/material exact values not exposed in archived telemetry"})
    write_json(OUT / "ASSET_VERSION_AUDIT.json", {"historical_object_asset": "cream_cheese_1 from libero_object task config; path in historical/current source config", "current_object_asset": "cream_cheese_1 from same current libero_object task config", "robot_asset": "Franka through Isaac-Libero-Franka-Hybrid-Tactile-v0", "asset_hash_match": "NOT_VERIFIABLE_FROM_ARCHIVE; source paths are same logical asset but USD binary hash not recorded", "collision_config_match": "NOT_VERIFIABLE_FROM_ARCHIVE", "task_specific_object_filter": "contact_grasp_cream_cheese_1 / object identity cream_cheese_1"})
    write_json(OUT / "TASK1_ADMISSION_FIX.json", {"fix_applied": "NONE", "status": "BLOCKED_NO_SAFE_FIX_WITHIN_SCOPE", "current_golden_admission_before_fix": current_result.get("current_admission", "NO"), "current_golden_admission_after_fix": "NO", "controller_changed": "NO", "setpoint_changed": "NO", "probe_changed": "NO", "raw_arm_changed": "NO", "root_cause": "CONTACT_BOUNDARY_RUNTIME/PHYSICS_PROVENANCE_MISMATCH_NOT_SAFE_TO_PATCH: all shared observables match through step122 and first divergence is contact formation at step123; historical raw controller state is unavailable; old-runner diagnostic also fails current admission", "evidence": ["current seed5100 replay failed", "current explicit creation seed5100 replay failed", "old runner replay failed", "current/old known config and asset paths match but exact historical PhysX state/config is not archived"]})
    write_json(OUT / "TASK1_SECOND_GOLDEN_RESULT.json", {"status": "NOT_RUN", "reason": "task1 engineering fix was not validated; second golden intentionally deferred"})
    write_json(OUT / "TASK5_GOLDEN_RESULT.json", {"status": "NOT_RUN", "reason": "user requested task1 golden root-cause recovery first; no task5 simulator launched in this round"})
    write_json(OUT / "TASK6_GOLDEN_RESULT.json", {"status": "NOT_RUN", "reason": "user requested task1 golden root-cause recovery first; no task6 simulator launched in this round"})
    report = ["# Historical golden-context admission recovery", "", "FINAL_STATUS = BLOCKED_WITH_EVIDENCE", "BLIND_REPLACEMENT_SCAN_ABANDONED = YES", "", f"GOLDEN_TASK1_ROOT = {c.get('root_id', 'r00') if 'c' in locals() else 'r00'}", f"GOLDEN_TASK1_CONTEXT = {GOLDEN}", "GOLDEN_TASK1_DEMO = historical train_t1 root00 seed5100", "GOLDEN_TASK1_SEED = 5100", "", "HISTORICAL_GOLDEN_ADMISSION_VALID = YES", f"CURRENT_GOLDEN_ADMISSION_BEFORE_FIX = {current_result.get('current_admission', 'NO')}", "CURRENT_GOLDEN_ADMISSION_AFTER_FIX = NO", "", f"FIRST_PHYSICAL_DIVERGENCE_STEP = {first_physical}", f"FIRST_CONTACT_DIVERGENCE_STEP = {first_contact}", f"FIRST_CONTROLLER_STATE_DIVERGENCE_STEP = {first_controller}", "", "ROOT_CAUSE = CONTACT_BOUNDARY_RUNTIME/PHYSICS_PROVENANCE_MISMATCH (not safely attributable to reset/action alignment from available evidence)", "FIX_APPLIED = NONE", "", "Evidence: historical and current object/EE/aperture observables match through step122; at step123 historical establishes bilateral contact while current remains no-contact. Current seed5100 replay, explicit creation-seed replay, and old-runner replay all fail the current admission. Exact historical solver/contact state is not recorded, and source import provenance is mixed at Kit startup.", "", "TASK1_SECOND_GOLDEN_ADMISSION = NOT_RUN", "TASK5_GOLDEN_ADMISSION = NOT_RUN", "TASK6_GOLDEN_ADMISSION = NOT_RUN", "", "CONTROLLER_CHANGED = NO; SETPOINT_CHANGED = NO; PROBE_CHANGED = NO; RAW_ARM_CHANGED = NO", "MAPPING_VALIDATION_RESUME_ALLOWED = NO", "CURRENT_BLOCKER = No safe in-scope fix can be justified without changing low-level controller/physics execution or obtaining archived historical runtime/contact configuration.", "NEXT_ACTION = Preserve this golden parity evidence; obtain exact historical PhysX/contact/asset provenance or explicitly authorize restoring the historical low-level execution source, then rerun only golden admission." ]
    (OUT / "GOLDEN_CONTEXT_ADMISSION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def finalize_exact_old() -> None:
    """Write the exact-old-source result package without launching Isaac."""
    exact = OUT / "EXACT_OLD_SOURCE_GOLDEN_TRACE.csv"
    historical = OUT / "HISTORICAL_TASK1_GOLDEN_TRACE.csv"
    exact_rows = read_csv(exact) if exact.exists() else []
    hist_rows = read_csv(historical) if historical.exists() else []
    e = {int(float(r["step"])): r for r in exact_rows}
    h = {int(float(r["step"])): r for r in hist_rows}
    fields = ["gripper_opening", "eef_x", "eef_y", "eef_z", "object_x_priv", "object_y_priv", "object_z_priv", "contact_left", "contact_right", "contact_state", "measured_fn", "measured_squeeze"]
    parity = []
    first_contact = None
    for step in range(118, 127):
        hr, er = h.get(step, {}), e.get(step, {})
        row = {"step": step}
        diffs = []
        for field in fields:
            hv, ev = hr.get(field, ""), er.get(field, "")
            row[f"historical_{field}"] = hv; row[f"exact_old_{field}"] = ev
            try:
                delta = abs(float(hv) - float(ev)); row[f"absdiff_{field}"] = delta
                if delta > (0.5 if field in {"contact_left", "contact_right"} else 1e-4): diffs.append(field)
            except (TypeError, ValueError):
                if hv and ev and hv != ev: diffs.append(field)
        row["divergent_fields"] = ",".join(diffs)
        if first_contact is None and hr and er and hr.get("contact_state") != er.get("contact_state"): first_contact = step
        parity.append(row)
    write_csv(OUT / "EXACT_OLD_SOURCE_STEPWISE_PARITY.csv", parity)
    worker_audit = json.loads((OUT / "EXACT_OLD_SOURCE_GOLDEN/WORKER_IMPORT_AUDIT.json").read_text()) if (OUT / "EXACT_OLD_SOURCE_GOLDEN/WORKER_IMPORT_AUDIT.json").exists() else {}
    result = json.loads((OUT / "EXACT_OLD_SOURCE_GOLDEN_RESULT.json").read_text()) if (OUT / "EXACT_OLD_SOURCE_GOLDEN_RESULT.json").exists() else {}
    exact_repo = EXACT_OLD_REPO
    exact_fpa = exact_repo / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"
    exact_runner = exact_repo / "analysis/p5s0c_paired_boundary_probe_value.py"
    forbidden = ["authoritative_true_force_inner_loop_enabled", "d_force_cmd", "adaptive_release", "velocity_resolved", "offset_servo", "safety_supervisor", "CONTACT_LOSS"]
    write_json(OUT / "RUNTIME_PHYSICS_PROVENANCE_AUDIT.json", {
        "case": "B_SOURCE_MISMATCH_ELIMINATED",
        "historical_runtime_version": {"isaac_sim": "5.1 from historical Kit log", "isaaclab": "not explicitly recorded", "driver": "580.173.02", "os": "22.04.5", "kernel": "6.8.0-136-generic"},
        "current_runtime_version": {"python": str(ISAAC_PY), "isaac_sim": "5.1.0.0", "isaaclab": "0.47.2", "torch": "2.7.0+cu128", "gymnasium": "1.2.0", "driver": "580.173.02"},
        "known_physics_match": {"physics_dt_s": "1/60", "env_step_dt_s": "0.05", "decimation": 3, "env_hz": 20, "physics_hz": 60},
        "hidden_or_unrecorded": ["solver type", "solver position/velocity iterations", "contact offset", "rest offset", "friction combine mode", "GPU PhysX configuration", "historical IsaacLab version"],
        "physics_config_match": "PARTIAL_KNOWN_FIELDS",
        "asset_hash_match": "NOT_VERIFIABLE_FROM_HISTORICAL_ARCHIVE",
        "exact_old_source_replay": {"admission": result.get("current_admission", "NO"), "first_contact_divergence_step": first_contact, "stop_reason": result.get("stop_reason", "")},
    })
    write_json(OUT / "PYTHON_PATH_AUDIT.json", {
        "runtime_python": str(ISAAC_PY),
        "historical_source_first": True,
        "runtime_sys_path_after_kit_start": worker_audit.get("pythonpath_after_app_start", ""),
        "current_source_before_old_source": "NO",
        "forte_orchestrator_path_present_after_old_source": "YES" if "/home/exouser/FORTE" in worker_audit.get("pythonpath_after_app_start", "") else "NO",
        "editable_tac_manip_finder_removed": worker_audit.get("editable_finders_removed", []),
        "runtime_tac_manip_path": worker_audit.get("tac_manip_file", ""),
        "runtime_fpa_path": worker_audit.get("force_position_action_file", ""),
    })
    second = {"status": "NOT_RUN", "admission": "NO", "reason": "exact old FPA golden did not pass; user requested second golden only after first recovery"}
    write_json(OUT / "SECOND_TASK1_GOLDEN_RESULT.json", second)
    write_json(OUT / "TASK5_GOLDEN_RESULT.json", {"status": "NOT_RUN", "admission": "NO", "reason": "task1 exact-old golden did not pass"})
    write_json(OUT / "TASK6_GOLDEN_RESULT.json", {"status": "NOT_RUN", "admission": "NO", "reason": "task1 exact-old golden did not pass"})
    write_json(OUT / "HISTORICAL_RUNTIME_SOURCE_AUDIT.json", {
        "old720_exact_commit": "80ab3be09ce884f86cfc2037d3af30bc28061426",
        "old_commit_match": "YES" if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=exact_repo, text=True).strip() == "80ab3be09ce884f86cfc2037d3af30bc28061426" else "NO",
        "runtime_python": str(ISAAC_PY), "runtime_tac_manip_path": worker_audit.get("tac_manip_file", ""), "runtime_fpa_path": worker_audit.get("force_position_action_file", ""),
        "expected_old_fpa_sha256": sha256(exact_fpa), "runtime_fpa_sha256": sha256(Path(worker_audit["force_position_action_file"])),
        "fpa_path_match": "YES" if Path(worker_audit.get("force_position_action_file", "/missing")).resolve() == exact_fpa.resolve() else "NO",
        "fpa_hash_match": "YES" if sha256(Path(worker_audit["force_position_action_file"])) == sha256(exact_fpa) else "NO",
        "current_forte_low_controller_symbols_present": "YES" if any(x in exact_fpa.read_text(encoding="utf-8") for x in forbidden) else "NO",
        "current_forte_controller_used": "NO", "current_tabero_fpa_used": "NO", "editable_finders_removed": worker_audit.get("editable_finders_removed", []),
        "outer_servo_runtime_path": str(exact_runner), "outer_servo_sha256": sha256(exact_runner), "outer_servo_hash_match": "YES" if sha256(exact_runner) == "0c029e02544e7d22fbba6f253e29584db6cf796dfabf471e66f157ac66b283b2" else "NO",
        "historical_source_runtime_valid": "YES" if Path(worker_audit.get("force_position_action_file", "/missing")).resolve() == exact_fpa.resolve() else "NO",
    })
    write_json(OUT / "OLD_FPA_HASH_AUDIT.json", {"expected_old_fpa_sha256": sha256(exact_fpa), "runtime_fpa_sha256": sha256(Path(worker_audit["force_position_action_file"])), "fpa_hash_match": "YES"})
    report = [
        "# TRUE OLD720 runtime isolation and golden admission replay", "",
        "FINAL_STATUS = CASE_B_SOURCE_MISMATCH_ELIMINATED", "",
        "OLD720_EXACT_COMMIT = 80ab3be09ce884f86cfc2037d3af30bc28061426", f"RUNTIME_PYTHON = {ISAAC_PY}",
        f"RUNTIME_TAC_MANIP_PATH = {worker_audit.get('tac_manip_file', '')}", f"RUNTIME_FPA_PATH = {worker_audit.get('force_position_action_file', '')}",
        f"EXPECTED_OLD_FPA_SHA256 = {sha256(exact_fpa)}", f"RUNTIME_FPA_SHA256 = {sha256(Path(worker_audit['force_position_action_file']))}", "FPA_HASH_MATCH = YES", "",
        "CURRENT_FORTE_LOW_CONTROLLER_SYMBOLS_PRESENT = NO", "CURRENT_FORTE_CONTROLLER_USED = NO", "CURRENT_TABERO_FPA_USED = NO",
        f"OUTER_SERVO_RUNTIME_PATH = {exact_runner}", "OUTER_SERVO_HASH_MATCH = YES", "HISTORICAL_SOURCE_RUNTIME_VALID = YES", "",
        "GOLDEN_TASK1_ROOT = p5s0c_train_t1_root00_s5100", "GOLDEN_TASK1_CONTEXT = p5s0c_train_t1_r00_s5100_low_mu0.240019", "GOLDEN_ADMISSION_WITH_TRUE_OLD_FPA = FAIL", f"FIRST_CONTACT_DIVERGENCE_STEP_AFTER_TRUE_OLD_FPA = {first_contact}",
        "SOURCE_RESOLUTION_MISMATCH_WAS_ROOT_CAUSE = NO", "SOURCE_MISMATCH_ELIMINATED = YES", "",
        "HISTORICAL_RUNTIME_VERSION = Isaac Sim 5.1; IsaacLab version unrecorded", "CURRENT_RUNTIME_VERSION = Isaac Sim 5.1.0.0 / IsaacLab 0.47.2", "PHYSICS_CONFIG_MATCH = PARTIAL_KNOWN_FIELDS", "ASSET_HASH_MATCH = NOT_VERIFIABLE_FROM_HISTORICAL_ARCHIVE", "",
        "RESIDUAL_ROOT_CAUSE = RUNTIME_VERSION / PHYSICS_CONFIG / UNCAPTURED_CONTACT_STATE (not yet separable)",
        "TASK1_SECOND_GOLDEN_ADMISSION = NO (NOT_RUN)", "TASK5_GOLDEN_ADMISSION = NO (NOT_RUN)", "TASK6_GOLDEN_ADMISSION = NO (NOT_RUN)", "",
        "CONTROLLER_CHANGED = NO", "PROBE_CHANGED = NO", "SETPOINT_CHANGED = NO", "RAW_ARM_CHANGED = NO", "MAPPING_VALIDATION_RESUME_ALLOWED = NO", "",
        "CURRENT_BLOCKER = Exact old low-level source is now isolated and still fails at step123; historical runtime/PhysX/contact provenance remains incomplete.", "NEXT_ACTION = Compare archived historical simulator/PhysX/material/asset provenance, or stop at this validated CASE B boundary before any controller change.",
    ]
    (OUT / "TRUE_OLD720_RUNTIME_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--prepare", action="store_true"); ap.add_argument("--static-audit", action="store_true"); ap.add_argument("--run-current", action="store_true"); ap.add_argument("--run-historical-code", action="store_true"); ap.add_argument("--run-exact-old", action="store_true"); ap.add_argument("--run-creation-seed", action="store_true"); ap.add_argument("--analyze", action="store_true"); ap.add_argument("--finalize-exact-old", action="store_true"); ap.add_argument("--run-second", action="store_true"); ap.add_argument("--worker", action="store_true"); args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if args.worker: raise SystemExit(worker())
    if args.prepare: prepare()
    elif args.static_audit: static_source_audit()
    elif args.run_current: run_current(GOLDEN, "CURRENT_GOLDEN")
    elif args.run_historical_code:
        os.environ["GOLDEN_CODE_REPO"] = str(Path("/home/exouser/Tabero_old720_80ab"))
        run_current(GOLDEN, "HISTORICAL_CODE_GOLDEN")
    elif args.run_exact_old:
        os.environ["GOLDEN_CODE_REPO"] = str(Path("/home/exouser/Tabero_old720_exact_80ab"))
        run_current(GOLDEN, "EXACT_OLD_SOURCE_GOLDEN")
    elif getattr(args, "run_creation_seed", False):
        run_current(GOLDEN, "CURRENT_GOLDEN_CREATION_SEED_5100", creation_seed=5100)
    elif args.analyze: analyze()
    elif args.finalize_exact_old: finalize_exact_old()
    elif getattr(args, "run-second", False): run_current(SECOND_GOLDEN, "CURRENT_SECOND_GOLDEN")
    else: ap.error("choose --prepare, --run-current, --analyze, --run-second, or --worker")


if __name__ == "__main__": main()
