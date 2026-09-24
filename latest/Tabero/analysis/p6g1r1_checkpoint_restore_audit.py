#!/usr/bin/env python3
"""Forensic audit for the frozen P6-G1-R1 OpenPI/Orbax runtime.

This file deliberately uses the same official policy construction call as the
P6-G1 B5 server.  It never writes to the checkpoint and never launches a
scientific rollout.  Each stage is intended to run in a fresh process.
"""
from __future__ import annotations

import argparse
import csv
import faulthandler
import hashlib
import importlib.util
import json
import os
import pathlib
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback
from typing import Any


ROOT = pathlib.Path("/home/exouser/Tabero")
VTLA = pathlib.Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA")
CHECKPOINT = pathlib.Path(
    "/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/"
    "checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999"
)
PARAMS = CHECKPOINT / "params"
NORM_STATS = CHECKPOINT / "assets/NathanWu7/tabero"
POLICY_CONFIG = "pi0_lora_tacfield_tabero"
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"
ISAAC_PY = pathlib.Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
SUCCESSFUL_P6G1_RUNNER = ROOT / "analysis/p6g1_primitive_ik_vla_grasp_realization.py"
WARP_CORE = pathlib.Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64")
OPENPI_CLIENT_SRC = ROOT / "benchmarks/openpi/openpi-client/src"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time_ns() % 1_000_000_000):09d}Z"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            parts = value.strip().split()
            if parts and parts[0].isdigit():
                out[key] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
    except Exception as exc:
        out["error"] = repr(exc)  # type: ignore[assignment]
    return out


def rss_bytes() -> int | None:
    try:
        for line in pathlib.Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def gpu_status() -> dict[str, Any]:
    result: dict[str, Any] = {"timestamp_utc": now()}
    for argv, key in [
        (["nvidia-smi"], "nvidia_smi"),
        (["nvidia-smi", "-L"], "nvidia_smi_L"),
        (["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], "query"),
    ]:
        try:
            p = subprocess.run(argv, text=True, capture_output=True, timeout=15)
            result[key] = {"returncode": p.returncode, "stdout": p.stdout[-4000:], "stderr": p.stderr[-4000:]}
        except Exception as exc:
            result[key] = {"error": repr(exc)}
    return result


class Audit:
    def __init__(self, out: pathlib.Path, stage: str):
        self.out = out
        self.stage = stage
        self.log = (out / f"P6G1R1_{stage.upper()}_AUDIT.log").open("a", encoding="utf-8", buffering=1)
        self.stack = (out / f"P6G1R1_{stage.upper()}_STACKS.log").open("a", encoding="utf-8", buffering=1)
        self.started = time.monotonic()

    def milestone(self, name: str, **extra: Any) -> None:
        row = {"timestamp_utc": now(), "stage": self.stage, "milestone": name, "elapsed_s": time.monotonic() - self.started, "rss_bytes": rss_bytes(), "host_mem": read_meminfo(), **extra}
        line = json.dumps(row, sort_keys=True, default=str)
        print(line, flush=True)
        self.log.write(line + "\n")

    def start(self) -> None:
        self.milestone(
            "stage_start",
            pid=os.getpid(),
            cwd=os.getcwd(),
            python=sys.executable,
            python_version=sys.version,
            platform=platform.platform(),
            env={k: os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_COMPILATION_CACHE_DIR", "PYTHONNOUSERSITE", "PYTHONPATH"]},
            gpu=gpu_status(),
        )
        faulthandler.enable(self.stack)
        faulthandler.dump_traceback_later(60, repeat=True, file=self.stack)

    def finish(self, ok: bool, **extra: Any) -> None:
        faulthandler.cancel_dump_traceback_later()
        self.milestone("stage_complete" if ok else "stage_failed", **extra)
        sentinel = self.out / f"P6G1R1_{self.stage.upper()}_{'COMPLETE' if ok else 'FAILED'}.json"
        sentinel.write_text(json.dumps({"stage": self.stage, "ok": ok, "timestamp_utc": now(), **extra}, indent=2, sort_keys=True, default=str) + "\n")
        self.log.close()
        self.stack.close()


def import_openpi():
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.training import config
    return policy_config, normalize, config


def make_policy(audit: Audit):
    audit.milestone("import_policy_modules_start")
    policy_config, normalize, config = import_openpi()
    audit.milestone("import_policy_modules_complete")
    norm_stats = normalize.load(NORM_STATS)
    audit.milestone("norm_stats_loaded", keys=sorted(norm_stats))
    train_config = config.get_config(POLICY_CONFIG)
    audit.milestone("train_config_loaded", config=POLICY_CONFIG, config_repr=repr(train_config))
    audit.milestone("official_create_trained_policy_start", checkpoint=str(CHECKPOINT))
    policy = policy_config.create_trained_policy(train_config, CHECKPOINT, norm_stats=norm_stats)
    audit.milestone("official_create_trained_policy_complete", metadata=getattr(policy, "metadata", {}))
    return policy, train_config


def flatten_tree(tree: Any) -> list[tuple[str, Any]]:
    try:
        from flax import traverse_util
        flat = traverse_util.flatten_dict(tree, sep="/")
        return sorted((str(k), v) for k, v in flat.items())
    except Exception:
        return [(str(i), v) for i, v in enumerate(tree)] if isinstance(tree, (list, tuple)) else [("<root>", tree)]


def normalize_checkpoint_path(path: str) -> str:
    """Orbax NNX metadata exposes state leaves with a trailing /value.

    openpi.models.model.restore_params removes this wrapper before comparing
    the restored tree with BaseModelConfig's pure-dict model state.
    """
    return path[:-6] if path.endswith("/value") else path


def leaf_info(value: Any) -> tuple[str, list[int], str, str]:
    shape = list(getattr(value, "shape", ()))
    dtype = str(getattr(value, "dtype", type(value).__name__))
    sharding = str(getattr(value, "sharding", ""))
    return str(type(value).__name__), shape, dtype, sharding


def checkpoint_metadata(audit: Audit) -> dict[str, Any]:
    import orbax.checkpoint as ocp
    audit.milestone("checkpoint_metadata_start", params=str(PARAMS))
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(PARAMS)
    audit.milestone("checkpoint_metadata_complete", top_keys=list(metadata))
    return metadata


def write_tree_csv(out: pathlib.Path, rows: list[dict[str, Any]], name: str) -> None:
    fields = ["source", "path", "type", "shape", "dtype", "sharding"]
    with (out / name).open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def stage_metadata(audit: Audit) -> dict[str, Any]:
    audit.milestone("filesystem_enumeration_start")
    files = []
    incomplete = []
    locks = []
    total = 0
    for p in sorted(PARAMS.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            total += size
            files.append({"path": str(p), "size_bytes": size, "mode": oct(p.stat().st_mode & 0o777), "sha256": sha256_file(p) if size <= 16 * 1024 * 1024 else "not_hashed_large_file"})
            if any(token in p.name.lower() for token in ["tmp", "lock", "incomplete", "_tmp"]):
                incomplete.append(str(p))
    metadata = checkpoint_metadata(audit)
    tree = metadata.get("params", metadata)
    rows = []
    for path, value in flatten_tree(tree):
        typ, shape, dtype, sharding = leaf_info(value)
        rows.append({"source": "checkpoint_metadata", "path": path, "type": typ, "shape": json.dumps(shape), "dtype": dtype, "sharding": sharding})
    write_tree_csv(audit.out, rows, "P6G1R1_CHECKPOINT_METADATA_TREE.csv")
    result = {"checkpoint_path": str(CHECKPOINT), "params_path": str(PARAMS), "files_readable": all(pathlib.Path(x["path"]).is_file() for x in files), "metadata_readable": True, "tree_discoverable": bool(rows), "file_count": len(files), "total_bytes": total, "incomplete_or_lock_artifacts": incomplete, "checkpoint_hash_sha256": sha256_path(CHECKPOINT), "metadata_top_keys": list(metadata), "sample_read_bps": sample_read(PARAMS)}
    (audit.out / "P6G1R1_CHECKPOINT_METADATA.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    audit.milestone("filesystem_metadata_complete", **result)
    return result


def sha256_path(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(path)).replace(os.sep, "/").encode())
        h.update(b"\0")
        h.update(sha256_file(p).encode())
        h.update(b"\0")
    return h.hexdigest()


def sample_read(path: pathlib.Path) -> float | None:
    candidates = sorted((p for p in path.rglob("*") if p.is_file() and p.stat().st_size > 0), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        return None
    p = candidates[0]
    n = min(p.stat().st_size, 256 * 1024 * 1024)
    t0 = time.monotonic()
    with p.open("rb") as fh:
        remaining = n
        while remaining:
            block = fh.read(min(8 * 1024 * 1024, remaining))
            if not block:
                break
            remaining -= len(block)
    dt = max(time.monotonic() - t0, 1e-9)
    return float((n - remaining) / dt)


def stage_model_tree(audit: Audit) -> dict[str, Any]:
    import jax
    from flax import nnx
    metadata = checkpoint_metadata(audit)
    ckpt_tree = metadata.get("params", metadata)
    audit.milestone("abstract_model_tree_start")
    _, _, config = import_openpi()
    train_config = config.get_config(POLICY_CONFIG)
    abstract_model = nnx.eval_shape(train_config.model.create, jax.random.key(0))
    _, state = nnx.split(abstract_model)
    expected_tree = state.to_pure_dict()
    rows = []
    for path, value in flatten_tree(expected_tree):
        typ, shape, dtype, sharding = leaf_info(value)
        rows.append({"source": "model_abstract_tree", "path": path, "type": typ, "shape": json.dumps(shape), "dtype": dtype, "sharding": sharding})
    write_tree_csv(audit.out, rows, "P6G1R1_MODEL_TREE.csv")
    ckpt_map = {normalize_checkpoint_path(p): v for p, v in flatten_tree(ckpt_tree)}
    exp_map = {p: v for p, v in flatten_tree(expected_tree)}
    diff = []
    for path in sorted(set(ckpt_map) | set(exp_map)):
        c = ckpt_map.get(path); e = exp_map.get(path)
        _, cs, cd, csh = leaf_info(c) if c is not None else ("", [], "", "")
        _, es, ed, esh = leaf_info(e) if e is not None else ("", [], "", "")
        if c is None or e is None or cs != es or cd != ed:
            kind = "missing_checkpoint" if c is None else "extra_checkpoint" if e is None else "shape_mismatch" if cs != es else "expected_restore_dtype_cast"
            diff.append({"path": path, "checkpoint_shape": json.dumps(cs), "model_shape": json.dumps(es), "checkpoint_dtype": cd, "model_dtype": ed, "checkpoint_sharding": csh, "model_sharding": esh, "kind": kind})
    with (audit.out / "P6G1R1_CHECKPOINT_TREE_DIFF.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["path", "checkpoint_shape", "model_shape", "checkpoint_dtype", "model_dtype", "checkpoint_sharding", "model_sharding", "kind"]); wr.writeheader(); wr.writerows(diff)
    structural = [x for x in diff if x["kind"] != "expected_restore_dtype_cast"]
    dtype_only = [x for x in diff if x["kind"] == "expected_restore_dtype_cast"]
    result = {"expected_leaf_count": len(exp_map), "checkpoint_leaf_count": len(ckpt_map), "mismatch_count": len(diff), "structural_mismatch_count": len(structural), "expected_dtype_cast_count": len(dtype_only), "model_tree_match": not structural, "official_restore_dtype": "bfloat16"}
    audit.milestone("abstract_model_tree_complete", **result)
    return result


def parameter_checksum(policy: Any) -> str:
    from flax import nnx
    state = nnx.state(policy._model).to_pure_dict()
    h = hashlib.sha256()
    for path, value in flatten_tree(state):
        import numpy as np
        arr = np.asarray(value)
        h.update(path.encode()); h.update(str(arr.shape).encode()); h.update(str(arr.dtype).encode()); h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def stage_restore(audit: Audit) -> dict[str, Any]:
    policy, _ = make_policy(audit)
    audit.milestone("parameter_checksum_start")
    checksum = parameter_checksum(policy)
    result = {"restore_completed": True, "parameter_checksum_sha256": checksum, "metadata": getattr(policy, "metadata", {})}
    (audit.out / f"P6G1R1_{audit.stage.upper()}_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    audit.milestone("parameter_checksum_complete", **result)
    return result


def stage_inference(audit: Audit) -> dict[str, Any]:
    import numpy as np
    import jax
    policy, train_config = make_policy(audit)
    audit.milestone("synthetic_model_observation_start")
    fake_obs = train_config.model.fake_obs(batch_size=1)
    out = policy._sample_actions(jax.random.key(0), fake_obs)
    arr = np.asarray(out)
    arr2 = np.asarray(policy._sample_actions(jax.random.key(0), fake_obs))
    finite = bool(np.isfinite(arr).all())
    repeatable = bool(np.array_equal(arr, arr2))
    np.save(audit.out / "P6G1R1_INFERENCE_OUTPUT.npy", arr)
    result = {"forward_pass_completed": True, "output_shape": list(arr.shape), "output_dtype": str(arr.dtype), "nan_or_inf": not finite, "repeatable_fixed_seed": repeatable, "parameter_checksum_sha256": parameter_checksum(policy)}
    (audit.out / "P6G1R1_INFERENCE_SMOKE.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    audit.milestone("inference_complete", **result)
    return result


def isaac_bootstrap(audit: Audit, restore_policy: bool = False) -> dict[str, Any]:
    if restore_policy:
        make_policy(audit)
        audit.milestone("policy_restore_before_isaac_complete")
    audit.milestone("isaac_import_start")
    # Match the non-interactive Isaac/Kit environment used by the successful
    # P6-G1 worker. This only acknowledges the already accepted local EULA;
    # it does not change simulation or policy semantics.
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=True, num_envs=1)
    simulation_app = app_launcher.app
    audit.milestone("app_launcher_complete")
    env = None
    result: dict[str, Any] = {"app_started": True, "reset_completed": False, "step_completed": False}
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        import torch
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        audit.milestone("isaac_python_imports_start")
        spec = importlib.util.spec_from_file_location("p6g1_successful_for_isaac", SUCCESSFUL_P6G1_RUNNER)
        mod = importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(mod)
        p6 = mod.import_p6g0()
        p4 = p6.import_p4_probe(1)
        p6.imported_p4 = p4
        audit.milestone("isaac_python_imports_complete")
        setup_task_objects(mod.TASK_SUITE, 1)
        cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 30.0
        try:
            cfg.sim.physx.enable_ccd = True
        except Exception:
            pass
        audit.milestone("isaac_env_make_start")
        env = gym.make(ENV_ID, cfg=cfg).unwrapped
        audit.milestone("isaac_env_make_complete")
        obs, info = env.reset(seed=1)
        result["reset_completed"] = True
        audit.milestone("isaac_reset_complete")
        result["observation_type"] = str(type(obs))
        result["action_space"] = str(getattr(env, "action_space", None))
        action_dim = int(env.action_space.shape[-1])
        env.step(torch.zeros((1, action_dim), device=env.device, dtype=torch.float32))
        result["step_completed"] = True
        audit.milestone("isaac_step_complete")
    finally:
        # Kit may terminate the interpreter while SimulationApp.close() is
        # unwinding. Persist the smoke result before asking Kit to shut down.
        audit.milestone("isaac_preclose_complete", **result)
        (audit.out / f"P6G1R1_{audit.stage.upper()}_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
        if audit.stage == "isaac-only":
            (audit.out / "P6G1R1_ISAAC_SMOKE.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
        (audit.out / f"P6G1R1_{audit.stage.upper()}_COMPLETE.json").write_text(json.dumps({"stage": audit.stage, "ok": bool(result.get("step_completed")), "result": result, "timestamp_utc": now()}, indent=2, sort_keys=True, default=str) + "\n")
        if env is not None:
            env.close()
        simulation_app.close()
    audit.milestone("isaac_bootstrap_complete", **result)
    return result


def runtime_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {"timestamp_utc": now(), "python": sys.version, "executable": sys.executable, "cwd": os.getcwd(), "env": dict(os.environ), "gpu": gpu_status(), "host_mem": read_meminfo(), "checkpoint": str(CHECKPOINT), "checkpoint_bytes": sum(p.stat().st_size for p in CHECKPOINT.rglob("*") if p.is_file()), "checkpoint_file_count": sum(1 for p in CHECKPOINT.rglob("*") if p.is_file()), "filesystem": shutil.disk_usage(CHECKPOINT.anchor)._asdict()}
    for module_name in ["jax", "jaxlib", "orbax.checkpoint", "flax", "tensorstore"]:
        try:
            mod = __import__(module_name)
            out.setdefault("versions", {})[module_name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            out.setdefault("versions", {})[module_name] = f"IMPORT_ERROR:{exc!r}"
    try:
        import jax
        out["jax_backend"] = jax.default_backend()
        out["jax_devices"] = [str(x) for x in jax.devices()]
        out["jax_local_device_count"] = jax.local_device_count()
    except Exception as exc:
        out["jax_error"] = repr(exc)
    return out


def run_joint_child(audit: Audit, label: str, argv: list[str], env: dict[str, str], cwd: pathlib.Path) -> dict[str, Any]:
    log_path = audit.out / f"P6G1R1_JOINT_{label}.log"
    audit.milestone("joint_child_start", label=label, argv=argv, cwd=str(cwd), log=str(log_path))
    t0 = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(argv, cwd=str(cwd), env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=1800)
        result = {"label": label, "returncode": proc.returncode, "duration_s": time.monotonic() - t0, "log": str(log_path), "ok": proc.returncode == 0}
    except Exception as exc:
        result = {"label": label, "duration_s": time.monotonic() - t0, "log": str(log_path), "ok": False, "exception": repr(exc)}
    audit.milestone("joint_child_complete", **result)
    return result


def stage_joint_runtime(audit: Audit) -> dict[str, Any]:
    server_python = "/media/volume/newdata/exouser/tabero/Tabero-VTLA/.venv/bin/python"
    isaac_python = str(ISAAC_PY)
    stubs = str(ROOT / "analysis/results/b5_tabero_neutral_20260822_040652/scripts/stubs")
    server_env = os.environ.copy()
    server_env.update({"CUDA_VISIBLE_DEVICES": "0", "PYTHONNOUSERSITE": "1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONPATH": os.pathsep.join([stubs, str(VTLA / "src"), str(VTLA / "packages/openpi-client/src")])})
    isaac_env = os.environ.copy()
    isaac_env.pop("CUDA_VISIBLE_DEVICES", None)
    isaac_env.update({"OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PYTHONNOUSERSITE": "1", "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(ROOT), str(OPENPI_CLIENT_SRC)]), "TABERO_ROOT": str(ROOT), "HDF5_TRAJ_SOURCE_DIR": str(ROOT / "benchmarks/datasets/libero/assembled_hdf5"), "LIBERO_CONFIG_DIR": str(ROOT / "benchmarks/datasets/libero/config"), "LIBERO_ASSETS_DATA_DIR": str(ROOT / "benchmarks/datasets/libero/USD")})
    script = str(pathlib.Path(__file__).resolve())
    order1 = [
        run_joint_child(audit, "ORDER1_RESTORE_THEN_ISAAC_RESTORE", [server_python, "-u", script, "--stage", "restore-gpu", "--out", str(audit.out / "joint_order1_restore")], server_env, VTLA),
        run_joint_child(audit, "ORDER1_RESTORE_THEN_ISAAC_ISAAC", [isaac_python, "-u", script, "--stage", "isaac-only", "--out", str(audit.out / "joint_order1_isaac")], isaac_env, ROOT),
    ]
    order2 = [
        run_joint_child(audit, "ORDER2_ISAAC_THEN_RESTORE_ISAAC", [isaac_python, "-u", script, "--stage", "isaac-only", "--out", str(audit.out / "joint_order2_isaac")], isaac_env, ROOT),
        run_joint_child(audit, "ORDER2_ISAAC_THEN_RESTORE_RESTORE", [server_python, "-u", script, "--stage", "restore-gpu", "--out", str(audit.out / "joint_order2_restore")], server_env, VTLA),
    ]
    result = {"topology": "separate authoritative server and Isaac worker processes", "order1_restore_then_isaac": order1, "order2_isaac_then_restore": order2, "order1_qualified": all(x["ok"] for x in order1), "order2_qualified": all(x["ok"] for x in order2), "qualified": all(x["ok"] for x in order1 + order2), "scientific_semantics_changed": False}
    (audit.out / "P6G1R1_JOINT_RUNTIME_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    (audit.out / "P6G1R1_JOINT_RUNTIME_LOG.txt").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["metadata", "model-tree", "restore-cpu", "restore-gpu", "inference", "isaac-only", "joint-runtime"])
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    audit = Audit(args.out, args.stage)
    audit.start()
    try:
        if args.stage == "metadata":
            result = stage_metadata(audit)
        elif args.stage == "model-tree":
            result = stage_model_tree(audit)
        elif args.stage in {"restore-cpu", "restore-gpu"}:
            result = stage_restore(audit)
        elif args.stage == "inference":
            result = stage_inference(audit)
        elif args.stage == "isaac-only":
            result = isaac_bootstrap(audit, restore_policy=False)
        elif args.stage == "joint-runtime":
            result = stage_joint_runtime(audit)
        else:
            result = isaac_bootstrap(audit, restore_policy=True)
        audit.finish(True, result=result)
        return 0
    except BaseException as exc:
        audit.milestone("exception", exception=repr(exc), traceback=traceback.format_exc())
        audit.finish(False, exception=repr(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
