"""Open-loop high-force collection: record one π₀ trajectory, replay across forces.

Claim boundary: controlled-motion / auxiliary force labels. Arm actions are
identical across force branches; only the bilateral force setpoint changes.
Not online-VLA closed-loop evidence.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np

import qualify_native_interfaces as base
from native_cartesian_kinematics import NativeCartesianKinematics
from native_original_force_controller import NativeOriginalForceController
from native_original_motion_features import build_features
from native_visual_mirror import (
    capture_visual_packet,
    native_observation_from_packet,
    observation_identity,
)
from qualify_original_p4_native import qualify as qualify_query


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def child_run(env, pipe, out, force, handoff, stock_drives, support, scope):
    controller = NativeOriginalForceController(
        env, env._af_grasp_arm_tag, force, handoff, stock_drives, support
    )
    env._update_render = lambda: None
    env.eval_video_path = None
    env.render_freq = 0
    original_step = env.scene.step
    trace_hash = hashlib.sha256()
    physics_steps = 0
    forces = []
    chunk_count = 0

    def rpc(kind):
        pipe.send({"kind": kind, "packet": capture_visual_packet(env)})
        response = pipe.recv()
        if response["kind"] == "error":
            raise RuntimeError(response["error"])
        return response

    def get_obs(*a, **kw):
        observation = rpc("observation")["observation"]
        env.now_obs = observation
        return observation

    env.get_obs = get_obs
    controller.install()
    try:
        with gzip.open(out / "physics_trace.jsonl.gz", "wt") as trace:

            def step():
                nonlocal physics_steps
                controller.before_physics()
                original_step()
                contact = base.contacts(env, env._af_grasp_arm_tag)
                measured = contact["measured_squeeze_n"]
                forces.append(measured)
                physics_steps += 1
                row = {
                    "physics_step": physics_steps,
                    "contact": contact,
                    "state": base.readback(env),
                    "original_squeeze_inner": controller.inner.last if controller.inner is not None else None,
                }
                encoded = json.dumps(row, sort_keys=True, separators=(",", ":"))
                trace_hash.update(encoded.encode())
                trace.write(encoded + "\n")
                controller.after_physics(measured)

            env.scene.step = step
            while not env.eval_success and env.take_action_cnt < env.step_lim:
                response = rpc("chunk")
                actions = np.asarray(response["actions"], np.float32)
                if actions.ndim != 2 or actions.shape[1] != 14 or not len(actions):
                    raise RuntimeError("Invalid chunk")
                np.save(out / f"chunk_{chunk_count:03d}.npy", actions)
                chunk_count += 1
                for action in actions:
                    controller.before_action(action)
                    env.take_action(action.copy(), action_type="qpos")
                    if env.eval_success or env.take_action_cnt >= env.step_lim:
                        break
        result = {
            "completed": True,
            "success": bool(env.eval_success),
            "official_final_check": bool(env.check_success()),
            "native_actions": env.take_action_cnt,
            "native_horizon": env.step_lim,
            "physics_steps": physics_steps,
            "trace_sha256": trace_hash.hexdigest(),
            "force_setpoint_bilateral_n": force,
            "measured_mean_squeeze_n": float(np.mean(forces)) if forces else None,
            "measured_max_squeeze_n": float(np.max(forces)) if forces else None,
            "release_actions": sum(r["vla_release_intent"] for r in controller.action_receipts),
            "controller_feedback_ticks": controller.feedback_ticks,
            "chunks": chunk_count,
            "arbitration_binding": controller.binding_receipt,
            "squeeze_controller_binding": (
                "ORIGINAL_OUTER_AND_ORIGINAL_INNER_POSITION_FEEDBACK"
                if controller.inner
                else "ORIGINAL_OUTER_ONLY_NATIVE_FORCE_CAP_DIAGNOSTIC"
            ),
            "old_irrecoverable_shortcut_used": False,
            "scope": scope,
            "motion_mode": "openloop_replay_or_record",
        }
        write_json(out / "arbitration.json", controller.action_receipts)
        write_json(out / "result.json", result)
        pipe.send({"kind": "done", "result": result})
    finally:
        controller.uninstall()
        env.scene.step = original_step


def _fork_workers(env, specs, handoff, stock, support):
    """Fork all workers from one clean handoff before any child mutates shared physics."""
    initial = base.readback(env)
    initial_hash = base.digest(initial)
    initial_count = env.take_action_cnt
    workers = []
    for spec in specs:
        branch = spec["path"]
        branch.mkdir(parents=True, exist_ok=False)
        parent, child = mp.Pipe()
        pid = os.fork()
        if pid == 0:
            parent.close()
            for old in workers:
                old["pipe"].close()
            try:
                command = child.recv()
                if command != "start":
                    raise RuntimeError("Expected start")
                child_run(env, child, branch, spec["force"], handoff, stock, support, spec["scope"])
                os._exit(0)
            except BaseException as exc:
                error = {"completed": False, "error": repr(exc), "traceback": traceback.format_exc()}
                write_json(branch / "error.json", error)
                child.send({"kind": "error", **error})
                os._exit(2)
        child.close()
        workers.append(
            {
                "pid": pid,
                "pipe": parent,
                "force": spec["force"],
                "path": branch,
                "index": spec["index"],
                "role": spec["role"],
            }
        )
        if base.readback(env) != initial or env.take_action_cnt != initial_count:
            raise RuntimeError("Fork changed shared source boundary")
    return workers, initial_hash


def _serve_worker(env, worker, initial_hash, chunk_provider, first_chunk_hashes):
    receipts = []
    result = None
    worker["pipe"].send("start")
    deadline = time.monotonic() + 1800
    chunk_count = 0
    while time.monotonic() < deadline:
        if not worker["pipe"].poll(0.2):
            ended, status = os.waitpid(worker["pid"], os.WNOHANG)
            if ended:
                worker["pid"] = None
                raise RuntimeError("Worker exited without terminal receipt: " + str(status))
            continue
        message = worker["pipe"].recv()
        if message["kind"] in ("done", "error"):
            result = message
            break
        packet = message["packet"]
        observation = native_observation_from_packet(env, packet)
        if message["kind"] == "observation":
            worker["pipe"].send({"kind": "observation", "observation": observation})
            continue
        if message["kind"] != "chunk":
            raise RuntimeError("Unknown worker request")
        if chunk_count == 0 and base.digest(packet["state"]) != initial_hash:
            raise RuntimeError("Candidate changed state before first policy chunk")
        actions = chunk_provider.next_actions(worker, chunk_count, observation, env, worker["path"].parent, packet)
        identity = hashlib.sha256(actions.tobytes()).hexdigest()
        receipts.append(
            {
                "chunk": chunk_count,
                "actions_sha256": identity,
                "observation": observation_identity(observation),
            }
        )
        if chunk_count == 0:
            first_chunk_hashes.append(identity)
            if identity != first_chunk_hashes[0]:
                raise RuntimeError("Strict first-chunk pairing failed")
        worker["pipe"].send({"kind": "actions", "actions": actions})
        chunk_count += 1
    if result is None:
        raise TimeoutError("Bounded rollout exceeded 30 minutes")
    write_json(worker["path"] / "policy_receipts.json", receipts)
    ended, _ = os.waitpid(worker["pid"], 0)
    worker["pid"] = None
    worker["pipe"].close()
    return result, chunk_count


def _cleanup_workers(workers):
    for worker in workers:
        if worker.get("pid") is not None:
            ended, _ = os.waitpid(worker["pid"], os.WNOHANG)
            if not ended:
                os.kill(worker["pid"], signal.SIGTERM)
                os.waitpid(worker["pid"], 0)
            worker["pid"] = None
        pipe = worker.get("pipe")
        if pipe is not None:
            try:
                pipe.close()
            except Exception:
                pass
            worker["pipe"] = None


class OnlineRecordProvider:
    mode = "record_online_pi0"

    def __init__(self, policy_seed: int, instruction: str, raw, kin, lock_dir: Path, query_out: Path):
        self.policy_seed = policy_seed
        self.instruction = instruction
        self.raw = raw
        self.kin = kin
        self.lock_dir = lock_dir
        self.query_out = query_out
        self.chunks = []
        self._client = None
        from scripts.eval_policy_xpolicylab import (
            build_policy_client,
            close_policy_client,
            normalize_action_chunk,
            robotwin_obs_to_xpolicylab,
            xpolicylab_action_to_robotwin,
        )

        self._build = build_policy_client
        self._close = close_policy_client
        self._normalize = normalize_action_chunk
        self._to_x = robotwin_obs_to_xpolicylab
        self._to_rw = xpolicylab_action_to_robotwin

    def __enter__(self):
        self._client = self._build(
            {
                "protocol": "ws",
                "host": "localhost",
                "port": 6001,
                "evaluation_id": "openloop-record",
                "trial_id": "record",
            }
        )
        self._client.call(
            func_name="prepare_case",
            obs={
                "task_name": "dump_bin_bigbin",
                "seed": 200002,
                "policy_seed": self.policy_seed,
                "instruction": self.instruction,
                "action_type": "joint",
            },
        )
        self._client.call(func_name="reset")
        return self

    def __exit__(self, *exc):
        if self._client is not None:
            self._close(self._client)
            self._client = None
        if self.chunks:
            stacked = np.stack(self.chunks, axis=0)
            np.save(self.lock_dir / "LOCKED_CHUNKS.npy", stacked)
            write_json(
                self.lock_dir / "LOCKED_TRAJECTORY_RECEIPT.json",
                {
                    "n_chunks": len(self.chunks),
                    "chunk_shape": list(self.chunks[0].shape),
                    "sha256": hashlib.sha256(stacked.tobytes()).hexdigest(),
                    "policy_seed": self.policy_seed,
                    "mode": self.mode,
                },
            )

    def next_actions(self, worker, chunk_count, observation, env, out, packet):
        payload = self._to_x(observation, instruction=self.instruction, env_idx=0, frequency=30, task_env=env)
        self._client.call(func_name="update_obs", obs=payload)
        response = self._normalize(self._client.call(func_name="get_action"))
        actions = []
        for action in response:
            flat, kind = self._to_rw(action, action_type="joint", current_observation=observation)
            if kind != "qpos":
                raise RuntimeError("Native action type changed")
            actions.append(flat)
        actions = np.asarray(actions, np.float32)
        self.chunks.append(actions.copy())
        if chunk_count == 0:
            import sapien

            body = env.deskbin.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            patch = json.loads((self.query_out / "patch_readbacks.json").read_text())[-1]
            feature = build_features(
                self.raw,
                patch["finger_joints"],
                np.asarray(body.linear_velocity),
                self.kin.future_xyz_base(actions),
            )
            write_json(worker["path"] / "original_motion_feature.json", feature)
            write_json(self.lock_dir / "PREACTION_FEATURE.json", feature)
        return actions


class OpenLoopReplayProvider:
    mode = "openloop_replay_fixed_chunks"

    def __init__(self, lock_dir: Path):
        path = lock_dir / "LOCKED_CHUNKS.npy"
        arr = np.load(path)
        if arr.ndim != 3:
            raise ValueError("LOCKED_CHUNKS must be [n_chunks, T, 14]")
        self.chunks = [np.asarray(arr[i], np.float32) for i in range(arr.shape[0])]
        receipt = json.loads((lock_dir / "LOCKED_TRAJECTORY_RECEIPT.json").read_text())
        stacked = np.stack(self.chunks, axis=0)
        if hashlib.sha256(stacked.tobytes()).hexdigest() != receipt["sha256"]:
            raise ValueError("Locked trajectory changed")
        feature = json.loads((lock_dir / "PREACTION_FEATURE.json").read_text())
        self.feature = feature
        self._hold_chunks = 0

    def next_actions(self, worker, chunk_count, observation, env, out, packet):
        if chunk_count < len(self.chunks):
            actions = np.asarray(self.chunks[chunk_count], np.float32)
            if chunk_count == 0:
                write_json(worker["path"] / "original_motion_feature.json", self.feature)
            return actions
        # Prior successful sources are short (~60 actions). On failing forces,
        # hold final qpos out to the native horizon.
        self._hold_chunks += 1
        last = np.asarray(self.chunks[-1][-1], np.float32)
        return np.tile(last[None, :], (self.chunks[-1].shape[0], 1))


def import_prior_successful_trajectory(lock_dir: Path, prior_branch: Path) -> dict:
    """Lock arm chunks from a previously successful fixed-force branch (no pi0)."""
    prior_branch = Path(prior_branch)
    result = json.loads((prior_branch / "result.json").read_text())
    if not result.get("success"):
        raise ValueError("Prior branch is not a success: " + str(prior_branch))
    chunks = sorted(prior_branch.glob("chunk_*.npy"))
    if not chunks:
        raise ValueError("Prior branch has no chunk_*.npy")
    arrays = [np.asarray(np.load(path, allow_pickle=False), np.float32) for path in chunks]
    if len({a.shape for a in arrays}) != 1 or arrays[0].ndim != 2 or arrays[0].shape[1] != 14:
        raise ValueError("Prior chunks have incompatible shapes")
    stacked = np.stack(arrays, axis=0)
    digest = hashlib.sha256(stacked.tobytes()).hexdigest()
    lock_dir.mkdir(parents=True, exist_ok=True)
    np.save(lock_dir / "LOCKED_CHUNKS.npy", stacked)
    feature_src = prior_branch / "original_motion_feature.json"
    if feature_src.exists():
        feature = json.loads(feature_src.read_text())
    else:
        feature = {"imported_without_feature": True, "prior_branch": str(prior_branch)}
    write_json(lock_dir / "PREACTION_FEATURE.json", feature)
    receipt = {
        "n_chunks": int(stacked.shape[0]),
        "chunk_shape": list(stacked.shape[1:]),
        "sha256": digest,
        "mode": "import_prior_successful_fixed_force",
        "prior_branch": str(prior_branch),
        "prior_force_N": float(result["force_setpoint_bilateral_n"]),
        "prior_native_actions": int(result["native_actions"]),
        "prior_success": True,
        "first_chunk_sha256": hashlib.sha256(arrays[0].tobytes()).hexdigest(),
    }
    write_json(lock_dir / "LOCKED_TRAJECTORY_RECEIPT.json", receipt)
    return receipt


def execute_openloop_context(env, out: Path, *, forces, record_force, support, policy_seed, prior_branch=None):
    out.mkdir(parents=True, exist_ok=False)
    stock = [
        (j, j.stiffness, j.damping, j.force_limit, j.drive_mode)
        for j, _, _ in getattr(env.robot, env._af_grasp_arm_tag + "_gripper")
    ]
    qualify_query(env, out / "query")
    report = json.loads((out / "query/qualification.json").read_text())
    if not report.get("completed") or report.get("probe_failure"):
        raise RuntimeError("Original P4 did not establish qualified query contact")
    import sapien
    from admit_original_query import admit

    body = env.deskbin.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    write_json(
        out / "query/ACTUAL_OBJECT_MATERIAL.json",
        {
            "shape_materials": [
                [float(s.physical_material.static_friction), float(s.physical_material.dynamic_friction)]
                for s in body.collision_shapes
            ]
        },
    )
    write_json(out / "QUERY_ADMISSION.json", admit(out / "query", require_material=True))
    controls = json.loads((out / "query/native_controls.json").read_text())
    handoff = float(controls[-1]["original_action13"][6])
    lock_dir = out / "locked_trajectory"
    lock_dir.mkdir()
    scope_replay = "openloop replay; fixed action sequence; force setpoint only"
    force_list = [float(f) for f in forces]

    if not prior_branch:
        raise RuntimeError("prior_branch is required: only import successful prior trajectories")
    receipt = import_prior_successful_trajectory(lock_dir, Path(prior_branch))
    write_json(out / "PRIOR_TRAJECTORY_IMPORT.json", receipt)
    if float(receipt["prior_force_N"]) != float(record_force):
        raise ValueError("Prior trajectory force does not match prospective record_force_N")

    replay_root = out / "replay"
    replay_root.mkdir()
    specs = [
        {
            "role": "replay",
            "index": index,
            "force": force,
            "path": replay_root / f"branch_{index}_{force:g}N",
            "scope": scope_replay,
        }
        for index, force in enumerate(force_list)
    ]
    workers, initial_hash = _fork_workers(env, specs, handoff, stock, support)
    try:
        replay_provider = OpenLoopReplayProvider(lock_dir)
        replay_results = []
        replay_first = []
        for worker in workers:
            result, _chunk_count = _serve_worker(env, worker, initial_hash, replay_provider, replay_first)
            replay_results.append(result)
            if result["kind"] != "done":
                raise RuntimeError("Incomplete open-loop replay")
            raw_result = result["result"]
            arb_n = len(json.loads((worker["path"] / "arbitration.json").read_text()))
            if int(raw_result["native_actions"]) != arb_n:
                raise RuntimeError("Replay action/arbitration count mismatch")
        if len(set(replay_first)) != 1:
            raise RuntimeError("Open-loop first chunks diverged")
        if replay_first[0] != receipt["first_chunk_sha256"]:
            raise RuntimeError("Replay first chunk differs from imported prior trajectory")
        replay_summary = {
            "common_handoff_state_sha256": initial_hash,
            "engineering_force_list_N": force_list,
            "force_support_N": list(support),
            "results": replay_results,
            "first_chunk_hashes": replay_first,
            "motion_mode": "openloop_replay_prior_successful_12N",
            "prior_branch": str(prior_branch),
            "hold_chunks_used": replay_provider._hold_chunks,
        }
        write_json(out / "REPLAY_SUMMARY.json", replay_summary)
    except Exception:
        _cleanup_workers(workers)
        raise

    outcomes = []
    for index, force in enumerate(forces):
        branch = out / "replay" / f"branch_{index}_{float(force):g}N"
        link = out / f"branch_{index}_{float(force):g}N"
        if not link.exists():
            link.symlink_to(branch)
        result = json.loads((branch / "result.json").read_text())
        if float(result["force_setpoint_bilateral_n"]) != float(force):
            raise ValueError("Force setpoint mismatch")
        outcomes.append(
            {
                "method": f"OpenLoop-{float(force):g}N",
                "commanded_force_N": float(force),
                "success": int(result["success"]),
                "measured_mean_squeeze_N": result["measured_mean_squeeze_n"],
                "measured_max_squeeze_N": result["measured_max_squeeze_n"],
                "native_actions": result["native_actions"],
                "result_sha256": hashlib.sha256((branch / "result.json").read_bytes()).hexdigest(),
                "motion_mode": "openloop_replay_fixed_chunks",
            }
        )
    write_json(
        out / "OPENLOOP_CONTEXT_RESULT.json",
        {
            "completed": True,
            "paired_rollouts": len(forces),
            "record_force_N": float(record_force),
            "outcomes": outcomes,
            "locked_trajectory_sha256": json.loads((lock_dir / "LOCKED_TRAJECTORY_RECEIPT.json").read_text())[
                "sha256"
            ],
            "common_handoff_state_sha256": replay_summary["common_handoff_state_sha256"],
            "first_chunk_sha256": replay_summary["first_chunk_hashes"][0],
            "claim_boundary": "controlled-motion auxiliary; prior-successful 12N trajectory; identical arm actions across forces",
        },
    )
    return outcomes
