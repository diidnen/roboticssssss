"""One fresh-process V6 minimum-force interface branch."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

V5_CODE = Path("/media/volume/data/exouser/activeforcing_table_push_v5_20260910/code")
V6_CODE = Path("/media/volume/data/exouser/activeforcing_table_push_v6_20260910/code")
for path in (V5_CODE, V6_CODE):
    sys.path.insert(0, str(path))
from r2h_branch import V2, array_hash, build_env, contact_force
from v6_controller import TorqueSafeMinimumForceAdapter

OUT = V6_CODE.parent


def atomic_json(path, value):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    os.close(fd)
    Path(tmp).write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def main():
    step, kind, command = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
    output = OUT / "r2" / f"s{step}_{kind}_{command:+.0f}N.json"
    if output.exists():
        raise RuntimeError(f"immutable branch exists: {output}")
    telemetry = [json.loads(line) for line in (V2 / "telemetry.jsonl").read_text().splitlines()]
    executed = np.asarray([row["executed_action"] for row in telemetry], dtype=float)
    nominal = np.asarray([row["nominal_action"] for row in telemetry], dtype=float)
    direction = np.asarray(json.loads((V2 / "receipt.json").read_text())["direction_record"]["direction_world_xy"])
    env, obs, table = build_env()
    for action in executed[:step]:
        obs, _, _, _ = env.step(action.tolist())
    pre_state = env.get_sim_state().copy()
    adapter = None
    if kind != "untouched":
        adapter = TorqueSafeMinimumForceAdapter(env, direction, kind == "minimum", command)
    post_obs, reward, done, info = env.step(nominal[step].tolist())
    post_state = env.get_sim_state().copy()
    post_force, contacts, raw_contacts = contact_force(env.sim.model, env.sim.data)
    arm_indices = env.robots[0]._ref_joint_actuator_indexes
    ctrl = env.sim.data.ctrl[arm_indices].copy()
    trace = [] if adapter is None else list(adapter.trace)
    document = {
        "step": step,
        "kind": kind,
        "minimum_force_n": command,
        "pre_state_sha256": array_hash(pre_state),
        "post_state_sha256": array_hash(post_state),
        "post_state": post_state.tolist(),
        "post_qpos": env.sim.data.qpos.copy().tolist(),
        "post_qvel": env.sim.data.qvel.copy().tolist(),
        "nominal_action_sha256": array_hash(nominal[step]),
        "post_aligned_force_n": float(post_force[:2] @ direction),
        "post_contact": bool(contacts),
        "raw_contacts": raw_contacts,
        "final_ctrl": ctrl.tolist(),
        "adapter_trace": trace,
        "finite_torque": all(item.get("finite", False) for item in trace) if trace else True,
        "total_clip_count": sum(item.get("clip_count", 0) for item in trace),
        "never_reduced_stock_axis": all(item.get("applied_axis_force_n", -np.inf) + 1e-12 >= item.get("stock_axis_force_n", np.inf) for item in trace) if kind == "minimum" else True,
    }
    if adapter is not None:
        adapter.close()
    env.close()
    atomic_json(output, document)


if __name__ == "__main__":
    main()
