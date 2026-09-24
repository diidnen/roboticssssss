"""Fresh-process V5 hybrid A3 tracking branch using prior-step feedback only."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

CODE = Path("/media/volume/data/exouser/activeforcing_table_push_v5_20260910/code")
sys.path.insert(0, str(CODE))
from r2h_branch import HybridForceAdapter, V2, array_hash, build_env, contact_force

OUT = CODE.parent
CONTRACT = json.loads((OUT / "A3_TRACKING_CONTRACT.json").read_text())


def atomic_json(path, value):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    os.close(fd)
    Path(tmp).write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


class CausalAbsoluteForcePI:
    def __init__(self, target):
        self.target = float(target)
        self.ema = None
        self.integral = 0.0
        self.command = None
        self.contact_losses = 0

    def observe_completed_step(self, measured_force, contact):
        if not contact:
            self.ema = None
            self.integral = 0.0
            self.command = None
            self.contact_losses += 1
            return
        alpha = float(CONTRACT["ema_alpha"])
        self.ema = measured_force if self.ema is None else alpha * measured_force + (1 - alpha) * self.ema
        error = self.target - self.ema
        proposed_integral = float(
            np.clip(self.integral + error, -CONTRACT["integral_cap"], CONTRACT["integral_cap"])
        )
        raw = self.target + CONTRACT["kp"] * error + CONTRACT["ki"] * proposed_integral
        low, high = CONTRACT["axis_command_limits_n"]
        clipped = float(np.clip(raw, low, high))
        if clipped == raw or (clipped == low and error > 0) or (clipped == high and error < 0):
            self.integral = proposed_integral
        if self.command is None:
            self.command = clipped
        else:
            ramp = float(CONTRACT["ramp_n"])
            self.command = float(np.clip(clipped, self.command - ramp, self.command + ramp))


def main():
    target = float(sys.argv[1])
    feedback = sys.argv[2] == "feedback"
    mode = "feedback" if feedback else "baseline"
    out = OUT / "tracking" / f"target_{target:.1f}_{mode}.json"
    if out.exists():
        raise RuntimeError(f"immutable tracking branch exists: {out}")
    gate = json.loads((OUT / "R2H_GATE_DECISION.json").read_text())
    if not gate["pass"]:
        raise RuntimeError("R2H physical gate did not pass")

    telemetry = [json.loads(line) for line in (V2 / "telemetry.jsonl").read_text().splitlines()]
    executed = np.asarray([row["executed_action"] for row in telemetry], dtype=float)
    nominal = np.asarray([row["nominal_action"] for row in telemetry], dtype=float)
    receipt = json.loads((V2 / "receipt.json").read_text())
    direction_xy = np.asarray(receipt["direction_record"]["direction_world_xy"], dtype=float)
    env, obs, table = build_env()
    for action in executed[:57]:
        obs, _, _, _ = env.step(action.tolist())

    controller = CausalAbsoluteForcePI(target)
    adapter = None
    rows = []
    for k in range(int(CONTRACT["horizon"])):
        command = None
        intervention = False
        if feedback and k > 0 and controller.command is not None:
            command = float(controller.command)
            if adapter is None:
                adapter = HybridForceAdapter(env, direction_xy, True, command)
            adapter.enabled = True
            adapter.axis_command_n = command
            adapter.trace = []
            intervention = True
        elif adapter is not None:
            adapter.enabled = False
            adapter.trace = []

        post_obs, reward, done, info = env.step(nominal[57 + k].tolist())
        post_force, identities, raw_contacts = contact_force(env.sim.model, env.sim.data)
        measured = float(post_force[:2] @ direction_xy)
        trace = [] if adapter is None else list(adapter.trace)
        rows.append(
            {
                "k": k,
                "action_index": 57 + k,
                "causal_measurement_index": None if k == 0 else k - 1,
                "first_action_untouched": bool(k != 0 or not intervention),
                "intervention": intervention,
                "absolute_axis_command_n": command,
                "post_aligned_force_n": measured,
                "contact": bool(identities),
                "contact_identities": identities,
                "raw_contacts": raw_contacts,
                "nominal_action_sha256": array_hash(nominal[57 + k]),
                "no_replan": True,
                "adapter_trace": trace,
                "reward": float(reward),
                "done": bool(done),
            }
        )
        if feedback:
            controller.observe_completed_step(measured, bool(identities))

    if adapter is not None:
        adapter.close()
    env.close()
    last_five = rows[-5:]
    retained = [row for row in last_five if row["contact"]]
    document = {
        "gate": "A3_TRACKING",
        "target_n": target,
        "feedback": feedback,
        "mode": mode,
        "rows": rows,
        "realized_last5_mean_n": float(np.mean([row["post_aligned_force_n"] for row in retained])),
        "mae_last5_n": float(np.mean([abs(row["post_aligned_force_n"] - target) for row in retained])),
        "contact_losses": sum(not row["contact"] for row in rows),
        "first_action_untouched": rows[0]["first_action_untouched"] and not rows[0]["intervention"],
        "all_nominal_actions_fixed": all(row["no_replan"] for row in rows),
        "new_task_outcomes": 0,
    }
    atomic_json(out, document)


if __name__ == "__main__":
    main()
