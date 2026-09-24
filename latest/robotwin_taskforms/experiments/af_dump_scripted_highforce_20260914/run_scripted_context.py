"""Pure scripted dump_bin_bigbin × multi-force collection.

Claim: controlled-motion / scripted full-task feasibility labels.
Uses repository play_once(); only af_force_limit_n (single-finger N) and
friction change across branches. No π₀ / no open-loop VLA chunks.
"""
from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def run_force(seed: int, friction: float, force_n: float, episode_id: int) -> dict:
    import sys
    from pathlib import Path as _Path

    repo = _Path(__file__).resolve().parents[2] / "RoboTwin"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from scripts.eval_policy_xpolicylab import class_decorator, load_task_args

    user_args = {
        "task_name": "dump_bin_bigbin",
        "task_config": "demo_clean",
        "policy_name": "expert",
        "ckpt_setting": "scripted-highforce",
        "activeforcing_enabled": True,
        "af_dynamic_evaluator": True,
        "af_contact_friction": float(friction),
        "af_force_limit_n": float(force_n),
    }
    task_args, _ = load_task_args(user_args)
    task_args["eval_mode"] = True
    task_args["render_freq"] = 0
    task = class_decorator("dump_bin_bigbin")
    try:
        task.setup_demo(now_ep_num=episode_id, seed=int(seed), is_test=True, **task_args)
        if float(task.af_contact_friction) != float(friction):
            raise RuntimeError("Friction not applied")
        if float(task.af_force_limit_n) != float(force_n):
            raise RuntimeError("Force limit not applied")
        # Apply single-finger drive limits on both grippers before scripted motion.
        task.activate_activeforcing_candidate_force()
        task.play_once()
        success = bool(task.check_success()) and bool(task.plan_success)
        metrics = task.compute_activeforcing_dynamic_metrics()
        return {
            "completed": True,
            "success": int(success),
            "plan_success": bool(task.plan_success),
            "official_final_check": bool(task.check_success()),
            "force_setpoint_single_finger_N": float(force_n),
            "friction": float(friction),
            "seed": int(seed),
            "left_force_limit_n": metrics.get("left_force_limit_n"),
            "right_force_limit_n": metrics.get("right_force_limit_n"),
            "contact_ratio": metrics.get("contact_ratio"),
            "measured_force_mean_n": metrics.get("measured_force_mean_n"),
            "measured_force_p95_n": metrics.get("measured_force_p95_n"),
            "samples": metrics.get("samples"),
            "irrecoverable_failure": metrics.get("irrecoverable_failure"),
            "motion_mode": "scripted_play_once",
            "finished_utc": now(),
        }
    except Exception as exc:
        unstable = type(exc).__name__ == "UnStableError" or "UnStableError" in repr(exc)
        return {
            "completed": True if unstable else False,
            "success": 0,
            "unstable_layout": bool(unstable),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "force_setpoint_single_finger_N": float(force_n),
            "friction": float(friction),
            "seed": int(seed),
            "motion_mode": "scripted_play_once",
            "finished_utc": now(),
        }
    finally:
        try:
            task.close_env()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    context = json.loads(args.context.read_text())["context"]
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    write(out / "CONTEXT.json", context)
    outcomes = []
    for index, force in enumerate(context["forces_N"]):
        branch = out / f"branch_{index}_{float(force):g}N"
        branch.mkdir()
        result = run_force(
            seed=int(context["seed"]),
            friction=float(context["friction"]),
            force_n=float(force),
            episode_id=index,
        )
        write(branch / "result.json", result)
        if result.get("unstable_layout"):
            outcomes.append(
                {
                    "method": f"Scripted-{float(force):g}N",
                    "force_N": float(force),
                    "success": 0,
                    "plan_success": False,
                    "unstable_layout": True,
                    "contact_ratio": None,
                    "measured_force_mean_n": None,
                    "measured_force_p95_n": None,
                    "samples": None,
                    "left_force_limit_n": float(force),
                    "right_force_limit_n": float(force),
                }
            )
            continue
        if not result.get("completed"):
            raise RuntimeError(f"Scripted branch failed: {branch} {result.get('error')}")
        if float(result["left_force_limit_n"]) != float(force) or float(result["right_force_limit_n"]) != float(force):
            raise RuntimeError("Force limit telemetry mismatch")
        outcomes.append(
            {
                "method": f"Scripted-{float(force):g}N",
                "force_N": float(force),
                "success": int(result["success"]),
                "plan_success": bool(result["plan_success"]),
                "unstable_layout": False,
                "contact_ratio": result["contact_ratio"],
                "measured_force_mean_n": result["measured_force_mean_n"],
                "measured_force_p95_n": result["measured_force_p95_n"],
                "samples": result["samples"],
                "left_force_limit_n": result["left_force_limit_n"],
                "right_force_limit_n": result["right_force_limit_n"],
            }
        )
    write(
        out / "SCRIPTED_CONTEXT_RESULT.json",
        {
            "completed": True,
            "context": context,
            "outcomes": outcomes,
            "paired_rollouts": len(outcomes),
            "motion_mode": "scripted_play_once",
            "force_convention": "RoboTwin AF single-finger drive limit N (both grippers)",
            "claim_boundary": "scripted controlled-motion auxiliary labels; not online AF efficacy",
            "finished_utc": now(),
        },
    )


if __name__ == "__main__":
    main()
