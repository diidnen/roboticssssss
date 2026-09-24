import argparse
import json

import numpy as np

from scripts.eval_policy_xpolicylab import class_decorator, load_task_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated absolute RoboTwin seeds")
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",")]
    records = []
    for episode_id, seed in enumerate(seeds):
        user_args = {
            "task_name": args.task,
            "task_config": "demo_clean",
            "policy_name": "expert",
            "ckpt_setting": "expert-dynamic-calibration",
            "activeforcing_enabled": True,
            "af_dynamic_evaluator": True,
        }
        task_args, _ = load_task_args(user_args)
        task_args["eval_mode"] = True
        task_args["render_freq"] = 0
        task = class_decorator(args.task)
        try:
            task.setup_demo(now_ep_num=episode_id, seed=seed, is_test=True, **task_args)
            task.play_once()
            record = {
                "task": args.task,
                "seed": seed,
                "plan_success": bool(task.plan_success),
                "native_or_dynamic_success": bool(task.check_success()),
                "metrics": task.compute_activeforcing_dynamic_metrics(),
            }
        except Exception as exc:
            record = {
                "task": args.task,
                "seed": seed,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        finally:
            try:
                task.close_env()
            except Exception:
                pass
        records.append(record)
        print("EXPERT_DYNAMIC_EVIDENCE " + json.dumps(record, sort_keys=True), flush=True)

    metric_names = (
        "position_path_m",
        "position_span_z_m",
        "angular_path_rad",
        "angular_excursion_rad",
        "direction_reversals",
        "contact_ratio",
        "measured_force_mean_n",
        "measured_force_p95_n",
    )
    valid = [record for record in records if "metrics" in record and record["plan_success"]]
    summary = {"task": args.task, "requested": len(seeds), "valid": len(valid), "q05": {}}
    for name in metric_names:
        values = [record["metrics"][name] for record in valid if name in record["metrics"]]
        if values:
            summary["q05"][name] = float(np.quantile(values, 0.05))
    print("EXPERT_DYNAMIC_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
