import json
from pathlib import Path

import numpy as np

ROOT = Path("/media/volume/data/exouser")
paths = {
    "v7_20.5": ROOT / "activeforcing_table_push_v7_20260911/online_pair/hybrid.jsonl",
    "v7_30": ROOT / "activeforcing_table_push_v7_20260911/online_calibration/mu_1.20_target_30.0N/hybrid.jsonl",
    "default_untouched": ROOT / "activeforcing_table_push_v6_20260910/online_calibration/mu_0.60_target_10.0N/untouched.jsonl",
}
for label, path in paths.items():
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    interventions = [row for row in rows if row.get("intervention")]
    traces = [
        item
        for row in interventions
        for item in row.get("controller_trace", [])
        if item.get("mode") == "torque_safe_minimum_axis_force"
    ]
    output = {
        "label": label,
        "steps": len(rows),
        "success_any": any(row["native_success_now"] for row in rows),
        "initial_plate_xy": rows[0]["plate_position"][:2],
        "final_plate_xy": rows[-1]["plate_position"][:2],
        "intervention_steps": len(interventions),
        "contact_steps": sum(row["post_contact"] for row in rows),
    }
    if traces:
        aligned = [row["post_aligned_force_n"] for row in interventions if row["post_contact"]]
        output.update(
            {
                "floor_active_substeps": sum(item["floor_active"] for item in traces),
                "total_controller_substeps": len(traces),
                "stock_axis_mean_n": float(np.mean([item["stock_axis_force_n"] for item in traces])),
                "applied_axis_mean_n": float(np.mean([item["applied_axis_force_n"] for item in traces])),
                "post_contact_aligned_mean_n": float(np.mean(aligned)) if aligned else None,
                "post_contact_aligned_max_n": float(np.max(aligned)) if aligned else None,
                "minimum_safety_alpha": float(min(item["safety_alpha"] for item in traces)),
            }
        )
    print(json.dumps(output, indent=2))
