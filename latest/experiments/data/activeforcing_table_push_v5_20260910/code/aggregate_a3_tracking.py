"""Aggregate immutable V5 A3 tracking branches."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np

OUT = Path("/media/volume/data/exouser/activeforcing_table_push_v5_20260910")
TARGETS = [19.5, 20.5, 21.5]


def atomic_json(path, value):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    os.close(fd)
    Path(tmp).write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def main():
    path = OUT / "A3_TRACKING_RESULT.json"
    if path.exists():
        raise RuntimeError("immutable A3 tracking result already exists")
    pairs = []
    for target in TARGETS:
        feedback = json.loads((OUT / "tracking" / f"target_{target:.1f}_feedback.json").read_text())
        baseline = json.loads((OUT / "tracking" / f"target_{target:.1f}_baseline.json").read_text())
        torque_ok = all(
            item["finite"] and item["shape"] == [7] and item["clip_count"] == 0
            for row in feedback["rows"]
            for item in row["adapter_trace"]
        )
        causal_ok = feedback["first_action_untouched"] and all(
            row["causal_measurement_index"] == (None if row["k"] == 0 else row["k"] - 1)
            for row in feedback["rows"]
        )
        improved = feedback["mae_last5_n"] <= 0.8 * baseline["mae_last5_n"]
        pair_pass = (
            improved
            and torque_ok
            and causal_ok
            and feedback["contact_losses"] < 2
            and feedback["all_nominal_actions_fixed"]
        )
        pairs.append(
            {
                "target_n": target,
                "feedback_realized_last5_mean_n": feedback["realized_last5_mean_n"],
                "feedback_mae_last5_n": feedback["mae_last5_n"],
                "baseline_mae_last5_n": baseline["mae_last5_n"],
                "mae_ratio": feedback["mae_last5_n"] / baseline["mae_last5_n"],
                "improved_by_required_20_percent": improved,
                "torque_valid_no_clipping": torque_ok,
                "causal_indexing": causal_ok,
                "contact_losses": feedback["contact_losses"],
                "pass": pair_pass,
            }
        )
    realized = [pair["feedback_realized_last5_mean_n"] for pair in pairs]
    ordered = bool(np.all(np.diff(realized) > 0))
    passed = ordered and all(pair["pass"] for pair in pairs)
    result = {
        "gate": "A3_TRACKING",
        "targets_n": TARGETS,
        "pairs": pairs,
        "realized_last5_n": realized,
        "strict_target_order": ordered,
        "pass": passed,
        "decision": "A3_TRACKING_PASS" if passed else "A3_TRACKING_FAIL_STOP_BEFORE_OUTCOME",
        "new_task_outcomes": 0,
    }
    atomic_json(path, result)


if __name__ == "__main__":
    main()
