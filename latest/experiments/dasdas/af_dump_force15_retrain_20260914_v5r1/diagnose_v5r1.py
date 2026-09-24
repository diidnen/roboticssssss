"""Summarize label and selector distributions for v5r1 diagnosis."""
from collections import Counter
import json
from pathlib import Path

import train_v5


HERE = Path(__file__).resolve().parent


def aggregate(records):
    result = {}
    for split in sorted({record["split"] for record in records}):
        rows = [
            (float(branch["force"]), int(branch["full_task_success_y"]), float(record["y"]))
            for record in records
            if record["split"] == split
            for branch in record["branches"]
        ]
        forces = sorted({force for force, _, _ in rows})
        frictions = sorted({friction for _, _, friction in rows})
        result[split] = {
            "overall": {
                str(force): {
                    "successes": sum(label for candidate, label, _ in rows if candidate == force),
                    "n": sum(candidate == force for candidate, _, _ in rows),
                }
                for force in forces
            },
            "by_friction": {
                str(friction): {
                    str(force): {
                        "successes": sum(label for candidate, label, value in rows if candidate == force and value == friction),
                        "n": sum(candidate == force and value == friction for candidate, _, value in rows),
                    }
                    for force in forces
                }
                for friction in frictions
            },
        }
    return result


def main():
    old, _, _ = train_v5.load_old()
    new, _, _ = train_v5.load_new(include_test_labels=True)
    decisions = json.loads((HERE / "models_v5/feasibility/TRAIN_VAL_DECISIONS.json").read_text())
    output = {
        "old": aggregate(old),
        "new": aggregate(new),
        "selector_by_split": {
            split: {
                "contexts": sum(row["split"] == split for row in decisions),
                "selected_force_counts": dict(sorted(Counter(float(row["selected_force_N"]) for row in decisions if row["split"] == split).items())),
            }
            for split in sorted({row["split"] for row in decisions})
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
