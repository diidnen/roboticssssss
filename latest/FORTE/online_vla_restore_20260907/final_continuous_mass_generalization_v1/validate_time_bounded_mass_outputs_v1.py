#!/usr/bin/env python3
"""Independent final QA for the time-bounded MASS evidence package."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from time_bounded_mass_common import HERE, load_frozen_subset, read, sha256
from train_time_bounded_mass_feasibility_v1 import metric


def close(a, b, tolerance=1e-10):
    if a is None or b is None: return a is b
    return bool(np.isclose(float(a), float(b), rtol=0, atol=tolerance))


def main() -> None:
    output = HERE / "TIME_BOUNDED_MASS_FINAL_VALIDATION.json"
    if output.exists(): raise FileExistsError(output)
    subset, subset_sha = load_frozen_subset()
    lock = read(HERE / "TIME_BOUNDED_MASS_FEASIBILITY_CHECKPOINT_SELECTION_LOCK.json")
    if lock.get("final_validator_sha256") != sha256(Path(__file__)):
        raise RuntimeError("final validator changed after checkpoint selection")
    data = read(HERE / "TIME_BOUNDED_MASS_DATA_AUDIT.json")
    held = read(HERE / "TIME_BOUNDED_MASS_HELDOUT_RESULTS.json")
    qualification = read(HERE / "TIME_BOUNDED_MASS_MODEL_QUALIFICATION.json")
    claims = read(HERE / "TIME_BOUNDED_MASS_CLAIM_AUDIT.json")
    terminal = read(HERE / "FINAL_TIME_BOUNDED_MASS_TERMINAL.json")
    pause = read(HERE / "ORIGINAL_648_QUEUE_PAUSE_AUDIT.json")
    with (HERE / "TABLE_TIME_BOUNDED_MASS_PRIMARY.csv").open() as stream: table = list(csv.DictReader(stream))
    with (HERE / "TIME_BOUNDED_MASS_HELDOUT_PREDICTIONS.csv").open() as stream: pred = list(csv.DictReader(stream))
    recomputed = metric([int(row["full_task_success_y"]) for row in pred], [float(row["p_success"]) for row in pred])
    metric_checks = {name: close(recomputed[name], held["metrics"][name]) for name in ("n", "positive_count", "NLL", "Brier", "AUROC", "AUPRC", "ECE_10bin")}
    report_text = (HERE / "FINAL_TIME_BOUNDED_MASS_REPORT.md").read_text()
    required_phrases = ["Executive conclusion", "Protocol change", "Reduced design", "Data quality",
                        "Mass belief", "HELDOUT", "Claim audit", "Paper recommendation", "Reproducibility"]
    pdf = HERE / "FIGURE_TIME_BOUNDED_MASS_COARSE_GRID.pdf"
    pdf_check = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=False)
    checks = {
        "subset_hash_exact": held["subset_sha256"] == subset_sha,
        "data_audit_pass": data["status"] == "PASS" and data["valid_branches"] == 216,
        "table_exact_216_unique": len(table) == 216 and len({(row["context_id"], row["force_N"]) for row in table}) == 216,
        "table_split_counts": {split: sum(row["split"] == split for row in table) for split in ("TRAIN", "VAL", "HELDOUT")} == {"TRAIN": 144, "VAL": 36, "HELDOUT": 36},
        "heldout_predictions_36": len(pred) == 36,
        "heldout_metrics_recomputed": all(metric_checks.values()),
        "heldout_access_once": qualification["heldout_access_count"] == 1 and held["heldout_labels_used_for_checkpoint_or_threshold_selection"] is False,
        "fine_claim_not_tested": claims["FINE_FORCE_CALIBRATION_CLAIM"] == "NOT_TESTED",
        "continuous_claim_limited": claims["CONTINUOUS_FORCE_RESPONSE_CLAIM"] == "LIMITED_BY_COARSE_GRID",
        "no_result_driven_reruns": terminal["RESULT_DRIVEN_RERUNS"] == 0,
        "original_protocol_preserved": all(pause["checks"].values()),
        "report_sections_present": all(phrase in report_text for phrase in required_phrases),
        "pdf_parseable_nonempty": pdf.is_file() and pdf.stat().st_size > 1000 and pdf_check.returncode == 0,
        "remaining_superset_math": terminal["REMAINING_BRANCHES_IF_648_LATER_REQUESTED"] == 394,
    }
    result = {
        "schema": "TIME_BOUNDED_MASS_FINAL_VALIDATION_V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "heldout_metric_checks": metric_checks,
        "artifact_sha256": {path.name: sha256(path) for path in (
            HERE / "TIME_BOUNDED_MASS_DATA_AUDIT.json", HERE / "TIME_BOUNDED_MASS_MODEL_QUALIFICATION.json",
            HERE / "TIME_BOUNDED_MASS_HELDOUT_RESULTS.json", HERE / "TIME_BOUNDED_MASS_CLAIM_AUDIT.json",
            HERE / "TABLE_TIME_BOUNDED_MASS_PRIMARY.csv", HERE / "FINAL_TIME_BOUNDED_MASS_REPORT.md", pdf,
        )},
        "source_sha256": sha256(Path(__file__)),
    }
    with output.open("x") as stream: json.dump(result, stream, indent=2, sort_keys=True); stream.write("\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS": raise SystemExit(2)


if __name__ == "__main__":
    main()
