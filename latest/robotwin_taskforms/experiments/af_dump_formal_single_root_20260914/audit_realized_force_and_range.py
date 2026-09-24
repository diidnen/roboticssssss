#!/usr/bin/env python3
"""Read-only formal 24-context audit: command vs squeeze, and offline 5-15N.

No robot rollouts. Frozen AF_original weights are queried only. Forces above
8 N are feature-OOD because the network encodes force as F/8.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V4 = HERE.parent / "af_dump_maxf8_20260913"
MODELS = V4 / "models_v4"
KEY = Path("/home/exouser/.ssh/codex_anvil_migration_20260912")
HOST = "x-csong7@anvil.rcac.purdue.edu"
REMOTE = (
    "/anvil/projects/x-cis250966/tabero-transfer/"
    "jetstream-activeforcing-20260912/formal_single_root_20260914/main"
)
PROBE = [5.0, 6.0, 8.0, 10.0, 12.0, 15.0]


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def mean(values):
    values = [float(v) for v in values]
    return float(np.mean(values)) if values else None


def median(values):
    values = [float(v) for v in values]
    return float(np.median(values)) if values else None


def ssh_args(command: str) -> list[str]:
    return [
        "ssh",
        "-i",
        str(KEY),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        HOST,
        command,
    ]


def utility(p, f, max_f):
    p = np.asarray(p, dtype=float)
    f = np.asarray(f, dtype=float)
    return p * (max_f - f) / max_f - (1.0 - p)


def build_execution_table() -> list[dict]:
    formal = read(HERE / "FORCE_RANGE_ROOT_CAUSE_DIAGNOSTIC.json")["formal"]["rows"]
    decisions = {row["context_id"]: row for row in read(HERE / "FORMAL_AF_DECISION_AUDIT.json")["rows"]}
    by_context = defaultdict(dict)
    for row in formal:
        by_context[row["context_id"]][row["method"]] = row
    table = []
    for context_id, methods in sorted(by_context.items()):
        af = methods["ActiveForcing"]
        f8 = methods["Fixed-Strong 8N"]
        nom = methods["Nominal Frozen VLA"]
        decision = decisions[context_id]
        command = float(af["commanded_force_N"])
        squeeze = float(af["measured_mean_squeeze_N"])
        contact = float(af["target_contact_fraction"])
        table.append(
            {
                "context_id": context_id,
                "friction": decision["friction"],
                "AF_success": int(af["success"]),
                "Fixed8_success": int(f8["success"]),
                "Nominal_success": int(nom["success"]),
                "AF_command_N": command,
                "AF_squeeze_N": squeeze,
                "AF_contact_fraction": contact,
                "AF_squeeze_per_contact_N": (
                    squeeze / contact if contact > 0 else None
                ),
                "AF_tracking_ratio": squeeze / command if command else None,
                "AF_sustained_contact_loss": bool(af["sustained_contact_loss"]),
                "Fixed8_command_N": 8.0,
                "Fixed8_squeeze_N": float(f8["measured_mean_squeeze_N"]),
                "Fixed8_contact_fraction": float(f8["target_contact_fraction"]),
                "Fixed8_squeeze_per_contact_N": f8["squeeze_per_contact_fraction_N"],
                "Fixed8_tracking_ratio": float(f8["measured_mean_squeeze_N"]) / 8.0,
                "Fixed8_sustained_contact_loss": bool(f8["sustained_contact_loss"]),
                "Nominal_squeeze_N": float(nom["measured_mean_squeeze_N"]),
                "predicted_at_selected": decision["predicted_at_selected"],
                "predicted_at_8N": decision["predicted_at_8N"],
                "probability_peak_force_N": decision["probability_peak_force_N"],
                "p_still_increasing_at_8N": decision["probability_still_increasing_at_8N"],
            }
        )
    return table


def summarize_execution(table: list[dict]) -> dict:
    af_fail = [row for row in table if not row["AF_success"]]
    af_ok = [row for row in table if row["AF_success"]]
    bins = [
        ("<4N", lambda c: c < 4),
        ("4-5.5N", lambda c: 4 <= c < 5.5),
        ("5.5-7N", lambda c: 5.5 <= c < 7),
        (">=7N", lambda c: c >= 7),
    ]
    by_bin = []
    for name, pred in bins:
        group = [row for row in table if pred(row["AF_command_N"])]
        by_bin.append(
            {
                "bin": name,
                "n": len(group),
                "successes": sum(row["AF_success"] for row in group),
                "mean_squeeze_N": mean(row["AF_squeeze_N"] for row in group),
                "mean_contact_fraction": mean(row["AF_contact_fraction"] for row in group),
                "contact_loss": sum(row["AF_sustained_contact_loss"] for row in group),
            }
        )
    high_cmd_fail = [
        row
        for row in af_fail
        if row["AF_command_N"] >= 5.0 and row["AF_tracking_ratio"] < 0.5
    ]
    low_cmd_fail = [row for row in af_fail if row["AF_command_N"] < 5.0]
    return {
        "AF_success_by_selected_force_bin": by_bin,
        "AF_fail_n": len(af_fail),
        "AF_fail_selected_lt_5N": len(low_cmd_fail),
        "AF_fail_selected_ge_5N_and_tracking_lt_0.5": len(high_cmd_fail),
        "AF_fail_with_contact_loss": sum(row["AF_sustained_contact_loss"] for row in af_fail),
        "AF_success_mean_command_N": mean(row["AF_command_N"] for row in af_ok),
        "AF_fail_mean_command_N": mean(row["AF_command_N"] for row in af_fail),
        "AF_success_mean_squeeze_N": mean(row["AF_squeeze_N"] for row in af_ok),
        "AF_fail_mean_squeeze_N": mean(row["AF_squeeze_N"] for row in af_fail),
        "AF_mean_squeeze_per_contact_N": mean(
            row["AF_squeeze_per_contact_N"]
            for row in table
            if row["AF_squeeze_per_contact_N"] is not None
        ),
        "Fixed8_mean_squeeze_per_contact_N": mean(
            row["Fixed8_squeeze_per_contact_N"]
            for row in table
            if row["Fixed8_squeeze_per_contact_N"] is not None
        ),
        "AF_mean_command_N": mean(row["AF_command_N"] for row in table),
        "AF_mean_squeeze_N": mean(row["AF_squeeze_N"] for row in table),
        "Fixed8_mean_squeeze_N": mean(row["Fixed8_squeeze_N"] for row in table),
    }


def fetch_all_preaction(dest: Path, context_ids: list[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    remote_script = (
        "set -euo pipefail; "
        "tmp=$(mktemp -d); "
        f"cd '{REMOTE}'; "
        "for f in formal_*.tar.gz; do "
        "id=${f%.tar.gz}; "
        "tar -xOf \"$f\" \"$id/job/PREACTION_FEATURE.json\" > \"$tmp/${id}_PREACTION_FEATURE.json\"; "
        "tar -xOf \"$f\" \"$id/job/PREACTION_POSTERIOR.json\" > \"$tmp/${id}_PREACTION_POSTERIOR.json\"; "
        "done; "
        "tar -C \"$tmp\" -czf - .; "
        "rm -rf \"$tmp\""
    )
    result = subprocess.run(ssh_args(remote_script), check=True, capture_output=True)
    import tarfile
    from io import BytesIO
    with tarfile.open(fileobj=BytesIO(result.stdout), mode="r:gz") as tf:
        tf.extractall(dest)
    missing = [
        context_id
        for context_id in context_ids
        if not (dest / f"{context_id}_PREACTION_FEATURE.json").exists()
        or not (dest / f"{context_id}_PREACTION_POSTERIOR.json").exists()
    ]
    if missing:
        raise FileNotFoundError(missing)


def run_extended_offline(table: list[dict]) -> dict:
    sys.path.insert(0, str(V4))
    from maxf8_runtime import MaxF8Feasibility

    training = read(MODELS / "TRAINING_COMPLETE.json")
    model = MaxF8Feasibility(
        str(MODELS / "feasibility" / "FEASIBILITY_MANIFEST.json"),
        manifest_sha256=training["feasibility_manifest_sha256"],
        device="cpu",
    )
    original_grid = np.asarray(model.force_grid, dtype=float)
    rows = []
    work = Path(tempfile.mkdtemp(prefix="formal_preaction_"))
    fetch_all_preaction(work, [item["context_id"] for item in table])
    for item in table:
        context_id = item["context_id"]
        feature = read(work / f"{context_id}_PREACTION_FEATURE.json")
        posterior = read(work / f"{context_id}_PREACTION_POSTERIOR.json")
        native = model.select(feature, posterior)
        if abs(float(native["selected_force_N"]) - item["AF_command_N"]) > 1e-6:
            raise ValueError(
                f"Replay drift {context_id}: {native['selected_force_N']} vs {item['AF_command_N']}"
            )
        model.force_grid = np.asarray(PROBE, dtype=float)
        try:
            curve = model.curve(np.asarray(feature["sequence"], np.float32), posterior)
        finally:
            model.force_grid = original_grid
        p = {float(f): float(c) for f, c in zip(PROBE, curve)}
        util15 = utility([p[f] for f in PROBE], PROBE, 15.0)
        pick15 = PROBE[int(np.argmax(util15))]
        rows.append(
            {
                "context_id": context_id,
                "replay_selected_force_N": float(native["selected_force_N"]),
                "p_probe": p,
                "arg_max_p": float(PROBE[int(np.argmax(curve))]),
                "hypothetical_selected_under_maxF15": float(pick15),
                "hypothetical_utility_maxF15": {
                    str(f): float(u) for f, u in zip(PROBE, util15)
                },
                "p_increases_8_to_15": p[15.0] > p[8.0],
                "frozen_utility_can_select_above_8N": False,
            }
        )
    return {
        "probe_forces_N": PROBE,
        "force_feature_encoding": "F/8 inside the frozen network",
        "forces_above_8N_are_feature_OOD": True,
        "frozen_expected_utility_rejects_F_gt_8": True,
        "n_select_15_under_hypothetical_maxF15": sum(
            row["hypothetical_selected_under_maxF15"] == 15.0 for row in rows
        ),
        "n_select_12_under_hypothetical_maxF15": sum(
            row["hypothetical_selected_under_maxF15"] == 12.0 for row in rows
        ),
        "n_select_10_under_hypothetical_maxF15": sum(
            row["hypothetical_selected_under_maxF15"] == 10.0 for row in rows
        ),
        "n_p_increases_8_to_15": sum(row["p_increases_8_to_15"] for row in rows),
        "n_argmax_p_above_8": sum(row["arg_max_p"] > 8.0 for row in rows),
        "rows": rows,
    }


def main() -> None:
    table = build_execution_table()
    execution = summarize_execution(table)
    offline = run_extended_offline(table)
    result = {
        "audit": "FORMAL_REALIZED_FORCE_AND_RANGE_V1",
        "robot_rollouts_added": 0,
        "formal_contexts": 24,
        "execution": execution,
        "contexts": table,
        "offline_extended_range": offline,
        "interpretation_constraints": [
            "Formal 24-context evaluation compared Nominal, Fixed-8, and AF only. It does not contain Fixed-5.",
            "Stage-I 16-context Fixed-5 7/16 is a different experiment and must not be cited as the formal 24-context result.",
            "Trajectory-mean squeeze is not a Newton setpoint. Nominal 13.757 N is mean squeeze, not a command.",
            "Frozen utility p*(8-F)/8-(1-p) hard-rejects F>8, so a 10-15 N candidate cannot be selected without changing the utility contract.",
        ],
    }
    write(HERE / "FORMAL_REALIZED_FORCE_AND_RANGE_AUDIT.json", result)
    print(json.dumps({"wrote": str(HERE / "FORMAL_REALIZED_FORCE_AND_RANGE_AUDIT.json"), **execution, **{k: offline[k] for k in offline if k != "rows"}}, indent=2))


if __name__ == "__main__":
    main()
