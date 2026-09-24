#!/usr/bin/env python3
"""Extract and audit frozen AF decision curves from archived formal contexts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def logloss(labels, probs):
    eps = 1e-12
    return mean(-(y * math.log(min(1 - eps, max(eps, p))) + (1 - y) * math.log(min(1 - eps, max(eps, 1 - p)))) for y, p in zip(labels, probs))


def auc(labels, probs):
    pos = [p for y, p in zip(labels, probs) if y]
    neg = [p for y, p in zip(labels, probs) if not y]
    if not pos or not neg:
        return None
    return sum(1 if p > n else 0.5 if p == n else 0 for p in pos for n in neg) / (len(pos) * len(neg))


def metrics(labels, probs):
    return {
        "n": len(labels),
        "successes": sum(labels),
        "empirical_rate": mean(labels),
        "mean_predicted": mean(probs),
        "auc": auc(labels, probs),
        "brier": mean((p - y) ** 2 for y, p in zip(labels, probs)),
        "logloss": logloss(labels, probs),
    }


def p_at(decision, force):
    grid = [float(v) for v in decision["force_grid_N"]]
    idx = min(range(len(grid)), key=lambda i: abs(grid[i] - force))
    if abs(grid[idx] - force) > 1e-8:
        raise AssertionError((force, grid[idx]))
    return float(decision["p_success"][idx])


def load_evidence(archive):
    prefix = f"{archive.name[:-7]}/job/"
    wanted = {
        prefix + "PREACTION_AF_DECISION.json": "decision",
        prefix + "PREACTION_FEATURE.json": "feature",
        prefix + "PREACTION_PI0_CHUNK.npy": "chunk",
        prefix + "PREACTION_POSTERIOR.json": "posterior",
    }
    found = {}
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            label = wanted.get(member.name)
            if label is None:
                continue
            stream = tf.extractfile(member)
            if stream is None:
                raise AssertionError(member.name)
            found[label] = stream.read()
            if len(found) == len(wanted):
                break
    if len(found) != len(wanted):
        raise AssertionError((archive, sorted(found)))
    return json.loads(found["decision"]), {name: hashlib.sha256(data).hexdigest() for name, data in found.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    core = json.loads(args.core.read_text())
    outcomes = {}
    friction = {}
    for record in core["records"]:
        context_id = record["context"]["id"]
        friction[context_id] = float(record["context"]["friction"])
        outcomes[context_id] = {o["method"]: int(o["success"]) for o in record["outcomes"]}

    rows = []
    for archive in sorted(args.archives.glob("formal_*.tar.gz")):
        context_id = archive.name[:-7]
        decision, evidence_hashes = load_evidence(archive)
        grid = [float(v) for v in decision["force_grid_N"]]
        probs = [float(v) for v in decision["p_success"]]
        peak_idx = max(range(len(probs)), key=probs.__getitem__)
        selected = float(decision["selected_force_N"])
        rows.append({
            "context_id": context_id,
            "friction": friction[context_id],
            "selected_force_N": selected,
            "predicted_at_selected": float(decision["predicted_success"]),
            "actual_AF": outcomes[context_id]["ActiveForcing"],
            "predicted_at_8N": p_at(decision, 8.0),
            "actual_Fixed8": outcomes[context_id]["Fixed-Strong 8N"],
            "probability_peak_force_N": grid[peak_idx],
            "probability_peak": probs[peak_idx],
            "p_at_7_5N": p_at(decision, 7.5),
            "edge_delta_p_8_minus_7_5": p_at(decision, 8.0) - p_at(decision, 7.5),
            "probability_still_increasing_at_8N": p_at(decision, 8.0) > p_at(decision, 7.5),
            "evidence_hashes": evidence_hashes,
        })

    af_labels = [r["actual_AF"] for r in rows]
    af_probs = [r["predicted_at_selected"] for r in rows]
    fixed_labels = [r["actual_Fixed8"] for r in rows]
    fixed_probs = [r["predicted_at_8N"] for r in rows]
    duplicate_groups = {}
    for label in ("decision", "feature", "chunk", "posterior"):
        groups = defaultdict(list)
        for row in rows:
            groups[row["evidence_hashes"][label]].append(row["context_id"])
        duplicate_groups[label] = [
            {"sha256": digest, "contexts": contexts}
            for digest, contexts in sorted(groups.items()) if len(contexts) > 1
        ]

    output = {
        "version": "FORMAL_AF_DECISION_AUDIT_V1",
        "contexts": len(rows),
        "selected_force_range_N": [min(r["selected_force_N"] for r in rows), max(r["selected_force_N"] for r in rows)],
        "selected_at_8N": sum(r["selected_force_N"] == 8.0 for r in rows),
        "probability_peak_at_8N": sum(r["probability_peak_force_N"] == 8.0 for r in rows),
        "probability_still_increasing_at_8N": sum(r["probability_still_increasing_at_8N"] for r in rows),
        "unique_evidence_hashes": {
            label: len({r["evidence_hashes"][label] for r in rows})
            for label in ("decision", "feature", "chunk", "posterior")
        },
        "duplicate_evidence_groups": duplicate_groups,
        "AF_selected_probability_calibration": metrics(af_labels, af_probs),
        "Fixed8_probability_calibration": metrics(fixed_labels, fixed_probs),
        "by_friction": {},
        "rows": rows,
    }
    for mu in sorted(set(r["friction"] for r in rows)):
        group = [r for r in rows if r["friction"] == mu]
        output["by_friction"][str(mu)] = {
            "AF": metrics([r["actual_AF"] for r in group], [r["predicted_at_selected"] for r in group]),
            "Fixed8": metrics([r["actual_Fixed8"] for r in group], [r["predicted_at_8N"] for r in group]),
            "mean_selected_force_N": mean(r["selected_force_N"] for r in group),
        }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
