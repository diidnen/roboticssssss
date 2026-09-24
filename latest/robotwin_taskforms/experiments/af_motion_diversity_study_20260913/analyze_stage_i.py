"""Population analysis for the precommitted 32-context Stage-I experiment."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIXED_METHODS = ["Fixed-1N", "Fixed-3N", "Fixed-5N", "Fixed-6N", "Fixed-8N"]
METHODS = ["ActiveForcing", *FIXED_METHODS]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def percentile_ci(values: list[float]) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def jaccard(a: set[float], b: set[float]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> None:
    complete = read(HERE / "STAGE_I_COMPLETE.json")
    protocol = read(HERE / "STAGE_I_PROTOCOL.json")
    lock = read(HERE / "STAGE_I_FREEZE_LOCK.json")
    if not complete["completed"] or complete["contexts"] != 32:
        raise ValueError("Stage I is incomplete")
    if sha(HERE / "STAGE_I_PROTOCOL.json") != lock["protocol_sha256"]:
        raise ValueError("Protocol drift")
    contexts = read(HERE / "CONTEXTS.json")
    records = []
    for context in contexts:
        record_path = HERE / "stage_i_records" / f"{context['id']}.json"
        receipt_path = HERE / "stage_i_archived" / f"{context['id']}.json"
        record = read(record_path)
        receipt = read(receipt_path)
        if record["context"] != context or not record["strict_matched"]:
            raise ValueError("Context/record mismatch")
        if not all(
            receipt[key]
            for key in [
                "remote_sha256_verified",
                "remote_gzip_test_passed",
                "remote_tar_listing_passed",
            ]
        ):
            raise ValueError("Unverified raw archive")
        records.append(record)

    rows = []
    context_rows = []
    for record in records:
        by_method = {row["method"]: row for row in record["branches"]}
        if set(by_method) != set(METHODS):
            raise ValueError("Missing method")
        fixed_successes = {
            by_method[method]["force_N"]
            for method in FIXED_METHODS
            if by_method[method]["success"]
        }
        fixed_outcomes = [
            (by_method[method]["force_N"], by_method[method]["success"])
            for method in FIXED_METHODS
        ]
        nonmonotonic = any(
            y1 == 1 and y2 == 0 and f1 < f2
            for (f1, y1), (f2, y2) in itertools.permutations(fixed_outcomes, 2)
        )
        af = by_method["ActiveForcing"]
        f8 = by_method["Fixed-8N"]
        paired_class = (
            "both_success"
            if af["success"] and f8["success"]
            else "AF_only"
            if af["success"]
            else "Fixed8_only"
            if f8["success"]
            else "both_failure"
        )
        context_rows.append(
            {
                "context_id": record["context"]["id"],
                "friction": record["context"]["friction"],
                "policy_seed": record["context"]["policy_seed"],
                "feasible_fixed_forces_N": sorted(fixed_successes),
                "nonmonotonic": nonmonotonic,
                "AF_success": af["success"],
                "AF_force_N": af["force_N"],
                "Fixed8_success": f8["success"],
                "paired_class": paired_class,
                "force_saving_N": (
                    8.0 - af["force_N"]
                    if af["success"] and f8["success"]
                    else None
                ),
            }
        )
        for branch in record["branches"]:
            rows.append(
                {
                    "context_id": record["context"]["id"],
                    "friction": record["context"]["friction"],
                    "policy_seed": record["context"]["policy_seed"],
                    "method": branch["method"],
                    "commanded_force_N": branch["force_N"],
                    "success": branch["success"],
                    "measured_mean_squeeze_N": branch["measured_mean_squeeze_N"],
                    "sustained_contact_loss": branch["trace_summary"][
                        "sustained_contact_loss"
                    ],
                    "first_sustained_contact_loss_step": branch["trace_summary"][
                        "first_sustained_contact_loss_step"
                    ],
                }
            )

    rng = np.random.default_rng(protocol["statistics_precommitted"]["bootstrap_seed"])
    flags = np.asarray([int(row["nonmonotonic"]) for row in context_rows])
    nonmono_boot = [
        float(rng.choice(flags, size=len(flags), replace=True).mean())
        for _ in range(protocol["statistics_precommitted"]["bootstrap_replicates"])
    ]

    landscape_pairs = []
    by_mu = defaultdict(list)
    for row in context_rows:
        by_mu[row["friction"]].append(row)
    for mu, group in sorted(by_mu.items()):
        if len(group) != 8:
            raise ValueError("Each friction must contain eight motions")
        for a, b in itertools.combinations(group, 2):
            set_a = set(a["feasible_fixed_forces_N"])
            set_b = set(b["feasible_fixed_forces_N"])
            sim = jaccard(set_a, set_b)
            landscape_pairs.append(
                {
                    "friction": mu,
                    "context_a": a["context_id"],
                    "context_b": b["context_id"],
                    "exact_set_mismatch": set_a != set_b,
                    "jaccard_similarity": sim,
                    "jaccard_distance": 1.0 - sim,
                    "minimum_successful_force_difference_N": (
                        abs(min(set_a) - min(set_b)) if set_a and set_b else None
                    ),
                    "maximum_successful_force_difference_N": (
                        abs(max(set_a) - max(set_b)) if set_a and set_b else None
                    ),
                }
            )

    # Hierarchical bootstrap: retain four friction strata, resample eight
    # motion contexts within each stratum, then recompute within-stratum pairs.
    landscape_boot = []
    for _ in range(protocol["statistics_precommitted"]["bootstrap_replicates"]):
        sample_pairs = []
        for mu in sorted(by_mu):
            group = by_mu[mu]
            sample = [group[i] for i in rng.integers(0, len(group), len(group))]
            for a, b in itertools.combinations(sample, 2):
                aa = set(a["feasible_fixed_forces_N"])
                bb = set(b["feasible_fixed_forces_N"])
                sample_pairs.append(
                    (float(aa != bb), 1.0 - jaccard(aa, bb))
                )
        landscape_boot.append(
            {
                "mismatch": float(np.mean([x[0] for x in sample_pairs])),
                "jaccard_distance": float(np.mean([x[1] for x in sample_pairs])),
            }
        )

    method_summary = []
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        successful = [row for row in group if row["success"]]
        failures = [row for row in group if not row["success"]]
        method_summary.append(
            {
                "method": method,
                "contexts": len(group),
                "successes": len(successful),
                "success_rate": len(successful) / len(group),
                "failures": len(failures),
                "mean_commanded_force_N": float(
                    np.mean([row["commanded_force_N"] for row in group])
                ),
                "mean_measured_force_N": float(
                    np.mean([row["measured_mean_squeeze_N"] for row in group])
                ),
                "mean_commanded_force_success_conditioned_N": (
                    float(np.mean([row["commanded_force_N"] for row in successful]))
                    if successful
                    else None
                ),
                "sustained_contact_loss_rate": float(
                    np.mean([row["sustained_contact_loss"] for row in group])
                ),
                "sustained_contact_loss_rate_on_failures": (
                    float(
                        np.mean(
                            [row["sustained_contact_loss"] for row in failures]
                        )
                    )
                    if failures
                    else None
                ),
            }
        )

    paired_counts = {
        name: sum(row["paired_class"] == name for row in context_rows)
        for name in ["both_success", "AF_only", "Fixed8_only", "both_failure"]
    }
    savings = [
        row["force_saving_N"]
        for row in context_rows
        if row["force_saving_N"] is not None
    ]
    min_differences = [
        row["minimum_successful_force_difference_N"]
        for row in landscape_pairs
        if row["minimum_successful_force_difference_N"] is not None
    ]
    max_differences = [
        row["maximum_successful_force_difference_N"]
        for row in landscape_pairs
        if row["maximum_successful_force_difference_N"] is not None
    ]
    result = {
        "completed_utc": now(),
        "contexts": 32,
        "robot_rollouts": 192,
        "nonmonotonic": {
            "count": int(flags.sum()),
            "rate": float(flags.mean()),
            "bootstrap_95pct_CI": percentile_ci(nonmono_boot),
        },
        "motion_dependent_landscape": {
            "within_friction_pairs": len(landscape_pairs),
            "exact_set_mismatch_rate": float(
                np.mean([row["exact_set_mismatch"] for row in landscape_pairs])
            ),
            "exact_set_mismatch_bootstrap_95pct_CI": percentile_ci(
                [row["mismatch"] for row in landscape_boot]
            ),
            "mean_jaccard_similarity": float(
                np.mean([row["jaccard_similarity"] for row in landscape_pairs])
            ),
            "mean_jaccard_distance": float(
                np.mean([row["jaccard_distance"] for row in landscape_pairs])
            ),
            "jaccard_distance_bootstrap_95pct_CI": percentile_ci(
                [row["jaccard_distance"] for row in landscape_boot]
            ),
            "mean_minimum_successful_force_difference_N": (
                float(np.mean(min_differences)) if min_differences else None
            ),
            "mean_maximum_successful_force_difference_N": (
                float(np.mean(max_differences)) if max_differences else None
            ),
        },
        "AF_vs_Fixed8": paired_counts,
        "force_saving_both_success": {
            "n": len(savings),
            "mean_N": float(np.mean(savings)) if savings else None,
            "median_N": float(np.median(savings)) if savings else None,
            "values_N": savings,
        },
        "methods": method_summary,
        "all_precommitted_contexts_included": True,
        "protocol_violations": [],
    }
    write(HERE / "STAGE_I_POPULATION_RESULTS.json", result)

    with (HERE / "STAGE_I_CONTEXT_RESULTS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "context_id",
                "friction",
                "policy_seed",
                "feasible_fixed_forces_N",
                "nonmonotonic",
                "AF_success",
                "AF_force_N",
                "Fixed8_success",
                "paired_class",
                "force_saving_N",
            ],
        )
        writer.writeheader()
        writer.writerows(context_rows)
    with (HERE / "STAGE_I_LANDSCAPE_PAIRS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(landscape_pairs[0]))
        writer.writeheader()
        writer.writerows(landscape_pairs)

    by_method = {row["method"]: row for row in method_summary}
    report = f"""# MOTION_EXPANSION_FROZEN_MODEL

## Scope

Precommitted frozen-model population experiment: root 200002, four frictions,
eight untouched policy seeds, 32 contexts, six paired methods, 192 rollouts.
All contexts are included. The earlier 502/602 pilot is excluded from these
population statistics.

## Q1 — Non-monotonic force feasibility

{result['nonmonotonic']['count']} / 32 contexts
({result['nonmonotonic']['rate']:.1%}; 95% context-bootstrap CI
{result['nonmonotonic']['bootstrap_95pct_CI'][0]:.1%}–{result['nonmonotonic']['bootstrap_95pct_CI'][1]:.1%})
contain F1<F2 with fixed-grid success at F1 and failure at F2.

## Q2 — Motion-dependent feasible-force landscape

Across {len(landscape_pairs)} within-friction motion pairs:

- exact-set mismatch: {result['motion_dependent_landscape']['exact_set_mismatch_rate']:.1%}
  (95% hierarchical-bootstrap CI
  {result['motion_dependent_landscape']['exact_set_mismatch_bootstrap_95pct_CI'][0]:.1%}–
  {result['motion_dependent_landscape']['exact_set_mismatch_bootstrap_95pct_CI'][1]:.1%});
- mean Jaccard distance:
  {result['motion_dependent_landscape']['mean_jaccard_distance']:.3f}
  (95% CI
  {result['motion_dependent_landscape']['jaccard_distance_bootstrap_95pct_CI'][0]:.3f}–
  {result['motion_dependent_landscape']['jaccard_distance_bootstrap_95pct_CI'][1]:.3f});
- mean difference in minimum successful force:
  {result['motion_dependent_landscape']['mean_minimum_successful_force_difference_N']} N;
- mean difference in maximum successful force:
  {result['motion_dependent_landscape']['mean_maximum_successful_force_difference_N']} N.

The primary feasible set uses the common fixed grid {{1,3,5,6,8}}. AF-selected
forces are reported separately because they differ across contexts.

## Q3 — Frozen AF success–force trade-off

- ActiveForcing: {by_method['ActiveForcing']['successes']}/32
  ({by_method['ActiveForcing']['success_rate']:.1%}), mean commanded force
  {by_method['ActiveForcing']['mean_commanded_force_N']:.3f} N.
- Fixed-8: {by_method['Fixed-8N']['successes']}/32
  ({by_method['Fixed-8N']['success_rate']:.1%}), mean commanded force 8 N.

Full method metrics are in `STAGE_I_POPULATION_RESULTS.json`.

## Q4/Q5 — Paired AF vs Fixed-8

- both success: {paired_counts['both_success']}
- AF-only success: {paired_counts['AF_only']}
- Fixed-8-only success: {paired_counts['Fixed8_only']}
- both fail: {paired_counts['both_failure']}

Conditional on both succeeding, AF saves mean
{result['force_saving_both_success']['mean_N']} N and median
{result['force_saving_both_success']['median_N']} N.

## Q6 — Pilot-phenomenon replication

Population frequencies above determine whether the pilot phenomena recur:
non-monotonic force landscapes are counted across all 32 contexts; AF-only and
Fixed8-only outcomes are reported without case selection. Narrow intermediate
solutions may be described only as case studies and are not inferred from this
coarse fixed grid.

## Gate

Stage I is complete and audited. Stage II may begin only with these eight policy
seeds excluded from both training and validation.
"""
    (HERE / "MOTION_EXPANSION_FROZEN_MODEL.md").write_text(report)
    write(
        HERE / "STAGE_I_GATE.json",
        {
            "stage_i_complete": True,
            "all_32_contexts_included": True,
            "raw_archives_verified": True,
            "report_sha256": sha(HERE / "MOTION_EXPANSION_FROZEN_MODEL.md"),
            "stage_ii_may_begin": True,
            "forbidden_train_val_policy_seeds": read(
                HERE / "PRECOMMITTED_MOTION_SEEDS.json"
            )["fresh_policy_seeds"],
            "created_utc": now(),
        },
    )


if __name__ == "__main__":
    main()
