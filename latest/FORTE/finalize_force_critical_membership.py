#!/usr/bin/env python3
"""QA frozen real sweeps and derive model-independent challenge membership."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "activeforcing_final_experiment_20260901_045000"
TASKS = (0, 5)
GRID = {0: np.arange(3.0, 5.0001, 0.25), 5: np.arange(3.0, 5.0001, 0.25)}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def truth(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def failure_class(row: pd.Series) -> str:
    if int(row.full_task_success_y) == 1:
        return "SUCCESS"
    if int(row.pick_success) == 0 or int(row.lift_success) == 0:
        return "UNDER_FORCE_IMMEDIATE"
    reason = " ".join(str(row.get(k, "")) for k in ("failure_reason", "failure_stage")).lower()
    if int(row.get("lost_in_transit", 0)) == 1 or int(row.get("transport_retention", 1)) == 0:
        if int(row.task) == 5 or "rotat" in reason or "turn" in reason:
            return "ROTATIONAL_SLIP"
        return "TRANSPORT_SLIP"
    if int(row.get("place_success", 1)) == 0 and (int(row.get("dropped", 0)) == 1 or "grip" in reason or "slip" in reason):
        return "PLACEMENT_GRIP_LOSS"
    if int(row.get("timeout", 0)) == 1:
        return "VLA_NON_FORCE_FAILURE"
    return "OTHER"


def main() -> None:
    protocol = json.loads((OUT / "ACTIVEFORCING_FINAL_EXPERIMENT_PROTOCOL.json").read_text())
    definition_path = OUT / "FORCE_CRITICAL_CHALLENGE_DEFINITION.json"
    if sha(definition_path) != protocol["definition_sha256"]:
        raise RuntimeError("frozen definition changed")
    contexts_expected = json.loads((OUT / "CHALLENGE_CONTEXTS.json").read_text())
    targets = json.loads((OUT / "CHALLENGE_TARGET_MANIFEST.json").read_text())["contexts"]

    contexts, branches, parity = [], [], []
    missing = []
    for task in TASKS:
        base = OUT / "collection_challenge" / f"task{task}" / f"task{task}"
        for name, sink in (("context.csv", contexts), ("branches.csv", branches), ("parity.csv", parity)):
            path = base / name
            if not path.exists():
                missing.append(str(path))
            else:
                q = pd.read_csv(path)
                q["source_path"] = str(path)
                sink.append(q)
    if missing:
        write_json(OUT / "FORCE_CRITICAL_CHALLENGE_QA.json", {
            "status": "INCOMPLETE_COLLECTION", "missing": missing,
            "definition_sha256": sha(definition_path), "model_queries": 0,
            "untouched_TEST_read": False,
        })
        raise SystemExit("collection incomplete; membership not computed")

    c = pd.concat(contexts, ignore_index=True)
    b = pd.concat(branches, ignore_index=True)
    p = pd.concat(parity, ignore_index=True)
    for col in ("full_task_success_y", "pick_success", "lift_success", "transport_retention", "place_success", "dropped", "lost_in_transit", "timeout", "state_parity"):
        if col in b:
            b[col] = truth(b[col])
    b["task"] = pd.to_numeric(b.task).astype(int)
    b["requested_force_N"] = pd.to_numeric(b.requested_force_N).astype(float)
    b["failure_class"] = b.apply(failure_class, axis=1)

    expected_ids = {str(r["context_id"]) for r in contexts_expected}
    committed_ids = set(c.context_id.astype(str))
    strict = truth(c.strict_matched) if "strict_matched" in c else pd.Series(np.zeros(len(c), int))
    duplicate_branch_ids = int(b.branch_id.astype(str).duplicated().sum())
    duplicate_context_ids = int(c.context_id.astype(str).duplicated().sum())
    checks = {
        "expected_contexts": len(expected_ids), "committed_contexts": len(committed_ids),
        "expected_branches": sum(len(v) for v in targets.values()), "committed_branches": len(b),
        "missing_contexts": sorted(expected_ids - committed_ids), "unexpected_contexts": sorted(committed_ids - expected_ids),
        "strict_matched_contexts": int(strict.sum()), "duplicate_context_ids": duplicate_context_ids,
        "duplicate_branch_ids": duplicate_branch_ids,
        "state_parity_failures": int((b.state_parity != 1).sum()) if "state_parity" in b else None,
        "parity_rows": len(p),
    }
    bad_cells = []
    for cid in sorted(expected_ids & committed_ids):
        q = b[b.context_id.astype(str) == cid]
        expected = targets[cid]
        expected_pairs = Counter((round(float(x["force_N"]), 6), int(x["repeat_index"])) for x in expected)
        # The branch label is authoritative because the legacy writer has had
        # a repeat-index display bug in some lineages.
        observed_pairs = Counter()
        for _, row in q.iterrows():
            label = str(row.branch_label)
            rep = 1 if "_R1" in label else (2 if "_R2" in label else int(row.repeat_index))
            observed_pairs[(round(float(row.requested_force_N), 6), rep)] += 1
        if expected_pairs != observed_pairs:
            bad_cells.append({"context_id": cid, "expected": dict((str(k), v) for k, v in expected_pairs.items()), "observed": dict((str(k), v) for k, v in observed_pairs.items())})
    checks["force_repeat_cell_mismatches"] = bad_cells
    integrity_pass = (
        checks["committed_contexts"] == checks["expected_contexts"] == 48
        and checks["committed_branches"] == checks["expected_branches"] == 864
        and not checks["missing_contexts"] and not checks["unexpected_contexts"]
        and checks["strict_matched_contexts"] == 48 and duplicate_branch_ids == 0
        and duplicate_context_ids == 0 and checks["state_parity_failures"] == 0
        and not bad_cells
    )
    if not integrity_pass:
        write_json(OUT / "FORCE_CRITICAL_CHALLENGE_QA.json", {
            "status": "COLLECTION_QA_FAIL_MEMBERSHIP_NOT_FROZEN", "checks": checks,
            "definition_sha256": sha(definition_path), "model_queries": 0,
            "untouched_TEST_read": False,
        })
        raise SystemExit("collection QA failed")

    # Per-context empirical force response. The pre-registered F*0.8 is the
    # smallest force with both repeats successful; unsupported contexts have NA.
    summaries = {}
    type1_pairs, type2_pairs = [], []
    for cid in sorted(expected_ids):
        q = b[b.context_id.astype(str) == cid].copy()
        task = int(q.task.iloc[0]); root_id = str(q.root_id.iloc[0]); mu = float(q.hidden_friction_analysis_only.iloc[0])
        cells = []
        for force, g in q.groupby("requested_force_N", sort=True):
            cells.append({
                "force_N": float(force), "n": len(g), "successes": int(g.full_task_success_y.sum()),
                "success_rate": float(g.full_task_success_y.mean()),
                "failure_classes": dict(Counter(g.failure_class[g.full_task_success_y == 0])),
            })
        supported = [x["force_N"] for x in cells if x["success_rate"] >= 0.8]
        frontier = min(supported) if supported else None
        summaries[cid] = {"context_id": cid, "task": task, "root_id": root_id, "mu": mu, "frontier_Fstar_0p8_N": frontier, "cells": cells}
        for low, high in zip(cells[:-1], cells[1:]):
            if low["success_rate"] == 0.0 and high["success_rate"] == 1.0:
                pair = {"context_id": cid, "task": task, "root_id": root_id, "mu": mu, "F_low_N": low["force_N"], "F_high_N": high["force_N"]}
                type1_pairs.append(pair)
                low_rows = q[np.isclose(q.requested_force_N, low["force_N"])]
                delayed = low_rows[
                    (low_rows.pick_success == 1) & (low_rows.lift_success == 1)
                    & low_rows.failure_class.isin(["TRANSPORT_SLIP", "ROTATIONAL_SLIP", "PLACEMENT_GRIP_LOSS"])
                ]
                if len(delayed) == len(low_rows):
                    type2_pairs.append({**pair, "failure_classes": sorted(delayed.failure_class.unique().tolist())})

    by_root = defaultdict(list)
    for row in summaries.values():
        by_root[(row["task"], row["root_id"])].append(row)
    type3 = []
    for (task, root_id), rows in sorted(by_root.items()):
        finite = [r for r in rows if r["frontier_Fstar_0p8_N"] is not None]
        if len(finite) < 2:
            continue
        lo, hi = min(finite, key=lambda r: r["mu"]), max(finite, key=lambda r: r["mu"])
        delta = abs(float(lo["frontier_Fstar_0p8_N"]) - float(hi["frontier_Fstar_0p8_N"]))
        if delta + 1e-12 >= 0.25:
            type3.append({
                "task": task, "root_id": root_id, "low_mu_context_id": lo["context_id"], "high_mu_context_id": hi["context_id"],
                "low_mu": lo["mu"], "high_mu": hi["mu"], "low_mu_frontier_N": lo["frontier_Fstar_0p8_N"],
                "high_mu_frontier_N": hi["frontier_Fstar_0p8_N"], "absolute_frontier_delta_N": delta,
            })
    member_ids = set(x["context_id"] for x in type1_pairs) | set(x["context_id"] for x in type2_pairs)
    for x in type3:
        member_ids.update([x["low_mu_context_id"], x["high_mu_context_id"]])
    members = [summaries[x] for x in sorted(member_ids)]
    manifest = {
        "status": "MEMBERSHIP_FROZEN_FROM_REAL_OUTCOMES_BEFORE_MODEL_QUERY",
        "definition_sha256": sha(definition_path), "selection_inputs": ["real full-task outcomes", "failure stage", "adjacent-force relation", "hidden-friction relation"],
        "forbidden_selection_inputs_used": [], "model_queries": 0, "untouched_TEST_read": False,
        "counts": {"type1_pairs": len(type1_pairs), "type2_pairs": len(type2_pairs), "type3_root_pairs": len(type3), "member_contexts": len(members), "independent_member_roots": len(set((x["task"], x["root_id"]) for x in members))},
        "type1_recoverable_near_frontier": type1_pairs,
        "type2_delayed_grip_failure": type2_pairs,
        "type3_hidden_friction_discordant": type3,
        "member_contexts": members,
    }
    write_json(OUT / "FORCE_CRITICAL_CHALLENGE_MANIFEST.json", manifest)
    qa = {
        "status": "PASS_MEMBERSHIP_FROZEN" if members else "PASS_COLLECTION_BUT_CHALLENGE_EMPTY",
        "definition_sha256": sha(definition_path), "manifest_sha256": sha(OUT / "FORCE_CRITICAL_CHALLENGE_MANIFEST.json"),
        "checks": checks, "membership_counts": manifest["counts"], "failure_taxonomy_counts": dict(Counter(b.failure_class)),
        "model_queries": 0, "untouched_TEST_read": False,
    }
    write_json(OUT / "FORCE_CRITICAL_CHALLENGE_QA.json", qa)
    print(json.dumps({"status": qa["status"], **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
