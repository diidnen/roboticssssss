#!/usr/bin/env python3
"""Reconstruct every frozen E1 Direct/Utility candidate score on CPU.

This script is analysis-only. It reads the authoritative shared artifacts and
frozen checkpoints, performs no fitting, and writes only inside Agent A's
isolated worktree.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path("/home/exouser/FORTE")
OUT = Path(__file__).resolve().parents[1] / "ACTIVEFORCING_E1_CRITICAL_CLOSURE_20260902_084000"
sys.path.insert(0, str(SOURCE))

import utility_and_causal_ablation_closure as u  # noqa: E402


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(u.ARCHIVE_DATA).reset_index(drop=True)
    members, _ = u.load_probe_members(data["context_id"])
    point_mu = np.mean(np.stack([members[m] for m in u.POSTERIOR_MEMBERS]), axis=0).astype(np.float32)
    prior = [float(x) for x in json.loads(u.NO_PROBE_PRIOR.read_text())["values"]]

    with tempfile.TemporaryDirectory(prefix="e1diag_direct_", dir="/tmp") as temp_dir:
        tpi, cf, full, cmap, traces, meta, audits, pairs, segs, norm = u.pooled.load_population(Path(temp_dir))
        md = u.ppv.frame(traces, meta).reset_index(drop=True)
        if not np.array_equal(md["branch_id"].astype(str).to_numpy(), data["branch_id"].astype(str).to_numpy()):
            raise RuntimeError("archive/inference row alignment failed")
        scenarios = {"active": point_mu, "gt": data["mu"].to_numpy(np.float32)}
        scenarios.update({f"prior_{i}": np.full(len(data), x, np.float32) for i, x in enumerate(prior)})
        probabilities = u.infer_direct_scenarios(full, traces, segs, data["fold"].to_numpy(int), scenarios)

    force = data["force_N"].to_numpy(float)
    fmax = data["task_Fmax_N"].to_numpy(float)
    p_active = probabilities["active"]
    p_gt = probabilities["gt"]
    p_prior_members = np.stack([probabilities[f"prior_{i}"] for i in range(len(prior))])
    u_prior_members = np.stack([u.utility_score(p, force, fmax) for p in p_prior_members])

    result = data[[
        "branch_id", "context_id", "root_id", "task", "repeat", "force_N", "mu",
        "success", "fold", "task_Fmax_N",
    ]].copy()
    result["predicted_mu_active"] = point_mu
    result["p_success_active"] = p_active
    result["expected_utility_active"] = u.utility_score(p_active, force, fmax)
    result["p_success_gt_friction"] = p_gt
    result["expected_utility_gt_friction"] = u.utility_score(p_gt, force, fmax)
    result["p_success_query_ignored"] = p_prior_members.mean(axis=0)
    result["expected_utility_query_ignored"] = u_prior_members.mean(axis=0)

    for prefix in ("active", "gt_friction", "query_ignored"):
        score_col = f"expected_utility_{prefix}"
        result[f"selected_{prefix}"] = 0
        for _, group in result.groupby(["context_id", "repeat"], sort=True):
            ordered = group.sort_values("force_N", kind="mergesort")
            best = ordered[score_col].max()
            idx = ordered[np.isclose(ordered[score_col], best, rtol=0.0, atol=1e-12)].index[0]
            result.loc[idx, f"selected_{prefix}"] = 1

    max_abs = float(np.max(np.abs(result["p_success_gt_friction"] - data["p_D_OOF_ensemble"])))
    if max_abs > 1e-6:
        raise RuntimeError(f"GT Direct reproduction mismatch: {max_abs}")
    result.to_csv(OUT / "E1_CANDIDATE_DECISION_CHAIN.csv", index=False)
    (OUT / "E1_DECISION_CHAIN_QA.json").write_text(
        json.dumps(
            {
                "rows": len(result),
                "unique_branches": int(result.branch_id.nunique()),
                "paired_episodes": int(result.groupby(["context_id", "repeat"]).ngroups),
                "candidates_per_episode": sorted(result.groupby(["context_id", "repeat"]).size().unique().tolist()),
                "gt_direct_reproduction_max_abs_error": max_abs,
                "fitted_or_tuned": False,
                "gpu_used": False,
                "sealed_test_read": False,
                "status": "PASS_FROZEN_CPU_RECONSTRUCTION",
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
