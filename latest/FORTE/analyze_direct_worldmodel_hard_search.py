#!/usr/bin/env python3
"""Post-specified native hard-decision search for the trajectory evaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FORTE = Path("/home/exouser/FORTE")
DIRECT_PRED = FORTE / "joint_decision_alignment_20260831_200441" / "JOINT_DECISIONALIGNED_ALL_PREDICTIONS.csv"
FMAX = {0: 5.0, 1: 6.0, 5: 5.0, 6: 4.0}
NATIVE_THRESHOLD = 0.5  # sigmoid(logit)>0.5 iff logit>0


def summarize(q: pd.DataFrame, name: str, total: int) -> dict:
    return {
        "policy": name,
        "total_contexts": total,
        "finite_decisions": int(len(q)),
        "finite_decision_rate": float(len(q) / total),
        "world_model_validated_rate": float(q.evaluator_binary_success.mean()),
        "success_rate": float(q.actual_success_rate.mean()),
        "under_force_rate": float(q.under_force.mean()),
        "mean_force_N": float(q.selected_force_N.mean()),
        "realized_utility": float(q.realized_utility.mean()),
        "changed_from_direct_rate": float(q.changed_from_direct.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    args = ap.parse_args()
    out = args.artifact
    world = pd.read_csv(out / "WORLD_MODEL_VERIFIER_DEV_PREDICTIONS.csv")
    direct = pd.read_csv(DIRECT_PRED)
    direct = direct[(direct.fraction == "100%") & (direct.method == "Direct")]
    dc = direct.groupby(["context_id", "scene_id", "task", "force_N"], as_index=False).agg(
        p_direct=("p_success", "mean"), actual_success=("actual_success", "mean")
    )
    wc = world.groupby(["context_id", "force_N"], as_index=False).agg(evaluator_score=("p_world", "mean"))
    cells = dc.merge(wc, on=["context_id", "force_N"], validate="one_to_one")
    selections = []
    gates = []
    total = int(cells.context_id.nunique())
    for cid, q0 in cells.groupby("context_id"):
        q = q0.sort_values("force_N").copy()
        fmax = FMAX[int(q.task.iloc[0])]
        q["direct_utility"] = q.p_direct * (fmax - q.force_N) + (1 - q.p_direct) * (-fmax)
        q["realized_utility"] = q.actual_success * (fmax - q.force_N) + (1 - q.actual_success) * (-fmax)
        boundary = float(q.loc[q.actual_success > 0, "force_N"].min()) if (q.actual_success > 0).any() else np.nan
        proposal = q.sort_values(["direct_utility", "force_N"], ascending=[False, True]).iloc[0]
        eligible = q[(q.force_N >= float(proposal.force_N) - 1e-9) & (q.evaluator_score > NATIVE_THRESHOLD)].sort_values("force_N")
        finite = bool(len(eligible))
        chosen = eligible.iloc[0] if finite else q.iloc[-1]
        base = {
            "context_id": cid,
            "scene_id": chosen.scene_id,
            "task": int(chosen.task),
            "direct_proposal_force_N": float(proposal.force_N),
            "selected_force_N": float(chosen.force_N),
            "evaluator_binary_success": int(finite),
            "evaluator_internal_score_audit_only": float(chosen.evaluator_score),
            "actual_success_rate": float(chosen.actual_success),
            "realized_utility": float(chosen.realized_utility),
            "under_force": int(np.isfinite(boundary) and float(chosen.force_N) < boundary),
            "changed_from_direct": int(abs(float(chosen.force_N) - float(proposal.force_N)) > 1e-9),
        }
        selections.append({**base, "policy": "HARD_SEARCH_WITH_EXPLICIT_MAX_FALLBACK"})
        if finite:
            selections.append({**base, "policy": "HARD_SEARCH_STRICT_FINITE_ONLY"})
        gates.append({
            "context_id": cid,
            "proposal_actual_success_rate": float(proposal.actual_success),
            "proposal_evaluator_internal_score_audit_only": float(proposal.evaluator_score),
            "proposal_binary_success": int(float(proposal.evaluator_score) > NATIVE_THRESHOLD),
            "any_higher_binary_success": int(finite),
        })
    s = pd.DataFrame(selections)
    g = pd.DataFrame(gates)
    strict = s[s.policy == "HARD_SEARCH_STRICT_FINITE_ONLY"]
    fallback = s[s.policy == "HARD_SEARCH_WITH_EXPLICIT_MAX_FALLBACK"]
    summary = {
        "status": "POST_SPECIFIED_RETROSPECTIVE_DIAGNOSTIC",
        "classifier_contract": "world model outputs H8 trajectory; evaluator returns SUCCESS iff logit>0; controller searches only upward",
        "probability_not_exposed_to_controller": True,
        "native_binary_boundary": "logit>0 (sigmoid score>0.5, stored audit-only)",
        "strict": summarize(strict, "HARD_SEARCH_STRICT_FINITE_ONLY", total),
        "max_fallback": summarize(fallback, "HARD_SEARCH_WITH_EXPLICIT_MAX_FALLBACK", total),
        "initial_gate": {
            "proposal_success": int(g.proposal_binary_success.sum()),
            "proposal_failure": int(total - g.proposal_binary_success.sum()),
            "correctly_rejected_imperfect": int(((g.proposal_binary_success == 0) & (g.proposal_actual_success_rate < 1)).sum()),
            "incorrectly_accepted_imperfect": int(((g.proposal_binary_success == 1) & (g.proposal_actual_success_rate < 1)).sum()),
            "incorrectly_rejected_perfect": int(((g.proposal_binary_success == 0) & (g.proposal_actual_success_rate == 1)).sum()),
        },
        "scope": "retrospective pooled fixed-scene DEV; not TEST or model selection",
    }
    s.to_csv(out / "DIRECT_WORLD_MODEL_HARD_SEARCH_SELECTIONS.csv", index=False)
    g.to_csv(out / "DIRECT_WORLD_MODEL_HARD_SEARCH_GATE.csv", index=False)
    (out / "DIRECT_WORLD_MODEL_HARD_SEARCH_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "DIRECT_WORLD_MODEL_HARD_SEARCH_REPORT.md").write_text(
        "# Direct utility → predicted trajectory → hard success search\n\n"
        "The world model outputs only an H8 physical trajectory. The separately trained trajectory evaluator emits a hard SUCCESS/FAIL decision at its native logit-zero boundary. The controller does not consume probability. Starting at the Direct-utility proposal, only higher forces are searched.\n\n"
        f"Strict finite decisions: {len(strict)}/{total}; finite SR={summary['strict']['success_rate']:.3f}, under-force={summary['strict']['under_force_rate']:.3f}, mean force={summary['strict']['mean_force_N']:.3f} N, realized utility={summary['strict']['realized_utility']:.3f}.\n\n"
        f"With explicitly labeled maximum-force fallback: SR={summary['max_fallback']['success_rate']:.3f}, under-force={summary['max_fallback']['under_force_rate']:.3f}, mean force={summary['max_fallback']['mean_force_N']:.3f} N, realized utility={summary['max_fallback']['realized_utility']:.3f}; world-model-validated rate remains {summary['max_fallback']['world_model_validated_rate']:.3f}.\n\n"
        "This rule was specified after inspecting the earlier p>0.8 diagnostic and is therefore retrospective only.\n",
        encoding="utf-8",
    )
    derived = [
        out / "DIRECT_WORLD_MODEL_HARD_SEARCH_SELECTIONS.csv",
        out / "DIRECT_WORLD_MODEL_HARD_SEARCH_GATE.csv",
        out / "DIRECT_WORLD_MODEL_HARD_SEARCH_SUMMARY.json",
        out / "DIRECT_WORLD_MODEL_HARD_SEARCH_REPORT.md",
    ]
    (out / "HARD_SEARCH_SHA256SUMS.txt").write_text(
        "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in derived),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
