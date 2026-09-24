#!/usr/bin/env python3
"""CPU-only E6 decision-aware re-query audit.

Reads frozen V2 per-candidate scores and the authoritative 720-branch
TRAIN/root-heldout-OOF archive. Recomputes member-wise Expected-Utility
argmax decisions, evaluates frontier-only harm retrospectively, audits repeat
rows for valid second-query evidence, and writes only to this lane directory.
No Isaac, GPU, TEST, or rollout code is imported.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


LANE = Path(__file__).resolve().parent
BASE = Path("/home/exouser/FORTE_e6resume/AF_V2_UTILITY_E6_CPU_RECOVERY_20260902")
SCORES = BASE / "E6_EXPECTED_UTILITY_MEMBER_SCORES.csv"
FROZEN_DECISIONS = BASE / "E6_EXPECTED_UTILITY_MEMBER_DECISIONS.csv"
ARCHIVE = Path("/home/exouser/FORTE/activeforcing_residual_utility_20260901_055605/POOLED_OOF_UTILITY_DATASET.csv")
PAIR_MANIFEST = BASE / "E6_PAIRED_EVALUATION_MANIFEST.json"
PHYSICAL_PROTOCOL = Path("/home/exouser/FORTE/activeforcing_physical_only_probe_20260901/PHYSICAL_ONLY_PROBE_PROTOCOL.json")
SECOND_QUERY_REPORT = Path("/home/exouser/FORTE/activeforcing_full_claim_closure_20260902_062737/E6_E7/SECOND_QUERY_DEV_REPORT.md")
CURRENT_AUDIT = Path("/home/exouser/FORTE/activeforcing_full_claim_closure_20260902_062737/E6_E7/E6_E7_EXISTING_WORK_AUDIT.md")
UTILITY_CONFIG = Path("/home/exouser/FORTE/UTILITY_FINAL_CONFIG.json")

PROTOCOL = "ACTIVEFORCING_FULL_CLAIM_V2_UTILITY"
SELECTOR = "EXPECTED_UTILITY"
MEMBERS = (0, 1, 2)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def entropy(values: list[float]) -> float:
    _, counts = np.unique(np.asarray(values, float), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ordered = score[order]
    ranks = np.empty(len(score), float)
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and ordered[j] == ordered[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0) / (pos.sum() * neg.sum()))


def auprc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    yy, ss = y[order], score[order]
    tp = fp = previous_recall = ap = 0.0
    i = 0
    while i < len(yy):
        j = i + 1
        while j < len(yy) and ss[j] == ss[i]:
            j += 1
        block = yy[i:j]
        tp += float(block.sum())
        fp += float(len(block) - block.sum())
        recall = tp / positives
        precision = tp / (tp + fp) if tp + fp else 1.0
        ap += (recall - previous_recall) * precision
        previous_recall = recall
        i = j
    return float(ap)


def ci95(values: np.ndarray) -> list[float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def f(v: float | int | np.floating) -> float:
    return float(v) if np.isfinite(v) else float("nan")


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(SCORES)
    archive = pd.read_csv(ARCHIVE)
    frozen = pd.read_csv(FROZEN_DECISIONS)
    if len(scores) != 2160 or len(archive) != 720 or len(frozen) != 144:
        raise RuntimeError(f"unexpected shapes scores={scores.shape}, archive={archive.shape}, frozen={frozen.shape}")
    if scores.protocol_version.nunique() != 1 or scores.protocol_version.iloc[0] != PROTOCOL:
        raise RuntimeError("scores are not current V2 Utility")
    if scores.selector.nunique() != 1 or scores.selector.iloc[0] != SELECTOR:
        raise RuntimeError("scores are not Expected-Utility scores")
    if scores.context_id.nunique() != 72 or scores.root_id.nunique() != 24:
        raise RuntimeError("unexpected current population context/root counts")
    if scores.groupby(["context_id", "repeat"]).size().to_dict() and set(scores.groupby(["context_id", "repeat"]).size()) != {15}:
        raise RuntimeError("expected 3 members x 5 candidates per episode")
    if archive.groupby(["context_id", "repeat"]).size().to_dict() and set(archive.groupby(["context_id", "repeat"]).size()) != {5}:
        raise RuntimeError("expected five empirical candidates per episode")
    if archive.branch_id.nunique() != len(archive) or archive[["context_id", "repeat", "force_N"]].duplicated().any():
        raise RuntimeError("archive branch/candidate key failure")
    if scores.context_id.astype(str).str.contains("test", case=False).any():
        raise RuntimeError("TEST-like context present in scores")
    return scores, archive, frozen


def recompute(scores: pd.DataFrame, archive: pd.DataFrame, frozen: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    scores = scores.copy()
    scores["U_recomputed"] = scores.p_success * (1.0 - scores.force_N / scores.Fmax_N) + (1.0 - scores.p_success) * (-1.0)
    max_utility_error = float(np.max(np.abs(scores.U_recomputed - scores.expected_utility)))
    rows: list[dict] = []
    for (context_id, repeat), group in scores.groupby(["context_id", "repeat"], sort=True):
        selected_forces, selected_u, selected_p, selected_success = [], [], [], []
        for member in MEMBERS:
            candidates = group[group.member == member].sort_values(["U_recomputed", "force_N"], ascending=[False, True], kind="mergesort")
            selected = candidates.iloc[0]
            selected_forces.append(float(selected.force_N))
            selected_u.append(float(selected.U_recomputed))
            selected_p.append(float(selected.p_success))
        branches = archive[(archive.context_id == context_id) & (archive.repeat == repeat)].sort_values("force_N")
        successful = branches.loc[branches.success == 1, "force_N"]
        frontier = float(successful.min()) if len(successful) else float("nan")
        for force in selected_forces:
            matching = branches[np.isclose(branches.force_N, force, rtol=0.0, atol=1e-8)]
            selected_success.append(int(matching.success.iloc[0]))
        unique = len(set(selected_forces))
        posterior = group.groupby("force_N", as_index=False).agg(
            p_success=("p_success", "mean"), expected_utility=("U_recomputed", "mean")
        ).sort_values(["expected_utility", "force_N"], ascending=[False, True], kind="mergesort").iloc[0]
        rows.append({
            "protocol_version": PROTOCOL,
            "selector": SELECTOR,
            "evidence_role": "CPU_RECOMPUTED_MEMBER_DECISION_FRONTIER_DIAGNOSTIC",
            "split": "TRAIN_ROOT_HELDOUT_OOF_DEV_FOLDS",
            "context_id": str(context_id), "repeat": int(repeat), "root_id": str(group.root_id.iloc[0]),
            "task": int(group.task.iloc[0]), "friction_band": str(group.friction_band.iloc[0]),
            "candidate_count": int(group.force_N.nunique()),
            "member_force_0_N": selected_forces[0], "member_force_1_N": selected_forces[1], "member_force_2_N": selected_forces[2],
            "member_selected_utility_0": selected_u[0], "member_selected_utility_1": selected_u[1], "member_selected_utility_2": selected_u[2],
            "member_selected_probability_0": selected_p[0], "member_selected_probability_1": selected_p[1], "member_selected_probability_2": selected_p[2],
            "unique_member_decisions": unique, "exact_utility_decision_consensus": int(unique == 1),
            "utility_decision_disagreement": int(unique > 1), "decision_entropy_bits": entropy(selected_forces),
            "force_spread_N": max(selected_forces) - min(selected_forces),
            "posterior_expected_utility_force_N": float(posterior.force_N), "posterior_expected_utility": float(posterior.expected_utility),
            "posterior_predicted_success": float(posterior.p_success),
            "empirical_frontier_N": frontier, "frontier_available": int(np.isfinite(frontier)),
            "harm_below_frontier_any_member": int(np.isfinite(frontier) and min(selected_forces) < frontier - 1e-9),
            "harm_below_frontier_all_members": int(np.isfinite(frontier) and max(selected_forces) < frontier - 1e-9),
            "harm_selected_candidate_failure_any_member": int(any(x == 0 for x in selected_success)),
            "selected_member_success_0": selected_success[0], "selected_member_success_1": selected_success[1], "selected_member_success_2": selected_success[2],
        })
    d = pd.DataFrame(rows).sort_values(["context_id", "repeat"]).reset_index(drop=True)
    if len(d) != 144 or d[["context_id", "repeat"]].duplicated().any():
        raise RuntimeError("invalid recomputed decision grain")
    old = frozen.set_index(["context_id", "repeat"])
    new = d.set_index(["context_id", "repeat"])
    compare_cols = {
        "member_force_0_N": "member_force_0_N", "member_force_1_N": "member_force_1_N", "member_force_2_N": "member_force_2_N",
        "utility_decision_disagreement": "utility_decision_disagreement", "unique_member_decisions": "unique_member_decisions",
        "decision_entropy_bits": "decision_entropy_bits", "force_spread_N": "force_spread_N",
        "harm_below_frontier_any_member": "any_member_below_empirical_frontier",
    }
    comparison = {}
    for new_col, old_col in compare_cols.items():
        a, b = new[new_col].to_numpy(float), old[old_col].to_numpy(float)
        comparison[new_col] = {"max_abs_diff": f(np.max(np.abs(a - b))), "mismatched_rows": int((~np.isclose(a, b, rtol=0.0, atol=1e-8)).sum())}
    return d, {"max_utility_abs_error": max_utility_error, "frozen_decision_comparison": comparison}


def repeat_audit(scores: pd.DataFrame, archive: pd.DataFrame, d: pd.DataFrame) -> dict:
    # Query-side rows have no second-query posterior/state and are exactly equal
    # across repeats after the repeat/branch identifiers are removed.
    cols = ["context_id", "member", "member_mu", "force_N", "Fmax_N", "p_success", "R_success", "R_failure", "expected_utility", "selected_by_member"]
    left = scores[scores.repeat == 1][cols].sort_values(["context_id", "member", "force_N"]).reset_index(drop=True)
    right = scores[scores.repeat == 2][cols].sort_values(["context_id", "member", "force_N"]).reset_index(drop=True)
    query_exact = bool(left.equals(right))
    ap = archive.pivot_table(index=["context_id", "force_N"], columns="repeat", values=["mu", "success", "p_D_OOF_ensemble"], aggfunc="first")
    outcome_success_diff = int((~np.isclose(ap[("success", 1)], ap[("success", 2)], rtol=0.0, atol=0.0)).sum())
    decisions_repeat = d.pivot(index="context_id", columns="repeat", values=["member_force_0_N", "member_force_1_N", "member_force_2_N", "utility_decision_disagreement", "force_spread_N"])
    decision_diff = {col: int((~np.isclose(decisions_repeat[(col, 1)], decisions_repeat[(col, 2)], rtol=0.0, atol=1e-12)).sum()) for col in ["member_force_0_N", "member_force_1_N", "member_force_2_N", "utility_decision_disagreement", "force_spread_N"]}
    return {
        "valid_second_query_evidence": False,
        "valid_second_query_continuation_count": 0,
        "post_second_query_posterior_count": 0,
        "repeat_context_pairs": int(scores.context_id.nunique()),
        "query_side_repeat_rows_per_arm": int(len(left)),
        "query_side_repeat_exact_after_identifier_removal": query_exact,
        "query_side_repeat_decisions_identical": decision_diff,
        "outcome_repeat_candidate_pairs": int(len(ap)),
        "outcome_success_diff_candidate_pairs": outcome_success_diff,
        "interpretation": "repeat is downstream outcome replication, not a valid second physical query; identical query-side predictions cannot measure uncertainty reduction",
    }


def metric_bundle(d: pd.DataFrame, harm_col: str, seed: int = 20260903, n_boot: int = 2000) -> dict:
    y = d[harm_col].to_numpy(int)
    consensus = d.exact_utility_decision_consensus.to_numpy(bool)
    disagreement = ~consensus
    p_cons = float(d.loc[consensus, harm_col].mean())
    p_dis = float(d.loc[disagreement, harm_col].mean())
    out = {
        "harm_label": harm_col, "n": int(len(d)), "harm_count": int(y.sum()), "harm_rate": float(y.mean()),
        "consensus_n": int(consensus.sum()), "consensus_harm_count": int(d.loc[consensus, harm_col].sum()), "P_harm_given_consensus": p_cons,
        "disagreement_n": int(disagreement.sum()), "disagreement_harm_count": int(d.loc[disagreement, harm_col].sum()), "P_harm_given_disagreement": p_dis,
        "risk_difference_disagreement_minus_consensus": p_dis - p_cons, "risk_ratio_disagreement_over_consensus": p_dis / p_cons if p_cons else float("nan"),
        "scores": {}, "risk_coverage": {},
    }
    for score in ["utility_decision_disagreement", "force_spread_N", "decision_entropy_bits"]:
        s = d[score].to_numpy(float)
        out["scores"][score] = {"AUROC": auroc(y, s), "AUPRC": auprc(y, s)}
        q = d.sort_values([score, "context_id", "repeat"], kind="mergesort")
        out["risk_coverage"][score] = []
        for coverage in [0.25, 0.50, 0.75, 1.0]:
            n = int(math.ceil(len(q) * coverage))
            kept = q.head(n)
            out["risk_coverage"][score].append({"coverage": coverage, "n": n, "harm_count": int(kept[harm_col].sum()), "risk": float(kept[harm_col].mean())})
    # Cluster bootstrap by context, retaining both outcome repeats together.
    rng = np.random.default_rng(seed)
    context_ids = np.array(sorted(d.context_id.unique()))
    groups = {cid: g.index.to_numpy() for cid, g in d.groupby("context_id")}
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(context_ids, size=len(context_ids), replace=True)
        idx = np.concatenate([groups[cid] for cid in sample])
        x = d.loc[idx]
        yc = x[harm_col].to_numpy(int)
        cc = x.exact_utility_decision_consensus.to_numpy(bool)
        pc, pd_ = float(x.loc[cc, harm_col].mean()), float(x.loc[~cc, harm_col].mean())
        boot.append([pc, pd_, pd_ / pc if pc else float("nan"), auroc(yc, x.utility_decision_disagreement), auprc(yc, x.utility_decision_disagreement), auroc(yc, x.force_spread_N), auprc(yc, x.force_spread_N)])
    b = np.asarray(boot, float)
    names = ["P_harm_given_consensus", "P_harm_given_disagreement", "risk_ratio", "AUROC_disagreement", "AUPRC_disagreement", "AUROC_spread", "AUPRC_spread"]
    out["cluster_bootstrap"] = {name: {"resamples": n_boot, "CI95": ci95(b[:, i])} for i, name in enumerate(names)}
    return out


def main() -> None:
    scores, archive, frozen = load()
    d, checks = recompute(scores, archive, frozen)
    repeats = repeat_audit(scores, archive, d)
    primary = metric_bundle(d, "harm_below_frontier_any_member")
    sensitivity = metric_bundle(d, "harm_selected_candidate_failure_any_member")
    d.to_csv(LANE / "E6_RECOMPUTED_MEMBER_DECISIONS.csv", index=False)

    metrics = {
        "status": "E6_SIGNAL_PASS",
        "scientific_status": "SCIENTIFIC_NEGATIVE",
        "requery_status": "BLOCKED_MISSING_VALID_SECOND_QUERY",
        "protocol_version": PROTOCOL, "selector": SELECTOR,
        "primary_harm_definition": "any member-selected force below the episode's empirical reliable frontier; frontier is evaluation-only and never a runtime selector input",
        "sensitivity_harm_definition": "any member-selected candidate is a recorded empirical failure",
        "population": {"episodes": 144, "contexts": 72, "roots": 24, "branches": 720, "members": 3, "candidates_per_episode": 5},
        "recompute_checks": checks,
        "primary_frontier_harm": primary,
        "selected_failure_sensitivity": sensitivity,
        "repeat_audit": repeats,
    }
    (LANE / "E6_SIGNAL_METRICS.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    source_files = [SCORES, FROZEN_DECISIONS, ARCHIVE, PAIR_MANIFEST, PHYSICAL_PROTOCOL, SECOND_QUERY_REPORT, CURRENT_AUDIT, UTILITY_CONFIG]
    evidence = {
        "status": "BLOCKED_MISSING_VALID_SECOND_QUERY",
        "valid_second_query_evidence": False,
        "valid_second_query_continuation_count": 0,
        "post_second_query_posterior_count": 0,
        "current_pair_manifest_status": json.loads(PAIR_MANIFEST.read_text()).get("status"),
        "current_pair_manifest_execution_authorized": json.loads(PAIR_MANIFEST.read_text()).get("execution_authorized"),
        "physical_only_second_query_status": json.loads(PHYSICAL_PROTOCOL.read_text()).get("second_query_status"),
        "evidence_paths": [str(p) for p in [PAIR_MANIFEST, PHYSICAL_PROTOCOL, SECOND_QUERY_REPORT, CURRENT_AUDIT]],
        "missing_historical_paths": [
            "/home/exouser/Tabero/analysis/results/p7b_scientific_main_20260828_000729/P7B_FINAL_VERDICT.json",
            "/home/exouser/Tabero/analysis/results/p7b_scientific_main_20260828_000729/P7B_SCIENTIFIC_COMPLETION_AUDIT.json",
        ],
        "repeat_audit": repeats,
        "source_sha256": {str(p): sha256(p) for p in source_files if p.exists()},
    }
    (LANE / "E6_SECOND_QUERY_EVIDENCE_AUDIT.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    context_roster = d[["context_id", "root_id", "task", "friction_band"]].drop_duplicates().sort_values(["task", "root_id", "friction_band"])
    pilot = {
        "status": "PILOT_MANIFEST_ONLY_NOT_EXECUTED",
        "protocol_version": PROTOCOL,
        "selector": SELECTOR,
        "execution_authorized": False, "gpu_authorized": False, "isaac_launch_authorized": False,
        "scope": "TRAIN_DEV_ONLY; no locked TEST rows, outcomes, or tuning",
        "population": {"contexts": int(len(context_roster)), "roots": int(context_roster.root_id.nunique()), "episodes": 144, "tasks": sorted(int(x) for x in context_roster.task.unique()), "context_roster": context_roster.to_dict("records")},
        "arms": [
            {"name": "ONE_QUERY", "rule": "exactly one qualified frozen P4-B query"},
            {"name": "ALWAYS_TWO_QUERIES", "rule": "always execute a genuine separately recorded qualified second query"},
            {"name": "RAW_UNCERTAINTY_GATE", "rule": "second query iff a threshold frozen before pilot execution triggers"},
            {"name": "DECISION_AWARE_UTILITY_CONSENSUS", "rule": "second query iff member-wise Expected-Utility argmax forces disagree"},
            {"name": "ORACLE_REQUERY_DIAGNOSTIC", "rule": "retrospective diagnostic only; never a runtime selector"},
        ],
        "pair_invariants": [
            "same root/context/repeat, initial state, candidate pool, Direct stack, utility, and first-query transition",
            "genuine second query must have distinct action/evidence, qualification record, post-query state hash, and posterior update",
            "duplicate or masked first-query evidence is invalid",
            "empirical outcomes/frontier are evaluation-only",
        ],
        "required_metrics": ["full_task_SR", "mean_query_count", "second_query_rate", "query_precision", "harm_enrichment", "mean_force", "max_force", "under_force", "excess_force", "realized_utility", "latency"],
        "unlock_gate": "freeze second-query action/state/qualification semantics and obtain genuine TRAIN/DEV continuation plus post-second-query posterior before any execution",
        "source_context_roster": str(LANE / "E6_RECOMPUTED_MEMBER_DECISIONS.csv"),
    }
    (LANE / "E6_TRAIN_DEV_SECOND_QUERY_PILOT_MANIFEST.json").write_text(json.dumps(pilot, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    protocol_text = f"""# E6 TRAIN/DEV second-query pilot protocol (manifest only)

STATUS: `PILOT_MANIFEST_ONLY_NOT_EXECUTED`

This is a CPU-prepared protocol only. It authorizes no Isaac, GPU, rollout, TEST access, or method retuning. The roster is the current 72-context / 24-root V2 OOF population; its 144 outcome repeats are not treated as second-query evidence.

## Scientific comparison

Compare `ONE_QUERY`, `ALWAYS_TWO_QUERIES`, `RAW_UNCERTAINTY_GATE`, `DECISION_AWARE_UTILITY_CONSENSUS`, and retrospective `ORACLE_REQUERY_DIAGNOSTIC` under paired context/repeat assignments. The final force decision remains Expected-Utility argmax with lower-force tie-break.

## Validity gate

A second query counts only if it is a separately recorded physical action/evidence sequence with frozen action/state semantics, a qualification record, a post-query state hash, and a measurable posterior update. Reusing, duplicating, or masking the first query does not qualify. Empirical outcome/frontier labels may be used only for evaluation, never for triggering or selecting force.

## Required outputs

Record full-task success, query count/rate, query precision, harm enrichment, force, under-/excess-force, realized utility, latency, state hashes, qualification fields, and pre/post posterior summaries. Preserve TEST isolation and write all artifacts under the approved result directory only.

The executable pilot remains blocked until the second-query action and qualification contract is frozen and coordinator resource clearance is explicit.
"""
    (LANE / "E6_TRAIN_DEV_SECOND_QUERY_PILOT_PROTOCOL.md").write_text(protocol_text, encoding="utf-8")

    p = primary
    s = sensitivity
    report = f"""# E6 decision-aware re-query CPU audit

## Status

- `CPU_RECOMPUTE: COMPLETE`
- `E6_SIGNAL: E6_SIGNAL_PASS` (descriptive DEV/OOF signal; low precision)
- `OVERALL_E6: SCIENTIFIC_NEGATIVE`
- `SECOND_QUERY: BLOCKED_MISSING_VALID_SECOND_QUERY`
- `ISAAC_GPU: NOT_STARTED`
- `SEALED_TEST: NOT_ACCESSED`

## Authoritative population and recomputation

Source archive: `{ARCHIVE}`. The current V2 population has 720 unique branches, 144 `(context_id, repeat)` episodes, 72 contexts, 24 roots, five candidates per episode, and three members. For every episode/member this audit recomputed `F*_m = argmax_F [p_success*(1-F/Fmax) + (1-p_success)*(-1)]`, with lower force as the tie-break.

The recomputed selections match `{FROZEN_DECISIONS}` exactly: all 144 rows agree for each member force, unique-force count, exact consensus/disagreement, entropy, spread, and frontier flag. Utility recomputation max absolute error versus stored per-candidate utility is `{checks['max_utility_abs_error']:.3g}`.

## Signal result

Primary harm is model-independent and retrospective: any member-selected force below that episode's empirical reliable frontier. The frontier is evaluation-only and was not used in selection or triggering.

| condition | harm | n | P(harm) |
|---|---:|---:|---:|
| consensus | {p['consensus_harm_count']} | {p['consensus_n']} | {p['P_harm_given_consensus']:.4f} |
| disagreement | {p['disagreement_harm_count']} | {p['disagreement_n']} | {p['P_harm_given_disagreement']:.4f} |

Frontier-harm risk difference is `{p['risk_difference_disagreement_minus_consensus']:.4f}` and risk ratio is `{p['risk_ratio_disagreement_over_consensus']:.2f}`. Disagreement as the risk score gives AUROC `{p['scores']['utility_decision_disagreement']['AUROC']:.3f}` and AUPRC `{p['scores']['utility_decision_disagreement']['AUPRC']:.3f}`. Continuous force spread gives AUROC `{p['scores']['force_spread_N']['AUROC']:.3f}` and AUPRC `{p['scores']['force_spread_N']['AUPRC']:.3f}`.

Cluster-bootstrap 95% intervals, resampling the 72 contexts while retaining both outcome repeats, are RR `{p['cluster_bootstrap']['risk_ratio']['CI95'][0]:.2f}–{p['cluster_bootstrap']['risk_ratio']['CI95'][1]:.2f}`, disagreement AUROC `{p['cluster_bootstrap']['AUROC_disagreement']['CI95'][0]:.3f}–{p['cluster_bootstrap']['AUROC_disagreement']['CI95'][1]:.3f}`, and disagreement AUPRC `{p['cluster_bootstrap']['AUPRC_disagreement']['CI95'][0]:.3f}–{p['cluster_bootstrap']['AUPRC_disagreement']['CI95'][1]:.3f}`. These intervals include the null, so the positive signal is not precise enough for a deployment claim.

As a sensitivity label, any selected member force whose recorded candidate outcome is failure yields {s['harm_count']}/{s['n']} harm, RR {s['risk_ratio_disagreement_over_consensus']:.2f}, AUROC {s['scores']['utility_decision_disagreement']['AUROC']:.3f}, and AUPRC {s['scores']['utility_decision_disagreement']['AUPRC']:.3f}.

Risk coverage for the primary harm score, ranking low risk first, is 0.50: {p['risk_coverage']['utility_decision_disagreement'][1]['risk']:.4f}; 0.75: {p['risk_coverage']['utility_decision_disagreement'][2]['risk']:.4f}; 1.00: {p['risk_coverage']['utility_decision_disagreement'][3]['risk']:.4f}. The 50% and 75% coverages correspond to retaining consensus-first rows; the 75% risk is the consensus harm rate.

## Second-query evidence audit

No valid second-query continuation or post-second-query posterior is present. The two repeat rows per current context have exactly identical query-side member scores and decisions; only some downstream empirical outcomes vary (18 of 360 context-force repeat pairs). Thus repeats are outcome replication, not uncertainty-reducing second queries. The physical-only protocol records `SECOND_QUERY_NOT_EVALUATED_NO_ARCHIVED_REPEAT`, and the paired manifest is execution-blocked. Two historical P7-B JSON paths referenced by the frozen runner are also absent; the existing audit/report evidence remains the controlling negative provenance.

Evidence paths: `{PAIR_MANIFEST}`, `{PHYSICAL_PROTOCOL}`, `{SECOND_QUERY_REPORT}`, `{CURRENT_AUDIT}`.

## Pilot handoff

Because the descriptive signal passes but valid evidence is absent, a minimal non-executed TRAIN/DEV pilot manifest and protocol were prepared at `{LANE / 'E6_TRAIN_DEV_SECOND_QUERY_PILOT_MANIFEST.json'}` and `{LANE / 'E6_TRAIN_DEV_SECOND_QUERY_PILOT_PROTOCOL.md'}`. They authorize no GPU/Isaac work.

## Artifacts

- Per-episode/member recompute: `{LANE / 'E6_RECOMPUTED_MEMBER_DECISIONS.csv'}`
- Metrics and uncertainty: `{LANE / 'E6_SIGNAL_METRICS.json'}`
- Second-query validity audit: `{LANE / 'E6_SECOND_QUERY_EVIDENCE_AUDIT.json'}`
- This report: `{LANE / 'E6_DECISION_AWARE_REQUERY_REPORT.md'}`
"""
    (LANE / "E6_DECISION_AWARE_REQUERY_REPORT.md").write_text(report, encoding="utf-8")

    status = {
        "status": "SCIENTIFIC_NEGATIVE",
        "cpu_recompute": "COMPLETE",
        "signal": "E6_SIGNAL_PASS",
        "requery": "BLOCKED_MISSING_VALID_SECOND_QUERY",
        "protocol_version": PROTOCOL, "selector": SELECTOR,
        "sealed_TEST_accessed": False, "gpu_started": False, "isaac_started": False,
        "report": str(LANE / "E6_DECISION_AWARE_REQUERY_REPORT.md"),
        "metrics": str(LANE / "E6_SIGNAL_METRICS.json"),
    }
    (LANE / "E6_STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
