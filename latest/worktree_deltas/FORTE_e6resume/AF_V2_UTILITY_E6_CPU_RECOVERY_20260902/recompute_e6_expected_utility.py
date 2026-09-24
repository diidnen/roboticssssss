#!/usr/bin/env python3
"""CPU-only E6 Expected-Utility disagreement and second-query readiness audit.

This runner reuses the final V2 Utility contract, the frozen grouped-root OOF
Direct stack, the three OOF physical-belief members, and the exact five-force
archive.  It deliberately does not import any legacy hard-rho decision table,
rho selector, Isaac/IsaacLab runtime, or rollout launcher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LANE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = LANE_ROOT / "AF_V2_UTILITY_E6_CPU_RECOVERY_20260902"
FORTE_ROOT = Path("/home/exouser/FORTE")
TABERO_ROOT = Path("/home/exouser/Tabero")

# Import only the already-audited CPU utility implementation.  The imported
# module has no Isaac launch in its import path and exposes the frozen OOF
# inference helpers used by the final Utility ablation.
sys.path.insert(0, str(FORTE_ROOT))
import utility_and_causal_ablation_closure as utility  # noqa: E402
import pooled_joint_novisual_current as pooled  # noqa: E402
import run_pooled_predictive_verifier as ppv  # noqa: E402


PROTOCOL_VERSION = "ACTIVEFORCING_FULL_CLAIM_V2_UTILITY"
SELECTOR = "EXPECTED_UTILITY"
EVIDENCE_ROLE = "CPU_DEV_DIAGNOSTIC_NOT_FINAL_PAIRED_OUTCOME_EVIDENCE"
FMAX = {0: 5.0, 1: 6.0, 5: 5.0, 6: 4.0}
MEMBERS = (0, 1, 2)

UTILITY_CONFIG = FORTE_ROOT / "UTILITY_FINAL_CONFIG.json"
AUTHORITATIVE_PROTOCOL = FORTE_ROOT / "ACTIVEFORCING_AUTHORITATIVE_PROTOCOL.md"
CORRECTED_HANDOFF = FORTE_ROOT / "PROTOCOL_RECONCILIATION/CORRECTED_HANDOFF_E6E7.md"
GLOBAL_STATUS = FORTE_ROOT / "ACTIVEFORCING_GLOBAL_STATUS_AUDIT_20260902_133241/FINAL_STATUS_SUMMARY.md"
P7B_DIR = TABERO_ROOT / "analysis/results/p7b_scientific_main_20260828_000729"
P7B_VERDICT = P7B_DIR / "P7B_FINAL_VERDICT.json"
P7B_COMPLETION = P7B_DIR / "P7B_SCIENTIFIC_COMPLETION_AUDIT.json"
PHYSICAL_ONLY_PROTOCOL = FORTE_ROOT / "activeforcing_physical_only_probe_20260901/PHYSICAL_ONLY_PROBE_PROTOCOL.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def entropy_bits(values: list[float]) -> float:
    _, counts = np.unique(np.asarray(values, float), return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def load_and_validate_sources() -> tuple[pd.DataFrame, dict[int, np.ndarray], pd.DataFrame, dict[str, Any]]:
    required = [
        utility.ARCHIVE_DATA,
        utility.PROBE_PREDICTIONS,
        utility.PROBE_AUDIT,
        utility.UTILITY_PROTOCOL,
        UTILITY_CONFIG,
        AUTHORITATIVE_PROTOCOL,
        CORRECTED_HANDOFF,
        GLOBAL_STATUS,
        P7B_VERDICT,
        P7B_COMPLETION,
        PHYSICAL_ONLY_PROTOCOL,
    ]
    required.extend(
        utility.OOF_DIR / "checkpoints" / f"DIRECT_POOLED_fold{fold}_seed{seed}.pt"
        for fold in utility.FOLDS
        for seed in utility.DIRECT_SEEDS
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"missing offline prerequisites: {missing}")

    config = json.loads(UTILITY_CONFIG.read_text())
    if config.get("selection") != "argmax_candidate_expected_utility":
        raise RuntimeError("final Utility config is not Expected-Utility argmax")
    if config.get("sealed_TEST_used") is not False:
        raise RuntimeError("final Utility config does not preserve sealed TEST isolation")
    if {int(k): float(v) for k, v in config["task_Fmax_N"].items()} != FMAX:
        raise RuntimeError("task Fmax values differ from authoritative V2 Utility")
    if config.get("tie_break") != "lower_force":
        raise RuntimeError("unexpected Utility tie break")

    data = pd.read_csv(utility.ARCHIVE_DATA).reset_index(drop=True)
    if len(data) != 720 or data.branch_id.nunique() != 720:
        raise RuntimeError("expected 720 unique archived branches")
    if data.groupby(["context_id", "repeat"]).size().value_counts().to_dict() != {5: 144}:
        raise RuntimeError("expected exactly five candidates for each of 144 paired episodes")
    if set(data.task.unique()) != set(FMAX):
        raise RuntimeError("unexpected task population")
    if data.duplicated("branch_id").any() or data[["context_id", "repeat", "force_N"]].duplicated().any():
        raise RuntimeError("duplicate branch or candidate key")
    data["task_Fmax_N"] = data.task.map(FMAX).astype(float)
    data["friction_band"] = data.context_id.astype(str).str.extract(
        r"_(low|mid|high)_", expand=False
    ).str.upper()
    if data.friction_band.isna().any():
        raise RuntimeError("could not recover friction band from every context ID")

    member_mu, member_pivot = utility.load_probe_members(data.context_id)
    source_hashes = {str(path): sha256_file(path) for path in required}
    return data, member_mu, member_pivot, {"config": config, "source_hashes": source_hashes}


def infer_member_probabilities(data: pd.DataFrame, member_mu: dict[int, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
    with tempfile.TemporaryDirectory(prefix="e6_v2_utility_cpu_", dir="/tmp") as temp_dir:
        _, _, full, _, traces, meta, audits, _, segs, _ = pooled.load_population(Path(temp_dir))
        metadata = ppv.frame(traces, meta).reset_index(drop=True)
        if not np.array_equal(metadata.branch_id.astype(str).to_numpy(), data.branch_id.astype(str).to_numpy()):
            raise RuntimeError("archive and frozen Direct inference population are not row-aligned")
        if any(bool(audits[task].get("TEST_read", True)) for task in FMAX):
            raise RuntimeError("source audit indicates TEST access")
        scenarios = {f"member_{member}": member_mu[member] for member in MEMBERS}
        scenarios["gt_reproduction_check"] = data.mu.to_numpy(np.float32)
        probabilities = utility.infer_direct_scenarios(
            full,
            traces,
            segs,
            data.fold.to_numpy(int),
            scenarios,
        )
    reproduction_error = float(
        np.max(np.abs(probabilities["gt_reproduction_check"] - data.p_D_OOF_ensemble.to_numpy(float)))
    )
    if reproduction_error > 1e-6:
        raise RuntimeError(f"frozen Direct reproduction mismatch: {reproduction_error:.3g}")
    return probabilities, reproduction_error


def candidate_pool_hash(data: pd.DataFrame) -> str:
    serialized = [
        {
            "context_id": str(context_id),
            "repeat": int(repeat),
            "forces_N": [float(x) for x in group.sort_values("force_N").force_N],
        }
        for (context_id, repeat), group in data.groupby(["context_id", "repeat"], sort=True)
    ]
    return hashlib.sha256(canonical_json(serialized).encode()).hexdigest()


def score_and_select(
    data: pd.DataFrame,
    member_mu: dict[int, np.ndarray],
    probabilities: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    force = data.force_N.to_numpy(float)
    fmax = data.task_Fmax_N.to_numpy(float)
    score_frames = []
    for member in MEMBERS:
        probability = probabilities[f"member_{member}"]
        reward_success = 1.0 - force / fmax
        reward_failure = np.full(len(data), -1.0)
        expected_utility = probability * reward_success + (1.0 - probability) * reward_failure
        score_frames.append(
            pd.DataFrame(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "selector": SELECTOR,
                    "split": "TRAIN_ROOT_HELDOUT_OOF_DEV_FOLDS",
                    "evidence_role": EVIDENCE_ROLE,
                    "context_id": data.context_id.astype(str),
                    "repeat": data.repeat.astype(int),
                    "root_id": data.root_id.astype(str),
                    "task": data.task.astype(int),
                    "friction_band": data.friction_band.astype(str),
                    "branch_id": data.branch_id.astype(str),
                    "member": member,
                    "member_mu": member_mu[member].astype(float),
                    "force_N": force,
                    "Fmax_N": fmax,
                    "p_success": probability,
                    "R_success": reward_success,
                    "R_failure": reward_failure,
                    "expected_utility": expected_utility,
                }
            )
        )
    scores = pd.concat(score_frames, ignore_index=True)
    scores["selected_by_member"] = 0

    decision_rows: list[dict[str, Any]] = []
    grouped = scores.groupby(["context_id", "repeat"], sort=True)
    for (context_id, repeat), episode_scores in grouped:
        member_forces: list[float] = []
        member_utilities: list[float] = []
        member_probabilities: list[float] = []
        selected_score_indexes: list[int] = []
        for member in MEMBERS:
            candidates = episode_scores[episode_scores.member == member].sort_values("force_N", kind="mergesort")
            maximum = float(candidates.expected_utility.max())
            selected = candidates[np.isclose(candidates.expected_utility, maximum, rtol=0.0, atol=1e-12)].iloc[0]
            member_forces.append(float(selected.force_N))
            member_utilities.append(float(selected.expected_utility))
            member_probabilities.append(float(selected.p_success))
            selected_score_indexes.append(int(selected.name))
        scores.loc[selected_score_indexes, "selected_by_member"] = 1

        branches = data[(data.context_id.astype(str) == context_id) & (data.repeat.astype(int) == int(repeat))].sort_values("force_N")
        successful = branches[branches.success == 1]
        frontier = float(successful.force_N.min()) if len(successful) else math.nan
        posterior_candidates = (
            episode_scores.groupby("force_N", as_index=False)
            .agg(p_success=("p_success", "mean"), expected_utility=("expected_utility", "mean"))
            .sort_values("force_N", kind="mergesort")
        )
        posterior_max = float(posterior_candidates.expected_utility.max())
        posterior = posterior_candidates[
            np.isclose(posterior_candidates.expected_utility, posterior_max, rtol=0.0, atol=1e-12)
        ].iloc[0]
        unique_forces = np.unique(np.asarray(member_forces, float))
        first = branches.iloc[0]
        context_member_rows = episode_scores.drop_duplicates("member").sort_values("member")
        mu_values = context_member_rows.member_mu.to_numpy(float)
        decision_rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "selector": SELECTOR,
                "split": "TRAIN_ROOT_HELDOUT_OOF_DEV_FOLDS",
                "evidence_role": EVIDENCE_ROLE,
                "context_id": context_id,
                "repeat": int(repeat),
                "root_id": str(first.root_id),
                "task": int(first.task),
                "friction_band": str(first.friction_band),
                "candidate_count": int(len(branches)),
                "candidate_forces_N_json": json_text([float(x) for x in branches.force_N]),
                "member_mu_0": float(mu_values[0]),
                "member_mu_1": float(mu_values[1]),
                "member_mu_2": float(mu_values[2]),
                "friction_member_std": float(np.std(mu_values, ddof=0)),
                "member_force_0_N": member_forces[0],
                "member_force_1_N": member_forces[1],
                "member_force_2_N": member_forces[2],
                "member_selected_utility_0": member_utilities[0],
                "member_selected_utility_1": member_utilities[1],
                "member_selected_utility_2": member_utilities[2],
                "member_selected_probability_0": member_probabilities[0],
                "member_selected_probability_1": member_probabilities[1],
                "member_selected_probability_2": member_probabilities[2],
                "unique_member_decisions": int(len(unique_forces)),
                "exact_utility_decision_consensus": int(len(unique_forces) == 1),
                "utility_decision_disagreement": int(len(unique_forces) > 1),
                "decision_entropy_bits": entropy_bits(member_forces),
                "force_spread_N": float(max(member_forces) - min(member_forces)),
                "posterior_expected_utility_force_N": float(posterior.force_N),
                "posterior_expected_utility": float(posterior.expected_utility),
                "posterior_predicted_success": float(posterior.p_success),
                "empirical_frontier_N": None if not math.isfinite(frontier) else frontier,
                "frontier_available": int(math.isfinite(frontier)),
                "any_member_below_empirical_frontier": int(
                    math.isfinite(frontier) and min(member_forces) < frontier - 1e-9
                ),
                "posterior_below_empirical_frontier": int(
                    math.isfinite(frontier) and float(posterior.force_N) < frontier - 1e-9
                ),
            }
        )
    decisions = pd.DataFrame(decision_rows)
    if len(decisions) != 144 or decisions.duplicated(["context_id", "repeat"]).any():
        raise RuntimeError("invalid disagreement output grain")
    if int(scores.selected_by_member.sum()) != 144 * len(MEMBERS):
        raise RuntimeError("each member must select exactly one candidate per paired episode")
    return scores, decisions


def summarize(decisions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("POOLED", decisions)]
    scopes.extend((f"task{task}", decisions[decisions.task == task]) for task in sorted(FMAX))
    scopes.extend(
        (f"task{task}_{band}", decisions[(decisions.task == task) & (decisions.friction_band == band)])
        for task in sorted(FMAX)
        for band in sorted(decisions.friction_band.unique())
    )
    for scope, group in scopes:
        disagreement = group[group.utility_decision_disagreement == 1]
        consensus = group[group.utility_decision_disagreement == 0]
        rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "selector": SELECTOR,
                "split": "TRAIN_ROOT_HELDOUT_OOF_DEV_FOLDS",
                "evidence_role": EVIDENCE_ROLE,
                "scope": scope,
                "n_episodes": int(len(group)),
                "n_contexts": int(group.context_id.nunique()),
                "n_roots": int(group.root_id.nunique()),
                "disagreement_count": int(group.utility_decision_disagreement.sum()),
                "disagreement_rate": float(group.utility_decision_disagreement.mean()),
                "consensus_count": int(group.exact_utility_decision_consensus.sum()),
                "mean_force_spread_N": float(group.force_spread_N.mean()),
                "median_force_spread_N": float(group.force_spread_N.median()),
                "max_force_spread_N": float(group.force_spread_N.max()),
                "mean_decision_entropy_bits": float(group.decision_entropy_bits.mean()),
                "any_member_below_frontier_rate": float(group.any_member_below_empirical_frontier.mean()),
                "posterior_below_frontier_rate": float(group.posterior_below_empirical_frontier.mean()),
                "P_any_member_below_frontier_given_disagreement": (
                    float(disagreement.any_member_below_empirical_frontier.mean()) if len(disagreement) else None
                ),
                "P_any_member_below_frontier_given_consensus": (
                    float(consensus.any_member_below_empirical_frontier.mean()) if len(consensus) else None
                ),
            }
        )
    return pd.DataFrame(rows)


def build_paired_manifest_rows(decisions: pd.DataFrame, pool_hash: str) -> pd.DataFrame:
    arms = [
        ("ONE_QUERY", "never", 1, True),
        ("ALWAYS_TWO_QUERIES", "always", 2, False),
        ("RAW_UNCERTAINTY_THRESHOLD", "V2 threshold not frozen", None, False),
        ("UTILITY_DECISION_CONSENSUS", "member Utility decisions disagree", None, False),
        ("ORACLE_REQUERY_DIAGNOSTIC", "diagnostic label unavailable before paired outcomes", None, False),
    ]
    rows = []
    for row in decisions.itertuples(index=False):
        for arm, trigger_rule, fixed_query_count, cpu_row_ready in arms:
            trigger: bool | None
            if arm == "ONE_QUERY":
                trigger = False
            elif arm == "ALWAYS_TWO_QUERIES":
                trigger = True
            elif arm == "UTILITY_DECISION_CONSENSUS":
                trigger = bool(row.utility_decision_disagreement)
            else:
                trigger = None
            planned_query_count = fixed_query_count
            if arm == "UTILITY_DECISION_CONSENSUS":
                planned_query_count = 2 if trigger else 1
            rows.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "selector": SELECTOR,
                    "split": "TRAIN_ROOT_HELDOUT_OOF_DEV_FOLDS",
                    "evidence_role": "PAIRED_EVALUATION_PLAN_NOT_OUTCOME_EVIDENCE",
                    "pair_id": f"{row.context_id}__repeat{row.repeat}",
                    "context_id": row.context_id,
                    "repeat": int(row.repeat),
                    "root_id": row.root_id,
                    "task": int(row.task),
                    "friction_band": row.friction_band,
                    "arm": arm,
                    "trigger_rule": trigger_rule,
                    "trigger_value": trigger,
                    "planned_query_count": planned_query_count,
                    "candidate_count": int(row.candidate_count),
                    "candidate_forces_N_json": row.candidate_forces_N_json,
                    "candidate_pool_sha256": pool_hash,
                    "same_initial_state_required": True,
                    "same_first_query_transition_required": True,
                    "first_query_archive_available": True,
                    "qualified_second_query_available": False,
                    "posterior_after_second_query_available": False,
                    "cpu_row_definition_ready": bool(cpu_row_ready or arm == "UTILITY_DECISION_CONSENSUS"),
                    "paired_outcome_ready": False,
                    "status": "BLOCKED_MISSING_QUALIFIED_SECOND_QUERY",
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != 144 * 5 or result.duplicated(["pair_id", "arm"]).any():
        raise RuntimeError("invalid paired manifest grain")
    return result


def write_report(
    out: Path,
    summary: pd.DataFrame,
    pool_hash: str,
    reproduction_error: float,
    readiness: dict[str, Any],
    utility_hash: str,
) -> None:
    pooled_summary = summary[summary.scope == "POOLED"].iloc[0]
    task_rows = summary[summary.scope.isin([f"task{x}" for x in sorted(FMAX)])]
    table = [
        "| scope | episodes | disagreement | rate | mean spread N | max spread N | posterior below frontier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pd.concat([summary[summary.scope == "POOLED"], task_rows]).itertuples(index=False):
        table.append(
            f"| {row.scope} | {row.n_episodes} | {row.disagreement_count} | {row.disagreement_rate:.2%} | "
            f"{row.mean_force_spread_N:.4f} | {row.max_force_spread_N:.4f} | {row.posterior_below_frontier_rate:.2%} |"
        )
    report = f"""# E6 Expected-Utility member disagreement and second-query readiness

## Assessment

**CPU recompute complete; paired re-query evaluation blocked.** The final V2 Expected-Utility member decisions are now reproducible on the frozen 720-branch, 144-episode root-held-out OOF archive. The five-arm comparison is not ready because no genuine qualified second-query continuation or post-second-query posterior exists.

Legacy hard-rho outputs are explicitly excluded. No `FINAL_RHO.json`, rho-selection table, minimum-passing decision, or rho-induced disagreement row was loaded as evidence or used in a calculation.

## Expected-Utility disagreement

Every member selected `argmax_F U(F|z_m,x)` from the same five archived candidates for the same `(context_id, repeat)` pair. Utility is `p_success*(1-F/Fmax) + (1-p_success)*(-1)` and ties choose the lower force.

{chr(10).join(table)}

Pooled disagreement is **{int(pooled_summary.disagreement_count)}/{int(pooled_summary.n_episodes)} ({pooled_summary.disagreement_rate:.2%})**. Mean member-force spread is **{pooled_summary.mean_force_spread_N:.4f} N**, with maximum **{pooled_summary.max_force_spread_N:.4f} N**. These are CPU/DEV diagnostics from grouped-root OOF folds, not a final paired controller outcome claim.

The empirical-frontier columns are retrospective diagnostics only. They never enter candidate selection or the re-query trigger.

## Source and calculation checks

- Archive grain: 720 unique branches = 144 paired episodes × five exact context-specific candidates; no duplicate branch or `(context, repeat, force)` key.
- Physical belief: three finite OOF member estimates for all 72 contexts.
- Direct: nine frozen fold/seed checkpoints were run on CPU; GT-physics reproduction maximum absolute error was `{reproduction_error:.3g}`.
- Utility file SHA-256: `{utility_hash}`; candidate-pool SHA-256: `{pool_hash}`.
- Split isolation: source audits report no sealed TEST read; output split is `TRAIN_ROOT_HELDOUT_OOF_DEV_FOLDS`.
- Per-candidate `p_success`, rewards, Expected Utility, and member selections are retained in `E6_EXPECTED_UTILITY_MEMBER_SCORES.csv`.

## Second-query readiness

The prerequisite remains unavailable:

- The P7-B scientific run attempted 100 contexts but qualified 44, all TRAIN; qualified DEV = 0 and qualified TEST = 0. Its frozen gate classified the comparison as `SCIENTIFIC_COMPARISON_INVALID_BY_FROZEN_GATE`.
- The later physical-only protocol explicitly records `SECOND_QUERY_NOT_EVALUATED_NO_ARCHIVED_REPEAT`.
- Duplicate first-query telemetry is forbidden as a substitute for a second query.
- Therefore uncertainty reduction, posterior updating after query two, and the matched One/Always-Two/Raw-Uncertainty/Utility-Consensus/Oracle comparison are not estimable offline.

`E6_PAIRED_EVALUATION_MANIFEST.json` freezes the five arms, exact pair keys, candidate-pool identity, and fail-closed gate. `E6_PAIRED_EVALUATION_ROWS.csv` contains 720 planned pair-arm rows. It authorizes no GPU work and no rollout launch.

## Validation verdict

- Member disagreement recompute: **READY TO SHARE AS CPU/DEV DIAGNOSTIC**.
- Final five-arm E6 result: **NEEDS PREREQUISITE; NOT EVALUATED**.
- Second-query launch readiness: **BLOCKED** until a genuine continuation passes the frozen action/state/query qualification contract and produces a post-query posterior.
- Recovery action: **remain idle**; do not launch GPU or rollout work from this bundle.

## Required caveat

This bundle closes the CPU rescoring task only. It does not convert E6 into a complete-negative scientific result: the final paired hypothesis is untested because the required second-query evidence is absent.
"""
    (out / "E6_EXPECTED_UTILITY_CPU_REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    data, member_mu, _, source = load_and_validate_sources()
    probabilities, reproduction_error = infer_member_probabilities(data, member_mu)
    pool_hash = candidate_pool_hash(data)
    expected_pool_hash = str(source["config"].get("archive_candidate_pool_sha256"))
    if pool_hash != expected_pool_hash:
        raise RuntimeError(f"candidate-pool hash mismatch: {pool_hash} != {expected_pool_hash}")

    scores, decisions = score_and_select(data, member_mu, probabilities)
    summary = summarize(decisions)
    paired_rows = build_paired_manifest_rows(decisions, pool_hash)
    scores.to_csv(out / "E6_EXPECTED_UTILITY_MEMBER_SCORES.csv", index=False)
    decisions.to_csv(out / "E6_EXPECTED_UTILITY_MEMBER_DECISIONS.csv", index=False)
    summary.to_csv(out / "E6_EXPECTED_UTILITY_MEMBER_SUMMARY.csv", index=False)
    paired_rows.to_csv(out / "E6_PAIRED_EVALUATION_ROWS.csv", index=False)

    p7b_verdict = json.loads(P7B_VERDICT.read_text())
    p7b_completion = json.loads(P7B_COMPLETION.read_text())
    physical_protocol = json.loads(PHYSICAL_ONLY_PROTOCOL.read_text())
    readiness = {
        "status": "BLOCKED_MISSING_QUALIFIED_SECOND_QUERY",
        "checks": {
            "expected_utility_member_disagreement_recomputed": True,
            "identical_candidate_pool_by_pair": True,
            "first_query_archive_available": True,
            "qualified_second_query_continuation_available": False,
            "post_second_query_posterior_available": False,
            "five_arm_paired_outcomes_available": False,
            "sealed_TEST_accessed": False,
            "legacy_hard_rho_used_as_final_evidence": False,
            "gpu_started": False,
            "rollout_started": False,
        },
        "p7b_gate": {
            "classification": p7b_verdict["STATUS"],
            "contexts": int(p7b_completion["contexts"]),
            "qualified_contexts": int(p7b_completion["qualified_contexts"]),
            "dev_qualified_contexts": int(p7b_completion["split_qualified_contexts"]["DEV"]),
            "test_qualified_contexts": int(p7b_completion["split_qualified_contexts"]["TEST"]),
        },
        "later_offline_audit": {
            "second_query_status": physical_protocol["second_query_status"],
        },
        "unlock_condition": (
            "genuine second-query continuation with frozen action/state semantics, query qualification, "
            "measurable uncertainty update, and a post-second-query posterior; duplicate query-one data forbidden"
        ),
    }

    pair_manifest = {
        "manifest_name": "E6_V2_EXPECTED_UTILITY_FIVE_ARM_PAIRED_EVALUATION",
        "manifest_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "selector": SELECTOR,
        "evidence_role": "PAIRED_EVALUATION_PLAN_NOT_OUTCOME_EVIDENCE",
        "status": readiness["status"],
        "execution_authorized": False,
        "gpu_authorized": False,
        "rollout_authorized": False,
        "population": {
            "split": "TRAIN_ROOT_HELDOUT_OOF_DEV_FOLDS",
            "pair_key": ["context_id", "repeat"],
            "pairs": 144,
            "contexts": 72,
            "roots": 24,
            "tasks": sorted(FMAX),
            "candidate_count_per_pair": 5,
            "candidate_pool_sha256": pool_hash,
        },
        "utility": {
            "equation": source["config"]["equation"],
            "task_Fmax_N": source["config"]["task_Fmax_N"],
            "tie_break": "lower_force",
            "utility_config_file_sha256": sha256_file(UTILITY_CONFIG),
        },
        "arms": [
            {"name": "ONE_QUERY", "query_rule": "exactly one qualified P4-B query"},
            {"name": "ALWAYS_TWO_QUERIES", "query_rule": "always execute the genuine frozen second query"},
            {
                "name": "RAW_UNCERTAINTY_THRESHOLD",
                "query_rule": "second query only when the separately frozen V2 raw-uncertainty threshold triggers",
                "threshold_status": "NOT_FROZEN_IN_AVAILABLE_FINAL_V2_ARTIFACTS",
            },
            {
                "name": "UTILITY_DECISION_CONSENSUS",
                "query_rule": "second query iff member-wise Expected-Utility argmax decisions disagree",
                "trigger_rows_available": True,
            },
            {
                "name": "ORACLE_REQUERY_DIAGNOSTIC",
                "query_rule": "retrospective diagnostic only; never deployable",
            },
        ],
        "pair_invariants": [
            "same root, context, repeat, initial state, candidate set, Direct stack, Utility, and first-query transition",
            "only query policy and resulting posterior/force may differ",
            "empirical outcome/frontier is never a runtime selector input",
        ],
        "required_result_metrics": [
            "full_task_SR",
            "mean_query_count",
            "second_query_rate",
            "query precision",
            "harmful-decision enrichment",
            "mean and max measured force",
            "under-force and excess-force",
            "realized utility",
            "latency",
        ],
        "readiness": readiness,
        "rows_file": "E6_PAIRED_EVALUATION_ROWS.csv",
    }
    write_json(out / "E6_PAIRED_EVALUATION_MANIFEST.json", pair_manifest)

    source_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "selector": SELECTOR,
        "legacy_hard_rho_final_evidence": "EXCLUDED",
        "source_hashes": source["source_hashes"],
        "derived_files": {},
    }
    write_report(out, summary, pool_hash, reproduction_error, readiness, sha256_file(UTILITY_CONFIG))

    derived_names = [
        "E6_EXPECTED_UTILITY_CPU_REPORT.md",
        "E6_EXPECTED_UTILITY_MEMBER_SCORES.csv",
        "E6_EXPECTED_UTILITY_MEMBER_DECISIONS.csv",
        "E6_EXPECTED_UTILITY_MEMBER_SUMMARY.csv",
        "E6_PAIRED_EVALUATION_MANIFEST.json",
        "E6_PAIRED_EVALUATION_ROWS.csv",
    ]
    source_manifest["derived_files"] = {name: sha256_file(out / name) for name in derived_names}
    write_json(out / "E6_CPU_SOURCE_AND_OUTPUT_MANIFEST.json", source_manifest)

    pooled_summary = summary[summary.scope == "POOLED"].iloc[0]
    status = {
        "status": "CPU_RECOMPUTE_COMPLETE_PAIRED_EVALUATION_BLOCKED",
        "protocol_version": PROTOCOL_VERSION,
        "selector": SELECTOR,
        "evidence_role": EVIDENCE_ROLE,
        "cpu_work_complete": True,
        "paired_evaluation_ready": False,
        "blocked_prerequisite": "QUALIFIED_GENUINE_SECOND_QUERY_CONTINUATION",
        "remain_idle": True,
        "gpu_started": False,
        "rollout_started": False,
        "sealed_TEST_accessed": False,
        "legacy_hard_rho_used_as_final_evidence": False,
        "population": {"branches": 720, "episodes": 144, "contexts": 72, "roots": 24},
        "expected_utility_disagreement": {
            "count": int(pooled_summary.disagreement_count),
            "rate": float(pooled_summary.disagreement_rate),
            "mean_force_spread_N": float(pooled_summary.mean_force_spread_N),
            "max_force_spread_N": float(pooled_summary.max_force_spread_N),
        },
        "candidate_pool_sha256": pool_hash,
        "utility_config_file_sha256": sha256_file(UTILITY_CONFIG),
        "direct_gt_reproduction_max_abs_error": reproduction_error,
        "source_output_manifest_sha256": sha256_file(out / "E6_CPU_SOURCE_AND_OUTPUT_MANIFEST.json"),
    }
    write_json(out / "RECOVERY_STATUS_E6_CPU_20260902.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
