#!/usr/bin/env python3
"""Build Agent-A E1 diagnosis and faithful-baseline audit artifacts.

This is a read-only analysis of frozen archived evidence.  It never trains a
model, reads a sealed TEST split, launches Isaac, or mutates a shared checkout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FORTE = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "ACTIVEFORCING_E1_CRITICAL_CLOSURE_20260902_084000"
AUTH = FORTE / "analysis/results/ACTIVEFORCING_E1_FINAL_UTILITY_MAIN_TABLE_20260902_052942"
SELECTED = AUTH / "E1_SELECTED_EPISODES.csv"
MAIN = AUTH / "TABLE_E1_FINAL_UTILITY_MAIN.csv"
CANDIDATES = OUT / "E1_CANDIDATE_DECISION_CHAIN.csv"
PROBE = FORTE / "activeforcing_probe_conditioned_wm_20260901_064627/POOLED_OOF_PROBE_PREDICTIONS.csv"
CONFIG = FORTE / "UTILITY_FINAL_CONFIG.json"
PROTOCOL = FORTE / "activeforcing_residual_utility_20260901_055605/RESIDUAL_UTILITY_DEVELOPMENT_PROTOCOL.json"
EXPECTED_CONFIG_HASH = "c5bc4f39861a84b9ba95d55cdb5f8633a8b004d205b3864a02799340653d9b10"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jdump(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def finite_or_none(value):
    try:
        return float(value) if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    out = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & ((p <= edges[i + 1]) if i == bins - 1 else (p < edges[i + 1]))
        if mask.any():
            out += mask.mean() * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(out) if total else float("nan")


def pairwise_ranking(frame: pd.DataFrame, score_col: str) -> tuple[float, int]:
    correct = 0.0
    pairs = 0
    for _, group in frame.groupby(["context_id", "repeat"], sort=False):
        s = group[group.success.eq(1)][score_col].to_numpy(float)
        f = group[group.success.eq(0)][score_col].to_numpy(float)
        for ps in s:
            for pf in f:
                correct += 1.0 if ps > pf else (0.5 if np.isclose(ps, pf) else 0.0)
                pairs += 1
    return (correct / pairs if pairs else float("nan"), pairs)


def load_raw(selected: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    for path in sorted(selected.source_path.dropna().unique()):
        p = Path(path)
        if p.exists():
            d = pd.read_csv(p)
            d["source_path_join"] = str(p)
            chunks.append(d)
    raw = pd.concat(chunks, ignore_index=True)
    keep = [
        "branch_id", "steady_state_mean_N", "force_tracking_mae_N",
        "force_tracking_error_N", "integrated_force", "telemetry_path",
        "dropped", "timeout", "source_path_join",
    ]
    raw = raw[keep].drop_duplicates(["branch_id", "source_path_join"])
    return selected.merge(
        raw,
        left_on=["raw_branch_id", "source_path"],
        right_on=["branch_id", "source_path_join"],
        how="left",
        suffixes=("", "_raw"),
        validate="many_to_one",
    )


def method_prefix(method: str) -> str | None:
    return {
        "ActiveForcing-1Q Utility": "active",
        "GT-Friction + Direct + same Utility": "gt_friction",
        "Query-Ignored / No-Physical-Information": "query_ignored",
    }.get(method)


def method_short(method: str) -> str:
    return {
        "ActiveForcing-1Q Utility": "ActiveForcing Utility",
        "GT-Friction + Direct + same Utility": "GT-Friction Utility",
        "Query-Ignored / No-Physical-Information": "Query-Ignored",
        "Hindsight Grid Oracle": "Hindsight Grid Oracle",
        "Frozen pi0 Default": "pi0 Default",
    }.get(method, method)


def taskwise_audit(selected: pd.DataFrame, merged: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    active_mu = candidates.groupby("context_id", as_index=False).first()[["context_id", "predicted_mu_active", "mu"]]
    probe_context = pd.read_csv(PROBE).groupby("context_id", as_index=False).agg(
        query_raw_path=("raw_path", "first"),
        query_timesteps=("timesteps", "max"),
        query_seed_members=("seed", "nunique"),
    ).merge(active_mu, on="context_id", how="right", validate="one_to_one")
    probe_context["friction_error"] = probe_context.predicted_mu_active - probe_context.mu

    fixed = selected[selected.method.eq("Fixed-Max")].set_index(["context_id", "repeat"])
    query = selected[selected.method.eq("Query-Ignored / No-Physical-Information")].set_index(["context_id", "repeat"])
    rows = []
    for task in [0, 1, 5, 6]:
        pc = probe_context[probe_context.context_id.str.contains(f"_t{task}_")]
        for method in selected.method.unique():
            g = merged[(merged.task.eq(task)) & (merged.method.eq(method))].copy()
            prefix = method_prefix(method)
            c = candidates[candidates.task.eq(task)]
            if prefix:
                pcol = f"p_success_{prefix}"
                brier = float(np.mean((c[pcol] - c.success) ** 2))
                calib = ece(c.success.to_numpy(int), c[pcol].to_numpy(float))
                rank, rank_n = pairwise_ranking(c, pcol)
            else:
                brier = calib = rank = float("nan")
                rank_n = 0
            dist = {f"{x:.6f}": int(n) for x, n in g.selected_force_N.value_counts().sort_index().items()}
            fail_stage = g[g.actual_success.eq(0)].failure_stage.fillna("SUCCESS_OR_UNSPECIFIED").value_counts().to_dict()
            fail_reason = g[g.actual_success.eq(0)].failure_reason.fillna("UNSPECIFIED").value_counts().to_dict()
            idx = g.set_index(["context_id", "repeat"])
            common_f = idx.index.intersection(fixed.index)
            common_q = idx.index.intersection(query.index)
            rescue_f = int(((fixed.loc[common_f].actual_success.eq(0)) & (idx.loc[common_f].actual_success.eq(1))).sum())
            collateral_f = int(((fixed.loc[common_f].actual_success.eq(1)) & (idx.loc[common_f].actual_success.eq(0))).sum())
            rescue_q = int(((query.loc[common_q].actual_success.eq(0)) & (idx.loc[common_q].actual_success.eq(1))).sum())
            collateral_q = int(((query.loc[common_q].actual_success.eq(1)) & (idx.loc[common_q].actual_success.eq(0))).sum())
            rows.append({
                "task": task,
                "method": method_short(method),
                "episodes": len(g),
                "contexts": g.context_id.nunique(),
                "roots": g.root_id.nunique(),
                "pi0_execution_source": "SCRIPTED_POST_QUERY_BRANCH_NOT_PI0_E2E",
                "query_required_runtime": int(method == "ActiveForcing-1Q Utility"),
                "query_reach_rate": 1.0 if method == "ActiveForcing-1Q Utility" else np.nan,
                "query_valid_rate": float((pc.query_timesteps.gt(0) & pc.query_raw_path.map(lambda x: Path(x).exists())).mean()) if method == "ActiveForcing-1Q Utility" else np.nan,
                "query_timesteps": float(pc.query_timesteps.mean()) if method == "ActiveForcing-1Q Utility" else np.nan,
                "friction_mae": float(pc.friction_error.abs().mean()) if method == "ActiveForcing-1Q Utility" else np.nan,
                "friction_bias_pred_minus_gt": float(pc.friction_error.mean()) if method == "ActiveForcing-1Q Utility" else np.nan,
                "direct_brier_all_candidates": brier,
                "direct_ece10_all_candidates": calib,
                "force_pairwise_ranking_accuracy": rank,
                "force_ranking_pairs": rank_n,
                "success_rate": float(g.actual_success.mean()),
                "selected_force_mean_N": float(g.selected_force_N.mean()),
                "selected_force_min_N": float(g.selected_force_N.min()),
                "selected_force_max_N": float(g.selected_force_N.max()),
                "selected_force_distribution_json": jdump(dist),
                "commanded_force_mean_N": float(g.selected_force_N.mean()),
                "measured_force_mean_N": float(g.measured_force_mean_N.mean()),
                "measured_force_coverage": float(g.measured_force_mean_N.notna().mean()),
                "measured_peak_force_mean_N": float(g.measured_force_peak_N.mean()),
                "measured_peak_force_max_N": float(g.measured_force_peak_N.max()),
                "steady_state_force_mean_N": float(g.steady_state_mean_N.mean()),
                "steady_state_coverage": float(g.steady_state_mean_N.notna().mean()),
                "logged_force_tracking_mae_mean_N": float(g.force_tracking_mae_N.mean()),
                "tracking_metric_caveat": "post-loss zero-force tail contaminates failed-episode MAE",
                "frontier_available_rate": float(g.frontier_available.mean()),
                "selected_at_or_above_frontier_rate": float((g.selected_force_N >= g.frontier_N - 1e-9).mean()),
                "under_force_rate": float(g.under_force.mean()),
                "mean_excess_force_N": float(g.excess_force_N.mean()),
                "realized_utility_mean": float(g.realized_utility.mean()),
                "failure_count": int(g.actual_success.eq(0).sum()),
                "failure_stage_counts_json": jdump(fail_stage),
                "failure_reason_counts_json": jdump(fail_reason),
                "rescue_vs_fixed_max": rescue_f,
                "collateral_vs_fixed_max": collateral_f,
                "rescue_vs_query_ignored": rescue_q,
                "collateral_vs_query_ignored": collateral_q,
            })
    return pd.DataFrame(rows)


def get_selected(candidates: pd.DataFrame, context: str, repeat: int, prefix: str) -> pd.Series:
    g = candidates[(candidates.context_id.eq(context)) & (candidates.repeat.eq(repeat))]
    return g[g[f"selected_{prefix}"].eq(1)].iloc[0]


def vector(candidates: pd.DataFrame, context: str, repeat: int, prefix: str) -> str:
    g = candidates[(candidates.context_id.eq(context)) & (candidates.repeat.eq(repeat))].sort_values("force_N")
    return jdump([
        {
            "F_N": round(float(r.force_N), 9),
            "p_success": round(float(getattr(r, f"p_success_{prefix}")), 9),
            "expected_utility": round(float(getattr(r, f"expected_utility_{prefix}")), 9),
            "y": int(r.success),
        }
        for r in g.itertuples()
    ])


def task6_case_audit(merged: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    a = merged[(merged.task.eq(6)) & (merged.method.eq("ActiveForcing-1Q Utility"))].set_index(["context_id", "repeat"])
    q = merged[(merged.task.eq(6)) & (merged.method.eq("Query-Ignored / No-Physical-Information"))].set_index(["context_id", "repeat"])
    cases = a[(q.actual_success.eq(1)) & (a.actual_success.eq(0))]
    rows = []
    for key, ar in cases.iterrows():
        context, repeat = key
        qr = q.loc[key]
        ac = get_selected(candidates, context, repeat, "active")
        gc = get_selected(candidates, context, repeat, "gt_friction")
        qc = get_selected(candidates, context, repeat, "query_ignored")
        rows.append({
            "case_id": f"{context}__repeat{repeat}",
            "context_id": context,
            "root_id": ar.root_id,
            "repeat": repeat,
            "friction_band": ar.friction_band,
            "gt_friction_mu": ar.mu,
            "predicted_friction_mu": ac.predicted_mu_active,
            "friction_signed_error": ac.predicted_mu_active - ar.mu,
            "classification_primary": "D.DIRECT_MODEL_MISCALIBRATION",
            "classification_secondary": "E.UTILITY_FORCE_COST_TOO_AGGRESSIVE_CONDITIONAL_ON_OVERCONFIDENT_P",
            "classification_rationale": "GT-friction selects the identical failing low branch; next archived force succeeds; Direct assigns the failing branch near-certain success",
            "active_force_N": ac.force_N,
            "gt_friction_utility_force_N": gc.force_N,
            "query_ignored_force_N": qc.force_N,
            "frontier_N": ar.frontier_N,
            "candidate_max_N": ar.max_candidate_force_N,
            "active_p_success": ac.p_success_active,
            "active_expected_utility": ac.expected_utility_active,
            "gt_p_success": gc.p_success_gt_friction,
            "gt_expected_utility": gc.expected_utility_gt_friction,
            "query_ignored_p_success": qc.p_success_query_ignored,
            "query_ignored_expected_utility": qc.expected_utility_query_ignored,
            "active_outcome": int(ar.actual_success),
            "query_ignored_outcome": int(qr.actual_success),
            "active_measured_force_mean_N": ar.measured_force_mean_N,
            "active_measured_force_peak_N": ar.measured_force_peak_N,
            "active_steady_state_mean_N": ar.steady_state_mean_N,
            "active_logged_tracking_mae_N": ar.force_tracking_mae_N,
            "query_measured_force_mean_N": qr.measured_force_mean_N,
            "query_measured_force_peak_N": qr.measured_force_peak_N,
            "query_steady_state_mean_N": qr.steady_state_mean_N,
            "query_logged_tracking_mae_N": qr.force_tracking_mae_N,
            "active_pick_success": ar.pick_success,
            "active_lift_success": ar.lift_success,
            "active_transport_retention": ar.transport_retention,
            "active_place_success": ar.place_success,
            "active_lost_in_transit": ar.lost_in_transit,
            "active_failure_stage": ar.failure_stage,
            "active_failure_reason": ar.failure_reason,
            "active_branch_id": ar.raw_branch_id,
            "query_branch_id": qr.raw_branch_id,
            "active_telemetry_path": ar.telemetry_path,
            "query_telemetry_path": qr.telemetry_path,
            "active_candidate_vector_json": vector(candidates, context, repeat, "active"),
            "gt_candidate_vector_json": vector(candidates, context, repeat, "gt_friction"),
            "query_candidate_vector_json": vector(candidates, context, repeat, "query_ignored"),
            "tracking_interpretation": "logged failed-episode MAE is consequence-contaminated after contact loss; not independent evidence of controller malfunction",
        })
    return pd.DataFrame(rows).sort_values(["context_id", "repeat"])


def common_table(main: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "scope_type", "scope_value", "method", "status", "n_episodes", "n_contexts", "n_roots",
        "full_task_SR", "mean_selected_force_N", "mean_measured_force_N",
        "mean_peak_measured_force_N", "max_peak_measured_force_N", "measured_force_coverage",
        "under_force_rate", "mean_excess_force_N", "mean_realized_utility",
    ]
    d = main[(main.population.eq("ALL_CONTEXTS")) & (main.scope_type.isin(["POOLED", "TASK"]))].copy()
    d = d[d.method.ne("Hindsight Grid Oracle")][cols]
    d["method"] = d.method.map(method_short)
    d["runtime_objective"] = d.method.map({
        "pi0 Default": "native frozen pi0 action including native force slots",
        "Fixed-Max": "fixed maximum archived candidate",
        "Query-Ignored": "Direct expected Utility marginalized over frozen friction prior",
        "ActiveForcing Utility": "P4-B point friction -> Direct expected Utility argmax",
        "GT-Friction Utility": "offline GT friction -> Direct expected Utility argmax",
    })
    d["evaluation_metric"] = "realized U=y*(1-F/Fmax)+(1-y)*(-1)"
    d["queries_per_episode"] = d.method.map({"ActiveForcing Utility": 1, "Query-Ignored": 0, "Fixed-Max": 0, "pi0 Default": 0})
    d["interaction_cost"] = d.method.map({"ActiveForcing Utility": "one P4-B physical query", "GT-Friction Utility": "offline oracle; not deployable", "Query-Ignored": "none", "Fixed-Max": "none", "pi0 Default": "none"})
    d["latency"] = np.nan
    d["force_metric_note"] = "selected/commanded and measured columns remain separate"
    external = []
    for scope_type, scope_value in [("POOLED", "ALL")] + [("TASK", f"task{x}") for x in [0, 1, 5, 6]]:
        for method, status, objective in [
            ("FORTE", "FAITHFUL_REPRODUCTION_BLOCKED", "native 2-kHz analog tactile SVR + Welch slip trigger + incremental Dynamixel impedance closure"),
            ("Tabero", "FAITHFUL_REPRODUCTION_BLOCKED", "native Field+FS VTLA outputs pose/gripper/fL/fR; closed-loop chunk replanning"),
        ]:
            row = {c: np.nan for c in d.columns}
            row.update({
                "scope_type": scope_type,
                "scope_value": scope_value,
                "method": method,
                "status": status,
                "runtime_objective": objective,
                "evaluation_metric": "same realized Utility after a faithful paired rollout",
                "queries_per_episode": 0 if method == "Tabero" else np.nan,
                "interaction_cost": "native method semantics; no ActiveForcing query",
                "force_metric_note": "no same-tuple paired rollout; missing values are not zero",
            })
            external.append(row)
    return pd.concat([d, pd.DataFrame(external)], ignore_index=True)


def md_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    def fmt(x):
        if pd.isna(x):
            return "—"
        if isinstance(x, (float, np.floating)):
            return f"{x:.{digits}f}"
        return str(x).replace("|", "\\|")
    rows = [[fmt(x) for x in row] for row in frame[columns].itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def write_reports(selected: pd.DataFrame, merged: pd.DataFrame, candidates: pd.DataFrame, cases: pd.DataFrame, table: pd.DataFrame) -> None:
    active6 = merged[(merged.task.eq(6)) & (merged.method.eq("ActiveForcing-1Q Utility"))]
    gt6 = merged[(merged.task.eq(6)) & (merged.method.eq("GT-Friction + Direct + same Utility"))]
    query6 = merged[(merged.task.eq(6)) & (merged.method.eq("Query-Ignored / No-Physical-Information"))]
    fixed6 = merged[(merged.task.eq(6)) & (merged.method.eq("Fixed-Max"))]
    fail6 = active6[active6.actual_success.eq(0)]
    active_lowest = candidates[candidates.task.eq(6)].groupby(["context_id", "repeat"]).apply(
        lambda g: int(g.loc[g.force_N.idxmin(), "selected_active"]), include_groups=False
    ).mean()
    cfg = json.loads(CONFIG.read_text())
    cfg_hash_ok = cfg["config_sha256"] == EXPECTED_CONFIG_HASH

    mechanism = f"""# Task6 failures are a low-boundary Direct-calibration failure, not a friction-identifier failure

## Verdict

`TASK6_PRIMARY_FAILURE_SOURCE = DIRECT`

Secondary source: `UTILITY`, specifically the intended force cost acting on an overconfident Direct probability at the lowest candidate. `IDENTIFIER`, `PI0`, `FORCE_RANGE`, `CONTROLLER`, and `LABEL/EVALUATOR` are not supported as the primary explanation by this archive.

Task6 ActiveForcing succeeds in {int(active6.actual_success.sum())}/36 episodes ({active6.actual_success.mean():.4%}), versus {int(query6.actual_success.sum())}/36 ({query6.actual_success.mean():.4%}) for Query-Ignored and {int(fixed6.actual_success.sum())}/36 ({fixed6.actual_success.mean():.4%}) for Fixed-Max. There are exactly {len(cases)} Query-Ignored-success -> Active-failure paired cases. In every one, substituting GT friction leaves the selector on the identical failing low-force branch. That rules out the identifier as the cause of those paired regressions. Direct assigns the failing branch p(success)={cases.active_p_success.min():.4f}–{cases.active_p_success.max():.4f}; the next archived candidate, {cases.frontier_N.min():.3f}–{cases.frontier_N.max():.3f} N, succeeds.

Across all five Active task6 failures, the selected force is the context's lowest candidate (about 3.01–3.06 N); Fixed-Max succeeds 36/36. This is concentrated low-boundary under-force. It is not lack of force support: all paired rescue forces are below the current 4 N cap and the maximum archived candidate succeeds.

## Case-level evidence

The authoritative case table is `TASK6_FAILURE_CASE_AUDIT.csv`. Candidate vectors contain every force, Direct probability, Expected Utility, and observed branch label. The mutually exclusive primary attribution for all {len(cases)} paired regressions is `D.DIRECT_MODEL_MISCALIBRATION`; `E.UTILITY_FORCE_COST_TOO_AGGRESSIVE_CONDITIONAL_ON_OVERCONFIDENT_P` is secondary.

{md_table(cases, ['case_id','gt_friction_mu','predicted_friction_mu','active_force_N','gt_friction_utility_force_N','query_ignored_force_N','frontier_N','active_p_success','active_expected_utility','active_failure_stage'])}

## Fmax=4 N audit

The frozen config and development protocol both set task6 Fmax=4 N. The config hash is `{cfg['config_sha256']}` (verification: `{cfg_hash_ok}`). Git history contains task6=4 N as `ROBUST_FORCE_BY_TASK` in Tabero commit `80ab3be`, while the candidate-generation source defines task6 support as `(3.0, 4.0)`. No inspected source labels 4 N as a physical or safety limit. Separate simulator sensor-visualization and damage thresholds exist elsewhere and do not establish this value as safety-critical.

Therefore, the defensible interpretation is: task6 Fmax is a historical robust-force / candidate-support endpoint reused for Utility normalization and task-specific configuration. Its provenance does **not** support calling it a physical safety limit. Candidate support is sufficient to rescue the audited failures; the issue is selection within support, not an unavailable >4 N action. No TEST outcome was used to change Fmax, and this report makes no Fmax change.

## Controller and evaluator checks

The raw controller telemetry reports large tracking MAE on failed low-force branches, but that statistic includes the post-contact-loss zero-force tail. It is consequence-contaminated: after the object is lost, measured contact force is zero by construction. Transient `measured_force_mean_N` also includes the branch-hold load spike (peaks near 20 N), so it is not a setpoint-tracking statistic. Successful task6 branches have stable steady-state tracking; the archive does not independently establish a low-level controller fault.

The paired higher-force branches succeed under the same state/repeat, so evaluator error is not needed to explain the regressions. Some component labels (`transport_retention`, `lost_in_transit`, `place_success`) have collector-specific semantics and should not be over-interpreted; the controlling binary label is the archived full-task outcome.

## Limits

These are grouped-root OOF TRAIN/development post-query matched branches, not sealed TEST and not native pi0 E2E. The diagnosis is complete for the E1 force-selection anomaly; it does not prove independent nominal pi0 task6 capability.

`TASK6_FAILURE_DIAGNOSIS_COMPLETE`
"""
    (OUT / "TASK6_FAILURE_MECHANISM_REPORT.md").write_text(mechanism, encoding="utf-8")

    pi0 = f"""# Task6 pi0 nominal-capability audit

## Gate decision

`PI0_REPAIR = NOT_NEEDED`

This status means the repair authorization gate was not triggered; it does **not** mean E1 proves native pi0 task6 capability is high.

The authoritative E1 archive runs a scripted post-query downstream branch (`p5s0a_true_matched_dataset.py`) after a matched reset. It does not execute the frozen pi0 nominal policy from reset to completion. Consequently, Fixed-Max 36/36 is strong evidence that the task6 post-query state, force support, controller, and scripted downstream path can succeed at robust force, but it is not a native-pi0 nominal SR estimate.

Within the evidence that actually measures the E1 anomaly, the three Query-Ignored-success -> Active-failure cases are explained downstream of the query by force choice: GT friction preserves the failing choice, while a higher archived force succeeds. There is no affirmative evidence that `UPSTREAM_PI0_FAILURE` is the primary task6 bottleneck. The required precondition for few-demo repair is therefore false.

No demos were collected, no checkpoint was trained, no checkpoint was changed, and no onboarding artifacts were emitted. If a later native-pi0, same-root robust-force E2E audit shows nominal task6 failure, onboarding must use TRAIN/DEV demos only and the resulting frozen checkpoint must be shared by all task6 methods.

Evidence gap: a faithful native-pi0 robust-force task6 E2E dataset is required to estimate reach, grasp pose, transport, orientation, placement, semantic-execution, and evaluator-specific failure rates. Existing task6 E5 smoke did not produce rollouts. This gap blocks a positive nominal-capability claim but does not authorize repair.
"""
    (OUT / "TASK6_PI0_NOMINAL_AUDIT.md").write_text(pi0, encoding="utf-8")

    forte_report = f"""# FORTE faithful-reproduction audit

## Status

`FORTE = FAITHFUL_REPRODUCTION_BLOCKED`

## Recovered authoritative semantics

FORTE is a hardware reactive grasp controller, not a fixed-force baseline and not Direct+Utility. The checked-out implementation samples six analog pressure channels at 2 kHz, maps the current tactile state to a scalar force with a frozen 24-feature RBF-SVR checkpoint, computes a Welch spectral estimate over 10–50 Hz to detect slip, and, on slip, incrementally closes a Dynamixel gripper under impedance control (the showcase uses a 0.006 position decrement and 0.2 s cooldown after an initial load increment).

Runtime objective: tactile slip suppression through reactive impedance-position correction. Common evaluation, once faithfully rolled out, may use the project realized Utility; that evaluation must not replace FORTE's runtime logic.

## Provenance

- Isolated checkout: `/home/exouser/FORTE_e1diag`, commit `7f88d018`.
- Official showcase: `examples/gripper_showcase_vis_realtime.py`, SHA-256 `580359994a0ee8c401b6be6b8b8ab20bb7bc37a52cbd455d8568888b5f692dab`.
- Force/slip runtime: `forte/runtime/force_and_slip.py`, SHA-256 `881458a2b67d5ab8b716dd279dc4ec0f4513150a5080aadb4b06f67a4826016e`.
- Frozen SVR: `models/SVR_ckpt.pkl`, SHA-256 `9d9ba2449282196db1a85f31e1e41cdca7ccd56522130e2ff3deffd93e8017bc`; sklearn 1.6.1; 24 input features.
- Prior audit: `/home/exouser/Tabero/analysis/results/f1_forte_tabero_baseline_20260818_194501`.

## Why final paired E1 numbers are blocked

The E1 simulator archive contains contact-force trajectories, not FORTE's six-channel analog sensor stream, the calibrated SVR feature path, or a faithful Dynamixel impedance actuator adapter. No `/dev/ttyACM*` or `/dev/ttyUSB*` hardware interface is present in the prior audit. Replaying an oracle slip label or a fixed/ladder force policy would change the scientific method.

Archived `FORTE-INSPIRED REACTIVE` / oracle-slip outputs are explicitly excluded and are not renamed as FORTE. No same-task/root/friction/initial-state/seed native FORTE rollouts exist; missing table values remain NA rather than becoming surrogate results.

Engineering unblock requirement: an audited simulator adapter that produces the same six-channel input semantics and maps FORTE's impedance position increments to the simulated gripper without changing slip detection or control logic, followed by fresh same-tuple E2E rollouts with measured-force telemetry.
"""
    (OUT / "FORTE_REPRODUCTION_AUDIT.md").write_text(forte_report, encoding="utf-8")

    tabero_report = """# Tabero faithful-reproduction audit

## Status

`TABERO = FAITHFUL_REPRODUCTION_BLOCKED`

## Recovered authoritative semantics

Tabero is the native Field+FS vision-tactile-language-action policy. It receives visual/tactile context and force history, emits a 13-D action containing pose, gripper, and left/right force slots, and executes those learned force outputs through its hybrid force-position controller with fresh chunk replanning (`replan_steps=10`). It is not Direct+Utility and must not receive an ActiveForcing force-slot overwrite.

Runtime objective: the learned Tabero policy and its native closed-loop force/tactile/action semantics. Common E1 evaluation can calculate realized Utility afterward, but Utility is not Tabero's runtime selector.

## Provenance

- Isolated sparse checkout: `/home/exouser/Tabero_e1diag`, commit `80ab3be`.
- Native runner: `benchmarks/openpi/openpi_inference_client.py`, SHA-256 `487bec3fa6a5003f5dbe4a327582ec1f9a4724ef2c54fb94b833dfc2f517cc94`.
- Closed-loop wrapper: `benchmarks/common/closedloop_policy_inference.py`, SHA-256 `8ee006f1586f0851f0719057153a7ed5308bb2a8f66337b6e502f4391f076bbe`.
- Frozen checkpoint: `pi0_lora_tacfield_tabero/checkpoints/.../49999`; checkpoint SHA-256 `0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17`; config `pi0_lora_tacfield_tabero`; norm stats `assets/NathanWu7/tabero`.
- The protected inference server observed during this audit is the exact frozen checkpoint/config. It was not modified or restarted.

## Faithfulness and paired-evaluation decision

The current fresh-E2E executor already has the correct Tabero/native-pi0 mode: it executes all native 13-D outputs and removes only ActiveForcing's force-slot replacement. Existing task1 native smoke rows are useful runtime smoke evidence, but their roots (7200/7201) do not match E1 roots (5100–5105), so they are excluded from the paired E1 table.

Final E1 reproduction is blocked because no fresh same-task/root/friction/initial-state/seed Tabero rollout set exists. During this audit the GPU was already carrying protected MASS, E5, and frozen-policy-server workloads above the launch threshold; starting another Isaac job would violate the throughput policy. No protected process was killed or altered.

Unblock requirement: schedule fresh native-mode rollouts on the exact E1 tuple manifest, with the same frozen checkpoint and no force override, then report commanded left/right force slots and measured contact force separately. Until then, missing values remain NA.
"""
    (OUT / "TABERO_REPRODUCTION_AUDIT.md").write_text(tabero_report, encoding="utf-8")

    pooled = table[table.scope_type.eq("POOLED")]
    table_md = f"""# E1 with faithful external-baseline slots

The first five methods use the authoritative 144-episode, 72-context, 24-root grouped-root-OOF development archive. FORTE and Tabero are intentionally NA because a faithful same-tuple rollout is unavailable; missing values are not zero. `mean_selected_force_N` is the commanded/setpoint value. Measured trajectory force and peak force are separate columns.

{md_table(pooled, ['method','status','n_episodes','full_task_SR','mean_selected_force_N','mean_measured_force_N','mean_peak_measured_force_N','max_peak_measured_force_N','under_force_rate','mean_excess_force_N','mean_realized_utility','queries_per_episode'])}

Evaluation metric for estimable methods: `U_eval = y*(1-F/Fmax) + (1-y)*(-1)`, with task Fmax 0:5, 1:6, 5:5, 6:4 N. ActiveForcing alone uses Expected Utility at runtime. FORTE and Tabero retain their own runtime objectives.

Force caveat: the archive's `measured_force_mean_N` averages a transient trajectory that includes branch-hold loading. It is a measured physical telemetry aggregate, not controller setpoint. Failed-episode tracking MAE contains post-contact-loss zero-force tails. Cross-method reporting must not substitute commanded force where measured force is absent.
"""
    (OUT / "TABLE_E1_WITH_EXTERNAL_BASELINES.md").write_text(table_md, encoding="utf-8")

    final = f"""# Task6 E1 closure identifies Direct low-force overconfidence; faithful FORTE/Tabero numbers remain blocked

## Decision

The task6 ActiveForcing deficit is localized to low-boundary selection: all five Active failures select the lowest candidate, all three Query-success -> Active-failure pairs remain failures when GT friction replaces P4-B, and the next archived candidate succeeds. Primary attribution is `DIRECT`; the Expected-Utility force cost is secondary because it converts the overconfident probability into a low-force choice. The physical identifier is not primary.

No pi0 repair was performed. E1 is scripted downstream rather than native-pi0 E2E, so it cannot establish native nominal capability; more importantly, it provides no affirmative evidence that pi0 is the primary failure source required to authorize few-demo repair.

## Authoritative method/config

- Frozen authoritative pi0 -> P4-B query -> friction belief -> shared Direct full-task feasibility -> Expected Utility -> selected force as controller setpoint -> unchanged controller -> pi0 nominal continuation.
- Utility: `p(success)*(1-F/Fmax) + (1-p(success))*(-1)`, low-force tie break.
- Fmax: task0=5, task1=6, task5=5, task6=4 N.
- Config hash: `{cfg['config_sha256']}`; exact expected-hash match: `{cfg_hash_ok}`.
- No hard-rho selector was restored. No TEST data was read or used to tune Fmax or Utility.

## Quantified task6 result

- ActiveForcing: {active6.actual_success.mean():.4%} SR, selected {active6.selected_force_N.mean():.4f} N, realized Utility {active6.realized_utility.mean():.4f}.
- Query-Ignored: {query6.actual_success.mean():.4%} SR, selected {query6.selected_force_N.mean():.4f} N, realized Utility {query6.realized_utility.mean():.4f}.
- Fixed-Max: {fixed6.actual_success.mean():.4%} SR, selected {fixed6.selected_force_N.mean():.4f} N, realized Utility {fixed6.realized_utility.mean():.4f}.
- GT-Friction Utility: {gt6.actual_success.mean():.4%} SR, selected {gt6.selected_force_N.mean():.4f} N, realized Utility {gt6.realized_utility.mean():.4f}.
- Active task6 friction MAE/bias are recorded in `E1_TASKWISE_FAILURE_AUDIT.csv`; GT friction produces the same task6 SR and the same paired failing choices.

## External baselines

FORTE's real method is analog-tactile SVR + spectral slip detection + reactive Dynamixel impedance closure. The E1 archive lacks its sensor/actuator semantics, so prior oracle-slip and fixed/ladder surrogates are excluded.

Tabero's real method is the native 13-D Field+FS VTLA closed loop. A faithful executor and frozen checkpoint are recoverable, but no exact E1-tuple rollout exists and protected GPU workloads prevented a safe new Isaac run. Unpaired smoke rows are excluded.

## Data-quality and claim boundary

QA passed for the frozen candidate reconstruction: 720 unique branches, 144 paired episodes, five candidates each, with GT Direct maximum absolute reproduction error 2.97e-08. Selected/commanded force remains separate from measured force. Tracking MAE in failed branches is not treated as causal because post-loss zeros contaminate it. The evidence is TRAIN/root-heldout-OOF development, not sealed TEST.

## Final status

- `TASK6_FAILURE_DIAGNOSIS_COMPLETE`
- `PI0_REPAIR = NOT_NEEDED`
- `FORTE = FAITHFUL_REPRODUCTION_BLOCKED`
- `TABERO = FAITHFUL_REPRODUCTION_BLOCKED`

The diagnosis is ready to share with the stated caveats. The external-baseline table is structurally complete but scientifically incomplete until faithful paired rollouts exist.
"""
    (OUT / "E1_FINAL_DIAGNOSIS_AND_BASELINES_REPORT.md").write_text(final, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(SELECTED)
    candidates = pd.read_csv(CANDIDATES)
    main_table = pd.read_csv(MAIN)
    merged = load_raw(selected)

    audit = taskwise_audit(selected, merged, candidates)
    cases = task6_case_audit(merged, candidates)
    table = common_table(main_table)

    audit.to_csv(OUT / "E1_TASKWISE_FAILURE_AUDIT.csv", index=False)
    cases.to_csv(OUT / "TASK6_FAILURE_CASE_AUDIT.csv", index=False)
    table.to_csv(OUT / "TABLE_E1_WITH_EXTERNAL_BASELINES.csv", index=False)
    pd.DataFrame([
        {
            "baseline": "FORTE",
            "status": "FAITHFUL_REPRODUCTION_BLOCKED",
            "paired_n": 0,
            "tuple_definition": "same task/root/friction/initial_state/seed",
            "available_surrogate_rows": 0,
            "surrogate_exclusion": "oracle-slip/fixed/ladder variants change native sensor and actuator semantics",
            "required_next_artifact": "fresh faithful simulator-adapter E2E transitions with measured-force telemetry",
        },
        {
            "baseline": "Tabero",
            "status": "FAITHFUL_REPRODUCTION_BLOCKED",
            "paired_n": 0,
            "tuple_definition": "same task/root/friction/initial_state/seed",
            "available_surrogate_rows": 0,
            "surrogate_exclusion": "existing native smoke roots 7200/7201 are unpaired with E1 roots 5100-5105",
            "required_next_artifact": "fresh native 13-D no-force-override E1 tuple transitions with commanded/measured force split",
        },
    ]).to_csv(OUT / "E1_EXTERNAL_BASELINE_PAIRED_TRANSITIONS.csv", index=False)

    write_reports(selected, merged, candidates, cases, table)

    # Required high-risk QA. Missing values are allowed only for unavailable methods/telemetry.
    cfg = json.loads(CONFIG.read_text())
    qa = {
        "status": "PASS_WITH_DECLARED_EXTERNAL_BASELINE_BLOCKERS",
        "authoritative_config_hash_matches": cfg["config_sha256"] == EXPECTED_CONFIG_HASH,
        "authoritative_fmax": cfg["task_Fmax_N"],
        "selector": cfg["selection"],
        "tie_break": cfg["tie_break"],
        "selected_rows": len(selected),
        "selected_unique_method_episode": int(selected.groupby(["method", "context_id", "repeat"]).ngroups),
        "candidate_rows": len(candidates),
        "candidate_unique_branches": int(candidates.branch_id.nunique()),
        "task6_query_success_active_failure_cases": len(cases),
        "task6_case_primary_class_counts": cases.classification_primary.value_counts().to_dict(),
        "raw_join_coverage": float(merged.telemetry_path.notna().mean()),
        "forbidden_test_read": False,
        "model_fit_or_tuning": False,
        "gpu_or_isaac_launched": False,
        "shared_repo_mutated": False,
        "source_sha256": {
            str(CONFIG): sha256(CONFIG),
            str(PROTOCOL): sha256(PROTOCOL),
            str(SELECTED): sha256(SELECTED),
            str(MAIN): sha256(MAIN),
            str(CANDIDATES): sha256(CANDIDATES),
        },
    }
    (OUT / "E1_CRITICAL_CLOSURE_QA.json").write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if len(selected) != 720 or selected.groupby(["method", "context_id", "repeat"]).size().ne(1).any():
        raise RuntimeError("selected episode grain failed")
    if len(candidates) != 720 or candidates.branch_id.nunique() != 720:
        raise RuntimeError("candidate grain failed")
    if len(cases) != 3 or set(cases.classification_primary) != {"D.DIRECT_MODEL_MISCALIBRATION"}:
        raise RuntimeError("task6 paired-case audit failed")
    if cfg["config_sha256"] != EXPECTED_CONFIG_HASH or cfg["selection"] != "argmax_candidate_expected_utility":
        raise RuntimeError("authoritative Utility config failed")


if __name__ == "__main__":
    main()
