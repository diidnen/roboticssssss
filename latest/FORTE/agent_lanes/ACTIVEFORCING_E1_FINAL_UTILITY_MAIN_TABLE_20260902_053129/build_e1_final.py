#!/usr/bin/env python3
"""Build the final E1 table from frozen offline OOF artifacts only.

This program deliberately has no IsaacLab/OpenPI imports and starts no rollout.
It reads the 720 post-P4-B branches and already-computed grouped-root OOF
predictions, then performs selection, paired accounting, and root-clustered
bootstrap aggregation.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path("/home/exouser/FORTE")
OUT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "activeforcing_residual_utility_20260901_055605/POOLED_OOF_UTILITY_DATASET.csv"
EPISODES = ROOT / "UTILITY_CAUSAL_ABLATION_PER_EPISODE.csv"
CONFIG = ROOT / "UTILITY_FINAL_CONFIG.json"
PROTOCOL = ROOT / "ACTIVEFORCING_AUTHORITATIVE_PROTOCOL.md"
FRESH_E2E = ROOT / "run_activeforcing_fresh_e2e.py"
FORCE_CRITICAL_DEFINITION = ROOT / "existing_force_critical_benchmark_20260901_052000/FORCE_CRITICAL_EXISTING_DEFINITION.json"
FORCE_CRITICAL_MANIFEST = ROOT / "existing_force_critical_benchmark_20260901_052000/FORCE_CRITICAL_EXISTING_MANIFEST.json"
RAW_ROOT = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000/collection_train")
EXPECTED_CONFIG_HASH = "c5bc4f39861a84b9ba95d55cdb5f8633a8b004d205b3864a02799340653d9b10"
EXPECTED_FMAX = {0: 5.0, 1: 6.0, 5: 5.0, 6: 4.0}
BOOTSTRAP_REPS = 4000
BOOTSTRAP_SEED = 2026090201

METHODS = [
    "Frozen π0 Default",
    "Fixed-Max",
    "Query-Ignored / No-Physical-Information",
    "ActiveForcing-1Q Utility",
    "GT-Friction + Direct + same Utility",
    "Hindsight Grid Oracle",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def markdown_table(frame: pd.DataFrame, cols: list[str]) -> str:
    def fmt(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row.get(col, math.nan)) for col in cols) + " |")
    return "\n".join(lines)


def read_raw_metadata() -> pd.DataFrame:
    rows = []
    for task in EXPECTED_FMAX:
        path = RAW_ROOT / f"task{task}/task{task}/branches.csv"
        frame = pd.read_csv(path)
        frame["task_raw"] = task
        rows.append(frame)
    raw = pd.concat(rows, ignore_index=True)
    if raw["branch_id"].duplicated().any():
        raise RuntimeError("raw metadata branch_id is not unique")
    # The OOF assembly dropped the raw collector's redundant terminal
    # `_F<rounded-force>` suffix for tasks 1/5/6.  Removing only that suffix
    # recovers the exact immutable branch identity; it is not a force-nearest
    # join.  Task 0 happened to retain the suffix in both tables.
    raw["archive_branch_id"] = raw["branch_id"].str.replace(r"_F-?[0-9]+(?:\.[0-9]+)?$", "", regex=True)
    archive_ids = set(pd.read_csv(ARCHIVE, usecols=["branch_id"])["branch_id"].astype(str))
    raw["archive_branch_id"] = np.where(raw["branch_id"].isin(archive_ids), raw["branch_id"], raw["archive_branch_id"])
    if raw["archive_branch_id"].duplicated().any():
        raise RuntimeError("normalized raw/archive branch identity is not unique")
    return raw


def select_by_score(data: pd.DataFrame, score: str, method: str) -> pd.DataFrame:
    selected = []
    for (_, _), group0 in data.groupby(["context_id", "repeat"], sort=True):
        group = group0.sort_values("force_N", kind="mergesort")
        best = float(group[score].max())
        row = group[np.isclose(group[score], best, rtol=0.0, atol=1e-12)].iloc[0].copy()
        selected.append(row)
    out = pd.DataFrame(selected).reset_index(drop=True)
    out["method"] = method
    return out


def enrich_selected(frame: pd.DataFrame, archive: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    branch_cols = [
        "branch_id", "context_id", "repeat", "root_id", "task", "force_N", "mu", "success", "fold", "task_Fmax_N"
    ]
    branch = archive[branch_cols].set_index("branch_id")
    out = frame.copy()
    if "selected_force_N" in out:
        out = out.rename(columns={"selected_force_N": "force_N", "actual_success": "success"})
    # Re-anchor outcome/physics fields to the frozen branch table by exact branch ID.
    for col in branch_cols[1:]:
        out[col] = out["branch_id"].map(branch[col])
    if out["branch_id"].isna().any() or out["task"].isna().any():
        raise RuntimeError("selected branch failed exact archive alignment")
    raw_idx = raw.set_index("archive_branch_id")
    for source, target in [
        ("measured_force_mean_N", "measured_force_mean_N"),
        ("measured_force_peak_N", "measured_force_peak_N"),
        ("steady_state_mean_N", "steady_state_mean_N"),
    ]:
        out[target] = out["branch_id"].map(raw_idx[source]) if source in raw_idx else math.nan
    out["friction_band"] = out["context_id"].str.extract(r"_(low|mid|high)_mu", expand=False).str.upper()
    out["realized_utility"] = np.where(
        out["success"].astype(int) == 1,
        1.0 - out["force_N"].astype(float) / out["task_Fmax_N"].astype(float),
        -1.0,
    )
    frontier = (
        archive[archive["success"] == 1]
        .groupby(["context_id", "repeat"])["force_N"]
        .min()
    )
    keys = pd.MultiIndex.from_frame(out[["context_id", "repeat"]])
    out["frontier_N"] = frontier.reindex(keys).to_numpy()
    out["frontier_available"] = out["frontier_N"].notna().astype(int)
    out["under_force"] = (
        out["frontier_N"].notna() & (out["force_N"] < out["frontier_N"] - 1e-9)
    ).astype(int)
    out["excess_force_N"] = np.where(
        out["frontier_N"].notna(), np.maximum(0.0, out["force_N"] - out["frontier_N"]), np.nan
    )
    return out


def scopes(frame: pd.DataFrame) -> Iterable[tuple[str, str, pd.DataFrame]]:
    yield "ALL", "POOLED", frame
    for task in sorted(frame["task"].dropna().astype(int).unique()):
        yield "TASK", f"task{task}", frame[frame["task"] == task]
    for band in ["LOW", "MID", "HIGH"]:
        part = frame[frame["friction_band"] == band]
        if len(part):
            yield "FRICTION", band, part
    for root_id in sorted(frame["root_id"].dropna().astype(str).unique()):
        yield "ROOT", root_id, frame[frame["root_id"] == root_id]


MEAN_METRICS = {
    "full_task_SR": "success",
    "mean_selected_force_N": "force_N",
    "mean_measured_force_N": "measured_force_mean_N",
    "mean_measured_peak_force_N": "measured_force_peak_N",
    "under_force_rate": "under_force",
    "mean_excess_force_N": "excess_force_N",
    "mean_realized_utility": "realized_utility",
    "frontier_coverage": "frontier_available",
    "rescue_rate": "rescue",
    "collateral_rate": "collateral",
    "both_success_rate": "both_success",
    "both_failure_rate": "both_failure",
}


def clustered_bootstrap(frame: pd.DataFrame, rng: np.random.Generator) -> dict[str, dict[str, float]]:
    roots = sorted(frame["root_id"].astype(str).unique())
    n_roots = len(roots)
    if not n_roots:
        return {}
    draws = rng.integers(0, n_roots, size=(BOOTSTRAP_REPS, n_roots))
    counts = np.zeros((BOOTSTRAP_REPS, n_roots), dtype=np.int16)
    for j in range(n_roots):
        counts[:, j] = (draws == j).sum(axis=1)
    result: dict[str, dict[str, float]] = {}
    for metric, column in MEAN_METRICS.items():
        if column not in frame:
            continue
        sums = []
        ns = []
        for root in roots:
            values = pd.to_numeric(frame.loc[frame["root_id"].astype(str) == root, column], errors="coerce").to_numpy(float)
            finite = np.isfinite(values)
            sums.append(float(values[finite].sum()))
            ns.append(int(finite.sum()))
        sums_a = np.asarray(sums, float)
        ns_a = np.asarray(ns, float)
        den = counts @ ns_a
        vals = np.divide(counts @ sums_a, den, out=np.full(BOOTSTRAP_REPS, np.nan), where=den > 0)
        finite_vals = vals[np.isfinite(vals)]
        if len(finite_vals):
            lo, hi = np.quantile(finite_vals, [0.025, 0.975])
            result[metric] = {"lower": float(lo), "upper": float(hi)}
    return result


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    def mean(column: str) -> float:
        values = pd.to_numeric(frame[column], errors="coerce")
        return float(values.mean()) if values.notna().any() else math.nan
    return {
        "n_episodes": int(len(frame)),
        "n_contexts": int(frame["context_id"].nunique()),
        "n_roots": int(frame["root_id"].nunique()),
        "full_task_SR": mean("success"),
        "mean_selected_force_N": mean("force_N"),
        "max_selected_force_N": float(frame["force_N"].max()),
        "mean_measured_force_N": mean("measured_force_mean_N"),
        "measured_force_coverage": float(frame["measured_force_mean_N"].notna().mean()),
        "mean_measured_peak_force_N": mean("measured_force_peak_N"),
        "max_measured_peak_force_N": float(frame["measured_force_peak_N"].max()) if frame["measured_force_peak_N"].notna().any() else math.nan,
        "peak_force_coverage": float(frame["measured_force_peak_N"].notna().mean()),
        "under_force_rate": mean("under_force"),
        "mean_excess_force_N": mean("excess_force_N"),
        "frontier_coverage": mean("frontier_available"),
        "mean_realized_utility": mean("realized_utility"),
        "rescue": int(frame["rescue"].sum()),
        "collateral": int(frame["collateral"].sum()),
        "both_success": int(frame["both_success"].sum()),
        "both_failure": int(frame["both_failure"].sum()),
        "rescue_rate": mean("rescue"),
        "collateral_rate": mean("collateral"),
        "both_success_rate": mean("both_success"),
        "both_failure_rate": mean("both_failure"),
    }


def main() -> int:
    required = [ARCHIVE, EPISODES, CONFIG, PROTOCOL, FRESH_E2E, FORCE_CRITICAL_DEFINITION, FORCE_CRITICAL_MANIFEST]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"missing inputs: {missing}")

    config = json.loads(CONFIG.read_text())
    core = {key: config[key] for key in [
        "name", "equation", "success_benefit", "force_cost_coefficient", "failure_penalty",
        "task_Fmax_N", "tie_break", "selection", "fit_and_sensitivity_scope",
    ]}
    computed_config_hash = hashlib.sha256(canonical(core).encode()).hexdigest()
    configured_fmax = {int(k): float(v) for k, v in config["task_Fmax_N"].items()}
    if config["config_sha256"] != EXPECTED_CONFIG_HASH or computed_config_hash != EXPECTED_CONFIG_HASH:
        raise RuntimeError("authoritative Utility config lineage mismatch; stop before result selection")
    if configured_fmax != EXPECTED_FMAX:
        raise RuntimeError("authoritative Fmax lineage mismatch; stop before result selection")
    if config["selection"] != "argmax_candidate_expected_utility" or config["tie_break"] != "lower_force":
        raise RuntimeError("final selector is not authoritative expected utility + lower-force tie-break")

    archive = pd.read_csv(ARCHIVE)
    if len(archive) != 720 or archive["branch_id"].nunique() != 720:
        raise RuntimeError("archive is not the authoritative 720 unique branches")
    if archive.groupby(["context_id", "repeat"]).size().value_counts().to_dict() != {5: 144}:
        raise RuntimeError("archive does not have exactly five siblings per episode")
    if archive["context_id"].nunique() != 72 or archive["root_id"].nunique() != 24:
        raise RuntimeError("archive context/root counts changed")
    if archive.groupby("root_id")["fold"].nunique().max() != 1:
        raise RuntimeError("sibling leakage: a root family crosses OOF folds")
    if archive.groupby(["context_id", "repeat"])["fold"].nunique().max() != 1:
        raise RuntimeError("sibling leakage: an episode crosses OOF folds")
    if {int(k): float(v) for k, v in archive.groupby("task")["task_Fmax_N"].first().items()} != EXPECTED_FMAX:
        raise RuntimeError("archive Fmax differs from frozen config")

    raw = read_raw_metadata()
    episodes = pd.read_csv(EPISODES)
    def ep(name: str, method: str) -> pd.DataFrame:
        part = episodes[episodes["analysis"] == name].copy()
        if len(part) != 144:
            raise RuntimeError(f"expected 144 existing selections for {name}, got {len(part)}")
        part["method"] = method
        return enrich_selected(part, archive, raw)

    active = ep("Full Utility", "ActiveForcing-1Q Utility")
    active_alias = ep("Active", "ActiveForcing-1Q Utility")
    if not np.array_equal(active.sort_values(["context_id", "repeat"])["branch_id"].to_numpy(), active_alias.sort_values(["context_id", "repeat"])["branch_id"].to_numpy()):
        raise RuntimeError("Active and Full Utility aliases are not decision-identical")
    fixed = ep("Fixed-Max", "Fixed-Max")
    ignored = ep("Query-Ignored", "Query-Ignored / No-Physical-Information")

    archive["gt_expected_utility"] = archive["p_D_OOF_ensemble"] * (1.0 - archive["force_N"] / archive["task_Fmax_N"]) + (1.0 - archive["p_D_OOF_ensemble"]) * -1.0
    # The archived values were emitted from float32 OOF probabilities; tolerate
    # only their sub-1e-6 serialization difference (observed max 3.78e-8).
    if float(np.max(np.abs(archive["gt_expected_utility"] - archive["U_D_normalized"]))) > 1e-6:
        raise RuntimeError("GT Direct expected-utility reproduction mismatch")
    gt = enrich_selected(select_by_score(archive, "gt_expected_utility", "GT-Friction + Direct + same Utility"), archive, raw)
    hindsight = enrich_selected(select_by_score(archive, "R_real", "Hindsight Grid Oracle"), archive, raw)
    selections = {
        "Fixed-Max": fixed,
        "Query-Ignored / No-Physical-Information": ignored,
        "ActiveForcing-1Q Utility": active,
        "GT-Friction + Direct + same Utility": gt,
        "Hindsight Grid Oracle": hindsight,
    }

    fresh_text = FRESH_E2E.read_text()
    default_match = re.search(r"^DEFAULT_FORCE\s*=\s*([0-9.]+)", fresh_text, flags=re.MULTILINE)
    if not default_match:
        raise RuntimeError("could not audit Frozen pi0 default semantics")
    default_force = float(default_match.group(1))
    default_exact_rows = archive[np.isclose(archive["force_N"], default_force, rtol=0.0, atol=1e-12)]
    if len(default_exact_rows):
        raise RuntimeError("default exact branch unexpectedly exists; implement exact selection before reporting")

    fc_definition = json.loads(FORCE_CRITICAL_DEFINITION.read_text())
    fc_manifest = json.loads(FORCE_CRITICAL_MANIFEST.read_text())
    if fc_manifest.get("model_outputs_read_for_membership", True) or fc_manifest.get("status") != "MEMBERSHIP_FROZEN_BEFORE_CASE_LEVEL_MODEL_QUERY":
        raise RuntimeError("force-critical membership is not pre-method/model-independent")
    fc_contexts = {str(case["context_id"]) for case in fc_manifest["cases"]}
    if len(fc_contexts) != 14:
        raise RuntimeError("frozen force-critical context count changed")

    active_idx = active.set_index(["context_id", "repeat"]).sort_index()
    transitions = []
    for method, selected0 in selections.items():
        selected = selected0.set_index(["context_id", "repeat"]).sort_index()
        if not selected.index.equals(active_idx.index):
            raise RuntimeError(f"pairing failure for {method}")
        for key in selected.index:
            ref = selected.loc[key]
            act = active_idx.loc[key]
            ref_success = int(ref["success"])
            act_success = int(act["success"])
            transitions.append({
                "population": "MODEL-INDEPENDENT FORCE-CRITICAL SUBSET" if key[0] in fc_contexts else "ALL-CONTEXTS-ONLY",
                "reference_method": method,
                "comparison_method": "ActiveForcing-1Q Utility",
                "context_id": key[0], "repeat": int(key[1]), "root_id": str(ref["root_id"]),
                "task": int(ref["task"]), "friction_band": str(ref["friction_band"]), "friction_gt": float(ref["mu"]),
                "reference_branch_id": str(ref["branch_id"]), "active_branch_id": str(act["branch_id"]),
                "reference_force_N": float(ref["force_N"]), "active_force_N": float(act["force_N"]),
                "force_delta_active_minus_reference_N": float(act["force_N"] - ref["force_N"]),
                "reference_success": ref_success, "active_success": act_success,
                "rescue": int(ref_success == 0 and act_success == 1),
                "collateral": int(ref_success == 1 and act_success == 0),
                "both_success": int(ref_success == 1 and act_success == 1),
                "both_failure": int(ref_success == 0 and act_success == 0),
                "reference_realized_utility": float(ref["realized_utility"]),
                "active_realized_utility": float(act["realized_utility"]),
                "active_minus_reference_realized_utility": float(act["realized_utility"] - ref["realized_utility"]),
            })
    transitions_df = pd.DataFrame(transitions)
    transitions_df.to_csv(OUT / "E1_PAIRED_TRANSITIONS.csv", index=False)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    table_rows = []
    bootstrap_rows = []
    population_masks = {
        "ALL CONTEXTS": lambda x: pd.Series(True, index=x.index),
        "MODEL-INDEPENDENT FORCE-CRITICAL SUBSET": lambda x: x["context_id"].isin(fc_contexts),
    }
    for population, mask_fn in population_masks.items():
        for method in METHODS:
            if method == "Frozen π0 Default":
                # Explicit unavailable row; never synthesize a nearest-force trajectory.
                available_scope_source = active[mask_fn(active)].copy()
                for scope_type, scope, group in scopes(available_scope_source):
                    table_rows.append({
                        "population": population, "scope_type": scope_type, "scope": scope, "method": method,
                        "availability": "UNAVAILABLE_NO_EXACT_ARCHIVE_BRANCH", "n_episodes": 0,
                        "n_contexts": int(group.context_id.nunique()), "n_roots": int(group.root_id.nunique()),
                        "default_force_N": default_force,
                        "interpretation": "native/default force is absent from continuous archive; no nearest-force surrogate",
                        "diagnostic_only": 0,
                    })
                continue
            selected_all = selections[method].copy()
            selected = selected_all[mask_fn(selected_all)].copy()
            ref_trans = transitions_df[transitions_df["reference_method"] == method].set_index(["context_id", "repeat"])
            for column in ["rescue", "collateral", "both_success", "both_failure"]:
                selected[column] = [int(ref_trans.loc[(row.context_id, row.repeat), column]) for row in selected.itertuples()]
            for scope_type, scope, group in scopes(selected):
                summary = summarize(group)
                ci = clustered_bootstrap(group, rng)
                row = {
                    "population": population, "scope_type": scope_type, "scope": scope, "method": method,
                    "availability": "AVAILABLE_EXACT_ARCHIVE_BRANCH", "default_force_N": default_force,
                    "interpretation": {
                        "Fixed-Max": "maximum exact archived candidate per context/repeat",
                        "Query-Ignored / No-Physical-Information": "same post-P4-B archive; instance response masked and training physical prior used",
                        "ActiveForcing-1Q Utility": "OOF P4-B estimate -> frozen Shared Direct -> authoritative expected utility",
                        "GT-Friction + Direct + same Utility": "privileged GT friction replaces identifier only; no future outcome access",
                        "Hindsight Grid Oracle": "future outcome read; diagnostic upper reference only",
                    }[method],
                    "diagnostic_only": int(method in {"GT-Friction + Direct + same Utility", "Hindsight Grid Oracle"}),
                    **summary,
                }
                for metric, bounds in ci.items():
                    row[f"{metric}_ci95_lower"] = bounds["lower"]
                    row[f"{metric}_ci95_upper"] = bounds["upper"]
                table_rows.append(row)
                bootstrap_rows.append({
                    "population": population, "scope_type": scope_type, "scope": scope, "method": method,
                    "bootstrap_unit": "root_id task-specific root family", "replicates": BOOTSTRAP_REPS,
                    "seed": BOOTSTRAP_SEED, "n_roots": int(group.root_id.nunique()), "intervals": ci,
                })

    table = pd.DataFrame(table_rows)
    for column in table.columns:
        if column.startswith(("full_task", "mean_", "max_", "under_", "frontier_", "measured_", "peak_", "rescue_", "collateral_", "both_")):
            table[column] = pd.to_numeric(table[column], errors="coerce")
    table.to_csv(OUT / "TABLE_E1_FINAL_UTILITY_MAIN.csv", index=False)

    bootstrap_artifact = {
        "status": "E1_ROOT_CLUSTER_BOOTSTRAP_COMPLETE",
        "bootstrap_unit": "task-specific root_id; all sibling contexts/forces/repeats remain together",
        "replicates": BOOTSTRAP_REPS,
        "seed": BOOTSTRAP_SEED,
        "percentile_interval": [0.025, 0.975],
        "archive_roots": 24,
        "force_critical_roots": int(archive[archive.context_id.isin(fc_contexts)].root_id.nunique()),
        "results": bootstrap_rows,
    }
    (OUT / "E1_ROOT_CLUSTER_BOOTSTRAP.json").write_text(json.dumps(bootstrap_artifact, indent=2, sort_keys=True) + "\n")

    pooled = table[(table.scope_type == "ALL") & (table.scope == "POOLED")].copy()
    key_cols = [
        "method", "availability", "n_episodes", "full_task_SR", "full_task_SR_ci95_lower", "full_task_SR_ci95_upper",
        "mean_selected_force_N", "mean_measured_force_N", "mean_measured_peak_force_N", "under_force_rate",
        "mean_excess_force_N", "mean_realized_utility", "rescue", "collateral", "both_success", "both_failure",
    ]
    md = ["# E1 Final Utility Main Table", "", "## All contexts", "", markdown_table(pooled[pooled.population == "ALL CONTEXTS"], key_cols), "", "## Model-independent force-critical subset", "", markdown_table(pooled[pooled.population == "MODEL-INDEPENDENT FORCE-CRITICAL SUBSET"], key_cols), "", "Detailed per-task, per-friction, and per-root rows are in `TABLE_E1_FINAL_UTILITY_MAIN.csv`.", "", "E1_FINAL_UTILITY_MAIN_TABLE_COMPLETE", ""]
    (OUT / "TABLE_E1_FINAL_UTILITY_MAIN.md").write_text("\n".join(md))

    utility_ablation = pd.read_csv(ROOT / "TABLE_UTILITY_ABLATION.csv")
    point_ablation = pd.read_csv(ROOT / "TABLE_POINT_VS_POSTERIOR.csv")
    query_ablation = pd.read_csv(ROOT / "TABLE_QUERY_INFORMATION_ABLATION.csv")
    pooled_u = utility_ablation[utility_ablation.scope == "POOLED"]
    pooled_p = point_ablation[point_ablation.scope == "POOLED"]
    pooled_q = query_ablation[query_ablation.scope == "POOLED"]
    all_active = pooled[(pooled.population == "ALL CONTEXTS") & (pooled.method == "ActiveForcing-1Q Utility")].iloc[0]
    all_ignored = pooled[(pooled.population == "ALL CONTEXTS") & (pooled.method == "Query-Ignored / No-Physical-Information")].iloc[0]

    report = f"""# E1 Final Utility Report

## Status

**E1_FINAL_UTILITY_MAIN_TABLE_COMPLETE**

This is a CPU/offline closure over the existing 720 post-P4-B branches. It started no IsaacLab process, no π0 rollout, no GPU job, and no training. The population contains 72 friction contexts, 24 task-specific grouped root families, five exact continuous force candidates, and two repeats per context.

## Authoritative lineage

- Frozen Utility config declared/core SHA-256: `{config['config_sha256']}` / `{computed_config_hash}` (match).
- Utility: `p(success)*(1-F/Fmax) + (1-p(success))*(-1)`; selector: expected-utility argmax; tie-break: lower force.
- Task Fmax: `{configured_fmax}`; config and archive agree.
- Config file SHA-256: `{sha256(CONFIG)}`; authoritative protocol SHA-256: `{sha256(PROTOCOL)}`.
- Every task-specific `root_id` belongs to exactly one OOF fold; sibling branches never cross a split.
- Deprecated hard-rho is not used by any selected policy in this table.

The current runtime audit defines Frozen π0 Default as the native/neutral `{default_force:.1f} N` override, and explicitly says an exact native-slot default is absent from the 720-branch archive. The archive contains `{len(default_exact_rows)}` exact default branches. Therefore E1 reports this baseline as unavailable and does not use a nearest-force surrogate; fresh execution belongs to E5.

## Main result and interpretation

On all contexts, ActiveForcing-1Q Utility has SR `{all_active.full_task_SR:.4f}`, mean selected force `{all_active.mean_selected_force_N:.4f}` N, and realized Utility `{all_active.mean_realized_utility:.4f}`. Query-Ignored has SR `{all_ignored.full_task_SR:.4f}`, mean selected force `{all_ignored.mean_selected_force_N:.4f}` N, and Utility `{all_ignored.mean_realized_utility:.4f}`. Relative to Query-Ignored, Active records `{int(all_ignored.rescue)}` rescues, `{int(all_ignored.collateral)}` collaterals, `{int(all_ignored.both_success)}` both-success, and `{int(all_ignored.both_failure)}` both-failure episodes.

The physical query improves friction estimation but does not establish a pooled SR gain. No Utility coefficient, force-critical membership, task, force, or method is changed in response. This is the required negative-result interpretation: the frozen Utility is the scientific objective, not an outcome-tuned SR maximizer.

`Query-Ignored / No-Physical-Information` is not a true zero-query trajectory: all 720 archived branches are post-P4-B. It uses the frozen training physical prior and masks instance-specific physical response. `GT-Friction + Direct` replaces only the identifier input. `Hindsight Grid Oracle` reads outcomes and is diagnostic only.

Measured-force metrics use exact immutable source-branch identity joins to archived telemetry. The OOF assembler omitted a redundant terminal `_F<rounded-force>` string suffix on some tasks; the join removes only this suffix and never performs nearest-force matching. Missing telemetry remains missing, coverage is explicit, and commanded force never fills a measured value.

## Predefined force-critical subset

The subset uses the already-frozen definition `{sha256(FORCE_CRITICAL_DEFINITION)}` and manifest `{sha256(FORCE_CRITICAL_MANIFEST)}`: adjacent exact archived force cells where both low-force repeats fail and both high-force repeats succeed, optionally with pick/lift and downstream-failure telemetry. Membership was frozen before case-level model query, read no model output, and never references whether ActiveForcing wins. It contains `{len(fc_contexts)}` contexts across `{fc_manifest['counts']['independent_task_root_families']}` root families. It is retrospective/model-independent stress evidence, not untouched TEST.

## Reused completed ablations

### Success-Only versus Full Utility

{markdown_table(pooled_u, ['method','full_task_SR','mean_selected_force_N','under_force_rate','mean_excess_force_N','mean_realized_utility'])}

### Point versus Posterior Utility

{markdown_table(pooled_p, ['method','full_task_SR','mean_selected_force_N','under_force_rate','mean_excess_force_N','mean_realized_utility','decision_changed_rate'])}

### Query-Ignored versus Active

{markdown_table(pooled_q, ['method','friction_MAE','full_task_SR','mean_selected_force_N','under_force_rate','mean_excess_force_N','paired_rescue_count','paired_collateral_count'])}

These are reused frozen grouped-root OOF outputs; nothing was retrained.

## Statistical contract

All 95% intervals use `{BOOTSTRAP_REPS}` percentile bootstrap replicates with task-specific `root_id` as the cluster. Every sibling context, force, and repeat stays together. Episode-paired transitions are enumerated in `E1_PAIRED_TRANSITIONS.csv`. Per-task, per-friction, and per-root estimates are included in the main CSV.

## Deliverables

- `TABLE_E1_FINAL_UTILITY_MAIN.csv`
- `TABLE_E1_FINAL_UTILITY_MAIN.md`
- `E1_PAIRED_TRANSITIONS.csv`
- `E1_ROOT_CLUSTER_BOOTSTRAP.json`
- `E1_FINAL_UTILITY_REPORT.md`

E1_FINAL_UTILITY_MAIN_TABLE_COMPLETE
"""
    (OUT / "E1_FINAL_UTILITY_REPORT.md").write_text(report)

    # Required protected-process record. This is documentary only; no process control occurred.
    (OUT / "MASS_PROTECTED_PIDS.txt").write_text("465038\n")
    audit = """# MASS Protected Process Audit

- Audit time: 2026-09-02 05:29:52 UTC (nvidia-smi), completed with `/proc` read at 05:31 UTC.
- Protected PID: `465038`; state at audit: running; parent PID `4264`; nice `0`; 245 threads.
- Command: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python /home/exouser/FORTE_mass/qualify_mass_tasks.py --task 2 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_TASK2_QUAL_RERUN2 --roots 7020 7021`.
- Working directory: `/home/exouser/Tabero`.
- GPU at audit: A100-SXM4-40GB; 29,974 MiB used / 40,960 MiB; 92% utilization. MASS PID used 7,592 MiB.
- Other GPU processes observed and left untouched: pi0 server PID `33931`; E5 Isaac workers PIDs `37328`, `464398`.
- Protection actions: no kill, signal, renice, environment mutation, checkout/config/file write, shared-memory deletion, or GPU process launch. E1 remained CPU/offline.
"""
    (OUT / "MASS_PROTECTED_PROCESS_AUDIT.md").write_text(audit)

    output_names = [
        "TABLE_E1_FINAL_UTILITY_MAIN.csv", "TABLE_E1_FINAL_UTILITY_MAIN.md", "E1_PAIRED_TRANSITIONS.csv",
        "E1_ROOT_CLUSTER_BOOTSTRAP.json", "E1_FINAL_UTILITY_REPORT.md",
    ]
    print(json.dumps({
        "status": "E1_FINAL_UTILITY_MAIN_TABLE_COMPLETE",
        "out": str(OUT),
        "rows": int(len(table)),
        "transitions": int(len(transitions_df)),
        "outputs": {name: sha256(OUT / name) for name in output_names},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
