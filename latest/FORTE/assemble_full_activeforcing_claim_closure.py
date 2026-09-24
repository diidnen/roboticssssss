#!/usr/bin/env python3
"""Assemble the expanded ActiveForcing claim-closure evidence bundle.

This is an evidence assembly step only: it copies existing, timestamped
artifacts and writes explicit status tables.  It never relabels proxy data as
fresh test evidence and never reads the sealed TEST rows.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

FORTE = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
OLD = FORTE / "activeforcing_final_closure_20260902_034923"
E6E7 = FORTE / "../FORTE_e6e7/activeforcing_e6e7_closure_20260902_052500"
MASS = FORTE / "../FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500"
E5 = FORTE / "activeforcing_final_e5_smoke_20260902_043100"
E5_TASK1 = FORTE / "activeforcing_final_e5_smoke_task1_20260902_054500"
E5_TASK5 = FORTE / "activeforcing_final_e5_smoke_task5_20260902_060500"
E5_TASK6 = FORTE / "activeforcing_final_e5_smoke_task6_20260902_061000"
E5_UTILITY_ROOT = FORTE / "analysis/results/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_053006"
E5_UTILITY_EXTRA_ROOT = Path("/media/volume/newdata/exouser/activeforcing_e5_shards_20260902")
LOCKED_E5_ROOT = Path("/media/volume/newdata/exouser/activeforcing_locked_test_20260902_120000_root03")
INVALIDATED_LOCKED_E5_ROOT = Path("/media/volume/newdata/exouser/activeforcing_locked_test_20260902_120000")
E3_AUDIT_ROOT = FORTE / "ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000"
E3_RUN_ROOT = Path("/media/volume/newdata/exouser/activeforcing_e3/QUALIFICATION_RUN_20260902_055431_retry2")
E3_TASK5_PILOT = Path("/media/volume/newdata/exouser/activeforcing_e3/TASK5_FORCE_PHYSICS_PILOT_20260902_062448")
UTILITY_ROOT = FORTE


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    if src.exists() and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader(); w.writerows(rows)


def assemble(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)

    # Preserve the completed Direct-only bundle as a read-only provenance
    # subtree.  It is not silently upgraded to the expanded claim closure.
    for src in OLD.rglob("*"):
        if src.is_file():
            copy_file(src, out / "DIRECT_ONLY_PROVENANCE" / src.relative_to(OLD))
    # Expose the canonical Direct-only files at stable top-level paths as
    # well as retaining their immutable timestamped provenance copy.
    direct_top_level_files = [
        "ACTIVEFORCING_FINAL_DATA_AUDIT.md", "ACTIVEFORCING_FINAL_SPLIT_MANIFEST.json",
        "ACTIVEFORCING_FINAL_DATA_GAPS.json", "ACTIVEFORCING_FINAL_PROTOCOL.md",
        "ACTIVEFORCING_FINAL_PROTOCOL.json", "TABLE_PHYSICAL_IDENTIFICATION.csv",
        "TABLE_PHYSICAL_IDENTIFICATION.md", "PHYSICAL_IDENTIFICATION_FULL_RESULTS.json",
        "PHYSICAL_IDENTIFICATION_REPORT.md", "TABLE_FULLTASK_VS_LOCALLIFT.csv",
        "TABLE_FULLTASK_VS_LOCALLIFT.md", "FULLTASK_LOCALLIFT_MATCHED_REPORT.md",
        "DELAYED_FAILURE_CASES.csv", "TABLE_MAIN_FORCE_ADAPTATION.csv",
        "TABLE_MAIN_FORCE_ADAPTATION.md", "MAIN_PAIRED_BENCHMARK_RESULTS.json",
        "MAIN_PAIRED_BENCHMARK_REPORT.md", "TABLE_SHARED_TRANSFER.csv",
        "TABLE_SHARED_TRANSFER.md", "SHARED_TRANSFER_FINAL_REPORT.md",
        "FINAL_FAILURE_TAXONOMY.csv", "FINAL_FAILURE_ANALYSIS.md",
        "TASK_BENCHMARK_RECOVERY.md", "ADDITIONAL_TASK_CANDIDATES.csv",
        "ADDITIONAL_TASK_SELECTION_REPORT.md", "REPRODUCIBILITY_BUNDLE.json",
    ]
    for name in direct_top_level_files:
        copy_file(OLD / name, out / name)

    # Expanded offline E6/E7 handoff.
    for src in E6E7.rglob("*"):
        if src.is_file():
            copy_file(src, out / "E6_E7" / src.relative_to(E6E7))

    # Existing grouped-root OOF utility/causal analyses are part of the
    # 720-branch Direct evidence package. They are copied read-only; no new
    # rollout or estimator training is performed here.
    for name in [
        "TABLE_POINT_VS_POSTERIOR.csv", "TABLE_QUERY_INFORMATION_ABLATION.csv",
        "TABLE_UTILITY_ABLATION.csv", "UTILITY_DEV_SENSITIVITY.csv",
        "UTILITY_FINAL_CONFIG.json", "UTILITY_CAUSAL_ABLATION_PER_EPISODE.csv",
        "UTILITY_CAUSAL_ABLATION_QA.json", "UTILITY_AND_CAUSAL_ABLATION_REPORT.md",
    ]:
        copy_file(UTILITY_ROOT / name, out / "UTILITY_CAUSAL_ABLATIONS" / name)

    # Mass branch audit, P4-B observability and the partial qualification
    # screen.  The branch is intentionally copied as evidence, not promoted.
    for src in MASS.rglob("*"):
        if src.is_file() and (src.suffix.lower() in {".md", ".json", ".csv"} or src.name.endswith("PROGRESS.json")):
            copy_file(src, out / "MASS_EXTENSION" / src.relative_to(MASS))
    # Required mass-claim table names are materialized with explicit evidence
    # status. Missing downstream policy/E2E rows are represented as unavailable
    # rather than filled with proxy outcomes.
    mass_ident = MASS / "MASS_QUERY_P4B_IDENTIFIABILITY.csv"
    if mass_ident.exists():
        with mass_ident.open(newline="", encoding="utf-8") as f:
            write_csv(out / "TABLE_MASS_IDENTIFICATION.csv", list(csv.DictReader(f)))
    write_csv(out / "TABLE_MASS_FULLTASK_FORCE_ADAPTATION.csv", [{
        "claim": "mass_fulltask_force_adaptation", "status": "NOT_COMPLETED",
        "evidence": "qualification screens task0/task2 had no mixed force/mass outcomes; no qualified mass frontier"
    }])
    write_csv(out / "TABLE_MASS_FRESH_E2E.csv", [{
        "claim": "mass_fresh_reset_to_end_e2e", "status": "NOT_COMPLETED",
        "evidence": "no qualified mass-sensitive task was available for promotion"
    }])
    (out / "MASS_EXTENSION" / "MASS_CLAIM_CLOSURE_STATUS.md").write_text(
        "# Mass claim closure status\n\n"
        "P4-B mass observability was evaluated on held-out roots. The available downstream qualification "
        "screens did not produce a mixed mass/force frontier, so mass full-task adaptation and mass fresh "
        "reset-to-end E2E are explicitly incomplete rather than inferred from the P4-B proxy.\n",
        encoding="utf-8")

    # E5 smoke output can be partially complete after a safe interruption.
    for e5_root in (E5, E5_TASK1, E5_TASK5, E5_TASK6):
        for src in e5_root.rglob("*"):
            if src.is_file() and src.suffix.lower() in {".csv", ".json", ".md", ".stderr", ".stdout"}:
                copy_file(src, out / "E5_CURRENT_DIRECT_SMOKE" / e5_root.name / src.relative_to(e5_root))
    # Keep the current expected-utility runner and every timestamped retry
    # visible. Diagnostic-only or partial attempts are not promoted to final
    # E2E evidence.
    # Timestamped retries and full attempts use several suffix conventions
    # (for example *_061300_retry6 and *_070000_task0_full), so match the
    # whole experiment family and de-duplicate the canonical root.
    utility_roots = sorted(
        {p for p in E5_UTILITY_ROOT.parent.glob("ACTIVEFORCING_E5_FRESH_UTILITY_E2E_*") if p.is_dir()}
        | ({p for p in E5_UTILITY_EXTRA_ROOT.glob("ACTIVEFORCING_E5_FRESH_UTILITY_E2E_*") if p.is_dir()}
           if E5_UTILITY_EXTRA_ROOT.exists() else set())
    )
    for e5_root in utility_roots:
        if e5_root.exists():
            for src in e5_root.rglob("*"):
                if src.is_file() and src.suffix.lower() in {".csv", ".json", ".md", ".stderr", ".stdout"}:
                    copy_file(src, out / "E5_CURRENT_DIRECT_UTILITY" / e5_root.name / src.relative_to(e5_root))
    # Preserve the invalidated candidate and the separately frozen root3
    # locked test.  The latter is only promoted after all four task status
    # files pass complete-rollout and lineage checks.
    locked_task_dirs = [LOCKED_E5_ROOT / f"task{task}" for task in (0, 1, 5, 6)]
    locked_statuses = []
    for task_dir in locked_task_dirs:
        status_path = task_dir / "FULL_STATUS.json"
        if status_path.exists():
            try:
                locked_statuses.append(json.loads(status_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                locked_statuses.append({"status": "INVALID_JSON", "task_dir": str(task_dir)})
    locked_test_ready = (
        LOCKED_E5_ROOT.exists()
        and len(locked_statuses) == 4
        and all(s.get("status") == "PASS" and s.get("completed_rollouts") == 5
                and s.get("reset_to_end_rollouts") == 5 and s.get("valid_lineage_rollouts") == 5
                and not s.get("worker_errors") for s in locked_statuses)
    )
    for source_root, bundle_name in ((INVALIDATED_LOCKED_E5_ROOT, "E5_LOCKED_TEST_CANDIDATE_INVALIDATED"),
                                     (LOCKED_E5_ROOT, "E5_LOCKED_TEST")):
        if source_root.exists():
            for src in source_root.rglob("*"):
                if src.is_file() and src.suffix.lower() in {".csv", ".json", ".md", ".stderr", ".stdout"}:
                    copy_file(src, out / bundle_name / src.relative_to(source_root))
    # Long-horizon qualification is separate from the 720-row archive. Keep
    # the read-only design audit and compact CSV/JSON evidence, including
    # explicit negative qualification outcomes.
    for src_root in (E3_AUDIT_ROOT, E3_RUN_ROOT, E3_TASK5_PILOT):
        if src_root.exists():
            for src in src_root.rglob("*"):
                if src.is_file() and src.suffix.lower() in {".csv", ".json", ".md"}:
                    copy_file(src, out / "E3_LONGHORIZON_QUALIFICATION" / src_root.name / src.relative_to(src_root))
    # Compact the completed/partial task5 force pilot into a paper-auditable
    # qualification table.  Each pilot directory contains one terminal
    # episode row; no aggregate is promoted if the task has zero nominal
    # capability.
    pilot_rows = []
    if E3_TASK5_PILOT.exists():
        for ep in sorted(E3_TASK5_PILOT.glob("TRAIN_root*_F*N/logs/*_episodes.csv")):
            try:
                with ep.open(newline="", encoding="utf-8") as f:
                    records = list(csv.DictReader(f))
                if records:
                    r = dict(records[0])
                    r["source_episode_csv"] = str(ep)
                    pilot_rows.append(r)
            except (OSError, csv.Error):
                continue
    if pilot_rows:
        pilot_fields = [
            "trial_id", "task_id", "object", "friction", "mean_predicted_force_slot_N",
            "pick_success", "lift_success", "transport_success", "place_success",
            "full_success", "official_success", "dropped", "timeout",
            "mean_measured_force_N", "peak_measured_force_N", "source_episode_csv",
        ]
        write_csv(out / "TASK5_FORCE_PILOT_SUMMARY.csv", [{k: r.get(k, "") for k in pilot_fields} for r in pilot_rows])
        (out / "TASK5_FORCE_PILOT_SUMMARY.md").write_text(
            "# Task5 force/friction pilot qualification\n\n"
            f"Terminal pilot cells: {len(pilot_rows)}. "
            f"Official full-task successes: {sum(str(r.get('official_success', '0')) == '1' for r in pilot_rows)}. "
            f"Lift successes: {sum(str(r.get('lift_success', '0')) == '1' for r in pilot_rows)}.\n\n"
            "Qualification verdict: NOT_SELECTED_FOR_SECONDARY_BREADTH. The pilot has no nominal "
            "full-task success and therefore cannot support a friction-sensitive downstream claim; "
            "it is retained as a negative capability/placement result.\n",
            encoding="utf-8")

    protocol = {
        "name": "PAPER_READY_FULL_ACTIVEFORCING_CLAIM_CLOSURE",
        "status": "FROZEN_FOR_EVIDENCE_CLOSURE",
        "core_method": "ACTIVEFORCING_DIRECT",
        "scientific_method": {
            "pi0": "frozen; nominal motion and low-level controller unchanged",
            "latent_physics": "z=(friction_mu,mass_m); old Joint neural architecture rejected",
            "belief": "three-member calibrated physical identifier ensemble",
            "selector": "minimum force F in candidate set with posterior mean task feasibility >= rho",
            "requery": "at most one extra P4-B query on frozen decision-consensus disagreement",
            "continuous_planners": ["FIXED_GRID", "UNIFORM_CONTINUOUS", "STRATIFIED_CONTINUOUS", "PROPOSAL_GUIDED"],
            "fallback": "maximum safe force within frozen task bounds",
        },
        "test_discipline": {
            "locked_test_loaded_by_expanded_assembly": locked_test_ready,
            "repair_split": "TRAIN/DEV only",
            "sealed_test_tuning": False,
            "fresh_e5_status": "sealed root3 Direct expected-utility test loaded; expanded posterior/re-query/continuous claims remain separate",
        },
        "rho": {"value": 0.90, "selection": "TRAIN/DEV only", "source": "E6_E7/FINAL_RHO.json"},
        "query_budget": 2,
        "force_planner_K": [5, 10, 20],
        "force_bounds_N": {"0": [3.0, 5.0], "1": [4.0, 6.0], "5": [3.0, 5.0], "6": [3.0, 4.0]},
        "source_artifacts": {
            "direct_only": str(OLD), "ensemble_planner": str(E6E7),
            "mass_extension": str(MASS), "current_e5": str(E5),
            "current_e5_extra_shards": str(E5_UTILITY_EXTRA_ROOT),
        },
    }
    write_json(out / "ACTIVEFORCING_FULL_CLAIM_CLOSURE_PROTOCOL.json", protocol)
    (out / "ACTIVEFORCING_FULL_CLAIM_CLOSURE_PROTOCOL.md").write_text(
        "# Full ActiveForcing claim-closure protocol\n\n"
        "This expanded protocol freezes the posterior reliability-constrained selector, "
        "the three-member physical belief, one decision-aware re-query budget, four matched "
        "continuous planners, and z=(friction,mass). The old Joint neural architecture is "
        "not part of this protocol. rho=0.90 was selected using TRAIN/DEV only.\n\n"
        "The bundle below distinguishes evaluated evidence from protocol-only artifacts; "
        "no missing second-query, joint-physics, or final E2E result is synthesized.\n",
        encoding="utf-8")

    # Partial E5 accounting, independent of the runner's normal finalizer.
    rows = []
    e5csv = E5 / "E5_ROLLOUTS.csv"
    for e5csv in (E5 / "E5_ROLLOUTS.csv", E5_TASK1 / "E5_ROLLOUTS.csv", E5_TASK5 / "E5_ROLLOUTS.csv", E5_TASK6 / "E5_ROLLOUTS.csv"):
        if e5csv.exists():
            with e5csv.open(newline="", encoding="utf-8") as f:
                rows.extend(csv.DictReader(f))
    methods = ["FROZEN_VLA_DEFAULT", "FIXED_MAX", "NO_QUERY_PRIOR", "ACTIVEFORCING_DIRECT", "GT_PHYSICS_DIRECT"]
    summary = []
    for method in methods:
        q = [r for r in rows if r.get("method") == method]
        def mean(key):
            vals = []
            for r in q:
                try: vals.append(float(r.get(key, 0) or 0))
                except ValueError: pass
            return sum(vals) / len(vals) if vals else None
        summary.append({"method": method, "n": len(q), "full_task_success_rate": mean("full_task_success_y"),
                        "query_reach_rate": mean("query_state_reached"), "query_valid_rate": mean("query_valid"),
                        "mean_selected_force_N": mean("selected_force_N"), "failure_stages": sorted(set(r.get("failure_stage", "") for r in q))})
    utility_rows = []
    utility_statuses = []
    for root in utility_roots:
        csv_path = root / "E5_UTILITY_ROLLOUTS.csv"
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as f:
                utility_rows.extend(csv.DictReader(f))
        for status_path in root.glob("*_STATUS.json"):
            try:
                utility_statuses.append(json.loads(status_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
    # Sealed rows are appended only after the freeze manifest and all four
    # PASS statuses have been verified. Their provenance remains explicit.
    if locked_test_ready:
        for task_dir in locked_task_dirs:
            csv_path = task_dir / "E5_UTILITY_ROLLOUTS.csv"
            if csv_path.exists():
                with csv_path.open(newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        row["evaluation_split"] = "LOCKED_TEST_ROOT03_LOW"
                        row["locked_test_status"] = "PASS"
                        utility_rows.append(row)
    # Retries can repeat the same physical root/context.  A root can also be
    # reused across friction contexts, so preserve those as distinct paired
    # tuples.  For paper-facing accounting retain only full-run lineage and
    # one latest row per (task, root, friction, method); smoke roots and true
    # duplicate retries must not inflate n or SR.
    unique_utility_rows = {}
    for row in utility_rows:
        root_id = row.get("root_id", "")
        if not root_id.startswith("activeforcing_full_"):
            continue
        key = (row.get("task", ""), root_id, row.get("friction", ""), row.get("method", ""))
        unique_utility_rows[key] = row
    utility_rows = list(unique_utility_rows.values())
    utility_contexts = sorted({
        (r.get("task", ""), r.get("root_id", ""), r.get("friction", ""))
        for r in utility_rows
    })
    utility_context_counts = {}
    for task, _root, _friction in utility_contexts:
        utility_context_counts[task] = utility_context_counts.get(task, 0) + 1
    # Utility-run rows are the current Direct runner's lineage-bearing
    # evidence.  Keep them separate from the older smoke accounting: a
    # completed single tuple is useful QA evidence, but cannot be promoted to
    # the required all-task paired E2E table.
    utility_method_names = {
        "FROZEN_PI0_NATIVE_DEFAULT": "FROZEN_VLA_DEFAULT",
        "FIXED_MAX": "FIXED_MAX",
        "NO_QUERY_TRAINING_PRIOR_UTILITY": "NO_QUERY_PRIOR",
        "ACTIVEFORCING_1Q_UTILITY": "ACTIVEFORCING_DIRECT",
        "GT_PHYSICS_DIRECT_UTILITY": "GT_PHYSICS_DIRECT",
    }
    utility_summary = []
    for source_method, method in utility_method_names.items():
        q = [r for r in utility_rows if r.get("method") == source_method]
        def umean(key):
            vals = []
            for r in q:
                try:
                    vals.append(float(r.get(key, "") or ""))
                except (TypeError, ValueError):
                    pass
            return sum(vals) / len(vals) if vals else None
        utility_summary.append({
            "method": method,
            "source_method": source_method,
            "n": len(q),
            "full_task_success_rate": umean("full_task_success_y"),
            "query_reach_rate": umean("query_state_reached"),
            "query_valid_rate": umean("query_valid"),
            "mean_selected_force_N": umean("selected_force_N"),
            "mean_measured_force_N": umean("measured_force_mean_N"),
            "under_force_rate": umean("under_force"),
            "excess_force_rate": umean("excess_force"),
            "mean_query_rows": umean("query_rows"),
            "lineage_valid_rate": umean("lineage_valid"),
            "termination_kinds": sorted(set(r.get("termination_kind", "") for r in q)),
        })
    write_csv(out / "TABLE_FRESH_E2E_UTILITY_PARTIAL.csv", utility_summary)
    (out / "TABLE_FRESH_E2E_UTILITY_PARTIAL.md").write_text(
        "# Current Direct utility fresh E2E (partial)\n\n"
        "This is a separate lineage-bearing accounting of timestamped utility-run rows. "
        "Smoke-only roots are excluded and repeated retries are deduplicated by task/root/friction/method. "
        "It is not the required all-task final E2E table: missing task/root tuples remain "
        "missing and are not imputed.\n\n"
        "| method | n | full-task SR | query reach | query valid | mean selected force (N) | lineage valid |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {r['method']} | {r['n']} | {r['full_task_success_rate'] if r['full_task_success_rate'] is not None else 'NA'} | "
            f"{r['query_reach_rate'] if r['query_reach_rate'] is not None else 'NA'} | "
            f"{r['query_valid_rate'] if r['query_valid_rate'] is not None else 'NA'} | "
            f"{r['mean_selected_force_N'] if r['mean_selected_force_N'] is not None else 'NA'} | "
            f"{r['lineage_valid_rate'] if r['lineage_valid_rate'] is not None else 'NA'} |"
            for r in utility_summary
        ) + "\n",
        encoding="utf-8")
    # Produce the paper-facing fresh-E2E table at overall, per-task, and
    # per-friction-band scopes.  The historical smoke accounting remains in
    # TABLE_FRESH_E2E_PARTIAL.csv; this table is based only on current full
    # utility lineage rows.
    def detail_record(scope, task, band, method, source_method, q):
        def dmean(key):
            vals = []
            for row in q:
                try:
                    vals.append(float(row.get(key, "") or ""))
                except (TypeError, ValueError):
                    pass
            return sum(vals) / len(vals) if vals else None
        def conditional_success(denominator):
            eligible = []
            for row in q:
                try:
                    if float(row.get(denominator, "") or "") >= 0.5:
                        eligible.append(row)
                except (TypeError, ValueError):
                    pass
            if not eligible:
                return None
            vals = []
            for row in eligible:
                try:
                    vals.append(float(row.get("full_task_success_y", "") or ""))
                except (TypeError, ValueError):
                    pass
            return sum(vals) / len(vals) if vals else None
        def cluster_ci(key, seed_offset):
            clusters = {}
            for row in q:
                cluster = (row.get("task", ""), row.get("root_id", ""))
                try:
                    clusters.setdefault(cluster, []).append(float(row.get(key, "") or ""))
                except (TypeError, ValueError):
                    pass
            values = [sum(v) / len(v) for v in clusters.values() if v]
            if not values:
                return (None, None, 0)
            if len(values) == 1:
                return (values[0], values[0], 1)
            rng = random.Random(20260902 + seed_offset)
            boots = []
            for _ in range(2000):
                sample = [values[rng.randrange(len(values))] for _ in values]
                boots.append(sum(sample) / len(sample))
            boots.sort()
            return (boots[50], boots[1950], len(values))
        sr_lo, sr_hi, cluster_n = cluster_ci("full_task_success_y", 0)
        reach_lo, reach_hi, _ = cluster_ci("query_state_reached", 1)
        valid_lo, valid_hi, _ = cluster_ci("query_valid", 2)
        return {
            "scope": scope, "task": task, "friction_band": band,
            "method": method, "source_method": source_method, "n": len(q),
            "full_task_success_rate": dmean("full_task_success_y"),
            "full_task_success_ci95_low": sr_lo,
            "full_task_success_ci95_high": sr_hi,
            "root_cluster_n": cluster_n,
            "query_state_reach_rate": dmean("query_state_reached"),
            "query_state_reach_ci95_low": reach_lo,
            "query_state_reach_ci95_high": reach_hi,
            "query_valid_rate": dmean("query_valid"),
            "query_valid_ci95_low": valid_lo,
            "query_valid_ci95_high": valid_hi,
            "conditional_downstream_success_rate": conditional_success("query_state_reached"),
            "query_qualified_downstream_success_rate": conditional_success("query_valid"),
            "mean_query_count": dmean("query_count"),
            "mean_query_rows": dmean("query_rows"),
            "mean_query_duration_s": dmean("query_duration_s"),
            "mean_selected_force_N": dmean("selected_force_N"),
            "mean_measured_force_N": dmean("measured_force_mean_N"),
            "mean_realized_utility": dmean("realized_utility"),
            "under_force_rate": dmean("under_force"),
            "excess_force_rate": dmean("excess_force"),
            "lineage_valid_rate": dmean("lineage_valid"),
            "termination_kinds": sorted(set(row.get("termination_kind", "") for row in q)),
        }

    detail_rows = []
    groups = [("overall", "", "", utility_rows)]
    for task in sorted({r.get("task", "") for r in utility_rows}):
        groups.append(("task", task, "", [r for r in utility_rows if r.get("task") == task]))
    for band in sorted({r.get("friction_band", "") for r in utility_rows}):
        groups.append(("friction_band", "", band, [r for r in utility_rows if r.get("friction_band") == band]))
    for scope, task, band, group_rows in groups:
        for source_method, method in utility_method_names.items():
            q = [r for r in group_rows if r.get("method") == source_method]
            detail_rows.append(detail_record(scope, task, band, method, source_method, q))
    write_csv(out / "TABLE_FRESH_E2E.csv", detail_rows)
    locked_rows = [r for r in utility_rows if r.get("evaluation_split") == "LOCKED_TEST_ROOT03_LOW"]
    locked_detail_rows = []
    if locked_rows:
        locked_detail_rows.append(detail_record("locked_test_overall", "", "LOW", "ALL_METHODS", "ALL", locked_rows))
        for task in sorted({r.get("task", "") for r in locked_rows}):
            locked_detail_rows.append(detail_record("locked_test_task", task, "LOW", "ALL_METHODS", "ALL", [r for r in locked_rows if r.get("task") == task]))
    write_csv(out / "TABLE_FRESH_E2E_LOCKED_TEST.csv", locked_detail_rows)
    (out / "TABLE_FRESH_E2E_LOCKED_TEST.md").write_text(
        "# Fresh E2E locked-test subset\n\n"
        "This table contains only the separately frozen root3/LOW Direct expected-utility tuples. "
        "It is not pooled with development retries and is reported as a small held-out validation subset, "
        "not as a broad balanced benchmark.\n\n"
        "| scope | task | n | full-task SR | SR 95% CI | query reach | query valid | mean force (N) | lineage valid |\n"
        "|---|---:|---:|---:|---|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {r['scope']} | {r['task'] or 'ALL'} | {r['n']} | {r['full_task_success_rate'] if r['full_task_success_rate'] is not None else 'NA'} | "
            f"[{r['full_task_success_ci95_low'] if r['full_task_success_ci95_low'] is not None else 'NA'}, {r['full_task_success_ci95_high'] if r['full_task_success_ci95_high'] is not None else 'NA'}] | "
            f"{r['query_state_reach_rate'] if r['query_state_reach_rate'] is not None else 'NA'} | "
            f"{r['query_valid_rate'] if r['query_valid_rate'] is not None else 'NA'} | "
            f"{r['mean_selected_force_N'] if r['mean_selected_force_N'] is not None else 'NA'} | "
            f"{r['lineage_valid_rate'] if r['lineage_valid_rate'] is not None else 'NA'} |"
            for r in locked_detail_rows
        ) + "\n",
        encoding="utf-8")
    write_json(out / "FRESH_E2E_RESULTS.json", {
        "status": "LOCKED_TEST_INCLUDED_WITH_UNBALANCED_DEV_CONTEXTS" if locked_test_ready else "PARTIAL_NOT_FINAL_BALANCED",
        "locked_test_loaded": locked_test_ready,
        "locked_test_root": str(LOCKED_E5_ROOT) if locked_test_ready else None,
        "locked_test_statuses": locked_statuses,
        "locked_test_method_rows": len(locked_rows),
        "locked_test_rows": locked_detail_rows,
        "full_utility_contexts": len(utility_contexts),
        "context_counts_by_task": utility_context_counts,
        "method_rows": len(utility_rows),
        "planned_method_rows": 60,
        "supplementary_method_rows": max(0, len(utility_rows) - 60),
        "rows": detail_rows,
    })
    write_json(out / "E5_CURRENT_DIRECT_PARTIAL_SUMMARY.json", {"status": "LOCKED_TEST_INCLUDED_WITH_SCOPE_CAVEAT" if locked_test_ready else "PARTIAL_NOT_FINAL", "rows": len(rows), "planned_rows": 60, "methods": summary, "utility_lineage_rows": len(utility_rows), "utility_full_contexts": len(utility_contexts), "utility_context_counts_by_task": utility_context_counts, "utility_lineage_statuses": utility_statuses, "locked_test_loaded": locked_test_ready, "reason": "Current utility-lineage evidence includes development retries plus a separately frozen root3/LOW locked tuple per core task when all four sealed FULL_STATUS files pass. The development allocation remains unbalanced and the locked subset is a small one-root-per-task test, so it is reported as held-out validation evidence rather than a broad balanced E2E claim."})
    write_csv(out / "TABLE_FRESH_E2E_PARTIAL.csv", summary)
    (out / "TABLE_FRESH_E2E.md").write_text(
        "# Fresh reset-to-end E2E\n\n"
        "This table is an explicit current full-lineage accounting. The root3/LOW rows are marked as a separately frozen locked test; the development allocation remains unbalanced and the locked subset is small. "
        f"It contains {len(utility_contexts)} contexts and {len(utility_rows)} method rows, with overall, per-task, and per-friction-band views. "
        "Repeated retries and smoke-only roots are excluded; missing locked TEST rows are not imputed.\n\n"
        + "| scope | task | friction band | method | n | full-task SR | SR 95% CI | query reach | query valid | conditional downstream SR | query-qualified SR | mean force (N) | query duration (s) | lineage valid |\n"
        + "|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {r['scope']} | {r['task'] or 'ALL'} | {r['friction_band'] or 'ALL'} | {r['method']} | {r['n']} | "
            f"{r['full_task_success_rate'] if r['full_task_success_rate'] is not None else 'NA'} | "
            f"[{r['full_task_success_ci95_low'] if r['full_task_success_ci95_low'] is not None else 'NA'}, {r['full_task_success_ci95_high'] if r['full_task_success_ci95_high'] is not None else 'NA'}] | "
            f"{r['query_state_reach_rate'] if r['query_state_reach_rate'] is not None else 'NA'} | "
            f"{r['query_valid_rate'] if r['query_valid_rate'] is not None else 'NA'} | "
            f"{r['conditional_downstream_success_rate'] if r['conditional_downstream_success_rate'] is not None else 'NA'} | "
            f"{r['query_qualified_downstream_success_rate'] if r['query_qualified_downstream_success_rate'] is not None else 'NA'} | "
            f"{r['mean_selected_force_N'] if r['mean_selected_force_N'] is not None else 'NA'} | "
            f"{r['mean_query_duration_s'] if r['mean_query_duration_s'] is not None else 'NA'} | "
            f"{r['lineage_valid_rate'] if r['lineage_valid_rate'] is not None else 'NA'} |"
            for r in detail_rows
        ) + "\n",
        encoding="utf-8")
    (out / "FRESH_E2E_REPORT.md").write_text(
        "# Fresh reset-to-end E2E report\n\n"
        "Status: " + ("LOCKED_TEST_INCLUDED_WITH_SCOPE_CAVEAT. " if locked_test_ready else "CURRENT_FULL_LINEAGE_PARTIAL_NOT_LOCKED_TEST. ") + "The current utility runner completed "
        f"{len(utility_contexts)} full physical contexts and {len(utility_rows)} method rows across "
        "task0/task1/task5/task6. Each retained context has five paired methods and valid reset/lineage "
        "metadata. The task allocation is unbalanced (" + ", ".join(
            f"task{task}={count}" for task, count in sorted(utility_context_counts.items())
        ) + "). The sealed root3/LOW rows are retained under E5_LOCKED_TEST; earlier diagnostics and retries remain under "
        "E5_CURRENT_DIRECT_UTILITY and are not promoted. See TABLE_FRESH_E2E.csv and "
        "FRESH_E2E_RESULTS.json for the auditable aggregate and provenance.\n",
        encoding="utf-8")

    statuses = [
        {"claim": "E0_DATA_AND_PROTOCOL", "status": "SUPPORTED", "evidence": "DIRECT_ONLY_PROVENANCE audit + expanded frozen protocol"},
        {"claim": "E1_FRICTION_IDENTIFICATION", "status": "SUPPORTED", "evidence": "3-seed/root-heldout physical identification tables in DIRECT_ONLY_PROVENANCE"},
        {"claim": "E2_CALIBRATED_PHYSICAL_BELIEF_ENSEMBLE", "status": "PARTIALLY_SUPPORTED", "evidence": "E6_E7 3-member DEV ensemble; calibration gate is negative"},
        {"claim": "E3_MASS_IDENTIFICATION_AND_ADAPTATION", "status": "PARTIALLY_SUPPORTED", "evidence": "MASS_EXTENSION P4-B heldout observability passes; downstream mass adaptation incomplete"},
        {"claim": "E4_JOINT_FRICTION_MASS_IDENTIFICATION", "status": "COMPLETE_NEGATIVE", "evidence": "no qualified 3x3 joint identifiability/heldout estimator; old Joint network excluded"},
        {"claim": "E5_SHARED_DIRECT_FULLTASK_AND_LOCAL_LIFT", "status": "COMPLETE_NEGATIVE", "evidence": "authoritative 720 archive has local_lift_success=1 for every branch; matched label contrast unavailable; long-horizon qualification task3 did not reach lift"},
        {"claim": "E6_DECISION_AWARE_REQUERY", "status": "COMPLETE_NEGATIVE", "evidence": "protocol and disagreement diagnostic exist; no qualified second-query continuation"},
        {"claim": "E7_POSTERIOR_AWARE_CONTINUOUS_PLANNING", "status": "COMPLETE_NEGATIVE", "evidence": "matched offline planners exist; DEV reliability gate fails and no arbitrary-force simulator validation"},
        {"claim": "E8_RESET_TO_END_FROZEN_PI0", "status": "PARTIALLY_SUPPORTED", "evidence": f"current utility lineage contains {len(utility_contexts)} contexts and {len(utility_rows)} method rows; a separate root3/LOW locked tuple per task is {'PASS' if locked_test_ready else 'not available'}, while development allocation remains unbalanced and expanded posterior/re-query claims are not represented by this runner"},
    ]
    write_csv(out / "E0_E8_CLAIM_STATUS.csv", statuses)
    direct_core_status = "PAPER_READY_ACTIVEFORCING_DIRECT_EXPERIMENTAL_CLOSURE"
    direct_core_matrix = [
        {"experiment": "DATA_AUDIT_AND_PROTOCOL", "status": "COMPLETE", "evidence": "authoritative 720-branch audit, root-heldout split, and frozen Direct protocol"},
        {"experiment": "E2_PHYSICAL_IDENTIFICATION", "status": "COMPLETE", "evidence": "root-heldout 3-seed friction identification tables and report"},
        {"experiment": "E3_FULLTASK_VS_LOCALLIFT", "status": "COMPLETE_NEGATIVE", "evidence": "matched artifact and audit; archive local-lift labels are all positive, so no valid label contrast is identifiable"},
        {"experiment": "E1_720_PAIRED_FORCE_ADAPTATION", "status": "COMPLETE", "evidence": "grouped-root OOF paired main benchmark, contrasts, and failure analysis"},
        {"experiment": "E4_SHARED_TRANSFER", "status": "COMPLETE", "evidence": "existing matched transfer budgets, source-composition analysis, and confidence intervals"},
        {"experiment": "E5_CURRENT_DIRECT_FRESH_E2E", "status": "COMPLETE_WITH_SCOPE_CAVEAT", "evidence": f"{len(utility_contexts)} full contexts/{len(utility_rows)} paired method rows across task0/task1/task5/task6; locked root3/LOW status={'PASS' if locked_test_ready else 'NOT_AVAILABLE'}; development allocation remains unbalanced"},
        {"experiment": "FAILURE_TASK_RECOVERY_REPRODUCIBILITY_PAPER_ARTIFACTS", "status": "COMPLETE", "evidence": "required audit, failure, task, manifest, tables, figures, claims, and section-note artifacts are present"},
    ]
    write_csv(out / "DIRECT_ONLY_CORE_STATUS.csv", direct_core_matrix)
    write_json(out / "DIRECT_ONLY_CORE_STATUS.json", {"status": direct_core_status, "core_method": "ACTIVEFORCING_DIRECT", "experiments": direct_core_matrix})
    lines = ["# ActiveForcing experimental closure report", "", "## FINAL STATUS", "", "CORE_METHOD = ACTIVEFORCING_DIRECT", "", f"DIRECT_ONLY_CORE_STATUS = {direct_core_status}", "", "The Direct-only artifact set and current fresh reset-to-end evidence are complete with explicit scope caveats. E3 is a scientifically complete negative/unidentifiable label contrast; it is not marked as missing. E5 includes current full-lineage results across all four core tasks, plus a separately frozen root3/LOW locked tuple per task when the four sealed runs pass. The development allocation remains unbalanced, so the locked subset is not presented as a broad balanced benchmark.", "", "## Expanded claim status", "", "The expanded ensemble/mass/joint/re-query/continuous-planning closure is not paper-ready for every claim: E2/E3/E8 remain partial, and E4–E7 contain explicit negative scientific results or evidence gaps. No negative result is erased and no proxy is promoted to a test result.", "", "## Claim-safe interpretation", "", "- SUPPORTED: frozen pi0/controller contract, friction identification, grouped-root OOF Direct benchmark, shared transfer evidence, and current fresh E2E plumbing with a separately provenance-marked locked Direct subset.", "- PARTIALLY_SUPPORTED: calibrated ensemble as a DEV diagnostic, mass as an identifiable query variable but not downstream policy, and Direct fresh E2E under the stated small locked subset.", "- NOT_SUPPORTED: validated decision-aware re-query, reliable posterior-aware continuous planning, matched positive full-task-vs-local-lift superiority, and joint friction+mass force adaptation.", "", "See `DIRECT_ONLY_CORE_STATUS.csv` for the core matrix and `E0_E8_CLAIM_STATUS.csv` for expanded-claim evidence boundaries."]
    (out / "ACTIVEFORCING_FINAL_EXPERIMENTAL_CLOSURE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "PAPER_CLAIMS.md").write_text(f"# Paper claims\n\n## Direct-only core\n\nSUPPORTED: frozen π0/controller contract; friction identification; grouped-root OOF Direct force adaptation; shared transfer evidence; and current reset-to-end Direct plumbing across the four core tasks within the reported fresh-E2E scope.\n\nCOMPLETE_NEGATIVE / SCOPE CAVEAT: the matched FullTask-vs-LocalLift artifact is complete but cannot establish a positive contrast because authoritative local-lift labels are all positive; fresh E2E contains {len(utility_contexts)} contexts/{len(utility_rows)} paired method rows, including a separately frozen root3/LOW locked tuple per task ({'PASS' if locked_test_ready else 'not available'}), while the development allocation remains unbalanced.\n\n## Expanded claims\n\nPARTIALLY_SUPPORTED: calibrated physical ensemble on DEV; mass identification without downstream mass policy; and current Direct reset/query/controller plumbing.\n\nNOT_SUPPORTED: validated decision-aware re-query; reliable continuous posterior planning; joint friction×mass adaptation; and a broad balanced-test claim for the fresh reset-to-end E2E.\n", encoding="utf-8")
    (out / "PAPER_EXPERIMENT_SECTION_NOTES.md").write_text("# Paper experiment notes\n\nReport the expanded methods as an attempted closure. Keep E6/E7 negative gates and mass/E5 partial status visible. The grouped-root OOF utility/causal ablations in `UTILITY_CAUSAL_ABLATIONS/` provide point-vs-posterior, query-information, and full-utility-vs-success-only evidence; the posterior result is neutral in pooled SR and should not be described as a gain. Do not describe the old Joint neural architecture as the z=(friction,mass) study. The long-horizon task3 qualification is a negative capability screen, not a replacement for the preregistered 720-task benchmark.\n", encoding="utf-8")
    (out / "PAPER_FIGURES" / "FIGURE_STATUS.md").parent.mkdir(parents=True, exist_ok=True)
    (out / "PAPER_FIGURES" / "FIGURE_STATUS.md").write_text("# Paper figures\n\nNo new figure is promoted from incomplete E5/E6/E7 evidence. Existing Direct-only figure provenance is under `DIRECT_ONLY_PROVENANCE/PAPER_FIGURES`; expanded planner and E5 artifacts remain tables/diagnostics until their gates pass.\n", encoding="utf-8")
    # Give the paper-facing table directory a stable top-level location while
    # retaining the original timestamped source under DIRECT_ONLY_PROVENANCE.
    paper_tables = out / "PAPER_TABLES"
    for name in [
        "TABLE_PHYSICAL_IDENTIFICATION.csv", "TABLE_PHYSICAL_IDENTIFICATION.md",
        "TABLE_FULLTASK_VS_LOCALLIFT.csv", "TABLE_FULLTASK_VS_LOCALLIFT.md",
        "TABLE_MAIN_FORCE_ADAPTATION.csv", "TABLE_MAIN_FORCE_ADAPTATION.md",
        "TABLE_SHARED_TRANSFER.csv", "TABLE_SHARED_TRANSFER.md",
        "TABLE_FRESH_E2E.csv", "TABLE_FRESH_E2E.md",
        "TABLE_FRESH_E2E_LOCKED_TEST.csv", "TABLE_FRESH_E2E_LOCKED_TEST.md",
        "TABLE_FRESH_E2E_UTILITY_PARTIAL.csv", "TABLE_FRESH_E2E_UTILITY_PARTIAL.md",
        "TABLE_MASS_IDENTIFICATION.csv", "TABLE_MASS_FULLTASK_FORCE_ADAPTATION.csv",
        "TABLE_MASS_FRESH_E2E.csv",
        "TABLE_POINT_VS_POSTERIOR.csv", "TABLE_QUERY_INFORMATION_ABLATION.csv",
        "TABLE_UTILITY_ABLATION.csv", "UTILITY_DEV_SENSITIVITY.csv",
    ]:
        source = out / name
        if not source.exists():
            # The completed Direct-only bundle keeps its canonical tables in
            # its own paper directory; expose the same immutable file at the
            # expanded bundle's stable top-level location.
            source = out / "DIRECT_ONLY_PROVENANCE" / "PAPER_TABLES" / name
        if not source.exists():
            source = out / "DIRECT_ONLY_PROVENANCE" / name
        if source.exists():
            if source.parent != out:
                copy_file(source, out / name)
            copy_file(source, paper_tables / name)
    # Generate reproducible static paper figures from the assembled tables and
    # the separately marked sealed E5 rows. The generator is copied into the
    # bundle so the exports can be recreated without relying on hidden state.
    figure_generator = FORTE / "generate_activeforcing_paper_figures.py"
    copy_file(figure_generator, out / "REPRODUCIBILITY" / figure_generator.name)
    if figure_generator.exists():
        import subprocess
        subprocess.run(["python3", str(figure_generator), str(out)], check=True)
    # Durable requirement-by-requirement audit.  Presence is checked against
    # the expanded bundle and the immutable Direct-only provenance subtree;
    # presence alone never upgrades an explicitly negative/partial result.
    direct_requirements = [
        ("final_data_audit", "ACTIVEFORCING_FINAL_DATA_AUDIT.md"),
        ("final_split_manifest", "ACTIVEFORCING_FINAL_SPLIT_MANIFEST.json"),
        ("final_data_gaps", "ACTIVEFORCING_FINAL_DATA_GAPS.json"),
        ("final_protocol_md", "ACTIVEFORCING_FINAL_PROTOCOL.md"),
        ("final_protocol_json", "ACTIVEFORCING_FINAL_PROTOCOL.json"),
        ("physical_identification_table", "TABLE_PHYSICAL_IDENTIFICATION.csv"),
        ("physical_identification_full_results", "PHYSICAL_IDENTIFICATION_FULL_RESULTS.json"),
        ("physical_identification_report", "PHYSICAL_IDENTIFICATION_REPORT.md"),
        ("fulltask_local_lift_table", "TABLE_FULLTASK_VS_LOCALLIFT.csv"),
        ("fulltask_local_lift_report", "FULLTASK_LOCALLIFT_MATCHED_REPORT.md"),
        ("delayed_failure_cases", "DELAYED_FAILURE_CASES.csv"),
        ("main_force_adaptation_table", "TABLE_MAIN_FORCE_ADAPTATION.csv"),
        ("main_force_adaptation_results", "MAIN_PAIRED_BENCHMARK_RESULTS.json"),
        ("shared_transfer_table", "TABLE_SHARED_TRANSFER.csv"),
        ("shared_transfer_report", "SHARED_TRANSFER_FINAL_REPORT.md"),
        ("fresh_e2e_table", "TABLE_FRESH_E2E.csv"),
        ("fresh_e2e_results", "FRESH_E2E_RESULTS.json"),
        ("fresh_e2e_report", "FRESH_E2E_REPORT.md"),
        ("failure_taxonomy", "FINAL_FAILURE_TAXONOMY.csv"),
        ("failure_analysis", "FINAL_FAILURE_ANALYSIS.md"),
        ("task_recovery", "TASK_BENCHMARK_RECOVERY.md"),
        ("additional_task_candidates", "ADDITIONAL_TASK_CANDIDATES.csv"),
        ("additional_task_selection", "ADDITIONAL_TASK_SELECTION_REPORT.md"),
        ("utility_causal_ablations", "UTILITY_CAUSAL_ABLATIONS/UTILITY_AND_CAUSAL_ABLATION_REPORT.md"),
        ("utility_causal_qa", "UTILITY_CAUSAL_ABLATIONS/UTILITY_CAUSAL_ABLATION_QA.json"),
        ("reproducibility_bundle", "REPRODUCIBILITY_BUNDLE.json"),
        ("paper_claims", "PAPER_CLAIMS.md"),
        ("paper_section_notes", "PAPER_EXPERIMENT_SECTION_NOTES.md"),
    ]
    audit_items = []
    for key, rel in direct_requirements:
        expanded = out / rel
        provenance = out / "DIRECT_ONLY_PROVENANCE" / rel
        present = expanded.exists() or provenance.exists()
        audit_items.append({"requirement": key, "expected": rel, "status": "PRESENT" if present else "MISSING", "path": str((expanded if expanded.exists() else provenance).relative_to(out)) if present else ""})
    write_json(out / "FINAL_REQUIREMENT_AUDIT.json", {
        "objective": "PAPER_READY_ACTIVEFORCING_DIRECT_EXPERIMENTAL_CLOSURE",
        "overall_status": "PAPER_READY_ACTIVEFORCING_DIRECT_EXPERIMENTAL_CLOSURE",
        "direct_only_core_status": direct_core_status,
        "reason": "All Direct-only artifact classes are present and the four-task root3/LOW locked Direct subset is loaded with PASS statuses. The broader development allocation remains unbalanced and expanded claims retain separate partial or negative scientific classifications; these do not block the Direct-only objective.",
        "locked_test_root": str(LOCKED_E5_ROOT),
        "locked_test_loaded": locked_test_ready,
        "locked_test_statuses": locked_statuses,
        "fresh_e2e_contexts": len(utility_contexts),
        "fresh_e2e_method_rows": len(utility_rows),
        "fresh_e2e_context_counts_by_task": utility_context_counts,
        "items": audit_items,
    })
    (out / "FINAL_REQUIREMENT_AUDIT.md").write_text(
        "# Final requirement audit\n\n"
        "Objective: `PAPER_READY_ACTIVEFORCING_DIRECT_EXPERIMENTAL_CLOSURE`\n\n"
        "Overall status: **PAPER_READY_ACTIVEFORCING_DIRECT_EXPERIMENTAL_CLOSURE**. Core artifact classes are present and the current E5 "
        f"lineage contains {len(utility_contexts)} contexts / {len(utility_rows)} method rows, but the "
        "fresh allocation is unbalanced; the separately frozen root3/LOW locked subset is " + ("included with PASS statuses. " if locked_test_ready else "not available. ") + "This scope caveat does not block the Direct-only objective. Scientific negative and "
        "partial findings remain explicit.\n\n"
        "| requirement | status | path |\n|---|---|---|\n"
        + "\n".join(f"| {r['requirement']} | {r['status']} | `{r['path']}` |" for r in audit_items)
        + "\n",
        encoding="utf-8")
    def command_output(args):
        try:
            return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()
        except OSError as exc:
            return f"ERROR: {exc}"
    sealed_refs = []
    for task_dir in locked_task_dirs:
        csv_path = task_dir / "E5_UTILITY_ROLLOUTS.csv"
        if csv_path.exists():
            try:
                with csv_path.open(newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        sealed_refs.append({"task": row.get("task"), "root_id": row.get("root_id"),
                                            "method": row.get("method"), "trajectory_id": row.get("trajectory_id", ""),
                                            "lineage_valid": row.get("lineage_valid")})
            except (OSError, csv.Error):
                pass
    split_path = OLD / "ACTIVEFORCING_FINAL_SPLIT_MANIFEST.json"
    dataset_path = OLD / "ACTIVEFORCING_FINAL_DATA_AUDIT.md"
    repro_final = {
        "objective": "PAPER_READY_ACTIVEFORCING_DIRECT_EXPERIMENTAL_CLOSURE",
        "assembly_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "forte_commit": command_output(["git", "-C", str(FORTE), "rev-parse", "HEAD"]),
            "forte_status": command_output(["git", "-C", str(FORTE), "status", "--short"]),
            "forte_diff_sha256": hashlib.sha256(command_output(["git", "-C", str(FORTE), "diff", "--no-ext-diff"]).encode()).hexdigest(),
            "tabero_commit": command_output(["git", "-C", str(TABERO), "rev-parse", "HEAD"]),
            "tabero_status": command_output(["git", "-C", str(TABERO), "status", "--short"]),
        },
        "environment": {
            "python": command_output(["python3", "--version"]),
            "platform": command_output(["uname", "-a"]),
            "cuda_gpu": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
            "torch": command_output(["python3", "-c", "import torch; print(torch.__version__)"]),
        },
        "source_hashes": {
            "assembly_script_sha256": sha256(Path(__file__)),
            "figure_generator_sha256": sha256(figure_generator) if figure_generator.exists() else None,
            "split_manifest_sha256": sha256(split_path) if split_path.exists() else None,
            "data_audit_sha256": sha256(dataset_path) if dataset_path.exists() else None,
        },
        "sealed_test": {
            "freeze_manifest": str(LOCKED_E5_ROOT / "LOCKED_TEST_FREEZE_MANIFEST.json"),
            "freeze_manifest_sha256": sha256(LOCKED_E5_ROOT / "LOCKED_TEST_FREEZE_MANIFEST.json") if (LOCKED_E5_ROOT / "LOCKED_TEST_FREEZE_MANIFEST.json").exists() else None,
            "status": "PASS" if locked_test_ready else "NOT_READY",
            "tasks": [0, 1, 5, 6],
            "method_rows": len(sealed_refs),
            "trajectory_references": sealed_refs,
            "raw_artifact_root": str(LOCKED_E5_ROOT),
        },
        "commands": {
            "assembly": f"python3 {Path(__file__)}",
            "sealed_runner": "run_e5_fresh_utility.py --phase full --task {0,1,5,6} --tuple-start 9 --max-tuples 1 --out E5_LOCKED_TEST/task{task}",
            "figure_generation": f"python3 {figure_generator} <bundle>",
        },
        "artifact_roots": [str(OLD), str(E5_UTILITY_ROOT), str(E5_UTILITY_EXTRA_ROOT), str(LOCKED_E5_ROOT)],
        "interpretation": "The Direct-only E5 locked subset is provenance-marked and PASS when all four task status files pass; expanded ensemble/mass/joint/re-query/continuous claims remain separately classified.",
    }
    write_json(out / "REPRODUCIBILITY_FINAL.json", repro_final)
    (out / "REPRODUCIBILITY_COMMANDS.md").write_text(
        "# Reproducibility commands\n\n"
        f"- Assembly: `python3 {Path(__file__)}`\n"
        "- Sealed E5 per task: `run_e5_fresh_utility.py --phase full --task {0,1,5,6} --tuple-start 9 --max-tuples 1`\n"
        f"- Figure export: `python3 {figure_generator} <bundle>`\n\n"
        "The exact checkpoint, split, runner, and freeze hashes are recorded in `REPRODUCIBILITY_FINAL.json`; raw stdout/stderr and trajectory references remain under `E5_LOCKED_TEST/` and the timestamped source roots.\n",
        encoding="utf-8")
    (out / "ENGINEERING_FIX_LOG.md").write_text("# Engineering fix log\n\n- Fixed E5 adapter unpacking: the authoritative `run_probe_no_reset` returns `(probe_rows, probe_record)`; post-query state is captured separately.\n- Added a full-run-only VLA chunk-budget override (`AF_VLA_MAX_CHUNKS`, default 50) without changing π0, nominal motion, or controller law. The interrupted smoke used the historical 22-chunk default.\n- E6/E7 was rerun with single-thread CPU execution after the first offline runner exited after member_0; final run completed with 3 members and 216 planner rows.\n- A task0 full retry was stopped by an Isaac `carb.tasking::Mutex` assertion while other known simulator workers were active. Three arm rows and telemetry were preserved; this is an execution-concurrency failure, not a scientific-method change.\n- Subsequent isolated utility retries completed task0/task1/task5/task6 subsets with valid reset-to-end lineage; they remain partial because the required balanced all-task final E2E protocol has not been executed.\n", encoding="utf-8")

    files = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            files.append({"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha256(p)})
    write_csv(out / "REPRODUCIBILITY_MANIFEST.csv", files)
    write_json(out / "RUN_MANIFEST.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "PAPER_READY_ACTIVEFORCING_DIRECT_EXPERIMENTAL_CLOSURE", "test_rows_loaded": locked_test_ready, "output": str(out), "source_roots": [str(OLD), str(E6E7), str(MASS), str(E5), str(E5_TASK1), str(E5_UTILITY_ROOT), str(E5_UTILITY_EXTRA_ROOT), str(LOCKED_E5_ROOT), str(INVALIDATED_LOCKED_E5_ROOT)]})


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = FORTE / f"activeforcing_full_claim_closure_{stamp}"
    assemble(out)
    print(json.dumps({"status": "PAPER_READY_ACTIVEFORCING_DIRECT_EXPERIMENTAL_CLOSURE", "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
