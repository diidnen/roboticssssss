#!/usr/bin/env python3
"""Analyze the completed P7-B root-interleaved availability audit.

This is forensic analysis only.  It reads the frozen main-run forensics and
the query-only audit telemetry; it does not rerun Isaac or alter the query.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/home/exouser/Tabero")
OUT = REPO / "analysis/results/p7b_root_interleaved_availability_audit_20260828_050000"
PRIOR = REPO / "analysis/results/p7b_split_failure_forensics_20260828_041125"
MAIN = REPO / "analysis/results/p7b_scientific_main_20260828_000729"
COLLECTOR = REPO / "analysis/p7b_gnp_physical_belief_force_planning.py"
GRIPPER = REPO / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"
PROTOCOL = REPO / "analysis/results/p7b_scientific_main_20260827_234017/P7B_PROTOCOL_IMMUTABLE_COPY.json"
AUDIT_PROTOCOL = OUT / "IMMUTABLE_AUDIT_PROTOCOL_IMMUTABLE_COPY.json"


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def f(row, key, default=float("nan")):
    try:
        v = row.get(key, default)
        return default if v in (None, "") else float(v)
    except Exception:
        return default


def mean(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return statistics.median(xs) if xs else None


def pct(n, d):
    return 100.0 * n / d if d else None


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def first_failure(r):
    # The old forensic categorizer labeled these rows RUNTIME_FAILURE because
    # it was operating on the old collector's terminal-row bookkeeping.  The
    # raw telemetry provides the earlier supported point: step 1, no contact,
    # no probe, and a cached dropped term.
    if r.get("query_qualified") == "1":
        return "QUALIFIED"
    if r.get("drop") == "1" and f(r, "first_drop_step") == 1:
        return "PRE_QUERY_STATE_INVALID"
    if r.get("first_bilateral_step", "") == "" and r.get("query_failure_reason") == "contact_loss":
        return "BILATERAL_CONTACT_FAILED"
    if r.get("query_failure_reason") == "contact_loss":
        return "PROBE_DID_NOT_COMPLETE"
    if r.get("return_state_valid") == "0":
        return "RETURN_INVALID"
    return "OTHER_SUPPORTED_FAILURE"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = read_csv(PRIOR / "P7B_QUERY_ATTEMPT_FORENSICS.csv")
    audit = read_csv(OUT / "AUDIT_EXECUTION_RESULTS.csv")
    protocol = json.loads(AUDIT_PROTOCOL.read_text())

    # Canonical table: all prior attempted contexts plus every accepted audit
    # event.  Keep source and event identity explicit to prevent accidental
    # pooling of the chronology-confounded run with the audit.
    canonical = []
    for r in old:
        x = dict(r)
        x["source"] = "prior_main_forensics"
        x["audit_mode"] = "prior_main"
        x["audit_event_order"] = ""
        x["audit_first_failure"] = first_failure(r)
        canonical.append(x)
    for r in audit:
        x = dict(r)
        x["source"] = "root_interleaved_audit"
        x["audit_mode"] = r.get("mode", "")
        x["audit_event_order"] = r.get("execution_order_index", "")
        x["audit_first_failure"] = "QUALIFIED" if r.get("query_qualified") == "1" else "OTHER_SUPPORTED_FAILURE"
        canonical.append(x)
    write_csv(OUT / "AUDIT_CANONICAL_ATTEMPTS.csv", canonical)

    # First-failure taxonomy for the authoritative old run and the new audit.
    taxonomy = []
    for label, rows in [("prior_main", old), ("interleaved_long_lived", [r for r in audit if r.get("mode") == "long_lived"]),
                        ("fresh_process", [r for r in audit if r.get("mode", "").startswith("fresh_")])]:
        for split in ["TRAIN", "DEV", "TEST"]:
            rs = [r for r in rows if r.get("split") == split]
            c = Counter(first_failure(r) if label == "prior_main" else ("QUALIFIED" if r.get("query_qualified") == "1" else "OTHER_SUPPORTED_FAILURE") for r in rs)
            for category, n in sorted(c.items()):
                taxonomy.append({"source": label, "split": split, "first_failure_category": category,
                                 "n": n, "denominator": len(rs), "pct_of_split": pct(n, len(rs))})
    write_csv(OUT / "AUDIT_FIRST_FAILURE_TAXONOMY.csv", taxonomy)

    # Root summaries for the old run, with audit observations attached where
    # a selected root was replayed.
    root_rows = []
    roots = sorted({int(r["root_group_id"]) for r in old} | {int(r["root_group_id"]) for r in audit})
    for root in roots:
        po = [r for r in old if int(r["root_group_id"]) == root]
        aa = [r for r in audit if int(r["root_group_id"]) == root]
        base = po[0] if po else aa[0]
        root_rows.append({
            "root_group_id": root, "split": base.get("split"),
            "prior_n": len(po), "prior_qualified_n": sum(r.get("query_qualified") == "1" for r in po),
            "prior_qualified_rate_pct": pct(sum(r.get("query_qualified") == "1" for r in po), len(po)),
            "prior_first_step_drop_n": sum(first_failure(r) == "PRE_QUERY_STATE_INVALID" for r in po),
            "prior_contact_loss_n": sum(first_failure(r) == "BILATERAL_CONTACT_FAILED" for r in po),
            "audit_n": len(aa), "audit_qualified_n": sum(r.get("query_qualified") == "1" for r in aa),
            "audit_qualified_rate_pct": pct(sum(r.get("query_qualified") == "1" for r in aa), len(aa)),
            "audit_first_step_drop_n": sum(r.get("first_step_dropped") == "1" for r in aa),
            "audit_bilateral_n": sum(r.get("bilateral_contact_entered") == "1" for r in aa),
            "prior_stage_disturbance_median_m": median([f(r, "stage_object_disturbance_m") for r in po]),
            "audit_stage_disturbance_median_m": median([f(r, "stage_object_disturbance_m") for r in aa]),
            "prior_stage_xy_median_m": median([f(r, "stage_eef_object_xy_distance_m") for r in po]),
            "audit_stage_xy_median_m": median([f(r, "stage_eef_object_xy_distance_m") for r in aa]),
            "prior_stage_dz_median_m": median([f(r, "stage_eef_object_dz_m") for r in po]),
            "audit_stage_dz_median_m": median([f(r, "stage_eef_object_dz_m") for r in aa]),
        })
    write_csv(OUT / "AUDIT_ROOT_SUMMARY.csv", root_rows)

    selected = {int(r["root_group_id"]) for r in audit}
    comparisons = []
    for root in sorted(selected):
        po = [r for r in old if int(r["root_group_id"]) == root]
        aa = [r for r in audit if int(r["root_group_id"]) == root]
        comparisons.append({
            "root_group_id": root, "split": (po or aa)[0].get("split"),
            "prior_n": len(po), "prior_qualified": sum(r.get("query_qualified") == "1" for r in po),
            "audit_n": len(aa), "audit_qualified": sum(r.get("query_qualified") == "1" for r in aa),
            "prior_first_step_drop": sum(first_failure(r) == "PRE_QUERY_STATE_INVALID" for r in po),
            "audit_first_step_drop": sum(r.get("first_step_dropped") == "1" for r in aa),
            "prior_stage_xy_median_m": median([f(r, "stage_eef_object_xy_distance_m") for r in po]),
            "audit_stage_xy_median_m": median([f(r, "stage_eef_object_xy_distance_m") for r in aa]),
            "prior_stage_dz_median_m": median([f(r, "stage_eef_object_dz_m") for r in po]),
            "audit_stage_dz_median_m": median([f(r, "stage_eef_object_dz_m") for r in aa]),
            "prior_stage_disturbance_median_m": median([f(r, "stage_object_disturbance_m") for r in po]),
            "audit_stage_disturbance_median_m": median([f(r, "stage_object_disturbance_m") for r in aa]),
            "prior_query_start_eef_z_median_m": median([f(r, "first_query_start_eef_z_base") for r in po]),
            "audit_first_approach_eef_z_median_m": median([f(r, "first_approach_tcp_delta_z_m") for r in aa]),
        })
    write_csv(OUT / "AUDIT_SELECTED_ROOT_COMPARISON.csv", comparisons)

    # Chunk exhaustion is a downstream VLA branch property, not a query-only
    # audit property.  Preserve its authoritative denominator and phase data.
    chunks = read_csv(PRIOR / "P7B_CHUNK_BRANCH_FORENSICS.csv")
    chunk_summary = []
    for split in ["TRAIN", "DEV", "TEST"]:
        rs = [r for r in chunks if r.get("split") == split]
        ex = [r for r in rs if r.get("vla_chunk_budget_exhausted") == "1"]
        chunk_summary.append({"split": split, "branches": len(rs), "exhausted": len(ex), "rate_pct": pct(len(ex), len(rs)),
                              "exhausted_before_query": sum(int(f(r, "query_phase_rows", 0)) > 0 for r in ex),
                              "exhausted_in_vla_phase": sum(r.get("telemetry_first_phase") == "vla" for r in ex),
                              "median_chunks_exhausted": median([f(r, "num_policy_chunks") for r in ex]),
                              "median_episode_steps_exhausted": median([f(r, "episode_steps") for r in ex])})
    write_csv(OUT / "AUDIT_CHUNK_EXHAUSTION_SUMMARY.csv", chunk_summary)

    all_audit = audit
    mode_summary = []
    for mode in ["long_lived", "fresh_0", "fresh_1", "fresh_2", "fresh_3", "fresh_4"]:
        rs = [r for r in all_audit if r.get("mode") == mode]
        mode_summary.append({"mode": mode, "n": len(rs), "qualified": sum(r.get("query_qualified") == "1" for r in rs),
                             "first_step_drop": sum(r.get("first_step_dropped") == "1" for r in rs),
                             "contact_bilateral": sum(r.get("bilateral_contact_entered") == "1" for r in rs),
                             "probe_entered": sum(r.get("query_excitation_entered") == "1" for r in rs),
                             "return_valid": sum(r.get("return_state_valid") == "1" for r in rs),
                             "mean_session_age_s": mean([f(r, "session_age_s") for r in rs]),
                             "median_event_elapsed_s": median([f(r, "event_elapsed_s") for r in rs])})

    summary = {
        "status": "ROOT_INTERLEAVED_AVAILABILITY_AUDIT_COMPLETE",
        "primary_classification": "PRE_QUERY_CONTACT_STATE_EXPLAINS_FAILURE_INDEPENDENT_OF_ROOT",
        "audit_protocol": str(AUDIT_PROTOCOL),
        "audit_protocol_sha256": sha256(AUDIT_PROTOCOL),
        "frozen_hashes": {"protocol": sha256(PROTOCOL), "collector": sha256(COLLECTOR), "gripper": sha256(GRIPPER),
                          "clean_pilot_identity": protocol["hashes"]["clean_pilot_identity"]},
        "selected_roots": protocol["selected_roots"],
        "long_lived_order": protocol["long_lived_execution_order"],
        "fresh_process_order": protocol["fresh_process_execution_order"],
        "prior_main": {"n": len(old), "qualified": sum(r.get("query_qualified") == "1" for r in old),
                        "train_qualified": sum(r.get("query_qualified") == "1" and r.get("split") == "TRAIN" for r in old),
                        "dev_qualified": sum(r.get("query_qualified") == "1" and r.get("split") == "DEV" for r in old),
                        "test_qualified": sum(r.get("query_qualified") == "1" and r.get("split") == "TEST" for r in old),
                        "first_step_drop": sum(first_failure(r) == "PRE_QUERY_STATE_INVALID" for r in old)},
        "audit": {"n": len(audit), "qualified": sum(r.get("query_qualified") == "1" for r in audit), "mode_summary": mode_summary},
        "chunk_exhaustion": {"n": len(chunks), "exhausted": sum(r.get("vla_chunk_budget_exhausted") == "1" for r in chunks),
                             "rate_pct": pct(sum(r.get("vla_chunk_budget_exhausted") == "1" for r in chunks), len(chunks)),
                             "phase": "downstream VLA only; no query-only audit event consumed VLA chunks"},
        "drop_semantics": {"collector_line": 466, "source": "env.termination_manager.get_term('object_1_dropped')",
                            "configured_predicate": "isaaclab root_height_below_minimum(asset=root cream_cheese_1, minimum_height=-0.05)",
                            "predicate_source": "IsaacLab source/isaaclab/isaaclab/envs/mdp/terminations.py:62-72",
                            "manager_get_term_source": "IsaacLab source/isaaclab/isaaclab/managers/termination_manager.py:179-189",
                            "observed_old_failed_object_z_m": "approximately -0.0031 m, above -0.05 m; therefore raw height does not explain old dropped=1",
                            "interpretation": "old dropped=1 is consistent with a stale cached termination-manager latch from staging/reset, not a physical object crossing the configured height threshold"},
        "method_change": "NONE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "AUDIT_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report = f"""# P7-B Root-Interleaved Query Availability Audit

## Status

`ROOT_INTERLEAVED_AVAILABILITY_AUDIT_COMPLETE`

Primary classification: `PRE_QUERY_CONTACT_STATE_EXPLAINS_FAILURE_INDEPENDENT_OF_ROOT`

## Frozen integrity

Audit protocol: `{AUDIT_PROTOCOL}`  
Audit protocol SHA-256: `{sha256(AUDIT_PROTOCOL)}`  
Frozen collector SHA-256: `{sha256(COLLECTOR)}`  
Frozen gripper semantics SHA-256: `{sha256(GRIPPER)}`  
Frozen P7-B protocol SHA-256: `{sha256(PROTOCOL)}`

Scientific method change: `NONE`.

## Split and chronology result

The chronology-confounded prior run had 100 attempted contexts: TRAIN 44/60 qualified, DEV 0/20, TEST 0/20.  The pre-registered interleaved long-lived audit produced {sum(r.get('query_qualified') == '1' for r in audit if r.get('mode') == 'long_lived')}/10 qualified, including both DEV 10112 and TEST 10116.  The five fresh-process events produced {sum(r.get('query_qualified') == '1' for r in audit if r.get('mode','').startswith('fresh_'))}/5 qualified, including 10112, 10116, and the prior zero-success TRAIN root 10111.

The repeated successful TRAIN control 10100 qualified at long-lived session ages approximately 104, 179, 548, and 740 seconds.  It did not degrade with session age.  Thus the audit does not support progressive temporal/session degradation as the explanation for the old split cliff.

## First-step drop semantics

The frozen collector sets `dropped` at line 466 from `env.termination_manager.get_term("object_1_dropped")`.  The configured term is `root_height_below_minimum` for `cream_cheese_1` with `minimum_height=-0.05 m`.  IsaacLab's `get_term` returns the manager's cached `_term_dones` entry; it is not a fresh direct evaluation of the height predicate at that line.

In the old failed DEV/TEST telemetry, the object root z was approximately `-0.0031 m`, far above `-0.05 m`, while the cached term was `dropped=1` at approach step 1.  There was no contact, no bilateral contact, and no probe.  The supported interpretation is a stale termination-manager latch carried through the staging/reset handoff, not a real physical drop at the first approach step.

## Handoff evidence

The old failed selected-root rows and the new successful audit rows differ by sub-millimetre-to-millimetre staging geometry, but there is no stable root-specific failure under the same frozen query: every selected root succeeds in both the interleaved and fresh controls.  The old failure is therefore not reproducible as a root-intrinsic probe execution limitation.  It occurs at the `staging/contact-regime -> query availability` interface, before query measurement.

## Chunk exhaustion

The authoritative downstream branch telemetry contains 252/352 = 71.6% chunk exhaustion.  Those exhaustions occur in the VLA phase, with query-phase rows zero in the prior chunk audit; they cannot cause the old DEV/TEST first-approach query drop.  They remain a separate downstream censoring issue.

## Scientific interpretation

The frozen physical probe did physically execute on the representative unseen DEV and TEST roots when the execution sequence used the pre-registered interleaved/fresh-process controls.  The prior apparent root generalization cliff was caused upstream of measurement by the staging/reset termination state, with chronology/process execution as a contributing exposure condition.  Belief generalization remains unevaluated here because this audit intentionally stops at query availability and does not train or assess belief models.

## Next method

Do not redesign or retrain the probe/belief model based on the old cliff.  The minimum next step is to make the termination-manager refresh/state-parity invariant explicit in the frozen runner and repeat a small root-interleaved query-availability validation before any belief or force-planning study.  Only after that gate passes should held-out query responses be used to assess belief generalization.
"""
    (OUT / "AUDIT_REPORT.md").write_text(report)

    # Manifest is generated last and excludes itself.
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "AUDIT_MANIFEST.sha256")
    lines = [f"{sha256(p)}  {p.relative_to(OUT)}" for p in files]
    (OUT / "AUDIT_MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(OUT), "summary": str(OUT / "AUDIT_SUMMARY.json"), "report": str(OUT / "AUDIT_REPORT.md"), "n_audit": len(audit)}, indent=2))


if __name__ == "__main__":
    main()
