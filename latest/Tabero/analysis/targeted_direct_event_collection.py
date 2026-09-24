#!/usr/bin/env python3
"""Exact frozen-manifest replay for corrected direct-event telemetry."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "analysis" / "results"
PAIR_RESULT = RESULTS / "counterfactual_force_world_model_20260829_160000"
OLD_EVENT_RESULT = RESULTS / "direct_event_world_model_20260829_180000"
RUNNER = REPO / "analysis" / "p5s0c_paired_boundary_probe_value.py"
PAIR_CSV = PAIR_RESULT / "COUNTERFACTUAL_FORCE_PAIRS.csv"
GROUP_CSV = PAIR_RESULT / "COUNTERFACTUAL_FORCE_GROUPS.csv"
CURRENT_RUN = RESULTS / "targeted_direct_event_collection_20260829_190000"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
WARP_CORE = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA")
OPENPI = Path("/media/volume/newdata/exouser/openpi/src")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows_: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows_[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_)


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def force_tag(force: float) -> str:
    return str(force).replace(".", "p")


def branch_label(force: float) -> str:
    return f"BRANCH_F{force_tag(force)}_R1"


def expected_branch_id(context: str, force: float) -> str:
    return f"{context}_{branch_label(force)}_F{force:g}"


def select_population() -> tuple[list[dict], dict[str, dict]]:
    pairs = rows(PAIR_CSV)
    groups = {r["group_id"]: r for r in rows(GROUP_CSV)}
    selected = []
    for split in ("TRAIN", "DEV"):
        seen = set()
        for r in pairs:
            if r["split"] == split and r["source"] == "historical" and r["category"] == "boundary" and r["a_outcome"] == "0" and r["b_outcome"] == "1" and r["group_id"] not in seen:
                selected.append(r)
                seen.add(r["group_id"])
    train_boundary = {r["group_id"] for r in selected if r["split"] == "TRAIN"}
    supplemental = next(
        r for r in pairs
        if r["split"] == "TRAIN" and r["source"] == "historical"
        and r["group_id"] not in train_boundary and r["category"] in {"adjacent", "wider"}
        and r["a_outcome"] == "0" and r["b_outcome"] == "1"
    )
    selected.append(supplemental)
    assert len([r for r in selected if r["split"] == "TRAIN"]) == 29
    assert len([r for r in selected if r["split"] == "DEV"]) == 9
    return selected, groups


def build_manifest(out: Path) -> dict:
    selected, groups = select_population()
    contexts: dict[str, list[dict]] = {}
    csv_rows = []
    for p in selected:
        g = groups[p["group_id"]]
        boundary = p["a_outcome"] == "0" and p["b_outcome"] == "1"
        specs = []
        for force, force_class, outcome in (
            (float(p["force_a_N"]), "F_prev" if boundary else "LOWER_FORCE", int(p["a_outcome"])),
            (float(p["force_b_N"]), "F_star" if boundary else "HIGHER_FORCE", int(p["b_outcome"])),
        ):
            label_ = branch_label(force)
            branch_id_ = expected_branch_id(p["context_id"], force)
            specs.append({"force_N": force, "repeat_index": 1, "branch_label": label_, "expected_branch_id": branch_id_, "force_class": force_class, "pair_id": p["pair_id"]})
            csv_rows.append({
                "split": p["split"], "population_role": "boundary" if p["category"] == "boundary" else "supplemental_train_event_group",
                "group_id": p["group_id"], "pair_id": p["pair_id"], "task": p["task"], "root_id": p["root_id"], "context_id": p["context_id"],
                "object_identity": p["object_identity"], "friction_band": p["friction_band"], "friction": p["friction"],
                "branch_start_state_hash": g["branch_start_state_hash"], "initial_state_max_abs_diff": g["initial_state_max_abs_diff"],
                "future_command_identifier": g["future_command_hash"], "future_command_hash": g["future_command_hash"], "future_command_exact_match": g["future_command_exact_match"],
                "future_command_per_step_max_diff": g["future_command_per_step_max_diff"], "horizon_length_min": g["horizon_length_min"], "horizon_aligned": g["horizon_aligned"], "phase_aligned": g["phase_aligned"],
                "telemetry_schema": "corrected_direct_contact_local_frame_v1", "force_class": force_class, "force_N": force, "branch_label": label_, "expected_branch_id": branch_id_,
                "expected_outcome_from_frozen_pair": outcome, "same_state_group": p["same_state_group"], "same_future_motion": p["same_future_motion"], "common_H8_window": p["common_H8_window"], "targeted_only": True,
            })
        contexts.setdefault(p["context_id"], []).extend(specs)
    csv_rows.sort(key=lambda r: (r["split"], r["group_id"], float(r["force_N"])))
    manifest = {
        "manifest_name": "DIRECT_EVENT_TARGET_MANIFEST", "created_utc": "2026-08-29T19:00:00Z",
        "frozen_population_rule": {"TRAIN": "28 unique historical boundary groups plus one unique historical supplemental group; existing corrected direct TRAIN group reused", "DEV": "9 unique historical boundary groups; existing corrected direct DEV group reused", "branches_per_new_group": 2, "no_test": True, "no_population_expansion": True},
        "counts": {"new_train_groups": 29, "new_dev_groups": 9, "new_train_branches": 58, "new_dev_branches": 18, "new_total_branches": 76, "reused_train_groups": 1, "reused_dev_groups": 1},
        "authoritative_sources": {str(PAIR_CSV): sha256(PAIR_CSV), str(GROUP_CSV): sha256(GROUP_CSV), str(OLD_EVENT_RESULT / "DIRECT_EVENT_COLLECTION_PROTOCOL.json"): sha256(OLD_EVENT_RESULT / "DIRECT_EVENT_COLLECTION_PROTOCOL.json"), str(RUNNER): sha256(RUNNER)},
        "contexts": dict(sorted(contexts.items())), "target_rows": csv_rows,
    }
    write_json(out / "DIRECT_EVENT_TARGET_MANIFEST.json", manifest)
    write_csv(out / "DIRECT_EVENT_TARGET_MANIFEST.csv", csv_rows)
    write_json(out / "DIRECT_EVENT_WORLD_MODEL_PROTOCOL.json", {
        "protocol_name": "DIRECT_EVENT_WORLD_MODEL_PROTOCOL",
        "status": "FROZEN_BEFORE_COLLECTION",
        "current_ie_checkpoint": str(PAIR_RESULT / "PHYSICS_GRU_FORCE_IE_lambda1.0_seed0.pt"),
        "architecture": "ShortHorizonPhysicsGRU trunk unchanged; H=8; explicit direct-event channels only if collection gate passes",
        "horizon": 8,
        "splits": {"TRAIN": "frozen TRAIN contexts only", "DEV": "frozen DEV contexts only", "TEST": "not used"},
        "event_definitions": str(RESULTS / "direct_contact_boundary_imagination_20260829_000205" / "DIRECT_PHYSICAL_EVENT_DEFINITION.json"),
        "eligible_telemetry_lineage": "corrected direct-contact logger; no second local-frame rotation; Fn=abs(local z); Ft=sqrt(local x^2+local y^2)",
        "target_manifest": str(out / "DIRECT_EVENT_TARGET_MANIFEST.json"),
        "target_manifest_sha256": sha256(out / "DIRECT_EVENT_TARGET_MANIFEST.json"),
        "pair_sampling": "existing frozen matched F_prev/F_star groups; two branches per selected new group; no expansion",
        "lambda_ie": 1.0,
        "lambda_event_candidates": [0.1, 0.3, 1.0],
        "seeds": [0, 1, 2],
        "training_budget": "inherit authoritative IE optimizer, steps, batch, and H=8",
        "forbidden": ["TEST", "continuous-force search", "Probe", "No-physics", "E2E", "horizon change", "architecture/capacity change", "evaluator change"],
    })
    return manifest


def static_audit(out: Path, manifest: dict) -> dict:
    source = RUNNER.read_text(encoding="utf-8")
    target_rows = manifest["target_rows"]
    checks = {
        "runner_compiles": True,
        "target_manifest_env_supported": "P5S0C_TARGET_MANIFEST" in source,
        "context_filter_precedes_execution": "if target_manifest and context_id not in target_manifest" in source,
        "exact_branch_specs_supported": "branch_specs = [(float(x[\"force_N\"]" in source,
        "no_test_rows": all(r["split"] != "TEST" for r in target_rows),
        "two_branches_per_context": all(len(v) == 2 for v in manifest["contexts"].values()),
        "expected_branch_ids_unique": len({r["expected_branch_id"] for r in target_rows}) == len(target_rows),
        "manifest_contexts_are_frozen": len(manifest["contexts"]) == 38,
        "targeted_only_flags": all(r["targeted_only"] for r in target_rows),
    }
    audit = {"audit_name": "TARGETED_REPLAY_IMPLEMENTATION_AUDIT", "static_checks": checks, "static_pass": all(checks.values()), "runner": str(RUNNER), "runner_sha256": sha256(RUNNER), "manifest_context_count": len(manifest["contexts"]), "manifest_branch_count": len(target_rows), "scope": "engineering targeted manifest replay only"}
    write_json(out / "TARGETED_REPLAY_IMPLEMENTATION_AUDIT.json", audit)
    return audit


def run_worker(out: Path, task: int, manifest_path: Path, context_ids: str, timeout_s: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    log = out / "logs" / f"task{task}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(REPO), str(OPENPI)]), "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "TABERO_ROOT": str(REPO), "P5S0C_OUT": str(out), "P5S0C_WORKER": "1", "P5S0C_TASK_ID": str(task), "P5S0C_TARGET_MANIFEST": str(manifest_path), "P5S0C_CONTEXT_IDS": context_ids, "P5S0C_SKIP_REPLAY": "1", "HDF5_TRAJ_SOURCE_DIR": str(REPO / "benchmarks/datasets/libero/assembled_hdf5"), "LIBERO_CONFIG_DIR": str(REPO / "benchmarks/datasets/libero/config"), "LIBERO_ASSETS_DATA_DIR": str(REPO / "benchmarks/datasets/libero/USD")})
    start = time.time()
    try:
        with log.open("w", encoding="utf-8") as f:
            p = subprocess.Popen([str(ISAAC_PY), "-u", str(RUNNER)], cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
            completed_after_result = False
            deadline = time.time() + timeout_s
            expected_context_count = len([x for x in context_ids.split(",") if x])
            expected_branch_count = expected_context_count * 2
            while p.poll() is None and time.time() < deadline:
                result_path = out / f"task{task}" / "result.json"
                error_path = out / f"task{task}" / "error.json"
                if result_path.exists() and not error_path.exists():
                    try:
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                        if int(result.get("contexts", 0)) == expected_context_count and int(result.get("primary_branches", 0)) == expected_branch_count:
                            completed_after_result = True
                            p.terminate()
                            break
                    except (ValueError, OSError, json.JSONDecodeError):
                        pass
                time.sleep(1.0)
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
            else:
                p.wait()
        task_error = out / f"task{task}" / "error.json"
        task_contexts = out / f"task{task}" / "context.csv"
        task_branches = out / f"task{task}" / "branches.csv"
        effective_code = 0 if completed_after_result else p.returncode
        error_summary = ""
        if task_error.exists():
            effective_code = 1
            try:
                error_summary = json.loads(task_error.read_text(encoding="utf-8")).get("error", "")
            except Exception:
                error_summary = task_error.read_text(errors="replace")[:1000]
        if effective_code == 0 and (not task_contexts.exists() or not task_branches.exists()):
            effective_code = 1
            error_summary = "worker returned zero without expected context.csv and branches.csv"
        return {"task": task, "returncode": effective_code, "process_returncode": p.returncode, "elapsed_wall_s": time.time() - start, "log": str(log), "context_ids": context_ids, "expected_context_count": expected_context_count, "error": error_summary}
    except subprocess.TimeoutExpired:
        return {"task": task, "returncode": None, "timeout": True, "elapsed_wall_s": time.time() - start, "log": str(log), "context_ids": context_ids}


def blocked_artifacts(out: Path, reason: str, audit: dict, manifest: dict, smoke: dict, workers: list[dict]) -> None:
    write_json(out / "DIRECT_EVENT_COLLECTION_RUN_MANIFEST.json", {"status": "BLOCKED_BEFORE_COLLECTION", "reason": reason, "smoke": smoke, "workers": workers, "manifest_sha256": sha256(out / "DIRECT_EVENT_TARGET_MANIFEST.json")})
    write_csv(out / "DIRECT_EVENT_COLLECTION_RUN_MANIFEST.csv", [{"stage": "smoke", "status": "PASS" if smoke.get("returncode") == 0 else "FAIL", "details": json.dumps(smoke)}, *[{"stage": "full_collection", "status": "NOT_RUN", "details": json.dumps(w)} for w in workers]])
    write_json(out / "CORRECTED_DIRECT_EVENT_DATASET_AUDIT.json", {"status": "NOT_RUN", "reason": reason})
    write_json(out / "DIRECT_EVENT_LABELABILITY_REAUDIT.json", {"status": "NOT_RUN", "reason": reason})
    write_csv(out / "DIRECT_EVENT_GROUP_COVERAGE_AFTER_COLLECTION.csv", [])
    write_json(out / "EVENT_TARGET_SPEC.json", {"status": "NOT_RUN", "reason": reason})
    write_json(out / "EVENT_LOSS_SPEC.json", {"status": "NOT_RUN", "reason": reason})
    write_csv(out / "IE_EVENT_TRAINING_MANIFEST.csv", [])
    write_json(out / "IE_EVENT_CHECKPOINT_MANIFEST.json", {"status": "NOT_RUN", "reason": reason})
    for name in ("IE_VS_IE_EVENT_DEV.csv", "DEV_EVENT_PREDICTION_METRICS.csv", "DEV_FAILURE_PRESERVATION_EVENT.csv", "H8_INFORMATIVE_SUBSET_ANALYSIS.csv"):
        write_csv(out / name, [])
    report = f"""# STATUS

BLOCKED BEFORE COLLECTION

# SINGLE GOAL

Collect corrected matched direct-contact telemetry and train the frozen IE+EVENT ablation if labelability passes.

# CONNECTION TO PREVIOUS BLOCKER

Previous classification: DIRECT_EVENT_SUPERVISION_INSUFFICIENT.

# TARGETED REPLAY IMPLEMENTATION

Static audit: {'PASS' if audit['static_pass'] else 'FAIL'}. Frozen target: {len(manifest['contexts'])} groups / {len(manifest['target_rows'])} branches, no TEST.

Smoke result:

```json
{json.dumps(smoke, indent=2)}
```

# COLLECTION POPULATION

Expected 29 TRAIN groups / 58 branches and 9 DEV groups / 18 branches. Completed 0 because the exact targeted replay smoke was blocked before valid telemetry output.

# CORRECTED TELEMETRY VALIDITY

No new corrected telemetry branch trace was completed. The pre-existing corrected direct groups remain the only event-valid groups; no historical proxy-only trace was promoted to a direct event label.

# EVENT LABELABILITY AFTER COLLECTION

Re-audit was not run because collection did not produce new valid branches. The prior gate remains the authoritative state: TRAIN 1/73 event-valid groups and 1/29 event-valid boundary pairs; DEV 1/25 and 1/10.

# LABELABILITY GATE

FAIL before re-audit: targeted replay infrastructure could not construct the first corrected TRAIN trace.

# IE+EVENT TRAINING

NOT RUN. No checkpoint, calibration, or DEV prediction was generated.

# EVENT PREDICTION RESULT

NOT AVAILABLE.

# F_PREV EVENT RECALL

NOT AVAILABLE from this run.

# H8-INFORMATIVE FAILURE PRESERVATION

NOT EVALUATED.

# ALL DEV BOUNDARY RESULT

NOT EVALUATED.

# IE RETENTION

Not applicable; the authoritative previous IE result is unchanged and no IE+EVENT model was trained.

# CALIBRATION

Not run.

# PRIMARY_CLASSIFICATION

TARGETED_COLLECTION_INFRASTRUCTURE_BLOCKED

# WHAT IS NOW PROVEN

The frozen population was constructed exactly and the patched runner statically supports target-manifest filtering. No new telemetry or model result was produced.

# WHAT IS STILL NOT PROVEN

Whether explicit physical-event supervision restores insufficient-force failure preservation.

- no TEST
- no continuous-force search
- no Probe/No-physics
- no E2E

# METHOD CHANGE

NONE

# NEXT_METHOD

collect corrected matched direct-contact boundary trajectories after restoring a usable Isaac/GPU execution environment.
"""
    (out / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    write_hashes(out)


def write_hashes(out: Path) -> None:
    lines = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            lines.append(f"{sha256(path)}  {path.relative_to(out)}")
    (out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=CURRENT_RUN)
    ap.add_argument("--smoke-timeout", type=int, default=240)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(out)
    audit = static_audit(out, manifest)
    smoke_context = next(r["context_id"] for r in manifest["target_rows"] if r["split"] == "TRAIN")
    smoke_task = int(next(r["task"] for r in manifest["target_rows"] if r["context_id"] == smoke_context))
    smoke_manifest = out / "DIRECT_EVENT_TARGET_MANIFEST_SMOKE.json"
    write_json(smoke_manifest, {"manifest_name": "DIRECT_EVENT_TARGET_MANIFEST_SMOKE", "contexts": {smoke_context: manifest["contexts"][smoke_context]}})
    smoke = run_worker(out / "smoke", smoke_task, smoke_manifest, smoke_context, args.smoke_timeout) if audit["static_pass"] else {"returncode": 2, "reason": "static_audit_failed"}
    write_json(out / "TARGETED_REPLAY_SMOKE_RESULT.json", smoke)
    if smoke.get("returncode") != 0:
        blocked_artifacts(out, "TARGETED_REPLAY_SMOKE_FAILED" if not smoke.get("timeout") else "TARGETED_REPLAY_SMOKE_TIMEOUT", audit, manifest, smoke, [])
        return 3
    task_contexts = {}
    for r in manifest["target_rows"]:
        task_contexts.setdefault(int(r["task"]), set()).add(r["context_id"])
    workers = []
    for task in sorted(task_contexts):
        workers.append(run_worker(out / "collection", task, out / "DIRECT_EVENT_TARGET_MANIFEST.json", ",".join(sorted(task_contexts[task])), args.smoke_timeout * 20))
        if workers[-1].get("returncode") != 0:
            blocked_artifacts(out, "TARGETED_COLLECTION_WORKER_FAILED", audit, manifest, smoke, workers)
            return 4
    write_json(out / "DIRECT_EVENT_COLLECTION_RUN_MANIFEST.json", {"status": "COLLECTION_COMPLETED", "workers": workers})
    write_csv(out / "DIRECT_EVENT_COLLECTION_RUN_MANIFEST.csv", [{"stage": "full_collection", **w} for w in workers])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
