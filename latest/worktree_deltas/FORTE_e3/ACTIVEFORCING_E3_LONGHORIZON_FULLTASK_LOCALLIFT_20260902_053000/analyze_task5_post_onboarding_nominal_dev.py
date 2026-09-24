#!/usr/bin/env python3
"""Fail-closed five-root nominal DEV gate for the locked 5-demo candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


DYNAMIC_VALIDATOR = Path("/home/exouser/E3_E6_E7_LANES/E3_FULLTASK_VS_LOCALLIFT/validate_e3_coordinator_final_gate_dynamic.py")
E5_ROOT = Path("/home/exouser/FORTE/analysis/results/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_053006")


ROOTS = (7600, 7601, 7602, 7603, 7604)
CELL_RE = re.compile(r"^NOMINAL_DEV_root(7600|7601|7602|7603|7604)_mu0\.6_F8N$")
CONFIG = "pi0_lora_tacfield_e3_task5_5demo_7dpf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("candidate_lock", type=Path)
    ap.add_argument("--e5-root", type=Path, default=E5_ROOT)
    args = ap.parse_args()
    lock = json.loads(args.candidate_lock.read_text())
    lock_sha = sha256(args.candidate_lock)
    if lock.get("status") != "CANDIDATE_LOCKED_FOR_NOMINAL_DEV_NOT_FINAL_FREEZE":
        raise RuntimeError("candidate lock status mismatch")
    if lock.get("selected_step") != 999 or lock.get("config") != CONFIG:
        raise RuntimeError("candidate lock config/step mismatch")
    fingerprint = lock["checkpoint_tree_sha256"]

    errors: list[str] = []
    records: list[dict[str, object]] = []
    seen: set[int] = set()
    for cell in sorted(path for path in args.root.iterdir() if path.is_dir()):
        match = CELL_RE.match(cell.name)
        if not match:
            continue
        root_seed = int(match.group(1))
        if root_seed in seen:
            errors.append(f"duplicate root {root_seed}")
            continue
        seen.add(root_seed)
        episodes = sorted((cell / "logs").glob("*_episodes.csv"))
        steps = sorted((cell / "logs").glob("*_steps.csv"))
        if len(episodes) != 1 or len(steps) != 1:
            errors.append(f"{cell}: expected one episode/step pair")
            continue
        ep_rows, step_rows = rows(episodes[0]), rows(steps[0])
        if len(ep_rows) != 1 or not step_rows:
            errors.append(f"{cell}: incomplete telemetry")
            continue
        ep = ep_rows[0]
        lock_copy = cell / "CANDIDATE_LOCK.json"
        if not lock_copy.is_file() or sha256(lock_copy) != lock_sha:
            errors.append(f"{cell}: copied candidate lock missing or mismatched")
        gate_copy = cell / "COORDINATOR_FINAL_GATE_DYNAMIC.json"
        if not gate_copy.is_file():
            errors.append(f"{cell}: copied second-silent gate missing")
            gate_sha = ""
        else:
            gate_sha = sha256(gate_copy)
            gate_record = json.loads(gate_copy.read_text())
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("e3_dynamic_gate", DYNAMIC_VALIDATOR)
                module = importlib.util.module_from_spec(spec)
                assert spec and spec.loader
                spec.loader.exec_module(module)
                module.common_checks(gate_record, "E3_TASK5_NOMINAL_DEV_CELL", root_seed, args.e5_root)
            except Exception as exc:
                errors.append(f"{cell}: copied dynamic gate evidence mismatch: {exc}")
        expected = {
            "method": "E3_POST_ONBOARDING_NOMINAL_DEV",
            "e3_policy_fingerprint": fingerprint,
            "e3_candidate_lock_sha256": lock_sha,
            "e3_policy_config": CONFIG,
            "e3_checkpoint_step": "999",
            "e3_coordinator_dynamic_gate_sha256": gate_sha,
        }
        mismatch = {key: (ep.get(key), value) for key, value in expected.items() if ep.get(key) != value}
        if mismatch:
            errors.append(f"{cell}: policy provenance mismatch {mismatch}")
        if int(number(ep, "task_id")) != 5:
            errors.append(f"{cell}: task identity mismatch")
        if abs(number(ep, "friction") - 0.6) > 1e-6:
            errors.append(f"{cell}: friction mismatch")
        if abs(number(ep, "peak_predicted_force_slot_N") - 8.0) > 1e-6:
            errors.append(f"{cell}: fixed force telemetry mismatch")
        root_hash = ep.get("root_state_hash", "")
        if len(root_hash) != 64:
            errors.append(f"{cell}: invalid root-state hash")
        telemetry_errors = cell / "logs" / "telemetry_errors.log"
        if telemetry_errors.is_file() and telemetry_errors.stat().st_size:
            errors.append(f"{cell}: nonempty telemetry_errors.log")
        query = max(int(number(row, "contact")) for row in step_rows)
        records.append({
            "root_seed": root_seed,
            "root_state_hash": root_hash,
            "query_state_reached": query,
            "official_fulltask_success": int(number(ep, "official_success")),
            "grasp": int(number(ep, "pick_success")),
            "lift": int(number(ep, "lift_success")),
            "transport": int(number(ep, "transport_success")),
            "placement": int(number(ep, "place_success")),
            "steps": int(number(ep, "steps")),
            "mean_measured_force_N": ep.get("mean_measured_force_N", ""),
            "peak_measured_force_N": ep.get("peak_measured_force_N", ""),
            "candidate_fingerprint": fingerprint,
            "candidate_lock_sha256": lock_sha,
            "coordinator_final_gate_v2_sha256": gate_sha,
            "cell": str(cell),
        })

    missing = sorted(set(ROOTS) - seen)
    extra = sorted(seen - set(ROOTS))
    root_hashes = [str(record["root_state_hash"]) for record in records]
    distinct_roots = len(root_hashes) == 5 and len(set(root_hashes)) == 5
    successes = sum(int(record["official_fulltask_success"]) for record in records)
    query_reaches = sum(int(record["query_state_reached"]) for record in records)
    gate_pass = not errors and not missing and not extra and distinct_roots and successes >= 3 and query_reaches >= 4
    result = {
        "protocol": "E3_POST_ONBOARDING_FULLTASK_VS_LOCALLIFT_V1",
        "status": "NOMINAL_DEV_PASS" if gate_pass else "NOMINAL_DEV_INCOMPLETE_OR_FAIL",
        "gate_pass": gate_pass,
        "expected_roots": list(ROOTS),
        "records": len(records),
        "missing_roots": missing,
        "extra_roots": extra,
        "distinct_root_state_hashes": distinct_roots,
        "official_fulltask_successes": successes,
        "query_state_reaches": query_reaches,
        "pass_rule": "official FullTask success >=3/5 and query-state reach >=4/5",
        "errors": errors,
        "candidate_lock": str(args.candidate_lock.resolve()),
        "candidate_lock_sha256": lock_sha,
        "candidate_fingerprint": fingerprint,
        "records_detail": records,
        "test_used": False,
        "utility_or_fmax_used": False,
        "activeforcing_outcomes_used": False,
        "failure_action": "increase nominal TRAIN demonstrations to 10; do not inspect TEST or force-method outcomes",
    }
    output = args.root / "E3_TASK5_POST_ONBOARDING_NOMINAL_DEV_GATE.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite nominal DEV gate: {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    table = args.root / "E3_TASK5_POST_ONBOARDING_NOMINAL_DEV.csv"
    with table.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]) if records else ["root_seed"])
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({key: value for key, value in result.items() if key != "records_detail"}, sort_keys=True))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
