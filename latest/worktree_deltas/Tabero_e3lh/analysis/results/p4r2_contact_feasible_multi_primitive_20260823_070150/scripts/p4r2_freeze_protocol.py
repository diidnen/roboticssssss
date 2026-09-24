#!/usr/bin/env python3
"""Freeze P4-R2 protocol before main evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(os.environ.get("P4R2_OUT", Path(__file__).resolve().parents[1])).resolve()
REPO = Path("/home/exouser/Tabero")
TASKS = [0, 1, 2, 5, 6]
FRICTIONS = [0.2, 0.5, 1.0]
DEV_OFFSET = int(os.environ.get("P4R2_DEV_SEED_OFFSET", "1000"))
DEV_N = int(os.environ.get("P4R2_DEV_N_SEEDS", "5"))
MAIN_OFFSET = int(os.environ.get("P4R2_MAIN_SEED_OFFSET", "2000"))
MAIN_N = int(os.environ.get("P4R2_MAIN_N_SEEDS", "20"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def seed_manifest(offset: int, n: int, split: str):
    rows = []
    for task in TASKS:
        for seed in range(offset, offset + n):
            for friction in FRICTIONS:
                rows.append({"split": split, "task": task, "seed": seed, "friction": friction})
    return rows


thresholds = {
    "base_force_N": 3.0,
    "preload_step_N": 0.5,
    "preload_cap_N": 4.5,
    "s_step_mm": 0.1,
    "l_step_mm": 0.1,
    "s_max_displacement_mm": 2.0,
    "l_max_displacement_mm": 1.2,
    "rho_cap": 0.08,
    "relative_normal_alpha": 0.55,
    "contact_force_eps_N": 0.20,
    "finger_force_min_N": 0.15,
    "marker_norm_stop": 0.12,
    "major_displacement_m": 0.01,
    "major_rotation_rad": 0.35,
    "selector_s_leakage_max": 0.10,
    "selector_s_alignment_min": 0.10,
    "selector_s_null_condition_max": 10000000.0,
    "selector_preload_balance_max": 0.65,
    "selector_preload_fn_min_factor": 0.65,
    "selector_s_preprobe_tangential_force_min_N": 0.01,
    "s_info_rho_impulse_min": 0.0025,
    "s_info_marker_tangential_min": 0.015,
    "l_info_load_impulse_min": 0.015,
    "l_info_force_delta_min_N": 0.08,
}

selector_rule = {
    "inputs_allowed": [
        "bilateral fingertip frame normals",
        "pre-probe bilateral contact state",
        "pre-probe normal force",
        "pre-probe force imbalance",
        "frozen nominal object-to-basket downstream direction",
        "gravity direction",
        "tactile marker baseline as contact proxy",
    ],
    "inputs_prohibited": ["task id", "object id", "friction", "mass", "main-evaluation outcome labels"],
    "rule": [
        "Compute Primitive S direction by projecting the nominal downstream direction into the regularized null space of left and right fingertip normals.",
        "Select S if pre-probe contact is bilateral, preload is strong enough, preload tangential contact force is at least 0.01N, force balance is within the global cap, actual projected normal leakage is <= 0.10, and downstream alignment is >= 0.10. A rank-1 opposing-normal pair is accepted when the two normals are mutually consistent because it defines the expected bilateral tangent plane.",
        "Otherwise select L if pre-probe contact is bilateral and normal force exceeds the global contact floor.",
        "Otherwise output NO_FEASIBLE_PROBE.",
    ],
    "thresholds": thresholds,
    "task_specific_lookup": False,
    "gt_hidden_physics_input": False,
}

primitive_definitions = {
    "METHOD_CHANGE": "CONTACT_FEASIBLE_MULTI_PRIMITIVE_PROBE_COMPOSER_ONLY",
    "Primitive_S": {
        "name": "bilateral-contact-feasible shear",
        "direction": "regularized bilateral normal null-space projection of frozen nominal downstream transport direction",
        "preload": "start 3N, increase by 0.5N only as needed for bilateral contact, cap 4.5N",
        "step_mm": thresholds["s_step_mm"],
        "max_displacement_mm": thresholds["s_max_displacement_mm"],
        "safety_stops": ["hard contact loss", "relative normal drop", "Ft/Fn cap", "marker budget", "2mm displacement cap", "return-to-start"],
        "informative_floor": "rho_impulse >= 0.0025 and marker_tangential_peak >= 0.015, each also above preload noise proxy",
    },
    "Primitive_L": {
        "name": "incipient support-transfer probe",
        "direction": "gravity-opposing support-transfer direction",
        "preload": "same global preload rule as S",
        "step_mm": thresholds["l_step_mm"],
        "max_displacement_mm": thresholds["l_max_displacement_mm"],
        "safety_stops": ["hard contact loss", "relative normal drop", "force imbalance cap", "marker budget", "support-load budget", "object displacement/rotation caps", "return-to-start"],
        "informative_floor": "support_load_impulse >= 0.015 and support_force_delta_peak >= 0.08N, each also above preload noise proxy",
    },
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    freeze_time = datetime.now(timezone.utc).isoformat()
    main_files = list(OUT.glob("P4R2_main_*_PROBE.csv"))
    no_main_observed = len(main_files) == 0
    protocol = {
        "name": "P4-R2 Contact-Feasible Multi-Primitive Probe Composer",
        "freeze_timestamp_utc": freeze_time,
        "method_change": "CONTACT_FEASIBLE_MULTI_PRIMITIVE_PROBE_COMPOSER_ONLY",
        "tasks": TASKS,
        "frictions_evaluation_only": FRICTIONS,
        "candidate_forces_unchanged_N": [3, 4, 5, 6, 8],
        "development_split": {"seed_offset": DEV_OFFSET, "n_seeds": DEV_N, "conditions": len(seed_manifest(DEV_OFFSET, DEV_N, "development"))},
        "main_split": {"seed_offset": MAIN_OFFSET, "n_seeds": MAIN_N, "conditions": len(seed_manifest(MAIN_OFFSET, MAIN_N, "main"))},
        "paired_primitives": ["S", "L"],
        "thresholds": thresholds,
        "selector_rule": selector_rule,
        "primitive_definitions": primitive_definitions,
        "success_gates": {
            "paired_library_coverage_per_task_min": 0.95,
            "selector_selected_qualified_per_task_min": 0.95,
            "selector_selected_qualified_aggregate_min": 0.95,
            "drop_rate_per_task_max": 0.05,
            "selector_regret_aggregate_max": 0.05,
        },
        "no_main_evaluation_outcome_observed_before_protocol_freeze": no_main_observed,
        "main_files_existing_at_freeze": [p.name for p in main_files],
        "task_specific_lookup": False,
        "gt_hidden_physics_input": False,
    }
    protocol_blob = json.dumps(protocol, indent=2, sort_keys=True)
    protocol_hash = sha256_text(protocol_blob)
    write_json(OUT / "P4R2_PROTOCOL.json", protocol)
    (OUT / "P4R2_PROTOCOL_HASH.txt").write_text(protocol_hash + "\n", encoding="utf-8")
    write_json(OUT / "P4R2_SELECTOR_RULE.json", selector_rule)
    write_json(OUT / "P4R2_PRIMITIVE_DEFINITIONS.json", primitive_definitions)
    (OUT / "P4R2_SELECTOR_RULE.md").write_text(
        "# P4-R2 Selector Rule\n\n"
        "Inputs are limited to pre-probe bilateral contact geometry, measured preload/contact forces, proprioception, tactile baseline proxies, the frozen nominal downstream direction, and gravity.\n\n"
        "1. Compute Primitive S as the closest unit direction to downstream transport in the regularized bilateral tangent/null space of the two fingertip normals.\n"
        "2. Select `S` only when all global feasibility thresholds pass: bilateral contact, preload force, preload balance, null-space condition, normal leakage, and downstream alignment.\n"
        "3. Otherwise select `L` if bilateral contact and normal force are present.\n"
        "4. Otherwise output `NO_FEASIBLE_PROBE`.\n\n"
        "No task ID, object name, friction, mass, hidden physics, or main-evaluation outcome is an input.\n",
        encoding="utf-8",
    )
    write_json(OUT / "P4R2_DEVELOPMENT_SEEDS.json", seed_manifest(DEV_OFFSET, DEV_N, "development"))
    write_json(OUT / "P4R2_MAIN_SEEDS.json", seed_manifest(MAIN_OFFSET, MAIN_N, "main"))
    write_json(OUT / "P4R2_RESET_PARITY.json", {"paired_reset_design": "same task, seed, friction separately reset for S and L", "parity_check": "deterministic seed/friction manifest; simulator state hash unavailable in existing P4 API", "strongest_available_check": "paired trial_id manifest and reset seed equality"})

    code_files = [OUT / "scripts" / "p4r2_collect_probe.py", OUT / "scripts" / "p4r2_analyze.py", OUT / "scripts" / "p4r2_freeze_protocol.py", OUT / "scripts" / "run_p4r2_collect_all.sh"]
    code_hashes = {p.name: sha256_file(p) for p in code_files if p.exists()}
    write_json(OUT / "P4R2_CODE_HASH.txt", {"git_commit": git_commit(), "file_sha256": code_hashes})
    env = {"timestamp_utc": freeze_time, "repo": str(REPO), "result_dir": str(OUT), "git_commit": git_commit(), "python": platform.python_version(), "platform": platform.platform(), "method_change": "CONTACT_FEASIBLE_MULTI_PRIMITIVE_PROBE_COMPOSER_ONLY"}
    try:
        env["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total", "--format=csv,noheader"], text=True).strip()
    except Exception as exc:
        env["nvidia_smi_error"] = repr(exc)
    write_json(OUT / "P4R2_ENVIRONMENT.json", env)
    print(json.dumps({"protocol_hash": protocol_hash, "freeze_timestamp_utc": freeze_time, "no_main_observed": no_main_observed}, indent=2))


if __name__ == "__main__":
    main()
