#!/usr/bin/env python3
"""Pre-outcome freeze for the final ActiveForcing normal/challenge evaluation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "activeforcing_final_experiment_20260901_045000"
TASKS = [0, 1, 5, 6]
PRIMARY_CHALLENGE_TASKS = [0, 5]
FORCES = {
    0: [3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
    1: [4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00],
    5: [3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
    6: [3.00, 3.25, 3.50, 3.75, 4.00],
}
FRICTIONS = [("LOW", .20), ("MID", .50), ("HIGH", 1.00)]
ROOT_SEEDS = {0: list(range(9100, 9108)), 5: list(range(9108, 9116))}
REPEATS = 2


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    definition_path = OUT / "FORCE_CRITICAL_CHALLENGE_DEFINITION.json"
    if definition_path.exists():
        print(json.dumps({"status": "ALREADY_FROZEN", "sha256": sha(definition_path)}, indent=2)); return

    sources = [
        ROOT / "prospective_visual_context_collect.py",
        Path("/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py"),
        ROOT / "run_pooled_predictive_verifier.py",
        ROOT / "pooled_predictive_verifier_20260901_033804/ACTIVEFORCING_POOLED_VERIFIER_FREEZE.json",
        Path("/home/exouser/Tabero/analysis/results/active_friction_imagination_20260828_211106/FRICTION_GRU.pt"),
        Path("/home/exouser/Tabero/analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542/P5S0C_NORMALIZATION.json"),
    ]
    for seed in [0, 1, 2]:
        sources += [
            ROOT / f"joint_mechanism_20260831/pooled_matched/POOLED_BASE_seed{seed}.pt",
            ROOT / f"joint_mechanism_20260831/pooled_matched/POOLED_JOINT_NOVISUAL_seed{seed}.pt",
        ]
    missing = [str(p) for p in sources if not p.exists()]
    if missing: raise RuntimeError("missing frozen source: " + json.dumps(missing))

    definition = {
        "status": "FROZEN_BEFORE_CHALLENGE_COLLECTION_AND_BEFORE_ANY_CHALLENGE_MODEL_QUERY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "name": "Force-Critical Challenge",
        "selection_is_model_independent": True,
        "membership_forbidden_inputs": ["Direct probability", "Direct selected force", "World Model prediction", "Verifier logit/decision", "Joint output", "any method success on challenge"],
        "membership_allowed_inputs": ["real simulator full-task outcomes", "real failure stage", "candidate-force adjacency", "hidden-friction relation", "task/root/object identity"],
        "unit": "friction-conditioned context; repeated branches define each force cell",
        "frontier": {"rho": .8, "definition": "minimum preregistered force whose real repeat success fraction is >=0.8", "unsupported": "NO_REAL_FRONTIER"},
        "types": {
            "TYPE1_RECOVERABLE_NEAR_FRONTIER": {
                "rule": "same task/root/friction has adjacent preregistered F_low,F_high; every valid repeat at F_low fails and every valid repeat at F_high succeeds",
                "minimum_valid_repeats_per_cell": REPEATS,
            },
            "TYPE2_DELAYED_GRIP_FAILURE": {
                "rule": "TYPE1 plus low-force pick_success=1 and lift_success=1; final failure is grip-related after lift; higher adjacent force completes full task",
                "allowed_failure_codes": ["TRANSPORT_SLIP", "ROTATIONAL_SLIP", "PLACEMENT_GRIP_LOSS", "OTHER_GRIP_RELATED_DOWNSTREAM_FAILURE"],
                "excluded": ["VLA semantic failure", "never grasped", "initial lift failure", "infrastructure timeout/error"],
            },
            "TYPE3_HIDDEN_FRICTION_DISCORDANT": {
                "rule": "same task/root/object/visual semantics with only hidden friction changed; two supported real Fstar_0.8 values differ by >= one preregistered grid step",
                "cross_friction_snapshot_claim": "same seeded root/state family, not bit-identical simulator state across friction",
            },
        },
        "failure_taxonomy": ["UNDER_FORCE_IMMEDIATE", "TRANSPORT_SLIP", "ROTATIONAL_SLIP", "PLACEMENT_GRIP_LOSS", "STOCHASTIC_BOUNDARY", "NO_VALID_FORCE", "VLA_NON_FORCE_FAILURE", "OTHER"],
        "failure_mapping": {
            "UNDER_FORCE_IMMEDIATE": "pick_success=0 or lift_success=0 with contact/grasp loss before stable lift",
            "TRANSPORT_SLIP": "pick_success=1,lift_success=1,final failure during transit with grip/contact loss",
            "ROTATIONAL_SLIP": "pick_success=1,lift_success=1,angular-slip signature before final failure",
            "PLACEMENT_GRIP_LOSS": "pick_success=1,lift_success=1,failure during over_basket/place with grip/contact loss",
            "STOCHASTIC_BOUNDARY": "same context/force has mixed real repeat outcomes",
            "VLA_NON_FORCE_FAILURE": "Fmax fails without grip-loss evidence or high-force gate shows semantic/task failure",
            "OTHER": "real execution failure not identified above",
        },
        "challenge_success_gate": {
            "verifier_rescue_rate_strictly_above_one_step": True,
            "under_force_lower_than_Direct": True,
            "collateral_rejection_max": .05,
            "mean_force_strictly_below_FixedMax": True,
            "benefit_independent_roots_min": 2,
            "benefit_tasks_min": 2,
            "strict_search_not_max_fallback": True,
        },
        "metrics": {
            "rescue_rate": "successful verifier executions / Direct-proposal failures having at least one higher successful candidate",
            "missed_rescue": "recoverable Direct failure ending in NO_VALID_FORCE or failed selected force",
            "collateral_escalation": "Direct proposal really succeeds but verifier selects higher force",
            "collateral_rejection": "Direct proposal really succeeds but strict verifier returns NO_VALID_FORCE",
            "selective_escalation_precision": "escalations whose Direct proposal really fails / all verifier escalations",
        },
        "no_mass_challenge": "not preregistered in primary run; current Probe estimates friction only",
        "root_scaling_isolation": "root_scaling_20260831 and all of its TEST/model prediction artifacts are forbidden inputs",
    }
    write_json(definition_path, definition)

    contexts, targets = [], {}
    for task in PRIMARY_CHALLENGE_TASKS:
        for ri, seed in enumerate(ROOT_SEEDS[task]):
            root_id = f"afc_t{task}_root{ri:02d}_s{seed}"
            for band, mu in FRICTIONS:
                cid = f"afc_challenge_t{task}_r{ri:02d}_s{seed}_{band.lower()}_mu{mu:.2f}"
                contexts.append({"context_id": cid, "root_id": root_id, "root_index": ri, "root_seed": seed, "task": task, "split": "CHALLENGE", "friction_band": band, "mu_GT": mu})
                specs = []
                for force in FORCES[task]:
                    for rep in range(1, REPEATS + 1):
                        tag = f"{force:.2f}".replace(".", "p")
                        specs.append({"force_N": force, "repeat_index": rep, "branch_label": f"CHALLENGE_F{tag}_R{rep}"})
                targets[cid] = specs
    write_json(OUT / "CHALLENGE_CONTEXTS.json", contexts)
    write_json(OUT / "CHALLENGE_TARGET_MANIFEST.json", {"status": "FROZEN_PREOUTCOME", "contexts": targets})

    manifest = {
        "status": "COLLECTION_POOL_FROZEN_MEMBERSHIP_PENDING_REAL_SWEEPS",
        "definition": str(definition_path), "definition_sha256": sha(definition_path),
        "primary_tasks": PRIMARY_CHALLENGE_TASKS, "resource_rule": "task0/task5 primary; task1/task6 expansion only after primary QA",
        "roots_per_task": 8, "frictions_per_root": 3, "repeats_per_force": REPEATS,
        "candidate_force_grids_N": {str(k): v for k, v in FORCES.items()},
        "primary_root_seeds": {str(k): v for k, v in ROOT_SEEDS.items()},
        "contexts": len(contexts), "planned_branches": sum(len(v) for v in targets.values()),
        "contexts_path": str(OUT / "CHALLENGE_CONTEXTS.json"), "contexts_sha256": sha(OUT / "CHALLENGE_CONTEXTS.json"),
        "target_path": str(OUT / "CHALLENGE_TARGET_MANIFEST.json"), "target_sha256": sha(OUT / "CHALLENGE_TARGET_MANIFEST.json"),
        "membership": [], "membership_not_yet_computed": True,
        "no_model_queried_during_construction": True, "untouched_TEST_read": False,
    }
    write_json(OUT / "FORCE_CRITICAL_CHALLENGE_MANIFEST.json", manifest)
    write_json(OUT / "FORCE_CRITICAL_CHALLENGE_QA.json", {
        "status": "PENDING_REAL_SWEEP_COLLECTION",
        "definition_frozen": True, "pool_frozen": True, "membership_computed": False,
        "model_outputs_used": False, "untouched_TEST_read": False,
        "required_before_evaluation": ["all planned primary branches atomically committed", "state parity PASS", "probe qualification PASS", "failure taxonomy derivable", "membership built from real outcomes only"],
    })

    protocol = {
        "status": "FROZEN_BEFORE_NEW_CHALLENGE_OUTCOMES",
        "methods": ["pi0-Default", "Tabero-Neutral", "Tabero-Oracle-Language [PRIVILEGED ORACLE LANGUAGE]", "FORTE-Reactive [frozen surrogate]", "Fixed-Max", "ActiveForcing-NoProbe", "ActiveForcing-Direct", "ActiveForcing-Direct-GT", "ActiveForcing-PredictiveVerifier", "ActiveForcing-Verifier-MaxFallback"],
        "method_names": {"main": "ActiveForcing-Direct", "extension": "ActiveForcing-PredictiveVerifier", "fallback": "ActiveForcing-Verifier-MaxFallback"},
        "control": {"Direct_proposal": "frozen expected-utility argmax; C_fail=task Fmax", "verifier": "SUCCESS iff logit>0", "search": "upward-only", "exhaustion": "NO_VALID_FORCE", "fallback_semantics": "execute Fmax; does not call it verifier-safe"},
        "physics_sources": {"NoProbe": [0.30, 0.56, 0.92], "Probe": "frozen FRICTION_GRU point estimate", "GT": "mu_GT oracle"},
        "architecture_tuning": False, "force_grid_tuning": False, "probability_threshold_tuning": False, "verifier_threshold_tuning": False, "probe_tuning": False, "model_tuning": False, "task_split_tuning": False,
        "normal_benchmark": "existing authoritative pooled TRAIN population with fully OOF grouped-root shared models for model-backed rows; new matched real rollouts required for pi0/Tabero/FORTE rows",
        "challenge_evaluation_lock": "no model may query challenge until FORCE_CRITICAL_CHALLENGE_QA.status=PASS_MEMBERSHIP_FROZEN",
        "source_hashes": {str(p): sha(p) for p in sources},
        "definition_sha256": sha(definition_path), "manifest_sha256": sha(OUT / "FORCE_CRITICAL_CHALLENGE_MANIFEST.json"),
        "untouched_TEST_read": False,
    }
    write_json(OUT / "ACTIVEFORCING_FINAL_EXPERIMENT_PROTOCOL.json", protocol)
    print(json.dumps({"status": protocol["status"], "out": str(OUT), "contexts": len(contexts), "branches": manifest["planned_branches"], "definition_sha256": sha(definition_path)}, indent=2))


if __name__ == "__main__": main()
