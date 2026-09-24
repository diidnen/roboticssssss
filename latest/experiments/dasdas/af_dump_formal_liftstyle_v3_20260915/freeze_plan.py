from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
OLD = BASE / "experiments/af_dump_original_restore_20260912"
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(HERE))
V4 = BASE / "experiments/af_dump_maxf8_20260913"
FORMAL = BASE / "experiments/af_dump_formal_single_root_20260914"
LIFT = BASE / "experiments/af_dump_liftstyle_feas_v3_fork_20260914"
V5R1 = Path("/media/volume/dasdas/exouser/af_dump_formal_af_v5r1_20260914")
RUNTIME = V4 / "additional_data_v1/RUNTIME_MANIFEST.json"
RUNTIME_SHA = "01d0289b14b812a4a12455e826a0b213ef4bf4fcdb27717892289b7e137b5034"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError("refusing to overwrite " + str(path))
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    plan = HERE / "plan"
    models = HERE / "models_deploy"
    feas = models / "feasibility"
    plan.mkdir(parents=True, exist_ok=True)
    if any(plan.iterdir()):
        raise RuntimeError("plan already populated")
    feas.mkdir(parents=True, exist_ok=True)
    belief_src = V4 / "models_v4/belief"
    belief_dst = models / "belief"
    if belief_dst.exists() or belief_dst.is_symlink():
        belief_dst.unlink()
    belief_dst.symlink_to(belief_src)

    checkpoints = []
    for seed in (0, 1, 2):
        src = LIFT / "models" / f"member_seed{seed}.pt"
        dest = feas / f"member_seed{seed}.pt"
        shutil.copy2(src, dest)
        ckpt = torch.load(dest, map_location="cpu", weights_only=False)
        checkpoints.append(
            {
                "seed": seed,
                "path": str(dest),
                "sha256": sha(dest),
                "selected_epoch": int(ckpt["best"]["epoch"]),
                "validation_posterior_nll": float(ckpt["best"]["val_nll"]),
            }
        )

    from native_original_motion_features import TASK_BINDING

    source_runtime = {
        str(HERE / "liftstyle_runtime.py"): sha(HERE / "liftstyle_runtime.py"),
        str(HERE / "max_force_utility.py"): sha(HERE / "max_force_utility.py"),
    }
    feas_manifest = {
        "native_task_binding": TASK_BINDING,
        "root_scope": [200002],
        "feature_shape": [8, 64],
        "label_target": "full_task_success_y",
        "calibration": "NONE_RAW",
        "posterior_interface": "CURRENT_MULTITASK58_SIGMA_AWARE_POSITIVE_MIXTURE_V1",
        "sigma_used": True,
        "force_support": [0.25, 8.0],
        "planner_grid_step": 0.25,
        "force_feature_normalization_N": 8.0,
        "utility_normalization_N": 8.0,
        "executed_selector": "argmax_p",
        "training_experiment": str(LIFT),
        "checkpoints": checkpoints,
        "source_hashes": source_runtime,
        "training_protocol_sha256": sha(LIFT / "plan/PROTOCOL.json"),
        "utility_definition": "logged only; executed rule is argmax_p",
        "online_prefix": "12N supplied grasp then dump probe then hold-chunk select",
    }
    write(feas / "FEASIBILITY_MANIFEST.json", feas_manifest)
    write(
        feas / "CHECKPOINT_SELECTION_LOCK.json",
        {
            "checkpoints": checkpoints,
            "test_labels_accessed_before_lock": False,
            "protocol_sha256": sha(LIFT / "plan/PROTOCOL.json"),
            "rule": "three-member v3 lift-style dump feasibility, VAL NLL per seed",
        },
    )
    belief_manifest_sha = sha(belief_dst / "BELIEF_MANIFEST.json")
    feas_manifest_sha = sha(feas / "FEASIBILITY_MANIFEST.json")
    write(
        models / "TRAINING_COMPLETE.json",
        {
            "completed": True,
            "test_groups_executed": 0,
            "belief_manifest_sha256": belief_manifest_sha,
            "feasibility_manifest_sha256": feas_manifest_sha,
            "belief_source": str(belief_src),
            "feasibility_source": str(LIFT / "models"),
            "force_support": [0.25, 8.0],
            "planner_grid_step": 0.25,
            "executed_selector": "argmax_p",
            "reused_nominal_formal": str(FORMAL / "FINAL_RESULTS.json"),
            "finished_utc": now(),
        },
    )

    shutil.copy2(V5R1 / "plan/REUSED_NOMINAL_OUTCOMES.json", plan / "REUSED_NOMINAL_OUTCOMES.json")
    sources = {
        str(HERE / name): sha(HERE / name)
        for name in [
            "EXPERIMENT_CARD.md",
            "audit_formal_results.py",
            "freeze_plan.py",
            "hold_chunk.py",
            "infer_af_context.py",
            "infer_liftstyle.py",
            "liftstyle_runtime.py",
            "max_force_utility.py",
            "run_af_queue.py",
        ]
    }
    protocol = {
        "version": "AF_DUMP_FORMAL_LIFTSTYLE_FEAS_V3",
        "created_utc": now(),
        "task": "dump_bin_bigbin",
        "physical_root": 200002,
        "claim_boundary": (
            "same-root AF-liftstyle v3 argmax_p [0.25,8]@0.25 vs reused Nominal; "
            "12N grasp then pi0 remainder; EU logged not executed"
        ),
        "formal_contexts": 24,
        "formal_frictions": [0.425, 0.575, 0.85],
        "formal_policy_seeds": [80200002, 80200003, 80200004, 80200005, 80200006, 80200007, 80200008, 80200009],
        "methods_executed": ["ActiveForcing"],
        "methods_reused": ["Nominal Frozen VLA"],
        "methods_skipped": ["Fixed-Strong 8N"],
        "selector": "LiftstyleFeasibility v3 argmax_p on [0.25,8] step 0.25",
        "established_grasp_N": 12.0,
        "pi0_remainder_only": True,
        "models_deploy": str(models),
        "runtime_manifest": str(RUNTIME),
        "runtime_manifest_sha256": RUNTIME_SHA,
        "reused_nominal_results": str(FORMAL / "FINAL_RESULTS.json"),
        "source_hashes": sources,
    }
    write(plan / "PROTOCOL.json", protocol)
    protocol_sha = sha(plan / "PROTOCOL.json")
    common = {
        "task": "dump_bin_bigbin",
        "root": 200002,
        "methods": ["ActiveForcing"],
        "models": str(models),
        "formal_protocol_path": str(plan / "PROTOCOL.json"),
        "formal_protocol_sha256": protocol_sha,
        "runtime_manifest_path": str(RUNTIME),
        "runtime_manifest_sha256": RUNTIME_SHA,
    }
    smoke = {
        **common,
        "id": "smoke_liftstyle_v3_mu0.575_root200002_ps40200002",
        "split": "SMOKE",
        "friction": 0.575,
        "policy_seed": 40200002,
        "excluded_from_main_analysis": True,
    }
    contexts = []
    for seed in protocol["formal_policy_seeds"]:
        for mu in protocol["formal_frictions"]:
            reused = f"formal_mu{mu:.3f}_root200002_ps{seed}"
            contexts.append(
                {
                    **common,
                    "id": f"liftstyle_v3_mu{mu:.3f}_root200002_ps{seed}",
                    "split": "FORMAL_LIFTSTYLE_V3",
                    "friction": mu,
                    "policy_seed": seed,
                    "reused_nominal_context_id": reused,
                }
            )
    write(plan / "SMOKE_CONTEXT.json", smoke)
    write(plan / "FORMAL_CONTEXTS.json", contexts)
    write(
        plan / "FREEZE_LOCK.json",
        {
            "protocol_sha256": protocol_sha,
            "smoke_context_sha256": sha(plan / "SMOKE_CONTEXT.json"),
            "formal_contexts_sha256": sha(plan / "FORMAL_CONTEXTS.json"),
            "reused_nominal_sha256": sha(plan / "REUSED_NOMINAL_OUTCOMES.json"),
        },
    )
    print(
        json.dumps(
            {
                "contexts": len(contexts),
                "belief_manifest_sha256": belief_manifest_sha,
                "feasibility_manifest_sha256": feas_manifest_sha,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
