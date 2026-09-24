#!/usr/bin/env python3
"""Zero-physics audit and prospective freeze for secondary baselines."""
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path


STAGE = Path("/home/exouser/FORTE/secondary_staging_v1/OUTPUT")
FINAL = Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_secondary_baselines_v1")
SOURCE = Path("/home/exouser/FORTE/secondary_staging_v1/SOURCE_SNAPSHOT")
PARENT_PATH = Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_ablation_confirmatory_v1/FINAL_ABLATION_RUNTIME_MANIFEST.json")
PARENT_PLAN = Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_ablation_confirmatory_v1/FINAL_ABLATION_CONTEXT_PLAN.json")
OLD_TABERO_CLIENT = Path("/media/volume/newdata/exouser/Tabero_e3lh/analysis/results/b5_tabero_neutral_20260822_040652/scripts/b5_tabero_neutral_client.py")
TABERO_AUDIT = Path("/home/exouser/FORTE/hidden_friction_baseline_20260831/FRICTION_BASELINE_IMPLEMENTATION_AUDIT.md")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(name, value):
    path = STAGE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def main():
    if STAGE.exists():
        raise RuntimeError("Staging output already exists")
    STAGE.mkdir(parents=True)
    (STAGE / "SOURCE_SNAPSHOT").mkdir()
    for path in SOURCE.glob("*.py"):
        (STAGE / "SOURCE_SNAPSHOT" / path.name).write_bytes(path.read_bytes())

    parent = json.loads(PARENT_PATH.read_text())
    checkpoint = parent["VLA_CHECKPOINT"]
    parent_record = {
        "version": "SECONDARY_BASELINE_RUNTIME_PARENT_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_manifest": str(PARENT_PATH),
        "parent_manifest_sha256": sha(PARENT_PATH),
        "ONLINE_VLA": parent["FINAL_RUNTIME_USES_ONLINE_VLA"],
        "VLA_CHECKPOINT_LOADED": parent["VLA_CHECKPOINT_LOADED"],
        "VLA_ACTION_PROVENANCE_VERIFIED": parent["VLA_ACTION_PROVENANCE_VERIFIED"],
        "VLA_CHECKPOINT": checkpoint,
        "VLA_CHECKPOINT_SHA256": parent["VLA_CHECKPOINT_SHA256"],
        "BELIEF_CHECKPOINTS": parent["PHYSICAL_BELIEF_58D_CHECKPOINTS"],
        "FEASIBILITY_CHECKPOINTS": parent["FEASIBILITY_CHECKPOINT"],
        "POSTERIOR_INTERFACE": parent["POSTERIOR_INTERFACE"],
        "UTILITY": parent["UTILITY"],
        "FORCE_SUPPORT": parent["FORCE_SUPPORT"],
        "FORCE_SEARCH_STEP": parent["FORCE_GRID_STEP_N"],
        "CONTROLLER": parent["CONTROLLER"],
        "EVALUATOR": parent["LABEL_CONTRACT"],
        "PROBE": parent["PROBE_VERSION"],
        "TASKS": parent["TASK_INSTRUCTIONS"],
        "FRICTION_VALUES": "frozen canonical LOW/MID/HIGH values in DEV_PLAN.json",
        "MEASURED_FORCE_CONTRACT": "branch_step>=1 AND vla_release_intent=false; per-frame 2*min(abs(left object-normal force),abs(right object-normal force)); missing contact retained as zero; branch mean then equal branch weighting",
        "PARENT_SCIENTIFIC_COMPONENT_CHANGED": False,
    }
    dump("SECONDARY_BASELINE_RUNTIME_PARENT_MANIFEST.json", parent_record)

    no_probe = {
        "TRUE_NOPROBE_PATHWAY_EXISTS": False,
        "NOPROBE_BELIEF_DEFINITION": "UNAVAILABLE_WITH_FROZEN_MODEL: the 58D ensemble requires P4-B RAW_PROBE.csv and CONTACT_PATCH_READBACK.json",
        "NOPROBE_STATE_DEFINITION": "UNAVAILABLE_WITH_FROZEN_MODEL: feasibility/runtime is qualified at the post-probe DECISION_STATE, not the established-grasp pre-probe state",
        "NOPROBE_VLA_MOTION_CONTEXT": "Could be inferred online from the pre-probe state, but would be paired with an unqualified feasibility state distribution",
        "NOPROBE_FEASIBILITY_INPUT_VALID": False,
        "forbidden_shortcuts": ["zero probe features", "fabricated probe trace", "pre-probe state into post-probe-qualified model", "probe then relabel as NoProbe"],
        "reason": "A legal no-observation belief/state adapter would require a new qualified model pathway; new training and method changes are forbidden in this round.",
        "evidence": {
            "worker": {"path": str(FINAL / "SOURCE_SNAPSHOT/worker.py"), "sha256": sha(SOURCE / "worker.py"), "lines": "66-94"},
            "existing_prior_adapter": {"path": str(FINAL.parent / "final_ablation_confirmatory_v1/SOURCE_SNAPSHOT/online_ablation_feasibility.py"), "sha256": sha(FINAL.parent / "final_ablation_confirmatory_v1/SOURCE_SNAPSHOT/online_ablation_feasibility.py"), "lines": "96-107; explicitly probe_executed=True, legal_no_probe_pathway=False"},
        },
        "physics_action": "SKIP_TRUE_NOPROBE",
    }
    dump("TRUE_NOPROBE_VALIDITY_AUDIT.json", no_probe)
    (STAGE / "TRUE_NOPROBE_VALIDITY_AUDIT.md").write_text(
        "# True NoProbe validity audit\n\n**Verdict: NO.** The frozen 58D belief consumes the actual P4-B probe telemetry and the frozen online feasibility path is qualified at the post-probe decision state. There is no frozen no-observation belief/state adapter. Zeroing or fabricating probe inputs, or feeding the established-grasp state into the post-probe model, would violate the requested contract. No True NoProbe physics is planned.\n"
    )

    gt = {
        "GT_PHYSICS_PATHWAY_VALID": True,
        "GT_PHYSICS_USES_PRIVILEGED_SIMULATOR_FRICTION": True,
        "only_change": "posterior quadrature nodes/weights -> [mu_GT]/[1.0] before the unchanged phase-free feasibility curve and utility",
        "unchanged": ["established grasp", "P4-B probe", "post-probe state", "online VLA", "feasibility weights", "utility", "dense force grid", "controller", "evaluator"],
        "outcome_not_used": True,
        "adapter": {"path": str(FINAL / "SOURCE_SNAPSHOT/secondary_variants.py"), "sha256": sha(SOURCE / "secondary_variants.py")},
    }
    dump("GT_PHYSICS_VALIDITY_AUDIT.json", gt)
    (STAGE / "GT_PHYSICS_VALIDITY_AUDIT.md").write_text(
        "# GT-Physics validity audit\n\n**Verdict: YES.** `GT_PHYSICS_DIRECT` retains the complete Full-AF pipeline and replaces only posterior quadrature by the privileged delta distribution at simulator friction. It is an information upper bound, not a deployable method.\n"
    )

    tabero = {
        "TABERO_NEUTRAL_MATCHED_VALID": True,
        "classification": "MATCHED_ONLINE_VLA_EXTERNAL_BASELINE_WITH_DIFFERENT_GRIPPER_CONTROLLER_PATH",
        "TABERO_VLA_CHECKPOINT": checkpoint,
        "TABERO_CHECKPOINT_SHA256": parent["VLA_CHECKPOINT_SHA256"],
        "TABERO_FORCE_BEHAVIOR": "During grasp, execute raw postprocessed VLA aperture and six native force slots; on the common frozen VLA open-intent threshold, canonical 0.04m and zero force preserve identical release/evaluator semantics.",
        "TABERO_GRIPPER_INTERFACE": "native 13D action: xyz,axis-angle,aperture,left 3D force,right 3D force",
        "TABERO_CONTROLLER_PATH": "same low-level ForcePositionAction, native VLA aperture/force targets; AF scalar squeeze servo bypassed",
        "TABERO_ARM_POLICY_PATH": "same current online checkpoint, prompt, observation adapter, predict50/execute10/requery10; first six action dimensions passed through identically",
        "same_established_grasp": True,
        "same_probe_and_postprobe_state_for_matching": True,
        "same_task_instruction": True,
        "same_online_checkpoint": True,
        "same_arm_semantics": True,
        "same_observation_requery_schedule": True,
        "same_evaluator": True,
        "same_release_semantics": True,
        "same_measured_force_definition": True,
        "caption_requirement": "External baseline; controller/gripper pathway differs, so do not describe as only scalar force strategy differing.",
        "historical_sources": {
            "official_client": {"path": str(OLD_TABERO_CLIENT), "sha256": sha(OLD_TABERO_CLIENT), "action_semantics_lines": "401-409"},
            "implementation_audit": {"path": str(TABERO_AUDIT), "sha256": sha(TABERO_AUDIT)},
        },
        "adapter": {"path": str(FINAL / "SOURCE_SNAPSHOT/secondary_variants.py"), "sha256": sha(SOURCE / "secondary_variants.py")},
    }
    dump("TABERO_NEUTRAL_VALIDITY_AUDIT.json", tabero)
    (STAGE / "TABERO_NEUTRAL_VALIDITY_AUDIT.md").write_text(
        "# Tabero-Neutral matched validity audit\n\n**Verdict: YES, as a separately captioned external baseline.** It uses the exact same online checkpoint, prompt, observations, arm action, predict/execute/re-query schedule, established-grasp scope, probe/post-probe state, release gate, evaluator, and measured-force metric. During grasp it executes the checkpoint's native aperture and six force slots instead of the AF scalar squeeze servo. Because the controller/gripper pathway differs, it is not described as a one-variable controlled force ablation.\n"
    )

    variants = {
        "version": "FINAL_SECONDARY_VARIANTS_V1",
        "frozen_before_physics": True,
        "valid_variants": ["ACTIVEFORCING", "GT_PHYSICS_DIRECT", "TABERO_NEUTRAL"],
        "skipped_variants": {"TRUE_NOPROBE": "NO_FROZEN_LEGAL_NO_OBSERVATION_PATHWAY"},
        "variants": {
            "ACTIVEFORCING": {"PHYSICS_INFORMATION_SOURCE": "P4B_PROBE_POSTERIOR", "PROBE_USED": True, "BELIEF_USED": "continuous positive-support posterior", "FEASIBILITY_USED": "frozen phase-free full-task", "UTILITY_USED": parent["UTILITY"], "FORCE_SEARCH_USED": "3.00:0.05:5.00 N", "GRIPPER_FORCE_SOURCE": "ActiveForcing servo", "ARM_ACTION_SOURCE": "online frozen VLA", "VLA_CHECKPOINT": checkpoint, "CONTROLLER": parent["CONTROLLER"], "EVALUATOR": parent["LABEL_CONTRACT"]},
            "GT_PHYSICS_DIRECT": {"PHYSICS_INFORMATION_SOURCE": "privileged simulator mu_GT delta belief", "PROBE_USED": True, "BELIEF_USED": "delta(mu-mu_GT)", "FEASIBILITY_USED": "same frozen phase-free full-task", "UTILITY_USED": parent["UTILITY"], "FORCE_SEARCH_USED": "3.00:0.05:5.00 N", "GRIPPER_FORCE_SOURCE": "same ActiveForcing servo", "ARM_ACTION_SOURCE": "online frozen VLA", "VLA_CHECKPOINT": checkpoint, "CONTROLLER": parent["CONTROLLER"], "EVALUATOR": parent["LABEL_CONTRACT"]},
            "TABERO_NEUTRAL": {"PHYSICS_INFORMATION_SOURCE": "none for force selection", "PROBE_USED": True, "BELIEF_USED": "computed for state parity, not used for native action", "FEASIBILITY_USED": "not used for executed gripper behavior", "UTILITY_USED": "not used", "FORCE_SEARCH_USED": "not used", "GRIPPER_FORCE_SOURCE": "raw online Tabero VLA native aperture/force slots with common canonical release", "ARM_ACTION_SOURCE": "same online frozen VLA", "VLA_CHECKPOINT": checkpoint, "CONTROLLER": "native Tabero gripper/force path through same low-level action manager", "EVALUATOR": parent["LABEL_CONTRACT"]},
        },
    }
    dump("FINAL_SECONDARY_VARIANT_MANIFEST.json", variants)

    roots = []
    for root in (170050, 170051):
        roots.append({
            "root_id": root,
            "root_seed": root,
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
            "generation_parameters": {"source": "outcome-blind contiguous prospective block", "block_start": 170050, "block_end": 170051, "tasks": [0, 1, 5, 6], "friction_bands": ["LOW", "MID", "HIGH"], "scene_sampling": "native authored randomization"},
            "snapshot_identity": {"template_source": str(PARENT_PLAN), "template_sha256": sha(PARENT_PLAN)},
        })
    root_doc = {"version": "NEW_SECONDARY_ROOTS_V1", "roots": roots, "root_list_frozen": True, "root_selection_used_outcomes": False, "physics_branches_started_before_freeze": 0, "root_id_semantics": "full integer IsaacLab reset seed", "excluded_prior_root_block": list(range(170040, 170050))}
    root_path = dump("NEW_SECONDARY_ROOTS.json", root_doc)
    (STAGE / "NEW_SECONDARY_ROOTS_SHA256.txt").write_text(sha(root_path) + "  NEW_SECONDARY_ROOTS.json\n")

    source_contexts = json.loads(PARENT_PLAN.read_text())["contexts"][:12]
    contexts = []
    for root in (170050, 170051):
        for base in source_contexts:
            contexts.append({**base, "root": root, "id": f"t{base['task']}_r{root}_{base['band'].lower()}"})
    dev_plan = {"role": "FINAL_SECONDARY_ONLY_UNTOUCHED", "roots": [170050, 170051], "contexts": contexts, "methods": variants["valid_variants"], "max_branches": 72, "main_final_roots_used": False}
    dump("DEV_PLAN.json", dev_plan)

    rng = random.Random(2026090901)
    queue, execution_index = [], 0
    for context in contexts:
        order = variants["valid_variants"].copy()
        rng.shuffle(order)
        for method_order, variant in enumerate(order, 1):
            queue.append({"execution_index": execution_index, "root": context["root"], "task": context["task"], "friction": context["band"], "context_id": context["id"], "variant": variant, "execution_order": method_order, "snapshot_id": "secondary_v1_frozen_postprobe_snapshot"})
            execution_index += 1
    plan_path = STAGE / "FINAL_SECONDARY_EXECUTION_PLAN.csv"
    with plan_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(queue[0]))
        writer.writeheader(); writer.writerows(queue)
    (STAGE / "FINAL_SECONDARY_EXECUTION_PLAN_SHA256.txt").write_text(sha(plan_path) + "  FINAL_SECONDARY_EXECUTION_PLAN.csv\n")

    retry = {"version": "FINAL_SECONDARY_RETRY_POLICY_V1", "allowed": ["ENOSPC before valid provenance", "simulator crash", "policy transport failure", "corrupted restore", "incomplete provenance"], "forbidden": ["physical failure", "drop", "bad placement", "low success rate", "strange force", "outcome harms paper"], "silent_overwrite": False, "preserve_original_attempt": True}
    dump("FINAL_SECONDARY_RETRY_POLICY.json", retry)

    external_sources = {str(PARENT_PATH): sha(PARENT_PATH), str(PARENT_PLAN): sha(PARENT_PLAN), str(OLD_TABERO_CLIENT): sha(OLD_TABERO_CLIENT), str(TABERO_AUDIT): sha(TABERO_AUDIT)}
    for ckpt in parent["PHYSICAL_BELIEF_58D_CHECKPOINTS"]:
        external_sources[ckpt["path"]] = ckpt["sha256"]
    for ckpt in parent["FEASIBILITY_CHECKPOINT"]["actually_loaded_source_checkpoints"]:
        external_sources[ckpt["path"]] = ckpt["sha256"]
    final_sources = {str(FINAL / "SOURCE_SNAPSHOT" / p.name): sha(p) for p in SOURCE.glob("*.py")}
    for name in ["SECONDARY_BASELINE_RUNTIME_PARENT_MANIFEST.json", "TRUE_NOPROBE_VALIDITY_AUDIT.json", "GT_PHYSICS_VALIDITY_AUDIT.json", "TABERO_NEUTRAL_VALIDITY_AUDIT.json", "FINAL_SECONDARY_VARIANT_MANIFEST.json", "NEW_SECONDARY_ROOTS.json", "DEV_PLAN.json", "FINAL_SECONDARY_EXECUTION_PLAN.csv", "FINAL_SECONDARY_RETRY_POLICY.json"]:
        final_sources[str(FINAL / name)] = sha(STAGE / name)
    candidate = dict(parent)
    candidate.update({
        "version": "FINAL_SECONDARY_ONLINE_VLA_RUNTIME_V1",
        "source_hashes": {**external_sources, **final_sources},
        "source_parent_runtime": str(PARENT_PATH),
        "source_parent_runtime_sha256": sha(PARENT_PATH),
        "scientific_parent_components_changed": False,
        "selected_secondary_methods": variants["valid_variants"],
        "true_noprobe_skipped": True,
        "secondary_roots": [170050, 170051],
        "secondary_plan_branches": 72,
        "POLICY_SERVER_VERSION": {**parent["POLICY_SERVER_VERSION"], "readiness_evidence": str(FINAL / "POLICY_SERVER_18885/SERVER_READY.json"), "same_checkpoint_and_source_new_process_instance": True},
    })
    dump("CANDIDATE_RUNTIME_MANIFEST.json", candidate)
    dump("PREPHYSICS_FREEZE_STATUS.json", {"variant_manifest_sha256": sha(STAGE / "FINAL_SECONDARY_VARIANT_MANIFEST.json"), "root_manifest_sha256": sha(root_path), "execution_plan_sha256": sha(plan_path), "candidate_runtime_manifest_sha256": sha(STAGE / "CANDIDATE_RUNTIME_MANIFEST.json"), "physics_branches_started": 0, "roots_frozen": True, "execution_order_frozen": True})


if __name__ == "__main__":
    main()
