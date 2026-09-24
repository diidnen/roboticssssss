#!/usr/bin/env python3
"""Generate the frozen visual-context availability audit and conditional-stop artifacts.

This script intentionally does not train or evaluate a new model.  The preregistered
representation precedence stops when no aligned pre-probe RGB or frozen visual feature
is available for the authoritative TRAIN/DEV contexts.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


OUT = Path("/home/exouser/FORTE/gnp_style_visual_context_20260831_003137")
GNP = Path("/home/exouser/Tabero/analysis/results/gnp_style_continuous_20260830_125107")
DEV = Path("/home/exouser/Tabero/analysis/results/continuous_probe_joint_20260830_110712")
P5 = Path("/home/exouser/Tabero/analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542")
CTX = Path("/home/exouser/Tabero/analysis/results/context_calibration_forensic_20260830_235151")
STOCH = Path("/home/exouser/Tabero/analysis/results/continuous_coverage_stochasticity_20260830_233650")
NOW = datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(name: str, obj: object) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict], fields: list[str] | None = None) -> None:
    p = OUT / name
    if fields is None:
        fields = list(rows[0]) if rows else ["status"]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finite_or_none(value: str):
    try:
        x = float(value)
        return None if x != x else x
    except Exception:
        return None


OUT.mkdir(parents=True, exist_ok=False)

# Population and frame-alignment inventory.
train_rows = read_csv(GNP / "CONTINUOUS_TRAIN_SUCCESS_DATA.csv")
dev_rows = read_csv(DEV / "REAL_CONTINUOUS_REPEAT_MANIFEST.csv")

train_contexts: dict[str, dict] = {}
for r in train_rows:
    train_contexts.setdefault(r["context_id"], r)
dev_contexts: dict[str, dict] = {}
for r in dev_rows:
    dev_contexts.setdefault(r["context_id"], r)

assert len(train_contexts) == 72, len(train_contexts)
assert len(dev_contexts) == 9, len(dev_contexts)

media_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".npy", ".npz", ".mp4", ".avi", ".gif"}
media_inventory = {}
for label, root in {"p5": P5, "gnp_train": GNP, "repeated_dev": DEV}.items():
    media = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in media_ext]
    media_inventory[label] = {"root": str(root), "eligible_media_file_count": len(media), "files": media}
assert sum(v["eligible_media_file_count"] for v in media_inventory.values()) == 0

alignment_rows: list[dict] = []
for split, contexts in (("TRAIN", train_contexts), ("DEV", dev_contexts)):
    for cid, r in sorted(contexts.items()):
        force_key = "requested_force_N" if split == "TRAIN" else "force_N"
        hash_key = "strict_preprobe_hash"
        alignment_rows.append(
            {
                "split": split,
                "context_id": cid,
                "root_id": r["root_id"],
                "task": r["task"],
                "friction_band": r["friction_band"],
                "friction": r["friction"],
                "example_force_N": r[force_key],
                "strict_preprobe_state_hash": r[hash_key],
                "expected_decision_frame": "STRICT_PREPROBE_LAST_HOLD",
                "agentview_rgb_path": "",
                "wrist_rgb_path": "",
                "pi0_visual_feature_path": "",
                "serialized_restorable_snapshot_path": "",
                "aligned_rgb_available": 0,
                "aligned_visual_feature_available": 0,
                "exact_offline_recovery_without_simulator_rerun": 0,
                "probe_motion_absent_verifiable_from_frame": 0,
                "postprobe_leakage_absent_verifiable_from_frame": 0,
                "status": "BLOCKED_ALIGNED_PREPROBE_VISUAL_OBSERVATION_NOT_ARCHIVED",
            }
        )
write_csv("VISUAL_CONTEXT_EXTRACTION_MANIFEST.csv", alignment_rows)

code_files = {
    "vla_observation_builder": Path("/home/exouser/Tabero/analysis/p6g1_primitive_ik_vla_grasp_realization.py"),
    "serving_wrapper": Path("/home/exouser/Tabero/analysis/results/b5_tabero_neutral_20260822_040652/scripts/b5_serve_policy_with_explicit_norm_stats.py"),
    "pi0_pytorch": Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA/src/openpi/models_pytorch/pi0_pytorch.py"),
    "paligemma_pytorch": Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA/src/openpi/models_pytorch/gemma_pytorch.py"),
    "tabero_input_transform": Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA/src/openpi/policies/libero_policy.py"),
    "strict_preprobe_collector": Path("/home/exouser/FORTE/gnp_style_continuous_collect.py"),
}
code_hashes = {k: {"path": str(v), "sha256": sha(v)} for k, v in code_files.items()}

vla_audit = {
    "generated_at": NOW,
    "status": "AUDITED_DEPLOYMENT_PATH_BUT_HISTORICAL_ALIGNMENT_BLOCKED",
    "policy": {
        "config": "pi0_lora_tacfield_tabero",
        "checkpoint": "/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999",
        "serving_interface": "WebsocketPolicyServer action inference",
        "serving_output": "actions and policy metadata; no hidden/image-token tensor",
        "pi0_modified_or_retrained": False,
    },
    "input_rgb_cameras": [
        {"runtime_name": "agentview_cam", "policy_key": "image/base_0_rgb", "shape": [224, 224, 3], "dtype": "uint8"},
        {"runtime_name": "eye_in_hand_cam", "policy_key": "wrist_image/left_wrist_0_rgb", "shape": [224, 224, 3], "dtype": "uint8"},
    ],
    "image_preprocessing": {
        "controller": "resize_frames_with_padding(..., (224,224,3), bgr_conversion=False, pad_img=True), then uint8 RGB",
        "policy_transform": "TaberoTacFieldInputs maps base and wrist RGB; right_wrist_0_rgb is zeros and masked false for PI0",
        "model_inference": "preprocess_observation_pytorch(train=False); resize-with-pad only if needed",
    },
    "visual_encoder": {
        "backbone_family": "PaliGemma vision tower (SigLIP path)",
        "entrypoint": "PaliGemmaWithExpertModel.embed_image -> paligemma.model.get_image_features",
        "token_width": 2048,
        "image_tokens_used_before_action_generation": True,
        "intermediate_feature_can_be_exposed_without_retraining_pi0": True,
        "intermediate_feature_exposed_by_current_serving_interface": False,
        "feature_persisted_for_authoritative_population": False,
    },
    "language_task": {
        "task_instruction_available": True,
        "prompt_tokens_embedded_separately": True,
        "frozen_language_embedding_archived_for_population": False,
        "task_identity_already_in_feasibility_backend": True,
    },
    "object_region": {
        "crop_or_mask_archived": False,
        "deployment_object_region_representation_already_present": False,
        "simulator_usd_geometry_seen_elsewhere": True,
        "simulator_usd_geometry_eligible": False,
        "reason": "privileged asset geometry is not the requested observable visual context and is not a deployed estimator",
    },
    "historical_alignment": {
        "train_contexts_required": len(train_contexts),
        "dev_contexts_required": len(dev_contexts),
        "aligned_preprobe_rgb_contexts": 0,
        "aligned_visual_feature_contexts": 0,
        "media_inventory": media_inventory,
        "state_hashes_archived": True,
        "restorable_scene_snapshots_archived": False,
        "collector_snapshot_semantics": "env.scene.get_state was retained only in memory; CSV/JSON contains its stable hash",
    },
    "actual_current_feasibility_input": {
        "task_one_hot": True,
        "phase_one_hot": True,
        "H8_nominal_cartesian_motion": True,
        "strict_preprobe_gripper_opening": True,
        "friction_mu": True,
        "candidate_force": True,
        "rgb_or_visual_feature": False,
    },
    "code_evidence": code_hashes,
}
write_json("VLA_VISUAL_CONTEXT_AUDIT.json", vla_audit)

alignment_audit = {
    "generated_at": NOW,
    "decision_moment": "strict pre-probe last-hold state before probe_out and before candidate-force branch",
    "required": {"TRAIN_contexts": 72, "frozen_DEV_contexts": 9, "total_contexts": 81},
    "available": {
        "context_rows_with_strict_state_hash": 81,
        "context_rows_with_aligned_agentview_rgb": 0,
        "context_rows_with_aligned_wrist_rgb": 0,
        "context_rows_with_frozen_pi0_visual_feature": 0,
        "context_rows_with_serialized_restorable_snapshot": 0,
    },
    "coverage": {"TRAIN": 0.0, "DEV": 0.0, "overall": 0.0},
    "checks": {
        "exact_frame_provenance": "FAIL_MISSING_FRAME",
        "timestamp_relation_to_pre_shear_hold": "NOT_VERIFIABLE_WITHOUT_FRAME",
        "no_active_probe_motion_yet": "STATE_CAPTURE_RULE_DOCUMENTED_BUT_NOT_VISUALLY_VERIFIABLE",
        "no_postprobe_leakage": "STATE_HASH_ONLY; NO_FRAME_TO_AUDIT",
        "outcome_independent_frame_selection": "NOT_APPLICABLE_NO_FRAME",
        "TEST_data": "NOT_LOADED",
        "context_level_reuse_across_forces_repeats": "INTENDED_BUT_NOT_POSSIBLE",
    },
    "snapshot_persistence_finding": "The collector stores the strict scene snapshot only in process memory for branch restores; it persists hashes and telemetry, not the scene state or RGB.",
    "scientific_consequence": "Replaying/re-rendering now would create a new observation not demonstrably identical to the historical decision-time RGB, so it cannot be silently treated as frozen aligned x.",
    "status": "FAIL_ALIGNED_PREPROBE_VISUAL_CONTEXT_UNAVAILABLE",
}
write_json("PREPROBE_VISUAL_ALIGNMENT_AUDIT.json", alignment_audit)

feature_spec = {
    "generated_at": NOW,
    "selection_precedence_frozen": [
        {"rank": 1, "candidate": "existing pi0 inference visual representation", "deployment_available": True, "historical_population_available": False, "reason": "not exposed or persisted"},
        {"rank": 2, "candidate": "exact pi0 vision-backbone output", "deployment_available": True, "historical_population_available": False, "reason": "requires missing aligned RGB"},
        {"rank": 3, "candidate": "pooled same-encoder representation recomputed from authoritative preprobe RGB", "deployment_available": True, "historical_population_available": False, "reason": "authoritative aligned RGB is absent"},
        {"rank": 4, "candidate": "existing project object-region representation", "deployment_available": False, "historical_population_available": False, "reason": "no crop/mask/region feature is archived or deployed"},
    ],
    "selected_x": None,
    "projection": None,
    "normalization": None,
    "feature_fishing_performed": False,
    "external_vision_model_used": False,
    "status": "NO_ELIGIBLE_REPRESENTATION_AFTER_PREREGISTERED_PRECEDENCE",
}
write_json("VISUAL_CONTEXT_FEATURE_SPEC.json", feature_spec)

protocol = {
    "protocol_name": "GNP_STYLE_VISUAL_CONTEXT_PROTOCOL",
    "frozen_at": NOW,
    "scientific_question": "Does deployment-time visual object/grasp context x fix continuous feasibility, and does Joint add value under matched x?",
    "claim_boundary": {
        "current_object_task_distribution_only": True,
        "cross_object": False,
        "unseen_object": False,
        "unseen_task": False,
        "original_TEST_loaded": False,
        "fresh_E2E": False,
    },
    "data": {
        "TRAIN_contexts": 72,
        "continuous_branches": 720,
        "historical_coarse_branches": 288,
        "feasibility_outcomes": 1008,
        "DEV_contexts": 9,
        "DEV_cells": 27,
        "DEV_repeats": 135,
    },
    "selection_rule": "Use first historically aligned, deployment-available representation in the user-specified four-level precedence; stop if none exists.",
    "selection_result": "STOP_DEPLOYABLE_VISUAL_CONTEXT_UNAVAILABLE",
    "selected_x": None,
    "conditional_training": {
        "executed": False,
        "reason": "no scientifically aligned x",
        "models": ["VISUAL_CONTEXT_CONTINUOUS_FEAS", "VISUAL_CONTEXT_CONTINUOUS_JOINT"],
        "seeds": [0, 1, 2],
        "epochs": 80,
        "probability_semantics": "raw mean sigmoid over 3 seeds",
        "rho": 0.8,
    },
    "GT_gate": {
        "real_frontier_coverage_min": 0.8,
        "finite_decision_coverage_min": 0.8,
        "probability_MAE_max": 0.2,
        "frontier_MAE_N_max": 0.2,
        "under_force_rate_max": 0.1,
        "systematic_nonmonotonicity_allowed": False,
    },
    "joint_value_criteria": {
        "under_force": "lower or safely non-worse",
        "frontier_MAE_improvement_N_min": 0.05,
        "probability_MAE_relative_improvement_min": 0.10,
        "dense_grid_monotonicity_required": True,
        "Brier_NLL_not_materially_worse": True,
    },
    "conditional_probe": {"executed": False, "reason": "visual-context GT gate not reached", "rho": 0.8, "grid_N": 0.05},
    "authoritative_inputs": [str(STOCH), str(CTX), str(GNP), str(DEV)],
    "code_hashes": code_hashes,
    "no_outcome_driven_feature_selection": True,
    "no_new_force_samples": True,
    "no_new_repeats": True,
    "pi0_modified": False,
    "architecture_search": False,
}
write_json("GNP_STYLE_VISUAL_CONTEXT_PROTOCOL.json", protocol)
(OUT / "GNP_STYLE_VISUAL_CONTEXT_PROTOCOL.json.sha256").write_text(
    sha(OUT / "GNP_STYLE_VISUAL_CONTEXT_PROTOCOL.json") + "  GNP_STYLE_VISUAL_CONTEXT_PROTOCOL.json\n",
    encoding="utf-8",
)

fairness = {
    "status": "NOT_EXECUTED_PRECONDITION_FAILED",
    "intended_same_outcome_data": True,
    "intended_same_1008_feasibility_outcomes": True,
    "intended_same_visual_context_x": True,
    "intended_same_seeds": [0, 1, 2],
    "intended_joint_only_extra_supervision": "corrected-valid physical telemetry/IE from same branches",
    "actual_training_runs": 0,
    "reason": "no aligned eligible x, so a fair matched visual comparison cannot be instantiated",
}
write_json("VISUAL_CONTEXT_FAIRNESS_AUDIT.json", fairness)

not_executed = [{"status": "NOT_EXECUTED", "reason": "DEPLOYABLE_VISUAL_CONTEXT_UNAVAILABLE", "selected_x": ""}]
write_csv("VISUAL_CONTEXT_LINEAR_PROBE.csv", not_executed)
write_csv("VISUAL_CONTEXT_FEAS_TRAINING_MANIFEST.csv", not_executed)
write_csv("VISUAL_CONTEXT_JOINT_TRAINING_MANIFEST.csv", not_executed)
write_json(
    "ALL_VISUAL_CONTEXT_CHECKPOINTS_FROZEN.json",
    {
        "status": "NOT_EXECUTED",
        "expected_checkpoints": 6,
        "completed_checkpoints": 0,
        "checkpoints": [],
        "reason": "DEPLOYABLE_VISUAL_CONTEXT_UNAVAILABLE",
    },
)

old_prob = read_csv(GNP / "CONTINUOUS_DEV_PROBABILITY_METRICS.csv")
prob_rows = []
for r in old_prob:
    if r["probability_semantics"] == "RAW_ENSEMBLE_PRIMARY":
        prob_rows.append(
            {
                "model": "OLD_" + ("FEAS" if r["backend"] == "FEASIBILITY_ONLY" else "JOINT"),
                "status": "AUTHORITATIVE_EXISTING",
                "probability_cells": r["probability_cells"],
                "probability_MAE": r["probability_MAE"],
                "Brier": r["Brier"],
                "NLL": r["negative_log_likelihood"],
                "signed_bias": "-0.1859674397555302" if r["backend"] == "FEASIBILITY_ONLY" else "-0.1705559770342778",
                "per_context_probability_MAE_available": 1,
            }
        )
for model in ("VISUAL_CONTEXT_FEAS", "VISUAL_CONTEXT_JOINT"):
    prob_rows.append({"model": model, "status": "NOT_EVALUATED_NO_X", "probability_cells": 0, "probability_MAE": "", "Brier": "", "NLL": "", "signed_bias": "", "per_context_probability_MAE_available": 0})
write_csv("VISUAL_CONTEXT_DEV_PROBABILITY_METRICS.csv", prob_rows)

front_rows = []
for r in old_prob:
    if r["probability_semantics"] == "RAW_ENSEMBLE_PRIMARY":
        front_rows.append(
            {
                "model": "OLD_" + ("FEAS" if r["backend"] == "FEASIBILITY_ONLY" else "JOINT"),
                "status": "AUTHORITATIVE_EXISTING",
                "valid_real_frontier_contexts": r["valid_frontier_contexts"],
                "finite_decision_contexts": r["valid_force_decision_contexts"],
                "finite_decision_coverage": 1.0 - float(r["decision_missing_rate"]),
                "frontier_MAE_N": r["frontier_MAE_N"],
                "under_force_rate": r["under_force_rate"],
                "mean_under_force_magnitude_N": r["mean_under_force_magnitude_N"],
                "mean_excess_force_N": r["mean_excess_force_N"],
                "within_0p25N_rate": r["within_0p25N_rate"],
                "within_0p50N_rate": r["within_0p50N_rate"],
                "monotonic_contexts": 9 if r["backend"] == "FEASIBILITY_ONLY" else 3,
            }
        )
for model in ("VISUAL_CONTEXT_FEAS", "VISUAL_CONTEXT_JOINT"):
    front_rows.append({"model": model, "status": "NOT_EVALUATED_NO_X", "valid_real_frontier_contexts": 8, "finite_decision_contexts": 0, "finite_decision_coverage": "", "frontier_MAE_N": "", "under_force_rate": "", "mean_under_force_magnitude_N": "", "mean_excess_force_N": "", "within_0p25N_rate": "", "within_0p50N_rate": "", "monotonic_contexts": ""})
write_csv("VISUAL_CONTEXT_DEV_FRONTIER_METRICS.csv", front_rows)

write_csv(
    "VISUAL_CONTEXT_JOINT_VS_FEAS.csv",
    [{
        "comparison": "VISUAL_CONTEXT_FEAS_vs_VISUAL_CONTEXT_JOINT",
        "status": "NOT_REACHED",
        "same_x": "NOT_INSTANTIABLE",
        "joint_probability_MAE_relative_improvement": "",
        "joint_frontier_MAE_improvement_N": "",
        "under_force_comparison": "",
        "joint_status": "EVIDENCE_LIMITED",
        "reason": "No aligned x; neither matched model was trained.",
    }],
)

gate = {
    "status": "NOT_REACHED",
    "evaluated_model": None,
    "selected_visual_backend": None,
    "criteria_frozen": protocol["GT_gate"],
    "pass": False,
    "reason": "DEPLOYABLE_VISUAL_CONTEXT_UNAVAILABLE; no visual-context checkpoint exists",
    "probe_stage": "NOT_REACHED",
}
write_json("VISUAL_CONTEXT_GT_GATE.json", gate)

report = f"""# STATUS

**COMPLETE WITH PREREGISTERED CONDITIONAL STOP.** The actual π0 visual pathway exists online, but no aligned strict-preprobe RGB, image token, visual feature, object crop, or restorable scene snapshot was archived for the 72 TRAIN and 9 frozen DEV contexts. The required visual-context experiment therefore cannot be run without creating a new observation dataset.

# SINGLE SCIENTIFIC GOAL

Test whether the remaining continuous-feasibility error is caused by an impoverished observable object/grasp context `x`, then test Joint under exactly the same repaired `x` and continue to Probe only if the GT gate passes.

# GNP CONNECTION

GNP conditions feasibility on observable context `x`, hidden dynamics `z`, and action `a`. Here `z≈friction μ` and `a=continuous force F` already exist. The intended new `x` was a frozen, deployment-time π0/object/grasp visual representation. This run audited whether that `x` could be aligned to the already collected outcomes; it did not assume availability from architecture documentation.

# CURRENT INPUT GAP

The current feasibility tensor contains task and phase one-hot features, H8 nominal π0 Cartesian motion, strict-preprobe gripper opening/state, μ, and F. It contains no RGB, visual token, object geometry representation, object crop, grasp-relative visual geometry, or π0 latent.

# FROZEN VLA / VISUAL CONTEXT AVAILABLE

At live deployment, yes: `agentview_cam` and `eye_in_hand_cam` supply 224×224 RGB to `TaberoTacFieldInputs`; π0 calls PaliGemma/SigLIP image feature extraction before action generation. In the current websocket deployment interface, however, only actions are returned. Hidden/image-token tensors are neither exposed nor persisted.

For this frozen historical experiment, **no**: all three authoritative result roots contain zero eligible image, video, image-array, or visual-feature files. The only aligned evidence is a strict-preprobe state hash.

# PRE-PROBE VISUAL ALIGNMENT

Coverage is **0/72 TRAIN** and **0/9 DEV**. The continuous collector captured `env.scene.get_state(...)` at the last hold, used it in memory for restores, and wrote hashes/telemetry. It did not serialize the snapshot or RGB. A state hash cannot reconstruct pixels. Rendering now would create a new observation and cannot be asserted to equal the historical decision-time frame.

# SELECTED VISUAL REPRESENTATION

None. All four preregistered precedence levels fail the historical-alignment requirement:

1. π0 visual representation: produced online but not exposed/persisted.
2. Exact backbone output: callable, but aligned RGB input is missing.
3. Recomputed pooled same-encoder feature: blocked by the same missing RGB.
4. Existing object-region representation: none is archived or deployed.

No external vision model was introduced and no layer fishing was performed.

# TRAINING FAIRNESS

The intended matched comparison was fair—same 1008 feasibility outcomes, same `x`, same μ/F/context, same seeds; Joint would receive only corrected-valid physical supervision. Because `x` is absent, this comparison cannot be instantiated. Zero new checkpoints were trained.

# OLD FEAS / OLD JOINT

The frozen old baselines remain unchanged:

| model | probability MAE | Brier | NLL | frontier MAE | under-force | finite decisions | monotonic contexts |
|---|---:|---:|---:|---:|---:|---:|---:|
| OLD FEAS | 0.245 | 0.086 | 0.463 | 0.325 N | 2/8 | 6/8 | 9/9 |
| OLD JOINT | 0.207 | 0.135 | 0.939 | 0.125 N | 3/8 | 8/8 | 3/9 |

# VISUAL FEAS

**NOT EXECUTED.** There is no eligible aligned `x`; training would silently associate outcomes with newly generated or mis-timed images.

# VISUAL JOINT

**NOT EXECUTED** for the same common-input reason. Joint is not prejudged.

# PROBABILITY ESTIMATION

No new probability estimate exists. The frozen OLD FEAS probability MAE remains 0.245. This run cannot measure the causal effect of visual context.

# CONTINUOUS FRONTIER

No new frontier estimate exists. The frozen OLD FEAS frontier MAE remains 0.325 N with 6/8 finite decisions.

# UNDER-FORCE / EXCESS FORCE

No new safety result exists. The frozen OLD FEAS under-force rate remains 2/8; OLD JOINT remains 3/8.

# DOES VISUAL CONTEXT FIX THE ERROR?

**NOT TESTED.** The scientific hypothesis remains plausible but unvalidated. The experiment was blocked by missing aligned historical observations, not by absence of a vision encoder.

# DOES JOINT HELP AFTER BOTH MODELS SEE THE SAME CONTEXT?

**EVIDENCE LIMITED.** Neither matched visual model could be trained, so no independent Joint value can be assigned.

# GT CONTINUOUS GATE

**NOT REACHED.** There is no visual-context backend to evaluate. The existing backend already failed the frozen gate.

# PROBE VS STRICT NO-PROBE

Not run because the GT visual-context gate was not reached.

# DOES PROBE NOW IMPROVE CONTINUOUS FORCE?

**NOT REACHED.**

# QUANTIZATION UNMASKING

Not applicable because no GT-valid visual-context backend exists.

# PRIMARY_CLASSIFICATION

**DEPLOYABLE_VISUAL_CONTEXT_UNAVAILABLE**

Here “unavailable” means unavailable as an aligned feature for this frozen historical TRAIN/DEV experiment. Live deployment does have RGB and a frozen π0 visual encoder.

# JOINT STATUS

**EVIDENCE_LIMITED**

# PROBE STATUS

**NOT_REACHED**

# WHAT IS NOW PROVEN

- The deployed π0 code path genuinely uses two RGB streams and PaliGemma/SigLIP image tokens.
- The current feasibility model does not receive those pixels or tokens.
- The authoritative continuous TRAIN/DEV artifacts do not preserve aligned preprobe pixels/features or a serialized restorable snapshot.
- Consequently, the proposed frozen historical visual-context comparison is not identifiable from current artifacts.

# WHAT IS STILL NOT PROVEN

- It is not proven that too little `x` caused the feasibility error.
- It is not proven that visual context fixes probability or frontier error.
- It is not proven whether Joint helps after matched visual context.
- There is no cross-object generalization claim.
- There is no unseen-task claim.
- No original TEST was loaded.
- No fresh E2E was run.
- There is no when-to-probe agent yet.

# FINAL METHOD IMPLICATION

The desired formulation remains:

`Frozen VLA visual/object/task context x + Active Probe friction belief + continuous force F → Full-Task Feasibility → minimum reliable grip force`.

It cannot yet be frozen as the method because `x` was not archived for the training/evaluation outcomes.

# NEXT_METHOD

Do not add visual layers blindly. First make the existing deployment observation auditable: at the strict-preprobe last-hold event, persist both 224×224 RGB inputs and one preregistered pooled output from the exact frozen π0 vision backbone, together with the strict-state hash, camera configuration/hash, timestamp/step, and no-probe/no-outcome provenance. Only then repeat the same matched Feas/Joint study on a prospectively aligned population. If pixels cannot explain same-μ/same-F differences, audit deployment-inferable mass/COM/object geometry rather than using simulator-private values.
"""
(OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

# Canonical portable report artifact.  The chart encodes binary audit gates, not model performance.
gate_rows = [
    {"gate": "π0 uses RGB", "passed": 1, "required": 1, "evidence": "PaliGemma/SigLIP image tokens in forward path", "population": "live deployment"},
    {"gate": "Serving exposes visual token", "passed": 0, "required": 1, "evidence": "websocket returns actions only", "population": "live deployment"},
    {"gate": "Aligned TRAIN RGB/features", "passed": 0, "required": 1, "evidence": "0/72 contexts", "population": "frozen TRAIN"},
    {"gate": "Aligned DEV RGB/features", "passed": 0, "required": 1, "evidence": "0/9 contexts", "population": "frozen repeated DEV"},
    {"gate": "Serialized restorable snapshot", "passed": 0, "required": 1, "evidence": "hash only; snapshot was in-memory", "population": "TRAIN + DEV"},
]
artifact = {
    "surface": "report",
    "manifest": {
        "surface": "report",
        "version": 1,
        "title": "GNP-Style Visual Context: Availability Gate",
        "description": "Technical audit of whether frozen π0 visual context can be aligned to the authoritative continuous TRAIN/DEV outcomes.",
        "generatedAt": NOW,
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# GNP-Style Visual Context: Availability Gate"},
            {"id": "executive", "type": "markdown", "body": "## Executive Summary\n\nπ0 genuinely uses RGB image tokens, but the frozen continuous experiment archived no aligned preprobe RGB/features and no restorable scene snapshot. The preregistered representation precedence therefore stops before training. Visual context remains an untested hypothesis; matched Joint value and Probe are not reached.", "sourceId": "audit_source"},
            {"id": "status", "type": "markdown", "body": "## Status\n\n**PRIMARY: DEPLOYABLE_VISUAL_CONTEXT_UNAVAILABLE.** This means unavailable for the frozen historical population, not unavailable online."},
            {"id": "gates_chart_block", "type": "chart", "chartId": "availability_gates"},
            {"id": "gates_table_block", "type": "table", "tableId": "availability_table"},
            {"id": "alignment", "type": "markdown", "body": "## Pre-probe Alignment\n\nThe strict state is hash-audited, but pixels were not persisted. Rendering now would be a new observation, so it cannot be silently paired with historical outcomes.", "sourceId": "alignment_source"},
            {"id": "decision", "type": "markdown", "body": "## Scientific Decision\n\n**JOINT: EVIDENCE_LIMITED. PROBE: NOT_REACHED.** No new model or DEV evaluation was run."},
            {"id": "next", "type": "markdown", "body": "## Next Method\n\nProspectively persist both preprobe RGB inputs and one preregistered frozen π0 pooled visual representation with state/camera hashes before repeating matched Feas/Joint evaluation."},
        ],
        "cards": [],
        "charts": [{
            "id": "availability_gates", "type": "horizontalBar", "compatibleTypes": ["horizontalBar", "bar"],
            "title": "Visual-context availability gates", "subtitle": "Binary audit outcome; 1 = passed, 0 = blocked.",
            "question": "Which prerequisites for a frozen matched visual-context experiment are satisfied?",
            "rationale": "The ordered binary gates show that model-side vision exists while data alignment fails.",
            "intent": "comparison", "layout": "full", "dataset": "availability_gates", "sourceId": "audit_source",
            "encodings": {"x": {"field": "gate", "label": "Gate", "type": "nominal"}, "y": {"field": "passed", "label": "Passed", "type": "quantitative", "format": "number"}},
            "xAxisTitle": "Gate", "yAxisTitle": "Passed (1=yes)", "valueFormat": "number"
        }],
        "tables": [{
            "id": "availability_table", "title": "Audit evidence by gate", "subtitle": "All required gates must pass before training.",
            "dataset": "availability_gates", "layout": "full", "density": "dense", "sourceId": "audit_source",
            "defaultSort": {"field": "passed", "direction": "desc"},
            "columns": [
                {"field": "gate", "label": "Gate", "type": "text"}, {"field": "population", "label": "Population", "type": "text"},
                {"field": "passed", "label": "Passed", "format": "number"}, {"field": "evidence", "label": "Evidence", "type": "text"}
            ]
        }],
        "sources": [
            {"id": "audit_source", "label": "VLA and historical artifact audit", "path": "VLA_VISUAL_CONTEXT_AUDIT.json", "query": {"engine": "python", "language": "python", "description": "Read code hashes and scan authoritative roots for aligned media/features.", "sql": "SELECT * FROM read_json_auto('VLA_VISUAL_CONTEXT_AUDIT.json')", "tables_used": ["VLA_VISUAL_CONTEXT_AUDIT.json", "VISUAL_CONTEXT_EXTRACTION_MANIFEST.csv"], "filters": ["72 TRAIN contexts", "9 frozen DEV contexts", "no TEST"], "metric_definitions": ["Passed = 1 only when the prerequisite is directly evidenced for the stated population."]}},
            {"id": "alignment_source", "label": "Strict pre-probe alignment audit", "path": "PREPROBE_VISUAL_ALIGNMENT_AUDIT.json"},
        ],
    },
    "snapshot": {"version": 1, "generatedAt": NOW, "status": "blocked", "accessIssues": [{"code": "ALIGNED_VISUAL_CONTEXT_MISSING", "message": "No aligned preprobe RGB/feature or serialized restorable snapshot exists for the authoritative population."}], "datasets": {"availability_gates": gate_rows}},
    "sources": [],
    "package_info": {"mode": "portable_html", "snapshot_generated_at": NOW},
}
write_json("artifact.json", artifact)


def write_sums() -> None:
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    (OUT / "SHA256SUMS.txt").write_text("".join(f"{sha(p)}  {p.name}\n" for p in files), encoding="utf-8")


write_sums()
print(OUT)
print(f"train_contexts={len(train_contexts)} dev_contexts={len(dev_contexts)} aligned=0")
