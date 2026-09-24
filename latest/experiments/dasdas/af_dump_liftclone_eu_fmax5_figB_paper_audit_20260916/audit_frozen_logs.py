#!/usr/bin/env python3
"""Re-verify formal liftclone EU fmax5 from frozen logs. No new sim, no overwrite of source runs."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

AF = Path("/media/volume/dasdas/exouser/af_dump_formal_liftclone_eu_fmax5_20260915")
NOM = Path("/media/volume/dasdas/exouser/af_dump_formal_liftclone_eu_fmax5_nominal_20260915")
FILL = Path("/media/volume/dasdas/exouser/af_dump_formal_liftclone_eu_fmax5_prefix_fill_20260915")
FIG = Path("/media/volume/dasdas/exouser/figB_dump_bin_af_frames_20260915")
FIG_BUILDER = Path("/home/exouser/artifacts/paper_figures_v1")
OUT = Path("/media/volume/dasdas/exouser/af_dump_liftclone_eu_fmax5_figB_paper_audit_20260916")
FEAS_TRAIN = Path("/media/volume/dasdas/exouser/af_dump_liftstyle_feas_v3_fmax5_20260915")
FEAS_FORK = Path("/media/volume/dasdas/exouser/af_dump_liftstyle_feas_v3_fork_20260914")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def mu_key(value) -> str:
    return f"{float(value):.3f}"


def context_key(root, friction, seed) -> tuple:
    return (int(root), mu_key(friction), int(seed))


def parse_id_parts(context_id: str) -> dict:
    # liftclone_eu_fmax5[_nominal]_mu{mu}_root{root}_ps{seed}
    body = context_id.replace("liftclone_eu_fmax5_nominal_", "").replace("liftclone_eu_fmax5_", "")
    mu_s, rest = body.split("_root", 1)
    mu = float(mu_s.replace("mu", ""))
    root_s, seed_s = rest.split("_ps")
    return {"mu": mu, "root": int(root_s), "policy_seed": int(seed_s)}


def load_af_record(path: Path) -> dict:
    rec = read(path)
    ctx = rec["context"]
    out = rec["outcomes"][0]
    decision = rec.get("af_decision") or {}
    raw = AF / "main_raw" / ctx["id"]
    result_path = next((raw / "job").glob("branch_0_*/result.json"))
    result = read(result_path)
    lock = read(raw / "job" / "PREACTION_SELECTION_LOCK.json")
    material = read(raw / "job" / "query" / "ACTUAL_OBJECT_MATERIAL.json")
    grasp = read(raw / "job" / "query" / "ESTABLISHED_GRASP.json")
    shear = read(raw / "job" / "query" / "DUMP_SHEAR_QUERY.json")
    qual = read(raw / "job" / "query" / "qualification.json")
    feature = read(raw / "job" / "PREACTION_FEATURE.json")
    seq = feature.get("sequence") or []
    return {
        "method": "ActiveForcing",
        "record_path": str(path),
        "raw_dir": str(raw),
        "id": ctx["id"],
        "root": int(ctx["root"]),
        "friction": float(ctx["friction"]),
        "policy_seed": int(ctx["policy_seed"]),
        "key": context_key(ctx["root"], ctx["friction"], ctx["policy_seed"]),
        "success": int(out["success"]),
        "official_final_check": int(bool(result["official_final_check"])),
        "success_equals_official": bool(result["success"]) == bool(result["official_final_check"]),
        "F_star_N": float(decision["selected_force_N"]),
        "commanded_force_N": float(out["commanded_force_N"]),
        "force_setpoint_bilateral_n": result.get("force_setpoint_bilateral_n"),
        "argmax_p_counterfactual_N": decision.get("argmax_p_counterfactual_selected_N"),
        "executed_selector": decision.get("executed_selector"),
        "feature_source": decision.get("feature_source"),
        "predicted_success": decision.get("predicted_success"),
        "measured_mean_squeeze_N": float(out["measured_mean_squeeze_N"]),
        "measured_max_squeeze_N": float(out["measured_max_squeeze_N"]),
        "native_actions": out.get("native_actions"),
        "physics_steps": out.get("physics_steps") or result.get("physics_steps"),
        "release_actions": result.get("release_actions"),
        "handoff_state_sha256": rec.get("common_handoff_state_sha256"),
        "first_chunk_sha256": lock.get("first_chunk_sha256"),
        "protocol_sha256": ctx.get("formal_protocol_sha256"),
        "result_keys": sorted(result.keys()),
        "object_mu_readback": material["shape_materials"][0][0] if material.get("shape_materials") else None,
        "grasp_force_N": grasp.get("grasp_force_N"),
        "query_force_N": grasp.get("query_force_N") or shear.get("query_force_n"),
        "query_displacement_m": grasp.get("query_displacement_m") or shear.get("commanded_displacement_m"),
        "p4_protocol_scope": qual.get("scope"),
        "p4_probe_failure": qual.get("probe_failure"),
        "mq_feature_len": len(seq),
        "mq_feature_width": len(seq[0]) if seq else None,
        "drop_field_in_result": "drop" in result or "pre_release_drop" in result or "object_dropped" in result,
    }


def load_nominal_record(path: Path) -> dict:
    rec = read(path)
    ctx = rec["context"]
    out = rec["outcomes"][0]
    raw = NOM / "main_raw" / ctx["id"]
    result_path = next((raw / "job").glob("branch_0_*/result.json"))
    result = read(result_path)
    lock = read(raw / "job" / "PREACTION_SELECTION_LOCK.json")
    return {
        "method": "Nominal Frozen VLA",
        "record_path": str(path),
        "raw_dir": str(raw),
        "id": ctx["id"],
        "root": int(ctx["root"]),
        "friction": float(ctx["friction"]),
        "policy_seed": int(ctx["policy_seed"]),
        "matched_af_context_id": rec.get("matched_af_context_id") or ctx["id"].replace("_nominal", ""),
        "key": context_key(ctx["root"], ctx["friction"], ctx["policy_seed"]),
        "success": int(out["success"]),
        "official_final_check": int(bool(result["official_final_check"])),
        "success_equals_official": bool(result["success"]) == bool(result["official_final_check"]),
        "F_star_N": None,
        "commanded_force_N": out.get("commanded_force_N"),
        "force_setpoint_bilateral_n": result.get("force_setpoint_bilateral_n"),
        "measured_mean_squeeze_N": float(out["measured_mean_squeeze_N"]),
        "measured_max_squeeze_N": float(out["measured_max_squeeze_N"]),
        "native_actions": out.get("native_actions"),
        "physics_steps": out.get("physics_steps") or result.get("physics_steps"),
        "release_actions": result.get("release_actions"),
        "handoff_state_sha256": rec.get("common_handoff_state_sha256"),
        "first_chunk_sha256": lock.get("first_chunk_sha256"),
        "protocol_sha256": ctx.get("formal_protocol_sha256"),
        "result_keys": sorted(result.keys()),
        "drop_field_in_result": "drop" in result or "pre_release_drop" in result or "object_dropped" in result,
        "force_controller_installed": result.get("force_controller_installed"),
        "original_squeeze_inner_disabled_after_query": result.get("original_squeeze_inner_disabled_after_query"),
    }


def figure_events(context: str) -> dict:
    rows = [json.loads(line) for line in (FIG / "raw" / context / "frames" / "index.jsonl").read_text().splitlines() if line.strip()]
    z = [float(r["deskbin_xyz"][2]) for r in rows]
    z0 = z[0]
    lift = next(i for i, val in enumerate(z) if val >= z0 + 0.03)
    pour_hits = [i for i, val in enumerate(z) if val >= 0.95]
    pour = pour_hits[0] if pour_hits else None
    return {
        "n_frames": len(rows),
        "z0": z0,
        "z_max": max(z),
        "z_terminal": z[-1],
        "handoff_idx": 0,
        "lift_idx": lift,
        "pour_idx": pour,
        "terminal_idx": len(rows) - 1,
        "pour_heuristic_found": pour is not None,
        "terminal_z_ge_1": z[-1] >= 1.0,
        "same_index_file": True,
    }


def main() -> None:
    protocol = read(AF / "plan/PROTOCOL.json")
    training = read(AF / "models_deploy/TRAINING_COMPLETE.json")
    feas_manifest = read(AF / "models_deploy/feasibility/FEASIBILITY_MANIFEST.json")
    belief_manifest = read(AF / "models_deploy/belief/BELIEF_MANIFEST.json")
    feas_train = read(FEAS_TRAIN / "PROTOCOL.json")
    feas_train_results = read(FEAS_TRAIN / "TRAIN_RESULTS.json")
    fork = read(FEAS_FORK / "plan/PROTOCOL.json")
    af_final = read(AF / "FINAL_RESULTS.json")
    nom_final = read(NOM / "FINAL_RESULTS.json")
    fill_claim = read(FILL / "CLAIM.json")
    fill_results = read(FILL / "FILL_RESULTS.json")
    fig_manifest = read(FIG_BUILDER / "figB_dump_panel_manifest.json")
    utility_src = (AF / "max_force_utility.py").read_text()

    feas_hashes = []
    for ckpt in feas_manifest["checkpoints"]:
        path = Path(ckpt["path"])
        digest = sha(path)
        feas_hashes.append(
            {
                "path": str(path),
                "seed": ckpt["seed"],
                "selected_epoch": ckpt["selected_epoch"],
                "manifest_sha256": ckpt["sha256"],
                "on_disk_sha256": digest,
                "hash_match": digest == ckpt["sha256"],
            }
        )

    belief_hashes = []
    for ckpt in belief_manifest["checkpoints"]:
        local = AF / "models_deploy/belief" / Path(ckpt["path"]).name
        source = Path(ckpt["path"])
        local_sha = sha(local) if local.exists() else None
        source_sha = sha(source) if source.exists() else None
        belief_hashes.append(
            {
                "manifest_path": str(source),
                "deploy_copy": str(local),
                "seed": ckpt["seed"],
                "selected_epoch": ckpt["selected_epoch"],
                "manifest_sha256": ckpt["sha256"],
                "deploy_copy_sha256": local_sha,
                "source_sha256": source_sha,
                "deploy_matches_manifest": local_sha == ckpt["sha256"],
                "source_matches_manifest": source_sha == ckpt["sha256"],
            }
        )

    af_records = [load_af_record(p) for p in sorted((AF / "main_records").glob("*.json"))]
    nom_records = [load_nominal_record(p) for p in sorted((NOM / "main_records").glob("*.json"))]
    af_by_key = {r["key"]: r for r in af_records}
    nom_by_key = {r["key"]: r for r in nom_records}

    planned = []
    for seed in protocol["formal_policy_seeds"]:
        for mu in protocol["formal_frictions"]:
            planned.append(context_key(protocol["physical_root"], mu, seed))

    af_skips = []
    for row in af_final["prefix_skips"]:
        parts = parse_id_parts(row["id"])
        af_skips.append({**row, "key": context_key(200002, parts["mu"], parts["policy_seed"]), "reason": row["reason"]})
    nom_skips = []
    for row in nom_final["prefix_skips"]:
        parts = parse_id_parts(row["id"])
        nom_skips.append({**row, "key": context_key(200002, parts["mu"], parts["policy_seed"]), "reason": row["reason"]})

    matched_keys = sorted(set(af_by_key) & set(nom_by_key))
    af_only_keys = sorted(set(af_by_key) - set(nom_by_key))
    nom_only_keys = sorted(set(nom_by_key) - set(af_by_key))

    both_success = both_fail = af_only_success = nom_only_success = 0
    matched_rows = []
    for key in matched_keys:
        a, n = af_by_key[key], nom_by_key[key]
        cell = (
            "both_success" if a["success"] and n["success"]
            else "both_fail" if (not a["success"] and not n["success"])
            else "AF_only_success" if a["success"]
            else "Nominal_only_success"
        )
        if cell == "both_success":
            both_success += 1
        elif cell == "both_fail":
            both_fail += 1
        elif cell == "AF_only_success":
            af_only_success += 1
        else:
            nom_only_success += 1
        matched_rows.append(
            {
                "match_key_root": key[0],
                "match_key_mu": key[1],
                "match_key_policy_seed": key[2],
                "match_rule": "physical_root + object friction (3dp) + pi0 policy_seed; NOT identical snapshot",
                "af_id": a["id"],
                "nominal_id": n["id"],
                "handoff_sha_equal": a["handoff_state_sha256"] == n["handoff_state_sha256"],
                "first_chunk_sha_equal": a["first_chunk_sha256"] == n["first_chunk_sha256"],
                "cell": cell,
                "AF_success": a["success"],
                "Nominal_success": n["success"],
                "AF_official_final_check": a["official_final_check"],
                "Nominal_official_final_check": n["official_final_check"],
                "AF_F_star_N": a["F_star_N"],
                "AF_commanded_force_N": a["commanded_force_N"],
                "Nominal_commanded_force_N": n["commanded_force_N"] if n["commanded_force_N"] is not None else "--",
                "AF_force_setpoint_bilateral_n": a["force_setpoint_bilateral_n"],
                "Nominal_force_setpoint_bilateral_n": n["force_setpoint_bilateral_n"],
                "AF_measured_mean_squeeze_N": a["measured_mean_squeeze_N"],
                "Nominal_measured_mean_squeeze_N": n["measured_mean_squeeze_N"],
                "AF_measured_max_squeeze_N": a["measured_max_squeeze_N"],
                "Nominal_measured_max_squeeze_N": n["measured_max_squeeze_N"],
                "AF_release_actions": a["release_actions"],
                "Nominal_release_actions": n["release_actions"],
                "AF_physics_steps": a["physics_steps"],
                "Nominal_physics_steps": n["physics_steps"],
                "weight": 1,
                "persistent_pre_release_drop": "MISSING",
                "drop_field_present_AF": a["drop_field_in_result"],
                "drop_field_present_Nominal": n["drop_field_in_result"],
            }
        )

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    af_matched = [af_by_key[k] for k in matched_keys]
    nom_matched = [nom_by_key[k] for k in matched_keys]
    af_squeeze_missing = sum(1 for r in af_matched if r["measured_mean_squeeze_N"] is None)
    nom_squeeze_missing = sum(1 for r in nom_matched if r["measured_mean_squeeze_N"] is None)
    squeeze_defs_match = af_squeeze_missing == 0 and nom_squeeze_missing == 0
    mean_af_squeeze = mean([r["measured_mean_squeeze_N"] for r in af_matched])
    mean_nom_squeeze = mean([r["measured_mean_squeeze_N"] for r in nom_matched])
    load_reduction = (
        1.0 - mean_af_squeeze / mean_nom_squeeze
        if squeeze_defs_match and mean_nom_squeeze not in (None, 0)
        else "MISSING"
    )

    af_fstar = [r["F_star_N"] for r in af_records]
    overlay = {1: 1.25, 2: 1.00, 3: 0.75}
    fig_rows_spec = [
        ("Low friction", "liftclone_eu_fmax5_mu0.425_root200002_ps80200002", 0.425),
        ("Mid friction", "liftclone_eu_fmax5_mu0.575_root200002_ps80200002", 0.575),
        ("High friction", "liftclone_eu_fmax5_mu0.850_root200002_ps80200002", 0.850),
    ]
    rgb_rows = []
    for row_i, (band, context_id, mu) in enumerate(fig_rows_spec, 1):
        official = af_by_key[context_key(200002, mu, 80200002)]
        fig_result = read(FIG / "raw" / context_id / "job" / "FORMAL_CONTEXT_RESULT.json")
        fig_lock = read(FIG / "raw" / context_id / "job" / "PREACTION_SELECTION_LOCK.json")
        events = figure_events(context_id)
        panels = [p for p in fig_manifest if p["context"] == context_id]
        panels = sorted(panels, key=lambda p: p["column"])
        stages = {
            1: ("handoff", events["handoff_idx"]),
            2: ("lift", events["lift_idx"]),
            3: ("pour", events["pour_idx"]),
            4: ("terminal", events["terminal_idx"]),
        }
        same_rollout = len({p["context"] for p in panels}) == 1 and all(
            Path(p["source_npz"]).parent == FIG / "raw" / context_id / "frames" for p in panels
        )
        in_matched_19 = official["key"] in set(matched_keys)
        rgb_rows.append(
            {
                "figure_row": row_i,
                "row_label": band,
                "mu_object_side_authored": mu,
                "mu_is": "object-side authored contact friction on 063_tabletrashbin shape materials; not an inferred effective mu",
                "root": 200002,
                "episode_root": 200002,
                "pi0_seed": 80200002,
                "run_id_official": official["id"],
                "run_id_figure_only": context_id,
                "official_F_star_N": official["F_star_N"],
                "official_commanded_force_N": official["commanded_force_N"],
                "official_measured_mean_squeeze_N": official["measured_mean_squeeze_N"],
                "official_full_task_success": official["success"],
                "official_official_final_check": official["official_final_check"],
                "official_in_matched_19": in_matched_19,
                "figure_only_F_star_N": fig_result["af_decision"]["selected_force_N"],
                "figure_only_commanded_force_N": fig_result["outcomes"][0]["commanded_force_N"],
                "figure_only_measured_mean_squeeze_N": fig_result["outcomes"][0]["measured_mean_squeeze_N"],
                "figure_only_success": fig_result["outcomes"][0]["success"],
                "figure_only_handoff_sha256": fig_result["common_handoff_state_sha256"],
                "official_handoff_sha256": official["handoff_state_sha256"],
                "same_rollout_as_official_18_19": fig_result["common_handoff_state_sha256"] == official["handoff_state_sha256"],
                "current_figure_overlay_F_N": overlay[row_i],
                "overlay_matches_official_F": overlay[row_i] == official["F_star_N"],
                "overlay_matches_figure_only_F": overlay[row_i] == fig_result["af_decision"]["selected_force_N"],
                "four_panels_same_figure_only_rollout": same_rollout,
                "stage_rule": "height heuristic on saved deskbin_xyz, not logged handoff/lift/pour/terminal phases",
                "handoff_frame": events["handoff_idx"],
                "lift_frame": events["lift_idx"],
                "pour_frame": events["pour_idx"],
                "terminal_frame": events["terminal_idx"],
                "n_frames": events["n_frames"],
                "terminal_z": events["z_terminal"],
                "terminal_z_ge_1_for_check_success_bin_height": events["terminal_z_ge_1"],
                "persistent_pre_release_drop": "MISSING",
                "npz_handoff": str(FIG / "raw" / context_id / "frames" / f"{events['handoff_idx']:04d}.npz"),
                "npz_lift": str(FIG / "raw" / context_id / "frames" / f"{events['lift_idx']:04d}.npz"),
                "npz_pour": str(FIG / "raw" / context_id / "frames" / f"{events['pour_idx']:04d}.npz") if events["pour_idx"] is not None else "MISSING",
                "npz_terminal": str(FIG / "raw" / context_id / "frames" / f"{events['terminal_idx']:04d}.npz"),
                "exported_png_handoff": panels[0]["exported_frame"] if len(panels) == 4 else "MISSING",
                "exported_png_lift": panels[1]["exported_frame"] if len(panels) == 4 else "MISSING",
                "exported_png_pour": panels[2]["exported_frame"] if len(panels) == 4 else "MISSING",
                "exported_png_terminal": panels[3]["exported_frame"] if len(panels) == 4 else "MISSING",
                "figure_only_extra_rpc": "figure-only extra observation RPC after each native action",
            }
        )

    unmatched = []
    for key in af_only_keys:
        a = af_by_key[key]
        nom_skip = next((s for s in nom_skips if s["key"] == key), None)
        unmatched.append(
            {
                "side": "AF_executed_no_Nominal_outcome",
                "key": list(key),
                "af_id": a["id"],
                "AF_success": a["success"],
                "AF_F_star_N": a["F_star_N"],
                "reason": nom_skip["reason"] if nom_skip else "Nominal record absent",
                "nominal_prefix_skip": bool(nom_skip),
            }
        )
    for key in nom_only_keys:
        n = nom_by_key[key]
        af_skip = next((s for s in af_skips if s["key"] == key), None)
        unmatched.append(
            {
                "side": "Nominal_executed_no_AF_outcome",
                "key": list(key),
                "nominal_id": n["id"],
                "Nominal_success": n["success"],
                "reason": af_skip["reason"] if af_skip else "AF record absent",
                "af_prefix_skip": bool(af_skip),
            }
        )
    for s in af_skips:
        if s["key"] not in af_by_key and s["key"] not in nom_by_key:
            unmatched.append({"side": "both_prefix_skip_or_AF_skip_unpaired", "key": list(s["key"]), "reason": s["reason"], "id": s["id"]})

    paper_claims = {
        "48_contexts": {
            "status": "UNSUPPORTED_for_this_dump_run",
            "found_instead": {
                "planned_online_contexts": 24,
                "AF_executed": 22,
                "Nominal_executed": 20,
                "matched": 19,
            },
            "note": "48 appears in the older 4-task frozen-VLA paper draft (4 tasks x 4 roots x 3 frictions), not in this dump liftclone EU fmax5 run.",
        },
        "648_continuations_or_72_contexts": {
            "status": "UNSUPPORTED_for_this_dump_run",
            "feasibility_fork_v3": {"main_contexts": fork["main_contexts"], "main_rollouts_0p25_to_8": fork["main_rollouts"]},
            "feasibility_v3_fmax5_rows": feas_train["n_rows"],
            "note": "v3 fork is 24 contexts x 32 forces = 768 scripted continuations; fmax5 clip is 480 rows. Not 72/648.",
        },
        "432_108_108": {
            "status": "UNSUPPORTED_for_this_dump_run",
            "feasibility_v3_fmax5_split_rows": feas_train_results["n_rows"],
            "note": "Actual fmax5 split is TRAIN/VAL/TEST = 160/160/160 rows (8 mu x 20 forces per seed). Not 432/108/108.",
        },
        "all_bins_10g": {
            "status": "UNSUPPORTED",
            "logged": {
                "garbage_sphere_mass_in_env_source": 0.0001,
                "deskbin_runtime_mass_in_this_run_logs": "MISSING",
                "af_object_mass_kg_in_this_run": "not present in liftclone EU fmax5 records",
            },
            "note": "Do not write 10 g from this dump run. Deskbin mass is not recorded in the frozen online logs.",
        },
    }

    run_contract = {
        "version_locked": "AF_DUMP_FORMAL_LIFTCLONE_EU_FMAX5",
        "do_not_mix": ["maxf8 confirmation", "current v4 relabel development 24", "friction-isolation / unit-test / pre-pour diagnostics"],
        "run_directory_AF": str(AF),
        "run_directory_Nominal": str(NOM),
        "prefix_fill_directory": str(FILL),
        "prefix_fill_excluded_from_official_18_19": bool(fill_claim.get("official_paired_18_19_not_modified")),
        "protocol": protocol,
        "feasibility": {
            "training_experiment": str(FEAS_TRAIN),
            "training_source_rows": str(FEAS_FORK),
            "architecture": feas_train_results["architecture"],
            "label_target": feas_manifest["label_target"],
            "force_feature_normalization_N": feas_manifest["force_feature_normalization_N"],
            "belief_unused_at_train_BCE": fork["belief"],
            "checkpoints": feas_hashes,
            "all_checkpoint_hashes_match": all(x["hash_match"] for x in feas_hashes),
        },
        "belief": {
            "description": protocol["belief"],
            "source": training["belief_source"],
            "normalization": {
                "fit_split": belief_manifest["normalization"]["fit_split"],
                "fit_roots": belief_manifest["normalization"]["fit_roots"],
                "feature_dim": belief_manifest["feature_dim"],
                "feature_schema_id": belief_manifest["feature_schema_id"],
            },
            "checkpoints": belief_hashes,
            "all_deploy_hashes_match": all(x["deploy_matches_manifest"] for x in belief_hashes),
        },
        "P4_and_query": {
            "probe_for_belief": protocol["probe_for_belief"],
            "probe_for_handoff": protocol["probe_for_handoff"],
            "established_grasp_N": protocol["established_grasp_N"],
            "established_grasp_rule": protocol["established_grasp_rule"],
            "dump_shear": "4 N / 0.012 m after 12 N cap",
            "example_from_first_AF": {
                "scope": af_records[0]["p4_protocol_scope"],
                "grasp_force_N": af_records[0]["grasp_force_N"],
                "query_force_N": af_records[0]["query_force_N"],
                "query_displacement_m": af_records[0]["query_displacement_m"],
            },
        },
        "EU": {
            "formula": "U(F)=p*(Fmax-F)/Fmax-(1-p)",
            "source_file": str(AF / "max_force_utility.py"),
            "source_contains_formula": "p * (maximum - f) / maximum - (1.0 - p)" in utility_src,
            "executed_selector": protocol["executed_selector"],
            "F_max_N": protocol["utility_normalization_N"],
            "grid_min": protocol["force_support"][0],
            "grid_max": protocol["force_support"][1],
            "grid_step": protocol["planner_grid_step"],
            "argmax_p_status": "diagnostic counterfactual only; 22/22 official AF records have argmax_p=5.0",
        },
        "remainder": {
            "training_remainder": "scripted snapshot-fork remainder (liftstyle v3, not pi0)",
            "deployment_remainder": "frozen pi0 native remainder only",
            "mq_training": "scripted 8-step EEF hold chunk (hold_chunk / 8x64)",
            "mq_deployment": "ONLINE_VLA_ACTION_CHUNK (first pi0 chunk, 8x64)",
            "example_mq_len": af_records[0]["mq_feature_len"],
            "example_mq_width": af_records[0]["mq_feature_width"],
        },
        "evaluation_set": {
            "task": "dump_bin_bigbin",
            "physical_root": 200002,
            "frictions": protocol["formal_frictions"],
            "policy_seeds": protocol["formal_policy_seeds"],
            "planned_contexts": 24,
            "AF_executed": len(af_records),
            "Nominal_executed": len(nom_records),
            "AF_prefix_skips": len(af_skips),
            "Nominal_prefix_skips": len(nom_skips),
            "matched_by_root_mu_seed": len(matched_keys),
            "handoff_sha_matches_in_matched": sum(1 for r in matched_rows if r["handoff_sha_equal"]),
            "first_chunk_sha_matches_in_matched": sum(1 for r in matched_rows if r["first_chunk_sha_equal"]),
        },
        "force_definition": {
            "AF_F": "bilateral squeeze force setpoint / force_setpoint_bilateral_n; not doubled; not an actuator-limit-only quantity",
            "Nominal_F": "none; commanded_force_N is null; native VLA gripper pass-through",
            "measured_squeeze": "mean over ALL remainder physics_trace steps of contact.measured_squeeze_n; no bilateral-contact filter in the official summary",
            "do_not": "do not multiply or divide by two; do not substitute commanded F for measured squeeze",
        },
        "official_success": {
            "source": "RoboTwin dump_bin_bigbin.check_success",
            "rule": "deskbin z>=1 and all 5 garbage spheres have z in [0.13, 0.25]; must equal official_final_check",
            "all_AF_success_equals_official": all(r["success_equals_official"] for r in af_records),
            "all_Nominal_success_equals_official": all(r["success_equals_official"] for r in nom_records),
        },
        "figure_B_rgb": {
            "rgb_source_run": str(FIG),
            "rgb_is_official_18_19_rollout": False,
            "current_overlay_forces": [1.25, 1.00, 0.75],
            "official_ps80200002_forces": [2.1, 2.0, 2.4],
            "figure_only_ps80200002_forces": [r["figure_only_F_star_N"] for r in rgb_rows],
            "mid_row_in_matched_19": False,
        },
        "paper_number_audit": paper_claims,
        "AF_all_22_Fstar_counts": dict(Counter(f"{x:.2f}" if x != int(x) else str(x) for x in af_fstar)),
        "AF_all_22_success": f"{sum(r['success'] for r in af_records)}/{len(af_records)}",
        "AF_all_22_mean_F": mean(af_fstar),
        "AF_all_22_mean_squeeze": mean([r["measured_mean_squeeze_N"] for r in af_records]),
    }

    matched_summary = [
        {
            "subset": "matched_19",
            "method": "ActiveForcing EU fmax5",
            "n": 19,
            "success_n": sum(r["success"] for r in af_matched),
            "success_pct": 100.0 * sum(r["success"] for r in af_matched) / 19,
            "drop_n": "MISSING",
            "drop_pct": "MISSING",
            "mean_commanded_F_N": mean([r["commanded_force_N"] for r in af_matched]),
            "mean_measured_squeeze_N": mean_af_squeeze,
            "squeeze_window": "all remainder physics steps; contact.measured_squeeze_n; unweighted mean of 19 contexts",
            "weighting": "one executed remainder per context",
            "missing_squeeze_telemetry": af_squeeze_missing,
        },
        {
            "subset": "matched_19",
            "method": "Nominal Frozen VLA",
            "n": 19,
            "success_n": sum(r["success"] for r in nom_matched),
            "success_pct": 100.0 * sum(r["success"] for r in nom_matched) / 19,
            "drop_n": "MISSING",
            "drop_pct": "MISSING",
            "mean_commanded_F_N": "--",
            "mean_measured_squeeze_N": mean_nom_squeeze,
            "squeeze_window": "all remainder physics steps; contact.measured_squeeze_n; unweighted mean of 19 contexts",
            "weighting": "one executed remainder per context",
            "missing_squeeze_telemetry": nom_squeeze_missing,
        },
        {
            "subset": "AF_all_executed_22_NOT_the_paired_table",
            "method": "ActiveForcing EU fmax5",
            "n": 22,
            "success_n": sum(r["success"] for r in af_records),
            "success_pct": 100.0 * sum(r["success"] for r in af_records) / 22,
            "drop_n": "MISSING",
            "drop_pct": "MISSING",
            "mean_commanded_F_N": mean(af_fstar),
            "mean_measured_squeeze_N": mean([r["measured_mean_squeeze_N"] for r in af_records]),
            "note": "Do not compare this 22-row mean to Nominal's 19-row mean.",
        },
        {
            "subset": "Nominal_all_executed_20_NOT_the_paired_table",
            "method": "Nominal Frozen VLA",
            "n": 20,
            "success_n": sum(r["success"] for r in nom_records),
            "success_pct": 100.0 * sum(r["success"] for r in nom_records) / 20,
            "mean_commanded_F_N": "--",
            "mean_measured_squeeze_N": mean([r["measured_mean_squeeze_N"] for r in nom_records]),
            "note": "Includes the one Nominal-only context mu0.575 ps80200009.",
        },
    ]

    paper_table = [
        {
            "Method": "ActiveForcing (EU, Fmax=5 N)",
            "Success (%)": f"{sum(r['success'] for r in af_matched)}/19 ({100.0 * sum(r['success'] for r in af_matched) / 19:.1f})",
            "Drop (%)": "MISSING",
            "mean F": f"{mean([r['commanded_force_N'] for r in af_matched]):.3f} N",
            "mean measured squeeze": f"{mean_af_squeeze:.3f} N",
        },
        {
            "Method": "Nominal Frozen VLA",
            "Success (%)": f"{sum(r['success'] for r in nom_matched)}/19 ({100.0 * sum(r['success'] for r in nom_matched) / 19:.1f})",
            "Drop (%)": "MISSING",
            "mean F": "--",
            "mean measured squeeze": f"{mean_nom_squeeze:.3f} N",
        },
    ]

    contingency = {
        "matched_n": 19,
        "both_success": both_success,
        "AF_only_success": af_only_success,
        "Nominal_only_success": nom_only_success,
        "both_fail": both_fail,
        "AF_success_on_matched": f"{both_success + af_only_success}/19",
        "Nominal_success_on_matched": f"{both_success + nom_only_success}/19",
        "both_success_is_not_AF_rate": True,
        "measured_load_reduction_matched_19": load_reduction,
        "load_reduction_formula": "1 - mean_squeeze_AF / mean_squeeze_Nominal on the same 19 contexts",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_contract.json").write_text(json.dumps(run_contract, indent=2, sort_keys=True, default=str) + "\n")
    (OUT / "unmatched_records.json").write_text(json.dumps(unmatched, indent=2, default=str) + "\n")
    (OUT / "contingency_matched_19.json").write_text(json.dumps(contingency, indent=2) + "\n")
    (OUT / "paper_table.json").write_text(json.dumps(paper_table, indent=2) + "\n")

    with (OUT / "figure_rgb_provenance.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rgb_rows[0].keys()))
        w.writeheader()
        w.writerows(rgb_rows)

    with (OUT / "matched_context_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(matched_rows[0].keys()))
        w.writeheader()
        w.writerows(matched_rows)

    with (OUT / "matched_summary.csv").open("w", newline="") as f:
        keys = sorted({k for row in matched_summary for k in row})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(matched_summary)

    with (OUT / "paper_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Method", "Success (%)", "Drop (%)", "mean F", "mean measured squeeze"])
        w.writeheader()
        w.writerows(paper_table)

    print(json.dumps({
        "out": str(OUT),
        "AF": len(af_records),
        "Nominal": len(nom_records),
        "matched": len(matched_keys),
        "contingency": contingency,
        "AF22": run_contract["AF_all_22_success"],
        "feas_hash_ok": run_contract["feasibility"]["all_checkpoint_hashes_match"],
        "belief_hash_ok": run_contract["belief"]["all_deploy_hashes_match"],
        "fig_official_F": [r["official_F_star_N"] for r in rgb_rows],
        "fig_only_F": [r["figure_only_F_star_N"] for r in rgb_rows],
        "fig_same_as_official": [r["same_rollout_as_official_18_19"] for r in rgb_rows],
        "mid_in_matched19": [r["official_in_matched_19"] for r in rgb_rows],
        "handoff_sha_matches": run_contract["evaluation_set"]["handoff_sha_matches_in_matched"],
        "AF_fail_ids": [r["id"] for r in af_records if not r["success"]],
        "matched_AF_fail": [r["af_id"] for r in matched_rows if not r["AF_success"]],
        "load_reduction": load_reduction,
        "mean_F_matched": mean([r["commanded_force_N"] for r in af_matched]),
        "mean_squeeze_AF": mean_af_squeeze,
        "mean_squeeze_Nom": mean_nom_squeeze,
    }, indent=2))


if __name__ == "__main__":
    main()
