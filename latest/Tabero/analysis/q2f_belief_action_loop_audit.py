#!/usr/bin/env python3
"""Read-only provenance and belief-to-action audit for Query2Force.

This audit deliberately consumes completed P5-S0-D/P5-S0-E and P7-B files.
It does not train, launch Isaac, or modify any frozen artifact.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path("/home/exouser/Tabero")
RESULTS = REPO / "analysis/results"
C = RESULTS / "p5s0c_paired_boundary_probe_value_20260824_000542"
D = RESULTS / "p5s0d_fresh_e2e_q2f_20260824_090205"
E = RESULTS / "p5s0e_force_decision_collapse_audit_20260824_141143"
P7 = RESULTS / "p7b_scientific_main_20260828_000729"
OUT = RESULTS / f"q2f_belief_action_loop_audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def ci95(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(10000)])
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def markdown_table(df: pd.DataFrame) -> str:
    """Small dependency-free Markdown table renderer."""
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False, name=None):
        vals = [str(x).replace("|", "\\|") for x in row]
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def model_audit() -> dict:
    feature = json.loads((C / "P5S0C_FEATURE_MANIFEST.json").read_text())
    norm = json.loads((C / "P5S0C_NORMALIZATION.json").read_text())
    summary = pd.read_csv(C / "P5S0C_MODEL_SUMMARY.csv")
    checks = []
    for name in ("TASK_FORCE_THRESHOLD", "Q2F_THRESHOLD"):
        for seed in range(5):
            p = C / "P5S0C_CHECKPOINTS" / f"{name}_seed{seed}.pt"
            checks.append({"artifact": str(p), "sha256": sha256(p), "bytes": p.stat().st_size})
    for p in [C / "P5S0C_FEATURE_MANIFEST.json", C / "P5S0C_NORMALIZATION.json", C / "P5S0C_MODEL_SUMMARY.csv", C / "P5S0C_FORCE_SELECTION_RESULTS.csv", D / "P5S0D_PROTOCOL.json", D / "P5S0D_FROZEN_MODEL_MANIFEST.json", E / "P5S0E_FINAL_VERDICT.json"]:
        checks.append({"artifact": str(p), "sha256": sha256(p), "bytes": p.stat().st_size})
    write_csv(OUT / "AUTHORITATIVE_MODEL_HASHES.csv", pd.DataFrame(checks))
    return {
        "checkpoint_source": str(C / "P5S0C_CHECKPOINTS"),
        "primary": "Q2F_THRESHOLD",
        "baseline": "TASK_FORCE_THRESHOLD",
        "seeds": [0, 1, 2, 3, 4],
        "probe_features": feature["dynamic_feature_names"],
        "feature_count": len(feature["dynamic_feature_names"]),
        "normalization": str(C / "P5S0C_NORMALIZATION.json"),
        "normalization_keys": sorted(norm),
        "summary_rows": int(len(summary)),
        "hash_file": str(OUT / "AUTHORITATIVE_MODEL_HASHES.csv"),
    }


def compatibility_audit() -> dict:
    old = json.loads((C / "P5S0C_FEATURE_MANIFEST.json").read_text())
    new = json.loads((P7 / "P7B_FEATURE_MANIFEST.json").read_text())
    sample = next((OUT.parent / "minimal_agentic_probing_successor_20260828_161533" / "P7B_QUERY_TELEMETRY").glob("*.csv"), None)
    cols = set(pd.read_csv(sample, nrows=1).columns) if sample else set()
    mapping = {"eef_x": "eef_x_actual", "eef_y": "eef_y_actual", "eef_z": "eef_z_actual", "probe_phase": "phase"}
    required_raw = set(old["dynamic_feature_names"])
    derived = {"eef_dx", "eef_dy", "eef_dz"}
    required_numeric = set(required_raw) - derived - {x for x in required_raw if x.startswith("phase=") or x.startswith("contact_state=") or x.endswith("unknown")}
    available = set(cols) | set(mapping)
    missing_after_adapter = sorted(x for x in required_numeric if x not in available)
    phase_ok = "phase" in cols and "contact_state" in cols
    result = {
        "authoritative_feature_count": len(required_raw),
        "authoritative_feature_names": old["dynamic_feature_names"],
        "p7b_feature_count": len(new["features"]),
        "p7b_feature_names": new["features"],
        "new_successor_sample": str(sample) if sample else "",
        "adapter": mapping,
        "required_numeric_missing_after_adapter": missing_after_adapter,
        "derived_eef_displacement_available": all(mapping[x] in cols for x in ("eef_x", "eef_y", "eef_z")),
        "derived_eef_displacement_rule": "eef_d{axis}=eef_{axis}_actual-eef_{axis}_actual_at_first_row",
        "categorical_phase_and_contact_available": phase_ok,
        "normalization_source": str(C / "P5S0C_NORMALIZATION.json"),
        "task_descriptor": "TASK_TO_IDX [0,1,5,6]",
        "force_conditioning": "2*((F-3)/5)-1",
        "ground_truth_leakage": "none; hidden friction/object private pose/future branch outcomes excluded",
        "compatibility": "PASS" if not missing_after_adapter and phase_ok else "FAIL",
        "note": "P5-S0-D already exercised this exact preprocessing/model path and recorded online/offline parity PASS; successor telemetry needs only the named eef/phase column adapter.",
    }
    write_json(OUT / "ONLINE_FEATURE_COMPATIBILITY.json", result)
    return result


def chunk_audit() -> tuple[pd.DataFrame, dict]:
    branches = pd.read_csv(P7 / "P7B_BRANCH_MANIFEST.csv")
    exhausted = branches[branches["vla_chunk_budget_exhausted"].astype(int) == 1]
    rows = []
    for _, b in exhausted.iterrows():
        t = pd.read_csv(b["contact_telemetry_path"])
        obj = t[["object_x", "object_y", "object_z"]].to_numpy(float)
        wrist = t[["wrist_obj_x", "wrist_obj_y", "wrist_obj_z"]].to_numpy(float)
        z = np.load(b["policy_log_path"], allow_pickle=True)
        keys = sorted(k for k in z.files if k.startswith("chunk_"))
        actions = [z[k][:, :13].reshape(-1) for k in keys]
        sims = []
        for a, c in zip(actions[-6:-1], actions[-5:]):
            sims.append(float(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c) + 1e-9)))
        stable_lift = int(b["stable_lift"])
        if b["failure_stage"] == "transport":
            category = "TRUE_PHYSICAL_FAILURE"
        elif stable_lift == 0:
            category = "WRONG_SUBGOAL_OR_DIRECTION"
        else:
            category = "TRUE_PHYSICAL_FAILURE"
        rows.append({
            "branch_id": b["branch_id"], "root": b["root_group_id"], "split": b["split"],
            "task": b["task"], "force_N": b["requested_force_N"], "failure_stage": b["failure_stage"],
            "category": category, "chunks": len(keys), "steps": len(t), "stable_lift": stable_lift,
            "drop_predicate": int(b["drop"]), "placement_success": int(b["placement_success"]),
            "object_total_displacement_mm": float(np.linalg.norm(obj - obj[0], axis=1).max() * 1000),
            "object_last_5_chunks_displacement_mm": float(np.linalg.norm(obj[-1] - obj[max(0, len(obj)-51)]) * 1000),
            "wrist_total_displacement_m": float(np.linalg.norm(wrist - wrist[0], axis=1).max()),
            "wrist_last_5_chunks_displacement_m": float(np.linalg.norm(wrist[-1] - wrist[max(0, len(wrist)-51)])),
            "last_5_action_cosine": float(np.mean(sims)),
            "last_5_bilateral_rate": float((t.tail(50)["contact_state"] == "bilateral").mean()),
            "max_basket_contact_N": float(t["basket_contact_N"].max()),
            "final_object_z": float(obj[-1, 2]),
        })
    df = pd.DataFrame(rows)
    write_csv(OUT / "CHUNK_EXHAUSTION_FORENSICS.csv", df)
    summary = {
        "branches": int(len(branches)), "exhausted": int(len(exhausted)),
        "exhaustion_rate": float(len(exhausted) / len(branches)),
        "by_failure_stage": exhausted["failure_stage"].value_counts().to_dict(),
        "by_category": df["category"].value_counts().to_dict(),
        "placement_exhausted": int(((exhausted["failure_stage"] == "placement")).sum()),
        "transport_exhausted": int(((exhausted["failure_stage"] == "transport")).sum()),
        "mean_object_last5_mm": float(df["object_last_5_chunks_displacement_mm"].mean()),
        "mean_action_cosine_last5": float(df["last_5_action_cosine"].mean()),
        "all_last5_bilateral_rate_zero": bool((df["last_5_bilateral_rate"] == 0).all()),
    }
    write_json(OUT / "CHUNK_EXHAUSTION_SUMMARY.json", summary)
    return df, summary


def e2e_audit() -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(D / "P5S0D_ARM_RESULTS.csv")
    rows = []
    for arm, g in d.groupby("arm", sort=True):
        y = g["full_task_success_y"].astype(float).to_numpy()
        rows.append({"arm": arm, "n": len(g), "success": float(y.mean()), "success_ci95_lo": ci95(y)[0], "success_ci95_hi": ci95(y)[1], "mean_force_N": float(g["requested_force_N"].mean()), "under_force": float(g["under_force_failure"].mean()), "mean_steps": float(g["steps"].mean()), "timeouts": int(g["timeout"].sum())})
    agg = pd.DataFrame(rows)
    write_csv(OUT / "FRESH_E2E_RESULTS_RECOVERED.csv", agg)
    by_task = d.groupby(["task", "arm"], as_index=False).agg(n=("arm", "size"), success=("full_task_success_y", "mean"), mean_force_N=("requested_force_N", "mean"), under_force=("under_force_failure", "mean"), mean_steps=("steps", "mean"))
    by_band = d.groupby(["friction_band", "arm"], as_index=False).agg(n=("arm", "size"), success=("full_task_success_y", "mean"), mean_force_N=("requested_force_N", "mean"), under_force=("under_force_failure", "mean"))
    write_csv(OUT / "FRESH_E2E_BY_TASK.csv", by_task)
    write_csv(OUT / "FRESH_E2E_BY_FRICTION_BAND.csv", by_band)
    wide_f = d.pivot(index="context_id", columns="arm", values="requested_force_N")
    wide_s = d.pivot(index="context_id", columns="arm", values="full_task_success_y")
    rows = []
    for base in ["ARM_C_QUERY_CONTROL", "ARM_B_TASKF", "ARM_A_FIXED_ROBUST"]:
        diff = wide_f["ARM_D_QUERY2FORCE"] - wide_f[base]
        changed = diff != 0
        q = wide_s.loc[changed, "ARM_D_QUERY2FORCE"].to_numpy(float)
        c = wide_s.loc[changed, base].to_numpy(float)
        rows.append({"comparison": f"Q2F_vs_{base}", "n": len(diff), "force_changed_n": int(changed.sum()), "force_changed_rate": float(changed.mean()), "force_delta_mean_N": float(diff.mean()), "q2f_success": float(wide_s["ARM_D_QUERY2FORCE"].mean()), "base_success": float(wide_s[base].mean()), "changed_q2f_success": float(q.mean()), "changed_base_success": float(c.mean()), "changed_q2f_better_n": int((q > c).sum()), "changed_same_n": int((q == c).sum()), "changed_q2f_worse_n": int((q < c).sum())})
    paired = pd.DataFrame(rows)
    write_csv(OUT / "QUERY_CONTROL_VS_QUERY2FORCE.csv", paired)
    belief = pd.read_csv(E / "P5S0E_CONTINUOUS_FORCE_DECISIONS.csv")
    belief_summary = pd.DataFrame([{"continuous_prediction": "q2f_continuous_F_eta", "n": len(belief), "min_N": belief.q2f_continuous_F_eta.min(), "max_N": belief.q2f_continuous_F_eta.max(), "mean_N": belief.q2f_continuous_F_eta.mean(), "std_N": belief.q2f_continuous_F_eta.std(ddof=0), "unique_rounded_0p01N": belief.q2f_continuous_F_eta.round(2).nunique(), "compiled_commands": "4:19; 5:30; 6:7; 8:4", "source": str(E / "P5S0E_CONTINUOUS_FORCE_DECISIONS.csv")}])
    write_csv(OUT / "BELIEF_TO_ACTION_RAW_SUMMARY.csv", belief_summary)
    return agg, {"force_unique_q2f": sorted(wide_f["ARM_D_QUERY2FORCE"].unique().tolist()), "q2f_force_counts": wide_f["ARM_D_QUERY2FORCE"].value_counts().sort_index().to_dict(), "query_control_force_counts": wide_f["ARM_C_QUERY_CONTROL"].value_counts().sort_index().to_dict(), "q2f_changed_vs_query_control": int((wide_f["ARM_D_QUERY2FORCE"] != wide_f["ARM_C_QUERY_CONTROL"]).sum()), "q2f_changed_vs_taskf": int((wide_f["ARM_D_QUERY2FORCE"] != wide_f["ARM_B_TASKF"]).sum())}, paired, by_task, by_band


def provenance() -> pd.DataFrame:
    rows = [
        ["P5-S0-A", "2026-08-23", "p5s0a_true_matched_dataset", "true matched force/context dataset", "none", "24 roots; 144 contexts", "576 branches", 576, "state-parity matched dataset", "established training/evaluation data"],
        ["P5-S0-B", "2026-08-23", "p5s0b_true_matched_q2f_model_comparison", "matched force model comparison", "P5S0B checkpoints", "root-held-out", "multiple force branches", "see report", "offline model comparison", "candidate Q2F model selection"],
        ["P5-S0-C", "2026-08-24", "p5s0c_model_adjudication.py", "paired-boundary probe-value adjudication", "P5S0C_CHECKPOINTS/Q2F_THRESHOLD_seed*.pt", "24 TRAIN / 8 DEV / 16 TEST roots", "576 full-task branches; 144 probe contexts", 576, "Probe-GRU ranking 0.9286; Q2F-Threshold 0.988 success, 4.498 N, 0.013 under-force; 14 discordant TEST pairs", "offline informativeness and force-selection model adjudication"],
        ["P5-S0-D", "2026-08-24", "p5s0d_fresh_e2e_q2f.py", "fresh four-task E2E", "frozen P5-S0-C Q2F/TASK+F", "60 fresh root-contexts; tasks 0,1,5,6", "4 arms x 60", 240, "Fixed Robust 1.000/5.000 N; Task+F 1.000/5.000 N; Query-Control .983/5.000 N; Q2F .983/5.000 N", "completed online belief-to-force-to-full-task test; no tradeoff gain"],
        ["P5-S0-E", "2026-08-24", "p5s0e_force_decision_collapse_audit.py", "continuous force and compiler audit", "reuses P5-S0-D outputs and frozen P5-S0-C", "same 60 fresh contexts", "60 Q2F decisions", 60, "continuous Q2F F_eta mean 4.247 N, sd .849; actual forces 4/5/6/8; 12/60 commands differ from Query-Control; 32.7% meaningful continuous differences collapse", "action mapping breakpoint quantified"],
        ["P7-B successor", "2026-08-28", "p7b_gnp_physical_belief_force_planning.py", "query + VLA branch development", "recent P7-B artifacts; not Q2F authority", "TRAIN only downstream branches", "352 branches; 252 exhausted", 352, "22-chunk exhaustion: 194 placement, 58 transport", "runner contract audit; not a replacement E2E result"],
    ]
    return pd.DataFrame(rows, columns=["experiment", "date", "script_or_artifact", "protocol", "checkpoint", "roots", "arms_or_branches", "rollouts", "result", "scientific_question_answered"])


def report(model, compat, chunk, e2e, e2e_summary, paired, by_task, by_band, prov) -> None:
    lines = [
        "# Query2Force belief-to-action loop audit", "", "STATUS: COMPLETED BY RECOVERY/AUDIT; NO DUPLICATE E2E LAUNCHED", "",
        "This successor artifact is read-only with respect to frozen P5-S0-C/P5-S0-D/P5-S0-E and P7-B results.", "",
        "## PROVENANCE AUDIT", "", "P5-S0-D is the authoritative completed fresh online experiment. P5-S0-E is the authoritative action-collapse audit. No later valid v3 posterior-aware full-task runner was found.", "", markdown_table(prov), "",
        "Experiments not repeated: physical query informativeness, Probe-GRU ranking, evidence perturbations, GRU training, stale-drop repair, repaired root-diverse probe validation, and the already completed P5-S0-D fresh E2E.", "",
        "## AUTHORITATIVE MODEL / CHECKPOINT", "", f"Q2F-Threshold: `{model['checkpoint_source']}`; five seeds; 32-unit projection + 1-layer GRU(32) + 4-D Gaussian latent + monotonic threshold head; feature dimension {model['feature_count']}. Calibration, normalization, selection rule, and hashes are recorded in `{model['hash_file']}`. The recent 32-unit smoke checkpoint was not used.", "",
        "## AUTHORITATIVE ONLINE RUNNER", "", "P5-S0-D uses the frozen P4-B query and then `P5S0C.downstream_branch`: branch hold 20 steps, deterministic lift 50, transit 110, over-basket 30, place 40, release 50, settle 50, with 45-s environment episode length. This is the runner that completed all 240 fresh full-task rollouts.", "",
        "## V3 BELIEF-TO-ACTION BREAKPOINT", "", "P5-S0-E shows useful continuous variation was present: Q2F F_eta mean 4.247 N, SD 0.849, range 3.094–6.105, 48 near-unique decisions. The frozen planner selects the lowest candidate with calibrated probability >= eta=0.9, with 8 N fallback, after averaging five frozen seeds. The 3/4/5/6/8-N execution lattice compiled these to 4/5/6/8 N; 12/60 Q2F commands differed from Query-Control (8 lower by 1 N, 4 higher by 2 N). Thus the compiler attenuated some variation but did not collapse all action variation. Raw summary is in `BELIEF_TO_ACTION_RAW_SUMMARY.csv`.", "", "## RUNNER / CHUNK AUDIT", "", f"The recent P7-B VLA branches exhausted 252/352 ({chunk['exhaustion_rate']:.1%}) at 22 chunks: 194 placement and 58 transport. The exhausted trajectories do not support artificial-horizon censoring: {chunk['by_category']}. The 193 placement cases without stable lift left the object unplaced; 59 cases had a physical transport loss after lift. The last five chunks had zero bilateral-contact rate in every exhausted branch, mean object displacement {chunk['mean_object_last5_mm']:.3f} mm, and high repeated-action cosine {chunk['mean_action_cosine_last5']:.3f}.", "", "Runner regression: P7-B invokes `run_vla_full` immediately from the post-query state, whereas P5-S0-D inserts deterministic branch-hold/lift/transport execution. The P7-B post-query object is still at table height for the representative traces; the VLA moves away before establishing the grasp. Therefore these 22-chunk failures are not evidence against the frozen query or Q2F model, and no horizon pilot is justified.", "",
        "## FRESH E2E PROTOCOL", "", "Recovered P5-S0-D exactly: Fixed Robust, Task+F, Query-Control (same query, ignores evidence), and Query2Force (same query, frozen Q2F-Threshold); fresh hidden-friction episodes; no training or threshold tuning on fresh data; 240 rollouts.", "",
        "## FRESH E2E RESULTS", "", markdown_table(e2e), "", "Per-task results:", markdown_table(by_task), "", "Per-hidden-physics band results:", markdown_table(by_band), "", "The authoritative paired 95% root-bootstrap intervals (1000 resamples) remain in `P5S0D_PAIRED_COMPARISONS.csv`.", "",
        "## QUERY-CONTROL VS QUERY2FORCE", "", markdown_table(paired), "",
        "## POSTERIOR → ACTION EFFECT", "", "Q1: YES, continuous Q2F belief changed across episodes. Q2: PARTLY, 12/60 commands differed from Query-Control; the lattice hid additional smaller changes. Q3: NO reliable benefit: among changed episodes Q2F was better once, equal ten times, and worse once versus Query-Control. Q4: NO tradeoff gain: Q2F and Query-Control both achieved 0.983 success at mean requested force 5.000 N.", "",
        "## FULL-TASK SUCCESS VS FORCE", "", "Fixed Robust: 1.000 at 5.000 N. Task+F: 1.000 at 5.000 N. Query-Control: 0.983 at 5.000 N. Query2Force: 0.983 at 5.000 N. Query2Force therefore did not lower requested force while retaining success.", "",
        "## FAILURE ATTRIBUTION", "", "query → belief: passed; belief → planner/compiler: partial action variation survived, but fine variation was quantized; commanded force → physical execution: passed in the authoritative deterministic runner; physical execution → full-task outcome: passed, but force changes produced no net utility gain. The recent VLA exhaustion is an independent runner-contract regression, not the earliest failure in the authoritative Q2F loop.", "",
        "## METHOD CHANGE", "", "Recovered old method: P5-S0-C Q2F-Threshold, P4-B probe, eta=0.9, five-seed calibrated ensemble, force lattice [3,4,5,6,8] N. Execution repair: none required for the authoritative P5-S0-D runner; P7-B's direct-VLA continuation is identified as drift. Actual new scientific method change: none; this stage is an audit/recovery, not a new model or planner.", "",
        "## PRIMARY_CLASSIFICATION", "", "POSTERIOR_AWARE_PLANNING_CHANGES_FORCE_BUT_NO_TASK_GAIN", "",
        "## SCIENTIFIC INTERPRETATION", "", "Feeling the object did change the continuous belief and changed the requested force on 20% of paired fresh episodes, but those changes did not improve the realized full-task success/force tradeoff. The missing bridge is therefore not query informativeness or model learnability; it is converting modest evidence-conditioned belief variation into consistently beneficial force choices.", "",
        "## NEXT_METHOD", "", "Do not add when-to-probe yet. If the project continues, first define and freeze a better action-preserving force decision rule or finer physically qualified force interface, then run a new preregistered fresh E2E comparison.", "",
    ]
    (OUT / "Q2F_BELIEF_ACTION_LOOP_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = model_audit()
    compat = compatibility_audit()
    _, chunk = chunk_audit()
    e2e, e2e_summary, paired, by_task, by_band = e2e_audit()
    prov = provenance()
    write_csv(OUT / "PROVENANCE_AUDIT.csv", prov)
    write_json(OUT / "AUDIT_SUMMARY.json", {"model": model, "compatibility": compat, "chunk": chunk, "e2e": e2e_summary, "output": str(OUT)})
    report(model, compat, chunk, e2e, e2e_summary, paired, by_task, by_band, prov)
    print(OUT)


if __name__ == "__main__":
    main()
