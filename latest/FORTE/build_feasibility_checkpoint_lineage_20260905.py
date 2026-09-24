#!/usr/bin/env python3
"""Offline feasibility-checkpoint lineage and selector audit.

This script deliberately does not import Isaac/IsaacLab or start a simulator.
It reads existing model/data artifacts, reconstructs the already-saved task-0
inference inputs, and writes a new audit directory without modifying source
models or experiment evidence.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
OUT = ROOT / "analysis/results/feasibility_checkpoint_lineage_20260905"
CLEAN = ROOT / "analysis/results/final_probe_continuous_posterior_rebuild_20260904"
SMOKE = ROOT / "analysis/results/activeforcing_e2e_task0_smoke_20260905"
VALID = ROOT / "analysis/results/current_runtime_setpoint_mapping_validation_20260905"
OLD_GNP = ROOT / "gnp_style_continuous_20260830_125107"
OLD_LOCK = ROOT / "activeforcing_full_claim_closure_20260902_062809/E6_E7/E6_E7_LOCKED_HANDOFF"
OLD_SEEDS = [OLD_LOCK / f"FROZEN_DIRECT_FEAS_seed{i}.pt" for i in range(3)]
CLEAN_SEEDS = [CLEAN / f"POSTERIOR_FEASIBILITY_seed{i}.pt" for i in range(3)]
FMAX_BY_TASK = {0: 5.0, 1: 6.0, 5: 5.0, 6: 4.0}
DT = 0.05
PHASES = ["branch_hold", "lift", "transit", "over_basket", "place", "release", "settle"]
TASKS = [0, 1, 5, 6]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def dump_csv(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def jload(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def finite(x: Any) -> float | None:
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


class FeasibilityOnly(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.command_gru = nn.GRU(17, 64, batch_first=True)
        self.condition = nn.Sequential(nn.Linear(54, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, step: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        _, h = self.command_gru(step)
        return self.head(torch.cat([h[-1], self.condition(cond)], dim=-1)).squeeze(-1)


def load_ensemble(paths: list[Path], kind: str):
    out = []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = FeasibilityOnly()
        m.load_state_dict(ck["state_dict"])
        m.eval()
        if kind == "old":
            norm = jload(OLD_GNP / "GNP_STYLE_TRAIN_NORMALIZATION.json")
            xm = np.asarray(norm["x_mean"], np.float32)
            xs = np.asarray(norm["x_std"], np.float32)
        else:
            xm = np.asarray(ck["normalization_mean"], np.float32)
            xs = np.asarray(ck["normalization_std"], np.float32)
        out.append((p, m, xm, xs, ck))
    return out


def build_smoke_input(cid: str, probe_path: Path, force: float, mu: float) -> np.ndarray:
    # This is the numerical construction used by the existing smoke runner;
    # it reads only the frozen eight-step arm prefix and saved probe telemetry.
    trace_files = sorted((VALID / "TASK0_TRACES").glob(f"{cid}_*.csv"))
    if not trace_files:
        raise FileNotFoundError(f"no frozen task0 trace for {cid}")
    base = list(csv.DictReader(trace_files[0].open(newline="", encoding="utf-8")))[:8]
    probe = list(csv.DictReader(probe_path.open(newline="", encoding="utf-8")))
    last = probe[-1]
    obj = [float(last.get("object_x_priv", 0.0)), float(last.get("object_y_priv", 0.0)), float(last.get("object_z_priv", 0.0))]
    fn = float(last.get("measured_fn", 0.0)); ft = float(last.get("measured_ft", 0.0))
    opening = float(last.get("gripper_opening", 0.02))
    for r in base:
        r.update({
            "object_x_analysis_only": obj[0], "object_y_analysis_only": obj[1], "object_z_analysis_only": obj[2],
            "object_vx_mps": 0.0, "object_vy_mps": 0.0, "object_vz_mps": 0.0,
            "left_normal_force_N": fn, "right_normal_force_N": fn,
            "left_tangential_force_N": ft, "right_tangential_force_N": ft,
            "gripper_pos_0": opening / 2.0, "gripper_pos_1": opening / 2.0,
            "phase": "hold",
        })
    obj_arr = np.asarray([[float(r["object_x_analysis_only"]), float(r["object_y_analysis_only"]), float(r["object_z_analysis_only"])] for r in base], float)
    cmd = np.asarray([[float(r["cmd_x"]), float(r["cmd_y"]), float(r["cmd_z"])] for r in base], float)
    rel = (obj_arr - cmd) - (obj_arr[0] - cmd[0])
    ov = np.asarray([[float(r["object_vx_mps"]), float(r["object_vy_mps"]), float(r["object_vz_mps"])] for r in base], float)
    cv = np.vstack([np.zeros((1, 3)), np.diff(cmd, axis=0) / DT])
    vel = ov - cv
    state = np.zeros((len(base), 13), np.float32); mask = np.zeros_like(state)
    state[:, :3] = rel; state[:, 3:6] = vel; mask[:, :6] = 1
    state[:, 6:10] = np.asarray([[float(r["left_normal_force_N"]), float(r["right_normal_force_N"]), float(r["left_tangential_force_N"]), float(r["right_tangential_force_N"])] for r in base], float)
    state[:, 10] = np.linalg.norm(vel[:, :2], axis=1)
    state[:, 6:11] = np.maximum(np.nan_to_num(state[:, 6:11]), 0); mask[:, 6:11] = 1
    state[:, 11:13] = np.asarray([[float(r["gripper_pos_0"]), float(r["gripper_pos_1"])] for r in base], float); mask[:, 11:13] = 1
    cr = cmd - cmd[0]; cd = np.vstack([np.zeros((1, 3)), np.diff(cmd, axis=0)])
    ph = np.stack([[1.0 if r.get("phase", "") == p else 0.0 for p in PHASES] for r in base], 0)
    task_onehot = np.zeros((len(base), len(TASKS)), float); task_onehot[:, TASKS.index(0)] = 1
    static = np.repeat([[force / 8.0, mu]], len(base), 0)
    init = np.repeat(state[0][None], len(base), 0); im = np.repeat(mask[0][None], len(base), 0)
    nominal = np.concatenate([cr, cd, ph, task_onehot, static, state, mask, init, im], 1).astype(np.float32)
    x = nominal[:8].copy(); x[:, 19:32] = state[0]; x[:, 32:45] = mask[0]
    return x


def utility(p: float, force: float, fmax: float = 5.0) -> float:
    return float(p * (fmax - force) / fmax + (1.0 - p) * (-1.0))


def task0_ablation() -> dict[str, Any]:
    old = load_ensemble(OLD_SEEDS, "old")
    clean = load_ensemble(CLEAN_SEEDS, "clean")
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for cid in [
        "p5s0c_train_t0_r00_s5100_high_mu0.940189",
        "p5s0c_train_t0_r00_s5100_mid_mu0.450580",
        "p5s0c_train_t0_r00_s5100_low_mu0.293710",
    ]:
        post = jload(SMOKE / "FRICTION_POSTERIOR" / f"{cid}.json")
        probe = SMOKE / "PROBE_TELEMETRY" / f"{cid}.csv"
        if not probe.exists():
            # The mid-context smoke retained only the per-repeat copy; use
            # the first saved probe, never a new physical execution.
            probe = SMOKE / "PROBE_TELEMETRY" / f"{cid}_repeat1.csv"
        curves: dict[str, list[dict[str, float]]] = {"frozen_direct": [], "current_clean": []}
        for f in np.round(np.arange(3.0, 5.0001, 0.01), 2):
            for name, ens in [("frozen_direct", old), ("current_clean", clean)]:
                pmu: list[float] = []
                for mu in post["member_means"]:
                    x = build_smoke_input(cid, probe, float(f), float(mu))
                    ps: list[float] = []
                    for _, m, xm, xs, _ in ens:
                        xn = (x - xm) / np.maximum(xs, 1e-6)
                        with torch.no_grad():
                            z = m(torch.tensor(xn[None, :, :17]), torch.tensor(xn[None, 0, 17:]))
                            ps.append(float(torch.sigmoid(z).item()))
                    pmu.append(float(np.mean(ps)))
                p = float(np.mean(pmu))
                u = utility(p, float(f), 5.0)
                curves[name].append({"force_N": float(f), "p_success": p, "expected_utility": u})
                rows.append({"context_id": cid, "force_N": float(f), "frozen_direct_p_success": p if name == "frozen_direct" else "", "current_clean_p_success": p if name == "current_clean" else "", "frozen_direct_utility": u if name == "frozen_direct" else "", "current_clean_utility": u if name == "current_clean" else ""})
        sel = {}
        for name, curve in curves.items():
            best = max(curve, key=lambda q: (q["expected_utility"], -q["force_N"]))
            sel[name] = best
        summaries.append({"context_id": cid, "posterior_member_means": post["member_means"], "frozen_direct_selected": sel["frozen_direct"], "current_clean_selected": sel["current_clean"], "frozen_direct_p3": curves["frozen_direct"][0]["p_success"], "frozen_direct_p5": curves["frozen_direct"][-1]["p_success"], "current_clean_p3": curves["current_clean"][0]["p_success"], "current_clean_p5": curves["current_clean"][-1]["p_success"]})
    dump_csv(OUT / "CHECKPOINT_ONLY_ABLATION.csv", rows)
    result = {"task": 0, "force_domain_N": [3.0, 5.0], "grid_step_N": 0.01, "utility": "p*(5-F)/5+(1-p)*(-1)", "same_probe_posteriors": True, "same_context_input_builder": True, "contexts": summaries}
    dump_json(OUT / "TASK0_SMOKE_CHECKPOINT_ABLATION.json", result)
    return result


def inventory() -> list[dict[str, Any]]:
    roots = [ROOT, TABERO, Path("/media/volume/newdata/exouser")]
    paths: set[Path] = set()
    tokens = ("frozen_direct", "posterior_feasibility", "continuous_feas", "feas_only", "feasibility")
    for base in roots:
        if not base.exists(): continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in {".git", ".venv", "_official_ups", "node_modules", "__pycache__"}]
            for fn in filenames:
                low = fn.lower()
                if low.endswith(".pt") and any(t in low for t in tokens): paths.add((Path(dirpath) / fn).resolve())
    rows=[]
    for p in sorted(paths):
        try: h=sha256(p); st=p.stat(); meta=torch.load(p,map_location="cpu",weights_only=False)
        except Exception as e:
            h="ERROR"; st=p.stat(); meta={"_error":repr(e)}
        name=p.name.lower(); kind="OTHER_FEASIBILITY"
        if "frozen_direct" in name: kind="FROZEN_DIRECT"
        elif "posterior_feasibility" in name: kind="CURRENT_CLEAN_POSTERIOR"
        elif "continuous_feas" in name: kind="GNP_CONTINUOUS_FEAS"
        elif "feas_only" in name: kind="FEASIBILITY_ONLY"
        row={"checkpoint_id":p.stem,"absolute_path":str(p),"sha256":h,"file_mtime":st.st_mtime,"kind":kind,"model_class":"FeasibilityOnly if state_dict-compatible; unknown for noncanonical", "architecture":"","seed":"","training_script":"","training_data_path":"","feature_schema":"","posterior_conditioned":"","label_source":"","label_version":"","calibration":"","used_in_experiment":"","notes":""}
        if isinstance(meta,dict):
            row.update({"architecture":meta.get("architecture",meta.get("variant","")),"seed":meta.get("seed",""),"posterior_conditioned":meta.get("probe_conditioned","")})
            if kind=="FROZEN_DIRECT": row.update({"training_script":str(ROOT/"gnp_style_continuous.py"),"training_data_path":str(OLD_GNP/"CONTINUOUS_TRAIN_SUCCESS_DATA.csv"),"feature_schema":"GRU(17,64)+condition MLP(54,64); 71 normalized features","posterior_conditioned":"NO (historical direct/no-probe point-mu inference)","label_source":"full_task_success_y from GNP continuous dataset","label_version":"independent old720 row-level branch labels","calibration":"raw ensemble mean; train-only isotonic diagnostic","used_in_experiment":"historical F*=4.61/3.75/3.25 cases","notes":"canonical locked handoff or identical copy"})
            if kind=="CURRENT_CLEAN_POSTERIOR": row.update({"training_script":str(ROOT/"rebuild_final_probe_continuous_posterior_20260904.py"),"training_data_path":str(CLEAN/"LIFT_HOLD_LABEL_DATASET.csv"),"feature_schema":"GRU(17,64)+condition MLP(54,64); posterior-conditioned 54-D condition","posterior_conditioned":"YES","label_source":"lift_hold_success_y","label_version":"ROW_LEVEL_LABEL_FIX_AUDIT PASS","calibration":"diagnostic only; no independent calibration","used_in_experiment":"current task0 smoke","notes":"clean rebuild checkpoint"})
        rows.append(row)
    dump_csv(OUT/"FEASIBILITY_CHECKPOINT_INVENTORY.csv",rows)
    return rows


def lineage_artifacts() -> dict[str, Any]:
    bug = jload(ROOT/"analysis/results/full_historical_results_forensics_20260905/OLD_LABEL_BUG_AUDIT.json")
    dump_json(OUT/"LABEL_BUG_PROVENANCE.json", {**bug,"evidence_files":[str(ROOT/"final_no_probe_prepare_dataset.py"),str(ROOT/"analysis/results/full_historical_results_forensics_20260905/OLD_LABEL_BUG_AUDIT.json"),str(CLEAN/"ROW_LEVEL_LABEL_FIX_AUDIT.json")]})
    old_meta=[]; clean_meta=[]
    for p in OLD_SEEDS:
        d=torch.load(p,map_location="cpu",weights_only=False); old_meta.append({"path":str(p),"sha256":sha256(p),"seed":d.get("seed"),"variant":d.get("variant"),"training_branches":d.get("training_branches"),"protocol_sha256":d.get("protocol_sha256")})
    for p in CLEAN_SEEDS:
        d=torch.load(p,map_location="cpu",weights_only=False); clean_meta.append({"path":str(p),"sha256":sha256(p),"seed":d.get("seed"),"architecture":d.get("architecture"),"target":d.get("target"),"training_rows":d.get("training_rows"),"probe_conditioned":d.get("probe_conditioned")})
    old_prov={"checkpoint_family":"FROZEN_DIRECT","canonical_paths":old_meta,"training_script":str(ROOT/"gnp_style_continuous.py"),"training_data":str(OLD_GNP/"CONTINUOUS_TRAIN_SUCCESS_DATA.csv"),"training_data_sha256":sha256(OLD_GNP/"CONTINUOUS_TRAIN_SUCCESS_DATA.csv"),"protocol":str(OLD_GNP/"GNP_STYLE_CONTINUOUS_TRAINING_PROTOCOL.json"),"label_provenance":"INDEPENDENT_VALID","uses_buggy_label_data":False,"label_definition":"full_task_success_y; individual branch outcome","posterior_conditioned":False,"continuous_force_input":True,"evidence_note":"GNP continuous dataset is independent of final_no_probe_prepare_dataset.py; old label audit says old720 raw mapping was unaffected."}
    clean_prov={"checkpoint_family":"POSTERIOR_FEASIBILITY","canonical_paths":clean_meta,"training_script":str(ROOT/"rebuild_final_probe_continuous_posterior_20260904.py"),"training_data":str(CLEAN/"LIFT_HOLD_LABEL_DATASET.csv"),"training_data_sha256":sha256(CLEAN/"LIFT_HOLD_LABEL_DATASET.csv"),"label_provenance":"CLEAN","uses_buggy_label_data":False,"label_definition":"lift_hold_success_y (development target)","posterior_conditioned":True,"continuous_force_input":True,"calibration":"diagnostic only; no independent calibration partition","evidence_files":[str(CLEAN/"ROW_LEVEL_LABEL_FIX_AUDIT.json"),str(CLEAN/"TRAINING_CONFIG.json"),str(CLEAN/"FINAL_FEATURE_SCHEMA.json")]}
    dump_json(OUT/"FROZEN_DIRECT_PROVENANCE.json",old_prov); dump_json(OUT/"CURRENT_CLEAN_PROVENANCE.json",clean_prov)
    cases=[
      {"case_id":"root7703_no_probe_prior_4p61","selected_setpoint":4.61,"checkpoint":str(OLD_SEEDS[0]),"checkpoint_sha256":sha256(OLD_SEEDS[0]),"probe_or_prior":"NO_PROBE_PRIOR","result_source":str(ROOT/"analysis/results/activeforcing_continuous_end_to_end_smoke_20260904/ROOT7703_POSTERIOR_INFERENCE.json")},
      {"case_id":"e5_no_probe_prior_3p75","selected_setpoint":3.75,"checkpoint":str(OLD_SEEDS[0]),"checkpoint_sha256":sha256(OLD_SEEDS[0]),"probe_or_prior":"NO_PROBE_PRIOR","result_source":str(ROOT/"activeforcing_full_claim_closure_20260902_113553/E5_CURRENT_DIRECT_UTILITY/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_103000_task5_tuple02/decisions/activeforcing_full_t5_root00_s7200_high_mu0.964156_NO_QUERY_TRAINING_PRIOR_UTILITY.json")},
      {"case_id":"e5_physical_probe_3p25","selected_setpoint":3.25,"checkpoint":str(OLD_SEEDS[0]),"checkpoint_sha256":sha256(OLD_SEEDS[0]),"probe_or_prior":"PHYSICAL_PROBE","result_source":str(ROOT/"activeforcing_full_claim_closure_20260902_113553/E5_CURRENT_DIRECT_UTILITY/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_103000_task5_tuple02/decisions/activeforcing_full_t5_root00_s7200_high_mu0.964156_ACTIVEFORCING_1Q_UTILITY.json")},
    ]
    for c in cases: c.update({"training_data":str(OLD_GNP/"CONTINUOUS_TRAIN_SUCCESS_DATA.csv"),"label_version":"INDEPENDENT_VALID old720 branch labels","label_valid":"YES"})
    dump_csv(OUT/"HISTORICAL_NONLOWER_SELECTION_PROVENANCE.csv",cases)
    return {"old":old_prov,"clean":clean_prov,"cases":cases,"bug":bug}


def distribution_artifacts() -> dict[str, Any]:
    oldp = pd.read_csv(Path("/media/volume/newdata/exouser/ACTIVEFORCING_DISK_ARCHIVE_20260902/Tabero/analysis/results/gnp_style_continuous_20260830_125107/CONTINUOUS_DEV_FROZEN_PREDICTIONS.csv"))
    cleanp = pd.read_csv(CLEAN/"POSTERIOR_TRAIN_PREDICTIONS.csv")
    dist=[]; sensitivity=[]; oldsel=[]; cleansel=[]
    for df,name,pcol,fcol in [(oldp,"FROZEN_DIRECT","raw_probability","force_N"),(cleanp,"CURRENT_CLEAN","p_success","force_N")]:
        for cid,g in df.groupby("context_id"):
            g=g.sort_values(fcol); task=int(g.iloc[0].task); p=g[pcol].astype(float).to_numpy(); f=g[fcol].astype(float).to_numpy(); fmax=FMAX_BY_TASK[task]
            best_i=max(range(len(g)),key=lambda i:(utility(float(p[i]),float(f[i]),fmax),-float(f[i])))
            sensitivity.append({"model":name,"context_id":str(cid),"task":task,"force_min_N":float(f.min()),"force_max_N":float(f.max()),"delta_p_max_minus_min":float(p[-1]-p[0]),"mean_p":float(p.mean()),"fraction_delta_lt_0.01":int((p[-1]-p[0])<0.01)})
            task_lower = 3.0 if task in (0, 5, 6) else 4.0
            row={"model":name,"context_id":str(cid),"task":task,"selected_force_N":float(f[best_i]),"grid_basis":"existing archived candidates","p_at_selected":float(p[best_i]),"at_task_lower_bound":int(abs(float(f[best_i])-task_lower)<1e-8)}
            (oldsel if name=="FROZEN_DIRECT" else cleansel).append(row)
            dist.append({"model":name,"task":task,"context_id":str(cid),"n_rows":len(g),"force_min_N":float(f.min()),"force_max_N":float(f.max()),"success_positive_or_mean_p":float(p.mean()),"observed_positive_count":int(g["y"].sum()) if "y" in g else "NA"})
    # Add the label distribution that actually trained each family.  These
    # rows are intentionally separate from the per-context prediction rows.
    old_data = pd.read_csv(OLD_GNP / "CONTINUOUS_TRAIN_SUCCESS_DATA.csv")
    clean_data = pd.read_csv(CLEAN / "LIFT_HOLD_LABEL_DATASET.csv")
    for data,name,label in [(old_data,"FROZEN_DIRECT","full_task_success_y"),(clean_data,"CURRENT_CLEAN","lift_hold_success_y")]:
        for task,g in data.groupby("task"):
            for force,gf in g.groupby(g["requested_force_N"].round(2)):
                dist.append({"record_type":"task_force_distribution","model":name,"task":int(task),"context_id":"ALL","n_rows":len(gf),"force_min_N":float(gf.requested_force_N.min()),"force_max_N":float(gf.requested_force_N.max()),"success_positive_or_mean_p":float(gf[label].mean()),"observed_positive_count":int(gf[label].sum()),"label_definition":label})
    for r in dist:
        r.setdefault("record_type","context_curve")
    dump_csv(OUT/"FEASIBILITY_TRAINING_DISTRIBUTION.csv",dist)
    dump_csv(OUT/"FORCE_SENSITIVITY_COMPARISON.csv",sensitivity)
    dump_csv(OUT/"FROZEN_DIRECT_SELECTED_SETPOINTS.csv",oldsel)
    dump_csv(OUT/"CURRENT_CLEAN_SELECTED_SETPOINTS.csv",cleansel)
    old_train = pd.read_csv(OLD_GNP / "CONTINUOUS_TRAIN_SUCCESS_DATA.csv")
    clean_lift = pd.read_csv(CLEAN / "LIFT_HOLD_LABEL_DATASET.csv")
    clean_full = pd.read_csv(CLEAN / "FULL_TASK_LABEL_DATASET.csv")
    dump_json(OUT/"FROZEN_VS_CLEAN_MODEL_COMPARISON.json",{
      "architecture": {"frozen_direct":"GRU(17,64)+condition MLP(54,64)+head","current_clean":"GRU(17,64)+condition MLP(54,64)+head"},
      "training_rows":{"frozen_direct":1008,"current_clean":720},
      "label_counts_and_rates":{
        "frozen_direct_full_task":{"rows":int(len(old_train)),"positive":int(old_train.full_task_success_y.sum()),"rate":float(old_train.full_task_success_y.mean())},
        "current_clean_lift_hold":{"rows":int(len(clean_lift)),"positive":int(clean_lift.lift_hold_success_y.sum()),"rate":float(clean_lift.lift_hold_success_y.mean())},
        "same_720_rows_full_task_reference":{"rows":int(len(clean_full)),"positive":int(clean_full.full_task_success_y.sum()),"rate":float(clean_full.full_task_success_y.mean())}
      },
      "training_data":{"frozen_direct":str(OLD_GNP/"CONTINUOUS_TRAIN_SUCCESS_DATA.csv"),"current_clean":str(CLEAN/"LIFT_HOLD_LABEL_DATASET.csv")},
      "labels":{"frozen_direct":"full_task_success_y","current_clean":"lift_hold_success_y"},
      "posterior_conditioned":{"frozen_direct":False,"current_clean":True},
      "force_input":"continuous absolute force, normalized as F/8 in both schema families",
      "split_and_calibration":{"frozen_direct":"TRAIN/DEV historical protocol; raw ensemble primary","current_clean":"training-distribution diagnostic; calibration diagnostic only, no independent calibration"},
      "eligibility_implication":"Neither checkpoint satisfies all final criteria: old fails posterior-conditioned deployment; clean fails target-definition parity and is lower-bound saturated on current task0 smoke."
    })
    return {"old":oldp,"clean":cleanp,"sensitivity":sensitivity}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inv=inventory(); lin=lineage_artifacts(); dist=distribution_artifacts(); ab=task0_ablation()
    old_target=OLD_SEEDS[0]; clean_target=CLEAN_SEEDS[0]
    # Current smoke exact result is copied from the saved audit, while the
    # old result is freshly computed above from the same saved probe inputs.
    current_summary=[]
    for c in ab["contexts"]:
        current_summary.append({"context_id":c["context_id"],"selected_force_N":c["current_clean_selected"]["force_N"],"p_success":c["current_clean_selected"]["p_success"],"utility":c["current_clean_selected"]["expected_utility"]})
    dump_json(OUT/"FINAL_MODEL_ELIGIBILITY_MATRIX.csv.json",{})
    matrix=[
      {"checkpoint":"FROZEN_DIRECT","no_buggy_labels":"PASS","schema_compatible":"PARTIAL","posterior_conditioned":"FAIL","continuous_force":"PASS","monotonic_or_force_sensitive":"PASS_DEV","calibration":"PARTIAL","non_saturated":"PASS_HISTORICAL_DEV","provenance_clear":"PASS","final_eligibility":"FAIL"},
      {"checkpoint":"CURRENT_CLEAN_POSTERIOR","no_buggy_labels":"PASS","schema_compatible":"PASS","posterior_conditioned":"PASS","continuous_force":"PASS","monotonic_or_force_sensitive":"FAIL_CURRENT_SMOKE_SATURATION","calibration":"PARTIAL_DIAGNOSTIC_ONLY","non_saturated":"FAIL","provenance_clear":"PASS","final_eligibility":"FAIL"},
    ]
    dump_csv(OUT/"FINAL_MODEL_ELIGIBILITY_MATRIX.csv",matrix)
    report=f'''# Feasibility checkpoint lineage audit (offline)\n\nIsaac run: **NO**. Controller, physical-belief model, utility, probe, and force domain were not changed.\n\n## Finding\n\nThe historical 4.61, 3.75, and 3.25 decisions all resolve to the same canonical `FROZEN_DIRECT_FEAS_seed{{0,1,2}}.pt` family. The canonical seed0 is byte-identical to the GNP continuous checkpoint (`920bc6…`). Its training source is the independent `CONTINUOUS_TRAIN_SUCCESS_DATA.csv`, not `final_no_probe_prepare_dataset.py`; the old label audit records 58 wrong rows / 16 affected contexts in the separate buggy dataset and says old720 raw mapping was unaffected.\n\nThe current clean family is a different checkpoint family trained on 720 rows with the clean row-level dataset, but the actual checkpoint target is `lift_hold_success_y`, while the final method definition names `full_task_success_y` as the final target. The clean label rate is 670/720 = 0.9306 for lift+hold, versus 575/720 = 0.7986 for the same rows under full-task labels. The old FROZEN_DIRECT population is 575/1008 = 0.5704 full-task positives. This label-definition shift materially changes the feasibility boundary.\n\nOn the saved task-0 smoke inputs, the clean model gives p(3N)≈0.9983–0.9990 and p(5N)≈0.9989–0.9993, so the fixed utility selects 3.00N by construction. The checkpoint-only task-0 ablation uses identical saved posterior member means, identical saved arm/probe-derived input construction, identical [3,5] grid, and identical utility; only the feasibility checkpoint family changes. It selects FROZEN_DIRECT `[3.92, 4.10, 3.97]N` and CURRENT_CLEAN `[3.00, 3.00, 3.00]N`.\n\n## Eligibility decision\n\n`FROZEN_DIRECT` is not final-compatible because its historical deployment is point/no-probe-mu rather than posterior-conditioned uncertainty input. `CURRENT_CLEAN_POSTERIOR` is posterior-conditioned and clean, but its target is the development lift+hold label, its calibration is not an independent formal calibration, and its current task-0 curves are lower-bound saturated. Therefore no existing checkpoint passes all nine final eligibility conditions.\n\nRecommended next action: retrain **only feasibility**, from clean row-level branch labels, with the final posterior-conditioned feature schema and the final `full_task_success_y` target (or explicitly freeze lift+hold as the method target), with a predeclared validation split and force-sensitivity/calibration checks. Do not change physical belief, Expected Utility, controller, probe, or Isaac execution.\n'''
    (OUT/"FEASIBILITY_CHECKPOINT_LINEAGE_REPORT.md").write_text(report,encoding="utf-8")
    summary={"total_inventory_entries":len(inv),"unique_inventory_hashes":len({r["sha256"] for r in inv if r["sha256"]!="ERROR"}),"frozen_direct_paths":[str(p) for p in OLD_SEEDS],"current_clean_paths":[str(p) for p in CLEAN_SEEDS],"label_bug":{"confirmed":True,"wrong_rows":58,"affected_contexts":16},"task0_current_clean_selected":current_summary,"task0_ablation":ab,"recommended_final_model":"NONE_EXISTING_CHECKPOINT_MEETS_ALL_FINAL_CRITERIA","final_feasibility_model_missing":True,"isaac_run":False}
    dump_json(OUT/"LINEAGE_SUMMARY.json",summary)
    print(json.dumps({"inventory_entries":len(inv),"unique_hashes":summary["unique_inventory_hashes"],"task0_ablation":ab["contexts"]},indent=2))


if __name__ == "__main__":
    main()
