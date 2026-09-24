#!/usr/bin/env python3
"""Read-only recovery of the historical E1 inference chain.

This file intentionally contains no Isaac/IsaacLab imports and no execution
worker.  It writes only to its own result directory.  Historical inputs are
loaded from frozen CSV/JSON/checkpoint artifacts; the old source tree is
materialized only as a temporary path-rewritten view under /tmp because the
original absolute mount prefix is no longer mounted.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "analysis/results/e1_pipeline_recovery_and_current_rerun_20260905"
OLD_SOURCE = Path("/media/volume/newdata/exouser/ACTIVEFORCING_DISK_ARCHIVE_20260902/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000")
ARCHIVE = ROOT / "activeforcing_residual_utility_20260901_055605/POOLED_OOF_UTILITY_DATASET.csv"
PROBE_PRED = ROOT / "activeforcing_probe_conditioned_wm_20260901_064627/POOLED_OOF_PROBE_PREDICTIONS.csv"
OOF = ROOT / "pooled_predictive_verifier_20260901_033804"
E1_TABLE = ROOT / "analysis/results/ACTIVEFORCING_E1_FINAL_UTILITY_MAIN_TABLE_20260902_052942/E1_SELECTED_EPISODES.csv"
E1_REPORT = ROOT / "analysis/results/ACTIVEFORCING_E1_FINAL_UTILITY_MAIN_TABLE_20260902_052942/E1_FINAL_UTILITY_REPORT.md"
E1_LINEAGE = ROOT / "analysis/results/ACTIVEFORCING_E1_FINAL_UTILITY_MAIN_TABLE_20260902_052942/E1_LINEAGE_AUDIT.json"
FRICTION_CKPT = Path("/home/exouser/Tabero/analysis/results/active_friction_imagination_20260828_211106/FRICTION_GRU.pt")
PROBE_OOF_DIR = ROOT / "activeforcing_probe_conditioned_wm_20260901_064627/probe"
EVIDENCE = Path("/home/exouser/Tabero/analysis/activeforcing_historical_transfer_20260904/posterior/evidence_46d.py")
EVIDENCE_NORM = Path("/home/exouser/Tabero/analysis/activeforcing_historical_transfer_20260904/posterior/NORMALIZATION_46D.json")
CURRENT_PROBE_DIR = ROOT / "analysis/results/activeforcing_e2e_task0_smoke_20260905/PROBE_TELEMETRY"
CURRENT_VALIDATION = ROOT / "analysis/results/current_runtime_setpoint_mapping_validation_20260905"
CURRENT_FEAS = ROOT / "analysis/results/final_probe_continuous_posterior_rebuild_20260904"
CURRENT_CIDS = {
    "HIGH": "p5s0c_train_t0_r00_s5100_high_mu0.940189",
    "MID": "p5s0c_train_t0_r00_s5100_mid_mu0.450580",
    "LOW": "p5s0c_train_t0_r00_s5100_low_mu0.293710",
}
FMAX = {0: 5.0, 1: 6.0, 5: 5.0, 6: 4.0}
GRID = np.round(np.arange(3.0, 5.0001, 0.01), 2)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader()
        w.writerows(rows)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def utility(p: np.ndarray | float, f: np.ndarray | float, fmax: float) -> np.ndarray | float:
    return np.asarray(p) * (1.0 - np.asarray(f) / fmax) + (1.0 - np.asarray(p)) * (-1.0)


def make_rewritten_source(tmp: Path) -> Path:
    """Copy only metadata CSVs and rewrite stale absolute paths to old mount."""
    for rel in [
        "PROSPECTIVE_TRAIN_RUN_MANIFEST.csv",
        "PROSPECTIVE_CONTEXT_MANIFEST.csv",
        "collection_train/visual_alignment_worker.csv",
    ]:
        src = OLD_SOURCE / rel
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text(encoding="utf-8")
        text = text.replace("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000", str(OLD_SOURCE))
        dst.write_text(text, encoding="utf-8")
    for task in [0, 1, 5, 6]:
        for name in ["context.csv", "branches.csv"]:
            rel = f"collection_train/task{task}/task{task}/{name}"
            src = OLD_SOURCE / rel
            dst = tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            text = src.read_text(encoding="utf-8")
            text = text.replace("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000", str(OLD_SOURCE))
            dst.write_text(text, encoding="utf-8")
    return tmp


def patch_population_modules(cl: Any, source: Path) -> None:
    """Point the existing offline loader at the recovered old720 mount."""
    early = cl.pooled.early
    # Constants that refer to absolute files use the temporary rewritten
    # metadata, while the loader's derived telemetry glob uses the real old
    # mount (the old CSV rows themselves are rewritten to that mount).
    early.SOURCE = OLD_SOURCE
    early.CONTEXT_CSV = source / "collection_train/task0/task0/context.csv"
    early.BRANCH_CSV = source / "collection_train/task0/task0/branches.csv"
    early.VISUAL_CSV = source / "collection_train/visual_alignment_worker.csv"
    tw = cl.pooled.taskwise
    tw.SOURCE = OLD_SOURCE
    tw.TRAIN_MANIFEST = source / "PROSPECTIVE_TRAIN_RUN_MANIFEST.csv"
    tw.CONTEXT_MANIFEST = source / "PROSPECTIVE_CONTEXT_MANIFEST.csv"
    tw.VISUAL_CSV = source / "collection_train/visual_alignment_worker.csv"


def simple_select(data: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    z = data.copy()
    z["score"] = scores
    rows = []
    for (cid, rep), g in z.groupby(["context_id", "repeat"], sort=True):
        g = g.sort_values("force_N", kind="mergesort")
        best = float(g.score.max())
        s = g[np.isclose(g.score, best, rtol=0.0, atol=1e-12)].iloc[0]
        rows.append({"context_id": str(cid), "repeat": int(rep), "selected_force_N": float(s.force_N), "branch_id": str(s.branch_id), "task": int(s.task), "mu": float(s.mu), "score": best, "actual_success": int(s.success)})
    return pd.DataFrame(rows)


class FrictionGRU(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int = 16, hidden_dim: int = 16):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, projection_dim), nn.ReLU())
        self.gru = nn.GRU(projection_dim, hidden_dim, batch_first=True)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        z = self.projection(x)
        _, h = self.gru(z)
        h = h[-1]
        return self.mu_head(h).squeeze(1), self.log_sigma_head(h).squeeze(1).clamp(-5.0, 1.5)


class ProbeGRU(nn.Module):
    """Exact E1 pooled OOF probe estimator architecture."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU())
        self.gru = nn.GRU(16, 16, batch_first=True)
        self.mu = nn.Linear(16, 1)
        self.logs = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        from torch.nn.utils.rnn import pack_padded_sequence
        z = self.proj(x)
        _, h = self.gru(pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False))
        h = h[-1]
        return self.mu(h).squeeze(1), self.logs(h).squeeze(1).clamp(-5.0, 1.5)


def e1_physical_inference(path: Path) -> dict[str, Any]:
    ev = load_module("e1_evidence_46d", EVIDENCE)
    feat = ev.Evidence46D(EVIDENCE_NORM)
    # E1 pooled OOF probe checkpoints carry their own fold TRAIN mean/std;
    # the 46-D adapter output must remain raw until that checkpoint-specific
    # normalization is applied. (The original single FRICTION_GRU lineage
    # used the transferred normalization directly, which is not this E1 OOF
    # contract.)
    arr = feat.csv(path).astype(np.float32)
    members, sigmas, paths = [], [], []
    for seed in [0, 1, 2]:
        p = PROBE_OOF_DIR / f"probe_fold0_seed{seed}.pt"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        model = ProbeGRU(46); model.load_state_dict(ck["state_dict"]); model.eval()
        mean = np.asarray(ck["mean"], np.float32); std = np.asarray(ck["std"], np.float32)
        x = (arr - mean) / np.maximum(std, 1e-6)
        with torch.no_grad():
            mu, log_sigma = model(torch.tensor(x[None]), torch.tensor([len(x)], dtype=torch.long))
        members.append(float(mu.item())); sigmas.append(float(torch.exp(log_sigma).item())); paths.append(str(p))
    return {"mu_hat": float(np.mean(members)), "sigma_mu": float(np.std(members, ddof=1)), "member_means": members, "member_sigma_diagnostic": sigmas, "feature_rows": int(arr.shape[0]), "feature_dim": int(arr.shape[1]), "checkpoint": paths, "posterior_semantics": "three equally weighted E1 grouped-root OOF probe point supports; posterior selector marginalizes utility over these supports"}


def replay_historical_probe_predictions(probe: pd.DataFrame) -> dict[str, Any]:
    """Recompute all 216 saved OOF probe members from their raw telemetry."""
    ev = load_module("e1_probe_replay_evidence", EVIDENCE)
    feat = ev.Evidence46D(EVIDENCE_NORM)
    rows = []
    original = "/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000"
    for r in probe[probe.seed.astype(str).isin(["0", "1", "2"])].itertuples(index=False):
        raw_path = Path(str(r.raw_path).replace(original, str(OLD_SOURCE)))
        arr = feat.csv(raw_path).astype(np.float32)
        ckpath = PROBE_OOF_DIR / f"probe_fold{int(r.fold)}_seed{int(r.seed)}.pt"
        ck = torch.load(ckpath, map_location="cpu", weights_only=False)
        model = ProbeGRU(46); model.load_state_dict(ck["state_dict"]); model.eval()
        x = (arr - np.asarray(ck["mean"], np.float32)) / np.maximum(np.asarray(ck["std"], np.float32), 1e-6)
        with torch.no_grad():
            mu, _ = model(torch.tensor(x[None]), torch.tensor([len(x)], dtype=torch.long))
        rows.append({"context_id": str(r.context_id), "fold": int(r.fold), "seed": int(r.seed), "historical_mu_hat": float(r.mu_hat), "replayed_mu_hat": float(mu.item()), "abs_error": float(abs(mu.item() - r.mu_hat)), "raw_rows": int(len(x))})
    errors = np.asarray([r["abs_error"] for r in rows], float)
    return {"rows": rows, "member_count": int(len(rows)), "max_abs_error": float(errors.max()), "mean_abs_error": float(errors.mean()), "exact_within_1e-6": bool(np.all(errors <= 1e-6)), "raw_telemetry_source": str(OLD_SOURCE)}


def current_prefix(cid: str, force: float, mu: float) -> np.ndarray:
    """Exact 8-step model-ready adapter copied from the current smoke."""
    trace = sorted((CURRENT_VALIDATION / "TASK0_TRACES").glob(f"{cid}_*.csv"))[0]
    with trace.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:8]
    probe = CURRENT_PROBE_DIR / f"{cid}.csv"
    if not probe.exists():
        probe = CURRENT_PROBE_DIR / f"{cid}_repeat1.csv"
    with probe.open(newline="", encoding="utf-8") as f:
        last = list(csv.DictReader(f))[-1]
    obj = [float(last.get(k, 0.0)) for k in ["object_x_priv", "object_y_priv", "object_z_priv"]]
    fn, ft = float(last.get("measured_fn", 0.0)), float(last.get("measured_ft", 0.0))
    for r in rows:
        r.update({"object_x_analysis_only": obj[0], "object_y_analysis_only": obj[1], "object_z_analysis_only": obj[2], "object_vx_mps": 0.0, "object_vy_mps": 0.0, "object_vz_mps": 0.0, "left_normal_force_N": fn, "right_normal_force_N": fn, "left_tangential_force_N": ft, "right_tangential_force_N": ft, "gripper_pos_0": float(last.get("gripper_opening", 0.02)) / 2.0, "gripper_pos_1": float(last.get("gripper_opening", 0.02)) / 2.0, "phase": "hold"})
    obj_a = np.asarray([[float(r["object_x_analysis_only"]), float(r["object_y_analysis_only"]), float(r["object_z_analysis_only"])] for r in rows])
    cmd = np.asarray([[float(r["cmd_x"]), float(r["cmd_y"]), float(r["cmd_z"])] for r in rows])
    rel = (obj_a - cmd) - (obj_a[0] - cmd[0])
    ov = np.asarray([[float(r["object_vx_mps"]), float(r["object_vy_mps"]), float(r["object_vz_mps"])] for r in rows])
    cv = np.vstack([np.zeros((1, 3)), np.diff(cmd, axis=0) / 0.05])
    vel = ov - cv
    state = np.zeros((len(rows), 13), np.float32); mask = np.zeros_like(state)
    state[:, :3] = rel; state[:, 3:6] = vel; mask[:, :6] = 1
    state[:, 6:10] = np.asarray([[float(r["left_normal_force_N"]), float(r["right_normal_force_N"]), float(r["left_tangential_force_N"]), float(r["right_tangential_force_N"])] for r in rows])
    state[:, 10] = np.linalg.norm(vel[:, :2], axis=1); state[:, 6:11] = np.maximum(np.nan_to_num(state[:, 6:11]), 0); mask[:, 6:11] = 1
    state[:, 11:13] = np.asarray([[float(r["gripper_pos_0"]), float(r["gripper_pos_1"])] for r in rows]); mask[:, 11:13] = 1
    cr = cmd - cmd[0]; cd = np.vstack([np.zeros((1, 3)), np.diff(cmd, axis=0)])
    phases = ["branch_hold", "lift", "transit", "over_basket", "place", "release", "settle"]
    ph = np.stack([[1.0 if str(r.get("phase", "")) == p else 0.0 for p in phases] for r in rows])
    task_one = np.zeros((len(rows), 4), np.float32); task_one[:, 0] = 1.0
    static = np.repeat([[force / 8.0, mu]], len(rows), axis=0)
    init = np.repeat(state[0][None], len(rows), axis=0); im = np.repeat(mask[0][None], len(rows), axis=0)
    full = np.concatenate([cr, cd, ph, task_one, static, state, mask, init, im], axis=1).astype(np.float32)
    x = full[:8].copy(); x[:, 19:32] = state[0]; x[:, 32:45] = mask[0]
    return x


def model_score(model: nn.Module, x: np.ndarray, norm_mean: np.ndarray, norm_std: np.ndarray) -> float:
    xn = (x - norm_mean) / np.maximum(norm_std, 1e-6)
    with torch.no_grad():
        logit = model(torch.tensor(xn[None, :, :17]), torch.tensor(xn[None, 0, 17:])).item()
    return float(1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, logit)))))


def load_direct_models(cl: Any, fold: int) -> list[tuple[nn.Module, np.ndarray, np.ndarray, str]]:
    models = []
    for seed in [0, 1, 2]:
        path = OOF / "checkpoints" / f"DIRECT_POOLED_fold{fold}_seed{seed}.pt"
        d = torch.load(path, map_location="cpu", weights_only=False)
        m = cl.pooled.early.load_module(f"e1_direct_full_{fold}_{seed}", cl.pooled.early.FULL_CODE).FeasibilityOnly()
        m.load_state_dict(d["state_dict"]); m.eval()
        norm = d["normalization"]
        models.append((m, np.asarray(norm["x_mean"], np.float32), np.asarray(norm["x_std"], np.float32), str(path)))
    return models


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ARCHIVE).reset_index(drop=True)
    probe = pd.read_csv(PROBE_PRED)
    e1 = pd.read_csv(E1_TABLE)
    active_hist = e1[e1.method == "ActiveForcing-1Q Utility"].copy().reset_index(drop=True)
    lineage = json.loads(E1_LINEAGE.read_text())
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    cl = load_module("e1_closure_offline", ROOT / "utility_and_causal_ablation_closure.py")

    # Provenance is written before inference so the audit remains useful if a
    # later optional computation fails.
    direct_paths = [OOF / "checkpoints" / f"DIRECT_POOLED_fold{f}_seed{s}.pt" for f in [0, 1, 2] for s in [0, 1, 2]]
    physical_paths = [PROBE_OOF_DIR / f"probe_fold{f}_seed{s}.pt" for f in [0, 1, 2] for s in [0, 1, 2]]
    provenance = {
        "status": "OFFLINE_ONLY",
        "E1_runner": [str(ROOT / "run_pooled_predictive_verifier.py"), str(ROOT / "run_residual_utility.py"), str(E1_TABLE.parent / "scripts/build_e1_final_utility_main.py")],
        "E1_git_commit": "UNRECORDED_UNTRACKED_SOURCE; workspace HEAD is not asserted as E1 commit",
        "E1_feasibility_model_paths": [str(x) for x in direct_paths],
        "E1_feasibility_model_sha256": {str(x): sha(x) for x in direct_paths},
        "E1_model_class": "FeasibilityOnly",
        "E1_model_architecture": "GRU(17,64)+MLP(54,64)+head(128->64->1)",
        "E1_training_data": str(ARCHIVE),
        "E1_label_source": "POOLED_OOF_UTILITY_DATASET.csv; task1 includes frozen terminal-height fallback rows",
        "E1_label_definition": "full-task success / lift_hold_success_y as assembled in pooled OOF archive",
        "E1_uses_buggy_58_row_labels": "NO (that separate final_no_probe cache bug is not the E1 source; E1 has a separate task1 reconstruction caveat)",
        "E1_physical_belief_paths": [str(x) for x in physical_paths],
        "E1_physical_belief_member_count": 3,
        "E1_physical_belief_sha256": {str(x): sha(x) for x in physical_paths},
        "probe_feature_schema": str(EVIDENCE_NORM),
        "probe_feature_dim": 46,
        "probe_normalization": str(EVIDENCE_NORM),
        "probe_checkpoint_normalization": "each probe_fold{fold}_seed{seed}.pt carries TRAIN-fitted mean/std; adapter output remains raw before checkpoint normalization",
        "posterior_construction": "three equally weighted grouped-root OOF probe point supports from probe_fold0/1/2_seed0/1/2; current root00 uses held fold0 members; no continuous density",
        "candidate_force_normalization": "F/8.0 in Direct feature vector; verified from E1 source and checkpoint input contract",
        "posterior_conditioning": "mu inserted at condition feature index 18 before Direct normalization",
        "selector_formula": "p_success*(1-F/Fmax)+(1-p_success)*(-1)",
        "force_domain": "historical archive: five saved candidate forces per context/repeat; current offline B: 3.00..5.00 step .01",
        "tie_break": "lower force on equal utility",
        "lineage_sha256": sha(E1_LINEAGE),
        "E1_report": str(E1_REPORT),
    }
    write_json(OUT / "E1_PIPELINE_PROVENANCE.json", provenance)

    # Recover the old source tree in a temporary path and run the original
    # frozen loader/inference function in memory. This does not write there.
    with tempfile.TemporaryDirectory(prefix="e1_recovery_source_", dir="/tmp") as ts:
        source = make_rewritten_source(Path(ts))
        patch_population_modules(cl, source)
        with tempfile.TemporaryDirectory(prefix="e1_recovery_loader_", dir="/tmp") as td:
            tpi, cf, full, cmap, traces, meta, audits, pairs, segs, norm = cl.pooled.load_population(Path(td))
            md = cl.ppv.frame(traces, meta).reset_index(drop=True)
            if len(md) != 720 or not np.array_equal(md.branch_id.astype(str).to_numpy(), data.branch_id.astype(str).to_numpy()):
                raise RuntimeError("old720 loader did not align exactly to the authoritative OOF archive")
            members, pivot = cl.load_probe_members(data.context_id)
            point_mu = np.mean(np.stack([members[m] for m in [0, 1, 2]]), axis=0).astype(np.float32)
            scenarios = {"gt": data.mu.to_numpy(np.float32), "point": point_mu}
            scenarios.update({f"member_{m}": members[m] for m in [0, 1, 2]})
            probs = cl.infer_direct_scenarios(full, traces, segs, data.fold.to_numpy(int), scenarios)

    probe_replay = replay_historical_probe_predictions(probe)
    write_rows(OUT / "E1_PROBE_OOF_REPLAY.csv", probe_replay["rows"])

    gt_err = float(np.max(np.abs(probs["gt"] - data.p_D_OOF_ensemble.to_numpy(float))))
    point_scores = utility(probs["point"], data.force_N.to_numpy(float), data.task.map(FMAX).to_numpy(float))
    replay_sel = simple_select(data, point_scores)
    hist_sel = active_hist[["context_id", "repeat", "selected_force_N", "branch_id", "task"]].rename(columns={"selected_force_N": "historical_selected_force_N", "branch_id": "historical_branch_id"})
    joined = replay_sel.merge(hist_sel, on=["context_id", "repeat", "task"], how="outer")
    selected_diff = np.abs(joined.selected_force_N - joined.historical_selected_force_N)

    # Golden contexts are selected from actual E1 decisions, not hand-picked
    # synthetic values. The targets make the non-lower-bound cases visible.
    targets = [3.25, 3.75, 4.61]
    golden_idx: set[int] = set()
    for target in targets:
        golden_idx.add(int(np.argmin(np.abs(active_hist.selected_force_N.to_numpy(float) - target))))
    golden_idx.add(int(active_hist.selected_force_N.idxmin()))
    golden_idx.add(int(active_hist.selected_force_N.idxmax()))
    for band in ["LOW", "HIGH"]:
        q = active_hist[active_hist.raw_friction_band.astype(str).str.upper() == band]
        if len(q): golden_idx.add(int(q.index[0]))
    golden = active_hist.loc[sorted(golden_idx)].head(10).copy()
    golden.to_csv(OUT / "E1_GOLDEN_INFERENCE_CASES.csv", index=False)
    replay_rows = []
    for _, h in golden.iterrows():
        q = data[(data.context_id.astype(str) == str(h["context_id"])) & (data["repeat"].astype(int) == int(h["repeat"]))].sort_values("force_N")
        for i in q.index:
            replay_rows.append({
                "task": int(h["task"]), "root": str(h["root_id"]), "context": str(h["context_id"]), "repeat": int(h["repeat"]),
                "historical_selected_setpoint": float(h["selected_force_N"]), "historical_force_candidate": float(data.loc[i, "force_N"]),
                "historical_p_success": float(data.loc[i, "p_D_OOF_ensemble"]), "replayed_p_success_point_mu": float(probs["point"][i]),
                "replayed_p_success_gt_mu": float(probs["gt"][i]), "historical_mu_point": float(point_mu[i]), "historical_mu_gt": float(data.loc[i, "mu"]),
                "p_abs_error_point": float(abs(probs["point"][i] - data.loc[i, "p_D_OOF_ensemble"])), "source_archive": str(ARCHIVE),
            })
    write_rows(OUT / "E1_EXACT_INFERENCE_REPLAY.csv", replay_rows)

    # Aggregate exact replay and friction-stratified decisions.
    hist_mean = float(active_hist.selected_force_N.mean())
    rep_mean = float(replay_sel.selected_force_N.mean())
    agg = {
        "historical_E1_SR": float(active_hist.actual_success.mean()),
        "historical_E1_mean_selected_setpoint": hist_mean,
        "replayed_E1_mean_selected_setpoint": rep_mean,
        "historical_E1_lower_bound_rate": float(np.mean(active_hist.selected_force_N <= 3.000001)),
        "replayed_E1_lower_bound_rate": float(np.mean(replay_sel.selected_force_N <= 3.000001)),
        "direct_gt_reproduction_max_abs_error": gt_err,
        "selected_force_exact_match_count": int(np.sum(selected_diff <= 1e-6)),
        "selected_force_exact_match_rate": float(np.mean(selected_diff <= 1e-6)),
        "posterior_replay_status": "RAW_216_MEMBER_OOF_PROBE_REPLAY",
        "probe_member_replay_max_abs_error": probe_replay["max_abs_error"],
        "probe_member_replay_mean_abs_error": probe_replay["mean_abs_error"],
        "probe_member_replay_parity": probe_replay["exact_within_1e-6"],
        "aggregate_replay_parity": bool(np.all(selected_diff <= 1e-6)),
        "task_group_means": {},
    }
    hband = active_hist[["context_id", "repeat", "raw_friction_band", "selected_force_N", "task"]].copy()
    rband = replay_sel.merge(hband[["context_id", "repeat", "raw_friction_band"]], on=["context_id", "repeat"], how="left")
    for task in [0, 5]:
        for band in ["LOW", "HIGH"]:
            hh = active_hist[(active_hist.task == task) & (active_hist.raw_friction_band.astype(str).str.upper() == band)]
            rr = rband[(rband.task == task) & (rband.raw_friction_band.astype(str).str.upper() == band)]
            agg["task_group_means"][f"task{task}_{band.lower()}_mu_mean_F_historical"] = float(hh.selected_force_N.mean()) if len(hh) else None
            agg["task_group_means"][f"task{task}_{band.lower()}_mu_mean_F_replayed"] = float(rr.selected_force_N.mean()) if len(rr) else None
    write_json(OUT / "E1_AGGREGATE_REPLAY.json", agg)

    # Current task0: exact historical FRICTION_GRU on the three saved probes,
    # then exact E1 Direct fold-0 heldout checkpoints on the current adapter.
    current_post = {}
    e1_curves = []
    selection = {}
    clean_models = []
    for seed in [0, 1, 2]:
        p = CURRENT_FEAS / f"POSTERIOR_FEASIBILITY_seed{seed}.pt"
        d = torch.load(p, map_location="cpu", weights_only=False)
        class CleanFeasibility(nn.Module):
            def __init__(self):
                super().__init__(); self.command_gru = nn.GRU(17,64,batch_first=True); self.condition = nn.Sequential(nn.Linear(54,64),nn.ReLU()); self.head = nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
            def forward(self, step, cond):
                _, h = self.command_gru(step); return self.head(torch.cat([h[-1], self.condition(cond)], dim=-1)).squeeze(-1)
        cm = CleanFeasibility(); cm.load_state_dict(d["state_dict"]); cm.eval()
        clean_models.append((cm, np.asarray(d["normalization_mean"], np.float32), np.asarray(d["normalization_std"], np.float32), str(p)))
    e1_models = load_direct_models(cl, 0)
    for band, cid in CURRENT_CIDS.items():
        probe_path = CURRENT_PROBE_DIR / f"{cid}.csv"
        if not probe_path.exists():
            # The mid base probe was preserved under the repeat-1 filename;
            # it is the same saved probe branch, not a new probe.
            probe_path = CURRENT_PROBE_DIR / f"{cid}_repeat1.csv"
        pp = e1_physical_inference(probe_path)
        current_post[band] = {"context_id": cid, "band": band, **pp, "probe_path": str(probe_path)}
        e1_vals = []; clean_vals = []
        for f in GRID:
            ep_members = []
            cp_members = []
            for member_mu in pp["member_means"]:
                xm_e = [model_score(m, current_prefix(cid, float(f), float(member_mu)), xm, xs) for m, xm, xs, _ in e1_models]
                xm_c = [model_score(m, current_prefix(cid, float(f), float(member_mu)), xm, xs) for m, xm, xs, _ in clean_models]
                ep_members.append(float(np.mean(xm_e))); cp_members.append(float(np.mean(xm_c)))
            pe, pc = float(np.mean(ep_members)), float(np.mean(cp_members))
            e1_vals.append({"context": cid, "band": band, "force_N": float(f), "p_success": pe, "expected_utility": float(utility(pe, float(f), 5.0)), "posterior_member_p_success": json.dumps(ep_members), "inference": "E1 Direct fold0; utility marginalization over 3 OOF probe supports"})
            clean_vals.append({"force_N": float(f), "p_success": pc, "expected_utility": float(utility(pc, float(f), 5.0)), "e1_p_success": pe, "e1_utility": float(utility(pe, float(f), 5.0)), "context": cid, "band": band, "posterior_member_p_success": json.dumps(cp_members)})
        e1_curves.extend(e1_vals)
        best = max(e1_vals, key=lambda r: (r["expected_utility"], -r["force_N"]))
        best_clean = max(clean_vals, key=lambda r: (r["expected_utility"], -r["force_N"]))
        selection[band] = {"context_id": cid, "E1_style_F_star": best["force_N"], "E1_style_p_success_at_F_star": best["p_success"], "E1_style_utility_at_F_star": best["expected_utility"], "current_clean_F_star": best_clean["force_N"], "current_clean_p_success_at_F_star": best_clean["p_success"], "same_input_checkpoint_only": True, "selection_semantics": "argmax expected utility after marginalizing over three E1 OOF probe supports"}
        # Add current-vs-clean values to a separate ablation table at all grid points.
        for r in clean_vals:
            pass
    write_json(OUT / "CURRENT_TASK0_E1_STYLE_POSTERIORS.json", current_post)
    write_rows(OUT / "E1_STYLE_CURRENT_TASK0_SUCCESS_CURVES.csv", e1_curves)
    write_json(OUT / "CURRENT_TASK0_E1_STYLE_SELECTION.json", selection)

    ablation_rows = []
    for band, cid in CURRENT_CIDS.items():
        pp = current_post[band]
        for f in GRID:
            ep_members = []; cp_members = []
            for member_mu in pp["member_means"]:
                x = current_prefix(cid, float(f), float(member_mu))
                ep_members.append(float(np.mean([model_score(m, x, xm, xs) for m, xm, xs, _ in e1_models])))
                cp_members.append(float(np.mean([model_score(m, x, xm, xs) for m, xm, xs, _ in clean_models])))
            pe, pc = float(np.mean(ep_members)), float(np.mean(cp_members))
            ablation_rows.append({"context": cid, "band": band, "force_N": float(f), "E1_checkpoint_p_success": pe, "current_clean_checkpoint_p_success": pc, "E1_utility": float(utility(pe,float(f),5.0)), "current_clean_utility": float(utility(pc,float(f),5.0)), "E1_posterior_member_p_success": json.dumps(ep_members), "current_posterior_member_p_success": json.dumps(cp_members), "E1_checkpoint_paths": ";".join(str(x[3]) for x in e1_models), "current_checkpoint_paths": ";".join(str(x[3]) for x in clean_models)})
    write_rows(OUT / "E1_VS_CURRENT_CHECKPOINT_ONLY_ABLATION.csv", ablation_rows)

    diff = {
        "first_causal_difference": "feasibility checkpoint lineage: E1 DIRECT_POOLED_fold*_seed*.pt versus current POSTERIOR_FEASIBILITY_seed*.pt clean rebuild",
        "E1_physical_belief_difference": "E1 archive used three OOF FRICTION_GRU probe members as point-support scenarios; current smoke uses a different 3-member PHYSICAL_BELIEF E6 lineage",
        "E1_vs_current_candidate_normalization": "both use F/8.0 in the 17-step command/condition input for the compared Direct/clean FeasibilityOnly models",
        "E1_vs_current_selector": "same expected utility and lower-force tie break; current smoke grid is [3,5] step .01; no utility change",
        "E1_vs_current_physical_contract": "not evaluated here; current task0 execution mapping remains the validated contract and no physics was rerun",
        "checkpoint_only_ablation": "same current probe-derived input, force grid and utility; E1 Direct versus current clean checkpoint",
        "historical_direct_replay_gt_max_abs_error": gt_err,
    }
    # Determine ablation-induced decision collapse directly from saved rows.
    ab = pd.DataFrame(ablation_rows)
    diff["checkpoint_only_E1_selection"] = {b: float(ab[ab.band == b].sort_values(["E1_utility","force_N"], ascending=[False,True]).iloc[0].force_N) for b in CURRENT_CIDS}
    diff["checkpoint_only_current_selection"] = {b: float(ab[ab.band == b].sort_values(["current_clean_utility","force_N"], ascending=[False,True]).iloc[0].force_N) for b in CURRENT_CIDS}
    diff["checkpoint_only_ablation_confirms_cause"] = bool(any(diff["checkpoint_only_E1_selection"][b] != diff["checkpoint_only_current_selection"][b] for b in CURRENT_CIDS))
    write_json(OUT / "E1_VS_CURRENT_PIPELINE_DIFF.json", diff)

    eligibility = {
        "E1_labels_clean": "PARTIAL",
        "E1_label_caveat": "task1 has 40 direct and 140 reconstructed terminal labels in the pooled archive",
        "E1_uses_separate_58_row_cache_bug": False,
        "E1_posterior_conditioned": "PARTIAL; archived Active used three OOF point supports, not a continuous posterior density",
        "E1_continuous_force_input": True,
        "E1_method_compatible": "PARTIAL",
        "E1_eligible_for_final_method": "PARTIAL/NO",
        "final_physical_run_allowed": False,
        "blocking_reason": "user boundary forbids new experiments this round; additionally E1 provenance is not clean enough to authorize final physical claim",
    }
    write_json(OUT / "E1_MODEL_ELIGIBILITY.json", eligibility)
    write_json(OUT / "FINAL_INFERENCE_RUNTIME_MANIFEST.json", {"status": "OFFLINE_RECOVERY_ONLY; PHYSICAL_RERUN_NOT_EXECUTED", "controller_changed": False, "utility_changed": False, "probe_changed": False, "E1_direct_sha256": {str(p): sha(p) for p in direct_paths}, "E1_physical_sha256": {str(p): sha(p) for p in physical_paths}, "preprocessing_sha256": {str(EVIDENCE): sha(EVIDENCE), str(EVIDENCE_NORM): sha(EVIDENCE_NORM)}, "selector": "historical utility; lower-force tie break", "physical_rerun_allowed": False})

    report = f"""# E1 pipeline recovery report

## Scope

This is a CPU/offline historical inference audit. No Isaac Sim, Isaac Lab, policy server, collector, rollout, training, or physics rerun was started. Historical artifacts were not modified.

## Recovered E1 chain

- Base runner: `{ROOT / 'run_pooled_predictive_verifier.py'}`.
- Utility assembly: `{ROOT / 'run_residual_utility.py'}`.
- E1 table builder: `{E1_TABLE.parent / 'scripts/build_e1_final_utility_main.py'}`.
- Feasibility model: nine `DIRECT_POOLED_fold{{0,1,2}}_seed{{0,1,2}}.pt` checkpoints, grouped-root OOF.
- Probe model: `activeforcing_probe_conditioned_wm_20260901_064627/probe/probe_fold0_seed{{0,1,2}}.pt` for the current root00 held fold; the lineage also records the original `FRICTION_GRU.pt`. The 46-D evidence adapter and TRAIN-only normalization are unchanged.
- Selector: `p*(1-F/Fmax)+(1-p)*(-1)`, lower-force tie break.

## Exact replay

The raw old720 metadata mount had stale `/home/exouser/Tabero/...` absolute paths. A temporary path-rewritten view under `/tmp` restored the same old files without altering them. The old720 loader aligned to 720 branches; all 216 saved grouped-root OOF probe members were also recomputed from raw telemetry with maximum absolute mu error `{probe_replay['max_abs_error']:.3g}`. The archived Direct probability reproduction error was `{gt_err:.3g}` maximum absolute probability error. Point-mu selected-force exact-match rate against the archived E1 Active table was `{agg['selected_force_exact_match_rate']:.6f}`.

Historical E1 mean selected setpoint: `{hist_mean:.10f}`. Replayed mean: `{rep_mean:.10f}`. Historical lower-bound rate: `{agg['historical_E1_lower_bound_rate']:.6f}`; replayed: `{agg['replayed_E1_lower_bound_rate']:.6f}`.

## Current task0 offline adapter

The three saved smoke probe CSVs were passed through the historical 46-D adapter and the E1 fold-0 grouped-root OOF probe checkpoints, then scored by E1 fold-0 Direct checkpoints on the current validated task0 prefix. The current clean checkpoints were scored on exactly the same input for checkpoint-only comparison. No selected force was sent to a simulator. The saved current probe files contain 206 rows, versus the historical E1 probe contract's 215 rows; this is recorded as a transfer caveat, not silently padded.

| context | E1-style F* | current clean F* |
|---|---:|---:|
""" + "\n".join(f"| {b} | {selection[b]['E1_style_F_star']:.2f} | {selection[b]['current_clean_F_star']:.2f} |" for b in CURRENT_CIDS) + f"""

## Eligibility and limitation

E1 is useful for regression/parity recovery, but is only `PARTIAL/NO` as the final paper model: the archived physical belief is point-support OOF evidence rather than a continuous density, and pooled task1 labels include reconstructed terminal labels. Therefore `PHYSICAL_RERUN_ALLOWED=NO` in this round.
"""
    (OUT / "E1_PIPELINE_RECOVERY_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
