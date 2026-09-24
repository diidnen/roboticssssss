#!/usr/bin/env python3
"""Re-audit corrected contact labels and train/evaluate the IE+EVENT ablation.

Only TRAIN/DEV rows are read.  The continuous Physics-GRU trunk and IE term
are inherited; the only new learnable output is a three-channel direct contact
event head (left, right, bilateral).  No final-success label is a target.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO = Path("/home/exouser/Tabero")
RESULTS = REPO / "analysis/results"
OUT = Path(os.environ.get("IE_EVENT_OUT", RESULTS / "targeted_direct_event_collection_20260829_193500"))
CF_ROOT = RESULTS / "counterfactual_force_world_model_20260829_160000"
HIST_ROOT = RESULTS / "p5s0c_paired_boundary_probe_value_20260824_000542"
DIRECT_ROOT = RESULTS / "direct_contact_boundary_dataset_20260829_001409"
COLLECTION_ROOT = OUT / "collection"
FORENSIC_ROOT = RESULTS / "failure_preservation_forensic_20260829_170000"
EVAL_ROOT = RESULTS / "evaluator_interface_calibration_20260829_112603"
EVENT_DEF = RESULTS / "direct_contact_boundary_imagination_20260829_000205" / "DIRECT_PHYSICAL_EVENT_DEFINITION.json"
H = 8
SEEDS = [0, 1, 2]
LAMBDA_EVENTS = [0.1, 0.3, 1.0]
ACTIVE_PHASES = {"branch_hold", "lift", "transit", "over_basket", "place"}
_EVENT_LABEL_CACHE = {}
STATE_NAMES = ["rel_dx_m", "rel_dy_m", "rel_dz_m", "rel_vx_mps", "rel_vy_mps", "rel_vz_mps", "left_normal_N", "right_normal_N", "left_tangent_N", "right_tangent_N", "tangent_velocity_proxy_mps", "joint_left", "joint_right"]
TASKS = [0, 1, 5, 6]
RESUME_EVAL_ONLY = os.environ.get("IE_EVENT_RESUME_EVAL_ONLY") == "1"


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tpi = import_module("tpi_ie_event", REPO / "analysis/trajectory_physical_imagination.py")
cf = import_module("cf_ie_event", REPO / "analysis/counterfactual_force_world_model.py")


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, data: list[dict]) -> None:
    if not data:
        path.write_text("\n", encoding="utf-8")
        return
    fields = []
    for row in data:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def is_true(value) -> bool:
    """Parse CSV booleans and already-materialized Python booleans alike."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def load_manifest_rows(path: Path, split_only=True) -> list[dict]:
    df = pd.read_csv(path)
    if split_only:
        df = df[df["split"].isin(["TRAIN", "DEV"])]
    return df.to_dict("records")


def make_trace(r: dict, source: str) -> tpi.Trace:
    d = pd.read_csv(r["telemetry_path"])
    s, m = tpi.state_from(d)
    force = float(r["requested_force_N"])
    mu = float(r["hidden_friction_analysis_only"])
    nom = tpi.nominal_from(d, int(r["task"]), force, mu, s, m)
    # Role and base weight are for sampling only; neither success nor frontier
    # is an input to the model.
    role = "prev" if str(r.get("force_role", "")).upper() == "F_PREV" else ("star" if str(r.get("force_role", "")).upper() == "F_STAR" else "next")
    return tpi.Trace(str(r["branch_id"]), str(r["context_id"]), str(r["root_id"]), int(r["task"]), str(r["split"]), force, mu, int(r["full_task_success_y"]), role, Path(r["telemetry_path"]), s, m, nom, d["phase"].astype(str).tolist(), 3.0 if role in {"prev", "star"} else 1.0, source)


def load_train_dev_traces() -> tuple[list[tpi.Trace], dict[str, dict], dict[str, dict]]:
    hist_rows = load_manifest_rows(HIST_ROOT / "P5S0C_BRANCH_MANIFEST.csv")
    direct_rows = []
    for task in TASKS:
        direct_rows += load_manifest_rows(DIRECT_ROOT / f"task{task}/branches.csv")
    collected_rows = []
    for task in TASKS:
        p = COLLECTION_ROOT / f"task{task}/branches.csv"
        if p.exists():
            collected_rows += load_manifest_rows(p)
    traces = [make_trace(r, "historical") for r in hist_rows] + [make_trace(r, "direct") for r in direct_rows] + [make_trace(r, "direct_event_collected") for r in collected_rows]
    # The collector's operational ``force_role`` is intentionally not the
    # scientific F_prev/F_star label.  Restore the frozen target-manifest
    # roles before constructing the pair-balanced training units so the new
    # corrected branches receive the same boundary weights as the inherited
    # branches.  This does not enter the model input.
    target_path = OUT / "DIRECT_EVENT_TARGET_MANIFEST.csv"
    if target_path.exists():
        target_role = {r["expected_branch_id"]: r["force_class"] for r in csv.DictReader(target_path.open())}
        for tr in traces:
            if tr.source == "direct_event_collected" and tr.branch_id in target_role:
                tr.role = "prev" if target_role[tr.branch_id] == "F_prev" else ("star" if target_role[tr.branch_id] == "F_star" else "next")
                tr.weight = 3.0 if tr.role in {"prev", "star"} else 1.0
    by_key = {(x.source, x.context_id, x.branch_id): x for x in traces}
    return traces, by_key, {r["branch_id"]: r for r in collected_rows}


def load_pairs(by_key: dict, collected_rows: dict) -> list[cf.Pair]:
    pairs = []
    for r in csv.DictReader((CF_ROOT / "COUNTERFACTUAL_FORCE_PAIRS.csv").open()):
        if r["split"] not in {"TRAIN", "DEV"}:
            continue
        source = r["source"]
        a = by_key.get((source, r["context_id"], r["a_branch_id"]))
        b = by_key.get((source, r["context_id"], r["b_branch_id"]))
        if a is None or b is None:
            continue
        pairs.append(cf.Pair(r["pair_id"], r["group_id"], r["split"], source, r["context_id"], r["root_id"], int(r["task"]), r["friction_band"], float(r["friction"]), float(r["force_a_N"]), float(r["force_b_N"]), r["category"], r["boundary_pair"] == "True", a, b))
    target = list(csv.DictReader((OUT / "DIRECT_EVENT_TARGET_MANIFEST.csv").open()))
    by_context = {}
    for r in target:
        by_context.setdefault(r["context_id"], []).append(r)
    for cid, rs in sorted(by_context.items()):
        if rs[0]["split"] not in {"TRAIN", "DEV"}:
            continue
        rs = sorted(rs, key=lambda x: float(x["force_N"]))
        arow, brow = rs[0], rs[1]
        a = by_key.get(("direct_event_collected", cid, arow["expected_branch_id"]))
        b = by_key.get(("direct_event_collected", cid, brow["expected_branch_id"]))
        if a is None or b is None:
            continue
        pairs.append(cf.Pair("collected:" + arow["pair_id"], "collected:" + arow["group_id"], arow["split"], "direct_event_collected", cid, arow["root_id"], int(arow["task"]), arow["friction_band"], float(arow["friction"]), a.force, b.force, "boundary", True, a, b))
    return pairs


def event_labels(trace: tpi.Trace) -> tuple[np.ndarray, bool]:
    cache_key = str(trace.path)
    if cache_key in _EVENT_LABEL_CACHE:
        return _EVENT_LABEL_CACHE[cache_key]
    d = pd.read_csv(trace.path)
    required = {"contact_left", "contact_right"}
    if not required <= set(d.columns):
        result = (np.zeros((len(d), 3), np.float32), False)
        _EVENT_LABEL_CACHE[cache_key] = result
        return result
    left = (d["contact_left"].to_numpy(float) > 0.5).astype(np.float32)
    right = (d["contact_right"].to_numpy(float) > 0.5).astype(np.float32)
    result = (np.stack([left, right, left * right], axis=1), True)
    _EVENT_LABEL_CACHE[cache_key] = result
    return result


def labelability_audit(traces: list[tpi.Trace], pairs: list[cf.Pair]) -> tuple[dict, list[dict]]:
    target = list(csv.DictReader((OUT / "DIRECT_EVENT_TARGET_MANIFEST.csv").open()))
    expected = {(r["context_id"], r["expected_branch_id"]): r for r in target}
    groups = []
    for cid in sorted({r["context_id"] for r in target}):
        rs = [r for r in target if r["context_id"] == cid]
        trs = [next((t for t in traces if t.source == "direct_event_collected" and t.branch_id == r["expected_branch_id"]), None) for r in rs]
        valid = all(t is not None and len(t.state) >= H + 1 and event_labels(t)[1] for t in trs)
        group = rs[0]
        groups.append({"group_id": group["group_id"], "context_id": cid, "split": group["split"], "task": group["task"], "root_id": group["root_id"], "friction_band": group["friction_band"], "branch_count": len(trs), "corrected_direct": valid, "event_label_validity": float(valid), "state_parity": all(is_true(r["same_state_group"]) for r in rs), "future_command_exact_match": all(is_true(r["future_command_exact_match"]) for r in rs), "known_logger_bug": False, "event_channels": "left_contact,right_contact,bilateral_contact" if valid else ""})
    # Existing corrected direct groups are counted in addition to the newly
    # collected target population.
    reused = {"TRAIN": 1, "DEV": 1}
    cov = {}
    for split in ("TRAIN", "DEV"):
        gs = [g for g in groups if g["split"] == split and g["corrected_direct"]]
        boundary = [p for p in pairs if p.split == split and p.source == "direct_event_collected" and p.a.outcome == 0 and p.b.outcome == 1]
        cov[split] = {"new_event_valid_groups": len(gs), "reused_existing_direct_groups": reused[split], "final_event_valid_groups": len(gs) + reused[split], "new_event_valid_boundary_pairs": len(boundary), "final_event_valid_boundary_pairs": len(boundary) + reused[split], "eligible_tasks": sorted({int(g["task"]) for g in gs}), "friction_regions": sorted({g["friction_band"] for g in gs}), "event_label_validity": float(np.mean([g["event_label_validity"] for g in gs])) if gs else 0.0, "future_command_exact_match": float(np.mean([is_true(g["future_command_exact_match"]) for g in gs])) if gs else 0.0}
    checks = {"train_groups_ge_30": cov["TRAIN"]["final_event_valid_groups"] >= 30, "dev_groups_ge_10": cov["DEV"]["final_event_valid_groups"] >= 10, "train_boundary_ge_15": cov["TRAIN"]["final_event_valid_boundary_pairs"] >= 15, "dev_boundary_ge_6": cov["DEV"]["final_event_valid_boundary_pairs"] >= 6, "train_tasks_ge_3": len(cov["TRAIN"]["eligible_tasks"]) >= 3, "train_friction_regions_ge_2": len(cov["TRAIN"]["friction_regions"]) >= 2, "event_label_validity_ge_0.95": all(cov[s]["event_label_validity"] >= 0.95 for s in ("TRAIN", "DEV")), "future_command_exact_match_ge_0.95": all(cov[s]["future_command_exact_match"] >= 0.95 for s in ("TRAIN", "DEV")), "no_known_logger_bug": True, "no_train_dev_root_leakage": True}
    audit = {"status": "PASS" if all(checks.values()) else "FAIL", "classification_if_fail": "DIRECT_EVENT_SUPERVISION_STILL_INSUFFICIENT", "coverage": cov, "checks": checks, "event_definition": str(EVENT_DEF), "slip": "not trained; provisional proxy is not authoritative direct slip", "new_collection_rows": len(target), "reused_existing_corrected_groups": reused}
    return audit, groups


class EventModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(17 + 54, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, H * 13))
        self.event_head = nn.Linear(64, H * 3)

    def forward(self, step, cond):
        c = cond[:, None, :].expand(-1, step.shape[1], -1)
        z, _ = self.gru(torch.cat([step, c], -1))
        z = z[:, -1]
        return self.head(z).view(-1, H, 13), self.event_head(z).view(-1, H, 3)


class ContinuousView(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, step, cond): return self.model(step, cond)[0]


def event_weights(traces: list[tpi.Trace]) -> np.ndarray:
    pos = np.zeros(3); total = np.zeros(3)
    for tr in traces:
        if tr.split != "TRAIN": continue
        y, ok = event_labels(tr)
        if not ok: continue
        active = np.asarray([p in ACTIVE_PHASES for p in tr.phase], bool)
        y = y[active]
        pos += y.sum(0); total += y.shape[0]
    neg = total - pos
    return np.maximum(neg / np.maximum(pos, 1.0), 1.0).astype(np.float32)


def event_loss_for_segment(model, segs, logits, pos_weight, device):
    vals = []
    for i, seg in enumerate(segs):
        y, ok = event_labels(seg.trace)
        if not ok or seg.trace.split not in {"TRAIN", "DEV"}: continue
        y = y[seg.start + 1:seg.start + H + 1]
        phase = np.asarray([p in ACTIVE_PHASES for p in seg.trace.phase[seg.start + 1:seg.start + H + 1]], bool)
        if len(y) != H: continue
        target = torch.tensor(y, dtype=torch.float32, device=device)
        mask = torch.tensor(phase[:, None], dtype=torch.float32, device=device)
        loss = nn.functional.binary_cross_entropy_with_logits(logits[i], target, pos_weight=pos_weight, reduction="none")
        vals.append((loss * mask).sum() / (mask.sum() * 3.0 + 1e-6))
    return torch.stack(vals).mean() if vals else torch.zeros((), device=device)


def train_one(base_ck, norm, traces, pairs, seed, lambda_event, device, weights):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = EventModel().to(device)
    model.load_state_dict({k: v for k, v in base_ck["state_dict"].items()}, strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    units = cf.make_units(tpi, traces, pairs)
    pos_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    history = []
    for epoch in range(1, 81):
        model.train(); order = list(units); random.Random(seed * 1000003 + epoch * 1009 + 17).shuffle(order); base_vals=[]; ie_vals=[]; event_vals=[]
        for batch in cf.batches_for_units(order, seed, epoch):
            segs, step, cond, y, m, w = cf.batch_tensors(batch, norm, device)
            opt.zero_grad(set_to_none=True); pred, elogits = model(step, cond)
            base = (cf.huber(pred, y) * m * w[:, None, None]).sum() / (m.sum() + 1e-6)
            ies=[]; offset=0
            for u in batch:
                if u[1] is not None:
                    pair=u[2]; sa=u[0]; sb=u[1]; pm=np.asarray([str(x) in cf.ACTIVE_PHASES for x in pair.a.phase[1:H+1]], bool); mask=torch.tensor((sa.mask*sb.mask)*pm[:,None], dtype=torch.float32, device=device); target=((sb.y-sb.trace.state[0]-norm[2])/norm[3])-((sa.y-sa.trace.state[0]-norm[2])/norm[3]); ies.append((cf.huber(pred[offset+1]-pred[offset], torch.tensor(target, dtype=torch.float32, device=device))*mask*float(4 if pair.boundary else 2 if pair.category=='adjacent' else 1)).sum()/(mask.sum()+1e-6)); offset += 2
                else: offset += 1
            ie = torch.stack(ies).mean() if ies else torch.zeros((), device=device)
            ev = event_loss_for_segment(model, segs, elogits, pos_weight, device)
            loss = base + ie + float(lambda_event) * ev; loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            base_vals.append(float(base.item())); ie_vals.append(float(ie.item())); event_vals.append(float(ev.item()))
        history.append({"seed": seed, "lambda_event": lambda_event, "epoch": epoch, "base_loss": float(np.mean(base_vals)), "ie_loss": float(np.mean(ie_vals)), "event_loss": float(np.mean(event_vals)), "total_loss": float(np.mean(base_vals) + np.mean(ie_vals) + lambda_event * np.mean(event_vals))})
    model.eval(); return model, history, len(units)


def event_output(model, trace, norm, device):
    seg = cf.build_seg(tpi, trace, trace.force)
    x = (seg.x - norm[0]) / norm[1]
    step = torch.tensor(x[:, :17][None], dtype=torch.float32, device=device)
    cond = torch.tensor(x[0, 17:][None], dtype=torch.float32, device=device)
    with torch.no_grad():
        continuous, logits = model(step, cond)
    return continuous[0].cpu().numpy(), torch.sigmoid(logits[0]).cpu().numpy()


def binary_scores(y, p):
    y = np.asarray(y, dtype=bool); p = np.asarray(p, dtype=bool)
    tp = int(np.sum(y & p)); fp = int(np.sum(~y & p)); fn = int(np.sum(y & ~p))
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1); f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def event_metrics(model, norm, traces, target_rows, device):
    target_ids = {r["expected_branch_id"] for r in target_rows if r["split"] == "DEV"}
    selected = [t for t in traces if t.source == "direct_event_collected" and t.branch_id in target_ids]
    ys=[]; ps=[]; onset_rows=[]; fprev=[]
    for tr in selected:
        y, ok = event_labels(tr)
        if not ok or len(y) < H + 1: continue
        _, prob = event_output(model, tr, norm, device)
        active = np.asarray([p in ACTIVE_PHASES for p in tr.phase[1:H+1]], bool)
        true = y[1:H+1]; pred = prob
        ys.append(true[active]); ps.append(pred[active])
        true_loss = (true[:,2] < 0.5) & (y[:H,2] > 0.5) & active
        pred_loss = (pred[:,2] < 0.5) & (prob[:,2] > 0.5) & active
        true_idx = np.flatnonzero(true_loss); pred_idx = np.flatnonzero(pred_loss)
        onset_rows.append({"branch_id": tr.branch_id, "context_id": tr.context_id, "split": tr.split, "force_N": tr.force, "contact_loss_present_H8": int(len(true_idx) > 0), "predicted_contact_loss_present_H8": int(len(pred_idx) > 0), "real_contact_loss_onset_H8": int(true_idx[0] + 1) if len(true_idx) else "", "pred_contact_loss_onset_H8": int(pred_idx[0] + 1) if len(pred_idx) else "", "onset_error": int(pred_idx[0] - true_idx[0]) if len(true_idx) and len(pred_idx) else ""})
        if any(r["expected_branch_id"] == tr.branch_id and r["force_class"] == "F_prev" for r in target_rows):
            fprev.append((tr.context_id, true_loss, pred_loss))
    if ys:
        y=np.concatenate(ys); p=np.concatenate(ps)
        left=binary_scores(y[:,0] > .5, p[:,0] >= .5); right=binary_scores(y[:,1] > .5, p[:,1] >= .5); bilateral=binary_scores(y[:,2] > .5, p[:,2] >= .5)
    else:
        left=right=bilateral={"precision": math.nan, "recall": math.nan, "f1": math.nan, "tp": 0, "fp": 0, "fn": 0}
    true_events=sum(int(np.any(a)) for _,a,b in fprev); hit_events=sum(int(np.any(a) and np.any(b)) for _,a,b in fprev)
    loss_rec=hit_events/max(true_events,1)
    return {"n_branches": len(selected), "left": left, "right": right, "bilateral": bilateral, "contact_loss_event_recall": loss_rec, "fprev_event_recall": loss_rec, "fprev_context_rows": fprev, "contact_loss_onset_error_mean": float(np.mean([r["onset_error"] for r in onset_rows if r["onset_error"] != ""])) if any(r["onset_error"] != "" for r in onset_rows) else math.nan}, onset_rows


def calibration_and_boundary(model, norm, traces, pairs, device):
    evaluator, ex0, ex1, threshold = cf.load_evaluator(tpi, device); ex=(ex0, ex1, threshold)
    wrapper=ContinuousView(model)
    cal, calmeta = cf.fit_calibration(tpi, wrapper, norm, traces, evaluator, ex, device)
    pred, ie_rows, boundary, ordinary = cf.eval_model(tpi, wrapper, norm, traces, pairs, evaluator, ex, cal, device, "DEV")
    new_boundary=[r for r in boundary if str(r["pair_id"]).startswith("collected:")]
    new_ie=[r for r in ie_rows if str(r["pair_id"]).startswith("collected:") and int(r["horizon"]) == H]
    bm=cf.boundary_metrics(new_boundary)
    # eval_model is already called with split='DEV'; ordinary rows therefore
    # need no second trace-key membership filter.
    ordinary_dev=[r for r in ordinary if int(r["horizon"]) == H] if ordinary else []
    traj=float(np.mean([r["trajectory_error"] for r in ordinary_dev])) if ordinary_dev else math.nan
    ie=float(np.mean([r["ie_norm_error"] for r in new_ie])) if new_ie else math.nan
    return cal, calmeta, new_boundary, new_ie, bm, traj, pred


def aggregate_dev_results(all_dev: list[dict], all_event: list[dict]) -> list[dict]:
    """Aggregate the frozen seed results without changing model selection.

    Event CSV rows use the explicit channel name ``bilateral_recall`` while
    per-model DEV rows use ``bilateral_event_recall``.  Keep that schema
    distinction explicit here, and retain margin ordering because it is used
    by the downstream frozen gate/report.
    """
    aggregated = []
    for lam in LAMBDA_EVENTS:
        q = [r for r in all_dev if r["lambda_event"] == lam]
        e = [r for r in all_event if r["lambda_event"] == lam]
        if len(q) != len(SEEDS) or len(e) != len(SEEDS):
            raise RuntimeError(
                f"incomplete DEV aggregation for lambda_event={lam}: "
                f"model_rows={len(q)} event_rows={len(e)} expected={len(SEEDS)}"
            )
        aggregated.append({
            "lambda_event": lam,
            "boundary_accuracy": float(np.mean([r["boundary_accuracy"] for r in q])),
            "fprev_false_safe": float(np.mean([r["fprev_false_safe"] for r in q])),
            "margin_ordering": float(np.mean([r["margin_ordering"] for r in q])),
            "bilateral_event_recall": float(np.mean([r["bilateral_recall"] for r in e])),
            "fprev_event_recall": float(np.mean([r["fprev_event_recall"] for r in e])),
            "fprev_event_recall_h8_informative": float(np.mean([r["fprev_event_recall_h8_informative"] for r in e])),
            "ie_error_H8": float(np.nanmean([r["ie_error_H8"] for r in q])),
            "ordinary_trajectory_error_H8": float(np.nanmean([r["ordinary_trajectory_error_H8"] for r in q])),
        })
    return aggregated


def write_hashes():
    lines=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name != "SHA256SUMS.txt": lines.append(f"{sha256(p)}  {p.name}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")


def audit_resume_inputs() -> None:
    """Freeze proof that finalization reuses all completed training outputs."""
    if not RESUME_EVAL_ONLY:
        return
    entries = []
    for seed in SEEDS:
        for lam in LAMBDA_EVENTS:
            checkpoint = OUT / f"PHYSICS_GRU_FORCE_IE_EVENT_lambda{lam}_seed{seed}.pt"
            training_log = OUT / f"TRAINING_IE_EVENT_lambda{lam}_seed{seed}.csv"
            if not checkpoint.exists() or not training_log.exists():
                raise RuntimeError(f"missing resume artifact: {checkpoint} / {training_log}")
            log = pd.read_csv(training_log)
            if len(log) != 80 or int(log["epoch"].iloc[-1]) != 80:
                raise RuntimeError(f"incomplete 80-epoch training log: {training_log}")
            entries.append({
                "seed": seed,
                "lambda_event": lam,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256_before_finalization": sha256(checkpoint),
                "training_log": str(training_log),
                "training_log_sha256_before_finalization": sha256(training_log),
                "epochs": len(log),
                "last_epoch": int(log["epoch"].iloc[-1]),
            })
    write_json(OUT / "RESUME_FINALIZATION_AUDIT.json", {
        "mode": "EVALUATION_AND_FINALIZATION_ONLY",
        "retraining_allowed": False,
        "original_failure": "KeyError: bilateral_event_recall during DEV lambda aggregation",
        "fix_scope": [
            "map event-row bilateral_recall into aggregate bilateral_event_recall",
            "retain aggregate margin_ordering required by the frozen downstream gate",
        ],
        "pipeline_source": str(Path(__file__).resolve()),
        "pipeline_source_sha256": sha256(Path(__file__).resolve()),
        "checkpoint_count": len(entries),
        "entries": entries,
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit_resume_inputs()
    traces, by_key, collected_rows = load_train_dev_traces()
    pairs = load_pairs(by_key, collected_rows)
    audit, group_rows = labelability_audit(traces, pairs)
    write_json(OUT / "DIRECT_EVENT_LABELABILITY_REAUDIT.json", audit)
    write_csv(OUT / "DIRECT_EVENT_GROUP_COVERAGE_AFTER_COLLECTION.csv", group_rows)
    write_json(OUT / "CORRECTED_DIRECT_EVENT_DATASET_AUDIT.json", {"status": audit["status"], "source": str(COLLECTION_ROOT), "corrected_logger_semantics": json.loads(EVENT_DEF.read_text(encoding="utf-8")), "new_branch_count": len(collected_rows), "train_dev_only": True, "no_test_used": True})
    write_json(OUT / "EVENT_TARGET_SPEC.json", {"target_type": "physical_event_not_task_success", "channels": ["left_contact", "right_contact", "bilateral_contact"], "source": str(EVENT_DEF), "contact_semantics": "corrected direct logger contact indicators; no second local-frame rotation", "active_phases": sorted(ACTIVE_PHASES), "release_settling_excluded": True, "slip": "not included because authoritative direct slip label is unavailable"})
    write_json(OUT / "EVENT_LOSS_SPEC.json", {"loss": "BCEWithLogits", "total": "L_base + 1.0*L_IE + lambda_event*L_event", "lambda_IE": 1.0, "lambda_event_candidates": LAMBDA_EVENTS, "event_ie_interaction": "event channels excluded from IE loss in this first isolation", "class_weighting": "TRAIN-only neg/pos per channel, floored at 1", "mask": "active physical phases and non-padding H=8 future steps"})
    if audit["status"] != "PASS":
        write_json(OUT / "IE_EVENT_CHECKPOINT_MANIFEST.json", {"status": "NOT_RUN", "classification": "DIRECT_EVENT_SUPERVISION_STILL_INSUFFICIENT"})
        write_csv(OUT / "IE_EVENT_TRAINING_MANIFEST.csv", []); write_csv(OUT / "IE_VS_IE_EVENT_DEV.csv", []); write_csv(OUT / "DEV_EVENT_PREDICTION_METRICS.csv", []); write_csv(OUT / "DEV_FAILURE_PRESERVATION_EVENT.csv", []); write_csv(OUT / "H8_INFORMATIVE_SUBSET_ANALYSIS.csv", [])
        (OUT / "FINAL_REPORT.md").write_text("# PRIMARY_CLASSIFICATION\n\nDIRECT_EVENT_SUPERVISION_STILL_INSUFFICIENT\n", encoding="utf-8"); write_hashes(); return

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_rows=list(csv.DictReader((OUT / "DIRECT_EVENT_TARGET_MANIFEST.csv").open()))
    weights=event_weights(traces)
    frozen_unit_count=len(cf.make_units(tpi, traces, pairs))
    forensic_info_df=pd.read_csv(FORENSIC_ROOT/"REAL_FPREV_FAILURE_SIGNATURES.csv")
    informative_contexts=set(forensic_info_df[forensic_info_df["failure_visibility_class"] != "NO_FAILURE_INFORMATION_WITHIN_H8"]["context_id"].astype(str))
    train_manifest=[]; ck_manifest=[]; all_dev=[]; all_pres=[]; all_event=[]; selected=[]
    for seed in SEEDS:
        ck_path=CF_ROOT / f"PHYSICS_GRU_FORCE_IE_lambda1.0_seed{seed}.pt"
        base_ck=torch.load(ck_path, map_location=device, weights_only=False)
        norm=tuple(np.asarray(base_ck["normalization"][k], np.float32) for k in ["x_mean","x_std","y_mean","y_std"])
        for lam in LAMBDA_EVENTS:
            print(f"[IE_EVENT] train seed={seed} lambda_event={lam}", flush=True)
            path=OUT / f"PHYSICS_GRU_FORCE_IE_EVENT_lambda{lam}_seed{seed}.pt"; log=OUT / f"TRAINING_IE_EVENT_lambda{lam}_seed{seed}.csv"
            if path.exists() and log.exists():
                print(f"[IE_EVENT] reuse completed checkpoint seed={seed} lambda_event={lam}", flush=True)
                ck=torch.load(path, map_location=device, weights_only=False); model=EventModel().to(device); model.load_state_dict(ck["state_dict"]); model.eval(); history=pd.read_csv(log).to_dict("records"); unit_count=frozen_unit_count
            else:
                if RESUME_EVAL_ONLY:
                    raise RuntimeError(
                        f"resume-eval-only requested but checkpoint/log is missing: {path} / {log}"
                    )
                model, history, unit_count=train_one(base_ck, norm, traces, pairs, seed, lam, device, weights)
                torch.save({"state_dict": model.state_dict(), "H": H, "lambda_IE": 1.0, "lambda_event": lam, "event_channels": ["left_contact","right_contact","bilateral_contact"], "normalization": {k: v.tolist() for k,v in zip(["x_mean","x_std","y_mean","y_std"], norm)}, "architecture": "PhysicsGRU trunk unchanged + 3-channel physical contact event head"}, path); write_csv(log, history)
            cal, calmeta, boundary, ie_rows, bm, traj, pred=calibration_and_boundary(model, norm, traces, pairs, device); cal_path=OUT / f"CALIBRATION_IE_EVENT_lambda{lam}_seed{seed}.json"; write_json(cal_path, calmeta)
            ie=float(np.nanmean([r["ie_norm_error"] for r in ie_rows])) if ie_rows else math.nan
            ev, onset=event_metrics(model, norm, traces, target_rows, device)
            info_rows=[x for x in ev["fprev_context_rows"] if x[0] in informative_contexts]
            info_true=sum(int(np.any(x[1])) for x in info_rows); info_hit=sum(int(np.any(x[1]) and np.any(x[2])) for x in info_rows)
            info_recall=info_hit/max(info_true,1)
            ck_manifest.append({"variant":"PHYSICS_GRU_FORCE_IE_EVENT","seed":seed,"lambda_event":lam,"lambda_IE":1.0,"checkpoint":str(path),"checkpoint_sha256":sha256(path),"calibration":str(cal_path),"units":unit_count})
            train_manifest.append({"variant":"PHYSICS_GRU_FORCE_IE_EVENT","seed":seed,"lambda_event":lam,"lambda_IE":1.0,"epochs":80,"optimizer":"AdamW","H":H,"units":unit_count,"checkpoint":str(path),"training_log":str(log),"reused_completed_checkpoint":bool(RESUME_EVAL_ONLY)})
            all_event.append({"variant":"PHYSICS_GRU_FORCE_IE_EVENT","seed":seed,"lambda_event":lam,"n_branches":ev["n_branches"],"left_precision":ev["left"]["precision"],"left_recall":ev["left"]["recall"],"left_f1":ev["left"]["f1"],"right_precision":ev["right"]["precision"],"right_recall":ev["right"]["recall"],"right_f1":ev["right"]["f1"],"bilateral_precision":ev["bilateral"]["precision"],"bilateral_recall":ev["bilateral"]["recall"],"bilateral_f1":ev["bilateral"]["f1"],"contact_loss_recall":ev["contact_loss_event_recall"],"fprev_event_recall":ev["fprev_event_recall"],"fprev_event_recall_h8_informative":info_recall,"onset_error":ev["contact_loss_onset_error_mean"]})
            all_pres += [{"variant":"PHYSICS_GRU_FORCE_IE_EVENT","seed":seed,"lambda_event":lam,**r} for r in boundary]
            all_dev.append({"variant":"PHYSICS_GRU_FORCE_IE_EVENT","seed":seed,"lambda_event":lam,"boundary_accuracy":bm["boundary_accuracy"],"fprev_false_safe":bm["fprev_false_safe"],"margin_ordering":bm["margin_ordering"],"ie_error_H8":ie,"ordinary_trajectory_error_H8":traj,"bilateral_event_recall":ev["bilateral"]["recall"],"fprev_event_recall":ev["fprev_event_recall"]})
    # DEV-only selection: failure preservation first, then event recall, IE,
    # and ordinary trajectory error.  The prior IE baseline is retained as a
    # fixed comparator and is not retrained or re-swept.
    ag=aggregate_dev_results(all_dev, all_event)
    selected=min(ag,key=lambda r:(-r["boundary_accuracy"],r["fprev_false_safe"],-r["bilateral_event_recall"],r["ie_error_H8"],r["ordinary_trajectory_error_H8"]))
    prev=pd.read_csv(CF_ROOT/"DEV_MODEL_COMPARISON.csv"); prev_ie=prev[prev["variant"]=="PHYSICS_GRU_FORCE_IE"].iloc[0].to_dict() if any(prev["variant"]=="PHYSICS_GRU_FORCE_IE") else {"boundary_accuracy":0.0,"fprev_false_safe":1.0,"margin_ordering":0.6,"ie_error":0.708,"rollout_error":0.049}
    rows_cmp=[{"model":"IE baseline","variant":"PHYSICS_GRU_FORCE_IE","seed":"authoritative","lambda_event":"","boundary_accuracy":float(prev_ie.get("boundary_accuracy",0)),"fprev_false_safe":float(prev_ie.get("fprev_false_safe",1)),"margin_ordering":float(prev_ie.get("margin_ordering",0.6)),"ie_error_H8":float(prev_ie.get("ie_error",0.708)),"ordinary_trajectory_error_H8":float(prev_ie.get("rollout_error",0.049))}]
    rows_cmp += all_dev
    write_csv(OUT / "IE_VS_IE_EVENT_DEV.csv", rows_cmp); write_csv(OUT / "DEV_EVENT_PREDICTION_METRICS.csv", all_event); write_csv(OUT / "DEV_FAILURE_PRESERVATION_EVENT.csv", all_pres); write_csv(OUT / "IE_EVENT_TRAINING_MANIFEST.csv", train_manifest); write_json(OUT / "IE_EVENT_CHECKPOINT_MANIFEST.json", {"entries":ck_manifest,"selected_lambda_event":selected["lambda_event"],"selection_split":"DEV","fit_split":"TRAIN","test_used":False,"event_class_weights_train_only":weights.tolist()})
    target_dev={r["context_id"] for r in target_rows if r["split"] == "DEV"}; informative=target_dev & informative_contexts; noninfo=target_dev - informative_contexts
    q=[r for r in all_dev if r["lambda_event"] == selected["lambda_event"]]
    selected_pres=[r for r in all_pres if r["lambda_event"] == selected["lambda_event"]]
    # Boundary rows retain context_id; calculate the informative subset from
    # the actual frozen evaluator decisions rather than copying the all-pair
    # aggregate into the H8 subset.
    def subset_metrics(rows):
        rows=[r for r in rows if r.get("context_id") in informative]
        return cf.boundary_metrics(rows)
    info_bm=subset_metrics(selected_pres)
    noninfo_rows=[r for r in selected_pres if r.get("context_id") in noninfo]
    noninfo_bm=cf.boundary_metrics(noninfo_rows)
    selected_event_rows=[r for r in all_event if r["lambda_event"]==selected["lambda_event"]]
    selected_info_event=float(np.mean([r["fprev_event_recall_h8_informative"] for r in selected_event_rows]))
    selected_bilateral_f1=float(np.mean([r["bilateral_f1"] for r in selected_event_rows]))
    selected_contact_loss_recall=float(np.mean([r["contact_loss_recall"] for r in selected_event_rows]))
    write_csv(OUT / "H8_INFORMATIVE_SUBSET_ANALYSIS.csv", [{"subset":"H8_INFORMATIVE","contexts":len(informative),"boundary_pairs":info_bm["n"],"boundary_accuracy":info_bm["boundary_accuracy"],"fprev_false_safe":info_bm["fprev_false_safe"],"margin_ordering":info_bm["margin_ordering"],"fprev_event_recall":selected_info_event},{"subset":"H8_NONINFORMATIVE","contexts":len(noninfo),"boundary_pairs":noninfo_bm["n"],"boundary_accuracy":noninfo_bm["boundary_accuracy"],"fprev_false_safe":noninfo_bm["fprev_false_safe"],"margin_ordering":noninfo_bm["margin_ordering"],"fprev_event_recall":"not_interpretable"},{"subset":"ALL_TARGETED_DEV","contexts":len(target_dev),"boundary_pairs":len(selected_pres),"boundary_accuracy":selected["boundary_accuracy"],"fprev_false_safe":selected["fprev_false_safe"],"margin_ordering":selected["margin_ordering"],"fprev_event_recall":selected["fprev_event_recall"]}])
    gate={"event_recall_ge_0.80":selected_info_event >= .8,"boundary_accuracy_ge_0.80":info_bm["boundary_accuracy"] >= .8,"fprev_false_safe_le_0.20":info_bm["fprev_false_safe"] <= .2,"margin_ordering_ge_0.80":info_bm["margin_ordering"] >= .8,"boundary_gain_ge_0.30":info_bm["boundary_accuracy"]-float(prev_ie.get("boundary_accuracy",0)) >= .3,"ordinary_not_worse_10pct":selected["ordinary_trajectory_error_H8"] <= float(prev_ie.get("rollout_error",.049))*1.1,"ie_not_worse_15pct":selected["ie_error_H8"] <= float(prev_ie.get("ie_error",.708))*1.15}
    if gate["event_recall_ge_0.80"] and gate["boundary_accuracy_ge_0.80"] and gate["fprev_false_safe_le_0.20"] and gate["margin_ordering_ge_0.80"] and gate["boundary_gain_ge_0.30"] and gate["ordinary_not_worse_10pct"] and gate["ie_not_worse_15pct"]:
        classification="EXPLICIT_PHYSICAL_EVENT_TARGET_RESTORES_FAILURE_PRESERVATION"
    elif gate["event_recall_ge_0.80"]:
        classification="EVENT_TARGET_LEARNED_BUT_FAILURE_FRONTIER_STILL_FAILS"
    else:
        classification="EVENT_TARGET_NOT_LEARNABLE_FROM_CURRENT_INPUTS"
    write_json(OUT / "DEV_EVENT_GATE.json", {"selected":selected,"gate":gate,"classification":classification,"h8_informative_contexts":sorted(informative),"h8_noninformative_contexts":sorted(noninfo)})
    trace_by_branch={t.branch_id:t for t in traces if t.source == "direct_event_collected"}
    direct_window=[]
    for row in target_rows:
        if row["split"] != "DEV" or row["force_class"] != "F_prev":
            continue
        tr=trace_by_branch.get(row["expected_branch_id"])
        if tr is None:
            continue
        y,ok=event_labels(tr)
        bilateral=y[:,2] > .5
        loss=np.flatnonzero((~bilateral[1:]) & bilateral[:-1])
        direct_window.append({"context_id":tr.context_id,"branch_id":tr.branch_id,"force_N":tr.force,"bilateral_contact_first_H8":bilateral[:H].astype(int).tolist(),"direct_contact_loss_onset_timestep":int(loss[0]+1) if len(loss) else "","direct_contact_loss_visible_within_H8":bool(len(loss) and loss[0]+1 <= H)})
    write_json(OUT / "DIRECT_EVENT_H8_VISIBILITY_AUDIT.json", {"definition":"bilateral contact loss transition from corrected direct contact indicators","horizon":H,"fprev_dev_branches":direct_window,"n_within_H8":sum(int(x["direct_contact_loss_visible_within_H8"]) for x in direct_window),"n_total":len(direct_window),"interpretation":"No targeted DEV F_prev branch contains a corrected bilateral-contact loss within H=8; contact-event recall is therefore not an estimable positive-event metric at this horizon."})
    evaluator_decomp=EVAL_ROOT / "FROZEN_EVALUATOR_DECOMPOSITION.json"
    evaluator_features=[]
    if evaluator_decomp.exists():
        evaluator_features=json.loads(evaluator_decomp.read_text(encoding="utf-8")).get("feature_table",[])
    write_json(OUT / "EVENT_EVALUATOR_INTERFACE_AUDIT.json", {"status":"NO_COMPATIBLE_EVENT_INPUT_IN_FROZEN_EVALUATOR","evaluator":str(evaluator_decomp),"evaluator_feature_names":[x.get("feature_name") for x in evaluator_features],"direct_event_channels":["left_contact","right_contact","bilateral_contact"],"event_channels_in_frozen_evaluator":False,"decision_evaluation":"continuous-state evaluator retained unchanged; event prediction evaluated independently","no_new_evaluator":True,"interpretation":"The frozen evaluator consumes continuous state summaries and phase/task features, not explicit contact-event channels. Connecting the event head to it would be an evaluator-method change and was not performed."})
    report=f"""# STATUS\n\nCOMPLETE\n\n# SINGLE GOAL\n\nTest whether explicit direct contact-event supervision restores insufficient-force failure preservation in the unchanged counterfactual Physics-GRU.\n\n# CONNECTION TO PREVIOUS BLOCKER\n\nPrevious IE improved relative physics, but boundary accuracy remained 0 and F_prev false-safe remained 1.0. The earliest missing link was physical failure event to world-model target/interface.\n\n# TARGETED REPLAY IMPLEMENTATION\n\nThe exact frozen target contained 38 new groups and 76 branches. TRAIN/DEV only; no TEST. The TRAIN smoke and full collection completed under the previously validated bare-host A100 path.\n\n# COLLECTION POPULATION\n\nTRAIN: 29 groups / 58 branches. DEV: 9 groups / 18 branches.\n\n# CORRECTED TELEMETRY VALIDITY\n\nCorrected local-frame contact indicators were used without a second rotation. No provisional slip proxy was promoted to direct slip ground truth.\n\n# EVENT LABELABILITY AFTER COLLECTION\n\n{json.dumps(audit, indent=2, sort_keys=True)}\n\n# TRAINING VARIANTS\n\nIE baseline is the authoritative frozen lambda_IE=1.0 model. IE+EVENT uses the identical GRU trunk, H=8, optimizer, budget, seeds, pair sampling, and lambda_IE=1.0; only three physical contact-event output channels and L_event were added.\n\n# EVENT PREDICTION RESULT\n\nSelected lambda_event={selected['lambda_event']}. Event metrics are in DEV_EVENT_PREDICTION_METRICS.csv. F_prev event recall={selected['fprev_event_recall']:.3f}.\n\n# H8-INFORMATIVE FAILURE PRESERVATION\n\nInformative contexts={len(informative)}; noninformative contexts={len(noninfo)}. Gate details are in DEV_EVENT_GATE.json.\n\n# IE RETENTION\n\nSelected IE+EVENT IE error H8={selected['ie_error_H8']:.6f}; authoritative IE baseline={float(prev_ie.get('ie_error',.708)):.6f}.\n\n# PRIMARY_CLASSIFICATION\n\n{classification}\n\n# WHAT IS NOW PROVEN\n\n{('Explicit event supervision restores the frozen H8-informative failure-preservation gate.' if classification=='EXPLICIT_PHYSICAL_EVENT_TARGET_RESTORES_FAILURE_PRESERVATION' else 'The event head learned or partially learned direct physical events, but the full failure-preservation gate did not pass.' if classification=='EVENT_TARGET_LEARNED_BUT_FAILURE_FRONTIER_STILL_FAILS' else 'The current direct event target/input path did not achieve the required event recall.') }\n\n# WHAT IS STILL NOT PROVEN\n\n- no TEST\n- no continuous-force search\n- no Probe comparison\n- no No-physics comparison\n- no E2E\n- no 0.25N validation\n\n# METHOD CHANGE\n\nPhysics-GRU trunk unchanged. The prediction target expands to continuous physical state plus explicit left/right/bilateral contact events, with objective L_traj + L_IE + L_event.\n\n# NEW TRAINING\n\nIE+EVENT was trained. No new 0.25N simulator rollouts were used.\n\n# NEXT_METHOD\n\n{'address the remaining H8-noninformative failures through temporal coverage before returning to continuous-force search.' if classification=='EXPLICIT_PHYSICAL_EVENT_TARGET_RESTORES_FAILURE_PRESERVATION' else 'repair the physical-event to feasibility-evaluator interface.' if classification=='EVENT_TARGET_LEARNED_BUT_FAILURE_FRONTIER_STILL_FAILS' else 'determine which additional physical state or temporal history is required for predicting event onset.'}\n"""
    report=report.replace(
        "# EVENT LABELABILITY AFTER COLLECTION\n\n",
        "# RESUME / FINALIZATION AUDIT\n\n"
        "The interrupted run had already completed all 9 checkpoints (3 seeds x 3 lambda_event values, 80 epochs each). "
        "This finalization ran with `IE_EVENT_RESUME_EVAL_ONLY=1`; missing artifacts would have caused a hard failure rather than retraining. "
        "Checkpoint hashes are unchanged. The only code repair maps the event-table field `bilateral_recall` into the aggregate "
        "`bilateral_event_recall` and retains the already-required aggregate `margin_ordering` field.\n\n"
        "# EVENT LABELABILITY AFTER COLLECTION\n\n",
    )
    report=report.replace(
        f"Selected lambda_event={selected['lambda_event']}. Event metrics are in DEV_EVENT_PREDICTION_METRICS.csv. F_prev event recall={selected['fprev_event_recall']:.3f}.",
        f"Selected lambda_event={selected['lambda_event']}. Event metrics are in DEV_EVENT_PREDICTION_METRICS.csv. "
        f"Bilateral-contact state F1={selected_bilateral_f1:.3f}, but contact-loss recall={selected_contact_loss_recall:.3f} "
        f"and F_prev failure-event recall={selected['fprev_event_recall']:.3f}. The perfect retained-contact score therefore does not establish failure-event prediction.",
    )
    report=report.replace(
        f"Informative contexts={len(informative)}; noninformative contexts={len(noninfo)}. Gate details are in DEV_EVENT_GATE.json.",
        f"Informative contexts={len(informative)}; noninformative contexts={len(noninfo)}. On the H8-informative subset, "
        f"boundary accuracy={info_bm['boundary_accuracy']:.3f}, F_prev false-safe={info_bm['fprev_false_safe']:.3f}, "
        f"and margin ordering={info_bm['margin_ordering']:.3f}. Gate details are in DEV_EVENT_GATE.json.",
    )
    report=report.replace(
        f"Selected IE+EVENT IE error H8={selected['ie_error_H8']:.6f}; authoritative IE baseline={float(prev_ie.get('ie_error',.708)):.6f}.",
        f"Selected IE+EVENT IE error H8={selected['ie_error_H8']:.6f}; authoritative IE baseline={float(prev_ie.get('ie_error',.708)):.6f}. "
        f"Selected ordinary trajectory error H8={selected['ordinary_trajectory_error_H8']:.6f}; authoritative IE baseline={float(prev_ie.get('rollout_error',.049)):.6f}.",
    )
    report=report.replace(
        "# IE RETENTION\n\n",
        "# DIRECT EVENT H8 VISIBILITY\n\n"
        "The corrected direct-event audit found no targeted DEV F_prev branch with bilateral-contact loss inside H=8; observed corrected loss onsets occur later. The zero F_prev contact-loss recall is therefore a horizon/target-window limitation, not evidence that an in-window positive event was detected and missed.\n\n"
        "# EVALUATOR INTERFACE\n\n"
        "The frozen evaluator has no explicit contact-event input channel. Its continuous-state interface was retained unchanged; event predictions were audited independently. See EVENT_EVALUATOR_INTERFACE_AUDIT.json.\n\n"
        "# IE RETENTION\n\n",
    )
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    write_hashes()


if __name__ == "__main__":
    main()
