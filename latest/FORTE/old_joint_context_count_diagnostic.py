#!/usr/bin/env python3
"""Retrospective fixed-subset context-count diagnostic for old Feas/Joint.

No architecture, objective, hyperparameter, epoch, seed, or evaluator rule is
selected from the results.  The old pooled TRAIN data are filtered only by a
frozen nested list of distinct context IDs.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import gnp_style_continuous as gnp


ROOT = Path("/home/exouser/FORTE")
OLD = ROOT / "gnp_style_continuous_20260830_125107"
OUT = ROOT / "old_joint_context_count_diagnostic_20260831"
SETTINGS = ["TASK0_ONLY_N18", "POOLED_N18", "POOLED_N36", "POOLED_N54", "POOLED_N72"]


def frozen_subsets(protocol: dict) -> dict[str, list[str]]:
    by_task: dict[int, list[dict]] = defaultdict(list)
    for c in protocol["train_context_population"]:
        by_task[int(c["task"])].append(c)
    for task in by_task:
        by_task[task] = sorted(by_task[task], key=lambda x: str(x["context_id"]))
    tasks = [0, 1, 5, 6]
    pooled = []
    for i in range(max(len(by_task[t]) for t in tasks)):
        for task in tasks:
            if i < len(by_task[task]):
                pooled.append(str(by_task[task][i]["context_id"]))
    if len(pooled) != 72 or len(set(pooled)) != 72:
        raise RuntimeError("pooled context order is not a permutation of 72 contexts")
    ans = {
        "TASK0_ONLY_N18": [str(x["context_id"]) for x in by_task[0]],
        "POOLED_N18": pooled[:18],
        "POOLED_N36": pooled[:36],
        "POOLED_N54": pooled[:54],
        "POOLED_N72": pooled[:72],
    }
    if not (set(ans["POOLED_N18"]) < set(ans["POOLED_N36"]) <
            set(ans["POOLED_N54"]) < set(ans["POOLED_N72"])):
        raise RuntimeError("nested subset invariant failed")
    return ans


def subset_pairs(cf, physical, meta):
    by = defaultdict(list)
    for tr in physical:
        m = meta[tr.branch_id]
        by[(tr.context_id, m.repeat_index)].append(tr)
    pairs = []
    for (cid, repeat), q in sorted(by.items()):
        q = sorted(q, key=lambda t: meta[t.branch_id].stratum_index)
        if len(q) != 5:
            raise RuntimeError(f"incomplete force strata in {cid}/R{repeat}")
        for a, b in zip(q[:-1], q[1:]):
            ma, mb = meta[a.branch_id], meta[b.branch_id]
            pairs.append(cf.Pair(
                f"subset:{cid}:r{repeat}:s{ma.stratum_index}_vs_s{mb.stratum_index}",
                f"subset:{cid}:repeat{repeat}", "TRAIN", "CORRECTED_CONTINUOUS",
                cid, a.root_id, a.task, ma.friction_band, a.mu, a.force, b.force,
                "adjacent", False, a, b,
            ))
    return pairs


def train_or_load(setting, cids, all_traces, all_physical, all_meta, tpi, cf, full, device, out):
    traces = [x for x in all_traces if x.context_id in cids]
    physical = [x for x in all_physical if x.context_id in cids]
    meta = {x.branch_id: all_meta[x.branch_id] for x in traces}
    pairs = subset_pairs(cf, physical, meta)
    segs = gnp.start_segments(cf, tpi, traces)
    norm = gnp.fit_shared_norm(traces, physical, segs, tpi)
    models = {"FEASIBILITY_ONLY": [], "JOINT": []}
    calibrations = {"FEASIBILITY_ONLY": [], "JOINT": []}
    checkpoint_rows = []
    histories = []
    for seed in gnp.SEEDS:
        fpath = out / f"{setting}_FEAS_seed{seed}.pt"
        if fpath.exists():
            ck = torch.load(fpath, map_location=device, weights_only=False)
            fm = full.FeasibilityOnly().to(device)
            fm.load_state_dict(ck["state_dict"])
            fm.eval()
            fsteps = int(ck["optimizer_steps"])
        else:
            fm, hist, fsteps = gnp.train_feas_seed(full, traces, segs, norm, meta, device, seed)
            histories.extend({"setting": setting, **x} for x in hist)
            torch.save({
                "diagnostic_only": True, "setting": setting, "backend": "FEASIBILITY_ONLY",
                "seed": seed, "state_dict": fm.state_dict(), "epochs": gnp.EPOCHS,
                "optimizer_steps": fsteps, "context_ids": sorted(cids),
                "outcome_branches": len(traces), "physical_branches": len(physical),
            }, fpath)
        flog = gnp.model_logits(fm, "FEAS", traces, segs, norm, device)
        fcal = cf.fit_iso([flog[t.branch_id] for t in traces], [t.outcome for t in traces])
        models["FEASIBILITY_ONLY"].append(fm)
        calibrations["FEASIBILITY_ONLY"].append((np.asarray(fcal["x"]), np.asarray(fcal["y"])))
        checkpoint_rows.append({
            "setting": setting, "backend": "FEASIBILITY_ONLY", "seed": seed,
            "checkpoint": str(fpath), "sha256": gnp.sha256(fpath),
            "optimizer_steps": fsteps,
        })

        jpath = out / f"{setting}_JOINT_seed{seed}.pt"
        base_ck, _, base_path = full.load_base(tpi, seed, device)
        if jpath.exists():
            ck = torch.load(jpath, map_location=device, weights_only=False)
            jm = full.JointIEFeasibility(tpi, base_ck["state_dict"]).to(device)
            jm.load_state_dict(ck["state_dict"])
            jm.eval()
            jsteps = int(ck["optimizer_steps"])
            units = int(ck["physical_units"])
        else:
            jm, hist, jsteps, units, base_path = gnp.train_joint_seed(
                full, cf, tpi, traces, physical, pairs, segs, norm, meta, device, seed
            )
            histories.extend({"setting": setting, **x} for x in hist)
            torch.save({
                "diagnostic_only": True, "setting": setting, "backend": "JOINT",
                "seed": seed, "state_dict": jm.state_dict(), "epochs": gnp.EPOCHS,
                "optimizer_steps": jsteps, "physical_units": units,
                "lambda_physics": 1.0, "lambda_IE": 1.0,
                "lambda_feasibility": gnp.LAMBDA_FEAS_AUTHORITATIVE,
                "context_ids": sorted(cids), "outcome_branches": len(traces),
                "physical_branches": len(physical), "adjacent_IE_pairs": len(pairs),
                "initial_checkpoint": str(base_path),
                "initial_checkpoint_sha256": gnp.sha256(Path(base_path)),
            }, jpath)
        jlog = gnp.model_logits(jm, "JOINT", traces, segs, norm, device)
        jcal = cf.fit_iso([jlog[t.branch_id] for t in traces], [t.outcome for t in traces])
        models["JOINT"].append(jm)
        calibrations["JOINT"].append((np.asarray(jcal["x"]), np.asarray(jcal["y"])))
        checkpoint_rows.append({
            "setting": setting, "backend": "JOINT", "seed": seed,
            "checkpoint": str(jpath), "sha256": gnp.sha256(jpath),
            "optimizer_steps": jsteps, "physical_units": units,
        })
    if histories:
        old = []
        hp = out / "OLD_CONTEXT_COUNT_TRAINING_MANIFEST.csv"
        if hp.exists():
            old = pd.read_csv(hp).to_dict("records")
        gnp.write_csv(hp, old + histories)
    return traces, physical, pairs, norm, models, calibrations, checkpoint_rows


def evaluate(setting, models, calibrations, norm, tpi, cf, contexts, curves, frontiers, device):
    cids = sorted(curves.context_id.astype(str).unique())
    gt_mu = {cid: [float(contexts[cid].mu_gt)] for cid in cids}
    metric_rows = []
    decision_rows = []
    original_seeds = list(gnp.SEEDS)
    for backend in ["FEASIBILITY_ONLY", "JOINT"]:
        gnp.SEEDS = original_seeds
        stack = (models[backend], calibrations[backend])
        preds, dec = gnp.eval_backend_condition(
            backend, "GT", gt_mu, contexts, curves, frontiers, tpi, cf, stack, norm, device
        )
        metric_rows.append({
            "setting": setting, "aggregation": "ENSEMBLE",
            **gnp.summarize_metrics(backend, "GT", dec, curves, preds, "RAW_ENSEMBLE_PRIMARY"),
        })
        decision_rows.extend({"setting": setting, "aggregation": "ENSEMBLE", **x} for x in dec)
        for idx, seed in enumerate(original_seeds):
            gnp.SEEDS = [seed]
            sp, sd = gnp.eval_backend_condition(
                backend, "GT", gt_mu, contexts, curves, frontiers, tpi, cf,
                ([models[backend][idx]], [calibrations[backend][idx]]), norm, device
            )
            metric_rows.append({
                "setting": setting, "aggregation": f"SEED_{seed}",
                **gnp.summarize_metrics(backend, "GT", sd, curves, sp, "RAW_ENSEMBLE_PRIMARY"),
            })
    gnp.SEEDS = original_seeds
    return metric_rows, decision_rows


def main(out: Path, requested_device: str):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    protocol = json.loads((OLD / "GNP_STYLE_CONTINUOUS_TRAINING_PROTOCOL.json").read_text())
    subsets = frozen_subsets(protocol)
    context_meta = {str(x["context_id"]): x for x in protocol["train_context_population"]}
    manifest = {
        "status": "FROZEN_RETROSPECTIVE_MECHANISM_DIAGNOSTIC",
        "created_before_subset_training": True,
        "selection_rule": "lexicographic context_id within task; round-robin tasks [0,1,5,6]",
        "nested": ["POOLED_N18", "POOLED_N36", "POOLED_N54", "POOLED_N72"],
        "architecture_loss_seed_epoch_evaluator_changed": False,
        "diagnostic_only_not_cross_task_transfer_claim": True,
        "subsets": {},
    }
    for setting, ids in subsets.items():
        by_task = defaultdict(int)
        roots = set()
        for cid in ids:
            by_task[int(context_meta[cid]["task"])] += 1
            roots.add(str(context_meta[cid]["root_id"]))
        manifest["subsets"][setting] = {
            "context_count": len(ids), "simulator_root_cluster_count": len(roots),
            "task_context_counts": dict(sorted(by_task.items())), "context_ids": ids,
        }
    mp = out / "OLD_CONTEXT_COUNT_SUBSET_MANIFEST.json"
    if mp.exists():
        frozen = json.loads(mp.read_text())["subsets"]
        current_ids = {k: v["context_ids"] for k, v in manifest["subsets"].items()}
        frozen_ids = {k: v["context_ids"] for k, v in frozen.items()}
        if frozen_ids != current_ids:
            raise RuntimeError("existing frozen subset context IDs differ")
    else:
        gnp.write_json(mp, manifest)

    tpi, cf, full, pre, active = gnp.modules()
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _, all_traces, all_physical, all_meta = gnp.load_training_population(OLD, tpi)
    curves = pd.read_csv(gnp.OLD_CONT / "REAL_CONTINUOUS_SUCCESS_CURVES.csv")
    frontiers = pd.read_csv(gnp.OLD_CONT / "REAL_FINE_FRONTIER_SUMMARY.csv")
    contexts = pre.all_contexts_for_split("DEV", active)
    all_metrics = []
    all_decisions = []
    all_checkpoints = []
    for setting in SETTINGS:
        ids = set(subsets[setting])
        print(f"[SETTING] {setting} contexts={len(ids)}", flush=True)
        traces, physical, pairs, norm, models, cals, checkpoints = train_or_load(
            setting, ids, all_traces, all_physical, all_meta, tpi, cf, full, device, out
        )
        metrics, decisions = evaluate(
            setting, models, cals, norm, tpi, cf, contexts, curves, frontiers, device
        )
        all_metrics.extend(metrics)
        all_decisions.extend(decisions)
        all_checkpoints.extend(checkpoints)
        gnp.write_csv(out / "OLD_CONTEXT_COUNT_METRICS_PARTIAL.csv", all_metrics)
        gnp.write_csv(out / "OLD_CONTEXT_COUNT_PER_ROOT_PARTIAL.csv", all_decisions)
        gnp.write_json(out / "OLD_CONTEXT_COUNT_CHECKPOINTS.json", {
            "status": "IN_PROGRESS", "checkpoints": all_checkpoints,
        })
        if device.type == "cuda":
            torch.cuda.empty_cache()
    gnp.write_csv(out / "OLD_CONTEXT_COUNT_METRICS.csv", all_metrics)
    gnp.write_csv(out / "OLD_CONTEXT_COUNT_PER_ROOT.csv", all_decisions)
    gnp.write_json(out / "OLD_CONTEXT_COUNT_CHECKPOINTS.json", {
        "status": "COMPLETE_ALL_FIXED_SUBSETS", "checkpoints": all_checkpoints,
    })
    print(pd.DataFrame(all_metrics).query("aggregation == 'ENSEMBLE'").to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = ap.parse_args()
    main(args.out, args.device)
