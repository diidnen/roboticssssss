#!/usr/bin/env python3
"""Freeze all-TRAIN pooled Direct/WM/V0 verifier replicas before challenge query."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import pooled_joint_novisual_current as pooled
import run_pooled_predictive_verifier as ppv
import run_predictive_verifier_development as pv


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "activeforcing_final_experiment_20260901_045000"
MODELS = ROOT / "joint_mechanism_20260831/pooled_matched"
SEEDS = (0, 1, 2)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    protocol = json.loads((OUT / "ACTIVEFORCING_FINAL_EXPERIMENT_PROTOCOL.json").read_text())
    if protocol["status"] != "FROZEN_BEFORE_NEW_CHALLENGE_OUTCOMES":
        raise RuntimeError("experiment protocol not frozen")
    scratch = OUT / "_alltrain_verifier_audit"; scratch.mkdir(exist_ok=True)
    tpi, cf, full, cmap, traces, meta, audits, pairs, segs, computed_norm = pooled.load_population(scratch)
    y = np.asarray([tr.outcome for tr in traces], np.float32)
    if len(traces) != 720 or len({tr.context_id for tr in traces}) != 72 or len({(tr.task, tr.root_id) for tr in traces}) != 24:
        raise RuntimeError("unexpected pooled TRAIN population")
    ckdir = OUT / "frozen_models"; ckdir.mkdir(exist_ok=True)
    rows = []
    for seed in SEEDS:
        direct_path = MODELS / f"POOLED_BASE_seed{seed}.pt"
        world_path = MODELS / f"POOLED_JOINT_NOVISUAL_seed{seed}.pt"
        dck = torch.load(direct_path, map_location="cpu", weights_only=False)
        wck = torch.load(world_path, map_location="cpu", weights_only=False)
        norm = tuple(np.asarray(wck["normalization"][k], np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"])
        if not all(np.allclose(a, b, rtol=0, atol=1e-7) for a, b in zip(norm, computed_norm)):
            raise RuntimeError(f"seed{seed} normalization differs from pooled TRAIN")
        base_ck, _, _ = full.load_base(tpi, seed, torch.device("cpu"))
        joint = full.JointIEFeasibility(tpi, base_ck["state_dict"]).cpu()
        joint.load_state_dict(wck["state_dict"]); joint.eval()
        traj, extra = ppv.predict_world(joint.physics, traces, norm, cf, tpi, torch.device("cpu"))
        verifier, stats, history = pv.fit_verifier("V0_LINEAR", traj, extra, y, np.arange(len(y)), seed, torch.device("cpu"))
        target = ckdir / f"ALLTRAIN_VERIFIER_WM_CURRENT_V0_LINEAR_seed{seed}.pt"
        torch.save({
            "world_model": "POOLED_JOINT_NOVISUAL all TRAIN", "architecture": "V0_LINEAR", "seed": seed,
            "state_dict": verifier.state_dict(), "stats": {k: v.tolist() if hasattr(v, "tolist") else v for k, v in stats.items()},
            "decision": "SUCCESS iff logit > 0", "threshold_tuning": False, "calibration": False,
            "epochs": pv.EPOCHS, "TRAIN_branches": 720, "TRAIN_contexts": 72, "TRAIN_root_families": 24,
            "challenge_used": False, "untouched_TEST_used": False, "final_train_bce": history[-1]["train_bce"],
        }, target)
        rows.append({
            "seed": seed, "direct_path": str(direct_path), "direct_sha256": sha(direct_path),
            "world_path": str(world_path), "world_sha256": sha(world_path),
            "verifier_path": str(target), "verifier_sha256": sha(target), "final_train_bce": history[-1]["train_bce"],
        })
    freeze = {
        "status": "FROZEN_BEFORE_FORCE_CRITICAL_MEMBERSHIP_AND_MODEL_QUERY",
        "scope": "all authoritative pooled task0/task1/task5/task6 TRAIN only",
        "models": rows,
        "direct_proposal": "mean sigmoid probability across three Direct seeds, then frozen expected-utility rule with Cfail=task Fmax",
        "verifier_architecture": "V0_LINEAR selected previously by pooled grouped-root CV",
        "verifier_replication": "three seed-matched systems reported individually and by arithmetic metric mean; no best-seed selection",
        "verifier_decision": "native logit > 0",
        "search": "upward only from Direct proposal; exhausted => NO_VALID_FORCE",
        "max_fallback": "separate deployment variant; not a verifier safety judgment",
        "architecture_tuning": False, "threshold_tuning": False, "challenge_used": False,
        "untouched_TEST_read": False,
    }
    path = OUT / "ACTIVEFORCING_FINAL_METHOD_FREEZE_PRECHALLENGE.json"
    path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": freeze["status"], "freeze": str(path), "sha256": sha(path)}, indent=2))


if __name__ == "__main__":
    main()
