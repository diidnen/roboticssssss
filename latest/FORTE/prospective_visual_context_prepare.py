#!/usr/bin/env python3
"""Freeze the prospective visual-aligned TRAIN/DEV population and grids."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/exouser/Tabero")
FORTE = Path("/home/exouser/FORTE")
RUNNER = ROOT / "analysis/p5s0c_paired_boundary_probe_value.py"
OUT = ROOT / "analysis/results/gnp_style_visual_context_prospective_20260831_011000"
SEED = 2026083102
DEV_CONTEXT_KEYS = {
    (0, 6, "LOW"), (0, 7, "LOW"),
    (1, 6, "LOW"), (1, 6, "MID"), (1, 7, "LOW"), (1, 7, "MID"),
    (5, 6, "LOW"), (5, 7, "LOW"),
    (6, 7, "LOW"),
}
TASK_RANGES = {0: (3.0, 5.0), 1: (4.0, 6.0), 5: (3.0, 5.0), 6: (3.0, 4.0)}
DEV_GRIDS = {
    0: [3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
    1: [4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00],
    5: [3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00],
    6: [3.00, 3.25, 3.50, 3.75, 4.00],
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_file():
        h.update(path.read_bytes())
    else:
        for f in sorted(p for p in path.rglob("*") if p.is_file()):
            h.update(str(f.relative_to(path)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_runner():
    spec = importlib.util.spec_from_file_location("prospective_p5", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cid(split: str, task: int, root_index: int, seed: int, band: str, mu: float) -> str:
    return f"pv_{split.lower()}_t{task}_r{root_index:02d}_s{seed}_{band.lower()}_mu{mu:.6f}"


def main() -> None:
    p5 = load_runner()
    OUT.mkdir(parents=True, exist_ok=True)
    contexts = []
    for root in p5.ROOT_PLAN:
        task, idx = int(root["task"]), int(root["root_index"])
        split = "TRAIN" if idx < 6 else ("DEV" if idx < 8 else "TEST")
        if split == "TEST":
            continue
        bands = ["LOW", "MID", "HIGH"]
        for band in bands:
            if split == "DEV" and (task, idx, band) not in DEV_CONTEXT_KEYS:
                continue
            mu = float(root["frictions"][band])
            contexts.append({
                "context_id": cid(split, task, idx, int(root["root_seed"]), band, mu),
                "split": split,
                "task": task,
                "root_id": f"pv_{split.lower()}_t{task}_root{idx:02d}_s{int(root['root_seed'])}",
                "source_root_id": root["root_id"],
                "root_index": idx,
                "root_seed": int(root["root_seed"]),
                "friction_band": band,
                "mu_GT": mu,
                "outcome_selection": "fixed authoritative split semantics before outcomes",
                "state_selection": "fresh prospective reset using authoritative root seed; no historical state-hash matching required",
                "eligible": 1,
            })
    contexts.sort(key=lambda r: (r["split"], r["task"], r["root_index"], r["friction_band"]))
    write_csv(OUT / "PROSPECTIVE_CONTEXT_MANIFEST.csv", contexts)

    rng = np.random.default_rng(SEED)
    train_force_rows = []
    train_cells = []
    for c in contexts:
        if c["split"] != "TRAIN":
            continue
        lo, hi = TASK_RANGES[int(c["task"])]
        for s in range(5):
            slo = lo + (hi - lo) * s / 5
            shi = lo + (hi - lo) * (s + 1) / 5
            force = float(rng.uniform(slo, shi))
            cell = {**c, "stratum_index": s, "stratum_low_N": slo, "stratum_high_N": shi, "force_N": force}
            train_cells.append(cell)
            train_force_rows.append({
                "context_id": c["context_id"], "split": "TRAIN", "task": c["task"],
                "root_id": c["root_id"], "friction_band": c["friction_band"], "mu_GT": c["mu_GT"],
                "stratum_index": s, "stratum_low_N": slo, "stratum_high_N": shi,
                "force_N": force, "force_sampling": "uniform_continuous_one_draw_per_equal_width_stratum",
                "global_rng_seed": SEED, "frozen_before_outcomes": 1,
            })
    write_csv(OUT / "PROSPECTIVE_TRAIN_FORCE_MANIFEST.csv", train_force_rows)

    train_specs = {}
    train_run_rows = []
    for cell in train_cells:
        specs = []
        for rep in range(1, 3):
            label = f"TRAIN_S{cell['stratum_index']}_F{cell['force_N']:.8f}_R{rep}"
            spec = {"force_N": cell["force_N"], "repeat_index": rep, "branch_label": label}
            specs.append(spec)
            train_run_rows.append({**cell, "repeat": rep, "branch_label": label, "scientific_retry": 0})
        train_specs[cell["context_id"]] = train_specs.get(cell["context_id"], []) + specs
    write_csv(OUT / "PROSPECTIVE_TRAIN_RUN_MANIFEST.csv", train_run_rows)
    write_json(OUT / "PROSPECTIVE_TRAIN_TARGET_MANIFEST.json", {
        "manifest_name": "PROSPECTIVE_TRAIN_TARGET_MANIFEST",
        "split": "TRAIN", "contexts": train_specs, "expected_contexts": len(train_specs),
        "expected_force_cells": len(train_cells), "expected_branches": len(train_run_rows),
        "branch_state": "strict pre-probe last-hold snapshot restored before every branch",
    })

    dev_specs, dev_run_rows = {}, []
    for c in contexts:
        if c["split"] != "DEV":
            continue
        for force in DEV_GRIDS[int(c["task"])] :
            specs = dev_specs.setdefault(c["context_id"], [])
            for rep in range(1, 6):
                label = f"DEV_F{force:.2f}_R{rep}"
                specs.append({"force_N": force, "repeat_index": rep, "branch_label": label})
                dev_run_rows.append({**c, "force_N": force, "repeat": rep, "branch_label": label, "scientific_retry": 0})
    write_json(OUT / "PROSPECTIVE_DEV_FORCE_PROTOCOL.json", {
        "protocol_name": "PROSPECTIVE_REPEATED_CONTINUOUS_DEV",
        "split": "DEV", "force_grids_N": DEV_GRIDS, "repeats_per_context_force": 5,
        "contexts_frozen_before_training": 1, "outcome_blind_collection": 1,
        "frontier_definition": "minimum tested force with k/5 >= 4/5",
        "wilson_interval": "95% Wilson score interval",
    })
    write_csv(OUT / "PROSPECTIVE_DEV_RUN_MANIFEST.csv", dev_run_rows)
    write_json(OUT / "PROSPECTIVE_DEV_TARGET_MANIFEST.json", {
        "manifest_name": "PROSPECTIVE_DEV_TARGET_MANIFEST",
        "split": "DEV", "contexts": dev_specs, "expected_contexts": len(dev_specs),
        "expected_force_cells": len(dev_run_rows) // 5, "expected_branches": len(dev_run_rows),
        "branch_state": "strict pre-probe last-hold snapshot restored before every branch",
    })

    write_json(OUT / "PROSPECTIVE_VISUAL_FEATURE_SPEC.json", {
        "source_module": "/media/volume/newdata/exouser/tabero/Tabero-VTLA/src/openpi/models/pi0.py",
        "exact_tensor_location": "Pi0.sample_actions -> Pi0.embed_prefix -> self.PaliGemma.img(image), before language/action suffix decoding",
        "source_code_hashes": {
            "pi0.py": sha(Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA/src/openpi/models/pi0.py")),
            "model.py": sha(Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA/src/openpi/models/model.py")),
            "policy.py": sha(Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA/src/openpi/policies/policy.py")),
            "diagnostic_server.py": sha(FORTE / "visual_pi0_server.py"),
        },
        "checkpoint_path": "/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999",
        "checkpoint_hash": sha(Path("/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999")),
        "input_cameras": ["agentview_cam -> base_0_rgb", "eye_in_hand_cam -> left_wrist_0_rgb"],
        "raw_camera_shape": [512, 512, 3], "policy_preprocessed_shape": [224, 224, 3], "raw_dtype": "uint8",
        "pooling": "mean over valid image-token sequence independently per camera; concatenate camera 0 then camera 1",
        "token_width": 2048, "unreduced_shape_per_camera": [256, 2048], "concatenated_shape": [4096],
        "pca": {"used": True, "components": 64, "fit": "TRAIN-only contexts/features; no DEV fitting", "implementation": "to be fit after prospective TRAIN visual capture"},
        "right_wrist_padding": "excluded because image_mask is false for PI0",
        "feedback_to_policy": False, "action_generation_changed": False,
    })
    write_json(OUT / "PROSPECTIVE_PROTOCOL_FREEZE.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "FROZEN_BEFORE_OUTCOMES",
        "contexts": len(contexts), "train_contexts": sum(c["split"] == "TRAIN" for c in contexts),
        "dev_contexts": sum(c["split"] == "DEV" for c in contexts), "train_force_cells": len(train_cells),
        "train_branches": len(train_run_rows), "dev_force_cells": len(dev_run_rows) // 5,
        "dev_branches": len(dev_run_rows), "test_roots_included": False,
        "source_runner": str(RUNNER), "source_runner_sha256": sha(RUNNER),
        "global_train_force_rng_seed": SEED,
    })
    print(json.dumps({"out": str(OUT), "train_contexts":  len(train_specs), "train_branches": len(train_run_rows), "dev_contexts": len(dev_specs), "dev_branches": len(dev_run_rows)}, indent=2))


if __name__ == "__main__":
    main()
