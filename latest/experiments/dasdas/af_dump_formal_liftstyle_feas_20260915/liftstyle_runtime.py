"""Lift-style dump feasibility in the original dump AF select() slot.

Training used hold-chunk zeros and point μ. Online uses the π0 pre-action
chunk (lift transfer) and frozen v4 belief quadrature. Executed selector is
argmax p on the training grid [1,5] N @ 0.5 N. Dump EU is logged only.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
OLD = BASE / "experiments/af_dump_original_restore_20260912"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(HERE))
from native_original_motion_features import TASK_BINDING
from max_force_utility import expected_utility, select_force


FORCE_NORM = 8.0
POSTERIOR_INTERFACE = "CURRENT_MULTITASK58_SIGMA_AWARE_POSITIVE_MIXTURE_V1"


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.command_gru = nn.GRU(10, 64, batch_first=True)
        self.condition = nn.Sequential(nn.Linear(54, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, step, cond):
        _, h = self.command_gru(step)
        return self.head(torch.cat([h[-1], self.condition(cond)], -1)).squeeze(-1)


class LiftstyleFeasibility:
    def __init__(self, path, *, manifest_sha256, device="cpu"):
        self.manifest_path = Path(path)
        content = self.manifest_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest_sha256:
            raise ValueError("Lift-style feasibility manifest changed")
        self.manifest = json.loads(content)
        manifest = self.manifest
        if manifest.get("native_task_binding") != TASK_BINDING or manifest.get("root_scope") != [200002]:
            raise ValueError("Native root-local task binding required")
        if manifest.get("feature_shape") != [8, 64] or manifest.get("label_target") != "full_task_success_y":
            raise ValueError("Phase-free full-task feature/label contract required")
        if manifest.get("posterior_interface") != POSTERIOR_INTERFACE:
            raise ValueError("Posterior interface mismatch")
        if manifest.get("force_support") != [1.0, 5.0]:
            raise ValueError("Lift-style support must be [1,5] N")
        if manifest.get("force_feature_normalization_N") != FORCE_NORM:
            raise ValueError("Training used F/8")
        if manifest.get("planner_grid_step") != 0.5:
            raise ValueError("Training grid step is 0.5 N")
        if manifest.get("executed_selector") != "argmax_p":
            raise ValueError("Executed selector must be argmax_p")
        sources = manifest.get("source_hashes", {})
        required = [
            Path(__file__).resolve(),
            Path(__file__).with_name("max_force_utility.py").resolve(),
        ]
        for source in required:
            if sources.get(str(source)) != hashlib.sha256(source.read_bytes()).hexdigest():
                raise ValueError("Missing/stale runtime source hash: " + str(source))
        entries = manifest.get("checkpoints", [])
        if len(entries) != 3 or {entry["seed"] for entry in entries} != {0, 1, 2}:
            raise ValueError("Three-member feasibility ensemble required")
        self.device = torch.device(device)
        self.models = []
        self.mean = self.std = None
        for entry in sorted(entries, key=lambda row: row["seed"]):
            checkpoint_path = Path(entry["path"])
            if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("Checkpoint changed")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            mean = np.asarray(checkpoint["mean"], np.float32)
            std = np.asarray(checkpoint["std"], np.float32)
            if mean.shape != (64,) or std.shape != (64,) or not np.isfinite(np.r_[mean, std]).all() or np.any(std <= 0):
                raise ValueError("Invalid checkpoint normalization")
            if self.mean is None:
                self.mean, self.std = mean, std
            elif not np.array_equal(self.mean, mean) or not np.array_equal(self.std, std):
                raise ValueError("Ensemble normalization mismatch")
            model = Network()
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.to(self.device).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.models.append(model)
        support = manifest["force_support"]
        step = float(manifest["planner_grid_step"])
        self.force_grid = np.round(np.arange(support[0], support[1] + 1e-9, step), 10)

    def curve(self, sequence, posterior):
        base = np.asarray(sequence, np.float32)
        if base.shape != (8, 64) or not np.isfinite(base).all():
            raise ValueError("Expected finite phase-free pre-action (8,64) features")
        if posterior.get("interface") != POSTERIOR_INTERFACE:
            raise ValueError("Posterior interface mismatch")
        if posterior.get("candidate_actions_executed") != 0 or posterior.get("hidden_friction_used") is not False:
            raise ValueError("Posterior is not pre-action observable")
        nodes = np.asarray(posterior["integration_nodes"], np.float32)
        weights = np.asarray(posterior["integration_weights"], float)
        if nodes.ndim != 1 or weights.shape != nodes.shape or len(nodes) == 0:
            raise ValueError("Invalid quadrature shape")
        if not np.isfinite(np.r_[nodes, weights]).all() or np.any(nodes <= 0) or np.any(weights < 0):
            raise ValueError("Invalid quadrature values")
        if not np.isclose(weights.sum(), 1, atol=1e-9):
            raise ValueError("Quadrature weights are not normalized")
        forces = np.repeat(self.force_grid, len(nodes))
        mus = np.tile(nodes, len(self.force_grid))
        values = np.broadcast_to(base, (len(forces),) + base.shape).copy()
        values[:, :, 10] = forces[:, None] / FORCE_NORM
        values[:, :, 11] = mus[:, None]
        values = (values - self.mean[None, None]) / self.std[None, None]
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        predictions = []
        with torch.no_grad():
            for model in self.models:
                predictions.append(
                    torch.sigmoid(model(tensor[:, :, :10], tensor[:, 0, 10:])).cpu().numpy()
                )
        p = np.mean(predictions, axis=0).reshape(len(self.force_grid), len(nodes))
        return p @ weights

    def select(self, feature, posterior):
        if not isinstance(feature, dict) or feature.get("task_binding") != TASK_BINDING:
            raise ValueError("Explicit native feature/task binding required")
        if feature.get("source") != "ONLINE_VLA_ACTION_CHUNK" or feature.get("candidate_actions_executed") != 0:
            raise ValueError("Pre-action feature required")
        sequence = np.asarray(feature["sequence"], np.float32)
        curve = self.curve(sequence, posterior)
        index = int(np.argmax(curve))
        eu = select_force(self.force_grid, curve, self.manifest["force_support"])
        return {
            "selected_force_N": float(self.force_grid[index]),
            "predicted_success": float(curve[index]),
            "utility": float(expected_utility(curve[index], self.force_grid[index], 5.0)),
            "force_grid_N": self.force_grid.tolist(),
            "p_success": curve.astype(float).tolist(),
            "expected_utility": expected_utility(curve, self.force_grid, 5.0).astype(float).tolist(),
            "executed_selector": "argmax_p",
            "eu_counterfactual_selected_N": eu["selected_force_N"],
            "eu_counterfactual_predicted_success": eu["predicted_success"],
            "native_task_binding": deepcopy(TASK_BINDING),
            "root_scope": [200002],
            "phase_representation": "NONE",
            "posterior_interface": POSTERIOR_INTERFACE,
            "posterior_sigma_used": True,
            "force_support": [1.0, 5.0],
            "planner_grid_step": 0.5,
            "force_feature_normalization_N": FORCE_NORM,
            "utility_normalization_N": 5.0,
            "utility_definition": "logged only; executed rule is argmax_p",
            "motion_channels": "pi0_chunk_transfer_not_hold_zeroed",
            "feasibility_manifest": str(self.manifest_path),
            "feasibility_manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
        }
