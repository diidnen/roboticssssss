"""Root-local phase-free feasibility runtime with force-15 support."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
OLD = BASE / "experiments/af_dump_original_restore_20260912"
SNAPSHOT = Path(
    "/media/volume/newdata/exouser/online_vla_activeforcing_20260907/"
    "final_continuous_friction_generalization_v1/SOURCE_SNAPSHOT"
)
sys.path.insert(0, str(OLD))
import native_original_feasibility_binding as original
from force15_utility import load_calibration, select_force


class Force15Feasibility(original.NativeOriginalFeasibility):
    def __init__(self, path, *, manifest_sha256, device="cpu"):
        self.manifest_path = Path(path)
        content = self.manifest_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest_sha256:
            raise ValueError("Force-15 feasibility manifest changed")
        self.manifest = json.loads(content)
        manifest = self.manifest
        if manifest.get("native_task_binding") != original.TASK_BINDING or manifest.get("root_scope") != [200002]:
            raise ValueError("Native root-local task binding required")
        if manifest.get("feature_shape") != [8, 64] or manifest.get("label_target") != "full_task_success_y":
            raise ValueError("Phase-free full-task feature/label contract required")
        if manifest.get("calibration") != "NONE_RAW" or manifest.get("posterior_interface") != "CURRENT_MULTITASK58_SIGMA_AWARE_POSITIVE_MIXTURE_V1":
            raise ValueError("Original probability/posterior contract changed")
        if manifest.get("sigma_used") is not True or manifest.get("force_support") != [0.5, 15.0]:
            raise ValueError("Force-15 posterior/support contract required")
        if manifest.get("force_feature_normalization_N") != 15.0 or manifest.get("planner_grid_step") != 0.05:
            raise ValueError("Force-15 normalization/grid contract required")
        if manifest.get("utility_engineering_scale_N") != 15.0:
            raise ValueError("Force-15 utility scale required")
        calibration_path = Path(manifest["realized_force_calibration_path"])
        self.realized_calibration = load_calibration(
            calibration_path, manifest["realized_force_calibration_sha256"]
        )
        sources = manifest.get("source_hashes", {})
        required_sources = [
            Path(__file__).resolve(),
            Path(__file__).with_name("force15_utility.py").resolve(),
            OLD / "native_original_feasibility_binding.py",
            SNAPSHOT / "phase_free_feasibility.py",
        ]
        for source in required_sources:
            if sources.get(str(source)) != hashlib.sha256(source.read_bytes()).hexdigest():
                raise ValueError("Missing/stale runtime source hash: " + str(source))
        entries = manifest.get("checkpoints", [])
        if len(entries) != 3 or {entry["seed"] for entry in entries} != {0, 1, 2}:
            raise ValueError("Three-member feasibility ensemble required")
        self.device = torch.device(device)
        self.models = []
        self.mean = self.std = None
        network = original.phase_free_network_class()
        for entry in sorted(entries, key=lambda row: row["seed"]):
            checkpoint_path = Path(entry["path"])
            if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("Checkpoint changed")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint.get("native_task_binding") != original.TASK_BINDING:
                raise ValueError("Checkpoint task binding mismatch")
            if checkpoint.get("feature_shape") != [8, 64] or checkpoint.get("target") != "full_task_success_y":
                raise ValueError("Checkpoint feature/target mismatch")
            if checkpoint.get("protocol_sha256") != manifest["training_protocol_sha256"]:
                raise ValueError("Checkpoint protocol mismatch")
            if checkpoint.get("normalization_fit_split") != "TRAIN":
                raise ValueError("Only TRAIN normalization is permitted")
            mean = np.asarray(checkpoint["normalization_mean"], np.float32)
            std = np.asarray(checkpoint["normalization_std"], np.float32)
            if mean.shape != (64,) or std.shape != (64,) or not np.isfinite(np.r_[mean, std]).all() or np.any(std <= 0):
                raise ValueError("Invalid checkpoint normalization")
            if self.mean is None:
                self.mean, self.std = mean, std
            elif not np.array_equal(self.mean, mean) or not np.array_equal(self.std, std):
                raise ValueError("Ensemble normalization mismatch")
            model = network()
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.to(self.device).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.models.append(model)
        support = manifest["force_support"]
        self.force_grid = np.round(np.arange(support[0], support[1] + 0.0001, 0.05), 8)

    def curve(self, sequence, posterior):
        base = np.asarray(sequence, np.float32)
        if base.shape != (8, 64) or not np.isfinite(base).all():
            raise ValueError("Expected finite phase-free pre-action (8,64) features")
        if posterior.get("interface") != self.manifest["posterior_interface"]:
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
        values[:, :, 10] = forces[:, None] / 15.0
        values[:, :, 11] = mus[:, None]
        values = (values - self.mean[None, None]) / self.std[None, None]
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        predictions = []
        with torch.no_grad():
            for model in self.models:
                chunks = []
                for start in range(0, len(tensor), 4096):
                    chunk = tensor[start : start + 4096]
                    chunks.append(torch.sigmoid(model(chunk[:, :, :10], chunk[:, 0, 10:])).cpu().numpy())
                predictions.append(np.concatenate(chunks))
        marginal = np.mean(predictions, axis=0).reshape(len(self.force_grid), len(nodes)) @ weights
        return marginal

    def select(self, feature, posterior):
        if not isinstance(feature, dict) or feature.get("task_binding") != original.TASK_BINDING:
            raise ValueError("Explicit native feature/task binding required")
        if feature.get("source") != "ONLINE_VLA_ACTION_CHUNK" or feature.get("candidate_actions_executed") != 0:
            raise ValueError("Pre-action feature required")
        curve = self.curve(feature["sequence"], posterior)
        result = select_force(
            self.force_grid,
            curve,
            self.manifest["force_support"],
            self.realized_calibration,
        )
        result.update(
            native_task_binding=deepcopy(original.TASK_BINDING),
            root_scope=[200002],
            phase_representation="NONE",
            posterior_interface=self.manifest["posterior_interface"],
            posterior_sigma_used=True,
            feasibility_manifest=str(self.manifest_path),
            feasibility_manifest_sha256=hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
            realized_force_calibration_sha256=self.manifest["realized_force_calibration_sha256"],
        )
        return result
