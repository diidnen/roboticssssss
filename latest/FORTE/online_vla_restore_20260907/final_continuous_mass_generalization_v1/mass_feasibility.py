"""Frozen phase-free full-task MASS feasibility and utility runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
INTERFACE = "CURRENT_MASS58_SIGMA_AWARE_POSITIVE_MIXTURE_V1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.command_gru = nn.GRU(10, 64, batch_first=True)
        self.condition = nn.Sequential(nn.Linear(54, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, step, condition):
        _, hidden = self.command_gru(step)
        return self.head(torch.cat([hidden[-1], self.condition(condition)], -1)).squeeze(-1)


class MassFeasibility:
    def __init__(self, manifest_path=None, device="cpu"):
        self.manifest_path = Path(manifest_path or HERE / "MASS_FEASIBILITY_MANIFEST.json")
        self.manifest = json.loads(self.manifest_path.read_text())
        m = self.manifest
        if not m.get("qualified") or m.get("posterior_interface") != INTERFACE:
            raise ValueError("unqualified/wrong MASS feasibility runtime")
        if m.get("label_target") != "full_task_success_y" or m.get("phase_representation") != "NONE":
            raise ValueError("wrong MASS feasibility target/input")
        for source, digest in m["source_hashes"].items():
            if sha(source) != digest: raise ValueError("MASS feasibility source changed: " + source)
        self.device = torch.device(device)
        self.models, means, stds = [], [], []
        for entry in sorted(m["checkpoints"], key=lambda row: row["seed"]):
            if sha(entry["path"]) != entry["sha256"]: raise ValueError("MASS feasibility checkpoint changed")
            checkpoint = torch.load(entry["path"], map_location="cpu", weights_only=False)
            model = Network(); model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.to(self.device).eval()
            for parameter in model.parameters(): parameter.requires_grad_(False)
            self.models.append(model)
            means.append(np.asarray(checkpoint["normalization_mean"], np.float32))
            stds.append(np.asarray(checkpoint["normalization_std"], np.float32))
        if len(self.models) != 3 or not all(np.array_equal(means[0], x) for x in means[1:]) or not all(np.array_equal(stds[0], x) for x in stds[1:]):
            raise ValueError("MASS feasibility ensemble mismatch")
        self.mean, self.std = means[0], stds[0]
        self.force_grid = np.round(np.arange(3.0, 5.0001, 0.05), 8)

    def curve(self, preaction_sequence, posterior):
        base = np.asarray(preaction_sequence, np.float32)
        if base.shape != (8, 64) or not np.isfinite(base).all():
            raise ValueError("expected finite phase-free (8,64) pre-action sequence")
        if posterior.get("interface") != INTERFACE or posterior.get("candidate_actions_executed") != 0:
            raise ValueError("wrong/non-preaction MASS posterior")
        if posterior.get("hidden_mass_used") is not False:
            raise ValueError("AF MASS posterior contains hidden mass")
        nodes = np.asarray(posterior["integration_nodes"], np.float32)
        weights = np.asarray(posterior["integration_weights"], float)
        if nodes.ndim != 1 or weights.shape != nodes.shape or not len(nodes): raise ValueError("invalid MASS quadrature")
        if np.any(nodes <= 0) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-9):
            raise ValueError("MASS quadrature must be positive and normalized")
        forces = np.repeat(self.force_grid, len(nodes)); masses = np.tile(nodes, len(self.force_grid))
        x = np.broadcast_to(base, (len(forces),) + base.shape).copy()
        x[:, :, 10] = forces[:, None] / 8.0
        x[:, :, 11] = masses[:, None]
        x = (x - self.mean[None, None]) / self.std[None, None]
        value = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probabilities = np.mean([torch.sigmoid(model(value[:, :, :10], value[:, 0, 10:])).cpu().numpy()
                                     for model in self.models], axis=0)
        return probabilities.reshape(len(self.force_grid), len(nodes)) @ weights

    def select(self, preaction_sequence, posterior):
        probability = self.curve(preaction_sequence, posterior)
        utility = probability * (5.0 - self.force_grid) / 5.0 + (1.0 - probability) * -1.0
        selected = int(np.argmax(utility))
        return {"selected_force_N": float(self.force_grid[selected]),
                "predicted_success": float(probability[selected]), "utility": float(utility[selected]),
                "force_grid_N": self.force_grid.tolist(), "p_success": probability.astype(float).tolist(),
                "expected_utility": utility.astype(float).tolist(), "posterior_interface": INTERFACE,
                "posterior_sigma_used": True, "feasibility_input_shape": [8, 64],
                "phase_representation": "NONE", "label_target": "full_task_success_y",
                "feasibility_manifest": str(self.manifest_path),
                "feasibility_manifest_sha256": sha(self.manifest_path)}
