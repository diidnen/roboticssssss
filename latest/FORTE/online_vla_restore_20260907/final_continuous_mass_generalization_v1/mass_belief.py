"""Qualified continuous positive-support MASS belief runtime."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = Path("/home/exouser/FORTE")
POSITIVE = ROOT / "analysis/results/current58_positive_posterior_contract_repair_20260905/positive_posterior.py"
sys.path.insert(0, str(ROOT))
from current_contract_belief_features import ProbeEvidence, SCHEMA_ID
from current_contract_physical_belief import FrictionMember, predict, verify_decision_prefix

spec = importlib.util.spec_from_file_location("mass_positive_posterior", POSITIVE)
positive = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = positive
spec.loader.exec_module(positive)

INTERFACE = "CURRENT_MASS58_SIGMA_AWARE_POSITIVE_MIXTURE_V1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integration(posterior):
    means, sigmas, _ = posterior.parameters()
    points = [float(m + k * s) for m, s in zip(means, sigmas) for k in (-8, -4, -2, -1, 0, 1, 2, 4, 8)]
    x, w, qa = posterior.quadrature(order=32, split_points=points)
    x2, w2, qa2 = posterior.quadrature(order=64, split_points=points)
    moments = posterior.moments()
    errors = [
        abs(float(w @ x) - moments["mean"]),
        abs(float(w @ (x * x)) - (moments["variance"] + moments["mean"] ** 2)),
        abs(float(w @ x) - float(w2 @ x2)),
        abs(float(w @ (x * x)) - float(w2 @ (x2 * x2))),
    ]
    if max(errors) > 1e-8 * max(1.0, moments["variance"] + moments["mean"] ** 2):
        raise ValueError("continuous MASS posterior quadrature did not converge")
    return x, w, {**qa, "cross_order_moment_errors": errors,
                  "validation_order": qa2["order_per_interval"], "sigma_used": True,
                  "three_member_means_only": False}


class ContinuousMassBelief:
    def __init__(self, manifest_path, *, manifest_sha256, diagnostic_only=False):
        path = Path(manifest_path)
        if sha(path) != manifest_sha256:
            raise ValueError("MASS belief manifest changed")
        self.manifest = json.loads(path.read_text())
        m = self.manifest
        if m.get("interface") != INTERFACE or m.get("feature_dim") != 58 or m.get("feature_schema_id") != SCHEMA_ID:
            raise ValueError("wrong MASS belief schema/interface")
        self.diagnostic_only = diagnostic_only
        if not diagnostic_only and not m.get("qualified_for_current_runtime"):
            raise ValueError("MASS belief has not passed qualification")
        for source, digest in m["source_hashes"].items():
            if sha(source) != digest:
                raise ValueError("MASS belief source changed: " + source)
        self.evidence = ProbeEvidence()
        normalization = m["normalization"]
        self.mean = np.asarray(normalization["mean"], np.float32)
        self.std = np.asarray(normalization["std"], np.float32)
        if self.mean.shape != (58,) or self.std.shape != (58,) or np.any(self.std <= 0):
            raise ValueError("invalid MASS normalization")
        if normalization.get("fit_split") != "TRAIN":
            raise ValueError("normalization must use TRAIN only")
        self.models = []
        entries = sorted(m["checkpoints"], key=lambda row: row["seed"])
        if len(entries) != 3 or {row["seed"] for row in entries} != {0, 1, 2}:
            raise ValueError("three frozen MASS ensemble members required")
        for entry in entries:
            if sha(entry["path"]) != entry["sha256"]:
                raise ValueError("MASS checkpoint changed")
            checkpoint = torch.load(entry["path"], map_location="cpu", weights_only=True)
            if checkpoint.get("input_dim") != 58 or checkpoint.get("feature_schema_id") != SCHEMA_ID:
                raise ValueError("MASS checkpoint schema mismatch")
            model = FrictionMember(58, 16, 16)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self.models.append(model)

    def array(self, raw_features):
        x = np.asarray(raw_features, np.float32)
        if x.ndim != 2 or x.shape[1] != 58 or not np.isfinite(x).all():
            raise ValueError("invalid MASS probe feature array")
        normalized = ((x - self.mean) / self.std).astype(np.float32)
        means, log_sigmas = [], []
        for model in self.models:
            mu, sigma = predict(model, [normalized])
            means.append(float(mu[0])); log_sigmas.append(float(np.log(sigma[0])))
        posterior = positive.PositivePosterior(tuple(means), tuple(log_sigmas), (1 / 3,) * 3)
        nodes, weights, qa = integration(posterior)
        return {
            "interface": INTERFACE,
            "feature_schema_id": SCHEMA_ID,
            "raw_features": x,
            "normalized_features": normalized,
            "member_means": means,
            "member_log_sigmas": log_sigmas,
            "continuous_posterior": posterior,
            "posterior_moments": posterior.moments(),
            "integration_nodes": nodes,
            "integration_weights": weights,
            "integration_qa": qa,
            "decision_state_is_pre_candidate": True,
            "hidden_mass_used": False,
            "diagnostic_only": self.diagnostic_only,
        }

    def rows(self, raw, readback, *, query_runtime_manifest_sha256):
        if query_runtime_manifest_sha256 != self.manifest["compatible_query_runtime_manifest_sha256"]:
            raise ValueError("wrong MASS query runtime")
        verify_decision_prefix(raw)
        tasks = {int(row["task_id"]) for row in raw}
        if len(tasks) != 1 or not tasks <= {0, 1, 5, 6}:
            raise ValueError("mixed/unsupported MASS task")
        if not self.diagnostic_only and next(iter(tasks)) not in self.manifest["qualified_tasks"]:
            raise ValueError("task-specific MASS qualification missing")
        return self.array(self.evidence.rows(raw, readback))
