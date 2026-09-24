#!/usr/bin/env python3
"""Self-contained historical P5-S0-C/P5-S0-D 46-D evidence adapter.

The feature order and row construction are copied from the transferred
`OnlineInference.sequence_array` implementation. This adapter has no Isaac,
model-checkpoint, or Q2F dependency and is intended for offline parity only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


NUMERIC_PROBE_COLS = [
    "t_s", "force_target", "measured_squeeze", "target_normal_force",
    "measured_fn", "measured_ft", "ft_over_fn", "left_fx", "left_fy",
    "left_fz", "right_fx", "right_fy", "right_fz", "force_imbalance",
    "force_imbalance_ratio", "gripper_opening", "contact_normal_x",
    "contact_normal_y", "contact_normal_z", "contact_tangent_x",
    "contact_tangent_y", "contact_tangent_z", "commanded_tangent_increment_mm",
    "accumulated_displacement_mm", "marker_motion", "marker_tangential",
    "marker_velocity", "marker_loading_unloading", "contact_left",
    "contact_right", "tactile_ok",
]


class Evidence46D:
    """Build the exact historical dynamic 46-D feature sequence."""

    def __init__(self, normalization_json: str | Path):
        self.norm = json.loads(Path(normalization_json).read_text(encoding="utf-8"))
        self.feature_names = list(self.norm["dynamic_feature_names"])
        self.phases = list(self.norm["phase_categories_from_train"])
        self.states = list(self.norm["contact_state_categories_from_train"])
        if len(self.feature_names) != 46:
            raise ValueError(f"expected 46 features, got {len(self.feature_names)}")

    @staticmethod
    def _f(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
        try:
            val = row.get(key, "")
            return default if val == "" else float(val)
        except (TypeError, ValueError):
            return default

    def rows(self, raw_rows: Iterable[Mapping[str, str]]) -> np.ndarray:
        raw_rows = list(raw_rows)
        if not raw_rows:
            return np.zeros((1, len(self.feature_names)), dtype=np.float32)
        ex0 = self._f(raw_rows[0], "eef_x", 0.0)
        ey0 = self._f(raw_rows[0], "eef_y", 0.0)
        ez0 = self._f(raw_rows[0], "eef_z", 0.0)
        rows: list[list[float]] = []
        for rr in raw_rows:
            feat: dict[str, float] = {}
            for col in NUMERIC_PROBE_COLS:
                feat[col] = self._f(rr, col, 0.0)
            feat["eef_dx"] = self._f(rr, "eef_x", 0.0) - ex0
            feat["eef_dy"] = self._f(rr, "eef_y", 0.0) - ey0
            feat["eef_dz"] = self._f(rr, "eef_z", 0.0) - ez0
            phase = str(rr.get("probe_phase", "NA"))
            state = str(rr.get("contact_state", "NA"))
            for p in self.phases:
                feat[f"phase={p}"] = 1.0 if phase == p else 0.0
            for s in self.states:
                feat[f"contact_state={s}"] = 1.0 if state == s else 0.0
            feat["probe_phase_unknown"] = 0.0 if phase in self.phases else 1.0
            feat["contact_state_unknown"] = 0.0 if state in self.states else 1.0
            rows.append([float(feat.get(name, 0.0)) for name in self.feature_names])
        return np.asarray(rows, dtype=np.float32)

    def csv(self, path: str | Path) -> np.ndarray:
        with Path(path).open(newline="", encoding="utf-8") as fh:
            return self.rows(csv.DictReader(fh))

    def normalize_dynamic(self, array: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.norm["dynamic_mean"], dtype=np.float32)
        std = np.asarray(self.norm["dynamic_std"], dtype=np.float32)
        if array.shape[-1] != 46 or mean.shape != (46,) or std.shape != (46,):
            raise ValueError("46-D normalization shape mismatch")
        return ((array - mean) / std).astype(np.float32)
