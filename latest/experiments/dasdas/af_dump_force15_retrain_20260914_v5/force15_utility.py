"""Controller-aware ActiveForcing utility for the force-15 retraining intervention."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


UTILITY_DEFINITION = "p*(15-realized_squeeze(command))/15-(1-p)"


def load_calibration(path: Path, expected_sha256: str | None = None) -> dict:
    import hashlib

    path = Path(path)
    content = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Realized-force calibration changed")
    calibration = json.loads(content)
    commands = np.asarray(calibration.get("command_knots_N"), dtype=float)
    realized = np.asarray(calibration.get("realized_squeeze_knots_N"), dtype=float)
    if commands.ndim != 1 or realized.shape != commands.shape or len(commands) < 2:
        raise ValueError("Invalid calibration knots")
    if not np.isfinite(np.r_[commands, realized]).all():
        raise ValueError("Nonfinite calibration")
    if np.any(np.diff(commands) <= 0) or np.any(np.diff(realized) < -1e-12):
        raise ValueError("Calibration must be increasing in command and nondecreasing in realization")
    if commands[0] != 0.5 or commands[-1] != 15.0:
        raise ValueError("Calibration must cover frozen support [0.5, 15] N")
    if np.any((realized < 0) | (realized > 15)):
        raise ValueError("Realized squeeze outside engineering scale")
    if calibration.get("fit_split") != "TRAIN" or calibration.get("feasibility_input") is not False:
        raise ValueError("Calibration leakage contract failed")
    return calibration


def realized_squeeze(force, calibration: dict):
    grid = np.asarray(force, dtype=float)
    commands = np.asarray(calibration["command_knots_N"], dtype=float)
    realized = np.asarray(calibration["realized_squeeze_knots_N"], dtype=float)
    if not np.isfinite(grid).all() or np.any((grid < commands[0]) | (grid > commands[-1])):
        raise ValueError("Force outside calibrated support")
    return np.interp(grid, commands, realized)


def expected_utility(probability, force, calibration: dict, engineering_scale_N: float = 15.0):
    p = np.asarray(probability, dtype=float)
    f = np.asarray(force, dtype=float)
    maximum = float(engineering_scale_N)
    if p.shape != f.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probability/force vectors")
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("Invalid engineering force scale")
    realized = realized_squeeze(f, calibration)
    return p * (maximum - realized) / maximum - (1.0 - p)


def select_force(force_grid, probability, force_support, calibration: dict) -> dict:
    grid = np.asarray(force_grid, dtype=float)
    p = np.asarray(probability, dtype=float)
    lower, upper = map(float, force_support)
    if grid.ndim != 1 or p.shape != grid.shape or grid.size == 0:
        raise ValueError("Grid and probability must be matching vectors")
    if [lower, upper] != [0.5, 15.0] or np.any(np.diff(grid) <= 0):
        raise ValueError("Frozen support and ascending grid required")
    if not np.isclose(grid[0], lower) or not np.isclose(grid[-1], upper):
        raise ValueError("Grid endpoints must match support")
    utility = expected_utility(p, grid, calibration, engineering_scale_N=15.0)
    index = int(np.argmax(utility))
    realized = realized_squeeze(grid, calibration)
    return {
        "selected_force_N": float(grid[index]),
        "predicted_success": float(p[index]),
        "predicted_realized_squeeze_N": float(realized[index]),
        "utility": float(utility[index]),
        "force_grid_N": grid.tolist(),
        "p_success": p.tolist(),
        "predicted_realized_squeeze_curve_N": realized.tolist(),
        "expected_utility": utility.tolist(),
        "force_feature_normalization_N": 15.0,
        "utility_engineering_scale_N": 15.0,
        "utility_definition": UTILITY_DEFINITION,
        "tie_break": "ascending force, first maximum",
    }
