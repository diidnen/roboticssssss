"""User-corrected ActiveForcing expected utility, parameterized by maxF."""
import numpy as np


def expected_utility(probability, force, max_force):
    p = np.asarray(probability, dtype=float)
    f = np.asarray(force, dtype=float)
    maximum = float(max_force)
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError('maxF must be finite and positive')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid success probability')
    if not np.isfinite(f).all() or np.any((f < 0) | (f > maximum)):
        raise ValueError('Force outside [0, maxF]')
    return p * (maximum - f) / maximum - (1.0 - p)


def select_force(force_grid, probability, force_support):
    grid, p = np.asarray(force_grid, dtype=float), np.asarray(probability, dtype=float)
    lower, maximum = map(float, force_support)
    if grid.ndim != 1 or p.shape != grid.shape or grid.size == 0:
        raise ValueError('Grid and probability must be matching vectors')
    if not 0 <= lower < maximum or np.any(np.diff(grid) <= 0):
        raise ValueError('Support and ascending grid required')
    if not np.isclose(grid[0], lower) or not np.isclose(grid[-1], maximum):
        raise ValueError('Grid endpoints must match configured force support')
    utility = expected_utility(p, grid, maximum)
    i = int(np.argmax(utility))
    return {'selected_force_N': float(grid[i]), 'predicted_success': float(p[i]),
            'utility': float(utility[i]), 'force_grid_N': grid.tolist(),
            'p_success': p.tolist(), 'expected_utility': utility.tolist(),
            'utility_normalization_N': maximum, 'utility_definition': 'p*(maxF-F)/maxF-(1-p)',
            'tie_break': 'ascending force, first maximum'}
