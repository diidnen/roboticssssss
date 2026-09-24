"""Frozen adapters for AF, privileged GT-Physics, and Fixed-4."""
from __future__ import annotations

import copy
import numpy as np

from phase_free_feasibility import PhaseFreeFeasibility


VALID_METHODS = ("ACTIVEFORCING", "GT_PHYSICS", "FIXED_4")


class ContinuousFrictionFeasibility(PhaseFreeFeasibility):
    """Uses the unchanged feasibility model; only GT swaps the belief measure."""

    def __init__(self, method: str, mu_gt: float, *args, **kwargs):
        if method not in VALID_METHODS:
            raise ValueError("Method outside frozen generalization plan")
        super().__init__(*args, **kwargs)
        self.method = method
        self.mu_gt = float(mu_gt)
        if not np.isfinite(self.mu_gt) or self.mu_gt <= 0:
            raise ValueError("Invalid object-side intervention friction")

    def select(self, preaction_sequence, posterior):
        used = copy.deepcopy(posterior)
        if self.method == "GT_PHYSICS":
            used.update(
                integration_nodes=[self.mu_gt],
                integration_weights=[1.0],
                posterior_moments={"mean": self.mu_gt, "std": 0.0, "variance": 0.0},
            )
        decision = super().select(preaction_sequence, used)
        model_selected = float(decision["selected_force_N"])
        executed = 4.0 if self.method == "FIXED_4" else model_selected
        decision.update(
            method=self.method,
            model_selected_force_N=model_selected,
            executed_force_N=executed,
            physics_information_source=(
                "PRIVILEGED_OBJECT_SIDE_MU_DELTA"
                if self.method == "GT_PHYSICS"
                else "P4B_PROBE_CONTINUOUS_POSTERIOR"
            ),
            gt_physics_used=self.method == "GT_PHYSICS",
            fixed_force_baseline=self.method == "FIXED_4",
            simulator_friction_value=(self.mu_gt if self.method == "GT_PHYSICS" else None),
            used_integration_nodes=used["integration_nodes"],
            used_integration_weights=used["integration_weights"],
            outcome_used_for_selection=False,
            model_or_utility_modified=False,
        )
        return decision
