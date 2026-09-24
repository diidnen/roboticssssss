"""AF-MASS, decision-oracle GT-MASS, and Fixed-4 execution adapters."""
from __future__ import annotations

import copy
import numpy as np

from mass_feasibility import MassFeasibility


VALID_METHODS = ("ACTIVEFORCING_MASS", "GT_MASS", "FIXED_4")


class MassMethod(MassFeasibility):
    def __init__(self, method, true_mass_kg=None, *args, **kwargs):
        if method not in VALID_METHODS: raise ValueError("method outside MASS protocol")
        if method == "GT_MASS":
            checked_mass = float(true_mass_kg)
            if not np.isfinite(checked_mass) or checked_mass <= 0: raise ValueError("invalid GT mass")
        elif true_mass_kg is not None:
            raise ValueError("true simulator mass must not enter AF-MASS or Fixed-4 planner construction")
        else:
            checked_mass = None
        super().__init__(*args, **kwargs)
        self.method = method
        self.true_mass_kg = checked_mass

    def select(self, preaction_sequence, posterior):
        used = copy.deepcopy(posterior)
        if self.method == "GT_MASS":
            used.update(integration_nodes=[self.true_mass_kg], integration_weights=[1.0],
                        posterior_moments={"mean": self.true_mass_kg, "std": 0.0, "variance": 0.0})
        decision = super().select(preaction_sequence, used)
        model_selected = float(decision["selected_force_N"])
        executed = 4.0 if self.method == "FIXED_4" else model_selected
        decision.update(
            method=self.method, model_selected_force_N=model_selected, executed_force_N=executed,
            physics_information_source=("PRIVILEGED_SIMULATOR_TOTAL_MASS_DELTA" if self.method == "GT_MASS"
                                        else "P4B_MASS_CONTINUOUS_POSTERIOR" if self.method == "ACTIVEFORCING_MASS"
                                        else "FIXED_4N_BASELINE"),
            gt_mass_used=self.method == "GT_MASS", fixed_force_baseline=self.method == "FIXED_4",
            simulator_mass_kg=(self.true_mass_kg if self.method == "GT_MASS" else None),
            used_integration_nodes=used["integration_nodes"], used_integration_weights=used["integration_weights"],
            outcome_used_for_selection=False, model_or_utility_modified=False,
        )
        return decision
