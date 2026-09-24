"""Prospectively frozen secondary-baseline adapters.

The parent ActiveForcing path is unchanged. GT_PHYSICS_DIRECT changes only
the quadrature measure supplied to the frozen feasibility model. The matched
Tabero baseline preserves the online policy arm action and native grasp-time
gripper/force outputs, while sharing the frozen canonical VLA release gate.
"""
import copy
import numpy as np

from phase_free_feasibility import PhaseFreeFeasibility
from arbitration import OPEN_INTENT_THRESHOLD_M


VALID_METHODS = ("ACTIVEFORCING", "GT_PHYSICS_DIRECT", "TABERO_NEUTRAL")


class SecondaryFeasibility(PhaseFreeFeasibility):
    def __init__(self, method, mu_gt, *args, **kwargs):
        if method not in VALID_METHODS:
            raise ValueError("Unfrozen secondary method")
        super().__init__(*args, **kwargs)
        self.method = method
        self.mu_gt = float(mu_gt)
        if not np.isfinite(self.mu_gt) or self.mu_gt <= 0:
            raise ValueError("Invalid simulator friction")

    def select(self, preaction_sequence, posterior):
        used = copy.deepcopy(posterior)
        if self.method == "GT_PHYSICS_DIRECT":
            used.update(
                integration_nodes=[self.mu_gt],
                integration_weights=[1.0],
                posterior_moments={"mean": self.mu_gt, "std": 0.0},
            )
        decision = super().select(preaction_sequence, used)
        decision.update(
            secondary_method=self.method,
            physics_information_source=(
                "PRIVILEGED_SIMULATOR_FRICTION"
                if self.method == "GT_PHYSICS_DIRECT"
                else "P4B_PROBE_POSTERIOR"
            ),
            gt_physics_used=self.method == "GT_PHYSICS_DIRECT",
            simulator_friction_value=(self.mu_gt if self.method == "GT_PHYSICS_DIRECT" else None),
            used_integration_nodes=used["integration_nodes"],
            used_integration_weights=used["integration_weights"],
            outcome_used_for_selection=False,
        )
        if self.method == "TABERO_NEUTRAL":
            decision["counterfactual_activeforcing_selected_force_N"] = decision["selected_force_N"]
            decision["selected_force_N"] = None
        return decision


class NativeTaberoArbitration:
    """Native Tabero grasp behavior plus the shared canonical release gate."""
    VERSION = "MATCHED_TABERO_NATIVE_GRASP_SHARED_CANONICAL_RELEASE_V1"

    def __init__(self):
        self.force = None

    def action(self, postprocessed):
        raw = np.asarray(postprocessed, dtype=np.float32)
        if raw.shape != (13,) or not np.isfinite(raw).all():
            raise ValueError("Invalid VLA action")
        opened = bool(raw[6] >= OPEN_INTENT_THRESHOLD_M)
        final = raw.copy()
        if opened:
            final[6] = np.float32(0.04)
            final[7:13] = 0.0
        if not np.array_equal(final[:6], raw[:6]):
            raise RuntimeError("VLA arm overridden")
        return final, {
            "raw_vla_arm_command": raw[:6].tolist(),
            "raw_vla_gripper_command": float(raw[6]),
            "raw_vla_force_command": raw[7:13].tolist(),
            "final_arm_command": final[:6].tolist(),
            "selected_force_setpoint": None,
            "active_force_setpoint": None,
            "final_gripper_command": float(final[6]),
            "vla_release_intent": opened,
            "vla_gripper_override_after_af_handoff": None,
            "tabero_native_grasp_action_preserved": not opened,
            "arbitration_version": self.VERSION,
        }

    def feedback(self, measured, opened, servo):
        return None
