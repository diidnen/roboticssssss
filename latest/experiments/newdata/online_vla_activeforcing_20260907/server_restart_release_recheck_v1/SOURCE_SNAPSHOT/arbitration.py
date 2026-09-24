"""Arm = online pi0. Squeeze = AF. Release intent is a VLA semantic request.

Raw aperture never becomes the force controller's aperture reference. While
grasping, the reference is the continuous AF servo state. A model-requested
open maps to canonical .04 and zero force, identically for every method.
No geometry or timestep can trigger a release or replace an arm target.
"""
import numpy as np
VERSION = 'ONLINE_VLA_ARM_AF_SQUEEZE_V2_CANONICAL_OPEN'
# The native aperture is an absolute per-finger position, not a binary score.
# E.g. .026 is the established soup grasp width. Only near-full canonical
# opening denotes release; 1mm tolerance is explicit development qualification.
OPEN_INTENT_THRESHOLD_M = .039

class Arbitration:
    def __init__(self, force, handoff):
        if not 3 <= force <= 5: raise ValueError('Force outside frozen support')
        if not 0 <= handoff <= .04: raise ValueError('Invalid command handoff')
        self.force = float(force); self.command = float(handoff)

    def action(self, postprocessed):
        raw = np.asarray(postprocessed, dtype=np.float32)
        if raw.shape != (13,) or not np.isfinite(raw).all(): raise ValueError('Invalid VLA action')
        opened = bool(raw[6] >= OPEN_INTENT_THRESHOLD_M)
        final = raw.copy(); final[7:13] = 0.
        final[6] = .04 if opened else self.command
        final[9] = final[12] = 0. if opened else self.force / 2
        if not np.array_equal(final[:6], raw[:6]): raise RuntimeError('VLA arm overridden')
        return final, {'raw_vla_arm_command':raw[:6].tolist(), 'raw_vla_gripper_command':float(raw[6]),
            'final_arm_command':final[:6].tolist(), 'selected_force_setpoint':self.force,
            'active_force_setpoint':0. if opened else self.force,
            'final_gripper_command':float(final[6]), 'vla_release_intent':opened,
            'vla_gripper_override_after_af_handoff':False, 'arbitration_version':VERSION}

    def feedback(self, measured, opened, servo):
        # Current frozen P4 force servo; no raw VLA aperture feedback.
        if not opened: self.command = float(servo(self.command, measured, self.force))
