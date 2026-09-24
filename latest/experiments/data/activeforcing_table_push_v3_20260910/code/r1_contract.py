"""Pure R1 safety primitives, tested before simulator use."""
import numpy as np
def final_action_safe(nominal_xy,residual_xy,margin=.01):
    return bool(np.all(np.abs(np.asarray(nominal_xy)+np.asarray(residual_xy)) < 1-margin))
def safe_symmetric_grid(nominal_xy,margin=.01,requested=(.02,.04,.06)):
    # Scalar residuals along runtime direction are filtered by caller with actual direction.
    headroom=float(np.min(1-margin-np.abs(np.asarray(nominal_xy))))
    allowed=[x for x in requested if x < headroom]
    if not allowed: raise ValueError('no symmetric residual has final-action headroom')
    x=max(allowed); return np.array([-x,-x/2,0.,x/2,x])
