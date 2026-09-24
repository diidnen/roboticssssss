"""Auditable planar residual control for LIBERO OSC_POSE actions.

The target is a measured robot-on-plate force in N, never an action component.
"""
from dataclasses import dataclass
import numpy as np

def unit_xy(v):
    v=np.asarray(v,dtype=float).copy(); v[2:]=0.; n=float(np.linalg.norm(v[:2]))
    if n < 1e-9: raise ValueError("causal push direction is degenerate")
    return v/n

@dataclass
class PushForceController:
    target_n: float
    kp: float = 0.004
    ki: float = 0.0002
    residual_limit: float = 0.020
    ema_alpha: float = 0.25
    enabled: bool = True
    integral_limit: float = 100.0
    ema_force: float | None = None
    integral: float = 0.0

    def apply(self, nominal, push_direction, raw_force_n, stable_contact):
        a=np.asarray(nominal,dtype=float).copy()
        if a.shape[-1] != 7: raise ValueError("expected 7-D OSC_POSE action")
        d=unit_xy(push_direction)
        if not self.enabled or not stable_contact:
            return a, {"raw_force_n":float(raw_force_n),"ema_force_n":self.ema_force,"residual_xy":[0.,0.],"clipped":False,"active":False}
        f=float(raw_force_n)
        self.ema_force=f if self.ema_force is None else self.ema_alpha*f+(1-self.ema_alpha)*self.ema_force
        err=self.target_n-self.ema_force
        candidate_integral=float(np.clip(self.integral+err,-self.integral_limit,self.integral_limit))
        scalar=self.kp*err+self.ki*candidate_integral
        clipped=abs(scalar)>self.residual_limit
        scalar=float(np.clip(scalar,-self.residual_limit,self.residual_limit))
        # anti-windup: retain integration only if output did not saturate toward error.
        if not clipped or np.sign(scalar)!=np.sign(err): self.integral=candidate_integral
        residual=scalar*d[:2]
        a[:2]=np.clip(a[:2]+residual,-1.,1.)
        return a, {"raw_force_n":f,"ema_force_n":float(self.ema_force),"error_n":float(err),"residual_xy":residual.tolist(),"residual_scalar":scalar,"clipped":clipped,"active":True}
