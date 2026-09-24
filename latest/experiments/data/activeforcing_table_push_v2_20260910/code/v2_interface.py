"""Preregistered causal direction and bounded XY-only calibration residual."""
import hashlib
import numpy as np

MIN_NORM=.01
N_SELECTED=15
def direction_from_fresh_chunk(chunk):
    a=np.asarray(chunk,dtype=float)
    if a.ndim!=2 or a.shape[1]!=7: raise ValueError('fresh action chunk must be [T,7]')
    candidates=[i for i,x in enumerate(a) if np.linalg.norm(x[:2])>=MIN_NORM]
    chosen=candidates[:N_SELECTED]
    if not chosen: raise ValueError('degenerate fresh chunk: no XY action >= 0.01')
    aggregate=np.median(a[chosen,:2],axis=0)
    norm=float(np.linalg.norm(aggregate))
    if norm<MIN_NORM: raise ValueError('degenerate fresh chunk median')
    return {'raw_chunk':a.tolist(),'chunk_sha256':hashlib.sha256(a.tobytes()).hexdigest(),
            'selected_indices':chosen,'aggregate_xy':aggregate.tolist(),
            'direction_world_xy':(aggregate/norm).tolist(),'min_norm':MIN_NORM}

class ResidualRamp:
    """A fixed normalized scalar calibration intervention, not a force controller."""
    def __init__(self,target,cap=.15,ramp=.02):
        if abs(target)>cap: raise ValueError('target exceeds hard cap')
        self.target=float(target);self.cap=float(cap);self.ramp=float(ramp);self.current=0.
    def apply(self,nominal,direction_xy,enabled):
        a=np.asarray(nominal,float).copy()
        before=self.current
        desired=self.target if enabled else 0.
        self.current=float(np.clip(desired,before-self.ramp,before+self.ramp))
        d=np.asarray(direction_xy,float)
        residual=self.current*d if enabled else np.zeros(2)
        a[:2]=np.clip(a[:2]+residual,-1,1)
        return a,{'residual_scalar':self.current,'residual_xy':residual.tolist(),'cap':self.cap,'ramp':self.ramp,
                  'clipped':bool(abs(self.target)>self.cap),'xy_only':bool(np.array_equal(a[2:],np.asarray(nominal)[2:]))}
