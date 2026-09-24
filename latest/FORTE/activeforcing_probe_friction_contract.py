"""Diagnostic friction budget; thresholds and raw belief tensors are unchanged.

Real PhysX friction patches in a common world frame, projected on the probe
direction and relative to ten genuine hold observations. NOT approved for
feasibility collection until posterior transfer is independently validated.
"""
import numpy as np


class FrictionProbeBudget:
    contract_id = 'PHYSX_FRICTION_PATCH_INCREMENT_ALONG_PROBE_WORLD_V1'

    def __init__(self, records, quat_apply):
        self.records = records
        self.quat_apply = quat_apply
        self.hold = []
        self.baseline = None

    def __call__(self, *, env, phase, tangent_base, step, legacy_rho):
        record = self.records[-1]
        if record['step'] != step:
            raise RuntimeError('Stale friction observation')
        patches = record['patches']
        if len(patches['normal']) != 2 or len(patches['friction']) != 2:
            raise RuntimeError('Unexpected contact sensor/filter layout')
        friction = np.sum([r['sum_friction_vector_world'] for r in patches['friction']],axis=0)
        normal = 2 * min(sum(float(v[0]) for v in r['normal_forces']) for r in patches['normal'])
        if phase == 'hold': self.hold.append(friction.copy())
        is_probe = phase in ('probe_out', 'probe_back', 'probe_hold')
        ratio = 0.
        if is_probe:
            if self.baseline is None:
                if len(self.hold) < 10: raise RuntimeError('Missing genuine pre-probe hold baseline')
                self.baseline = np.mean(self.hold[-10:],axis=0)
            root_q=env.scene['robot'].data.root_quat_w[0].detach().cpu().numpy()
            tangent_world=self.quat_apply(root_q,np.asarray(tangent_base,float))
            tangent_world=tangent_world/np.linalg.norm(tangent_world)
            incremental=abs(float(np.dot(friction-self.baseline,tangent_world)))
            ratio=incremental/max(normal,1e-6)
        record['termination_signal']={'contract_id':self.contract_id,'phase':phase,
            'legacy_normal_projection_ratio':legacy_rho,'physical_friction_world':friction.tolist(),
            'normal_patch_bilateral_N':normal,'baseline_friction_world':None if self.baseline is None else self.baseline.tolist(),
            'incremental_directional_friction_ratio':ratio}
        return ratio
