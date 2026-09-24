import gzip,json
from pathlib import Path
import numpy as np
folder=Path(__file__).parent/'force_zero_friction_v1'
with gzip.open(folder/'zero_friction.json.gz','rt') as f: rows=json.load(f)
result=[]
for i in range(2):
    data=[r['fingers'][i] for r in rows]
    reads=[f['native_joint_readback'] for f in data]
    normals=np.array([f['force_world_n'] for f in data])
    errors=np.array([r['unqualified_contact_balance_world_n'] for r in reads])-normals
    velocity=np.array([r['com_linear_velocity_world'] for r in reads])
    acceleration=np.array([r['com_linear_acceleration'] for r in reads])
    gravity=np.array([r['gravity_world'] for r in reads])
    force=np.array([r['joint_force_world_n'] for r in reads])
    mass=reads[0]['mass_kg']
    fd=np.diff(velocity,axis=0)/.004
    fd_error=mass*(fd-gravity[1:])-force[1:]-normals[1:]
    result.append({'finger':data[0]['finger'],'mass_kg':mass,
                   'gravity_disabled':reads[0]['gravity_disabled'],
                   'cache_error_mean':errors.mean(0).tolist(), 'fd_error_mean':fd_error.mean(0).tolist(),
                   'velocity_min':velocity.min(0).tolist(),'velocity_max':velocity.max(0).tolist(),
                   'cache_accel_vs_fd_p95':np.percentile(abs(fd-acceleration[1:]),95,axis=0).tolist(),
                   'first':{'error':errors[0].tolist(),'velocity':velocity[0].tolist(),
                            'qf':rows[0]['qf'],'native':reads[0]}})
print(json.dumps(result,indent=2))
(folder/'RESIDUAL_DIAGNOSIS.json').write_text(json.dumps(result,indent=2))
