"""Cold-start replay unit test, NOT a scientific task replacement."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from test_force_cache_dynamics import make_scene
from qualify_native_interfaces import Snapshot,readback,digest
import af_native_joint_readback as native

scene,art,link,joints=make_scene('contact',True)
art.set_qf(np.zeros(art.dof))
joints[-1].set_drive_target(.005)
for _ in range(400): scene.step()
env=SimpleNamespace(scene=scene,now_obs={},robot=SimpleNamespace())
snapshot=Snapshot(env)
rows=[]
for branch in range(3):
    receipt=native.cold_solver_reset(link,flush=os.environ.get('AF_DIAGNOSTIC_FLUSH_BUFFERS')=='1',
                                    canonical_ids=os.environ.get('AF_DIAGNOSTIC_CANONICAL_ACTOR_IDS')=='1',
                                    canonical_shapes=os.environ.get('AF_DIAGNOSTIC_CANONICAL_SHAPE_IDS')=='1')
    snapshot.restore()
    trace=[]
    for _ in range(100):
        scene.step()
        trace.append({'state':readback(env),'force':native.read_link(link)})
    rows.append({'branch':branch,'receipt':receipt,'trace_sha256':digest(trace)})
    joints[-1].set_drive_target(-.005)
    for _ in range(75): scene.step()
snapshot.restore()
report={'scope':'cold solver replay UNIT TEST','branches':rows,
        'passed':len({r['trace_sha256'] for r in rows})==1}
suffix='_CANONICAL_IDS' if os.environ.get('AF_DIAGNOSTIC_CANONICAL_ACTOR_IDS')=='1' else ''
if os.environ.get('AF_DIAGNOSTIC_CANONICAL_SHAPE_IDS')=='1':suffix+='_SHAPES'
Path(__file__).with_name('COLD_SOLVER_REPLAY_UNIT_TEST'+suffix+'.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
assert report['passed']
