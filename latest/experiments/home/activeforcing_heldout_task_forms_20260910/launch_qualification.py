"""Fixed frozen context order with prospective 5/6 futility stopping."""
from pathlib import Path
import os,subprocess,json,hashlib,resource
OUT=Path(__file__).resolve().parent
python=Path('/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python')
env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',
    PYTHONPATH=os.pathsep.join([str(python.parent.parent/'lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64'),'/home/exouser/Tabero','/home/exouser/Tabero/benchmarks/openpi/openpi-client/src']),
    TABERO_ROOT='/home/exouser/Tabero',HDF5_TRAJ_SOURCE_DIR='/home/exouser/Tabero/benchmarks/datasets/libero/assembled_hdf5')
(OUT/'qualification').mkdir(exist_ok=True)
freeze=OUT/'qualification/EXECUTION_FREEZE.json'
if not freeze.exists():freeze.write_text(json.dumps({'AF_calls':0,'task_order':['libero_10:5','libero_goal:9','libero_spatial:4'],
    'threshold':'5/6','futility_stop':'second valid failed context','files':{f:hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in ['qualify_worker.py','launch_qualification.py','QUALIFICATION_PROTOCOL.md','QUALIFICATION_EXECUTION_ADDENDUM.md']}},indent=2))
manifest=json.loads(freeze.read_text())
assert all(hashlib.sha256((OUT/f).read_bytes()).hexdigest()==h for f,h in manifest['files'].items())
for suite,task in [('libero_10',5),('libero_goal',9),('libero_spatial',4)]:
    failures=0
    for root in [5100,6100]:
        for mu in [.3,.5,.9]:
            job=OUT/'qualification'/f'{suite}_{task}_r{root}_mu{mu:.2f}'
            if failures>=2:continue
            receipt=job/'QUALIFICATION_RESULT.json'
            if not receipt.exists():
                log=OUT/'qualification'/f'{job.name}.log'
                with log.open('x') as f:
                    result=subprocess.run([str(python),str(OUT/'qualify_worker.py'),'--suite',suite,'--task',str(task),'--root',str(root),'--mu',str(mu)],cwd='/home/exouser/Tabero',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
            if not receipt.exists():raise RuntimeError('Missing worker receipt; infrastructure stop')
            r=json.loads(receipt.read_text());print(job.name,json.dumps({k:r.get(k) for k in ['valid_outcome','full_screen_success','steps','failure','error']}),flush=True)
            if not r['valid_outcome']:raise RuntimeError('Invalid qualification attempt retained; no automatic rerun')
            failures+=not r['full_screen_success']
