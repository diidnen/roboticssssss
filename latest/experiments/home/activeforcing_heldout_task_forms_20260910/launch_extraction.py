"""Isolated extraction preflight/qualification, keeping original results intact."""
from pathlib import Path
import os,subprocess,json,hashlib,sys
P=Path(__file__).resolve().parent
python=Path('/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python')
env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',AF_PREFLIGHT_DIR='preflight_open_drawer',
    PYTHONPATH=os.pathsep.join([str(P),str(python.parent.parent/'lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64'),'/home/exouser/Tabero','/home/exouser/Tabero/benchmarks/openpi/openpi-client/src']),
    TABERO_ROOT='/home/exouser/Tabero',HDF5_TRAJ_SOURCE_DIR='/home/exouser/Tabero/benchmarks/datasets/libero/assembled_hdf5')
if '--preflight' in sys.argv:
    (P/'preflight_open_drawer').mkdir(exist_ok=True)
    with (P/'preflight_open_drawer/PREFLIGHT.log').open('x') as f:
        r=subprocess.run([str(python),str(P/'preflight_extraction.py'),'--suite','libero_spatial','--task','4'],env=env,cwd='/home/exouser/Tabero',stdout=f,stderr=subprocess.STDOUT,timeout=240)
    print('preflight_exit',r.returncode)
else:
    pre=json.loads((P/'preflight_open_drawer/libero_spatial_4/PREFLIGHT_RESULT.json').read_text())
    names=pre['cabinet_joint_names'];pos=pre['cabinet_joint_positions'][0]
    assert abs(pos[names.index('top_level')]+.1523311883211136)<1e-5
    freeze=P/'qualification/EXTRACTION_COMPATIBILITY_FREEZE.json'
    if not freeze.exists():freeze.write_text(json.dumps({'task':'libero_spatial:4','new_task_outcomes_seen':0,'threshold':'unchanged 5/6','context_order':'unchanged',
        'source_hashes':{f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ['qualify_worker.py','qualify_extraction.py','extraction_reset_compatibility.py','launch_extraction.py']},
        'preflight':pre},indent=2))
    m=json.loads(freeze.read_text());assert all(hashlib.sha256((P/f).read_bytes()).hexdigest()==h for f,h in m['source_hashes'].items())
    failed=0
    for root in [5100,6100]:
        for mu in [.3,.5,.9]:
            if failed>=2:continue
            name=f'libero_spatial_4_r{root}_mu{mu:.2f}';receipt=P/'qualification'/name/'QUALIFICATION_RESULT.json'
            if not receipt.exists():
                with (P/'qualification'/f'{name}.log').open('x') as f:
                    r=subprocess.run([str(python),str(P/'qualify_extraction.py'),'--suite','libero_spatial','--task','4','--root',str(root),'--mu',str(mu)],env=env,cwd='/home/exouser/Tabero',stdout=f,stderr=subprocess.STDOUT,timeout=900)
            x=json.loads(receipt.read_text());print(name,{k:x.get(k) for k in ['valid_outcome','full_screen_success','failure','error']},flush=True)
            if not x['valid_outcome']:raise RuntimeError('Invalid extraction screen; retained')
            failed+=not x['full_screen_success']
