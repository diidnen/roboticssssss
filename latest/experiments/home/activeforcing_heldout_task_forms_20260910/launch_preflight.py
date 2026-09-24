"""Sequential native scene preflight; never evaluates a force-selection method."""
from pathlib import Path
import os,subprocess,json
OUT=Path(__file__).resolve().parent
python=Path('/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python')
env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',
    PYTHONPATH=os.pathsep.join([str(python.parent.parent/'lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64'),'/home/exouser/Tabero','/home/exouser/Tabero/benchmarks/openpi/openpi-client/src']),
    TABERO_ROOT='/home/exouser/Tabero',HDF5_TRAJ_SOURCE_DIR='/home/exouser/Tabero/benchmarks/datasets/libero/assembled_hdf5')
directory=os.environ.get('AF_PREFLIGHT_DIR','preflight')
(OUT/directory).mkdir(exist_ok=True)
for suite,task in [('libero_10',5),('libero_goal',9),('libero_spatial',4),('libero_goal',0)]:
    if (OUT/directory/f'{suite}_{task}'/'PREFLIGHT_RESULT.json').exists():continue
    log=OUT/directory/f'{suite}_{task}.log'
    with log.open('x') as f:
        result=subprocess.run([str(python),str(OUT/'preflight_worker.py'),'--suite',suite,'--task',str(task)],cwd='/home/exouser/Tabero',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
    receipt=OUT/directory/f'{suite}_{task}'/'PREFLIGHT_RESULT.json'
    print(suite,task,result.returncode,receipt.read_text()[:700] if receipt.exists() else 'NO_RECEIPT',flush=True)
