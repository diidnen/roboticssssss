"""Launch original audited policy from its own immutable source snapshot."""
import argparse
import os
import shutil
import subprocess
from pathlib import Path
from common import HERE,VTLA,write,sha

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--port',type=int,default=18885)
    a=p.parse_args();a.out.mkdir(exist_ok=False);snapshot=a.out/'SOURCE_SNAPSHOT';snapshot.mkdir()
    for name in ('server.py','common.py'):shutil.copy2(HERE/name,snapshot/name)
    env=os.environ.copy();env.update(PYTHONPATH=os.pathsep.join([
        '/media/volume/newdata/exouser/tabero/uv-cache/archive-v0/FiR5so0BgJNw-JER',
        '/media/volume/newdata/exouser/softvtbench/openpi-venv/lib/python3.11/site-packages',
        str(VTLA/'src'),str(VTLA/'packages/openpi-client/src'),'/home/exouser/Tabero/benchmarks/openpi/openpi-client/src']),
        XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='4')
    cmd=[str(VTLA/'.venv/bin/python'),'-u',str(snapshot/'server.py'),'--out',str(a.out),'--port',str(a.port)]
    with (a.out/'SERVER.log').open('x') as f:
        process=subprocess.Popen(cmd,cwd=VTLA,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write(a.out/'PROCESS.json',dict(pid=process.pid,command=cmd,source_hashes={str(x):sha(x) for x in snapshot.glob('*.py')}))
    print(process.pid)
