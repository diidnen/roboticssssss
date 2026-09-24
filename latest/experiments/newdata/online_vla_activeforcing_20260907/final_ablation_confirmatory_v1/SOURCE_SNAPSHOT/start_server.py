import os, subprocess
from pathlib import Path
from common import HERE, VTLA, write

out = HERE/'server_v2'
out.mkdir(exist_ok=False)
env = os.environ.copy()
env.update(PYTHONPATH=os.pathsep.join(['/media/volume/newdata/exouser/tabero/uv-cache/archive-v0/FiR5so0BgJNw-JER',
    '/media/volume/newdata/exouser/softvtbench/openpi-venv/lib/python3.11/site-packages',
    str(VTLA/'src'),str(VTLA/'packages/openpi-client/src'),'/home/exouser/Tabero/benchmarks/openpi/openpi-client/src']),
    XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='4')
cmd = [str(VTLA/'.venv/bin/python'), '-u', str(HERE/'server.py'), '--out',str(out),'--port','18885']
with (out/'SERVER.log').open('x') as f:
    proc = subprocess.Popen(cmd, cwd=VTLA, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
write(out/'PROCESS.json',dict(pid=proc.pid,command=cmd))
print(proc.pid)
