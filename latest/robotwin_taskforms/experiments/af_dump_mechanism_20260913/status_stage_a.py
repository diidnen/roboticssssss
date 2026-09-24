"""Read-only compact status for the finite force-window queue."""
import json,os,shutil
from pathlib import Path
H=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
protocol=read(H/'PROTOCOL.json');rows=[]
for case in protocol['cases']:
    folder=H/case['id'];job=folder/'job';result=[]
    for branch in sorted(job.glob('branch_*')):
        if (branch/'result.json').exists():
            r=read(branch/'result.json');result.append({'branch':branch.name,'force':r['force_setpoint_bilateral_n'],
                'success':r['success'],'trace_sha256':r['trace_sha256']})
    item={'id':case['id'],'completed_branches':len(result),'results':result,'audit_complete':(folder/'CASE_AUDIT.json').exists()}
    if (job/'PREACTION_SELECTION_LOCK.json').exists():
        current=read(job/'PREACTION_SELECTION_LOCK.json');old=read(case['historical_lock'])
        item.update(historical_state_exact=current['state_sha256']==old['state_sha256'],
                    historical_first_chunk_exact=current['first_chunk_sha256']==old['first_chunk_sha256'])
    if (folder/'EXIT.json').exists():item['exit']=read(folder/'EXIT.json')
    rows.append(item)
pid=read(H/'QUEUE_PID.json')['pid']
try:os.kill(pid,0);live=True
except ProcessLookupError:live=False
print(json.dumps({'queue_pid':pid,'queue_live':live,'planned_rollouts':12,
    'completed_rollouts':sum(r['completed_branches'] for r in rows),
    'stage_a_complete':(H/'STAGE_A_COMPLETE.json').exists(),'free_bytes':shutil.disk_usage(H).free,'cases':rows}))
