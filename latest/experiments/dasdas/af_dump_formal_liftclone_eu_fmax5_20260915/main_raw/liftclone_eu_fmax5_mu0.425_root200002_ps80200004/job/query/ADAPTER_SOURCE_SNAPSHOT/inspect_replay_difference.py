"""Find first exposed-state divergence without altering archived observations."""
import gzip
import json
from pathlib import Path
import sys
import numpy as np

root=Path(sys.argv[1])
left=json.loads(gzip.open(root/'original.json.gz','rt').read())
right=json.loads(gzip.open(root/'original_replay.json.gz','rt').read())

def differences(a,b,path=''):
    if isinstance(a,dict):
        return [r for k in a for r in differences(a[k],b[k],path+'/'+k)]
    if isinstance(a,list):
        if len(a)!=len(b):return [{'path':path,'kind':'different lengths','a':len(a),'b':len(b)}]
        return [r for i in range(len(a)) for r in differences(a[i],b[i],path+'/'+str(i))]
    if a!=b:
        return [{'path':path,'a':a,'b':b,'delta':float(b-a) if isinstance(a,(int,float)) else None}]
    return []

summary=[]
for index in (0,1,2,len(left)-1):
    ds=differences(left[index].get('exposed_state',{}),right[index].get('exposed_state',{}))
    summary.append({'step':index,'different_state_scalars':len(ds),'state_differences':ds[:18],
        'squeeze':[left[index]['measured_squeeze_n'],right[index]['measured_squeeze_n']],
        'actual_joints':[left[index]['actual_finger_joint_m'],right[index]['actual_finger_joint_m']]})
report={'scope':'diagnosis only; raw evidence retained','first_differences':summary}
(root/'FIRST_REPLAY_DIFFERENCES.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
