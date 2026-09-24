"""Locate loss of target contact in complete paired engineering traces."""
import argparse
import gzip
import heapq
import json
from pathlib import Path
import numpy as np


def review(branch):
    peak=[];streak=0;first_loss=None;last_contact=None;normal=[];target=[];other=0
    sampled=[]
    with gzip.open(branch/'physics_trace.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);c=row['contact'];step=row['physics_step']
            normal.append(c['measured_squeeze_n']);target.append(c['target_measured_squeeze_n'])
            other_names=sorted(set(p['other'] for finger in c['fingers'] for p in finger['points'] if not p['is_target']))
            other+=bool(other_names)
            compact={'step':step,'squeeze_n':c['measured_squeeze_n'],
                     'target_squeeze_n':c['target_measured_squeeze_n'],
                     'finger_joints_m':c['actual_finger_joint_m'],'other_contacts':other_names,
                     'target_point_counts':[sum(p['is_target'] for p in f['points']) for f in c['fingers']]}
            if len(peak)<5:heapq.heappush(peak,(c['measured_squeeze_n'],step,compact))
            elif c['measured_squeeze_n']>peak[0][0]:heapq.heapreplace(peak,(c['measured_squeeze_n'],step,compact))
            if c['target_measured_squeeze_n']>.2:streak=0;last_contact=step
            else:
                streak+=1
                if streak==25 and first_loss is None:first_loss={'first_step':step-24,'receipt':compact}
            if step in (1,5,10,25,50,100,200,300,500,1000):sampled.append(compact)
    result={'scope':'descriptive force/contact diagnostics, not a causal claim','branch':branch.name,
            'physics_steps':len(normal),'target_contact_fraction':float(np.mean(np.asarray(target)>.2)),
            'first_25step_low_target_contact_streak':first_loss,'last_target_contact_step':last_contact,
            'steps_with_non_target_contact':other,'largest_squeeze_samples':[p[2] for p in sorted(peak,reverse=True)],
            'prefix_samples':sampled}
    (branch/'CONTACT_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args()
    for branch in sorted(args.run.glob('branch_*')):
        if branch.name.startswith('branch_1_'):continue
        print(json.dumps(review(branch),indent=2))
