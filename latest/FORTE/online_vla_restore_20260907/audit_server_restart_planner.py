"""Offline effect of process-restart numerical variation on the frozen planner."""
import argparse
import sys
from pathlib import Path
import numpy as np
from common import ROOT,read,write,sha
sys.path.insert(0,str(ROOT))
from phase_free_feasibility import PhaseFreeFeasibility

def run(qualification,reinference,out):
    jobs=sorted((qualification/'branches').glob('*__ACTIVEFORCING'))
    if len(jobs)!=12:raise RuntimeError('Require all12 qualified decision contexts')
    if any(not (reinference/p.name/'RPC/0001.npz').exists() for p in jobs):
        raise RuntimeError('Not all12 new-server initial observations have been inferred')
    model=PhaseFreeFeasibility(device='cpu');rows=[]
    for job in jobs:
        new_path=reinference/job.name/'RPC/0001.npz'
        with np.load(new_path) as a:commands=a['postprocessed_vla_action'][:8,:3].astype(np.float32).astype(np.float64)
        old_x=np.load(job/'PREACTION_SEQUENCE.npy');new_x=old_x.copy()
        new_x[:,:3]=commands-commands[0]
        new_x[:,3:6]=np.vstack([np.zeros((1,3)),np.diff(commands,axis=0)])
        posterior=read(job/'PREACTION_POSTERIOR.json');old=read(job/'PLANNER_DECISION.json')
        original=model.select(old_x,posterior);new=model.select(new_x,posterior)
        if not np.allclose(original['p_success'],old['p_success'],atol=1e-7,rtol=0):
            raise RuntimeError('Frozen planner no longer reproduces qualification')
        rows.append(dict(context=read(job/'BRANCH_RESULT.json')['plan']['id'],old_force=old['selected_force_N'],
            new_force=new['selected_force_N'],force_equal=old['selected_force_N']==new['selected_force_N'],
            max_probability_delta=float(np.max(np.abs(np.array(new['p_success'])-old['p_success']))),
            old_decision_sha256=sha(job/'PLANNER_DECISION.json'),new_initial_chunk_sha256=sha(new_path),
            new_decision=new))
    write(out,dict(role='COMPLETE_OFFLINE_RESTART_NUMERICS_PLANNER_CHECK',contexts=12,rows=rows,
        all_selected_forces_equal=all(r['force_equal'] for r in rows),
        max_probability_delta=max(r['max_probability_delta'] for r in rows),
        simulator_steps=0,training_steps=0,final_roots_used=False,
        implementation_sha256=sha(__file__)))
    print('RESTART_PLANNER_REVIEW',len(rows),all(r['force_equal'] for r in rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--reinference',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.qualification,a.reinference,a.out)
