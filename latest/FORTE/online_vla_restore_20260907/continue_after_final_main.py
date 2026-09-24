"""Audited main table, then frozen dev ablations, without outcome retries."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from common import HERE,ROOT,read,write,sha

PYTHON=Path('/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python')

def run(control,final,main_pid,ablation,main_results,figures,ablation_results):
    control.mkdir(exist_ok=False)
    commands=[
        ('main_results',[final/'SOURCE_SNAPSHOT/build_final_online_results.py','--out',final,'--destination',main_results]),
        ('main_figures',[HERE/'render_online_vla_main.py','--source',main_results/'FINAL_ONLINE_VLA_MAIN_RESULTS.json','--destination',figures]),
        ('online_ablations',[ablation/'SOURCE_SNAPSHOT/run_online_ablations.py','--out',ablation]),
        ('ablation_results',[HERE/'build_online_ablation_results.py','--out',ablation,'--destination',ablation_results]),
    ]
    bound={str(args[0]):sha(args[0]) for _,args in commands}
    write(control/'POST_MAIN_QUEUE_STARTED.json',dict(pid=os.getpid(),waiting_main_pid=main_pid,
        source_final=str(final),source_ablation=str(ablation),stage_sources=bound,
        stages=[dict(name=n,arguments=list(map(str,a))) for n,a in commands],
        no_outcome_retry=True,main_outcomes_cannot_tune_ablations=True,
        paper_package_and_final_completion_audit_still_required=True))
    done=final/'FINAL_MAIN_EXECUTION_COMPLETE.json'
    while not done.exists():
        if (control/'STOP_AUTOMATION.json').exists():return
        p=Path(f'/proc/{main_pid}/cmdline')
        if not p.exists() or str(final).encode() not in p.read_bytes():
            raise RuntimeError('Main coordinator ended before complete192 evidence')
        time.sleep(5)
    if read(done)['branches']!=192:raise RuntimeError('Incomplete main denominator')
    for p,d in bound.items():
        if sha(p)!=d:raise RuntimeError('Queued analysis/ablation source changed')
    env=os.environ.copy();env.update(PYTHONPATH=os.pathsep.join((str(HERE),str(ROOT))),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for name,args in commands:
        if (control/'STOP_AUTOMATION.json').exists():return
        with (control/(name+'.log')).open('x') as log:
            child=subprocess.Popen([str(PYTHON),'-u',*map(str,args)],cwd=HERE,env=env,
                stdout=log,stderr=subprocess.STDOUT)
            write(control/(name+'_PROCESS.json'),dict(pid=child.pid,command=[str(PYTHON),*map(str,args)]))
            print('POST_MAIN_STAGE_STARTED',name,child.pid,flush=True)
            rc=child.wait()
        write(control/(name+'_EXIT.json'),dict(exit_code=rc))
        if rc!=0:raise RuntimeError('Stage failed; no automatic retry: '+name)
        print('POST_MAIN_STAGE_COMPLETED',name,flush=True)
    write(control/'ANALYSIS_AND_ABLATIONS_COMPLETE_PAPER_PENDING.json',dict(
        main_results=str(main_results),main_figures=str(figures),ablation_results=str(ablation_results),
        final_goal_complete=False,paper_package_and_completion_audit_required=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True)
    p.add_argument('--final',type=Path,required=True);p.add_argument('--main-pid',type=int,required=True)
    p.add_argument('--ablation',type=Path,required=True);p.add_argument('--main-results',type=Path,required=True)
    p.add_argument('--figures',type=Path,required=True);p.add_argument('--ablation-results',type=Path,required=True)
    a=p.parse_args()
    try:run(a.control,a.final,a.main_pid,a.ablation,a.main_results,a.figures,a.ablation_results)
    except Exception as e:
        if a.control.exists() and not (a.control/'AUTOMATION_FAILED.json').exists():
            write(a.control/'AUTOMATION_FAILED.json',dict(error=repr(e),no_automatic_retry=True))
        raise
