"""Continue authorized stages only after each complete scientific gate passes.

No automatic retries, changes of seeds/models, or replacement of failed results.
This coordinator preserves a separate process/log/exit receipt for every stage.
"""
import argparse
import os
import subprocess
import time
from pathlib import Path
from common import HERE,ROOT,read,write,sha

PYTHON=Path('/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python')

def run(control,current,coordinator_pid,final):
    control.mkdir(exist_ok=False)
    scripts=('review_current_server36.py','prepare_final_online_vla.py','run_final_online_vla.py')
    hashes={str(HERE/name):sha(HERE/name) for name in scripts}
    write(control/'AUTOMATION_STARTED.json',dict(pid=os.getpid(),waiting_for_pid=coordinator_pid,
        current_qualification=str(current),final_destination=str(final),source_hashes=hashes,
        final_physics_allowed_only_after_positive_complete36_review=True,no_outcome_retry=True,
        automatic_stages=['current36_review','runtime_freeze','final192_main'],
        final_tables_ablations_and_paper_still_required=True))
    done=current/'CURRENT_SERVER36_EXECUTION_COMPLETE.json'
    while not done.exists():
        if (control/'STOP_AUTOMATION.json').exists():
            write(control/'STOPPED_BEFORE_FINAL.json',dict(reason='Explicit stop file'));return
        command=Path(f'/proc/{coordinator_pid}/cmdline')
        if not command.exists() or str(current).encode() not in command.read_bytes():
            raise RuntimeError('Qualification coordinator ended without complete36 evidence')
        time.sleep(5)
    # The coordinator writes completion after every child exited and audited.
    if read(done)['branches']!=36:raise RuntimeError('Qualification count invalid')
    for p,d in hashes.items():
        if sha(p)!=d:raise RuntimeError('Queued stage implementation changed; manual review required')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join((str(HERE),str(ROOT))))
    def stage(name,args):
        if (control/'STOP_AUTOMATION.json').exists():raise RuntimeError('Explicit stop requested')
        log=control/(name+'.log')
        with log.open('x') as f:
            child=subprocess.Popen([str(PYTHON),'-u',*map(str,args)],cwd=HERE,env=env,
                                   stdout=f,stderr=subprocess.STDOUT)
            write(control/(name+'_PROCESS.json'),dict(pid=child.pid,command=[str(PYTHON),*map(str,args)]))
            print('STAGE_STARTED',name,child.pid,flush=True);rc=child.wait()
        write(control/(name+'_EXIT.json'),dict(exit_code=rc))
        if rc!=0:raise RuntimeError('Stage failed, no automatic retry: '+name)
        print('STAGE_COMPLETED',name,flush=True)
    review=HERE/'CURRENT_SERVER36_QUALIFICATION_REVIEW.json'
    admission=HERE/'FINAL_VLA_QUALIFICATION_ADMISSION_V2.json'
    stage('current36_review',[HERE/'review_current_server36.py','--current',current,
        '--old-admission',HERE/'FINAL_VLA_QUALIFICATION_ADMISSION.json',
        '--server-ready',HERE/'server_v3_frozen/SERVER_READY.json',
        '--parity-review',HERE/'SERVER_RESTART14_OUTCOME_PARITY_REVIEW.json','--out',review])
    if not read(review)['passed'] or read(admission)['EXISTING_FEASIBILITY_VLA_TRANSFER']!='PASS':
        raise RuntimeError('Scientific gate did not pass')
    stage('runtime_freeze',[HERE/'prepare_final_online_vla.py','--qualification',current,
        '--admission',admission,'--fresh-plan',HERE/'FINAL_FRESH_ROOT_PLAN.json','--out',final,
        '--server-ready',HERE/'server_v3_frozen/SERVER_READY.json'])
    # The frozen main driver validates its own immutable source and every gate
    # before the first fresh-root reference or branch can start.
    stage('final192_main',[final/'SOURCE_SNAPSHOT/run_final_online_vla.py','--out',final])
    if read(final/'FINAL_MAIN_EXECUTION_COMPLETE.json')['branches']!=192:
        raise RuntimeError('Final main ended without complete192 evidence')
    write(control/'MAIN_EXECUTION_COMPLETE_REVIEW_REMAINING.json',dict(branches=192,
        requires_independent_tables=True,requires_online_dev_ablations=True,requires_paper_package=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True)
    p.add_argument('--current',type=Path,required=True);p.add_argument('--coordinator-pid',type=int,required=True)
    p.add_argument('--final',type=Path,required=True);a=p.parse_args()
    try:run(a.control,a.current,a.coordinator_pid,a.final)
    except Exception as e:
        if a.control.exists() and not (a.control/'AUTOMATION_FAILED.json').exists():
            write(a.control/'AUTOMATION_FAILED.json',dict(error=repr(e),no_automatic_retry=True))
        raise
