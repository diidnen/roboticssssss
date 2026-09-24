"""Outcome-blind confirmatory coordinator using the already frozen worker/runtime.
Method order is read only from the precomputed plan; no outcome can alter it.
"""
import argparse, csv, json
from pathlib import Path
from datetime import datetime, timezone
from common import read, write, sha
from run_final_online_vla import validate, launch
from audit_rollout import audit
from audit_geometric_label import audit as audit_geometry

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); out=a.out
    manifest, contexts=validate(out)
    plan_rows=list(csv.DictReader((out/'CONFIRMATORY_EXECUTION_PLAN.csv').open()))
    by={c['context_id']:[] for c in plan_rows}
    for r in plan_rows: by[r['context_id']].append(r)
    if len(plan_rows)!=192 or any(len(v)!=4 for v in by.values()): raise RuntimeError('Invalid frozen execution plan')
    write(out/'CONFIRMATORY_QUEUE_STARTED.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'planned_branches':192,'methods':['FIXED_3','FIXED_4','FIXED_5','ACTIVEFORCING'],'method_order_source':str(out/'CONFIRMATORY_EXECUTION_PLAN.csv'),'method_order_sha256':sha(out/'CONFIRMATORY_EXECUTION_PLAN.csv'),'no_outcome_retry':True})
    rows=[]
    for i,ctx in enumerate(contexts):
        cid=ctx['id']; launch(out,i,'REFERENCE')
        ordered=sorted(by[cid],key=lambda r:int(r['method_order']))
        for er in ordered:
            meth=er['method']; job=launch(out,i,meth)
            proof=audit(job); geom=audit_geometry(job)
            if not proof['passed'] or not geom['passed']:
                write(job/'FINAL_EVIDENCE_QUARANTINE.json',{'provenance':proof,'geometry':geom,'execution_index':er['execution_index']})
                raise RuntimeError('Invalid branch quarantined; confirmatory queue stopped: '+str(job))
            write(job/'FINAL_EVIDENCE_ADMISSION.json',{'provenance':proof,'geometry':geom,'final_manifest_sha256':sha(out/'FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json'),'execution_index':er['execution_index'],'method_order':er['method_order']})
            rows.append({'execution_index':int(er['execution_index']),'context':cid,'method':meth,'method_order':int(er['method_order']),'job':str(job),'result_sha256':sha(job/'BRANCH_RESULT.json'),'success':proof['outcome']['full_task_success_y']})
        print('CONFIRMATORY_CONTEXT_DONE',cid,flush=True)
    if len(rows)!=192: raise RuntimeError('Incomplete confirmatory table')
    write(out/'CONFIRMATORY_MAIN_EXECUTION_COMPLETE.json',{'rows':rows,'branches':192,'contexts':48,'roots':sorted({c['root'] for c in contexts}),'source':'ONLINE_VLA','method_order_sha256':sha(out/'CONFIRMATORY_EXECUTION_PLAN.csv'),'tables_and_paper_validation_still_required':True})
if __name__=='__main__': main()
