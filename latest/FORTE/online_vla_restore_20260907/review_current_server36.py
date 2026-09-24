"""Qualify complete current-server evidence with the original transfer gates.

Old/new outcome agreement is reported, never used to drop a failed branch.
No admission is written before all 36 paired cells have completed.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from common import HERE, read, write, sha
from review_qualification import review

BOUNDARIES=('t0_r5100_low','t1_r5100_mid','t5_r5100_low','t6_r5100_low')
RUNTIME=('worker.py','runtime.py','arbitration.py','phase_free_feasibility.py','common.py')

def transfer_gates(report):
    rows=report['probabilities']['rows']; metrics=report['probabilities']['main_probability']
    y=np.asarray([r['y'] for r in rows]);p=np.asarray([r['p'] for r in rows])
    ece5=0.
    for i in range(5):
        mask=(p>=i/5)&(p<(i+1)/5 if i<4 else p<=1)
        if mask.any():ece5+=float(mask.mean()*abs(p[mask].mean()-y[mask].mean()))
    af=[r for r in rows if r['method']=='ACTIVEFORCING']
    f5=[r for r in rows if r['method']=='FIXED_5']
    boundary={cid:report['contexts'].get(cid,{}).get('low_force_failure_higher_force_success',False)
              for cid in BOUNDARIES}
    gates=dict(primary_complete=len(rows)==36 and len(af)==12 and len(f5)==12,
        both_outcomes_observed=len(set(y))==2,
        auroc_at_least_0_70=metrics['AUROC'] is not None and metrics['AUROC']>=.70,
        brier_at_most_0_22=metrics['Brier']<=.22,ece5_at_most_0_20=ece5<=.20,
        af_within_two_successes_of_fixed5=sum(r['y'] for r in af)>=sum(r['y'] for r in f5)-2,
        af_not_constant_force=len({r['force'] for r in af})>1,
        force_boundary_all_four_tasks=all(boundary.values()))
    return dict(gates=gates,passed=all(gates.values()),ece5=ece5,
        boundaries=boundary,metrics=metrics,
        original_thresholds_reused_without_relaxation=True)

def qualify(current, old_admission_path, ready_path, parity_path, out):
    complete=read(current/'CURRENT_SERVER36_EXECUTION_COMPLETE.json')
    if complete['branches']!=36 or complete['new_branches']!=22 or complete['reused_branches']!=14:
        raise RuntimeError('Full second qualification is incomplete')
    old=read(old_admission_path);ready=read(ready_path)
    protocol=read(current/'COMPLETION_PROTOCOL.json')
    if protocol['combined_primary_max']!=72 or protocol['no_outcome_retry'] is not True:
        raise RuntimeError('Unexpected qualification scope')
    for path,digest in old['evidence_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Historical evidence changed')
    for path,digest in read(current/'CANDIDATE_RUNTIME_MANIFEST.json')['source_hashes'].items():
        if sha(path)!=digest:raise RuntimeError('Current qualification source changed')
    for name in RUNTIME:
        if sha(current/'SOURCE_SNAPSHOT'/name)!=old['qualified_runtime_sha256'][name]:
            raise RuntimeError('Model or physical runtime changed')
    prior=Path(old['qualification_directory'])
    plans=read(current/'DEV_PLAN.json')['contexts']
    expected={(p['id'],m) for p in plans for m in ('ACTIVEFORCING','FIXED_3','FIXED_5')}
    report=review(current)
    rows=report['rows']
    if len(rows)!=36 or {(r['context'],r['method']) for r in rows}!=expected:
        raise RuntimeError('Incomplete or duplicated current coverage')
    differences=[];evidence=dict(old['evidence_sha256'])
    for r in rows:
        job=Path(r['job']);cid=r['context']
        if read(job/'VLA_SERVER_METADATA.json')!=ready:
            raise RuntimeError('Mixed server instances')
        before=read(prior/'branches'/job.name/'BRANCH_RESULT.json')['outcome']
        now=r['outcome']
        if before['full_task_success_y']!=now['full_task_success_y']:
            differences.append(dict(context=cid,method=r['method'],old_outcome=before,new_outcome=now))
        for name in ('PROCESS_EXIT.json','BRANCH_RESULT.json','BRANCH_TRACE.json','PLANNER_DECISION.json','VLA_SERVER_METADATA.json'):
            evidence[str(job/name)]=sha(job/name)
    for plan in plans:
        ref=current/'references'/plan['id'];original=prior/'references'/plan['id']
        if (ref/'RAW_PROBE.csv').read_bytes()!=(original/'RAW_PROBE.csv').read_bytes():
            raise RuntimeError('Physical probe mismatch')
        for name in ('CONTACT_PATCH_READBACK.json','PREACTION_POSTERIOR.json'):
            if read(ref/name)!=read(original/name):raise RuntimeError('Probe/posterior mismatch')
    transfer=transfer_gates(report)
    behavior=(report['early_regrasp_candidate_branch_rate']==0 and
              report['mean_per_branch_action_anomaly_rate']==0)
    parity=read(parity_path)
    # Keep the failed strict restart-parity review as evidence, not an admission.
    if parity['reviewed_branches']!=14 or parity['numerical_diagnostic_requests']!=1400:
        raise RuntimeError('Missing restart diagnostic coverage')
    for path,digest in parity['evidence_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Restart diagnostic changed')
    evidence.update(parity['evidence_sha256'])
    for path in (old_admission_path,ready_path,parity_path,current/'COMPLETION_PROTOCOL.json',
                 current/'CURRENT_SERVER36_EXECUTION_COMPLETE.json',Path(__file__),
                 HERE/'review_qualification.py',HERE/'qualification_metrics.py'):
        evidence[str(path)]=sha(path)
    passed=transfer['passed'] and behavior and report['all_provenance_and_geometry_valid'] and report['all_common_random_numbers_and_initial_X_valid']
    result=dict(role='COMPLETE_CURRENT_SERVER36_QUALIFICATION_REVIEW',passed=passed,
        reviewed_branches=36,combined_primary_branches=72,source_qualification=str(current),
        original_qualification=str(prior),server_ready=str(ready_path),server_ready_sha256=sha(ready_path),
        checkpoint_sha256=ready['checkpoint_sha256'],same_current_server_instance=True,
        VLA_SEED_MATCHING_VALID=report['all_common_random_numbers_and_initial_X_valid'],
        VLA_POSTPROBE_BEHAVIOR_VALID=behavior,all_four_force_boundaries_valid=all(transfer['boundaries'].values()),
        numerical_diagnostic_requests=1400,transfer=transfer,full_review=report,
        outcome_changes=differences,all_original_outcomes_reproduced=not differences,
        no_outcome_retry=True,no_parameters_changed=True,final_roots_used=False,
        evidence_sha256=evidence,interpretation='Complete repeated-context development qualification; current outcomes retained even when changed. Same original engineering gates; no new independent root or final SR claim.')
    write(out,result)
    print('CURRENT_SERVER36_QUALIFICATION',passed,'OUTCOME_CHANGES',len(differences),flush=True)
    if not passed:raise RuntimeError('Current-server transfer/behavior gate failed; no final admission')
    admission=dict(old)
    admission.update(qualification_directory=str(current),completed_primary_branches=36,
        completed_boundary_fixed4=0,historical_boundary_fixed4=4,combined_primary_branches=72,
        server_requalification=str(out),real_online_requests=report['real_online_requests'],
        created_utc=datetime.now(timezone.utc).isoformat(),EXISTING_FEASIBILITY_VLA_TRANSFER='PASS',
        evidence_sha256={**evidence,str(out):sha(out)},
        acceptance_reasoning=[
            'All36 current-server branches pass joined online inference, arm/force arbitration, geometric-label, same-X and native-noise audits.',
            'Same original complete36 engineering transfer gates pass; all four prespecified paired force boundaries remain.',
            'Postprobe physical references match prior sensor-diagnosed references. No early repeated-grasp or >10cm/step arm anomaly in current36.',
            'Old and current outcomes are both retained; complete current-server qualification replaces failed strict cross-process outcome-parity admission.',
            'Exactly the same pretrained VLA, phase-free feasibility weights, 64 causal inputs and gripper rules. No tuning/retraining or new VLA training rows.'
        ],current_server_outcome_changes=differences)
    write(out.with_name('FINAL_VLA_QUALIFICATION_ADMISSION_V2.json'),admission)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--current',type=Path,required=True)
    p.add_argument('--old-admission',type=Path,required=True);p.add_argument('--server-ready',type=Path,required=True)
    p.add_argument('--parity-review',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();qualify(a.current,a.old_admission,a.server_ready,a.parity_review,a.out)
