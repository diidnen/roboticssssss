"""Evidence-bound review of bounded post-restart dev qualification."""
import argparse
from pathlib import Path
from common import HERE,read,write,sha
from audit_rollout import audit
from audit_geometric_label import audit as geometry
from behavior_audit import audit as behavior
from matched_seeds import matched_native_prefixes

def review(boundary,release,numerical,ready_path,out):
    source=read(HERE/'SERVER_COMMON_SOURCE_DRIFT_RESOLUTION.json')
    if not source['all_function_asts_identical'] or not source['all_native_openpi_source_hashes_unchanged']:
        raise RuntimeError('Native policy/source equivalence unresolved')
    ready=read(ready_path)
    for p,d in ready['source_hashes'].items():
        if sha(p)!=d:raise RuntimeError('Frozen new-server source changed')
    n=read(numerical);planner=read(HERE/'SERVER_RESTART_PLANNER_AUDIT.json')
    if n['requests']!=1400 or not n['input_noise_checkpoint_preprocessing_parity']:
        raise RuntimeError('Incomplete native-input numerical diagnostic')
    if not planner['all_selected_forces_equal'] or planner['contexts']!=12:
        raise RuntimeError('Recheck planner numerical variation before admission')
    expected={('t0_r5100_low',m) for m in ('FIXED_3','FIXED_5','ACTIVEFORCING')}
    expected.update((c,m) for c in ('t1_r5100_mid','t5_r5100_low','t6_r5100_low') for m in ('FIXED_3','FIXED_5','ACTIVEFORCING'))
    expected.update(('t5_r5100_mid',m) for m in ('FIXED_3','FIXED_5'))
    rows=[];evidence={};groups={}
    for directory,required in ((boundary,12),(release,2)):
        complete=directory/'RECHECK_EXECUTION_COMPLETE.json';raw=read(complete)
        if raw['branches']!=required or len(raw['rows'])!=required:raise RuntimeError('Incomplete bounded recheck')
        evidence[str(complete)]=sha(complete)
        for r in raw['rows']:
            job=Path(r['job']);p=audit(job);g=geometry(job);b=behavior(job)
            if not p['passed'] or not g['passed']:raise RuntimeError('Invalid online recheck evidence')
            if read(job/'VLA_SERVER_METADATA.json')!=ready:raise RuntimeError('Recheck used a different server')
            if not read(directory/'references'/r['context']/'ORIGINAL_QUALIFICATION_PROBE_PARITY.json')['passed']:
                raise RuntimeError('Recheck changed physical probe reference')
            identity=read(job/'INITIAL_ONLINE_CHUNK_IDENTITY.json')
            seeds=[{k:q[k] for k in ('step','noise_seed','noise_sha256')} for q in
                (read(path) for path in sorted((job/'RPC').glob('*.json')))]
            groups.setdefault(r['context'],[]).append((identity,seeds))
            result=read(job/'BRANCH_RESULT.json');decision=read(job/'PLANNER_DECISION.json')
            old=Path(r['old_job']);old_result=read(old/'BRANCH_RESULT.json')
            same_y=result['outcome']['full_task_success_y']==old_result['outcome']['full_task_success_y']
            same_f=decision['executed_force_N']==read(old/'PLANNER_DECISION.json')['executed_force_N']
            rows.append(dict(context=r['context'],method=r['method'],job=str(job),
                full_success=result['outcome']['full_task_success_y'],selected_force=decision['executed_force_N'],
                same_original_outcome=same_y,same_selected_force=same_f,
                early_regrasp_candidate=b['early_regrasp_candidate_present'],
                regrasp_candidate=b['regrasp_candidate_present'],
                action_anomaly_rate=b['VLA_ACTION_ANOMALY_RATE'],
                measured_drop=g['independent_measured_drop'],real_online_requests=p['real_online_requests']))
            evidence[str(job/'RECHECK_EVIDENCE.json')]=sha(job/'RECHECK_EVIDENCE.json')
            evidence[str(job/'BRANCH_RESULT.json')]=sha(job/'BRANCH_RESULT.json')
    if {(r['context'],r['method']) for r in rows}!=expected or len(rows)!=14:
        raise RuntimeError('Unexpected recheck coverage or duplicates')
    for group in groups.values():
        if not all(x[0]==group[0][0] for x in group) or not matched_native_prefixes([x[1] for x in group]):
            raise RuntimeError('Recheck lost common decisionX / native-noise matching')
    boundaries={}
    for cid in ('t0_r5100_low','t1_r5100_mid','t5_r5100_low','t6_r5100_low'):
        group={r['method']:r for r in rows if r['context']==cid}
        boundaries[cid]=group['FIXED_3']['full_success']==0 and group['FIXED_5']['full_success']==1 and group['ACTIVEFORCING']['full_success']==1
    same=all(r['same_original_outcome'] and r['same_selected_force'] for r in rows)
    continuation=not any(r['early_regrasp_candidate'] or r['action_anomaly_rate'] for r in rows)
    passed=same and all(boundaries.values()) and continuation
    exact_arbitration=read(HERE/'SERVER_RESTART_EXACT_ARBITRATION_DIAGNOSTIC.json')
    if exact_arbitration['positions']!=14000 or not exact_arbitration['old_trace_release_flags_all_reproduced']:
        raise RuntimeError('Incomplete actual arbitration-rule numerical diagnostic')
    for path in (numerical,ready_path,HERE/'SERVER_COMMON_SOURCE_DRIFT_RESOLUTION.json',HERE/'SERVER_RESTART_PLANNER_AUDIT.json',HERE/'SERVER_RESTART_REPEAT_DIAGNOSTIC.json',HERE/'SERVER_RESTART_EXACT_ARBITRATION_DIAGNOSTIC.json'):
        evidence[str(path)]=sha(path)
    result=dict(role='PHYSICAL_RESTART_QUALIFICATION_ADMISSION_REVIEW',passed=passed,
        reviewed_branches=14,real_online_requests=sum(r['real_online_requests'] for r in rows),rows=rows,
        all_original_outcomes_and_forces_reproduced=same,all_four_force_boundaries_valid=all(boundaries.values()),
        VLA_POSTPROBE_BEHAVIOR_VALID=continuation,
        boundaries=boundaries,VLA_SEED_MATCHING_VALID=True,same_current_server_instance=True,
        server_ready_sha256=sha(ready_path),server_ready=str(ready_path),source_hashes=ready['source_hashes'],
        source_qualification=n['source_qualification'],checkpoint_sha256=ready['checkpoint_sha256'],
        numerical_diagnostic_requests=1400,cross_process_byte_exact=False,
        all12_AF_selected_forces_unchanged=True,final_roots_used=False,
        numerical_compilation_cause='Cross-process low-precision/kernel numerical variation is consistent with unchanged model code/weights/preprocessed inputs/explicit noise and stable within-process repeats; specific compiler kernel cause not directly proven.',
        evidence_sha256=evidence,implementation_sha256=sha(__file__),
        interpretation='All54 development branches retained; fourteen restart rechecks are repeated-context engineering evidence, not independent-root SR. No force/policy/feasibility tuning from these outcomes.')
    write(out,result)
    print('RESTART_PHYSICAL_ADMISSION',passed)
    if not passed:raise RuntimeError('Restart physical qualification needs investigation; no final admission')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--boundary',type=Path,required=True);p.add_argument('--release',type=Path,required=True)
    p.add_argument('--numerical',type=Path,required=True);p.add_argument('--server-ready',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();review(a.boundary,a.release,a.numerical,a.server_ready,a.out)
