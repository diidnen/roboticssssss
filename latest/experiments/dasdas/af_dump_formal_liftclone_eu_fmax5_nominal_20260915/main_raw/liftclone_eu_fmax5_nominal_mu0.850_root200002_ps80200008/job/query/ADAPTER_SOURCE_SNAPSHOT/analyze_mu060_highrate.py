"""Compare passive diagnostic traces without changing acquisition or admission."""
import gzip
import hashlib
import json
import numpy as np
from rootlocal_collection_contract import HERE, read, write, sha
from audit_original_collected_group import audit_query


def analyze(case):
    query = case/'query'
    audit_query(query)
    summary=read(query/'original_probe_summary.json')
    receipt=read(query/'DIAGNOSTIC_TRACE_RECEIPT.json')
    low=[]; gate_transitions=[]; forces=[]; observed=[]; previous=None; h=hashlib.sha256(); maxerr=0.
    with gzip.open(query/'DIAGNOSTIC_PHYSICS_TRACE.jsonl.gz','rt',encoding='utf-8') as stream:
        for count,line in enumerate(stream,1):
            h.update(line.rstrip('\n').encode()); row=json.loads(line)
            contact=row['contact']; axis=np.asarray(contact['normal_axis_world'])
            projected=[]
            for finger in contact['fingers']:
                summed=sum(((1 if p['finger_body_index']==0 else -1)*np.asarray(p['impulse_ns'])/.004
                            for p in finger['points']),start=np.zeros(3))
                projected.append(float(summed@axis))
            rebuilt=2*min(map(abs,projected)); f=contact['measured_squeeze_n']
            maxerr=max(maxerr,abs(rebuilt-f))
            if maxerr>1e-8: raise ValueError('Native impulse/force mismatch')
            if row['physics_step']!=count: raise ValueError('Trace index mismatch')
            if row['query_step']>=191:
                short={'physics_step':count,'query_step':row['query_step'],'force_after_N':f,
                    **row['inner_before_step'],'finger_targets':row['finger_drive_targets'],
                    'finger_positions':contact['actual_finger_joint_m'],
                    'arm_max_speed':float(np.max(abs(np.asarray(row['qvel'])[row['arm_indices']])))}
                forces.append(f)
                if f<.2: low.append(short)
                if previous and row['inner_before_step']['effective_reference_N']!=previous['inner_before_step']['effective_reference_N']:
                    gate_transitions.append(short)
                if len(observed)<1:observed.append(row['finger_drive_properties'])
            previous=row
    if count!=receipt['physics_steps'] or h.hexdigest()!=receipt['trace_rows_sha256']:
        raise ValueError('Trace hash/count mismatch')
    rows=read(query/'original_raw_rows.json')
    result={'case':str(case),'probe_failure':summary['probe_failure'],'stop':summary['stop_trigger'],
            'query_rows':len(rows),'highrate_probe_steps':len(forces),'min_probe_force_N':min(forces),
            'highrate_below_0p2_steps':low,'feedforward_gate_transitions':gate_transitions,
            'max_independent_force_error_N':maxerr,'finger_drive_properties':observed,
            'recorded_failed_query_steps':[r['step'] for r in rows if r['probe_phase'].startswith('probe_') and
                                         (r['contact_state']!='bilateral' or r['measured_fn']<.2)],
            'diagnostic_only_no_formal_admission':True,'trace_sha256':sha(query/'DIAGNOSTIC_PHYSICS_TRACE.jsonl.gz')}
    write(case/'HIGH_RATE_ANALYSIS.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    formal=HERE/'original_rootlocal_dataset_v1/groups/train_mu0.600_root200002/attempt_001/job/query'
    audit_query(formal)
    base=HERE/'mu060_contact_diagnostic_repeats_v1'
    for case in sorted(base.glob('repeat_*')):
        if (case/'EXIT.json').exists() and read(case/'EXIT.json')['exit_code']==0 and not (case/'HIGH_RATE_ANALYSIS.json').exists():
            analyze(case)
