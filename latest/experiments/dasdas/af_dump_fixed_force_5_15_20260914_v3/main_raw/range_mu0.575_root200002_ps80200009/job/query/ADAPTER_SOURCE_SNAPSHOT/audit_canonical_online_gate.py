"""Independent raw audit of the three-branch preparation engineering gate."""
from pathlib import Path
from rootlocal_collection_contract import HERE, read, sha, write
from audit_original_collected_group import audit_query, audit_branch


def main():
    out=HERE/'canonical_ready_online_gate_v1'
    if not read(out/'COMPLETE.json')['completed']:raise ValueError('Gate not terminal')
    job=(out/'job').resolve()
    query=audit_query(job/'query')
    summary=read(job/'online_qualification.json')
    forces=[3.,3.,8.]
    reports=[]
    for index,force in enumerate(forces):
        branch=job/f'branch_{index}_{force:g}N'
        report=audit_branch(branch)
        result=read(branch/'result.json')
        if result['force_setpoint_bilateral_n']!=force:raise ValueError('Wrong engineering force')
        reports.append({'branch':str(branch),'success':result['success'],
                        'audit_sha256':sha(branch/'INDEPENDENT_BRANCH_AUDIT.json'),
                        'raw_trace_hash':report['raw_trace_hash_recomputed']})
    if reports[0]['raw_trace_hash']!=reports[1]['raw_trace_hash']:raise ValueError('Duplicate raw replay differs')
    if len(summary['first_chunk_hashes'])!=3 or len(set(summary['first_chunk_hashes']))!=1:
        raise ValueError('Unpaired first policy chunk')
    if len({sha(Path(r['branch'])/'original_motion_feature.json') for r in reports})!=1:
        raise ValueError('Unpaired motion descriptor')
    report={'passed':True,'formal_labels_added':0,'query_rows':query['query_rows'],
            'full_duplicate_trace_exact':True,'all_three_independently_audited':True,
            'branches':reports,'source_sha256':sha(__file__),
            'query_audit_sha256':sha(job/'query/INDEPENDENT_NATIVE_FORCE_REBUILD.json')}
    write(out/'INDEPENDENT_ENGINEERING_AUDIT.json',report)
    print(report,flush=True)


if __name__=='__main__':main()
