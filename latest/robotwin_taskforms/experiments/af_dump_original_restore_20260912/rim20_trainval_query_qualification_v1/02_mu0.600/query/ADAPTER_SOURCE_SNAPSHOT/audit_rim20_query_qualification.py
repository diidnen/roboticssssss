"""Apply the original complete evidence admission, not only process exit status."""
import json
from rootlocal_collection_contract import HERE,read,write,sha,verify_runtime
from admit_original_query import admit


def main():
    out=HERE/'rim20_trainval_query_qualification_v1'
    protocol=read(out/'PROTOCOL.json')
    for source,digest in protocol['source_hashes'].items():
        if sha(source)!=digest:raise ValueError('Qualification source drift')
    verify_runtime(HERE/'original_rootlocal_dataset_v2_canonical/RUNTIME_MANIFEST.json',protocol['frozen_runtime_sha256'])
    completed=[]
    for index,mu in enumerate(protocol['frictions'],1):
        case=out/f'{index:02d}_mu{mu:.3f}'
        # The receipt is written after closure, independent force audit and pack.
        receipt=HERE.parent/('af_dump_rim20_qualification_'+case.name+'_20260913.tar.receipt.json')
        if not receipt.exists():continue
        if read(case/'EXIT.json')['exit_code']:raise ValueError('Retained process failure '+case.name)
        if read(case/'query/original_probe_summary.json')['probe_failure']:
            raise ValueError('Retained original probe failure '+case.name)
        report_path=case/'ORIGINAL_FULL_ADMISSION.json'
        admission=admit((case/'query').resolve())
        write(report_path,admission)
        highrate=read(case/'HIGH_RATE_ANALYSIS.json')
        completed.append({'case':case.name,'friction':mu,'original_admitted':admission['admitted'],
            'admission_sha256':sha(report_path),'archive_receipt_sha256':sha(receipt),
            'highrate_min_bilateral_N':highrate['min_probe_force_N'],
            'highrate_below_0p2_count':len(highrate['highrate_below_0p2_steps']),
            'highrate_diagnostics_do_not_replace_original_admission':True})
    print(json.dumps({'completed_original_admissions':completed},indent=2),flush=True)
    if (out/'COMPLETE.json').exists():
        if len(completed)!=17:raise ValueError('Missing original admissions')
        replay=read(out/'REPLAY_AUDIT.json')
        if not replay['raw_rows_byte_equal'] or replay['mismatched_steps']:raise ValueError('Replay mismatch')
        write(out/'INDEPENDENT_FULL_QUALIFICATION.json',{'all_original_admissions_passed':True,
            'cases':completed,'formal_labels_added':0,'TEST_used':False,
            'replay_audit_sha256':sha(out/'REPLAY_AUDIT.json'),'source_sha256':sha(__file__)})


if __name__=='__main__':main()
