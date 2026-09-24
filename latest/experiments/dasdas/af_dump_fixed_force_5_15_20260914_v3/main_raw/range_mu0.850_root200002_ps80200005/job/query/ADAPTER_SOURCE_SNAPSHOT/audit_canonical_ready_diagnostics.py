"""Check the completed two-run initialization intervention, without admission."""
import json
import numpy as np
from rootlocal_collection_contract import HERE, read, write, sha
from analyze_mu060_highrate import analyze


def main():
    out=HERE/'mu060_canonical_ready_diagnostics_v1'
    if not read(out/'COMPLETE.json')['completed']:raise ValueError('Not terminal')
    cases=[out/f'repeat_{i:02d}' for i in (1,2)]
    for case in cases:
        if read(case/'EXIT.json')['exit_code']!=0:raise ValueError('Incomplete diagnostic process')
        if not (case/'HIGH_RATE_ANALYSIS.json').exists():analyze(case)
    query=[case/'query' for case in cases]
    init=[read(q/'CANONICAL_OPEN_READY_INITIALIZATION.json') for q in query]
    controls=[read(q/'native_controls.json') for q in query]
    comparisons={}
    for key in ['actual_qpos','actual_qvel','actual_ready_world_pose','target_qpos']:
        comparisons[key+'_exact']=init[0][key]==init[1][key]
        comparisons[key+'_max_abs_difference']=float(np.max(abs(np.asarray(init[0][key])-np.asarray(init[1][key]))))
    for key in ['original_action13','joint_targets','actual_eef_base']:
        a,b=[np.asarray([c[key] for c in run]) for run in controls]
        comparisons[key+'_shapes']=[list(a.shape),list(b.shape)]
        comparisons[key+'_exact']=bool(np.array_equal(a,b))
        comparisons[key+'_first_exact']=bool(np.array_equal(a[0],b[0]))
        comparisons[key+'_max_abs_difference']=float(np.max(abs(a-b))) if a.shape==b.shape else None
    summaries=[read(q/'original_probe_summary.json') for q in query]
    comparisons['both_probe_qualified']=all(s['probe_failure']==0 for s in summaries)
    comparisons['probe_failures']=[s['probe_failure'] for s in summaries]
    comparisons['both_raw_observation_sequences_identical']=sha(query[0]/'original_raw_rows.json')==sha(query[1]/'original_raw_rows.json')
    comparisons['both_full_highrate_traces_identical']=read(query[0]/'DIAGNOSTIC_TRACE_RECEIPT.json')['trace_rows_sha256']==read(query[1]/'DIAGNOSTIC_TRACE_RECEIPT.json')['trace_rows_sha256']
    report={'comparisons':comparisons,'diagnostic_only':True,'formal_admission':False,
            'causality_of_original_failure_not_established_by_two_runs':True,'source_sha256':sha(__file__),
            'highrate_audit_hashes':[sha(case/'HIGH_RATE_ANALYSIS.json') for case in cases]}
    write(out/'INITIALIZATION_COMPARISON.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
