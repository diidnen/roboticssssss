"""Independent descriptive audit; never changes qualification or selects data."""
import gzip
import json
import numpy as np
from rootlocal_collection_contract import HERE,read,write,sha,verify_runtime
from analyze_mu060_highrate import analyze


def main():
    out=HERE/'mu070_rim_alignment_diagnostics_v1'
    if not read(out/'COMPLETE.json')['completed']:raise ValueError('Not complete')
    protocol=read(out/'PROTOCOL.json')
    for path,digest in protocol['source_hashes'].items():
        if sha(path)!=digest:raise ValueError('Source drift '+path)
    dataset=HERE/'original_rootlocal_dataset_v2_canonical'
    verify_runtime(dataset/'RUNTIME_MANIFEST.json',protocol['frozen_runtime_sha256'])
    results=[]
    for name,offset in protocol['cases']:
        case=out/name
        if read(case/'EXIT.json')['exit_code']:raise ValueError('Process failed')
        if not (case/'HIGH_RATE_ANALYSIS.json').exists():analyze(case)
        analysis=read(case/'HIGH_RATE_ANALYSIS.json')
        sampled=read(case/'query/original_raw_rows.json')
        probe_steps={row['step'] for row in sampled if row['probe_phase'].startswith('probe_')}
        forces=[]; weak=[];geometry=[]
        with gzip.open(case/'query/DIAGNOSTIC_PHYSICS_TRACE.jsonl.gz','rt') as stream:
            for line in stream:
                row=json.loads(line)
                if row['query_step'] not in probe_steps:continue
                contact=row['contact'];axis=np.asarray(contact['normal_axis_world'])
                normals=[];zs=[]
                for finger in contact['fingers']:
                    force=sum(((1 if p['finger_body_index']==0 else -1)*np.asarray(p['impulse_ns'])/.004
                               for p in finger['points']),start=np.zeros(3))
                    normals.append(abs(float(force@axis)))
                    loaded=[p for p in finger['points'] if p['is_target'] and np.linalg.norm(p['impulse_ns'])>1e-9]
                    weights=[np.linalg.norm(p['impulse_ns']) for p in loaded]
                    zs.append(float(np.average([p['position_world'][2] for p in loaded],weights=weights)) if loaded else None)
                minimum=min(normals);forces.append(minimum)
                if minimum<.15:weak.append({'physics_step':row['physics_step'],'query_step':row['query_step'],'per_pad_normal_N':normals})
                if all(z is not None for z in zs):geometry.append(abs(zs[0]-zs[1]))
        results.append({'case':name,'offset_m':offset,'original_probe_failure':analysis['probe_failure'],
            'recorded_failed_steps':analysis['recorded_failed_query_steps'],
            'probe_physics_steps':len(forces),'minimum_per_pad_normal_N':min(forces),
            'highrate_per_pad_below_original_0p15N':weak,
            'median_loaded_contact_height_asymmetry_m':float(np.median(geometry)) if geometry else None,
            'force_rebuild_max_error_N':analysis['max_independent_force_error_N'],
            'trace_sha256':analysis['trace_sha256']})
    report={'cases':results,'formal_labels_added':0,'original_decision_rules_unchanged':True,
            'highrate_statistics_are_diagnostic_not_new_admission_rules':True,
            'automatic_geometry_adoption':False,'source_sha256':sha(__file__)}
    write(out/'INDEPENDENT_GEOMETRY_AUDIT.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
