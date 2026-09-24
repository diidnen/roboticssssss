"""Locate first cross-process divergence without assuming exact reset replay."""
import json
import numpy as np
from rootlocal_collection_contract import HERE, read, write, sha

queries = [HERE/'original_rootlocal_dataset_v1/groups/train_mu0.600_root200002/attempt_001/job/query'] + [
    HERE/f'mu060_contact_diagnostic_repeats_v1/repeat_{i:02d}/query' for i in (1,2)]


def main():
    data=[]
    for q in queries:
        data.append({'path':str(q),'geometry':read(q/'geometry_binding.json'),
            'rows':read(q/'original_raw_rows.json'),'controls':read(q/'native_controls.json'),
            'reset_contacts':read(q/'reset.json')['contacts']})
    reports=[]
    for left,right in [(0,1),(1,2)]:
        a,b=data[left],data[right]
        report={'left':a['path'],'right':b['path'],'geometry_equal':a['geometry']==b['geometry'],
                'geometry_left':a['geometry'],'geometry_right':b['geometry'],
                'reset_finger_positions_left':a['reset_contacts']['actual_finger_joint_m'],
                'reset_finger_positions_right':b['reset_contacts']['actual_finger_joint_m']}
        for label,key in [('command','original_action13'),('arm_target','joint_targets'),('eef','actual_eef_base')]:
            differences=[]
            for i,(x,y) in enumerate(zip(a['controls'],b['controls']),1):
                delta=float(np.max(abs(np.asarray(x[key])-np.asarray(y[key]))))
                if delta: differences.append({'step':i,'max_abs_delta':delta})
            report[label+'_first_differences']=differences[:5]
        keys=['eef_x','eef_y','eef_z','object_x_priv','object_y_priv','object_z_priv','measured_fn']
        report['first_row_left']={k:a['rows'][0][k] for k in keys}
        report['first_row_right']={k:b['rows'][0][k] for k in keys}
        reports.append(report)
    out=HERE/'mu060_contact_diagnostic_repeats_v1/PREFIX_COMPARISON.json'
    write(out,{'comparisons':reports,'source_sha256':sha(__file__),
               'comparison_is_not_a_whole_memory_replay_test':True})
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
