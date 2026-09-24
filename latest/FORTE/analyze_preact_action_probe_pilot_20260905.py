"""CPU-only incremental posterior and contact-source provenance audit."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
import activeforcing_e2e_task0_smoke_20260905 as smoke

OUT=Path('/home/exouser/FORTE/analysis/results/preaction_decision_contract_20260905')


class BeliefNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=torch.nn.Sequential(torch.nn.Linear(46,16),torch.nn.ReLU())
        self.gru=torch.nn.GRU(16,16,batch_first=True)
        self.mu_head=torch.nn.Linear(16,1)
        self.log_sigma_head=torch.nn.Linear(16,1)


def main():
    torch.set_num_threads(2)
    adapter=smoke.load_module('decision_audit_evidence',smoke.EVIDENCE_PATH).Evidence46D(smoke.EVIDENCE_NORM)
    models=[]
    for i in range(3):
        m=BeliefNet();m.load_state_dict(torch.load(smoke.BELIEF_DIR/f'PHYSICAL_BELIEF_member_{i}.pt',map_location='cpu',weights_only=False)['state_dict']);m.eval();models.append(m)
    summary=[];detail=[]
    for run in ['normal_vs_friction_readback_gpu','corrected_friction_stop']:
        for band in ['high','mid','low']:
            job=OUT/run/band
            if not (job/'RESULT.json').exists():continue
            with (job/'RAW_PROBE.csv').open() as f:raw=list(csv.DictReader(f))
            readback=json.loads((job/'CONTACT_PATCH_READBACK.json').read_text())
            result=json.loads((job/'RESULT.json').read_text())
            x=torch.tensor(adapter.normalize_dynamic(adapter.rows(raw)))[None]
            mus=[];sigmas=[]
            for m in models:
                with torch.no_grad():
                    h,_=m.gru(m.projection(x));mus.append(m.mu_head(h)[0,:,0].numpy())
                    sigmas.append(m.log_sigma_head(h)[0,:,0].clamp(-5,1.5).exp().numpy())
            mus=np.array(mus).T;sigmas=np.array(sigmas).T
            np.testing.assert_allclose(mus[-1],result['posterior_support'],atol=2e-6,rtol=0)
            # Explicit prefix recomputation proves these diagnostics never look ahead.
            for end in [1,80,150,190,len(raw)]:
                for k,m in enumerate(models):
                    with torch.no_grad():
                        _,h=m.gru(m.projection(x[:,:end]));pred=m.mu_head(h[-1])[0,0].item()
                    if abs(pred-mus[end-1,k])>2e-6:raise AssertionError('Posterior prefix leakage')
            incremental=[]
            normal_projection_error=[]
            for i,(r,rb) in enumerate(zip(raw,readback)):
                normals=np.array([a['sum_normal_vector_world'] for a in rb['patches']['normal']])
                friction=np.array([a['sum_friction_vector_world'] for a in rb['patches']['friction']])
                recorded=np.array([rb['target_object_force']['F_obj_left_world'],rb['target_object_force']['F_obj_right_world']])
                normal_projection_error.append(float(np.max(abs(normals-recorded))))
                d={'run':run,'context':band,'step':i+1,'phase':r['probe_phase'],
                   'legacy_projected_normal_ratio':float(r['ft_over_fn']),
                   'true_friction_left_N':float(np.linalg.norm(friction[0])),
                   'true_friction_right_N':float(np.linalg.norm(friction[1])),
                   'normal_vector_z_sum_N':float(normals[:,2].sum()),
                   'friction_vector_z_sum_N':float(friction[:,2].sum()),
                   'aperture_m':rb['aperture_m'],'posterior_mean':float(mus[i].mean()),
                   'support':json.dumps(mus[i].tolist()),'member_sigmas':json.dumps(sigmas[i].tolist()),
                   'diagnostic_only':True,'candidate_actions_executed':0,
                   'new_stop_ratio':rb.get('termination_signal',{}).get('incremental_directional_friction_ratio')}
                incremental.append(d);detail.append(d)
            smoke.write_csv(job/'PER_STEP_POSTERIOR_AND_CONTACT.csv',incremental)
            reference=Path('/home/exouser/FORTE/analysis/results/probe_only_instrumented_20260905/jobs')/band/'RAW_PROBE.csv'
            raw46=adapter.rows(raw);old46=adapter.csv(reference)
            # Control/readback run must exactly match previously recorded policy inputs.
            if run=='normal_vs_friction_readback_gpu':np.testing.assert_array_equal(raw46,old46)
            summary.append({'run':run,'context':band,'outward_steps':result['outward_steps'],
                'total_steps':result['steps'],'posterior_mean':result['posterior_mean'],
                'posterior_support':result['posterior_support'],
                'stop':result['probe_record']['stop_trigger'],
                'normal_patch_vs_filtered_sensor_max_diff_N':max(normal_projection_error),
                'feature_parity_max_error':result['decision_feature_parity']['MAX_FEATURE_DIFF'],
                'causal_prefix_posterior_parity':True,
                'preprobe_raw46_max_diff':float(np.max(abs(raw46[:190]-old46[:190]))),
                'full_task_branches':0})
    smoke.write_csv(OUT/'PROBE_POSTERIOR_SUMMARY.csv',summary)
    smoke.write_csv(OUT/'PER_STEP_POSTERIOR_AND_CONTACT.csv',detail)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
