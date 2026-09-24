"""Post-lock diagnostics. No fitting, calibration, or simulator launch."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from current_contract_physical_belief import FrictionMember,CurrentPhysicalBelief,predict,verify_decision_prefix
from current_contract_belief_features import ProbeEvidence
from train_current_contract_belief_20260905 import OUT,ROOT,distribution_metrics,write,csvout,sha


def main():
    torch.set_num_threads(2)
    if not (OUT/'CHECKPOINT_SELECTION_LOCK.json').exists():raise RuntimeError('No checkpoint lock')
    runtime=CurrentPhysicalBelief(OUT/'PHYSICAL_BELIEF_CANDIDATE_MANIFEST.json',diagnostic_only=True)
    protocol=json.loads((OUT/'PROTOCOL.json').read_text());comparison=[];perstep=[];parity=[]
    evidence=ProbeEvidence();old_models=[]
    old_dir=ROOT/'activeforcing_full_claim_closure_20260902_062809/E6_E7/E6_E7_LOCKED_HANDOFF'
    for i in range(3):
        model=FrictionMember(input_dim=46)
        ck=torch.load(old_dir/f'PHYSICAL_BELIEF_member_{i}.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['state_dict']);model.eval();old_models.append(model)
    cohorts={s:[] for s in ['TRAIN','VAL','TEST','DIAGNOSTIC']}
    for plan in protocol['contexts']:
        job=Path(plan['data_path']);record=json.loads((job/'RESULT.json').read_text())
        if not record.get('probe_qualified',True):continue
        with (job/'RAW_PROBE.csv').open() as stream:raw=list(csv.DictReader(stream))
        patches=json.loads((job/'CONTACT_PATCH_READBACK.json').read_text())
        verify_decision_prefix(raw)
        x=evidence.rows(raw,patches);posterior=runtime.rows(raw,patches)
        y=plan['hidden_friction_analysis_only'];new_mu=np.array(posterior['member_means']);new_sd=np.exp(posterior['member_log_sigma'])
        legacy_input=evidence.legacy.normalize_dynamic(x[:,:46]);old_mu=[];old_sd=[]
        for model in old_models:
            m,s=predict(model,[legacy_input]);old_mu.append(float(m[0]));old_sd.append(float(s[0]))
        comparison.append({'context_id':plan['context_id'],'root_seed':plan['root_seed'],'split':plan['split'],
            'target_mu':y,'old_mean':float(np.mean(old_mu)),'new_mean':posterior['mean'],
            'old_abs_error':abs(float(np.mean(old_mu))-y),'new_abs_error':abs(posterior['mean']-y),
            'old_support':json.dumps(old_mu),'new_support':json.dumps(new_mu.tolist()),
            'probe_out_steps':record['outward_steps']})
        cohorts[plan['split']].append((y,new_mu,new_sd,old_mu,old_sd))
        normalized=torch.tensor((x-runtime.mean)/runtime.std)[None]
        mus=[];sigmas=[]
        for model in runtime.models:
            with torch.no_grad():
                h,_=model.gru(model.projection(normalized))
                mus.append(model.mu_head(h)[0,:,0].numpy())
                sigmas.append(model.log_sigma_head(h)[0,:,0].clamp(-5,1.5).exp().numpy())
        mus=np.array(mus).T;sigmas=np.array(sigmas).T
        max_error=0.
        for end in [1,80,150,190,len(raw)]:
            prefix=runtime.rows(raw[:end],patches[:end])
            max_error=max(max_error,float(np.max(abs(mus[end-1]-prefix['member_means']))))
        assert max_error<3e-6
        parity.append({'context_id':plan['context_id'],'prefix_model_max_error':max_error,
            'evidence_rows_exactly_causal':all(np.array_equal(x[:n],evidence.rows(raw[:n],patches[:n])) for n in [1,80,190,len(raw)]),
            'feature_rows':len(raw),'candidate_actions':0})
        if plan['split']=='DIAGNOSTIC':
            for i,r in enumerate(raw):perstep.append({'context':plan['friction_band'],'step':i+1,'phase':r['probe_phase'],
                'posterior_mean':float(mus[i].mean()),'support':json.dumps(mus[i].tolist()),
                'member_sigma':json.dumps(sigmas[i].tolist()),'diagnostic_only':True})
    metrics={}
    for split,values in cohorts.items():
        if not values:continue
        y=np.array([v[0] for v in values])
        metrics[split]={model:distribution_metrics(y,np.array([v[a] for v in values]),np.array([v[b] for v in values]))
            for model,a,b in [('NEW_CURRENT_CONTRACT',1,2),('OLD_E6',3,4)]}
    csvout('OLD_VS_NEW_POSTERIOR.csv',comparison)
    csvout('TASK0_DIAGNOSTIC_PER_STEP_POSTERIOR.csv',perstep)
    csvout('PREFIX_AND_RUNTIME_PARITY.csv',parity)
    write('OLD_VS_NEW_METRICS.json',metrics)
    write('OLD_BELIEF_PROVENANCE.json',{'checkpoints':{str(p):sha(p) for p in old_dir.glob('PHYSICAL_BELIEF_member_*.pt')},
        'comparison_uses_same_new_probe_observations':True,'old_model_uses_legacy46_current_model_uses58':True,
        'training_or_model_selection_performed_by_evaluation':False})
    print(json.dumps({s:{m:{k:v[k] for k in ['MAE','RMSE','GAUSSIAN_MIXTURE_NLL']} for m,v in mm.items()} for s,mm in metrics.items() if s in ['VAL','TEST','DIAGNOSTIC']},indent=2))


if __name__=='__main__':main()
