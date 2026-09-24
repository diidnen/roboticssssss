"""CPU-only E6 raw-input replay and signal-shift audit; no training functions called."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import e1_verified_inference as rt

ROOT=rt.ROOT
TAB=Path('/home/exouser/Tabero')
OUT=ROOT/'analysis/results/physical_belief_transfer_root_cause_20260905'
E6=ROOT/'activeforcing_full_claim_closure_20260902_062809/E6_E7'
P5=TAB/'analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542'

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,default=str)+'\n')

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    source=rt.module('e6_readonly_source',Path('/home/exouser/FORTE_e6e7/activeforcing_e6e7_closure.py'))
    feature_source=rt.module('p5_readonly_features',TAB/'analysis/p5s0c_model_adjudication.py')
    contexts=pd.read_csv(P5/'P5S0C_CONTEXT_MANIFEST.csv')
    contexts=contexts[contexts.split.isin(['TRAIN','DEV'])].copy()
    seqs,_,lengths,meta=feature_source.load_features(contexts)
    adapter=rt.module('e6_raw_adapter',rt.TRANSFER/'evidence_46d.py').Evidence46D(rt.TRANSFER/'NORMALIZATION_46D.json')
    norm_error=max(float(np.max(abs(np.asarray(meta[k])-np.asarray(adapter.norm[k])))) for k in ['dynamic_mean','dynamic_std'])
    assert meta['dynamic_feature_names']==adapter.feature_names
    examples=[];profiles=[];raw_by_cid={};feature_error=0.
    for r in contexts.itertuples():
        raw=adapter.csv(r.probe_telemetry_path);raw_by_cid[r.context_id]=raw
        parsed=adapter.normalize_dynamic(raw)
        feature_error=max(feature_error,float(np.max(abs(parsed-seqs[r.context_id]))))
        examples.append({'sequence':seqs[r.context_id],'friction':r.hidden_friction_analysis_only})
        df=pd.read_csv(r.probe_telemetry_path)
        profiles.append({'context_id':r.context_id,'task':r.task,'split':r.split,'rows':len(df),
            'outward_steps':int((df.probe_phase=='probe_out').sum()),
            'stop':str(df.stop_trigger.iloc[-1]),'hold_mean_fn':float(df[df.probe_phase=='hold'].measured_fn.mean()),
            'hold_mean_ft':float(df[df.probe_phase=='hold'].measured_ft.mean()),'path':r.probe_telemetry_path})
    models=[];mus=[];sigmas=[];hashes={}
    for i in range(3):
        path=E6/'E6_E7_LOCKED_HANDOFF'/f'PHYSICAL_BELIEF_member_{i}.pt'
        ck=torch.load(path,map_location='cpu',weights_only=False)
        m=source.FrictionMember(ck['input_dim'],ck['projection_dim'],ck['hidden_dim']);m.load_state_dict(ck['state_dict']);m.eval()
        mm,ss=source.predict_member(m,examples);mus.append(mm);sigmas.append(ss);models.append(m)
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    mus=np.asarray(mus).T;sigmas=np.asarray(sigmas).T
    ref=pd.read_csv(E6/'PHYSICAL_BELIEF_PREDICTIONS.csv').set_index('context_id').loc[contexts.context_id]
    mu_error=float(np.max(abs(mus-ref[[f'member_mu_{i}' for i in range(3)]].to_numpy())))
    scale=json.loads((E6/'PHYSICAL_BELIEF_MANIFEST.json').read_text())['interval_scale']
    total=(mus.var(1,ddof=1)+(sigmas**2).mean(1))*scale**2
    var_error=float(np.max(abs(total-ref.total_variance.to_numpy())))
    result=contexts[['context_id','task','split']].copy()
    for i in range(3):result[f'mu_{i}']=mus[:,i];result[f'sigma_{i}']=sigmas[:,i]
    result['mu_mean']=mus.mean(1);result['calibrated_total_sd']=np.sqrt(total)
    result.to_csv(OUT/'E6_HISTORICAL_RAW_REPLAY.csv',index=False)
    pd.DataFrame(profiles).to_csv(OUT/'E6_PROBE_SUPPORT.csv',index=False)
    assert norm_error<1e-5 and feature_error<1e-4 and mu_error<1e-5 and var_error<1e-5
    current=[];shift=[];phase=[]
    train_raw=np.concatenate([raw_by_cid[r.context_id] for r in contexts.itertuples() if r.split=='TRAIN'])
    train_lo=np.min(train_raw,axis=0);train_hi=np.max(train_raw,axis=0)
    normsd=np.asarray(meta['dynamic_std'])
    for band,mu in [('high','0.940189'),('mid','0.450580'),('low','0.293710')]:
        cid=f'p5s0c_train_t0_r00_s5100_{band}_mu{mu}'
        historical_path=Path(contexts[contexts.context_id==cid].iloc[0].probe_telemetry_path)
        for repeat in [1,2]:
            path=ROOT/'analysis/results/activeforcing_e2e_task0_smoke_20260905/PROBE_TELEMETRY'/f'{cid}_repeat{repeat}.csv'
            raw=adapter.csv(path);x=adapter.normalize_dynamic(raw)
            ex=[{'sequence':x,'friction':0}];mm=[];ss=[]
            for model in models:
                a,b=source.predict_member(model,ex);mm.append(float(a[0]));ss.append(float(b[0]))
            epistemic=float(np.var(mm,ddof=1));alea=float(np.mean(np.square(ss)))
            current.append({'context_id':cid,'repeat':repeat,'reference_mu_analysis_only':float(mu),'member_mus':mm,'member_sigmas':ss,'mu_mean':float(np.mean(mm)),
                'smoke_reported_epistemic_sd':float(np.sqrt(epistemic)), 'historical_total_sd_semantics':float(np.sqrt((epistemic+alea)*scale**2)),
                'historical_same_context_mu_mean':float(ref.loc[cid,'ensemble_mean']), 'raw_path':str(path)})
            for j,name in enumerate(meta['dynamic_feature_names']):
                outside=(raw[:,j]<train_lo[j]-1e-6)|(raw[:,j]>train_hi[j]+1e-6)
                shift.append({'context_id':cid,'repeat':repeat,'feature':name,'outside_train_row_range_count':int(outside.sum()),'current_rows':len(raw),
                    'historical_train_min':float(train_lo[j]),'historical_train_max':float(train_hi[j]),'current_min':float(raw[:,j].min()),'current_max':float(raw[:,j].max())})
            for label,pp in [('historical',historical_path),('current',path)]:
                df=pd.read_csv(pp)
                for part,g in df.groupby('probe_phase',sort=False):
                    phase.append({'context_id':cid,'repeat':repeat,'source':label,'phase':part,'rows':len(g),
                        'mean_fn':float(g.measured_fn.mean()),'mean_ft':float(g.measured_ft.mean()),'mean_rho':float(g.ft_over_fn.mean()),
                        'mean_opening':float(g.gripper_opening.mean()),'last_stop':str(g.stop_trigger.iloc[-1])})
    pd.DataFrame(current).to_csv(OUT/'CURRENT_E6_TRANSFER.csv',index=False)
    pd.DataFrame(shift).to_csv(OUT/'CURRENT_RAW_FEATURE_SUPPORT.csv',index=False)
    pd.DataFrame(phase).to_csv(OUT/'E6_HISTORICAL_VS_CURRENT_PHASES.csv',index=False)
    dump('E6_REPLAY_PARITY.json',{'passed':True,'contexts':len(contexts),'members':3,'normalization_max_error':norm_error,'raw_feature_max_error':feature_error,
         'member_mu_max_error':mu_error,'calibrated_total_variance_max_error':var_error,'checkpoint_hashes':hashes,
         'conclusion':'Current E6 checkpoints and adapter reproduce original TRAIN/DEV. Poor current predictions are not explained by missing/double normalization.',
         'uncertainty_semantics_difference':'Smoke std is member spread only; historical total includes sigma heads plus TRAIN-fitted scale. This reporting distinction does not fix current mean discrimination.',
         'training_performed':False,'physics_performed':False})
    print(json.dumps({'norm_error':norm_error,'feature_error':feature_error,'mu_error':mu_error,'var_error':var_error},indent=2))
    print(pd.DataFrame(profiles).groupby(['rows','outward_steps']).size())
    print(pd.DataFrame(current)[['context_id','repeat','mu_mean','historical_same_context_mu_mean','historical_total_sd_semantics']].to_string(index=False))

if __name__=='__main__':main()
