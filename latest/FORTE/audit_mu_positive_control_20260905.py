"""Oracle mu is a diagnostic positive control ONLY, never a runtime fix."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import e1_verified_inference as rt
import activeforcing_feasibility_features as feat
import activeforcing_e2e_task0_smoke_20260905 as smoke

ROOT=rt.ROOT
OUT=ROOT/'analysis/results/physical_belief_transfer_root_cause_20260905'

def main():
    torch.set_num_threads(2)
    full=ROOT/'analysis/results/final_fulltask_posterior_feasibility_20260905'
    ens=[]
    for i in range(3):
        ck=torch.load(full/f'FINAL_POSTERIOR_FULLTASK_FEAS_seed{i}.pt',map_location='cpu',weights_only=False)
        m=rt.DirectNet();m.load_state_dict(ck['state_dict']);m.eval()
        ens.append((m,np.asarray(ck['normalization_mean'],np.float32),np.asarray(ck['normalization_std'],np.float32)))
    fs=np.round(np.arange(3,5.0001,.01),2);results=[];curves=[];features=[]
    train=np.load(full/'TRAINING_FEATURES.npz')['x']
    labels=pd.read_csv(ROOT/'analysis/results/final_probe_continuous_posterior_rebuild_20260904/FULL_TASK_LABEL_DATASET.csv')
    lo=train[labels.task==0].min((0,1));hi=train[labels.task==0].max((0,1))
    for band,reference in [('high',.940189),('mid',.450580),('low',.293710)]:
        cid=f'p5s0c_train_t0_r00_s5100_{band}_mu{reference:.6f}'
        probe=smoke.OUT/'PROBE_TELEMETRY'/f'{cid}_repeat1.csv'
        command=sorted((smoke.VALIDATION/'TASK0_TRACES').glob(f'{cid}_*.csv'))[0]
        current=smoke.posterior_from_probe(probe)
        actual,_=feat.from_saved_probe(cid,probe,command,0,3,current['mean'])
        e1=rt.e1_nominal_from_saved(probe,command,0,3,current['mean'])
        for j in range(19,71):
            features.append({'context':cid,'index':j,'current':float(actual[0,j]),'task0_training_min':float(lo[j]),'task0_training_max':float(hi[j]),
                'outside_range':bool(actual[0,j]<lo[j]-1e-6 or actual[0,j]>hi[j]+1e-6)})
        for model in ['E1_OOF','CURRENT_LIFTHOLD','EXISTING_FULLTASK']:
            template=e1 if model=='E1_OOF' else actual
            for mode,mus in [('CURRENT_E6_SUPPORT',current['member_means']),('ORACLE_ANALYSIS_ONLY',[reference]),('SHARED_MU_DIAGNOSTIC',[.5])]:
                x=np.repeat(template[None],len(fs)*len(mus),axis=0)
                x[:,:,17]=np.repeat(fs/8,len(mus))[:,None];x[:,:,18]=np.tile(mus,len(fs))[:,None]
                if model=='E1_OOF':p=rt.probabilities(x,0)
                elif model=='CURRENT_LIFTHOLD':p=rt.probabilities(x,0,'clean')
                else:
                    ps=[]
                    for m,mean,std in ens:
                        with torch.no_grad():ps.append(torch.sigmoid(m(torch.tensor((x-mean)/std))).numpy())
                    p=np.mean(ps,axis=0)
                p=p.reshape(len(fs),len(mus)).mean(1)
                results.append({'context':cid,'model':model,'mu_source':mode,'input_contract':'E1 strict preprobe' if model=='E1_OOF' else 'corrected current post-probe',
                    'not_deployable':True,'p3':float(p[0]),'p5':float(p[-1]),**rt.select(fs,p,5)})
                curves.extend({'context':cid,'model':model,'mu_source':mode,'setpoint':float(f),'p_success':float(v),'utility':float(v*(2-f/5)-1)} for f,v in zip(fs,p))
    pd.DataFrame(results).to_csv(OUT/'MU_POSITIVE_CONTROL.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'MU_POSITIVE_CONTROL_CURVES.csv',index=False)
    pd.DataFrame(features).to_csv(OUT/'CORRECTED_CURRENT_STATE_SUPPORT.csv',index=False)
    print(pd.DataFrame(results)[['context','model','mu_source','selected_setpoint','p3','p5']].to_string(index=False))

if __name__=='__main__':main()
