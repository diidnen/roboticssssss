"""Evaluate existing clean checkpoints with the repaired CURRENT feature schema."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import e1_verified_inference as rt
import activeforcing_feasibility_features as feat
import activeforcing_e2e_task0_smoke_20260905 as smoke

ROOT=rt.ROOT
OUT=ROOT/'analysis/results/e1_verified_recovery_revision_20260905'

def main():
    torch.set_num_threads(2)
    fs=np.round(np.arange(3,5.0001,.01),2)
    full=ROOT/'analysis/results/final_fulltask_posterior_feasibility_20260905'
    families={'CURRENT_LIFTHOLD':[smoke.FEAS_DIR/f'POSTERIOR_FEASIBILITY_seed{s}.pt' for s in range(3)],
              'EXISTING_FULLTASK':[full/f'FINAL_POSTERIOR_FULLTASK_FEAS_seed{s}.pt' for s in range(3)]}
    models={}
    for family,paths in families.items():
        models[family]=[]
        for path in paths:
            ck=torch.load(path,map_location='cpu',weights_only=False)
            m=rt.DirectNet();m.load_state_dict(ck['state_dict']);m.eval()
            models[family].append((m,np.asarray(ck['normalization_mean'],np.float32),np.asarray(ck['normalization_std'],np.float32)))
    results=[];curves=[]
    for band,mu in [('high','0.940189'),('mid','0.450580'),('low','0.293710')]:
        cid=f'p5s0c_train_t0_r00_s5100_{band}_mu{mu}'
        command=sorted((smoke.VALIDATION/'TASK0_TRACES').glob(f'{cid}_*.csv'))[0]
        for repeat in [1,2]:
            probe=smoke.OUT/'PROBE_TELEMETRY'/f'{cid}_repeat{repeat}.csv'
            posterior=smoke.posterior_from_probe(probe)
            mus=posterior['member_means']
            base,provenance=feat.from_saved_probe(cid,probe,command,0,3,mus[0])
            x=np.repeat(base[None],len(fs)*len(mus),axis=0)
            x[:,:,17]=np.repeat(fs/8,len(mus))[:,None];x[:,:,18]=np.tile(mus,len(fs))[:,None]
            for family,ensemble in models.items():
                outputs=[]
                for model,mean,std in ensemble:
                    with torch.no_grad():outputs.append(torch.sigmoid(model(torch.tensor((x-mean)/np.maximum(std,1e-6)))).numpy())
                p=np.mean(outputs,axis=0).reshape(len(fs),len(mus)).mean(1)
                results.append({'context':cid,'repeat':repeat,'model':family,'mu_mean':posterior['mean'],'mu_std_sample':posterior['std'],
                                'p_at_3':float(p[0]),'p_at_5':float(p[-1]),'delta_p':float(p[-1]-p[0]),
                                'input_provenance':provenance,**rt.select(fs,p,5)})
                curves.extend({'context':cid,'repeat':repeat,'model':family,'setpoint':float(f),'p':float(v),'utility':float(v*(2-f/5)-1)} for f,v in zip(fs,p))
    pd.DataFrame(results).to_csv(OUT/'CURRENT_CORRECTED_CLEAN_MODEL_COMPARISON.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'CURRENT_CORRECTED_CLEAN_MODEL_CURVES.csv',index=False)
    manifest=json.loads((full/'FINAL_FEASIBILITY_RUNTIME_MANIFEST.json').read_text())
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for paths in families.values() for p in paths}
    for entry in manifest['checkpoints']:assert hashes[entry['path']]==entry['sha256']
    (OUT/'EXISTING_CLEAN_MODEL_AUDIT.json').write_text(json.dumps({'checkpoint_hashes':hashes,
        'fulltask_saved_manifest_status':manifest['status'],'saved_manifest_runtime_authorized':manifest['runtime_use_authorized'],
        'current_corrected_all_selected_lower_bound':all(r['selected_setpoint']==3 for r in results),
        'no_training_performed':True,'no_physical_run_performed':True,
        'causal_scope':'Actual current feature contract + current E6 posterior. Separate from E1-schema controlled weights ablation.',
        'retrain_required':'NOT_ESTABLISHED; extant clean fulltask does not by itself recover adaptation'},indent=2)+'\n')
    print(pd.DataFrame(results).drop(columns=['input_provenance']).to_string(index=False))

if __name__=='__main__':main()
