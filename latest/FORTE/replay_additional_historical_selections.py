"""Independent replay of three non-E1 historical decisions; CPU only."""
import json
import numpy as np
import pandas as pd
import torch
import e1_verified_inference as rt
import build_feasibility_checkpoint_lineage_20260905 as lineage
import run_two_method_formal as legacy
import activeforcing_continuous_e2e_smoke_20260904 as rootcase
import activeforcing_feasibility_features as features

ROOT=rt.ROOT
OUT=ROOT/'analysis/results/e1_verified_recovery_revision_20260905'

def main():
    torch.set_num_threads(2)
    tpi=rt.module('additional_replay_tpi',rootcase.TPI_SOURCE)
    ensemble=lineage.load_ensemble(lineage.OLD_SEEDS,'old')
    handoff=rootcase.build_handoff_dataframe();state,mask=tpi.state_from(handoff)
    saved=json.loads((rootcase.OUT/'ROOT7703_POSTERIOR_INFERENCE.json').read_text())
    cases=[{'case':'root7703 prior','expected':4.61,'x':tpi.nominal_from(handoff,5,3.,.3,state,mask),
            'mus':rootcase.PRIOR_MUS,'f':[r['force_N'] for r in saved['force_curve']],
            'p':[r['p_success'] for r in saved['force_curve']], 'probe_used':False,
            'source':str(rootcase.OUT/'ROOT7703_POSTERIOR_INFERENCE.json')}]
    e5=ROOT/'analysis/results/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_103000_task5_tuple02'
    cid='activeforcing_full_t5_root00_s7200_high_mu0.964156'
    estimator=legacy.DirectRuntime()
    mu,sd=estimator.estimate_mu(e5/'probe'/f'{cid}.csv')
    for tag,expected,skeleton in [('NO_QUERY_TRAINING_PRIOR_UTILITY',3.75,'query_skeleton_root'),('ACTIVEFORCING_1Q_UTILITY',3.25,'query_skeleton_post')]:
        path=e5/'decisions'/f'{cid}_{tag}.json';decision=json.loads(path.read_text())
        cmd=pd.read_csv(e5/skeleton/f'{cid}.csv')[['cmd_x','cmd_y','cmd_z']].to_numpy()[:8]
        if expected==3.75:
            s=m=np.zeros(13,np.float32);mus=decision['prior_support_mu'];mu_error=None
        else:
            s,m,_,_=legacy.strict_state_from_probe(features.rows(e5/'probe'/f'{cid}.csv'))
            mus=[mu];mu_error=abs(mu-decision['mu_hat'])
            assert mu_error<1e-6,(mu,decision['mu_hat'])
        cases.append({'case':tag,'expected':expected,'x':legacy.nominal_segment(cmd,5,3.,mus[0],s,m),
            'mus':mus,'f':[r['candidate_force_N'] for r in decision['scores']],
            'p':[r['raw_probability'] for r in decision['scores']],'probe_used':expected==3.25,
            'raw_probe_mu_error':mu_error,'source':str(path)})
    rows=[];curves=[]
    for c in cases:
        f=np.asarray(c['f']);mus=np.asarray(c['mus'])
        x=np.repeat(c['x'][None],len(f)*len(mus),axis=0)
        x[:,:,17]=np.repeat(f/8,len(mus))[:,None];x[:,:,18]=np.tile(mus,len(f))[:,None]
        ps=[]
        for _,model,mean,std,_ in ensemble:
            xn=(x-mean)/std
            with torch.no_grad():ps.append(torch.sigmoid(model(torch.tensor(xn[:,:,:17]),torch.tensor(xn[:,0,17:]))).numpy())
        p=np.mean(ps,axis=0).reshape(len(f),len(mus)).mean(1)
        dec=rt.select(f,p,5);error=float(np.max(abs(p-c['p'])))
        assert error<2e-6 and abs(dec['selected_setpoint']-c['expected'])<1e-8
        rows.append({k:v for k,v in c.items() if k not in ['x','mus','f','p']}|dec|{'max_probability_error':error})
        curves.extend({'case':c['case'],'setpoint':float(a),'replayed_p':float(b),'historical_p':float(z)} for a,b,z in zip(f,p,c['p']))
    pd.DataFrame(rows).to_csv(OUT/'ADDITIONAL_CHAINED_REPLAY.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'ADDITIONAL_CHAINED_CURVES.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
