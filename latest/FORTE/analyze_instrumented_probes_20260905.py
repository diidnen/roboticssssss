"""Analyze the authorized probe-only readbacks; no additional physics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import e1_verified_inference as rt
import activeforcing_e2e_task0_smoke_20260905 as smoke
import activeforcing_feasibility_features as feat

ROOT=rt.ROOT
OUT=ROOT/'analysis/results/probe_only_instrumented_20260905'
P5=Path('/home/exouser/Tabero/analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542')

def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,default=str)+'\n')

def main():
    torch.set_num_threads(2)
    summaries=[];contacts=[];selections=[];curves=[];posteriors=[]
    full=ROOT/'analysis/results/final_fulltask_posterior_feasibility_20260905'
    ens=[]
    for seed in range(3):
        ck=torch.load(full/f'FINAL_POSTERIOR_FULLTASK_FEAS_seed{seed}.pt',map_location='cpu',weights_only=False)
        m=rt.DirectNet();m.load_state_dict(ck['state_dict']);m.eval()
        ens.append((m,np.asarray(ck['normalization_mean'],np.float32),np.asarray(ck['normalization_std'],np.float32)))
    fs=np.round(np.arange(3,5.0001,.01),2)
    for band in ['high','mid','low']:
        job=OUT/'jobs'/band
        if not (job/'RESULT.json').exists():continue
        result=json.loads((job/'RESULT.json').read_text());cid=result['context']['context_id']
        new=job/'RAW_PROBE.csv';previous=smoke.OUT/'PROBE_TELEMETRY'/f'{cid}_repeat1.csv'
        df=pd.read_csv(new);old=pd.read_csv(previous)
        xa=rt.raw_features(new);xb=rt.raw_features(previous)
        feature_error=None if xa.shape!=xb.shape else float(np.max(abs(xa-xb)))
        data=json.loads((job/'CONTACT_READBACK.json').read_text())
        if len(df)!=len(data):raise ValueError('Instrumentation took a different number of steps')
        frame_errors=[];extra_contacts=[]
        for r,row in zip(data,df.itertuples()):
            assert r['step']==row.step
            obj=r.get('target_object_force')
            if obj is None:continue
            local=np.asarray(r['policy_local_force']).reshape(-1,2,3)[-1]
            filtered=np.asarray([obj['F_obj_left_local'],obj['F_obj_right_local']])
            allw=np.asarray(r['all_contact_world']).reshape(-1,2,3)[-1]
            objw=np.asarray([obj['F_obj_left_world'],obj['F_obj_right_world']])
            worlderror=float(np.max(abs(allw-objw)));localerror=float(np.max(abs(local-filtered)))
            extra_contacts.append(worlderror);frame_errors.append(localerror)
            contacts.append({'band':band,'step':r['step'],'phase':row.probe_phase,'all_vs_object_world_max_error':worlderror,
                'policy_vs_object_local_max_error':localerror,'measured_fn':row.measured_fn,'measured_ft':row.measured_ft,'rho':row.ft_over_fn,
                'object_filtered_bilateral_n':obj['F_obj_bilateral_n'],'object_bilateral_contact':obj['target_object_bilateral_contact'],
                'sum_all_contact_world_x':float(allw[:,0].sum()),'sum_all_contact_world_y':float(allw[:,1].sum()),'sum_all_contact_world_z':float(allw[:,2].sum())})
        hold=df[df.probe_phase=='hold'];out=df[df.probe_phase=='probe_out']
        mats=np.asarray(result['post_object_material']).reshape(-1,3)
        historical=pd.read_csv(P5/'P5S0C_PROBE_TELEMETRY'/f'{cid}_probe_timesteps.csv')
        historicalhold=historical[historical.probe_phase=='hold']
        summaries.append({'band':band,'context_id':cid,'rows':len(df),'outward_steps':len(out),'stop':result['probe_record']['stop_trigger'],
            'requested_mu':result['context']['hidden_friction_analysis_only'],'readback_static_mu':float(mats[:,0].mean()),'readback_dynamic_mu':float(mats[:,1].mean()),
            'mass_kg':float(np.asarray(result['post_object_masses']).sum()),'instrumentation_errors':len(result['instrumentation_errors']),
            'old_smoke_raw46_max_error':feature_error,'hold_mean_fn':float(hold.measured_fn.mean()),'hold_mean_ft':float(hold.measured_ft.mean()),
            'historical_hold_mean_ft':float(historicalhold.measured_ft.mean()),'first_outward_rho':float(out.ft_over_fn.iloc[0]),
            'max_non_object_contact_world_error':max(extra_contacts),'max_local_coordinate_difference':max(frame_errors),
            'policy_contact_override_ever_enabled':any(bool(r.get('action_debug',{}).get('target_contact_override_enabled',False)) for r in data),
            'authoritative_inner_loop_ever_enabled':any(bool(r.get('action_debug',{}).get('authoritative_true_force_inner_loop_enabled',False)) for r in data)})
        e6=smoke.posterior_from_probe(new);e1members=rt.probe_members(xa[None],0)[0]
        posteriors.append({'band':band,'context':cid,'E6':e6,'E1_members':e1members.tolist(),'E1_mean':float(e1members.mean()),'E1_std_population':float(e1members.std()),'source_raw_probe':str(new)})
        command=sorted((smoke.VALIDATION/'TASK0_TRACES').glob(f'{cid}_*.csv'))[0]
        e1x=rt.e1_nominal_from_saved(new,command,0,3,float(e1members.mean()))
        cleanx,_=feat.from_saved_probe(cid,new,command,0,3,e6['mean'],state_path=job/'POST_PROBE_STATE.pt')
        scenarios=[('E1_PRIMARY_POINT',e1x,[float(e1members.mean())]),('E1_POSTERIOR',e1x,e1members),
                   ('CURRENT_CLEAN_LIFTHOLD',cleanx,e6['member_means']),('EXISTING_CLEAN_FULLTASK',cleanx,e6['member_means'])]
        for model,base,mus in scenarios:
            x=np.repeat(base[None],len(fs)*len(mus),axis=0)
            x[:,:,17]=np.repeat(fs/8,len(mus))[:,None];x[:,:,18]=np.tile(mus,len(fs))[:,None]
            if model.startswith('E1'):p=rt.probabilities(x,0)
            elif model=='CURRENT_CLEAN_LIFTHOLD':p=rt.probabilities(x,0,'clean')
            else:
                ps=[]
                for m,mean,std in ens:
                    with torch.no_grad():ps.append(torch.sigmoid(m(torch.tensor((x-mean)/std))).numpy())
                p=np.mean(ps,axis=0)
            p=p.reshape(len(fs),len(mus)).mean(1)
            selections.append({'band':band,'model':model,'diagnostic_only':True,'deployment_approved':False,'p3':float(p[0]),'p5':float(p[-1]),**rt.select(fs,p,5)})
            curves.extend({'band':band,'model':model,'setpoint':float(f),'p':float(v),'utility':float(v*(2-f/5)-1)} for f,v in zip(fs,p))
    pd.DataFrame(summaries).to_csv(OUT/'PROBE_ONLY_SUMMARY.csv',index=False)
    pd.DataFrame(contacts).to_csv(OUT/'PER_STEP_CONTACT_DIAGNOSIS.csv',index=False)
    pd.DataFrame(selections).to_csv(OUT/'NEW_PROBE_OFFLINE_SELECTION.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'NEW_PROBE_OFFLINE_CURVES.csv',index=False)
    dump('NEW_PROBE_POSTERIORS.json',posteriors)
    dump('ANALYSIS_STATUS.json',{'completed_contexts':len(summaries),'all_three_completed':len(summaries)==3,'physical_task_branches':0,
        'readback_object_material_matches_requested':all(abs(r['readback_static_mu']-r['requested_mu'])<1e-6 and abs(r['readback_dynamic_mu']-r['requested_mu'])<1e-6 for r in summaries),
        'all_new_probe_raw_features_match_previous_smoke':all(r['old_smoke_raw46_max_error']==0 for r in summaries),
        'final_physical_rerun_allowed':False,'training_performed':False})
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(selections)[['band','model','selected_setpoint','p3','p5']].to_string(index=False))

if __name__=='__main__':main()
