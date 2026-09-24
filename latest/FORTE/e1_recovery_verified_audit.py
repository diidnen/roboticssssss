"""Independent chained replay and current signal diagnostics; CPU only."""
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import e1_verified_inference as runtime

ROOT = runtime.ROOT
OUT = ROOT / 'analysis/results/e1_verified_recovery_revision_20260905'
PREVIOUS = ROOT / 'analysis/results/e1_pipeline_recovery_and_current_rerun_20260905'


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    old = runtime.module('previous_recovery_readonly', PREVIOUS / 'recover_e1_offline.py')
    cl = runtime.module('verified_closure', ROOT / 'utility_and_causal_ablation_closure.py')
    data = pd.read_csv(old.ARCHIVE)
    historical = pd.read_csv(old.E1_TABLE)
    hist = historical[historical.method == 'ActiveForcing-1Q Utility'].copy()
    probe = pd.read_csv(old.PROBE_PRED)
    contexts = probe[probe.seed.astype(str) == '0'].sort_values('probe_index')
    raw = np.stack([runtime.raw_features(Path(str(r.raw_path).replace('/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000', str(old.OLD_SOURCE)))) for r in contexts.itertuples()])
    pm = np.empty((72, 3), np.float32)
    for fold in range(3):
        ids = np.flatnonzero(contexts.fold.to_numpy() == fold)
        pm[ids] = runtime.probe_members(raw[ids], fold)
    mu_by_context = {cid: values for cid, values in zip(contexts.context_id, pm)}
    ref_pm = probe[probe.seed.astype(str).isin(['0','1','2'])].pivot(index='context_id', columns='seed', values='mu_hat').reindex(contexts.context_id)[['0','1','2']].to_numpy()
    probe_error = float(np.max(abs(ref_pm-pm)))
    with tempfile.TemporaryDirectory(prefix='e1_verified_metadata_') as st, tempfile.TemporaryDirectory(prefix='e1_verified_loader_') as ld:
        old.patch_population_modules(cl, old.make_rewritten_source(Path(st)))
        tpi, cf, full, cmap, traces, meta, audits, pairs, segs, norm = cl.pooled.load_population(Path(ld))
        assert [t.branch_id for t in traces] == data.branch_id.tolist()
        x = np.stack([segs[t.branch_id].x for t in traces])
    row_mu = np.stack([mu_by_context[c] for c in data.context_id])
    scenarios = {'point': row_mu.mean(1), 'GT': data.mu.to_numpy(np.float32)}
    scenarios.update({f'member{s}': row_mu[:,s] for s in range(3)})
    probs = {}
    for key, mu in scenarios.items():
        probs[key] = np.empty(len(x), np.float32)
        for fold in range(3):
            ids = np.flatnonzero(data.fold.to_numpy() == fold)
            xx = x[ids].copy(); xx[:,:,18] = mu[ids,None]
            probs[key][ids] = runtime.probabilities(xx, fold)
    probs['posterior'] = np.stack([probs[f'member{s}'] for s in range(3)]).mean(0)
    allcurves, decisions = [], []
    for (cid, repeat), g in data.groupby(['context_id','repeat']):
        ids = g.index.to_numpy(); fs = g.force_N.to_numpy(); fmax = old.FMAX[int(g.task.iloc[0])]
        for variant in ['point','posterior']:
            dec = runtime.select(fs, probs[variant][ids], fmax)
            row = {'context_id':cid, 'repeat':int(repeat), 'variant':variant, 'task':int(g.task.iloc[0]), **dec}
            historical_rows = hist[(hist.context_id==cid)&(hist['repeat']==repeat)]
            if variant=='point':
                hr = historical_rows.iloc[0]
            else:
                table = pd.read_csv(ROOT/'UTILITY_CAUSAL_ABLATION_PER_EPISODE.csv')
                hr = table[(table.analysis=='Posterior')&(table.context_id==cid)&(table['repeat']==repeat)].iloc[0]
            row.update(historical_setpoint=float(hr.selected_force_N), historical_predicted_success=float(hr.predicted_success_probability), selected_abs_error=abs(dec['selected_setpoint']-float(hr.selected_force_N)), probability_abs_error=abs(dec['predicted_success']-float(hr.predicted_success_probability)))
            decisions.append(row)
            for i in ids:
                allcurves.append({'context_id':cid,'repeat':repeat,'variant':variant,'setpoint':float(data.force_N.iloc[i]),'p_success':float(probs[variant][i]),'utility':float(probs[variant][i]*(2-data.force_N.iloc[i]/fmax)-1)})
    dd = pd.DataFrame(decisions)
    dd.to_csv(OUT/'CHAINED_ALL_DECISIONS.csv',index=False)
    pd.DataFrame(allcurves).to_csv(OUT/'CHAINED_ALL_CURVES.csv',index=False)
    gt_error = float(np.max(abs(probs['GT']-data.p_D_OOF_ensemble.to_numpy())))
    parity = {'raw_probe_member_max_error':probe_error,'GT_probability_max_error':gt_error,'point_decisions':144,'posterior_decisions':144,'setpoint_max_error':float(dd.selected_abs_error.max()),'selected_probability_max_error':float(dd.probability_abs_error.max()),'all_mu_inferred_from_raw':True,'historical_mu_used_as_input':False,'passed':bool(probe_error<1e-6 and gt_error<1e-6 and dd.selected_abs_error.max()<1e-6 and dd.probability_abs_error.max()<1e-6),'point_mean':float(dd[dd.variant=='point'].selected_setpoint.mean()),'posterior_mean':float(dd[dd.variant=='posterior'].selected_setpoint.mean())}
    dump('CHAINED_PARITY.json',parity)
    assert parity['passed'], parity
    np.savez_compressed(OUT/'VERIFIED_HISTORICAL_INPUTS.npz', raw_probe=raw, probe_members=pm, feasibility_inputs=x)
    current_rows=[]; curve_rows=[]; phase_rows=[]
    ck = torch.load(runtime.DIRECT/'DIRECT_POOLED_fold0_seed0.pt',map_location='cpu',weights_only=False)
    shared_norm = [ck['normalization'][k] for k in ['x_mean','x_std']]
    for band,cid in old.CURRENT_CIDS.items():
        for repeat in [1,2]:
            path=old.CURRENT_PROBE_DIR/f'{cid}_repeat{repeat}.csv'
            rows=list(csv.DictReader(path.open()))
            rr=runtime.raw_features(path)
            members=runtime.probe_members(rr[None],0)[0]
            gate=runtime.validate_live_probe(rows,members)
            hist_cid=cid.replace('p5s0c_train','pv_train')
            hi=int(np.flatnonzero(contexts.context_id.to_numpy()==hist_cid)[0])
            hist_path=Path(str(contexts.iloc[hi].raw_path).replace('/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000',str(old.OLD_SOURCE)))
            for source,pp in [('historical',hist_path),('current',path)]:
                df=pd.read_csv(pp)
                for phase,g in df.groupby('probe_phase',sort=False):
                    phase_rows.append({'context':cid,'repeat':repeat,'source':source,'phase':phase,'rows':len(g),'mean_fn':float(g.measured_fn.mean()),'mean_ft':float(g.measured_ft.mean()),'max_ft_over_fn':float(g.ft_over_fn.max()),'end_displacement_mm':float(g.accumulated_displacement_mm.iloc[-1]),'last_stop_trigger':str(g.stop_trigger.iloc[-1])})
            # Audit both templates; actual inference uses current frozen commands.
            template=x[data.context_id==hist_cid][0].copy()
            hold=[r for r in rows if r['probe_phase']=='hold']
            state=np.zeros(13,np.float32); mask=np.zeros(13,np.float32)
            opening=float(hold[-1]['gripper_opening']); state[11:13]=[opening,-opening]; mask[:6]=1; mask[11:13]=1
            template[:,19:32]=state; template[:,32:45]=mask; template[:,45:58]=state; template[:,58:71]=mask
            command_paths = sorted((ROOT/'analysis/results/current_runtime_setpoint_mapping_validation_20260905/TASK0_TRACES').glob(f'{cid}_*.csv'))
            live_template = runtime.e1_nominal_from_saved(path,command_paths[0],0,float(template[0,17]*8),float(template[0,18]))
            template_difference = float(np.max(abs(template-live_template)))
            template = live_template
            for mode in ['point','posterior']:
                supports=[float(members.mean())] if mode=='point' else members
                batch=np.stack([template.copy() for f in old.GRID for mu in supports])
                batch[:,:,17]=np.repeat(old.GRID/8,len(supports))[:,None]
                batch[:,:,18]=np.tile(supports,len(old.GRID))[:,None]
                for model_kind,norm_kind in [('e1','native'),('clean','native'),('clean','shared_e1')]:
                    p=runtime.probabilities(batch,0,model_kind,shared_norm=shared_norm if norm_kind=='shared_e1' else None).reshape(len(old.GRID),len(supports)).mean(1)
                    selection=runtime.select(old.GRID,p,5)
                    current_rows.append({'context':cid,'band':band,'repeat':repeat,'mode':mode,'model':model_kind,'normalization':norm_kind,'probe_members':members.tolist(),'probe_mean':float(members.mean()),'probe_std':float(members.std(ddof=0)),'admitted':gate['admitted'],'reasons':gate['reasons'],'inference_status':'DIAGNOSTIC_ONLY' if not gate['admitted'] else 'SUPPORT_ADMITTED','command_source':str(command_paths[0]),'historical_template_max_difference':template_difference,**selection})
                    curve_rows.extend({'context':cid,'repeat':repeat,'mode':mode,'model':model_kind,'normalization':norm_kind,'setpoint':float(f),'p':float(pv),'U':float(pv*(2-f/5)-1)} for f,pv in zip(old.GRID,p))
    pd.DataFrame(phase_rows).to_csv(OUT/'PROBE_PHASE_COMPARISON.csv',index=False)
    pd.DataFrame(current_rows).to_csv(OUT/'CURRENT_DIAGNOSTIC_SELECTIONS.csv',index=False)
    pd.DataFrame(curve_rows).to_csv(OUT/'CURRENT_DIAGNOSTIC_CURVES.csv',index=False)
    dump('DEPLOYMENT_GATE.json',{'exact_historical_parity':parity['passed'],'current_probes_admitted':bool(all(r['admitted'] for r in current_rows)),'physical_rerun_allowed':False,'reason':'Current probes outside verified E1 support; posterior transfer and live nominal input unresolved. User already authorized conditional physical run; gate is evidentiary, not permission.'})
    paths=[Path(__file__),Path(runtime.__file__)]+list(runtime.PROBE.glob('probe_fold*_seed*.pt'))+list(runtime.DIRECT.glob('DIRECT_POOLED_fold*_seed*.pt'))
    dump('RUNTIME_HASHES.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    print(json.dumps(parity,indent=2),flush=True)
    print(pd.DataFrame(current_rows)[['band','repeat','mode','model','normalization','selected_setpoint','admitted']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
