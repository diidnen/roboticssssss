#!/usr/bin/env python3
"""LEGACY_E1_REPLAY: isolated, checksum-locked, CPU-only, no training/physics."""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def dump(out,name,value):(out/name).write_text(json.dumps(value,indent=2,default=str)+'\n')

def verify(manifest):
    for item in manifest['files']:
        p=HERE/item['path']
        if sha(p)!=item['sha256']:raise RuntimeError(f'LOCKED_BUNDLE_HASH_MISMATCH:{p}')

def legacy_nominal(commands,phases,task,force,mu,opening):
    # Exact original task0_visual_context_early.strict_preprobe_state,
    # trajectory_physical_imagination.nominal_from and cf.build_seg contract.
    state=np.zeros(13,np.float32);mask=np.zeros(13,np.float32)
    state[11:13]=[opening,-opening];mask[:6]=1.;mask[11:13]=1.
    ph=np.stack([(phases==p).astype(float) for p in ['branch_hold','lift','transit','over_basket','place','release','settle']],1)
    taskvec=np.zeros((8,4));taskvec[:,[0,1,5,6].index(task)]=1
    repeat=lambda v:np.repeat(np.asarray(v)[None],8,axis=0)
    return np.concatenate([commands-commands[0],np.vstack([np.zeros((1,3)),np.diff(commands,axis=0)]),ph,taskvec,
                           repeat([force/8.,mu]),repeat(state),repeat(mask),repeat(state),repeat(mask)],axis=1).astype(np.float32)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args()
    manifest=json.loads((HERE/'MANIFEST.json').read_text());verify(manifest)
    if args.output:
        out=args.output.resolve()
        if out.exists():raise ValueError('Use a new output directory; never overwrite a previous replay')
        if HERE not in out.parents:raise ValueError('Replay outputs must stay inside LEGACY_E1_REPLAY')
        out.mkdir(parents=True)
    else:
        (HERE/'runs').mkdir(exist_ok=True)
        out=Path(tempfile.mkdtemp(prefix='replay_',dir=HERE/'runs'))
    torch.set_num_threads(2)
    rt=module('locked_e1_runtime',HERE/'vendor/e1_verified_inference.py')
    rt.PROBE=HERE/'checkpoints/probe';rt.DIRECT=HERE/'checkpoints/direct';rt.TRANSFER=HERE/'vendor'
    old=module('locked_legacy407_runtime',HERE/'vendor/recover_e1_offline.py')
    old.PROBE_OOF_DIR=rt.PROBE;old.EVIDENCE=HERE/'vendor/evidence_46d.py';old.EVIDENCE_NORM=HERE/'vendor/NORMALIZATION_46D.json'
    old.CURRENT_PROBE_DIR=HERE/'inputs/legacy_407/probes';old.CURRENT_VALIDATION=HERE/'inputs/legacy_407/commands'
    data=pd.read_csv(HERE/'reference/OOF_DATA.csv');probe=pd.read_csv(HERE/'reference/OOF_PROBE.csv')
    contexts=probe[probe.seed.astype(str)=='0'].sort_values('probe_index');assert len(contexts)==72 and len(data)==720
    verified=np.load(HERE/'reference/VERIFIED_HISTORICAL_INPUTS.npz');commands=np.load(HERE/'inputs/COMMAND_PREFIXES.npz')
    assert data.branch_id.tolist()==commands['branch_ids'].tolist()
    raw=[];openings={};support=[]
    for row in contexts.itertuples():
        path=HERE/f'inputs/historical_probes/{row.context_id}.csv';df=pd.read_csv(path)
        assert len(df)==215 and int((df.probe_phase=='probe_out').sum())==10
        hold=df[df.probe_phase=='hold'].iloc[-1];assert int(hold.step)==190
        openings[row.context_id]=float(hold.gripper_opening);raw.append(rt.raw_features(path))
        support.append(dict(context_id=row.context_id,rows=len(df),outward_steps=10,sha256=sha(path)))
    raw=np.stack(raw);np.testing.assert_array_equal(raw,verified['raw_probe'])
    members=np.empty((72,3),np.float32)
    for fold in range(3):
        ids=np.flatnonzero(contexts.fold.to_numpy()==fold);members[ids]=rt.probe_members(raw[ids],fold)
    reference=probe[probe.seed.astype(str).isin(['0','1','2'])].pivot(index='context_id',columns='seed',values='mu_hat').reindex(contexts.context_id)[['0','1','2']].to_numpy(float)
    mu_error=float(np.max(abs(members-reference)));assert mu_error<=1e-6
    row_members=np.stack([members[list(contexts.context_id).index(c)] for c in data.context_id])
    x=np.stack([legacy_nominal(commands['commands'][i],commands['phases'][i],int(r.task),float(r.force_N),float(r.mu),openings[r.context_id]) for i,r in enumerate(data.itertuples())])
    feature_error=float(np.max(abs(x-verified['feasibility_inputs'])));assert feature_error==0.,feature_error
    probs={};scenarios={'POINT_MU':row_members.mean(1)}
    scenarios.update({f'MEMBER{s}':row_members[:,s] for s in range(3)})
    for name,mu in scenarios.items():
        pp=np.empty(720,np.float32)
        for fold in range(3):
            ids=np.flatnonzero(data.fold.to_numpy()==fold);xx=x[ids].copy();xx[:,:,18]=mu[ids,None]
            pp[ids]=rt.probabilities(xx,fold,kind='e1')
        probs[name]=pp
    probs['POSTERIOR_ABLATION']=np.stack([probs[f'MEMBER{s}'] for s in range(3)]).mean(0)
    hist=pd.read_csv(HERE/'reference/E1_SELECTED_EPISODES.csv');hist=hist[hist.method=='ActiveForcing-1Q Utility']
    posterior=pd.read_csv(HERE/'reference/POSTERIOR_REFERENCE.csv');posterior=posterior[posterior.analysis=='Posterior']
    decisions=[];curves=[]
    for (cid,rep),g in data.groupby(['context_id','repeat']):
        ids=g.index.to_numpy();fs=g.force_N.to_numpy();fmax=old.FMAX[int(g.task.iloc[0])]
        for mode,ref in [('POINT_MU',hist),('POSTERIOR_ABLATION',posterior)]:
            sel=rt.select(fs,probs[mode][ids],fmax);rr=ref[(ref.context_id==cid)&(ref['repeat']==rep)];assert len(rr)==1;rr=rr.iloc[0]
            chosen=g.iloc[np.argmin(abs(fs-sel['selected_setpoint']))]
            decisions.append(dict(context_id=cid,repeat=int(rep),task=int(g.task.iloc[0]),mode=mode,**sel,
                 historical_selected_F=float(rr.selected_force_N),selected_error=abs(sel['selected_setpoint']-rr.selected_force_N),
                 probability_error=abs(sel['predicted_success']-rr.predicted_success_probability),historical_outcome=int(chosen.success),
                 branch_id=chosen.branch_id,branch_match=bool(chosen.branch_id==rr.branch_id)))
            curves.extend(dict(context_id=cid,repeat=int(rep),mode=mode,force=float(data.force_N.iloc[i]),p_success=float(probs[mode][i])) for i in ids)
    dd=pd.DataFrame(decisions);assert len(dd)==288 and dd.selected_error.max()<1e-10 and dd.probability_error.max()<1e-6 and dd.branch_match.all()
    dd.to_csv(out/'E1_ALL_DECISIONS.csv',index=False);pd.DataFrame(curves).to_csv(out/'E1_ALL_SUCCESS_CURVES.csv',index=False)
    golden=pd.read_csv(HERE/'reference/ORIGINAL_SEVEN_GOLDEN.csv')[['context_id','repeat']].drop_duplicates()
    golden.merge(dd,on=['context_id','repeat']).to_csv(out/'E1_SEVEN_GOLDEN_REPLAY.csv',index=False)
    pd.DataFrame(support).to_csv(out/'HISTORICAL_PROBE_SUPPORT.csv',index=False)
    pd.DataFrame(members,columns=['mu_seed0','mu_seed1','mu_seed2']).assign(context_id=contexts.context_id.to_numpy()).to_csv(out/'RAW_PROBE_RECOMPUTED_MEMBERS.csv',index=False)
    print('E1_MAIN + POSTERIOR: 288/288 matched',flush=True)
    # Reproduce the old reported 4.07/4.52/4.19 numbers with their actual inputs.
    # Do not repair its prefix and do not send it through current runtime.
    direct=[]
    for seed in range(3):
        ck=torch.load(rt.DIRECT/f'DIRECT_POOLED_fold0_seed{seed}.pt',map_location='cpu',weights_only=False)
        model=rt.DirectNet();model.load_state_dict(ck['state_dict']);model.eval()
        direct.append((model,np.asarray(ck['normalization']['x_mean'],np.float32),np.asarray(ck['normalization']['x_std'],np.float32)))
    def score(model,xx,mean,std):
        xx=(xx-mean)/np.maximum(std,1e-6)
        with torch.no_grad():z=model(torch.tensor(xx[None])).item()
        return 1/(1+math.exp(-max(-50.,min(50.,z))))
    reference_sel=json.loads((HERE/'reference/LEGACY_407_SELECTION.json').read_text());reference_curve=pd.read_csv(HERE/'reference/LEGACY_407_CURVES.csv')
    current=[];current_curves=[]
    for r in manifest['current_fixture_contexts']:
        band,cid=r['band'],r['context_id'];post=old.e1_physical_inference(old.CURRENT_PROBE_DIR/f'{cid}.csv')
        base=old.current_prefix(cid,3.,0.);rows=[]
        for f in old.GRID:
            ps=[]
            for mu in post['member_means']:
                xx=base.copy();xx[:,17]=float(f)/8.;xx[:,18]=float(mu)
                ps.append(float(np.mean([score(m,xx,xm,xs) for m,xm,xs in direct])))
            pp=float(np.mean(ps));rows.append(dict(band=band,force=float(f),p_success=pp,utility=float(old.utility(pp,float(f),5))))
        best=max(rows,key=lambda z:(z['utility'],-z['force']));expected=reference_sel[band]['E1_style_F_star']
        assert best['force']==expected,(band,best,expected)
        rc=reference_curve[reference_curve.band==band].sort_values('force_N')
        error=float(np.max(abs(rc.p_success.to_numpy()-np.array([z['p_success'] for z in rows]))));assert error<1e-6,error
        current.append(dict(band=band,selected_setpoint=best['force'],historical_selected_setpoint=expected,
                            curve_max_error=error,probe_rows=post['feature_rows'],posterior=post,
                            input_contract='EXACT_ORIGINAL_LEGACY_PREFIX_BUGS_RETAINED_OFFLINE_ONLY'))
        current_curves+=rows
        print(f'LEGACY407 {band}={best["force"]:.2f}; curve error={error:.3g}',flush=True)
    pd.DataFrame(current_curves).to_csv(out/'LEGACY_407_SUCCESS_CURVES.csv',index=False);dump(out,'LEGACY_407_SELECTION.json',current)
    protected_ok=all(sha(p['path'])==p['sha256'] for p in manifest['protected_current_files'])
    assert protected_ok,'Current pipeline changed since bundle freeze; investigate without overwriting it'
    verify(manifest)
    result={'FINAL_STATUS':'LEGACY_E1_EXACT_OFFLINE_REPLAY_PASS','E1_POINT_MATCHED':144,'E1_POSTERIOR_MATCHED':144,
            'E1_POINT_MEAN_F':float(dd[dd['mode']=='POINT_MU'].selected_setpoint.mean()),
            'E1_POSTERIOR_MEAN_F':float(dd[dd['mode']=='POSTERIOR_ABLATION'].selected_setpoint.mean()),
            'LEGACY_HIGH_MID_LOW':[next(z['selected_setpoint'] for z in current if z['band']==b) for b in ['HIGH','MID','LOW']],
            'GOLDEN_CASE_COUNT':len(golden),'SELECTED_SETPOINT_MAX_ERROR':float(dd.selected_error.max()),
            'SELECTED_PROBABILITY_MAX_ERROR':float(dd.probability_error.max()),'RAW_PROBE_MEMBER_MAX_ERROR':mu_error,
            'FEATURE_MAX_ERROR':feature_error,'CURRENT_PIPELINE_CHANGED':False,'MODEL_RETRAINED':False,'PHYSICS_EXECUTED':False,
            'REPLAY_COMMAND':f'/usr/bin/python3 {HERE}/replay.py','MANIFEST_SHA256':sha(HERE/'MANIFEST.json'),
            'OUTPUT_DIR':str(out),'historical_success_rate_reselected_from_archive':float(dd[dd['mode']=='POINT_MU'].historical_outcome.mean()),
            'physical_success_rate_remeasured':False,'torch_version':torch.__version__,'numpy_version':np.__version__}
    dump(out,'REPLAY_RESULT.json',result)
    report='# LEGACY_E1_REPLAY — PASS\n\n'+json.dumps(result,indent=2)+'\n\n'
    report+='E1 main is point-mu; its posterior ablation is separate. Legacy407 uses its original206-row saved probes and buggy prefix, not the main-table fixture. All three lanes reproduce their own historical decisions. No new physical rollout or retraining occurred.\n'
    (out/'REPLAY_REPORT.md').write_text(report)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
