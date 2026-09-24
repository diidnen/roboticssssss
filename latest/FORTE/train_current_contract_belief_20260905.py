"""One predeclared recipe, TRAIN-only normalization, VAL-only selection.

Test labels are loaded only AFTER all three checkpoints have been locked.
No simulator or feasibility training is imported or launched.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import norm,spearmanr
from scipy.optimize import brentq

ROOT=Path('/home/exouser/FORTE');OUT=ROOT/'analysis/results/current_contract_physical_belief_20260905'
sys.path.insert(0,str(ROOT))
from current_contract_belief_features import ProbeEvidence,SCHEMA_ID,ADAPTER,LEGACY_SCHEMA
from current_contract_physical_belief import FrictionMember,batch,predict,CurrentPhysicalBelief
from collect_current_contract_belief_20260905 import check_protocol


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def write(name,v): (OUT/name).write_text(json.dumps(clean(v),indent=2,allow_nan=False)+'\n')
def csvout(name,rows):
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def load_split(protocol,split):
    result=[]
    for p in protocol['contexts']:
        if p['split']!=split:continue
        job=Path(p['data_path']);record=json.loads((job/'RESULT.json').read_text())
        if split=='DIAGNOSTIC':qualified=not any(record['probe_record'].get(k,0) for k in ['probe_failure','contact_lost_probe','dropped','major_disturbance'])
        else:
            qualified=record['probe_qualified']
            for name,h in record['sha256'].items():
                if sha(job/name)!=h:raise RuntimeError('Acquisition artifact changed')
        if not qualified:continue
        x=ProbeEvidence().load(job)
        if (job/'RAW_FEATURES.npy').exists():np.testing.assert_array_equal(x,np.load(job/'RAW_FEATURES.npy'))
        result.append({'context_id':p['context_id'],'root_seed':p['root_seed'],'split':split,
            'band':p['friction_band'],'y':float(p['hidden_friction_analysis_only']),'raw':x,'path':job})
    if not result:raise RuntimeError('No admitted probes in '+split)
    return result


def distribution_metrics(y,mu,sigma):
    means=mu.mean(1)
    nll=-(logsumexp(norm.logpdf(y[:,None],loc=mu,scale=sigma),axis=1)-np.log(mu.shape[1]))
    intervals={}
    for level in [.5,.8,.9,.95]:
        bounds=[]
        for m,s in zip(mu,sigma):
            lo=float(m.min()-12*s.max()-1);hi=float(m.max()+12*s.max()+1)
            cdf=lambda z:float(norm.cdf(z,loc=m,scale=s).mean())
            bounds.append([brentq(lambda z:cdf(z)-(1-level)/2,lo,hi),brentq(lambda z:cdf(z)-(1+level)/2,lo,hi)])
        a=np.asarray(bounds);intervals[str(level)]={'coverage':float(((y>=a[:,0])&(y<=a[:,1])).mean()),
            'mean_width':float((a[:,1]-a[:,0]).mean()),'bounds':bounds}
    metrics={'count':len(y),'MAE':float(np.abs(y-means).mean()),'RMSE':float(np.sqrt(((y-means)**2).mean())),
        'GAUSSIAN_MIXTURE_NLL':float(nll.mean()),
        'SPEARMAN':float(spearmanr(y,means).statistic) if np.std(means)>1e-10 else None,
        'POSITIVE_MEMBER_MEAN_FRACTION':float((mu>0).mean()),
        'MEAN_NEGATIVE_MASS_GAUSSIAN_DIAGNOSTIC':float(norm.cdf(0,loc=mu,scale=sigma).mean()),
        'intervals':intervals}
    return metrics


def root_order(examples,mu):
    groups={}
    for ex,m in zip(examples,mu.mean(1)):groups.setdefault(ex['root_seed'],{})[ex['band']]=float(m)
    complete={r:v for r,v in groups.items() if set(v)=={'LOW','MID','HIGH'}}
    rows=[{'root_seed':r,**v,'ordered':v['LOW']<v['MID']<v['HIGH']} for r,v in complete.items()]
    return {'complete_roots':len(rows),'ordered_fraction':float(np.mean([r['ordered'] for r in rows])) if rows else None,'roots':rows}


def main():
    torch.set_num_threads(2);protocol=check_protocol()
    if not (OUT/'COLLECTION_COMPLETE.json').exists():raise RuntimeError('Collection incomplete')
    if (OUT/'CHECKPOINT_SELECTION_LOCK.json').exists():raise RuntimeError('Already trained; no overwrite or lucky-seed rerun')
    train=load_split(protocol,'TRAIN');val=load_split(protocol,'VAL')
    joined=np.concatenate([x['raw'] for x in train]);mean=joined.mean(0);std=np.maximum(joined.std(0),1e-6)
    normalization={'mean':mean.tolist(),'std':std.tolist(),'fit_split':'TRAIN','fit_contexts':len(train),
        'fit_roots':sorted({x['root_seed'] for x in train}),'fit_unpadded_steps':len(joined),
        'axes':'sample and real time steps; exclude padding','all_channels':True,'std_floor':1e-6}
    write('NORMALIZATION.json',normalization)
    for ex in train+val:ex['x']=((ex['raw']-mean)/std).astype(np.float32)
    groups={r:[ex for ex in train if ex['root_seed']==r] for r in sorted({x['root_seed'] for x in train})}
    checkpoints=[];histories=[];models=[]
    for seed in protocol['training_seeds']:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=FrictionMember();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        rng=np.random.default_rng(seed);best=float('inf');best_state=None;best_epoch=None
        for epoch in range(1,81):
            ids=rng.choice(list(groups),size=len(groups),replace=True)
            fit=[ex for r in ids for ex in groups[r]]
            x,lengths=batch([ex['x'] for ex in fit]);y=torch.tensor([ex['y'] for ex in fit],dtype=torch.float32)
            model.train();opt.zero_grad(set_to_none=True);mu,ls=model(x,lengths)
            loss=(.5*(((y-mu)/ls.exp())**2+2*ls)).mean()+.05*torch.abs(mu-y).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            vm,vs=predict(model,[ex['x'] for ex in val]);vy=np.array([ex['y'] for ex in val])
            vnll=float(-norm.logpdf(vy,vm,vs).mean())
            histories.append({'seed':seed,'epoch':epoch,'updates':epoch,'train_loss':float(loss),'val_gaussian_nll':vnll,
                'val_MAE':float(np.abs(vy-vm).mean()),'sampled_train_roots':json.dumps(ids.tolist())})
            if vnll<best:
                best=vnll;best_epoch=epoch;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best_state);model.eval();path=OUT/f'CURRENT_CONTRACT_PHYSICAL_BELIEF_seed{seed}.pt'
        torch.save({'state_dict':best_state,'input_dim':58,'projection_dim':16,'hidden_dim':16,'seed':seed,
            'selected_epoch':best_epoch,'optimizer_updates_total':80,'selected_val_nll':best,
            'feature_schema_id':SCHEMA_ID,'normalization':normalization,'protocol_sha256':sha(OUT/'PROTOCOL.json')},path)
        checkpoints.append({'seed':seed,'path':str(path),'sha256':sha(path),'selected_epoch':best_epoch,'selected_val_nll':best})
        models.append(model);print('LOCKED_SEED',seed,best_epoch,best,flush=True)
    csvout('TRAINING_HISTORY.csv',histories)
    write('CHECKPOINT_SELECTION_LOCK.json',{'checkpoints':checkpoints,'rule':protocol['checkpoint_rule'],
        'test_labels_used_for_selection':False,'diagnostic_root_used_for_selection':False,
        'training_script_sha256':sha(__file__)})
    # Only now access TEST labels/features for final assessment.
    test=load_split(protocol,'TEST');diagnostic=load_split(protocol,'DIAGNOSTIC')
    for ex in test+diagnostic:ex['x']=((ex['raw']-mean)/std).astype(np.float32)
    train_y=np.array([x['y'] for x in train]);prior_mu=float(train_y.mean());prior_sigma=max(float(train_y.std()),1e-3)
    metrics={};predictions=[];cached={}
    for split,examples in [('TRAIN',train),('VAL',val),('TEST',test),('DIAGNOSTIC',diagnostic)]:
        pm=[];ps=[]
        for model in models:
            m,s=predict(model,[ex['x'] for ex in examples]);pm.append(m);ps.append(s)
        pm=np.array(pm).T;ps=np.array(ps).T;y=np.array([ex['y'] for ex in examples]);cached[split]=(examples,pm,ps)
        single=[distribution_metrics(y,pm[:,i:i+1],ps[:,i:i+1]) for i in range(3)]
        metrics[split]={'ensemble':distribution_metrics(y,pm,ps),
            'prior':distribution_metrics(y,np.full((len(y),1),prior_mu),np.full((len(y),1),prior_sigma)),
            'seed_mean_std':{key:{'mean':float(np.mean([v[key] for v in single])),'std':float(np.std([v[key] for v in single],ddof=1))} for key in ['MAE','RMSE','GAUSSIAN_MIXTURE_NLL']},
            'per_seed':single,'root_order':root_order(examples,pm)}
        for ex,m,s in zip(examples,pm,ps):
            predictions.append({'context_id':ex['context_id'],'root_seed':ex['root_seed'],'split':split,'friction_band_analysis_only':ex['band'],
                'target_mu':ex['y'],'posterior_mean':float(m.mean()),'epistemic_std':float(np.std(m,ddof=1)),
                **{f'mu_{i}':float(m[i]) for i in range(3)},**{f'sigma_{i}':float(s[i]) for i in range(3)},
                'probe_path':str(ex['path'])})
    csvout('PHYSICAL_BELIEF_PREDICTIONS.csv',predictions)
    passed_probability=all(metrics[s]['ensemble']['MAE']<metrics[s]['prior']['MAE'] and
        metrics[s]['ensemble']['GAUSSIAN_MIXTURE_NLL']<metrics[s]['prior']['GAUSSIAN_MIXTURE_NLL'] for s in ['VAL','TEST'])
    admission=len(test)/sum(p['split']=='TEST' for p in protocol['contexts'])
    order=metrics['TEST']['root_order']['ordered_fraction']
    positive=all(metrics[s]['ensemble']['POSITIVE_MEMBER_MEAN_FRACTION']==1 for s in ['VAL','TEST'])
    coverage=metrics['TEST']['ensemble']['intervals']['0.9']['coverage']
    qualified=bool(passed_probability and admission>=.9 and order is not None and order>=.75 and positive and coverage>=.75)
    metrics['eligibility']={'task0_candidate_qualified':qualified,'probability_better_than_train_prior':passed_probability,
        'test_probe_admission_fraction':admission,'test_root_order_fraction':order,
        'test_positive_member_means':positive,'test_90_interval_coverage':coverage,
        'scope':'task0 admitted probes at this grasp configuration and sampled friction bands ONLY',
        'matched_physics_executed':False,'feasibility_collection_authorized':False}
    write('TRAINING_METRICS.json',metrics)
    sources=[ROOT/'current_contract_physical_belief.py',ROOT/'current_contract_belief_features.py',ADAPTER,LEGACY_SCHEMA,
             ROOT/'activeforcing_current_probe.py',ROOT/'activeforcing_probe_friction_contract.py']
    manifest={'feature_schema_id':SCHEMA_ID,'task0_candidate_qualified':qualified,'checkpoints':checkpoints,
        'normalization':normalization,'source_hashes':{**protocol['sha256'],**{str(p):sha(p) for p in sources}},
        'protocol_path':str(OUT/'PROTOCOL.json'),'protocol_sha256':sha(OUT/'PROTOCOL.json'),
        'selector_formula_changed':False,'posterior_interface_changed':False,'sigma_heads_integrated_by_planner':False,
        'scope':'task0 candidate, not four-task final model','physical_execution_authorized':False}
    write('PHYSICAL_BELIEF_CANDIDATE_MANIFEST.json',manifest)
    runtime=CurrentPhysicalBelief(OUT/'PHYSICAL_BELIEF_CANDIDATE_MANIFEST.json',diagnostic_only=True)
    parity=0.;diagnostic_result={}
    for split,(examples,pm,ps) in cached.items():
        for ex,m in zip(examples,pm):
            result=runtime.array(ex['raw']);parity=max(parity,float(np.max(np.abs(np.array(result['member_means'])-m))))
            if split=='DIAGNOSTIC':diagnostic_result[ex['band']]=result
    assert parity<3e-6
    write('TASK0_HIGH_MID_LOW_POSTERIORS.json',diagnostic_result)
    write('TRAIN_RUNTIME_PARITY.json',{'max_member_mean_diff':parity,'passed':True,'feature_parity':'exact',
        'context_count':sum(len(v[0]) for v in cached.values())})
    audit=[]
    for p in protocol['contexts']:
        job=Path(p['data_path']);r=json.loads((job/'RESULT.json').read_text())
        audit.append({'context_id':p['context_id'],'root_seed':p['root_seed'],'split':p['split'],'target_mu':p['hidden_friction_analysis_only'],
            'rows':r['steps'],'outward_steps':r['outward_steps'],'source':str(job),
            'sha256':{name:sha(job/name) for name in ['RAW_PROBE.csv','CONTACT_PATCH_READBACK.json','DECISION_STATE.pt']},
            'qualified':r.get('probe_qualified',True),'excluded_reasons':r.get('exclusion_reasons',[])})
    dataset_digest=hashlib.sha256(json.dumps(audit,sort_keys=True).encode()).hexdigest()
    write('DATASET_PROVENANCE.json',{'examples':audit,'old720_used':False,'feasibility_labels':0,
        'dataset_digest':dataset_digest,
        'root_leakage':False,'normalization_uses_test':False,'training_recipe_search':False})
    manifest.update(training_dataset_digest=dataset_digest,split_manifest_path=str(OUT/'SPLIT_MANIFEST.json'),
                    split_manifest_sha256=sha(OUT/'SPLIT_MANIFEST.json'))
    write('PHYSICAL_BELIEF_CANDIDATE_MANIFEST.json',manifest)
    print(json.dumps(clean({'metrics':{s:{k:metrics[s]['ensemble'][k] for k in ['MAE','RMSE','GAUSSIAN_MIXTURE_NLL','SPEARMAN']} for s in ['VAL','TEST']},
        'eligibility':metrics['eligibility'],'diagnostic_means':{k:v['mean'] for k,v in diagnostic_result.items()}}),indent=2))


if __name__=='__main__':main()
