"""Offline-only 647-row phase intervention; never imports simulator runtime."""
import sys, json, random, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0, '/home/exouser/FORTE')
import train_current_fulltask_feasibility_baseline_20260906 as t

HERE = Path(__file__).resolve().parent
OUT = HERE / 'offline_feature_audit_v3'
KEEP = list(range(6)) + list(range(13,17))

class Removed(t.FeasibilityOnly):
    def __init__(self):
        super().__init__()
        self.command_gru = nn.GRU(10,64,batch_first=True)
    def forward(self, step, cond):
        return super().forward(step[:,:,KEEP],cond)

def write(name, value):
    t.write(OUT/name,value)

def intervention(base, mode, mean, key):
    x=base.copy()
    if mode=='PHASE_MASKED':
        # Missingness is external metadata. Legacy model has NO phase-mask input.
        # Neutral imputation removes all normalized phase contribution.
        x[:,6:13]=mean[6:13]
    elif mode=='PHASE_ZEROED': x[:,6:13]=0
    elif mode=='PHASE_PERMUTED':
        # Actual permutation among examples/timesteps is identically unchanged:
        # asserted globally before evaluation. Do not invent varying labels.
        pass
    elif mode=='PHASE_CHANNEL_SHUFFLED_DIAGNOSTIC':
        seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
        perm=np.random.default_rng(seed).permutation(7)
        x[:,6:13]=x[:,6:13][:,perm]
    return x

def prepare(rows,mean,std,mode,planner=False):
    groups={}
    for r in rows:groups.setdefault(r['row']['context_id'],[]).append(r)
    batches=[]; specs=[]; offset=0
    for cid, group in sorted(groups.items()):
        ref=Path(group[0]['row']['reference']);post=t.load(ref/'PREACTION_POSTERIOR.json')
        base=intervention(np.load(ref/'PREACTION_SEQUENCE.npy'),mode,mean,cid)
        nodes=np.asarray(post['integration_nodes'],np.float32); weights=np.asarray(post['integration_weights'])
        assert np.isclose(weights.sum(),1)
        forces=t.FORCE_GRID if planner else np.array([r['row']['force'] for r in group])
        n=len(forces)*len(nodes)
        step=(base[:,:17]-mean[:17])/std[:17]
        cond=np.broadcast_to(base[0,17:],(n,54)).copy()
        cond[:,0]=np.repeat(forces,len(nodes))/8;cond[:,1]=np.tile(nodes,len(forces))
        cond=(cond-mean[17:])/std[17:]
        batches.append((step,cond));specs.append((cid,group,forces,weights,offset,n));offset+=n
    return batches,specs

def predict(models,prepared):
    x,specs=prepared
    # All force/mu combinations in a context share one sequence. Evaluate that
    # GRU once, then broadcast its hidden state to the condition-only branches.
    # This is an algebraic factorization, not a changed model or approximation.
    per_model=[]
    with torch.no_grad():
        for m in models:
            m.eval();outputs=[]
            for (step_array,cond_array),(_,_,_,_,start,n) in zip(x,specs):
                step=torch.as_tensor(step_array[None],dtype=torch.float32)
                cond=torch.as_tensor(cond_array,dtype=torch.float32)
                if isinstance(m,Removed):step=step[:,:,KEEP]
                _,h=m.command_gru(step)
                out=m.head(torch.cat([h[-1].expand(n,-1),m.condition(cond)],dim=-1)).squeeze(-1)
                outputs.append(torch.sigmoid(out).numpy())
            per_model.append(np.concatenate(outputs))
    p=np.mean(per_model,axis=0)
    results=[]
    for cid,group,forces,w,start,n in specs:
        values=p[start:start+n].reshape(len(forces),len(w))@w
        results.append((cid,group,forces,values))
    return results

def evaluate(models,rows,mean,std,mode):
    results={}; predictions=[]; plans=[]
    for split,records in rows.items():
        cooked=predict(models,prepare(records,mean,std,mode))
        ys=[];ps=[]
        for cid,group,forces,probs in cooked:
            for r,f,p in zip(group,forces,probs):
                ys.append(r['y']);ps.append(float(p))
                predictions.append(dict(split=split,context_id=cid,force=float(f),y=r['y'],p=float(p)))
        results[split]=t.metric(ys,ps)
        # All 72 contexts reported; heldout aggregation remains separate.
        for cid,group,forces,p in predict(models,prepare(records,mean,std,mode,True)):
            u=p*(5-forces)/5+(1-p)*-1; k=int(np.argmax(u));r=group[0]['row']
            plans.append(dict(split=split,context_id=cid,task=r['task'],root=r['root'],band=r['band'],
                selected_force=float(forces[k]),p3=float(p[0]),p5=float(p[-1]),curve=p.tolist()))
    def summary(v):
        f=np.array([r['selected_force'] for r in v]); groups={}
        for r in v:groups.setdefault((r['task'],r['root']),[]).append(r['selected_force'])
        return dict(contexts=len(v),mean_selected_force=float(f.mean()),lower_bound_selection_rate=float(np.mean(f==3)),
            context_sensitive_force_rate=float(np.mean([np.ptp(g)>.001 for g in groups.values()])),
            task0_by_band={b:[dict(root=r['root'],force=r['selected_force']) for r in v if r['task']==0 and r['band']==b] for b in ('LOW','MID','HIGH')})
    result=dict(metrics=results,planner_all=summary(plans),planner_heldout=summary([r for r in plans if r['split']!='TRAIN']))
    write(mode+'_PREDICTIONS.json',predictions);write(mode+'_PLANNER.json',plans)
    write(mode+'_RESULT.json',result)
    print(mode,json.dumps(result['metrics']),flush=True)
    return result,predictions,plans

def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    rows={s:t.load_rows(s) for s in ('TRAIN','VAL','TEST')}
    assert sum(map(len,rows.values()))==647
    allx=np.stack([r['x'] for rs in rows.values() for r in rs])
    assert np.array_equal(np.unique(allx[:,:,6:13].reshape(-1,7),axis=0),[[1,0,0,0,0,0,0]])
    checkpoints=[t.OUT/f'CURRENT_FULLTASK_FEAS_seed{s}.pt' for s in t.SEEDS]
    states=[torch.load(p,map_location='cpu',weights_only=False) for p in checkpoints]
    mean=states[0]['normalization_mean'];std=states[0]['normalization_std']
    assert np.array_equal(mean[6:13],[1,0,0,0,0,0,0])
    models=[];removed=[]
    for p in states:
        m=t.FeasibilityOnly();m.load_state_dict(p['state_dict']);m.eval();models.append(m)
        r=Removed();sd={k:v.clone() for k,v in p['state_dict'].items()}
        sd['command_gru.weight_ih_l0']=sd['command_gru.weight_ih_l0'][:,KEEP]
        r.load_state_dict(sd);r.eval();removed.append(r)
    protocol=dict(created_utc=t.now(),source_sha256=t.sha(__file__),new_physics_runs=0,
        source_checkpoints={str(p):t.sha(p) for p in checkpoints},rows_by_split={k:len(v) for k,v in rows.items()},
        dataset_row_hashes={str(p):t.sha(p) for s in rows for p in t.row_paths(s)},
        phase_unique_vectors=[[1,0,0,0,0,0,0]],phase_constant=True,
        phase_masked='External missing mask plus TRAIN-mean imputation; legacy architecture has no phase-mask input. Normalized phase is zero.',
        phase_permuted='Seeded sample/time permutation is mathematically a no-op because all phase vectors are identical.',
        additional_diagnostic='Seeded channel permutation is separately named; not confused with empirical permutation importance.',
        removed_exact='Delete seven normalized-constant-zero GRU input columns. Same existing weights elsewhere; no training.',
        retrain=dict(seeds=list(t.SEEDS),epochs=t.EPOCHS,batch=t.BATCH,lr=t.LR,weight_decay=t.WEIGHT_DECAY,
            optimizer='AdamW',gradient_clip=1,selection='lowest VAL posterior-marginalized NLL',
            device='CPU',original_device='CUDA',hidden_capacity_unchanged=True,
            original_parameter_count=sum(p.numel() for p in models[0].parameters()),
            removed_parameter_count=sum(p.numel() for p in removed[0].parameters()),
            capacity_caveat='Deleting 7 GRU input columns necessarily removes 1344 parameters; recurrent hidden, condition and head sizes unchanged. Exact-removal arm isolates phase without optimization confound.',
            test_use='Previously exposed old scripted TEST. Never used for checkpoint selection; not fresh VLA evidence.'))
    write('PROTOCOL.json',protocol)
    check=prepare(rows['VAL'][:1],mean,std,'ORIGINAL_PHASE')
    factored=predict(models,check)[0][3]
    step,cond=check[0][0]
    expanded=np.concatenate([np.broadcast_to(step,(len(cond),8,17)),np.broadcast_to(cond[:,None,:],(len(cond),8,54))],axis=-1)
    direct=np.mean([t.model_probabilities(m,expanded,'cpu') for m in models],axis=0).reshape(1,-1)@check[1][0][3]
    assert np.max(np.abs(factored-direct))<1e-6
    write('FACTORIZED_INFERENCE_PARITY.json',dict(max_probability_error=float(np.max(np.abs(factored-direct))),passed=True))
    write('CHANNEL_STATISTICS.json',[dict(index=i,minimum=float(allx[:,:,i].min()),maximum=float(allx[:,:,i].max()),
        train_mean=float(mean[i]),train_std=float(std[i]),constant=bool(np.ptp(allx[:,:,i])==0)) for i in range(71)])
    results={};full={}
    for mode in ('ORIGINAL_PHASE','PHASE_MASKED','PHASE_ZEROED','PHASE_PERMUTED','PHASE_CHANNEL_SHUFFLED_DIAGNOSTIC','PHASE_REMOVED_EXACT'):
        result,pred,plan=evaluate(removed if mode=='PHASE_REMOVED_EXACT' else models,rows,mean,std,mode)
        results[mode]=result;full[mode]=(pred,plan)
    # Cache posterior-marginalized validation batches once; identical original selection criterion.
    val=prepare(rows['VAL'],mean,std,'PHASE_REMOVED_RETRAIN')
    tr_step,tr_cond,tr_y=t.tensors(rows['TRAIN'],mean,std,'cpu')
    histories=[];trained=[];selected=[]
    for seed in t.SEEDS:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=Removed();optimizer=torch.optim.AdamW(model.parameters(),lr=t.LR,weight_decay=t.WEIGHT_DECAY)
        rng=np.random.default_rng(seed);best=None
        for epoch in range(1,t.EPOCHS+1):
            model.train();perm=rng.permutation(len(tr_y));losses=[]
            for start in range(0,len(perm),t.BATCH):
                ix=torch.as_tensor(perm[start:start+t.BATCH]);optimizer.zero_grad(set_to_none=True)
                loss=nn.functional.binary_cross_entropy_with_logits(model(tr_step[ix],tr_cond[ix]),tr_y[ix]);loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
            ys=[];ps=[]
            for _,group,_,p in predict([model],val):ys.extend(r['y'] for r in group);ps.extend(p)
            nll=t.metric(ys,ps)['nll'];histories.append(dict(seed=seed,epoch=epoch,train_bce=float(np.mean(losses)),val_nll=nll))
            if best is None or nll<best['val_nll']:
                best=dict(epoch=epoch,val_nll=nll,state={k:v.detach().clone() for k,v in model.state_dict().items()})
        model.load_state_dict(best['state']);model.eval();trained.append(model)
        path=OUT/f'PHASE_REMOVED_RETRAIN_seed{seed}.pt'
        torch.save(dict(state_dict=best.pop('state'),normalization_mean=mean,normalization_std=std,keep_sequence_indices=KEEP,seed=seed,**best),path)
        selected.append(dict(seed=seed,path=str(path),sha256=t.sha(path),**best))
        print('retrained',seed,best,flush=True)
    write('RETRAIN_SELECTION_LOCK.json',selected);write('RETRAIN_HISTORY.json',histories)
    result,pred,plan=evaluate(trained,rows,mean,std,'PHASE_REMOVED_RETRAIN')
    results['PHASE_REMOVED_RETRAIN']=result;full['PHASE_REMOVED_RETRAIN']=(pred,plan)
    baseline_p=np.array([r['p'] for r in full['ORIGINAL_PHASE'][0]])
    baseline_f=np.array([r['selected_force'] for r in full['ORIGINAL_PHASE'][1]])
    for mode,(p,f) in full.items():
        results[mode]['vs_original']=dict(max_probability_change=float(np.max(np.abs(np.array([r['p'] for r in p])-baseline_p))),
            planner_changed_fraction=float(np.mean(np.abs(np.array([r['selected_force'] for r in f])-baseline_f)>.001)))
    exact=results['PHASE_REMOVED_EXACT']['vs_original']
    assert exact['max_probability_change']<1e-6 and exact['planner_changed_fraction']==0
    write('PHASE_SENSITIVITY_SUMMARY.json',dict(results=results,PHASE_DEPENDENCE_STRONG='NO',
        PHASE_REQUIRED_FOR_CURRENT_FEASIBILITY='NO',
        conclusion='Phase contains no information in all 647 rows. Exact normalized-input column removal preserves existing model predictions and planner. Raw-zero OOD perturbation is not evidence phase has learned semantic value.',
        FINAL_PHASE_REPRESENTATION='NONE; remove seven phase channels',PHASE_PROXY_REQUIRED=False,
        limits='Does not establish online VLA transfer. Six motion channels were nearly constant in scripted training; live chunk motion is a separate coverage shift.'))

if __name__=='__main__':main()
