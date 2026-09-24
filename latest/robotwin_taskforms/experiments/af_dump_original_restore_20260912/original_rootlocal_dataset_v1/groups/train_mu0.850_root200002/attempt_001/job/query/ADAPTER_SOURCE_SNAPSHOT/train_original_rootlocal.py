"""Original recipes on a declared one-root/new-task dataset, not legacy weights.

Dataset adapter only: complete query groups replace the historical multi-root
LIBERO registry. Models, losses, sampling, optimizers and selection stay original.
TEST is physically uncollected until both selection locks exist.
"""
import argparse
import ast
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
from scipy.stats import norm
from torch import nn
from rootlocal_collection_contract import HERE,FORTE,SNAPSHOT,read,write,sha,now,verify_runtime
from run_original_rootlocal_collection import valid_completion
from native_original_motion_features import TASK_BINDING
from native_original_evidence_adapter import original_evidence
from native_original_belief_binding import native_loader_module,load_native_belief,SOURCE as BELIEF_LOADER
from native_original_feasibility_binding import NativeOriginalFeasibility,phase_free_network_class,SOURCE as FEAS_RUNTIME
sys.path.insert(0,str(FORTE))
from current_contract_physical_belief import FrictionMember,batch,predict,verify_decision_prefix
from current_contract_belief_features import SCHEMA_ID

BELIEF_TRAINER=FORTE/'analysis/results/current_belief_training_prep_v2_20260905/train_current58.py'
FEAS_TRAINER=FORTE/'analysis/results/current_fulltask_feasibility_baseline_v1_20260906/train_current_fulltask_feasibility_baseline_20260906.py'


def recipe():
    node=next(n for n in ast.parse(BELIEF_TRAINER.read_text(encoding='utf-8')).body
              if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='RECIPE')
    return ast.literal_eval(node.value)


def load_dataset(dataset):
    dataset=Path(dataset).resolve();lock=read(dataset/'FREEZE_LOCK.json')
    if sha(dataset/'CONTEXTS.json')!=lock['contexts_sha256']:raise ValueError('Schedule changed')
    manifest=verify_runtime(dataset/'RUNTIME_MANIFEST.json',lock['runtime_sha256'])
    if sha(dataset/'COLLECTION_PROTOCOL.json')!=lock['protocol_sha256']:raise ValueError('Protocol changed')
    complete=read(dataset/'COLLECTION_COMPLETE.json')
    planned=[c for c in read(dataset/'CONTEXTS.json') if c['split'] in ('TRAIN','VAL')]
    entries=complete['contexts']
    if sorted(e['context']['id'] for e in entries)!=sorted(c['id'] for c in planned):raise ValueError('Incomplete groups')
    if complete['total_rows']!=128 or complete['test_groups_executed']!=0:raise ValueError('Wrong stage coverage')
    records=[];evidence_hash_splits=defaultdict(set);inputs={}
    for entry in entries:
        context=entry['context'];attempt=Path(entry['attempt']);job=attempt/'job'
        if context not in planned or not valid_completion(attempt,context):raise ValueError('Invalid group receipt')
        query=job/'query';raw=read(query/'original_raw_rows.json');patch=read(query/'patch_readbacks.json')
        verify_decision_prefix(raw);features=original_evidence(raw,patch)
        np.testing.assert_array_equal(features,np.load(query/'original58_engineering.npy',allow_pickle=False))
        admission=read(job/'QUERY_ADMISSION.json')
        if not admission['admitted'] or not admission['material_readback_required']:raise ValueError('Missing query admission')
        material=read(query/'ACTUAL_OBJECT_MATERIAL.json')['shape_materials']
        if not material or not np.allclose(material,context['friction'],atol=1e-6,rtol=0):raise ValueError('Wrong friction label')
        evidence_hash_splits[hashlib.sha256(features.tobytes()).hexdigest()].add(context['split'])
        rows=read(job/'COLLECTED_ROWS.json')
        if [r['force'] for r in rows]!=context['forces_N'] or len(rows)!=8:raise ValueError('Candidate coverage mismatch')
        for row in rows:
            if row['context_id']!=context['id'] or row['split']!=context['split'] or row['root']!=200002:
                raise ValueError('Branch split or root mismatch')
            if row['full_task_success_y'] not in (0,1):raise ValueError('Unknown outcome cannot be a label')
        feature=read(Path(rows[0]['job'])/'original_motion_feature.json')
        if feature['task_binding']!=TASK_BINDING or feature['candidate_actions_executed']!=0:raise ValueError('Feature binding failed')
        records.append({'id':context['id'],'root':200002,'split':context['split'],'y':context['friction'],
                        'raw':features,'query_raw':raw,'patch':patch,'branches':rows,'feature':feature,
                        'reference':str(job),'context':context})
        for p in [attempt/'PROCESS.json',attempt/'PROCESS_EXIT.json',job/'LOGICAL_COMPLETION.json',
                  job/'COLLECTED_ROWS.json',job/'QUERY_ADMISSION.json',query/'original_raw_rows.json',
                  query/'patch_readbacks.json',query/'original58_engineering.npy',query/'ACTUAL_OBJECT_MATERIAL.json']:
            inputs[str(p)]=sha(p)
        for row in rows:
            for name in ['result.json','original_motion_feature.json']:
                p=Path(row['job'])/name;inputs[str(p)]=sha(p)
    if any(len(s)>1 for s in evidence_hash_splits.values()):raise ValueError('Exact query evidence duplicated across splits')
    if sum(r['split']=='TRAIN' for r in records)!=12 or sum(r['split']=='VAL' for r in records)!=4:
        raise ValueError('Wrong independent query counts')
    return records,inputs,lock


def frozen_training_protocol(dataset,out,records,inputs,lock):
    out.mkdir(parents=True,exist_ok=False)
    protocol={'version':'DUMP_ROOTLOCAL_ORIGINAL_RECIPES_V1','created_utc':now(),
              'dataset':str(dataset),'collection_lock':lock,'input_hashes':inputs,
              'native_task_binding':TASK_BINDING,'root_scope':[200002],
              'split_override':'user requested within root; complete friction/query groups remain disjoint',
              'belief_recipe':recipe(),
              'belief_single_root_bootstrap_degenerates_to_all_train_contexts':True,
              'feasibility_recipe':{'seeds':[0,1,2],'epochs':80,'batch':64,'lr':.0008,'weight_decay':.0001,
                 'gradient_clip':1.,'loss':'BCE logits','selection':'first minimum VAL posterior marginal raw NLL',
                 'shape':[8,64],'normalization':'TRAIN population mean/std; std<1e-6 replaced by1'},
              'force_support':[.5,8.],'planner_step':.05,'utility_normalization_N':5.,
              'test_access_before_both_checkpoint_locks':False,
              'source_hashes':{str(p):sha(p) for p in [Path(__file__).resolve(),BELIEF_TRAINER,FEAS_TRAINER,
                  BELIEF_LOADER,FEAS_RUNTIME,SNAPSHOT/'phase_free_feasibility.py',
                  HERE/'native_original_belief_binding.py',HERE/'native_original_feasibility_binding.py']}}
    write(out/'TRAINING_PROTOCOL.json',protocol)
    write(out/'TRAINING_PROTOCOL_LOCK.json',{'sha256':sha(out/'TRAINING_PROTOCOL.json')})
    return out/'TRAINING_PROTOCOL.json'


def train_belief(records,out,protocol,runtime_hash):
    out.mkdir()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    joined=np.concatenate([r['raw'] for r in records if r['split']=='TRAIN'])
    mean=joined.mean(0);std=np.maximum(joined.std(0),1e-6)
    normalization={'mean':mean.tolist(),'std':std.tolist(),'fit_split':'TRAIN','fit_unpadded_steps':len(joined),
                   'fit_roots':[200002],'all_channels':True,'axes':'sample and real time steps; exclude padding','std_floor':1e-6}
    write(out/'NORMALIZATION.json',normalization)
    for r in records:r['x']=((r['raw']-mean)/std).astype(np.float32)
    training=[r for r in records if r['split']=='TRAIN'];validation=[r for r in records if r['split']=='VAL']
    groups={200002:training};checkpoints=[];history=[];models=[]
    for seed in (0,1,2):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=FrictionMember();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        rng=np.random.default_rng(seed);best=float('inf');state=None;bestepoch=None
        for epoch in range(1,81):
            ids=rng.choice(list(groups),size=len(groups),replace=True);fit=[r for root in ids for r in groups[root]]
            x,lengths=batch([r['x'] for r in fit]);y=torch.tensor([r['y'] for r in fit],dtype=torch.float32)
            model.train();opt.zero_grad(set_to_none=True);mu,ls=model(x,lengths)
            loss=(.5*(((y-mu)/ls.exp())**2+2*ls)).mean()+.05*torch.abs(mu-y).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            vm,vs=predict(model,[r['x'] for r in validation]);vy=np.array([r['y'] for r in validation])
            vnll=float(-norm.logpdf(vy,vm,vs).mean())
            if not np.isfinite(vnll):raise ValueError('Nonfinite validation NLL')
            history.append({'seed':seed,'update':epoch,'loss':float(loss.detach()),'val_nll':vnll,'sampled_roots':ids.tolist()})
            if vnll<best:best=vnll;bestepoch=epoch;state={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(state);model.eval();models.append(model)
        path=out/f'CURRENT_MULTITASK58_seed{seed}.pt'
        with path.open('xb') as stream:torch.save({'state_dict':state,'input_dim':58,'projection_dim':16,'hidden_dim':16,
            'seed':seed,'normalization':normalization,'feature_schema_id':SCHEMA_ID,'selected_epoch':bestepoch,
            'selected_val_nll':best,'optimizer_updates_total':80,'native_task_binding':TASK_BINDING,
            'root_scope':[200002],'training_protocol_sha256':sha(protocol)},stream)
        checkpoints.append({'path':str(path),'sha256':sha(path),'seed':seed,'selected_epoch':bestepoch,'val_nll':best})
    write(out/'CHECKPOINT_SELECTION_LOCK.json',{'checkpoints':checkpoints,'rule':recipe()['checkpoint_selection'],
                                              'test_labels_used':False,'protocol_sha256':sha(protocol)})
    write(out/'TRAINING_HISTORY.json',history)
    module=native_loader_module()
    sources=[BELIEF_LOADER,module.POSITIVE,FORTE/'current_contract_belief_features.py',
             FORTE/'current_contract_physical_belief.py',HERE/'native_original_belief_binding.py']
    manifest={'interface':module.INTERFACE,'feature_dim':58,'feature_schema_id':SCHEMA_ID,
              'qualified_for_current_runtime':True,'qualified_tasks':['dump_bin_bigbin'],
              'native_task_binding':TASK_BINDING,'native_loader_binding':module.native_binding_receipt,
              'root_scope':[200002],'normalization':normalization,'checkpoints':checkpoints,
              'source_hashes':{str(p):sha(p) for p in sources},'compatible_runtime_manifest_sha256':[runtime_hash],
              'qualification_scope':'sensor/schema/decision-prefix and exact checkpoint reload, not an accuracy guarantee',
              'training_protocol_sha256':sha(protocol)}
    # Write candidate separately; final deployment manifest only after real
    # TRAIN/VAL feature, member-prediction and quadrature reload parity below.
    manifest['qualified_for_current_runtime']=False
    write(out/'CANDIDATE_MANIFEST.json',manifest)
    diagnostic=module.ContinuousBelief(out/'CANDIDATE_MANIFEST.json',
        manifest_sha256=sha(out/'CANDIDATE_MANIFEST.json'),diagnostic_only=True)
    receipt=[]
    for record in records:
        value=diagnostic.rows(record['query_raw'],record['patch'],runtime_manifest_sha256=runtime_hash)
        np.testing.assert_array_equal(value['raw_features'],record['raw'])
        np.testing.assert_array_equal(value['normalized_features'],record['x'])
        for i,model in enumerate(models):
            mu,sigma=predict(model,[record['x']])
            if value['member_means'][i]!=float(mu[0]) or value['member_log_sigmas'][i]!=float(np.log(sigma[0])):
                raise ValueError('Original member reload parity failed')
        receipt.append({'id':record['id'],'feature_member_reload_exact':True,'integration_qa':value['integration_qa']})
    write(out/'RUNTIME_RELOAD_QUALIFICATION.json',receipt)
    manifest['qualified_for_current_runtime']=True
    write(out/'BELIEF_MANIFEST.json',manifest)
    belief=load_native_belief(out/'BELIEF_MANIFEST.json',manifest_sha256=sha(out/'BELIEF_MANIFEST.json'))
    for record in records:
        value=belief.rows(record['query_raw'],record['patch'],runtime_manifest_sha256=runtime_hash)
        value.update(candidate_actions_executed=0)
        record['posterior']=value
        serial={k:v for k,v in value.items() if k not in ('continuous_posterior','raw_features','normalized_features')}
        serial={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in serial.items()}
        write(out/(record['id']+'_POSTERIOR.json'),serial)
    return belief


def feas_rows(records,split):
    rows=[]
    for record in records:
        if record['split']!=split:continue
        base=np.asarray(record['feature']['sequence'],np.float32)
        for branch in record['branches']:
            x=base.copy();x[:,10]=float(branch['force'])/8.;x[:,11]=record['y']
            rows.append({'record':record,'branch':branch,'x':x,'y':float(branch['full_task_success_y'])})
    return rows


def posterior_predictions(models,rows,mean,std,device):
    by_context=defaultdict(list)
    for row in rows:by_context[row['record']['id']].append(row)
    predictions={}
    for cid,group in by_context.items():
        record=group[0]['record'];p=record['posterior']
        nodes=np.asarray(p['integration_nodes'],np.float32);weights=np.asarray(p['integration_weights'],float)
        forces=np.asarray([r['branch']['force'] for r in group],np.float32)
        force=np.repeat(forces,len(nodes));mu=np.tile(nodes,len(forces))
        base=np.asarray(record['feature']['sequence'],np.float32)
        x=np.broadcast_to(base,(len(force),)+base.shape).copy();x[:,:,10]=force[:,None]/8.;x[:,:,11]=mu[:,None]
        x=(x-mean[None,None])/std[None,None]
        values=[]
        for model in models:
            model.eval();chunks=[]
            with torch.no_grad():
                for start in range(0,len(x),4096):
                    v=torch.as_tensor(x[start:start+4096],dtype=torch.float32,device=device)
                    chunks.append(torch.sigmoid(model(v[:,:,:10],v[:,0,10:])).cpu().numpy())
            values.append(np.concatenate(chunks))
        marginal=np.mean(values,axis=0).reshape(len(forces),len(nodes))@weights
        for row,prob in zip(group,marginal):predictions[cid,row['branch']['force']]=float(prob)
    return np.asarray([predictions[r['record']['id'],r['branch']['force']] for r in rows])


def train_feasibility(records,out,protocol):
    out.mkdir();device=torch.device('cuda')
    if not torch.cuda.is_available():raise RuntimeError('Original feasibility trainer requires CUDA')
    # Historical trainers were separate processes. Do not accidentally inherit
    # belief trainer's CPU deterministic-algorithm global into this recipe.
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(4)
    training=feas_rows(records,'TRAIN');validation=feas_rows(records,'VAL')
    values=np.stack([r['x'] for r in training]);mean=values.mean(axis=(0,1)).astype(np.float32)
    std=values.std(axis=(0,1)).astype(np.float32);std[std<1e-6]=1.
    x=(values-mean[None,None])/std[None,None]
    tr_step=torch.as_tensor(x[:,:,:10],device=device);tr_cond=torch.as_tensor(x[:,0,10:],device=device)
    tr_y=torch.as_tensor([r['y'] for r in training],dtype=torch.float32,device=device)
    vy=np.asarray([r['y'] for r in validation]);checkpoints=[];histories=[];models=[]
    for seed in (0,1,2):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=phase_free_network_class()().to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
        rng=np.random.default_rng(seed);best=None
        for epoch in range(1,81):
            model.train();permutation=rng.permutation(len(training));losses=[]
            for start in range(0,len(permutation),64):
                index=torch.as_tensor(permutation[start:start+64],device=device)
                optimizer.zero_grad(set_to_none=True)
                loss=nn.functional.binary_cross_entropy_with_logits(model(tr_step[index],tr_cond[index]),tr_y[index])
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                losses.append(float(loss.detach().cpu()))
            p=np.clip(posterior_predictions([model],validation,mean,std,device),1e-7,1-1e-7)
            vnll=float(np.mean(-(vy*np.log(p)+(1-vy)*np.log(1-p))))
            if not np.isfinite(vnll):raise ValueError('Nonfinite posterior validation NLL')
            histories.append({'seed':seed,'epoch':epoch,'train_minibatch_bce':float(np.mean(losses)),
                              'val_posterior_nll':vnll})
            if best is None or vnll<best['val_nll']:
                best={'epoch':epoch,'val_nll':vnll,'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
        model.load_state_dict(best['state']);model.eval();models.append(model)
        path=out/f'CURRENT_FULLTASK_FEAS_seed{seed}.pt'
        with path.open('xb') as stream:torch.save({'state_dict':best['state'],'seed':seed,'selected_epoch':best['epoch'],
            'validation_posterior_nll':best['val_nll'],'normalization_mean':mean.tolist(),'normalization_std':std.tolist(),
            'target':'full_task_success_y','protocol_sha256':sha(protocol),'native_task_binding':TASK_BINDING,
            'root_scope':[200002],'feature_shape':[8,64],'normalization_fit_split':'TRAIN'},stream)
        checkpoints.append({'seed':seed,'path':str(path),'sha256':sha(path),'selected_epoch':best['epoch'],
                            'validation_posterior_nll':best['val_nll']})
        print('ORIGINAL_FEASIBILITY_SEED_SELECTED '+json.dumps(checkpoints[-1]),flush=True)
    write(out/'CHECKPOINT_SELECTION_LOCK.json',{'checkpoints':checkpoints,'test_labels_accessed_before_lock':False,
                                             'protocol_sha256':sha(protocol)})
    write(out/'TRAINING_HISTORY.json',histories)
    manifest={'native_task_binding':TASK_BINDING,'root_scope':[200002],'feature_shape':[8,64],
              'label_target':'full_task_success_y','calibration':'NONE_RAW',
              'posterior_interface':'CURRENT_MULTITASK58_SIGMA_AWARE_POSITIVE_MIXTURE_V1','sigma_used':True,
              'utility_normalization_N':5.,'planner_grid_step':.05,'force_support':[.5,8.],
              'source_hashes':{str(p):sha(p) for p in [FEAS_RUNTIME,SNAPSHOT/'phase_free_feasibility.py',
                                                    HERE/'native_original_feasibility_binding.py']},
              'checkpoints':checkpoints,'training_protocol_sha256':sha(protocol)}
    write(out/'FEASIBILITY_MANIFEST.json',manifest)
    runtime=NativeOriginalFeasibility(out/'FEASIBILITY_MANIFEST.json',manifest_sha256=sha(out/'FEASIBILITY_MANIFEST.json'),device='cuda')
    expected=posterior_predictions(models,validation,mean,std,device);actual=[];decisions=[]
    for record in records:
        decision=runtime.select(record['feature'],record['posterior'])
        decisions.append({'id':record['id'],'split':record['split'],**decision})
        if record['split']=='VAL':
            grid=np.asarray(decision['force_grid_N']);curve=np.asarray(decision['p_success'])
            for branch in record['branches']:
                index=np.flatnonzero(np.isclose(grid,branch['force'],atol=1e-8,rtol=0))
                if len(index)!=1:raise ValueError('Collected force absent from continuous grid')
                actual.append(float(curve[index[0]]))
    error=float(np.max(np.abs(expected-np.asarray(actual))))
    if error>1e-6:raise ValueError('Posterior runtime/checkpoint reload mismatch: '+str(error))
    write(out/'RUNTIME_RELOAD_QUALIFICATION.json',{'passed':True,'max_probability_error':error,
        'tested_real_VAL_branches':len(actual),'positive_mixture_sigma_used':True,'utility_normalization_N':5.})
    write(out/'TRAIN_VAL_DECISIONS.json',decisions)


def train(dataset,out):
    dataset=Path(dataset).resolve();out=Path(out).resolve()
    records,inputs,lock=load_dataset(dataset)
    protocol=frozen_training_protocol(dataset,out,records,inputs,lock)
    train_belief(records,out/'belief',protocol,lock['runtime_sha256'])
    train_feasibility(records,out/'feasibility',protocol)
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError('Dataset changed during training')
    write(out/'TRAINING_COMPLETE.json',{'completed':True,'protocol_sha256':sha(protocol),
        'belief_manifest_sha256':sha(out/'belief/BELIEF_MANIFEST.json'),
        'feasibility_manifest_sha256':sha(out/'feasibility/FEASIBILITY_MANIFEST.json'),
        'test_groups_executed':0,'final_online_confirmation_still_required':True,'finished_utc':now()})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('dataset',type=Path);ap.add_argument('out',type=Path)
    args=ap.parse_args();train(args.dataset,args.out)
