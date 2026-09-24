#!/usr/bin/env python3
"""Retrain the existing ActiveForcing networks on repeated, context-balanced rows."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from torch import nn
from train_af_taskforms_models import BeliefNet, FeasibilityNet, query_sequence, task_onehot, fit_normalization, sha256, binary_metrics

FORCES=(3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0)

def read(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def belief_rows(rows):
    out={}
    for r in rows:
        if r['valid']: out.setdefault((r['task'],r['root_slot'],float(r['friction'])),r)
    return list(out.values())

def belief_batch(rows,device):
    from torch.nn.utils.rnn import pad_sequence
    seq=[torch.tensor(query_sequence(r),dtype=torch.float32) for r in rows]
    return pad_sequence(seq,batch_first=True).to(device),torch.tensor([len(x) for x in seq],dtype=torch.long).to(device),torch.tensor([r['friction'] for r in rows],dtype=torch.float32).to(device)

def train_belief_model(rows,out,device):
    br=belief_rows(rows); by={s:[r for r in br if r['split']==s] for s in ('train','validation','test')}; models=[]; report={'counts':{s:len(v) for s,v in by.items()},'members':[]}
    for seed in (41,42,43):
        torch.manual_seed(seed); np.random.seed(seed); model=BeliefNet().to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); tr=belief_batch(by['train'],device); va=belief_batch(by['validation'],device); best=None; best_mae=float('inf'); patience=0
        for epoch in range(1500):
            model.train(); opt.zero_grad(); loss=nn.functional.mse_loss(model(*tr[:2]),tr[2]); loss.backward(); opt.step()
            if epoch%10==0:
                model.eval();
                with torch.no_grad(): mae=float(torch.mean(torch.abs(model(*va[:2])-va[2])))
                if mae<best_mae-1e-6: best_mae=mae; best={k:v.detach().cpu() for k,v in model.state_dict().items()}; patience=0
                else: patience+=1
                if patience>=30: break
        model.load_state_dict(best); model.eval(); ck=out/f'belief_member_{seed}.pt'; torch.save({'state_dict':best,'input_dim':15},ck); report['members'].append({'seed':seed,'best_validation_mae':best_mae,'checkpoint':str(ck),'sha256':sha256(ck)}); models.append(model.cpu())
    return models,report

def feas_batch(rows,norm,device):
    acts=np.asarray([r['evidence']['preaction_sequence'] for r in rows],dtype=np.float32); acts=(acts-norm['action_mean'])/norm['action_std']; states=np.asarray([r['evidence']['preaction_state'] for r in rows],dtype=np.float32); states=(states-norm['state_mean'])/norm['state_std']; cond=np.asarray([list(states[i])+[r['friction'],r['force_n']/5.0]+task_onehot(r['task']) for i,r in enumerate(rows)],dtype=np.float32); y=np.asarray([bool(r['evidence']['success']) for r in rows],dtype=np.float32); return torch.tensor(acts,device=device),torch.tensor(cond,device=device),torch.tensor(y,device=device)

def train_feas(rows,out,device):
    valid=[r for r in rows if r['valid'] and r['evidence'].get('preaction_sequence') is not None and r['evidence'].get('preaction_state') is not None]; by={s:[r for r in valid if r['split']==s] for s in ('train','validation','test')}; norm=fit_normalization(valid); report={'counts':{s:len(v) for s,v in by.items()},'context_balanced':True,'monotone_regularization_weight':0.0,'members':[]}; models=[]
    # Each context contributes unit total weight, so repeats do not dominate.
    ctx_counts=defaultdict(int)
    for r in by['train']: ctx_counts[(r['task'],r['root_slot'],float(r['friction']))]+=1
    for seed in (71,72,73):
        torch.manual_seed(seed); model=FeasibilityNet().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); tr=feas_batch(by['train'],norm,device); va=feas_batch(by['validation'],norm,device); weights=torch.tensor([1.0/ctx_counts[(r['task'],r['root_slot'],float(r['friction']))] for r in by['train']],device=device); best=None; best_brier=float('inf'); patience=0
        for epoch in range(1200):
            model.train(); opt.zero_grad(); logits=model(tr[0],tr[1]); per=nn.functional.binary_cross_entropy_with_logits(logits,tr[2],reduction='none'); loss=(per*weights).sum()/max(weights.sum(),1.0); loss.backward(); opt.step()
            if epoch%10==0:
                model.eval();
                with torch.no_grad(): p=torch.sigmoid(model(va[0],va[1])); brier=float(torch.mean((p-va[2])**2))
                if brier<best_brier-1e-6: best_brier=brier; best={k:v.detach().cpu() for k,v in model.state_dict().items()}; patience=0
                else: patience+=1
                if patience>=40: break
        model.load_state_dict(best); model.eval(); ck=out/f'feasibility_member_{seed}.pt'; torch.save({'state_dict':best,'force_input':'continuous_requested_force_n_divided_by_5','force_support_n':list(FORCES),'monotone_weight':0.0},ck); member={'seed':seed,'best_validation_brier':best_brier,'checkpoint':str(ck),'sha256':sha256(ck)}
        for s in ('train','validation','test'):
            b=feas_batch(by[s],norm,device)
            with torch.no_grad(): p=torch.sigmoid(model(b[0],b[1])).cpu().numpy()
            member[s]=binary_metrics(p,b[2].cpu().numpy())
        report['members'].append(member); models.append(model.cpu())
    report['normalization']={k:v.tolist() for k,v in norm.items()}; return models,report

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--records',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--device',default='cpu'); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); rows=read(a.records); bm,br=train_belief_model(rows,a.out,a.device); fm,fr=train_feas(rows,a.out,a.device); report={'schema_id':'AF_FORCE_SUPERVISION_REPAIR_TRAINING_V1','records':str(a.records),'records_sha256':sha256(a.records),'force_support_n':list(FORCES),'force_conditioning':'continuous scalar','tasks':sorted({r['task'] for r in rows}),'belief':br,'feasibility':fr,'root_disjoint':True,'test_used_for_model_selection':False,'repeated_supervision':True,'monotonic_regularization_disabled':True}; (a.out/'training_summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); (a.out/'training_summary.md').write_text('# Repaired ActiveForcing training\n\n- Existing architecture retained.\n- Repeated rows use context-balanced feasibility loss.\n- Monotonic regularization disabled for diagnosis.\n- Test outcomes were not used for model selection.\n\n```json\n'+json.dumps({'counts':fr['counts'],'belief_counts':br['counts']},indent=2)+'\n```\n'); print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__': main()
