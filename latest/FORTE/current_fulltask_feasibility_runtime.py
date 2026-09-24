"""Frozen current-contract full-task feasibility runtime and EU selector."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn

DEFAULT_MANIFEST=Path('/home/exouser/FORTE/analysis/results/current_fulltask_feasibility_runtime_freeze_v2_20260906/FINAL_FEASIBILITY_RUNTIME_MANIFEST.json')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

class Network(nn.Module):
    def __init__(self):
        super().__init__();self.command_gru=nn.GRU(17,64,batch_first=True)
        self.condition=nn.Sequential(nn.Linear(54,64),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,step,cond):
        _,h=self.command_gru(step);return self.head(torch.cat([h[-1],self.condition(cond)],-1)).squeeze(-1)

class CurrentFeasibility:
    def __init__(self,manifest=DEFAULT_MANIFEST,device='cpu'):
        self.manifest_path=Path(manifest);self.manifest=read(self.manifest_path)
        m=self.manifest
        if m['version']!='CURRENT_FULLTASK_FEASIBILITY_RUNTIME_V2' or not m['runtime_reload_parity']:
            raise ValueError('Unqualified feasibility runtime manifest')
        if m['label_target']!='full_task_success_y' or m['calibration']!='NONE_RAW':raise ValueError('Unexpected model contract')
        if m['posterior_interface']!='CURRENT_MULTITASK58_SIGMA_AWARE_POSITIVE_MIXTURE_V1' or not m['sigma_used']:
            raise ValueError('Wrong posterior interface')
        if m['force_support']!=[3.0,5.0] or m['planner_grid_step']!=.05:raise ValueError('Force support changed')
        self.device=torch.device(device);self.models=[];means=[];stds=[]
        for item in m['checkpoints']:
            path=Path(item['path'])
            if sha(path)!=item['sha256']:raise ValueError('Checkpoint changed: '+str(path))
            p=torch.load(path,map_location='cpu',weights_only=False)
            if p['protocol_sha256']!=m['training_protocol_sha256'] or p['target']!='full_task_success_y':raise ValueError('Checkpoint provenance mismatch')
            model=Network();model.load_state_dict(p['state_dict'],strict=True);model.to(self.device).eval();self.models.append(model)
            means.append(np.asarray(p['normalization_mean'],np.float32));stds.append(np.asarray(p['normalization_std'],np.float32))
        if not all(np.array_equal(means[0],x) for x in means[1:]) or not all(np.array_equal(stds[0],x) for x in stds[1:]):
            raise ValueError('Ensemble normalization mismatch')
        self.mean,self.std=means[0],stds[0]
        self.force_grid=np.round(np.arange(3.,5.0001,.05),8)

    def curve(self,preaction_sequence,posterior):
        base=np.asarray(preaction_sequence,np.float32)
        if base.shape!=(8,71) or not np.isfinite(base).all():raise ValueError('Expected finite pre-action (8,71) feature')
        if posterior.get('interface')!=self.manifest['posterior_interface']:raise ValueError('Posterior interface mismatch')
        if posterior.get('candidate_actions_executed')!=0 or posterior.get('hidden_friction_used') is not False:raise ValueError('Posterior is not deployment-preamble safe')
        nodes=np.asarray(posterior['integration_nodes'],np.float32);weights=np.asarray(posterior['integration_weights'],float)
        if nodes.ndim!=weights.ndim or nodes.ndim!=1 or len(nodes)==0 or len(nodes)!=len(weights):raise ValueError('Invalid posterior quadrature')
        if np.any(nodes<=0) or np.any(weights<0) or not np.isclose(weights.sum(),1.,atol=1e-9):raise ValueError('Invalid positive normalized posterior')
        forces=np.repeat(self.force_grid,len(nodes));mus=np.tile(nodes,len(self.force_grid))
        x=np.broadcast_to(base,(len(forces),)+base.shape).copy();x[:,:,17]=forces[:,None]/8.;x[:,:,18]=mus[:,None]
        x=(x-self.mean[None,None])/self.std[None,None]
        value=torch.as_tensor(x,dtype=torch.float32,device=self.device)
        with torch.no_grad():
            p=np.mean([torch.sigmoid(model(value[:,:,:17],value[:,0,17:])).cpu().numpy() for model in self.models],axis=0)
        return p.reshape(len(self.force_grid),len(nodes))@weights

    def select(self,preaction_sequence,posterior):
        curve=self.curve(preaction_sequence,posterior)
        utility=curve*(5.-self.force_grid)/5.+(1.-curve)*-1.
        i=int(np.argmax(utility)) # ascending grid implements frozen low-force tie break
        return {'selected_force_N':float(self.force_grid[i]),'predicted_success':float(curve[i]),'utility':float(utility[i]),
          'force_grid_N':self.force_grid.tolist(),'p_success':curve.astype(float).tolist(),'expected_utility':utility.astype(float).tolist(),
          'posterior_interface':self.manifest['posterior_interface'],'posterior_sigma_used':True,
          'feasibility_manifest':str(self.manifest_path),'feasibility_manifest_sha256':sha(self.manifest_path)}
