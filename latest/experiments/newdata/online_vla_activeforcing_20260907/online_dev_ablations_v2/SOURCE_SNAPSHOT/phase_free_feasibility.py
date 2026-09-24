"""Exact phase-column removal from the existing frozen feasibility ensemble.

The public input is 8x64: 10 online chunk/task channels and 54 conditions.
No phase values, phase imputation, scripted stage or waypoint is accepted.
Weights are inherited unchanged except seven deleted GRU input columns that
multiplied constant normalized zeros in all 647 original training rows.
"""
import numpy as np
import torch
from torch import nn
from current_fulltask_feasibility_runtime import CurrentFeasibility

KEEP_SEQUENCE=list(range(6))+list(range(13,17))
KEEP_INPUT=KEEP_SEQUENCE+list(range(17,71))

class PhaseFreeFeasibility(CurrentFeasibility):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not np.array_equal(self.mean[6:13],[1,0,0,0,0,0,0]):
            raise RuntimeError('Exact phase removal requires audited constant TRAIN phase')
        for model in self.models:
            original=model.command_gru.state_dict()
            replacement=nn.GRU(10,64,batch_first=True).to(self.device)
            original['weight_ih_l0']=original['weight_ih_l0'][:,KEEP_SEQUENCE].clone()
            replacement.load_state_dict(original,strict=True)
            model.command_gru=replacement
            model.eval()
            for parameter in model.parameters():parameter.requires_grad_(False)
        self.mean=self.mean[KEEP_INPUT];self.std=self.std[KEEP_INPUT]

    def curve(self,preaction_sequence,posterior):
        base=np.asarray(preaction_sequence,np.float32)
        if base.shape!=(8,64) or not np.isfinite(base).all():
            raise ValueError('Expected finite phase-free pre-action (8,64) features')
        if posterior.get('interface')!=self.manifest['posterior_interface']:
            raise ValueError('Posterior interface mismatch')
        if posterior.get('candidate_actions_executed')!=0 or posterior.get('hidden_friction_used') is not False:
            raise ValueError('Posterior is not pre-action observable')
        nodes=np.asarray(posterior['integration_nodes'],np.float32)
        weights=np.asarray(posterior['integration_weights'],float)
        if nodes.ndim!=1 or weights.shape!=nodes.shape or len(nodes)==0 or not np.isfinite(nodes).all() or not np.isfinite(weights).all():
            raise ValueError('Invalid quadrature')
        if np.any(nodes<=0) or np.any(weights<0) or not np.isclose(weights.sum(),1,atol=1e-9):
            raise ValueError('Invalid positive normalized posterior')
        forces=np.repeat(self.force_grid,len(nodes));mus=np.tile(nodes,len(self.force_grid))
        x=np.broadcast_to(base,(len(forces),)+base.shape).copy()
        x[:,:,10]=forces[:,None]/8;x[:,:,11]=mus[:,None]
        x=(x-self.mean[None,None])/self.std[None,None]
        value=torch.as_tensor(x,dtype=torch.float32,device=self.device)
        with torch.no_grad():
            p=np.mean([torch.sigmoid(model(value[:,:,:10],value[:,0,10:])).cpu().numpy() for model in self.models],axis=0)
        return p.reshape(len(self.force_grid),len(nodes))@weights

    def select(self,preaction_sequence,posterior):
        result=super().select(preaction_sequence,posterior)
        result.update(feasibility_derivation='EXACT_REMOVAL_OF_7_NORMALIZED_CONSTANT_PHASE_COLUMNS',
            feasibility_input_shape=[8,64],phase_representation='NONE',feasibility_retrained=False)
        return result
