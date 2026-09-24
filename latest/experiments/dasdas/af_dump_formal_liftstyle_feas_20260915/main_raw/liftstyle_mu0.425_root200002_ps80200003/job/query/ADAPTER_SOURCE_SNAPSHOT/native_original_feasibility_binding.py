"""Task-bound original phase-free network, curve and exact original EU rule.

No probability-threshold selector, calibration replacement, monotonic penalty
or prior task-ID alias. Require newly trained native-bound ensemble artifacts.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from native_original_motion_features import TASK_BINDING

ROOT=Path('/home/exouser/FORTE')
SNAPSHOT=Path('/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1/SOURCE_SNAPSHOT')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(SNAPSHOT))
from current_fulltask_feasibility_runtime import CurrentFeasibility
from phase_free_feasibility import PhaseFreeFeasibility
SOURCE=ROOT/'current_fulltask_feasibility_runtime.py'


def phase_free_network_class():
    tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
    original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Network')
    cls=deepcopy(original)
    calls=[n for n in ast.walk(cls) if isinstance(n,ast.Call) and ast.unparse(n.func)=='nn.GRU']
    if len(calls)!=1 or ast.unparse(calls[0].args[0])!='17':raise RuntimeError('Original architecture drift')
    old=calls[0].args[0];calls[0].args[0]=ast.Constant(10)
    executable=deepcopy(cls)
    calls[0].args[0]=old
    if ast.dump(cls)!=ast.dump(original):raise RuntimeError('Unexpected network change')
    namespace={'torch':torch,'nn':nn}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[executable],type_ignores=[])),str(SOURCE),'exec'),namespace)
    return namespace['Network']


class NativeOriginalFeasibility(CurrentFeasibility):
    def __init__(self,path,*,manifest_sha256,device='cpu'):
        self.manifest_path=Path(path);content=self.manifest_path.read_bytes()
        if hashlib.sha256(content).hexdigest()!=manifest_sha256:raise ValueError('Native feasibility manifest changed')
        self.manifest=json.loads(content);m=self.manifest
        if m.get('native_task_binding')!=TASK_BINDING or m.get('root_scope')!=[200002]:
            raise ValueError('Native root-local task binding required')
        if m.get('feature_shape')!=[8,64] or m.get('label_target')!='full_task_success_y' or m.get('calibration')!='NONE_RAW':
            raise ValueError('Original phase-free/raw target contract required')
        if m.get('posterior_interface')!='CURRENT_MULTITASK58_SIGMA_AWARE_POSITIVE_MIXTURE_V1' or m.get('sigma_used') is not True:
            raise ValueError('Original full posterior contract required')
        if m.get('utility_normalization_N')!=5. or m.get('planner_grid_step')!=.05:
            raise ValueError('Original utility or continuous force-grid rule changed')
        support=m.get('force_support',[])
        if len(support)!=2 or not 0<float(support[0])<=float(support[1])<=8:
            raise ValueError('Force range outside authorized bounded engineering support')
        sources=m.get('source_hashes',{})
        for required in (SOURCE,SNAPSHOT/'phase_free_feasibility.py',Path(__file__).resolve()):
            if sources.get(str(required))!=hashlib.sha256(required.read_bytes()).hexdigest():
                raise ValueError('Missing/stale required original/runtime binding hash: '+str(required))
        for source,digest in sources.items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=digest:raise ValueError('Runtime source changed')
        entries=m.get('checkpoints',[])
        if len(entries)!=3 or {e['seed'] for e in entries}!={0,1,2}:raise ValueError('Original three member seeds required')
        self.device=torch.device(device);self.models=[]
        self.mean=self.std=None
        network=phase_free_network_class()
        for entry in sorted(entries,key=lambda e:e['seed']):
            file=Path(entry['path'])
            if hashlib.sha256(file.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Checkpoint changed')
            ck=torch.load(file,map_location='cpu',weights_only=True)
            if ck.get('native_task_binding')!=TASK_BINDING or ck.get('feature_shape')!=[8,64]:
                raise ValueError('Legacy/unbound weights cannot be relabeled for dump')
            if ck.get('protocol_sha256')!=m['training_protocol_sha256'] or ck.get('target')!='full_task_success_y':
                raise ValueError('Training provenance mismatch')
            if ck.get('normalization_fit_split')!='TRAIN':raise ValueError('Only TRAIN normalization permitted')
            mean=np.asarray(ck['normalization_mean'],np.float32);std=np.asarray(ck['normalization_std'],np.float32)
            if mean.shape!=(64,) or std.shape!=(64,) or not np.isfinite(np.r_[mean,std]).all() or np.any(std<=0):
                raise ValueError('Invalid normalization')
            if self.mean is None:self.mean,self.std=mean,std
            elif not np.array_equal(self.mean,mean) or not np.array_equal(self.std,std):raise ValueError('Ensemble norm mismatch')
            model=network();model.load_state_dict(ck['state_dict'],strict=True);model.to(self.device).eval()
            for parameter in model.parameters():parameter.requires_grad_(False)
            self.models.append(model)
        self.force_grid=np.round(np.arange(float(support[0]),float(support[1])+.0001,.05),8)

    def curve(self,sequence,posterior):
        if posterior.get('native_task_binding')!=TASK_BINDING or posterior.get('root_scope')!=[200002]:
            raise ValueError('Only qualified native posterior can enter native feasibility')
        return PhaseFreeFeasibility.curve(self,sequence,posterior)

    def select(self,feature,posterior):
        if not isinstance(feature,dict) or feature.get('task_binding')!=TASK_BINDING:
            raise ValueError('Explicit native feature/task binding required')
        if feature.get('source')!='ONLINE_VLA_ACTION_CHUNK' or feature.get('candidate_actions_executed')!=0:
            raise ValueError('Must lock actual pre-action VLA feature')
        result=CurrentFeasibility.select(self,np.asarray(feature['sequence'],np.float32),posterior)
        result.update(native_task_binding=deepcopy(TASK_BINDING),root_scope=[200002],
                      utility_normalization_N=5.,phase_representation='NONE')
        return result


if __name__=='__main__':
    from current_fulltask_feasibility_runtime import Network
    torch.manual_seed(913);original=Network().eval();native=phase_free_network_class()().eval()
    weights=deepcopy(original.state_dict());weights['command_gru.weight_ih_l0']=weights['command_gru.weight_ih_l0'][:,list(range(6))+list(range(13,17))]
    native.load_state_dict(weights)
    x=torch.randn(100,8,17);x[:,:,6:13]=0;cond=torch.randn(100,54)
    with torch.no_grad():
        a=original(x,cond);b=native(x[:,:,[*range(6),*range(13,17)]],cond)
    error=float(abs(a-b).max())
    assert error<1e-6
    print('ORIGINAL_PHASE_FREE_NETWORK_PARITY_PASSED',error,
          'parameters',sum(p.numel() for p in native.parameters()))
