"""Four predeclared, phase-free dev ablations; never changes primary AF.

All inputs come from current online VLA chunk and postprobe observations.
Local-Lift reuses the historical auxiliary lift checkpoint, with exact deletion
of constant normalized phase columns. No fitting occurs in this module.
"""
import copy
import pickle
import types
from pathlib import Path
import numpy as np
import torch
from common import BASE, read, sha
from phase_free_feasibility import PhaseFreeFeasibility, KEEP_SEQUENCE, KEEP_INPUT

METHODS = ('POSTERIOR_MEAN','COARSE_GRID','PRIOR_NO_POSTERIOR','LOCAL_LIFT')
LOCAL = BASE/'icra_final_experiment_package_v1_20260907/local_lift_model/LOCAL_LIFT_RUNTIME_MANIFEST.json'
STAGE = BASE/'current_matched_stage1_648_v1_20260906/STAGE_MANIFEST.json'


class _NumpyNamespaceUnpickler(pickle.Unpickler):
    """Read hash-verified NumPy2 checkpoint metadata under Isaac NumPy1.

    Remap only the renamed NumPy core namespace; no module aliases, package
    installation, tensor changes or global runtime monkeypatches are applied.
    """
    def find_class(self, module, name):
        if module == 'numpy._core' or module.startswith('numpy._core.'):
            module = module.replace('numpy._core','numpy.core',1)
        return super().find_class(module,name)


def load_local_checkpoint(path):
    compatibility = types.SimpleNamespace(__name__='numpy_namespace_compatibility',
        Unpickler=_NumpyNamespaceUnpickler,load=pickle.load,loads=pickle.loads)
    return torch.load(path,map_location='cpu',weights_only=False,pickle_module=compatibility)


class AblationFeasibility(PhaseFreeFeasibility):
    def __init__(self, method, *args, **kwargs):
        if method not in METHODS:
            raise ValueError('Unknown prespecified ablation')
        super().__init__(*args, **kwargs)
        self.method = method
        self.ablation_sources = {}
        if method == 'COARSE_GRID':
            self.force_grid = np.asarray([3.,4.,5.])
        elif method == 'PRIOR_NO_POSTERIOR':
            stage = read(STAGE); roots = set(stage['selected_roots']['TRAIN'])
            self.prior_nodes = np.asarray([p['mu'] for p in stage['contexts'] if p['root'] in roots],float)
            if len(self.prior_nodes) != 48 or np.any(self.prior_nodes <= 0):
                raise RuntimeError('Expected original48 TRAIN-context empirical prior')
            self.prior_weights = np.full(48,1/48.)
            self.ablation_sources[str(STAGE)] = sha(STAGE)
        elif method == 'LOCAL_LIFT':
            local = read(LOCAL)
            if local['target'] != 'lift_success_y':
                raise RuntimeError('Wrong local checkpoint target')
            models, means, stds = [], [], []
            for index, item in enumerate(local['checkpoints']):
                if sha(item['path']) != item['sha256']:
                    raise RuntimeError('Local-Lift checkpoint changed')
                checkpoint = load_local_checkpoint(item['path'])
                if checkpoint['target'] != 'lift_success_y' or checkpoint['protocol_sha256'] != local['training_protocol_sha256']:
                    raise RuntimeError('Local-Lift checkpoint provenance mismatch')
                mean = np.asarray(checkpoint['normalization_mean'],np.float32)
                std = np.asarray(checkpoint['normalization_std'],np.float32)
                if not np.array_equal(mean[6:13],[1,0,0,0,0,0,0]):
                    raise RuntimeError('Exact phase deletion premise fails for Local-Lift')
                state = dict(checkpoint['state_dict'])
                state['command_gru.weight_ih_l0'] = state['command_gru.weight_ih_l0'][:,KEEP_SEQUENCE].clone()
                model = copy.deepcopy(self.models[index])
                model.load_state_dict(state,strict=True);model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                models.append(model);means.append(mean[KEEP_INPUT]);stds.append(std[KEEP_INPUT])
                self.ablation_sources[item['path']] = item['sha256']
            if not all(np.array_equal(means[0],x) for x in means) or not all(np.array_equal(stds[0],x) for x in stds):
                raise RuntimeError('Local-Lift ensemble normalization mismatch')
            self.models,self.mean,self.std = models,means[0],stds[0]
            self.manifest,self.manifest_path = local,LOCAL
            self.ablation_sources[str(LOCAL)] = sha(LOCAL)

    def ablated_posterior(self, posterior):
        result = copy.deepcopy(posterior)
        if self.method == 'POSTERIOR_MEAN':
            value = float(posterior['posterior_moments']['mean'])
            result.update(integration_nodes=[value],integration_weights=[1.],
                posterior_moments={'mean':value,'std':0.})
        elif self.method == 'PRIOR_NO_POSTERIOR':
            mean = float(np.sum(self.prior_nodes*self.prior_weights))
            result.update(integration_nodes=self.prior_nodes.tolist(),integration_weights=self.prior_weights.tolist(),
                posterior_moments={'mean':mean,'std':float(np.sqrt(np.sum((self.prior_nodes-mean)**2*self.prior_weights)))})
        return result

    def select(self, preaction_sequence, posterior):
        used = self.ablated_posterior(posterior)
        decision = super().select(preaction_sequence,used)
        decision.update(ablation_method=self.method,probe_executed=True,legal_no_probe_pathway=False,
            scientific_label='Prior / No-Posterior' if self.method=='PRIOR_NO_POSTERIOR' else self.method,
            model_target='lift_success_y' if self.method=='LOCAL_LIFT' else 'full_task_success_y',
            posterior_sigma_used=self.method in ('COARSE_GRID','LOCAL_LIFT'),
            runtime_hidden_friction_used=False,outcome_used_for_selection=False,
            ablation_source_hashes=self.ablation_sources,
            used_integration_nodes=used['integration_nodes'],used_integration_weights=used['integration_weights'],
            training_runtime_phase_parity=True,phase_representation='NONE',
            local_lift_training_label_source='SCRIPTED_AUXILIARY_LIFT_LABELS' if self.method=='LOCAL_LIFT' else None)
        return decision
