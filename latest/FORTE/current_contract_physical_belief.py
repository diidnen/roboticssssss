"""Offline/runtime estimator for the frozen current probe contract.

The planning interface remains empirical support at three member means.
Gaussian-mixture density metrics are uncertainty diagnostics; sigma heads
are not silently added to the planning integration interface.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from current_contract_belief_features import ProbeEvidence, SCHEMA_ID, PHASES


class FrictionMember(nn.Module):
    def __init__(self,input_dim=58,projection_dim=16,hidden_dim=16):
        super().__init__()
        self.projection=nn.Sequential(nn.Linear(input_dim,projection_dim),nn.ReLU())
        self.gru=nn.GRU(projection_dim,hidden_dim,batch_first=True)
        self.mu_head=nn.Linear(hidden_dim,1)
        self.log_sigma_head=nn.Linear(hidden_dim,1)

    def forward(self,x,lengths):
        z=self.projection(x)
        packed=pack_padded_sequence(z,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,h=self.gru(packed);h=h[-1]
        return self.mu_head(h).squeeze(1),self.log_sigma_head(h).squeeze(1).clamp(-5.,1.5)


def batch(sequences):
    lengths=torch.tensor([len(x) for x in sequences],dtype=torch.long)
    x=np.zeros((len(sequences),int(lengths.max()),sequences[0].shape[1]),np.float32)
    for i,s in enumerate(sequences):x[i,:len(s)]=s
    return torch.tensor(x),lengths


def predict(model,sequences):
    model.eval()
    with torch.no_grad():mu,ls=model(*batch(sequences))
    return mu.numpy(),ls.exp().numpy()


class CurrentPhysicalBelief:
    def __init__(self,manifest_path,*,diagnostic_only=False):
        self.path=Path(manifest_path)
        self.diagnostic_only=diagnostic_only
        self.manifest=json.loads(self.path.read_text())
        if self.manifest['feature_schema_id']!=SCHEMA_ID:raise ValueError('Wrong physical evidence schema')
        if not diagnostic_only and not self.manifest.get('task0_candidate_qualified',False):
            raise ValueError('Physical-belief candidate is not qualified')
        for path,want in self.manifest['source_hashes'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=want:raise ValueError('Belief source changed: '+path)
        self.models=[]
        for entry in self.manifest['checkpoints']:
            path=Path(entry['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Checkpoint changed')
            ck=torch.load(path,map_location='cpu',weights_only=False)
            model=FrictionMember(ck['input_dim']);model.load_state_dict(ck['state_dict']);model.eval();self.models.append(model)
        self.mean=np.array(self.manifest['normalization']['mean'],np.float32)
        self.std=np.array(self.manifest['normalization']['std'],np.float32)
        self.evidence=ProbeEvidence()

    def rows(self,raw,readback):
        # Only task0 is currently qualified. Caller must use this with a
        # matching task0 probe; no object/private state enters this model.
        if any(int(r['task_id'])!=0 for r in raw):raise ValueError('Task0-only physical-belief candidate')
        if not self.diagnostic_only:verify_decision_prefix(raw)
        x=self.evidence.rows(raw,readback)
        return self.array(x,decision_verified=True)

    def array(self,x,*,decision_verified=False):
        if not self.diagnostic_only and not decision_verified:
            raise ValueError('Deployment must use observed rows with a completed probe, not an unverified array')
        if x.ndim!=2 or x.shape[1]!=58 or not np.isfinite(x).all():raise ValueError('Invalid raw evidence')
        normalized=((x-self.mean)/self.std).astype(np.float32)
        means=[];sigmas=[]
        for model in self.models:
            mu,sigma=predict(model,[normalized]);means.append(float(mu[0]));sigmas.append(float(sigma[0]))
        positive=all(v>0 for v in means)
        return {'member_means':means,'member_log_sigma':np.log(sigmas).tolist(),
            'posterior_samples_mu':means,'posterior_sample_count':3,
            'mean':float(np.mean(means)),'std':float(np.std(means,ddof=1)),
            'epistemic_variance':float(np.var(means,ddof=1)),
            'aleatoric_variance':float(np.mean(np.square(sigmas))),
            'posterior_representation':'unchanged equal-weight empirical support at three member means',
            'sigma_heads_integrated_by_planner':False,'feature_schema_id':SCHEMA_ID,
            'positive_support':positive,'task0_candidate_qualified':bool(self.manifest['task0_candidate_qualified'] and positive),
            'physical_execution_authorized':False}


def verify_decision_prefix(raw):
    if not raw or len({r['trial_id'] for r in raw})!=1:raise ValueError('Mixed or empty probe context')
    if any(r['probe_phase'] not in PHASES for r in raw):raise ValueError('Unknown/candidate phase')
    order=[PHASES.index(r['probe_phase']) for r in raw]
    if order!=sorted(order):raise ValueError('Probe phases are out of causal order')
    counts=Counter(r['probe_phase'] for r in raw)
    expected={'approach':45,'descend':35,'close':70,'hold':40,'probe_back':10,'probe_hold':5}
    if any(counts.get(k)!=v for k,v in expected.items()) or not 1<=counts.get('probe_out',0)<=25:
        raise ValueError('Incomplete probe; no decision-time posterior authorization')
    if raw[-1]['probe_phase']!='probe_hold':raise ValueError('Not a post-probe decision state')
    for r in raw:
        if r['probe_phase'] in ('probe_out','probe_back','probe_hold') and not (int(r['contact_left']) and int(r['contact_right'])):
            raise ValueError('Contact-lost probe is not deployment qualified')
