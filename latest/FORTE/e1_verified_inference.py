"""CPU-only E1 inference, with explicit checkpoint and feature contracts.

No training or simulator entry points. Callers must provide raw probe CSVs
and model-ready nominal inputs; invalid deployment inputs fail closed.
"""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROBE = ROOT / 'activeforcing_probe_conditioned_wm_20260901_064627/probe'
DIRECT = ROOT / 'pooled_predictive_verifier_20260901_033804/checkpoints'
TRANSFER = Path('/home/exouser/Tabero/analysis/activeforcing_historical_transfer_20260904/posterior')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class ProbeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Sequential(torch.nn.Linear(46, 16), torch.nn.ReLU())
        self.gru = torch.nn.GRU(16, 16, batch_first=True)
        self.mu = torch.nn.Linear(16, 1)
        self.logs = torch.nn.Linear(16, 1)

    def forward(self, x):
        lengths = torch.full((len(x),), x.shape[1], dtype=torch.long)
        packed = torch.nn.utils.rnn.pack_padded_sequence(self.proj(x), lengths, batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.mu(h[-1]).squeeze(1)


class DirectNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.command_gru = torch.nn.GRU(17, 64, batch_first=True)
        self.condition = torch.nn.Sequential(torch.nn.Linear(54, 64), torch.nn.ReLU())
        self.head = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))

    def forward(self, x):
        _, h = self.command_gru(x[:, :, :17])
        return self.head(torch.cat([h[-1], self.condition(x[:, 0, 17:])], -1)).squeeze(-1)


def raw_features(path):
    e = module('verified_e1_evidence', TRANSFER / 'evidence_46d.py')
    return e.Evidence46D(TRANSFER / 'NORMALIZATION_46D.json').csv(path)


def probe_members(raw, fold):
    raw = np.asarray(raw, np.float32)
    if raw.ndim != 3 or raw.shape[-1] != 46 or not np.isfinite(raw).all():
        raise ValueError('Expected finite raw [batch,time,46] evidence')
    outputs = []
    for seed in range(3):
        ck = torch.load(PROBE / f'probe_fold{fold}_seed{seed}.pt', map_location='cpu', weights_only=False)
        model = ProbeNet(); model.load_state_dict(ck['state_dict']); model.eval()
        x = (raw - ck['mean']) / ck['std']
        with torch.no_grad():
            outputs.append(model(torch.tensor(x, dtype=torch.float32)).numpy())
    return np.stack(outputs, 1)


def probabilities(x, fold, kind='e1', shared_norm=None):
    """Batch score. shared_norm fixes normalization for weight-only ablation."""
    x = np.asarray(x, np.float32)
    if kind not in ('e1', 'clean') or fold not in (0, 1, 2):
        raise ValueError('Unknown checkpoint family or historical fold')
    if x.ndim != 3 or x.shape[1:] != (8, 71) or not np.isfinite(x).all():
        raise ValueError('Expected finite model-ready [batch,8,71] inputs')
    result = []
    for seed in range(3):
        path = DIRECT / f'DIRECT_POOLED_fold{fold}_seed{seed}.pt' if kind == 'e1' else ROOT / f'analysis/results/final_probe_continuous_posterior_rebuild_20260904/POSTERIOR_FEASIBILITY_seed{seed}.pt'
        ck = torch.load(path, map_location='cpu', weights_only=False)
        m = DirectNet(); m.load_state_dict(ck['state_dict']); m.eval()
        if shared_norm is not None:
            mean, std = shared_norm
        elif kind == 'e1':
            mean, std = ck['normalization']['x_mean'], ck['normalization']['x_std']
        else:
            mean, std = ck['normalization_mean'], ck['normalization_std']
        xx = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
        with torch.no_grad():
            result.append(torch.sigmoid(m(torch.tensor(xx))).numpy())
    return np.stack(result).mean(0)


def select(forces, p, fmax):
    forces, p = np.asarray(forces, float), np.asarray(p, float)
    if (forces.ndim != 1 or not len(forces) or forces.shape != p.shape
            or not np.isfinite(forces).all() or not np.isfinite(p).all()
            or not np.isfinite(fmax) or fmax <= 0
            or np.any((p < 0) | (p > 1))):
        raise ValueError('Invalid candidate probabilities')
    u = p * (1 - forces / fmax) - (1 - p)
    ids = np.flatnonzero(np.isclose(u, u.max(), rtol=0, atol=1e-12))
    i = ids[np.argmin(forces[ids])]
    return {'selected_setpoint': float(forces[i]), 'predicted_success': float(p[i]), 'utility': float(u[i])}


def validate_live_probe(rows, members):
    """Training support check, separate from physical safety qualification."""
    phases = [r['probe_phase'] for r in rows]
    reasons = []
    if not np.isfinite(np.asarray(members)).all():
        reasons.append('NONFINITE_FRICTION_MEMBER')
    if len(rows) != 215 or phases.count('probe_out') != 10:
        reasons.append('PROBE_SEQUENCE_OUTSIDE_VERIFIED_E1_SUPPORT')
    if np.any(np.asarray(members) <= 0):
        reasons.append('NONPOSITIVE_FRICTION_MEMBER')
    return {'admitted': not reasons, 'reasons': reasons, 'rows': len(rows), 'probe_out_steps': phases.count('probe_out'), 'support_rule': 'historical E1 training probes: 215 rows / 10 outward steps; no padding or clipping'}


def e1_nominal_from_saved(probe_path, command_path, task, force, mu):
    """Historical E1 feature contract, NOT the clean post-probe contract.

    Commands must be an independently frozen nominal prefix. Only their
    commands/phases are read; no branch outcomes enter inference. Preserve
    E1's exact preprobe convention even where it differs from modern sensors.
    """
    import csv
    from activeforcing_feasibility_features import nominal_input
    with Path(probe_path).open() as f:
        probe = list(csv.DictReader(f))
    hold = sorted((r for r in probe if r['probe_phase'] == 'hold'), key=lambda r: int(r['step']))
    if not hold or int(hold[-1]['step']) != 190:
        raise ValueError('E1 requires strict preprobe hold step 190')
    with Path(command_path).open() as f:
        prefix = list(csv.DictReader(f))[:8]
    if len(prefix) != 8:
        raise ValueError('Missing frozen eight-command nominal prefix')
    state = np.zeros(13, np.float32); mask = np.zeros(13, np.float32)
    opening = float(hold[-1]['gripper_opening'])
    state[11:13] = [opening, -opening]
    mask[:6] = 1; mask[11:13] = 1
    return nominal_input([[float(r['cmd_'+a]) for a in 'xyz'] for r in prefix],
                         [r['phase'] for r in prefix], task, force, mu, state, mask)
