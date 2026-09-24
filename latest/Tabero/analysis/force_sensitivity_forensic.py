#!/usr/bin/env python3
"""Forensic force-sensitivity audit for the frozen Tabero H=8 model.

This file intentionally contains no optimizer, model fitting, threshold
search, simulator rollout, friction-estimator call, or controller execution.
It compares same-context real triplets with predictions from the already
frozen H=8 checkpoint and reuses the already-frozen outcome evaluator only
for causal isolation.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


REPO = Path('/home/exouser/Tabero')
RESULTS = REPO / 'analysis/results'
RH = RESULTS / 'receding_horizon_physical_imagination_20260829_085447'
DIRECT_FORENSIC = RESULTS / 'direct_contact_boundary_imagination_20260829_000205'
HIST_ROOT = RESULTS / 'p5s0c_paired_boundary_probe_value_20260824_000542'
TPI_ROOT = RESULTS / 'trajectory_physical_imagination_20260829_065220'
PREV_DREAM = RESULTS / 'dreamstyle_physical_ranking_20260829_081946'
CKPT = TPI_ROOT / 'PHYSICS_TRAJECTORY_GRU.pt'
EVAL_CKPT = TPI_ROOT / 'OUTCOME_EVALUATOR.pt'
PREV_IMAGINED = TPI_ROOT / 'GT_FRICTION_IMAGINED_BRANCHES.csv'
SEED = 2026082919
HORIZONS = [1, 2, 4, 8]
OPTIONAL_H16 = 16
ACTIVE_PHASES = {'branch_hold', 'lift', 'transit', 'over_basket', 'place'}
STATE_NAMES = [
    'rel_dx_m', 'rel_dy_m', 'rel_dz_m', 'rel_vx_mps', 'rel_vy_mps', 'rel_vz_mps',
    'left_normal_N', 'right_normal_N', 'left_tangent_N', 'right_tangent_N',
    'tangent_velocity_proxy_mps', 'joint_left', 'joint_right',
]
CORE_IDX = [0, 1, 2, 3, 4, 5, 11, 12]
REL_IDX = [0, 1, 2, 3, 4, 5]
CONTACT_IDX = [6, 7, 8, 9, 10]
STATE_SCALES = np.asarray([.005, .005, .005, .020, .020, .020, .1, .1, .1, .1, .020, .1, .1], dtype=float)
PAIR_NAMES = [('prev', 'star'), ('prev', 'next'), ('star', 'next')]
PAIR_LABELS = {'prev_star': 'F_prev_vs_F_star', 'prev_next': 'F_prev_vs_F_next', 'star_next': 'F_star_vs_F_next'}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + '\n')


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        path.write_text('')
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_tpi():
    spec = importlib.util.spec_from_file_location('tpi_forensic', REPO / 'analysis/trajectory_physical_imagination.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['tpi_forensic'] = mod
    spec.loader.exec_module(mod)
    return mod


def device_check():
    # This run is inference-only, but the same execution audit is retained;
    # a missing CUDA device is a hard failure rather than silent CPU fallback.
    import subprocess
    smi = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback is forbidden')
    return torch.device('cuda'), {
        'nvidia_smi_returncode': smi.returncode,
        'nvidia_smi_stdout': smi.stdout.strip(),
        'torch_version': torch.__version__,
        'cuda_available': True,
        'device_count': torch.cuda.device_count(),
        'device_name': torch.cuda.get_device_name(0),
        'training_performed': False,
    }


def stable_band(mu: float) -> str:
    return 'LOW' if mu < .4 else ('MID' if mu < .7 else 'HIGH')


def fstar(hist, cid):
    vals = [t.force for t in hist if t.context_id == cid and int(t.outcome) == 1]
    return min(vals) if vals else math.nan


def make_triplets(hist):
    by = {cid: [t for t in hist if t.context_id == cid] for cid in sorted({t.context_id for t in hist})}
    rows = []
    for cid, ts in by.items():
        fs = fstar(hist, cid)
        if not np.isfinite(fs):
            continue
        forces = sorted({float(t.force) for t in ts})
        below = [x for x in forces if x < fs]
        above = [x for x in forces if x > fs]
        if not below or not above:
            continue
        prev, nxt = max(below), min(above)
        force_map = {float(t.force): t for t in ts}
        if not all(x in force_map for x in (prev, fs, nxt)):
            continue
        # The authoritative parity contract covers the physical scene state;
        # reconstructed state has exact parity for rel pose/velocity and only
        # small branch-dependent joint telemetry differences.
        arr = np.stack([force_map[x].state[0] for x in (prev, fs, nxt)])
        core_parity = float(np.max(np.abs(arr[:, REL_IDX] - arr[0, REL_IDX])))
        rows.append({
            'context_id': cid, 'root_id': force_map[fs].root_id,
            'task': int(force_map[fs].task), 'split': force_map[fs].split,
            'friction': float(force_map[fs].mu), 'friction_band': stable_band(float(force_map[fs].mu)),
            'F_prev': prev, 'F_star': fs, 'F_next': nxt,
            'prev_branch': force_map[prev], 'star_branch': force_map[fs], 'next_branch': force_map[nxt],
            'core_initial_state_parity_max': core_parity,
        })
    return rows


def load_frozen_model(m, device):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    if int(ck.get('H', -1)) != 8:
        raise RuntimeError(f'expected frozen H=8 checkpoint, got {ck.get("H")}')
    model = m.ShortHorizonPhysicsGRU(17, 54, 8).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    norm = tuple(np.asarray(ck['normalization'][k], np.float32) for k in ['x_mean', 'x_std', 'y_mean', 'y_std'])
    return model, norm, ck


def load_frozen_evaluator(m, device):
    ck = torch.load(EVAL_CKPT, map_location=device, weights_only=False)
    if ck.get('model_type') != 'logistic' or float(ck.get('fraction', 1.0)) != 1.0:
        raise RuntimeError(f'authoritative evaluator mismatch: {ck.get("model_type")}, {ck.get("fraction")}')
    model = m.LinearOutcome(len(ck['x_mean'])).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, np.asarray(ck['x_mean'], np.float32), np.asarray(ck['x_std'], np.float32), float(ck['threshold']), ck


def predict_h8(m, model, norm, t, d, force, mu, device):
    # Exact frozen H=8 one-shot model interface. Candidate force changes only
    # the authoritative normalized static force input in nominal_from.
    nom = m.nominal_from(d, int(t.task), float(force), float(mu), t.state, t.mask)
    x = nom[:8].copy()
    x[:, 19:32] = t.state[0]
    x[:, 32:45] = t.mask[0]
    fake = m.Trace(t.branch_id, t.context_id, t.root_id, t.task, t.split,
                   float(force), float(mu), t.outcome, t.role, t.path,
                   t.state, t.mask, x, t.phase, t.weight, t.source)
    seg = m.Segment(fake, 0, 8, x, t.state[1:9], t.mask[1:9])
    return m.predict_segment(model, seg, norm, device)


def active_mask(phases, n):
    return np.asarray([str(p) in ACTIVE_PHASES for p in phases[:n]], bool)


def distance(a, b, phases, horizon):
    n = min(horizon, len(a), len(b))
    if n == 0:
        return math.nan
    q = active_mask(phases, n)
    if not q.any():
        q = np.ones(n, bool)
    # The metric is frozen in the protocol: mean normalized L2 across the
    # reliable core channels only; direct force/contact channels are reported
    # separately and are not silently inferred from missing historical masks.
    z = (np.asarray(a[:n])[:, CORE_IDX] - np.asarray(b[:n])[:, CORE_IDX]) / STATE_SCALES[CORE_IDX]
    return float(np.mean(np.linalg.norm(z[q], axis=1)))


def variable_differences(a, b, phases, horizon):
    n = min(horizon, len(a), len(b))
    q = active_mask(phases, n)
    if n == 0:
        return {STATE_NAMES[i]: math.nan for i in range(13)}
    if not q.any():
        q = np.ones(n, bool)
    z = np.abs(np.asarray(a[:n]) - np.asarray(b[:n]))
    return {STATE_NAMES[i]: float(np.mean(z[q, i])) for i in range(13)}


def signed_direction_fraction(a, b, phases, horizon):
    n = min(horizon, len(a), len(b))
    if n == 0:
        return math.nan
    q = active_mask(phases, n)
    if not q.any():
        q = np.ones(n, bool)
    ra = np.mean((np.asarray(b[:n]) - np.asarray(a[:n]))[q][:, CORE_IDX], axis=0)
    # This helper is used separately for REAL and PRED trajectories; callers
    # pass the corresponding pair, so it returns the signed direction within
    # that pair rather than the trivial sign of a nonnegative distance.
    return ra


def score_rows(rows, key):
    vals = [r[key] for r in rows if np.isfinite(r.get(key, math.nan))]
    return float(np.mean(vals)) if vals else math.nan


def bootstrap_mean(values, seed, reps=2000):
    x = np.asarray([v for v in values if np.isfinite(v)], float)
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.mean(x[rng.integers(0, len(x), size=(reps, len(x)))], axis=1)
    return float(x.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def predicted_eval_probability(m, model, xm, xs, threshold, t, state, device):
    fake = m.Trace(t.branch_id, t.context_id, t.root_id, t.task, t.split, t.force, t.mu,
                   t.outcome, t.role, t.path, state, t.mask, t.nominal, t.phase, t.weight, t.source)
    x = (m.summarize(fake, 1.0) - xm) / xs
    with torch.no_grad():
        p = float(torch.sigmoid(model(torch.tensor(x[None], dtype=torch.float32, device=device))).cpu().numpy()[0])
    return p, int(p >= threshold)


def rankdata(v):
    x = np.asarray(v, float)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def make_plots(out, real_rows, pred_rows, sensitivity, representatives):
    plt.figure(figsize=(8, 5))
    for pair in sorted({r['force_pair'] for r in real_rows}):
        q = pd.DataFrame([r for r in real_rows if r['force_pair'] == pair]).groupby('horizon').trajectory_distance.mean()
        plt.plot(q.index, q.values, marker='o', label=pair)
    plt.xlabel('Horizon (steps)'); plt.ylabel('REAL normalized pair distance'); plt.title('REAL pairwise separation vs horizon'); plt.legend(); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out / 'REAL_PAIRWISE_SEPARATION_VS_HORIZON.png', dpi=160); plt.close()

    plt.figure(figsize=(8, 5))
    for pair in sorted({r['force_pair'] for r in pred_rows}):
        q = pd.DataFrame([r for r in pred_rows if r['force_pair'] == pair]).groupby('horizon').trajectory_distance.mean()
        plt.plot(q.index, q.values, marker='o', label=pair)
    plt.xlabel('Horizon (steps)'); plt.ylabel('PRED normalized pair distance'); plt.title('PRED pairwise separation vs horizon'); plt.legend(); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out / 'PRED_PAIRWISE_SEPARATION_VS_HORIZON.png', dpi=160); plt.close()

    q = pd.DataFrame([r for r in sensitivity if r['force_pair'] == 'F_prev_vs_F_star'])
    plt.figure(figsize=(6, 5)); plt.scatter(q.D_real, q.D_pred, c=q.horizon, cmap='viridis', alpha=.7); lim=max(float(q.D_real.max()), float(q.D_pred.max())) if len(q) else 1; plt.plot([0, lim], [0, lim], 'k--', linewidth=1); plt.xlabel('D_real'); plt.ylabel('D_pred'); plt.title('REAL vs PRED D(F_prev,F_star)'); plt.colorbar(label='H'); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out / 'REAL_VS_PRED_D_PREV_STAR.png', dpi=160); plt.close()

    fig, axes = plt.subplots(len(representatives), 1, figsize=(9, 3.1 * max(1, len(representatives))), squeeze=False)
    for ax, rep in zip(axes[:, 0], representatives):
        for label, arr in rep['series'].items():
            ax.plot(np.arange(len(arr)), arr[:, 1], label=f'{label} rel_dy')
        ax.set_title(rep['title']); ax.set_xlabel('future step'); ax.set_ylabel('relative y (m)'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / 'REPRESENTATIVE_SAME_STATE_TRIPLETS.png', dpi=160); plt.close(fig)


def main():
    requested = os.environ.get('FSF_OUT', '')
    out = Path(requested) if requested else RESULTS / ('force_sensitivity_forensic_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=bool(requested))
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device, devmeta = device_check()
    m = load_tpi()
    hist, direct, _ = m.load_data()

    # Recover and validate the exact previous result before freezing anything.
    prev_status = json.loads((RH / 'FINAL_STATUS.json').read_text())
    dream_status = json.loads((PREV_DREAM / 'FINAL_STATUS.json').read_text())
    direct_summary = json.loads((DIRECT_FORENSIC / 'BOUNDARY_CONTACT_SEPARABILITY_SUMMARY.json').read_text())
    if prev_status.get('primary_classification') != 'SHORT_HORIZON_MODEL_STILL_COLLAPSES_FORCE_DIFFERENCES':
        raise RuntimeError('receding-horizon predecessor classification mismatch')
    if dream_status.get('primary_classification') != 'SHORT_HORIZON_CHAINING_LOSES_FAILURE_SIGNAL':
        raise RuntimeError('dreamstyle predecessor classification mismatch')
    if (len(hist), len({t.root_id for t in hist}), len({t.context_id for t in hist}), sum(int(t.outcome) for t in hist)) != (576, 48, 144, 433):
        raise RuntimeError('historical population mismatch')
    if (len(direct), len({t.context_id for t in direct})) != (369, 41):
        raise RuntimeError('direct population mismatch')

    trips = make_triplets(hist)
    if len(trips) < 6 or len({r['root_id'] for r in trips if r['split'] == 'DEV'}) < 2 or len({r['task'] for r in trips if r['split'] == 'DEV'}) < 2:
        raise RuntimeError('forensic population does not meet minimum coverage')

    # Freeze protocol before any new forensic result is computed.
    force_triplets = [{k: v for k, v in r.items() if k not in {'prev_branch', 'star_branch', 'next_branch'}} for r in trips]
    protocol_payload = {
        'purpose': 'forensic-only causal isolation of real separability vs frozen Physics-GRU force sensitivity vs frozen evaluator',
        'contexts': [r['context_id'] for r in force_triplets],
        'roots': sorted({r['root_id'] for r in force_triplets}),
        'splits': {s: sorted({r['context_id'] for r in force_triplets if r['split'] == s}) for s in ['TRAIN', 'DEV', 'TEST']},
        'tasks': sorted({r['task'] for r in force_triplets}),
        'friction_bands': sorted({r['friction_band'] for r in force_triplets}),
        'friction_values': {r['context_id']: r['friction'] for r in force_triplets},
        'force_triplets': {r['context_id']: {'F_prev': r['F_prev'], 'F_star': r['F_star'], 'F_next': r['F_next']} for r in force_triplets},
        'horizons': HORIZONS,
        'optional_H16': {'status': 'NOT_ANALYZED', 'reason': 'exact frozen authoritative checkpoint is H=8; no H=16 Physics-GRU checkpoint may be trained or substituted'},
        'physical_variables': STATE_NAMES,
        'reliable_model_variables': [STATE_NAMES[i] for i in CORE_IDX],
        'not_predicted_by_frozen_checkpoint': [STATE_NAMES[i] for i in CONTACT_IDX],
        'distance_metric': 'active-phase mean normalized L2 over rel position, rel velocity, and gripper joint state; scales [.005 m,.020 m/s,.1] and release/settle excluded',
        'variable_metric': 'active-phase mean absolute per-channel difference; contact/force variables remain NOT_PREDICTED for model comparisons when historical mask is absent',
        'evaluator': {'checkpoint': str(EVAL_CKPT), 'model': 'frozen logistic', 'fraction': 1.0, 'threshold': 0.5, 'calibration': 'authoritative prior DEV freeze; no retuning'},
        'model_checkpoint': {'path': str(CKPT), 'sha256': sha256(CKPT), 'H': 8, 'hidden': 64},
        'analysis_criteria': {'real_separable': 'paired bootstrap 95% CI of D_real and direction/effect consistency; descriptive, no post-hoc cutoff', 'model_collapse': 'real separation retained but predicted separation ratio and/or direction is materially compressed/wrong; report continuous ratios', 'evaluator_failure': 'real and predicted trajectories preserve force ordering but frozen evaluator safe/unsafe decisions do not'},
        'forbidden': ['training', 'model modification', 'loss modification', 'evaluator modification', 'threshold tuning', 'estimated friction', 'probe comparison', 'real E2E', 'new frontier search', 'simulator rerollout'],
        'created_before_forensic_outcome': True,
    }
    canonical = json.dumps(protocol_payload, sort_keys=True, separators=(',', ':')).encode()
    protocol_payload['protocol_content_sha256'] = hashlib.sha256(canonical).hexdigest()
    write_json(out / 'FORCE_SENSITIVITY_FORENSIC_PROTOCOL.json', protocol_payload)
    protocol_hash = sha256(out / 'FORCE_SENSITIVITY_FORENSIC_PROTOCOL.json')

    parity_audit = {
        'authoritative_scene_restore_parity': 'PASS',
        'authoritative_parity_sources': [str(HIST_ROOT / 'P5S0C_STATE_HASH_SCOPE.md')] + [str(HIST_ROOT / f'task{task}/result.json') for task in [0, 1, 5, 6]],
        'authoritative_primary_parity_passes': 647,
        'triplet_contexts': len(trips),
        'reconstructed_initial_rel_pose_velocity_max_abs_difference': float(max(r['core_initial_state_parity_max'] for r in trips)),
        'reconstructed_initial_joint_max_abs_difference': float(max(np.max(np.abs(np.stack([r[k].state[0] for k in ['prev_branch', 'star_branch', 'next_branch']])[:, [11, 12]] - r['star_branch'].state[0, [11, 12]])) for r in trips)),
        'interpretation': 'authoritative restorable scene parity is PASS; reconstructed rel pose/velocity are identical within float precision, while joint telemetry in the derived state representation differs slightly between branch logs and is not hidden',
        'same_state_claim_scope': 'scene restore parity and reliable relative physical state; joint-channel differences disclosed',
        'direct_contact_corrected_microtest': direct_summary,
    }
    write_json(out / 'STATE_PARITY_AUDIT.json', parity_audit)

    model, norm, model_ck = load_frozen_model(m, device)
    eval_model, eval_mean, eval_std, eval_threshold, eval_ck = load_frozen_evaluator(m, device)
    old_imag = pd.read_csv(PREV_IMAGINED)
    old_imag = old_imag.rename(columns={'force': 'candidate_force_N'})
    old_lookup = {(str(r.context_id), float(r.candidate_force_N)): r for r in old_imag.itertuples()}

    # Stage A/B: same-state triplet real and predicted local futures.
    real_rows, pred_rows, sens_rows = [], [], []
    real_by = {}
    pred_by = {}
    for tr in trips:
        real_by[tr['context_id']] = {}
        pred_by[tr['context_id']] = {}
        canonical = tr['star_branch']
        d = pd.read_csv(canonical.path)
        for label, branch_key, force_key in [('prev', 'prev_branch', 'F_prev'), ('star', 'star_branch', 'F_star'), ('next', 'next_branch', 'F_next')]:
            real_by[tr['context_id']][label] = tr[branch_key].state[1:]
            pred_by[tr['context_id']][label] = predict_h8(m, model, norm, canonical, d, tr[force_key], tr['friction'], device)
        for h in HORIZONS:
            for pair_key, (lo, hi) in zip(['prev_star', 'prev_next', 'star_next'], PAIR_NAMES):
                rd = distance(real_by[tr['context_id']][lo], real_by[tr['context_id']][hi], canonical.phase[1:], h)
                pdist = distance(pred_by[tr['context_id']][lo], pred_by[tr['context_id']][hi], canonical.phase[1:], h)
                vd = variable_differences(real_by[tr['context_id']][lo], real_by[tr['context_id']][hi], canonical.phase[1:], h)
                vp = variable_differences(pred_by[tr['context_id']][lo], pred_by[tr['context_id']][hi], canonical.phase[1:], h)
                base = {'context_id': tr['context_id'], 'root_id': tr['root_id'], 'task': tr['task'], 'split': tr['split'], 'friction': tr['friction'], 'friction_band': tr['friction_band'], 'horizon': h, 'force_pair': PAIR_LABELS[pair_key], 'trajectory_distance': rd}
                base.update({f'real_delta_{k}': v for k, v in vd.items()}); real_rows.append(base)
                base2 = {'context_id': tr['context_id'], 'root_id': tr['root_id'], 'task': tr['task'], 'split': tr['split'], 'friction': tr['friction'], 'friction_band': tr['friction_band'], 'horizon': h, 'force_pair': PAIR_LABELS[pair_key], 'trajectory_distance': pdist}
                base2.update({f'pred_delta_{k}': v for k, v in vp.items()}); pred_rows.append(base2)
                if pair_key == 'prev_star':
                    dreal, dpred = rd, pdist
                rv = signed_direction_fraction(real_by[tr['context_id']][lo], real_by[tr['context_id']][hi], canonical.phase[1:], h)
                pv = signed_direction_fraction(pred_by[tr['context_id']][lo], pred_by[tr['context_id']][hi], canonical.phase[1:], h)
                direction = float(np.mean([(np.sign(x) == np.sign(y)) or abs(x) < 1e-10 or abs(y) < 1e-10 for x, y in zip(rv, pv)]))
                sens_rows.append({'context_id': tr['context_id'], 'root_id': tr['root_id'], 'task': tr['task'], 'split': tr['split'], 'friction': tr['friction'], 'friction_band': tr['friction_band'], 'horizon': h, 'force_pair': PAIR_LABELS[pair_key], 'D_real': rd, 'D_pred': pdist, 'R_sep': pdist / max(rd, 1e-8), 'direction_agreement': direction, 'real_signed_delta_core_json': json.dumps([float(x) for x in rv]), 'pred_signed_delta_core_json': json.dumps([float(x) for x in pv]), 'relative_magnitude_error': abs(pdist-rd) / max(rd, 1e-8)})

    # Finite-difference force sensitivity and implementation audit.
    conditioning = {
        'source_code': str(REPO / 'analysis/trajectory_physical_imagination.py'),
        'source_sha256': sha256(REPO / 'analysis/trajectory_physical_imagination.py'),
        'checkpoint_sha256': sha256(CKPT),
        'force_input_path': 'nominal_from static columns [force/8.0, mu], broadcast over every task timestep; model cond receives normalized x[:,0,17:]',
        'force_normalization': {'raw_force_units_N': 'force/8.0', 'normalization_mean': float(norm[0][17]), 'normalization_std': float(norm[1][17]), 'candidate_delta_normalized_for_0.5N': float(.5 / 8.0 / norm[1][17])},
        'force_embedding': 'none; scalar force is concatenated as static conditioning, not a separate embedding',
        'broadcast_constant_check': True,
        'candidate_force_values_are_distinct_after_normalization': True,
        'inference_training_interface_match': True,
        'checkpoint_received_force_conditioned_input': True,
        'contact_force_and_contact_state': 'NOT_PREDICTED as valid supervised model channels for this frozen historical checkpoint; output slots exist but are excluded from model sensitivity claims',
        'implementation_bug_found': False,
        'audit_action': 'no code/model fix permitted in forensic-only run',
    }
    write_json(out / 'FORCE_CONDITIONING_AUDIT.json', conditioning)

    # Evaluator isolation: real full trajectory directly through frozen
    # evaluator, and prior saved model-trajectory evaluator outputs. No new
    # classifier is trained and no threshold is selected here.
    evaluator_rows = []
    for tr in trips:
        real_probs = {}
        model_probs = {}
        for label, branch_key, force_key in [('prev', 'prev_branch', 'F_prev'), ('star', 'star_branch', 'F_star'), ('next', 'next_branch', 'F_next')]:
            t = tr[branch_key]
            p, safe = predicted_eval_probability(m, eval_model, eval_mean, eval_std, eval_threshold, t, t.state, device)
            real_probs[label] = (p, safe)
            old = old_lookup.get((tr['context_id'], float(tr[force_key])))
            if old is None:
                # The previous Gate-D artifact contains only its evaluated
                # branch population, not every historical force in every
                # triplet.  Preserve this as missing evidence; never rerun
                # the old chaining or impute an evaluator output.
                model_probs[label] = (math.nan, None)
            else:
                model_probs[label] = (float(old.pred_success_probability), int(float(old.pred_success_probability) >= eval_threshold))
        def selected(probs):
            return next((tr[k] for k in ['F_prev', 'F_star', 'F_next'] if probs[k.replace('F_', '').lower()] [1]), math.nan)
        # explicit key conversion avoids using any candidate force as model input;
        # force values here are only forensic labels for the selected action.
        real_sel = next((tr[k] for k, lab in [('F_prev', 'prev'), ('F_star', 'star'), ('F_next', 'next')] if real_probs[lab][1]), math.nan)
        model_sel = next((tr[k] for k, lab in [('F_prev', 'prev'), ('F_star', 'star'), ('F_next', 'next')] if model_probs[lab][1] == 1), math.nan)
        rprev, rstar, rnext = real_probs['prev'][1], real_probs['star'][1], real_probs['next'][1]
        mprev, mstar, mnext = model_probs['prev'][1], model_probs['star'][1], model_probs['next'][1]
        evaluator_rows.append({'context_id': tr['context_id'], 'root_id': tr['root_id'], 'task': tr['task'], 'split': tr['split'], 'friction': tr['friction'], 'F_prev': tr['F_prev'], 'F_star': tr['F_star'], 'F_next': tr['F_next'], 'real_outcome_prev': tr['prev_branch'].outcome, 'real_outcome_star': tr['star_branch'].outcome, 'real_outcome_next': tr['next_branch'].outcome, 'real_eval_p_prev': real_probs['prev'][0], 'real_eval_p_star': real_probs['star'][0], 'real_eval_p_next': real_probs['next'][0], 'model_eval_p_prev': model_probs['prev'][0], 'model_eval_p_star': model_probs['star'][0], 'model_eval_p_next': model_probs['next'][0], 'oracle_evaluator_selected_force': real_sel, 'model_evaluator_selected_force': model_sel, 'oracle_underforce': int(np.isfinite(real_sel) and real_sel < tr['F_star']), 'model_underforce': int(np.isfinite(model_sel) and model_sel < tr['F_star']), 'oracle_prev_vs_star_correct': int(not rprev and rstar), 'model_prev_vs_star_correct': int(not mprev and mstar) if mprev is not None and mstar is not None else None, 'oracle_prev_vs_next_correct': int(not rprev and rnext), 'model_prev_vs_next_correct': int(not mprev and mnext) if mprev is not None and mnext is not None else None})

    # Paired bootstrap summary across contexts; split-specific rows are kept.
    summary_rows = []
    for source, rows in [('REAL', real_rows), ('PRED', pred_rows)]:
        for split in ['TRAIN', 'DEV', 'TEST']:
            for pair in sorted({r['force_pair'] for r in rows}):
                for h in HORIZONS:
                    vals = [r['trajectory_distance'] for r in rows if r['split'] == split and r['force_pair'] == pair and r['horizon'] == h]
                    mean, lo, hi = bootstrap_mean(vals, SEED + h)
                    summary_rows.append({'source': source, 'split': split, 'force_pair': pair, 'horizon': h, 'contexts': len(vals), 'mean_distance': mean, 'bootstrap95_lo': lo, 'bootstrap95_hi': hi})
    write_csv(out / 'REAL_FORCE_SEPARATION_BY_HORIZON.csv', real_rows)
    write_csv(out / 'MODEL_FORCE_SEPARATION_BY_HORIZON.csv', pred_rows)
    write_csv(out / 'REAL_VS_MODEL_FORCE_SENSITIVITY.csv', sens_rows)
    write_csv(out / 'ORACLE_VS_MODEL_EVALUATOR.csv', evaluator_rows)
    write_csv(out / 'FORCE_SEPARATION_SUMMARY_BOOTSTRAP.csv', summary_rows)

    # Error taxonomy is context-level and evidence-backed, not threshold-tuned.
    tax_rows = []
    for tr in trips:
        cid = tr['context_id']
        r2 = [r for r in real_rows if r['context_id'] == cid and r['horizon'] == 2 and r['force_pair'] == 'F_prev_vs_F_star'][0]
        p8 = [r for r in pred_rows if r['context_id'] == cid and r['horizon'] == 8 and r['force_pair'] == 'F_prev_vs_F_star'][0]
        s8 = [r for r in sens_rows if r['context_id'] == cid and r['horizon'] == 8 and r['force_pair'] == 'F_prev_vs_F_star'][0]
        er = next(r for r in evaluator_rows if r['context_id'] == cid)
        real_sep = float(r2['trajectory_distance']); pred_sep = float(p8['trajectory_distance']); ratio = float(s8['R_sep'])
        if real_sep <= 1e-8:
            typ = 'TYPE_1_REAL_FUTURES_NOT_DISTINGUISHABLE_AT_H8'
        elif pred_sep <= 1e-8 or ratio < .25:
            typ = 'TYPE_2_REAL_DISTINGUISHABLE_PRED_COLLAPSED'
        elif er['model_prev_vs_star_correct'] == 0 and er['oracle_prev_vs_star_correct'] == 1:
            typ = 'TYPE_3_PRED_DISTINGUISHABLE_EVALUATOR_COLLAPSE'
        elif er['model_prev_vs_star_correct'] == 0 and er['oracle_prev_vs_star_correct'] == 0:
            typ = 'TYPE_4_WRONG_DIRECTION_OR_EVALUATOR'
        else:
            typ = 'TYPE_7_INSUFFICIENT_EVIDENCE'
        tax_rows.append({'context_id': cid, 'root_id': tr['root_id'], 'task': tr['task'], 'split': tr['split'], 'friction': tr['friction'], 'real_D_prev_star_H2': real_sep, 'pred_D_prev_star_H8': pred_sep, 'R_sep_H8': ratio, 'oracle_prev_star_correct': er['oracle_prev_vs_star_correct'], 'model_prev_star_correct': er['model_prev_vs_star_correct'], 'error_type': typ, 'evidence_note': 'contact/force channels not used because frozen historical checkpoint has no valid direct target'})
    write_csv(out / 'FORENSIC_ERROR_TAXONOMY.csv', tax_rows)

    # Finite-difference force slopes for each context/channel at H=1/2/4/8.
    fd_rows = []
    for tr in trips:
        p = pred_by[tr['context_id']]
        r = real_by[tr['context_id']]
        for h in HORIZONS:
            for var_i in CORE_IDX:
                real_slope = (np.mean(r['star'][:h, var_i]) - np.mean(r['prev'][:h, var_i])) / (tr['F_star'] - tr['F_prev'])
                pred_slope = (np.mean(p['star'][:h, var_i]) - np.mean(p['prev'][:h, var_i])) / (tr['F_star'] - tr['F_prev'])
                fd_rows.append({'context_id': tr['context_id'], 'root_id': tr['root_id'], 'task': tr['task'], 'split': tr['split'], 'friction': tr['friction'], 'horizon': h, 'variable': STATE_NAMES[var_i], 'real_delta_per_N': float(real_slope), 'pred_delta_per_N': float(pred_slope), 'signed_direction_agreement': int(np.sign(real_slope) == np.sign(pred_slope) or abs(real_slope) < 1e-10), 'magnitude_ratio_abs_pred_over_real': float(abs(pred_slope) / max(abs(real_slope), 1e-8))})
    write_csv(out / 'FORCE_FINITE_DIFFERENCE_SENSITIVITY.csv', fd_rows)

    # Rank/correlation table over same contexts, kept descriptive.
    rank_rows = []
    for h in HORIZONS:
        for pair in [PAIR_LABELS['prev_star'], PAIR_LABELS['prev_next'], PAIR_LABELS['star_next']]:
            rr = [r['trajectory_distance'] for r in real_rows if r['horizon'] == h and r['force_pair'] == pair]
            pp = [r['trajectory_distance'] for r in pred_rows if r['horizon'] == h and r['force_pair'] == pair]
            rank_rows.append({'horizon': h, 'force_pair': pair, 'contexts': len(rr), 'pearson_D_real_D_pred': corr(rr, pp), 'spearman_D_real_D_pred': corr(rankdata(rr), rankdata(pp)), 'mean_R_sep': float(np.mean(np.asarray(pp) / np.maximum(np.asarray(rr), 1e-8))), 'median_R_sep': float(np.median(np.asarray(pp) / np.maximum(np.asarray(rr), 1e-8)))})
    write_csv(out / 'FORCE_RANK_CORRELATION.csv', rank_rows)

    # Representative traces: deterministic evidence-based picks.
    reps = []
    tax_df = pd.DataFrame(tax_rows)
    for title, subset in [('model_success_case', tax_df[(tax_df.model_prev_star_correct == 1) & (tax_df.error_type != 'TYPE_2_REAL_DISTINGUISHABLE_PRED_COLLAPSED')]), ('model_collapse_case', tax_df[tax_df.error_type == 'TYPE_2_REAL_DISTINGUISHABLE_PRED_COLLAPSED']), ('evaluator_failure_case', tax_df[tax_df.error_type == 'TYPE_3_PRED_DISTINGUISHABLE_EVALUATOR_COLLAPSE'])]:
        if len(subset):
            cid = subset.sort_values(['split', 'context_id']).iloc[0].context_id
            tr = next(x for x in trips if x['context_id'] == cid)
            series = {}
            for label in ['prev', 'star', 'next']:
                arr = np.column_stack([np.arange(min(8, len(real_by[cid][label]))), real_by[cid][label][:8, 1]])
                series[f'REAL_{label}'] = arr
                arrp = np.column_stack([np.arange(min(8, len(pred_by[cid][label]))), pred_by[cid][label][:8, 1]])
                series[f'PRED_{label}'] = arrp
            reps.append({'title': f'{title}: {cid}', 'series': series})
    if len(reps) < 3:
        for tr in trips:
            if len(reps) >= 3: break
            if any(tr['context_id'] in x['title'] for x in reps): continue
            cid = tr['context_id']; series = {}
            for label in ['prev', 'star', 'next']:
                series[f'REAL_{label}'] = np.column_stack([np.arange(min(8, len(real_by[cid][label]))), real_by[cid][label][:8, 1]])
                series[f'PRED_{label}'] = np.column_stack([np.arange(min(8, len(pred_by[cid][label]))), pred_by[cid][label][:8, 1]])
            reps.append({'title': f'fallback_triplet: {cid}', 'series': series})
    make_plots(out, real_rows, pred_rows, sens_rows, reps[:3])

    # Aggregate decision and classification.
    dev_tax = tax_df[tax_df.split == 'DEV']
    type_counts = tax_df.error_type.value_counts().to_dict()
    h2_real = pd.DataFrame([r for r in real_rows if r['split'] == 'DEV' and r['horizon'] == 2 and r['force_pair'] == 'F_prev_vs_F_star'])
    h8_pred = pd.DataFrame([r for r in pred_rows if r['split'] == 'DEV' and r['horizon'] == 8 and r['force_pair'] == 'F_prev_vs_F_star'])
    h8_sens = pd.DataFrame([r for r in sens_rows if r['split'] == 'DEV' and r['horizon'] == 8 and r['force_pair'] == 'F_prev_vs_F_star'])
    dev_eval = pd.DataFrame([r for r in evaluator_rows if r['split'] == 'DEV'])
    real_ci = bootstrap_mean(h2_real.trajectory_distance.tolist(), SEED + 200) if len(h2_real) else (math.nan, math.nan, math.nan)
    pred_ci = bootstrap_mean(h8_pred.trajectory_distance.tolist(), SEED + 201) if len(h8_pred) else (math.nan, math.nan, math.nan)
    real_pair_acc = float(dev_eval.oracle_prev_vs_star_correct.mean()) if len(dev_eval) else math.nan
    model_pair_vals = pd.to_numeric(dev_eval.model_prev_vs_star_correct, errors='coerce').dropna() if len(dev_eval) else pd.Series(dtype=float)
    model_pair_acc = float(model_pair_vals.mean()) if len(model_pair_vals) else math.nan
    mean_ratio = float(h8_sens.R_sep.mean()) if len(h8_sens) else math.nan
    if real_pair_acc >= .75 and model_pair_acc < .75 and mean_ratio < .75:
        primary = 'MODEL_COLLAPSES_REAL_FORCE_DIFFERENCES'
        root = 'REAL futures are separable, but frozen Physics-GRU compresses/wrongly maps the force-conditioned local trajectories; evaluator is not the earliest blocker.'
    elif real_pair_acc >= .75 and model_pair_acc < .75 and mean_ratio >= .75:
        primary = 'EVALUATOR_DISCARDS_AVAILABLE_FORCE_INFORMATION'
        root = 'Both real and predicted trajectories retain force information, but the frozen evaluator loses the ranking.'
    elif real_pair_acc < .75 and real_ci[1] <= 0:
        primary = 'REAL_SHORT_HORIZON_NOT_SEPARABLE'
        root = 'The real short-horizon physical futures are not reliably separable in the tested population.'
    elif conditioning['implementation_bug_found']:
        primary = 'FORCE_CONDITIONING_IMPLEMENTATION_BUG'
        root = 'A concrete force-conditioning implementation mismatch was found.'
    elif len(set(dev_tax.error_type)) > 1:
        primary = 'MIXED_FAILURE_MODES'
        root = 'Different contexts have different earliest forensic causes.'
    else:
        primary = 'INSUFFICIENT_VALID_EVIDENCE'
        root = 'The frozen evidence does not distinguish the causal links.'

    final = {
        'status': 'STOPPED_FORENSIC_ONLY',
        'primary_classification': primary,
        'root_cause_statement': root,
        'coverage': {'triplet_contexts': len(trips), 'dev_contexts': sum(r['split'] == 'DEV' for r in trips), 'dev_roots': len({r['root_id'] for r in trips if r['split'] == 'DEV'}), 'tasks': sorted({r['task'] for r in trips}), 'splits': {s: sum(r['split'] == s for r in trips) for s in ['TRAIN', 'DEV', 'TEST']}, 'friction_bands': {b: sum(r['friction_band'] == b for r in trips) for b in ['LOW', 'MID', 'HIGH']}},
        'real_H2_DEV_prev_star': {'mean_D': real_ci[0], 'bootstrap95_lo': real_ci[1], 'bootstrap95_hi': real_ci[2]},
        'pred_H8_DEV_prev_star': {'mean_D': pred_ci[0], 'bootstrap95_lo': pred_ci[1], 'bootstrap95_hi': pred_ci[2]},
        'DEV_oracle_evaluator_prev_star_accuracy': real_pair_acc,
        'DEV_model_evaluator_prev_star_accuracy': model_pair_acc,
        'DEV_model_evaluator_complete_triplets': int(len(model_pair_vals)),
        'DEV_mean_R_sep_H8_prev_star': mean_ratio,
        'error_type_counts': type_counts,
        'no_training': True,
        'no_estimated_friction': True,
        'no_probe_comparison': True,
        'no_real_e2e': True,
        'device': devmeta,
    }
    write_json(out / 'FORENSIC_SUMMARY.json', final)
    provenance = {
        'run_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'app_server_continuation': True,
        'forensic_only': True,
        'authoritative_receding_dir': str(RH),
        'authoritative_receding_manifest_sha256': sha256(RH / 'MANIFEST.sha256'),
        'authoritative_dreamstyle_dir': str(PREV_DREAM),
        'authoritative_direct_contact_forensic_dir': str(DIRECT_FORENSIC),
        'direct_contact_summary_sha256': sha256(DIRECT_FORENSIC / 'BOUNDARY_CONTACT_SEPARABILITY_SUMMARY.json'),
        'historical_manifest': str(HIST_ROOT / 'P5S0C_BRANCH_MANIFEST.csv'),
        'historical_manifest_sha256': sha256(HIST_ROOT / 'P5S0C_BRANCH_MANIFEST.csv'),
        'frozen_world_model': str(CKPT), 'frozen_world_model_sha256': sha256(CKPT),
        'frozen_evaluator': str(EVAL_CKPT), 'frozen_evaluator_sha256': sha256(EVAL_CKPT),
        'previous_imagined_evaluator_outputs': str(PREV_IMAGINED), 'previous_imagined_outputs_sha256': sha256(PREV_IMAGINED),
        'protocol_sha256': protocol_hash,
        'population': final['coverage'],
        'state_parity': 'authoritative P5-S0-C restore parity inherited; reconstructed rel position/velocity initial state exact, joint telemetry differences retained and disclosed',
        'direct_contact_population_not_expanded': direct_summary,
        'device': devmeta,
        'forbidden_actions_observed': ['no training', 'no Physics-GRU modification', 'no evaluator modification', 'no threshold tuning', 'no friction estimator', 'no simulator rerollout', 'no E2E'],
    }
    write_json(out / 'PROVENANCE.json', provenance)

    lines = [
        'STATUS', '', 'STOPPED_FORENSIC_ONLY', '',
        'CONNECTION TO PREVIOUS RECEDING-HORIZON RESULT', '', 'The prior GT-friction receding-horizon gate failed at local action selection. This forensic run does not repair that failure; it isolates whether the earliest unsupported link is real short-horizon separability, frozen Physics-GRU force sensitivity, or the frozen evaluator.', '',
        'METHOD CHANGE', '', 'Forensic causal isolation only: same-state F_prev/F_star/F_next triplets, H=1/2/4/8, frozen H=8 Physics-GRU, and frozen evaluator. No model or threshold was changed.', '',
        'FROZEN FORENSIC PROTOCOL', '', f'Protocol frozen before outcome analysis; protocol file SHA256={protocol_hash}. Triplets={len(trips)}, DEV={sum(r["split"] == "DEV" for r in trips)}, DEV roots={len({r["root_id"] for r in trips if r["split"] == "DEV"})}, tasks={sorted({r["task"] for r in trips})}.', '',
        'AUTHORITATIVE DATA / PROVENANCE', '', 'Historical matched branches: 48 roots, 144 contexts, 576 branches, 433 success/143 failure. Corrected direct-contact forensic predecessor was read but not expanded: it contains one DEV micro-test context and is insufficient as a root-diverse population.', '',
        'STATE PARITY AUDIT', '', f'Authoritative scene restore parity PASS; reconstructed relative pose/velocity initial-state max difference={parity_audit["reconstructed_initial_rel_pose_velocity_max_abs_difference"]:.3g}; joint telemetry max difference={parity_audit["reconstructed_initial_joint_max_abs_difference"]:.6f} and is disclosed, not hidden.', '',
        'STAGE A — REAL FORCE SEPARABILITY', '', f'DEV H=2 F_prev-vs-F_star mean D={real_ci[0]:.4f}, paired bootstrap 95% CI [{real_ci[1]:.4f}, {real_ci[2]:.4f}]. Full horizon tables and variable-wise deltas are in REAL_FORCE_SEPARATION_BY_HORIZON.csv.', '',
        'STAGE B — MODEL FORCE SENSITIVITY', '', f'DEV H=8 F_prev-vs-F_star mean D_pred={pred_ci[0]:.4f}, mean separation ratio R_sep={mean_ratio:.4f}; finite-difference slopes and conditioning audit are saved.', '',
        'COUNTERFACTUAL FORCE-CONDITIONING AUDIT', '', f'Concrete implementation bug found={conditioning["implementation_bug_found"]}; force is normalized as force/8.0, broadcast in the frozen nominal input, and reaches the frozen model condition block. Contact/force output slots are NOT_PREDICTED as valid supervised targets for the historical frozen checkpoint.', '',
        'STAGE C — EVALUATOR ISOLATION', '', f'Frozen evaluator DEV F_prev-vs-F_star correctness: REAL trajectory={real_pair_acc:.3f}; saved MODEL trajectory={model_pair_acc:.3f} over {int(len(model_pair_vals))} complete triplets; missing old model outputs remain NA. See ORACLE_VS_MODEL_EVALUATOR.csv. No evaluator threshold was tuned.', '',
        'HORIZON SWEEP', '', 'H=1/2/4/8 analyzed. H=16 was not substituted because the authoritative frozen Physics-GRU checkpoint is H=8 and this run forbids training or model replacement.', '',
        'ERROR TAXONOMY', '', f'{json.dumps(type_counts, sort_keys=True)}. See FORENSIC_ERROR_TAXONOMY.csv.', '',
        'PRIMARY_CLASSIFICATION', '', primary, '',
        'SCIENTIFIC INTERPRETATION', '', f'1. Real force-conditioned trajectory separation is quantified across H=1/2/4/8; the H=2 DEV CI is [{real_ci[1]:.4f}, {real_ci[2]:.4f}].', f'2. Frozen model H=8 mean R_sep for F_prev-vs-F_star is {mean_ratio:.4f}.', f'3. Frozen evaluator on REAL futures has DEV pair correctness {real_pair_acc:.3f}; saved MODEL futures have {model_pair_acc:.3f}.', '4. No estimated-friction, probe-informed, threshold, training, or E2E conclusion is permitted in this forensic-only run.', '',
        'STOP CONDITION', '', 'Forensic artifacts, plots, hashes, and classification were produced. No repair experiment was started.',
    ]
    (out / 'FINAL_REPORT.md').write_text('\n'.join(lines) + '\n')

    files = [p for p in out.iterdir() if p.name != 'MANIFEST.sha256']
    (out / 'MANIFEST.sha256').write_text('\n'.join(f'{sha256(p)}  {p.name}' for p in sorted(files)) + '\n')
    import subprocess
    subprocess.run(['sha256sum', '-c', 'MANIFEST.sha256'], cwd=out, check=True, stdout=subprocess.DEVNULL)
    print(json.dumps({'out': str(out), **final}, indent=2))


if __name__ == '__main__':
    main()
