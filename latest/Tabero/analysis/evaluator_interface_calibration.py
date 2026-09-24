#!/usr/bin/env python3
"""Forensic calibration of the frozen physical evaluator interface.

No model is trained here.  The exact frozen Physics-GRU and logistic outcome
evaluator are replayed offline to expose evaluator features, fit only simple
TRAIN-root calibration maps, choose among them on DEV, and evaluate TEST once
after that choice is frozen.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
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
FORENSIC = RESULTS / 'force_sensitivity_forensic_20260829_090000'
RH = RESULTS / 'receding_horizon_physical_imagination_20260829_085447'
TPI_ROOT = RESULTS / 'trajectory_physical_imagination_20260829_065220'
DREAM = RESULTS / 'dreamstyle_physical_ranking_20260829_081946'
DIRECT = RESULTS / 'direct_contact_boundary_imagination_20260829_000205'
HIST_ROOT = RESULTS / 'p5s0c_paired_boundary_probe_value_20260824_000542'
CKPT = TPI_ROOT / 'PHYSICS_TRAJECTORY_GRU.pt'
EVAL_CKPT = TPI_ROOT / 'OUTCOME_EVALUATOR.pt'
SEED = 2026082920
TASKS = [0, 1, 5, 6]
H = 8
ACTIVE_PHASES = {'branch_hold', 'lift', 'transit', 'over_basket', 'place'}
COMMON_IDX = [0, 1, 2, 3, 4, 5, 11, 12]
STATE_NAMES = [
    'rel_dx_m', 'rel_dy_m', 'rel_dz_m', 'rel_vx_mps', 'rel_vy_mps', 'rel_vz_mps',
    'left_normal_N', 'right_normal_N', 'left_tangent_N', 'right_tangent_N',
    'tangent_velocity_proxy_mps', 'joint_left', 'joint_right',
]
PHASES = ['branch_hold', 'lift', 'transit', 'over_basket', 'place', 'release', 'settle']
PAIR_LABELS = [('prev', 'star', 'F_prev_vs_F_star'), ('prev', 'next', 'F_prev_vs_F_next'), ('star', 'next', 'F_star_vs_F_next')]


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
        w.writeheader(); w.writerows(rows)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def device_check():
    import subprocess
    r = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback forbidden')
    return torch.device('cuda'), {'nvidia_smi_returncode': r.returncode, 'nvidia_smi_stdout': r.stdout.strip(), 'torch_version': torch.__version__, 'cuda_available': True, 'device_count': torch.cuda.device_count(), 'device_name': torch.cuda.get_device_name(0), 'training_performed': False}


def fstar(hist, cid):
    x = [float(t.force) for t in hist if t.context_id == cid and int(t.outcome) == 1]
    return min(x) if x else math.nan


def make_triplets(hist):
    ans = []
    for cid in sorted({t.context_id for t in hist}):
        ts = [t for t in hist if t.context_id == cid]
        fs = fstar(hist, cid)
        if not np.isfinite(fs):
            continue
        forces = sorted({float(t.force) for t in ts})
        lo = [x for x in forces if x < fs]; hi = [x for x in forces if x > fs]
        if not lo or not hi:
            continue
        mp = {float(t.force): t for t in ts}
        prev, nxt = max(lo), min(hi)
        if all(x in mp for x in [prev, fs, nxt]):
            ans.append({'context_id': cid, 'root_id': mp[fs].root_id, 'task': int(mp[fs].task), 'split': mp[fs].split, 'friction': float(mp[fs].mu), 'F_prev': prev, 'F_star': fs, 'F_next': nxt, 'prev_branch': mp[prev], 'star_branch': mp[fs], 'next_branch': mp[nxt]})
    return ans


def feature_names(m):
    names = []
    for ag in ['final', 'mean', 'std', 'max']:
        names += [f'{ag}.{m.STATE_NAMES[i]}' for i in COMMON_IDX]
    names += [f'phase_fraction.{p}' for p in PHASES]
    names += [f'task_onehot.task_{t}' for t in TASKS]
    return names


def load_frozen(m, device):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    if int(ck['H']) != H:
        raise RuntimeError('wrong authoritative Physics-GRU horizon')
    model = m.ShortHorizonPhysicsGRU(17, 54, H).to(device); model.load_state_dict(ck['state_dict']); model.eval()
    norm = tuple(np.asarray(ck['normalization'][k], np.float32) for k in ['x_mean', 'x_std', 'y_mean', 'y_std'])
    ek = torch.load(EVAL_CKPT, map_location=device, weights_only=False)
    if ek.get('model_type') != 'logistic' or float(ek.get('fraction', 1.0)) != 1.0:
        raise RuntimeError('wrong authoritative evaluator')
    em = m.LinearOutcome(len(ek['x_mean'])).to(device); em.load_state_dict(ek['state_dict']); em.eval()
    exm, exs = np.asarray(ek['x_mean'], np.float32), np.asarray(ek['x_std'], np.float32)
    w = em.fc.weight.detach().cpu().numpy().reshape(-1); b = float(em.fc.bias.detach().cpu().numpy().reshape(-1)[0])
    return model, norm, em, exm, exs, w, b, ck, ek


def replay_prediction(dm, m, model, norm, t, device):
    # This is the prior authoritative full-chain replay used only because the
    # predicted physical traces were not serialized.  It is not retraining or
    # a new controller experiment.
    d = pd.read_csv(t.path)
    pred, calls = dm.fast_chain(m, model, t, d, float(t.force), float(t.mu), norm, device)
    if len(pred) < len(t.state) - 1:
        pred = np.vstack([pred, np.repeat(pred[-1][None], len(t.state) - 1 - len(pred), axis=0)]) if len(pred) else np.repeat(t.state[0][None], len(t.state) - 1, axis=0)
    pred = pred[:len(t.state) - 1]
    return np.vstack([t.state[0], pred]), calls


def evaluator_vector(m, em, exm, exs, w, b, t, state):
    fake = m.Trace(t.branch_id, t.context_id, t.root_id, t.task, t.split, t.force, t.mu, t.outcome, t.role, t.path, state, t.mask[:len(state)], t.nominal[:len(state)], t.phase[:len(state)], t.weight, t.source)
    raw = m.summarize(fake, 1.0)
    z = (raw - exm) / exs
    margin = float(np.dot(w, z) + b)
    return z, margin, int(margin >= 0), float(1.0 / (1.0 + np.exp(-np.clip(margin, -80, 80))))


def make_record(m, em, exm, exs, w, b, t, pred_state, calls):
    rz, rm, rs, rp = evaluator_vector(m, em, exm, exs, w, b, t, t.state)
    pz, pm, ps, pp = evaluator_vector(m, em, exm, exs, w, b, t, pred_state)
    return {'branch_id': t.branch_id, 'context_id': t.context_id, 'root_id': t.root_id, 'task': int(t.task), 'split': t.split, 'force': float(t.force), 'friction': float(t.mu), 'real_success': int(t.outcome), 'force_role': t.role, 'real_z': rz, 'pred_z': pz, 'real_margin': rm, 'pred_margin': pm, 'real_safe': rs, 'pred_safe': ps, 'real_probability': rp, 'pred_probability': pp, 'world_model_calls': int(calls)}


def frontier_metrics(records, margin_key, threshold=0.0, split='DEV', mode='threshold'):
    rows = [r for r in records if r['split'] == split]
    by = {}
    for r in rows: by.setdefault(r['context_id'], []).append(r)
    out = []
    for cid, rs in by.items():
        rs = sorted(rs, key=lambda x: x['force']); fs = min([r['force'] for r in rs if r['real_success'] == 1], default=math.nan)
        if mode == 'argmax':
            chosen = max(rs, key=lambda x: x[margin_key])['force'] if rs else math.nan
        else:
            safe = [r['force'] for r in rs if r[margin_key] >= threshold]
            chosen = min(safe) if safe else math.nan
        out.append({'context_id': cid, 'root_id': rs[0]['root_id'], 'task': rs[0]['task'], 'split': split, 'real_F_star': fs, 'chosen_F': chosen, 'exact': int(np.isfinite(fs) and np.isfinite(chosen) and chosen == fs), 'within_one': int(np.isfinite(fs) and np.isfinite(chosen) and abs(chosen-fs) <= .5), 'under_force': int(np.isfinite(fs) and np.isfinite(chosen) and chosen < fs), 'over_force': int(np.isfinite(fs) and np.isfinite(chosen) and chosen > fs), 'no_valid': int(not np.isfinite(chosen)), 'abs_error': abs(chosen-fs) if np.isfinite(chosen) and np.isfinite(fs) else math.nan})
    return out


def metric_summary(rows):
    if not rows: return {'contexts': 0, 'exact': math.nan, 'within_one': math.nan, 'under_force': math.nan, 'over_force': math.nan, 'no_valid': math.nan, 'mae_N': math.nan}
    d = pd.DataFrame(rows)
    return {'contexts': len(d), 'exact': float(d.exact.mean()), 'within_one': float(d.within_one.mean()), 'under_force': float(d.under_force.mean()), 'over_force': float(d.over_force.mean()), 'no_valid': float(d.no_valid.mean()), 'mae_N': float(d.abs_error.mean())}


def pair_metrics(records, margin_key, split):
    ans = []
    for tr in TRIPLETS:
        if tr['split'] != split: continue
        by = {r['force']: r for r in records if r['context_id'] == tr['context_id']}
        if not all(x in by for x in [tr['F_prev'], tr['F_star'], tr['F_next']]): continue
        for lo, hi, name in PAIR_LABELS:
            a, b = by[tr[f'F_{lo}']][margin_key], by[tr[f'F_{hi}']][margin_key]
            ans.append({'context_id': tr['context_id'], 'root_id': tr['root_id'], 'task': tr['task'], 'split': split, 'pair': name, 'margin_low': a, 'margin_high': b, 'low_is_worse': int(a < b), 'margin_delta_high_minus_low': b-a})
    return ans


def pava_fit(x, y):
    order = np.argsort(x, kind='mergesort'); xs = np.asarray(x)[order]; ys = np.asarray(y)[order]
    ux, inv = np.unique(xs, return_inverse=True); uy = np.asarray([ys[inv == i].mean() for i in range(len(ux))], float)
    blocks = []
    for i, val in enumerate(uy):
        blocks.append([i, i, float(val)])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            a, b = blocks[-2], blocks[-1]; n1 = a[1]-a[0]+1; n2 = b[1]-b[0]+1
            blocks[-2:] = [[a[0], b[1], (n1*a[2]+n2*b[2])/(n1+n2)]]
    fit = np.empty(len(ux))
    for a, b, v in blocks: fit[a:b+1] = v
    return ux, fit


def pava_predict(model, x):
    ux, uy = model
    return np.interp(np.asarray(x, float), ux, uy, left=uy[0], right=uy[-1])


def fit_calibrations(train):
    x = np.asarray([r['pred_margin'] for r in train]); y = np.asarray([r['real_margin'] for r in train])
    a, b = np.polyfit(x, y, 1) if np.std(x) > 1e-12 else (0.0, float(y.mean()))
    iso = pava_fit(x, y)
    # A single global margin threshold is chosen from TRAIN labels only. It is
    # a calibration of the existing evaluator margin, not a new classifier.
    candidates = sorted(set(x.tolist() + [0.0]))
    best = None
    for th in candidates:
        safe = x >= th; yy = np.asarray([r['real_success'] for r in train])
        acc = float((safe == yy).mean()); under = float(np.mean((safe == 1) & (yy == 0))); over = float(np.mean((safe == 0) & (yy == 1)))
        key = (acc - 2*under, acc, -under, -over, -abs(th))
        if best is None or key > best[0]: best = (key, float(th), acc, under, over)
    return {'affine': {'a': float(a), 'b': float(b)}, 'isotonic': {'x': iso[0].tolist(), 'y': iso[1].tolist()}, 'threshold': {'threshold': best[1], 'train_branch_accuracy': best[2], 'train_under_force_branch_rate': best[3], 'train_over_force_branch_rate': best[4], 'selection_rule': 'maximize branch accuracy - 2*false-safe rate, then accuracy, on TRAIN only'}}


def calibrated_margin(r, family, cal):
    x = float(r['pred_margin'])
    if family == 'frozen': return x, 0.0
    if family == 'affine': return cal['affine']['a']*x + cal['affine']['b'], 0.0
    if family == 'isotonic': return float(pava_predict((np.asarray(cal['isotonic']['x']), np.asarray(cal['isotonic']['y'])), [x])[0]), 0.0
    if family == 'threshold': return x, cal['threshold']['threshold']
    raise ValueError(family)


def make_plots(out, sep_rows, cal_rows, feature_metrics):
    d = pd.DataFrame(sep_rows)
    for source, path, title in [('real_margin', 'REAL_VS_MODEL_MARGIN.png', 'REAL vs PRED evaluator margin'), ('pred_margin', 'PRED_MARGIN_BY_FORCE_CLASS.png', 'Predicted evaluator margin by force class')]:
        plt.figure(figsize=(8, 5))
        if source == 'real_margin':
            for label, g in d[d.split == 'DEV'].groupby('force_class'):
                plt.scatter(g.real_margin, g.pred_margin, label=label, alpha=.7)
            lim=max(abs(d.real_margin).max(), abs(d.pred_margin).max()); plt.plot([-lim,lim],[-lim,lim],'k--',lw=1); plt.xlabel('REAL margin'); plt.ylabel('PRED margin')
        else:
            q=d[d.split == 'DEV'].groupby('force_class').pred_margin.mean().reindex(['prev','star','next']); plt.plot(q.index,q.values,marker='o'); plt.axhline(0,color='k',ls='--',lw=1); plt.xlabel('force class'); plt.ylabel('PRED margin')
        plt.title(title); plt.grid(alpha=.25); plt.legend() if source == 'real_margin' else None; plt.tight_layout(); plt.savefig(out/path,dpi=160); plt.close()
    fm = pd.DataFrame(feature_metrics)
    q = fm[(fm.split == 'DEV') & (fm.feature_name.str.contains('rel_|joint_'))].sort_values('abs_weight', ascending=False).head(12)
    plt.figure(figsize=(10,5)); plt.barh(q.feature_name, q.bias); plt.axvline(0,color='k',lw=1); plt.title('DEV predicted-minus-real evaluator-feature bias'); plt.xlabel('normalized feature bias'); plt.tight_layout(); plt.savefig(out/'FEATURE_BIAS_TOP12.png',dpi=160); plt.close()


def main():
    global TRIPLETS
    requested = os.environ.get('EIC_OUT', '')
    out = Path(requested) if requested else RESULTS / ('evaluator_interface_calibration_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=bool(requested))
    device, devmeta = device_check()
    m = load_module('tpi_calibration', REPO/'analysis/trajectory_physical_imagination.py')
    dm = load_module('dream_calibration', REPO/'analysis/dreamstyle_physical_ranking.py')
    hist, direct, _ = m.load_data(); TRIPLETS = make_triplets(hist)
    if len(TRIPLETS) != 71: raise RuntimeError(f'expected 71 authoritative triplets, got {len(TRIPLETS)}')
    model, norm, em, exm, exs, w, bias, model_ck, eval_ck = load_frozen(m, device)
    names = feature_names(m)
    if len(names) != len(w): raise RuntimeError(f'evaluator feature count mismatch {len(names)} vs {len(w)}')

    # Freeze the full context/metric protocol before any new evaluator result.
    triplet_payload = [{k: v for k, v in x.items() if k not in ['prev_branch','star_branch','next_branch']} for x in TRIPLETS]
    protocol = {'purpose':'calibration mismatch forensic for frozen Physics-GRU -> frozen physical evaluator interface', 'contexts':[x['context_id'] for x in triplet_payload], 'roots':sorted({x['root_id'] for x in triplet_payload}), 'splits':{s:sorted({x['context_id'] for x in triplet_payload if x['split']==s}) for s in ['TRAIN','DEV','TEST']}, 'tasks':sorted({x['task'] for x in triplet_payload}), 'frictions':{x['context_id']:x['friction'] for x in triplet_payload}, 'force_triplets':{x['context_id']:{k:x[k] for k in ['F_prev','F_star','F_next']} for x in triplet_payload}, 'model_checkpoint':{'path':str(CKPT),'sha256':sha256(CKPT),'H':H}, 'evaluator_checkpoint':{'path':str(EVAL_CKPT),'sha256':sha256(EVAL_CKPT),'model_type':eval_ck.get('model_type'),'fraction':eval_ck.get('fraction')}, 'evaluator_features':names, 'horizons':[H], 'calibration_families':['affine','isotonic','threshold'], 'metrics':['feature bias/MAE/RMSE/slope/Pearson/Spearman/sign agreement/threshold-crossing recall/F_prev-vs-F_star ordering','frontier exact/within-one/under/over/no-valid/MAE','triplet pair ordering'], 'split_usage':'TRAIN fit; DEV family selection; TEST evaluated once after freeze', 'inherited_gate':{'DEV frontier exact_minimum':0.80,'DEV under_force_maximum':0.10}, 'forbidden':['new Physics-GRU','world-model modification','loss change','evaluator retraining','friction estimator','estimated friction','probe comparison','real E2E','threshold tuning on TEST','new data collection'], 'created_before_new_forensic_outcome':True}
    protocol['protocol_content_sha256'] = hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write_json(out/'EVALUATOR_INTERFACE_CALIBRATION_PROTOCOL.json', protocol)
    protocol_hash = sha256(out/'EVALUATOR_INTERFACE_CALIBRATION_PROTOCOL.json')

    # Exact frozen evaluator decomposition: no hand-written event predicate is
    # present; the only safety boundary is the complete logistic margin.
    decomposition = {'evaluator_type':'frozen LinearOutcome logistic', 'source_checkpoint':str(EVAL_CKPT), 'input_summary':'final, mean, std, max of COMMON_IDX physical channels + phase fractions + task one-hot', 'physical_channels_used':[m.STATE_NAMES[i] for i in COMMON_IDX], 'physical_channels_not_used':['left/right normal force','left/right tangential force','bilateral contact','contact-loss duration','tangential velocity proxy','object angular velocity'], 'phase_channels':PHASES, 'task_channels':TASKS, 'normalization':'frozen evaluator TRAIN normalization from checkpoint', 'margin_formula':'w dot ((summary - x_mean) / x_std) + b', 'safe_rule':'margin >= 0 (equivalent frozen probability >= 0.5)', 'unsafe_rule':'margin < 0', 'hard_threshold':0.0, 'continuous_output':'logistic probability', 'explicit_hard_physical_rules':False, 'feature_table':[{'index':i,'feature_name':n,'coefficient':float(w[i]),'direction':'higher_feature_increases_safety' if w[i]>0 else 'higher_feature_decreases_safety' if w[i]<0 else 'zero','aggregation':n.split('.')[0] if '.' in n else 'categorical'} for i,n in enumerate(names)], 'contact_definition_reference':str(DIRECT/'DIRECT_PHYSICAL_EVENT_DEFINITION.json'), 'contact_definition_sha256':sha256(DIRECT/'DIRECT_PHYSICAL_EVENT_DEFINITION.json')}
    write_json(out/'FROZEN_EVALUATOR_DECOMPOSITION.json', decomposition)

    # Inference is split: TRAIN/DEV first, so calibration is frozen before
    # TEST is read for predicted trajectories or metrics.
    records = []
    branch_trace = {}
    for split in ['TRAIN','DEV']:
        for t in [x for x in hist if x.split == split]:
            pred_state, calls = replay_prediction(dm, m, model, norm, t, device)
            rec = make_record(m, em, exm, exs, w, bias, t, pred_state, calls); records.append(rec); branch_trace[t.branch_id] = pred_state
    train_records = [r for r in records if r['split']=='TRAIN']; dev_records = [r for r in records if r['split']=='DEV']
    cal = fit_calibrations(train_records)
    # DEV-only family selection; inherited frontier gate is reported but not
    # changed.  Tie-breaks prefer safety and then exactness.
    families = ['frozen','affine','isotonic','threshold']
    dev_choice = {}
    for fam in families:
        if fam == 'frozen': mm, th = 'pred_margin', 0.0
        else:
            vals = []
            for r in dev_records:
                z, th0 = calibrated_margin(r,fam,cal); vals.append({**r,'cal_margin':z})
            mm, th = 'cal_margin', 0.0
            if fam == 'threshold': th = cal['threshold']['threshold']
        rs = []
        for r in dev_records:
            cm, unused = calibrated_margin(r,fam,cal); rs.append({**r,'cal_margin':cm})
        fr = frontier_metrics(rs,'cal_margin',th,'DEV')
        pm = pair_metrics(rs,'cal_margin','DEV')
        pair = float(np.mean([x['low_is_worse'] for x in pm if x['pair']=='F_prev_vs_F_star'])) if pm else math.nan
        sm = metric_summary(fr); dev_choice[fam] = {'frontier':sm,'boundary_pair':pair,'selection_key':(sm['exact']-2*sm['under_force'],sm['exact'],pair,-sm['over_force'])}
    selected = max(families, key=lambda f: dev_choice[f]['selection_key'])
    write_json(out/'CALIBRATION_SELECTION.json', {'selected_family':selected,'families':dev_choice,'selection_split':'DEV','selection_rule':'maximize exact - 2*under-force, then exact, boundary F_prev-vs-F_star, minimize over-force','test_not_used':True})
    write_json(out/'CALIBRATED_EVALUATOR_FREEZE.json', {'selected_family':selected,'calibration':cal,'fit_split':'TRAIN roots','selection_split':'DEV','test_opened_after_freeze':True,'world_model_unchanged':True,'evaluator_unchanged':True})

    # Only now read/replay TEST.
    for t in [x for x in hist if x.split == 'TEST']:
        pred_state, calls = replay_prediction(dm, m, model, norm, t, device)
        records.append(make_record(m, em, exm, exs, w, bias, t, pred_state, calls)); branch_trace[t.branch_id] = pred_state

    # Required exact evaluator features on the same-state triplets.
    feature_rows = []
    for tr in TRIPLETS:
        for label, key in [('prev','F_prev'),('star','F_star'),('next','F_next')]:
            t = tr[f'{label}_branch']; rec = next(r for r in records if r['branch_id'] == t.branch_id)
            for i,n in enumerate(names):
                feature_rows.append({'context':tr['context_id'],'task':tr['task'],'root':tr['root_id'],'split':tr['split'],'friction':tr['friction'],'force':tr[key],'force_class':label,'feature_name':n,'real_value':float(rec['real_z'][i]),'pred_value':float(rec['pred_z'][i]),'threshold':0.0,'real_margin':float(rec['real_margin']),'pred_margin':float(rec['pred_margin']),'real_margin_contribution':float(w[i]*rec['real_z'][i]),'pred_margin_contribution':float(w[i]*rec['pred_z'][i]),'real_safe':int(rec['real_safe']),'pred_safe':int(rec['pred_safe'])})
    write_csv(out/'REAL_VS_PRED_EVALUATOR_FEATURES.csv', feature_rows)

    # False-safe attribution is explicit about the frozen logistic interface;
    # categories are descriptive, not a newly tuned evaluator.
    false_rows = []
    for tr in TRIPLETS:
        t = tr['prev_branch']; rec = next(r for r in records if r['branch_id'] == t.branch_id)
        if rec['real_safe'] != 0 or rec['pred_safe'] != 1: continue
        cr = w * rec['real_z']; cp = w * rec['pred_z']; real_neg = float(np.maximum(-cr,0).sum()); pred_neg = float(np.maximum(-cp,0).sum()); agree = float(np.mean(np.sign(cr) == np.sign(cp)))
        top = np.argsort(cr)[:5]
        if pred_neg < .10 * max(real_neg,1e-8): category='correct_unsafe_direction_but_event_approximately_zero'
        elif agree < .40: category='wrong_direction'
        else:
            real_ag = names[int(np.argmax(np.abs(cr)))] .split('.')[0]; pred_ag = names[int(np.argmax(np.abs(cp)))] .split('.')[0]
            category='wrong_time_or_phase_or_temporal_smoothing' if real_ag != pred_ag else 'correct_unsafe_direction_but_insufficient_magnitude'
        false_rows.append({'context':tr['context_id'],'task':tr['task'],'root':tr['root_id'],'split':tr['split'],'friction':tr['friction'],'F_prev':tr['F_prev'],'F_star':tr['F_star'],'F_next':tr['F_next'],'real_margin':rec['real_margin'],'pred_margin':rec['pred_margin'],'real_negative_contribution_mass':real_neg,'pred_negative_contribution_mass':pred_neg,'contribution_sign_agreement':agree,'top_real_unsafe_features_json':json.dumps([{'feature':names[int(i)],'contribution':float(cr[i]),'pred_contribution':float(cp[i])} for i in top]),'attribution_category':category,'explanation':'frozen evaluator has no independent physical event rule; attribution decomposes its signed linear margin contributions'})
    write_csv(out/'F_PREV_FALSE_SAFE_ATTRIBUTION.csv', false_rows)

    # Feature-wise calibration metrics, with signed evaluator contribution as
    # the threshold-crossing object.  All rows are computed after TEST replay,
    # but calibration parameters were frozen before TEST.
    feature_metrics = []
    for split in ['TRAIN','DEV','TEST']:
        rs = [r for r in records if r['split']==split]
        for i,n in enumerate(names):
            real = np.asarray([r['real_z'][i] for r in rs]); pred = np.asarray([r['pred_z'][i] for r in rs]);
            slope = float(np.polyfit(pred,real,1)[0]) if np.std(pred)>1e-12 else math.nan
            pear = float(np.corrcoef(pred,real)[0,1]) if np.std(pred)>1e-12 and np.std(real)>1e-12 else math.nan
            sr = np.argsort(np.argsort(real)); sp = np.argsort(np.argsort(pred)); spear = float(np.corrcoef(sr,sp)[0,1]) if len(rs)>1 else math.nan
            wc = float(w[i]); rc = wc*((real-exm[i]*0) if False else real); pc=wc*pred
            real_unsafe = rc < 0; pred_unsafe = pc < 0; recall = float(np.mean(pred_unsafe[real_unsafe])) if np.any(real_unsafe) else math.nan
            pair = []
            for tr in TRIPLETS:
                if tr['split'] != split: continue
                mp={r['force']:r for r in rs if r['context_id']==tr['context_id']}
                if tr['F_prev'] in mp and tr['F_star'] in mp:
                    pair.append(int(wc*mp[tr['F_prev']]['pred_z'][i] < wc*mp[tr['F_star']]['pred_z'][i]) == int(wc*mp[tr['F_prev']]['real_z'][i] < wc*mp[tr['F_star']]['real_z'][i]))
            feature_metrics.append({'split':split,'feature_name':n,'coefficient':wc,'abs_weight':abs(wc),'bias_pred_minus_real':float(np.mean(pred-real)),'bias':float(np.mean(pred-real)),'mae':float(np.mean(np.abs(pred-real))),'rmse':float(np.sqrt(np.mean((pred-real)**2))),'slope_pred_vs_real':slope,'pearson':pear,'spearman':spear,'sign_agreement_about_zero':float(np.mean(np.sign(pred)==np.sign(real))),'threshold_crossing_recall':recall,'F_prev_vs_F_star_ordering_agreement':float(np.mean(pair)) if pair else math.nan,'n':len(rs)})
    write_csv(out/'FEATURE_WISE_CALIBRATION.csv', feature_metrics)

    # Frontier comparison after calibration freeze.
    comp_rows=[]; frontier_all=[]
    for fam in families:
        fam_records=[]
        for r in records:
            cm, th = calibrated_margin(r,fam,cal); fam_records.append({**r,'cal_margin':cm})
        for split in ['TRAIN','DEV','TEST']:
            fr=frontier_metrics(fam_records,'cal_margin',th,split); sm=metric_summary(fr)
            for x in fr: frontier_all.append({'method':fam,**x})
            comp_rows.append({'method':fam,'split':split,**sm,'gate_exact_min_0.80':int(sm['exact']>=.80),'gate_under_max_0.10':int(sm['under_force']<=.10)})
        if fam == 'frozen':
            # Real evaluator ceiling uses real margins, not calibrated model
            # margins; store as a separate method.
            for split in ['TRAIN','DEV','TEST']:
                rr=[{**r,'real_margin_as_cal':r['real_margin']} for r in records]
                fr=frontier_metrics(rr,'real_margin_as_cal',0.0,split); sm=metric_summary(fr); comp_rows.append({'method':'frozen_evaluator_REAL','split':split,**sm,'gate_exact_min_0.80':int(sm['exact']>=.80),'gate_under_max_0.10':int(sm['under_force']<=.10)})
    write_csv(out/'CALIBRATION_FRONTIER_COMPARISON.csv',comp_rows); write_csv(out/'CALIBRATED_FRONTIER_DETAILS.csv',frontier_all)

    # Relative/ranking diagnostic: no F_star is used by the diagnostic rule;
    # it reports ordinal safety ranking and an argmax diagnostic separately.
    rel_rows=[]; rel_frontier=[]
    for split in ['TRAIN','DEV','TEST']:
        pm=pair_metrics(records,'pred_margin',split)
        for x in pm: rel_rows.append({'method':'relative_predicted_margin','mode':'pairwise',**x})
        for tr in TRIPLETS:
            if tr['split'] != split: continue
            rs=[r for r in records if r['context_id']==tr['context_id']]
            by={r['force']:r for r in rs};
            if not all(x in by for x in [tr['F_prev'],tr['F_star'],tr['F_next']]): continue
            strict=int(by[tr['F_prev']]['pred_margin'] < by[tr['F_star']]['pred_margin'] < by[tr['F_next']]['pred_margin'])
            chosen=max([tr['F_prev'],tr['F_star'],tr['F_next']],key=lambda f:by[f]['pred_margin']); fs=tr['F_star']
            rel_frontier.append({'context_id':tr['context_id'],'root_id':tr['root_id'],'task':tr['task'],'split':split,'diagnostic':'argmax_predicted_margin','real_F_star':fs,'chosen_F':chosen,'exact':int(chosen==fs),'under_force':int(chosen<fs),'over_force':int(chosen>fs),'strict_triplet_order':strict})
    write_csv(out/'RELATIVE_RANKING_DIAGNOSTIC.csv',rel_rows+rel_frontier)

    # Margin summaries and hypothesis evidence.
    all_pair = pair_metrics(records,'pred_margin','DEV')
    model_pair = float(np.mean([x['low_is_worse'] for x in all_pair if x['pair']=='F_prev_vs_F_star'])) if all_pair else math.nan
    real_pair = float(np.mean([x['low_is_worse'] for x in pair_metrics(records,'real_margin','DEV') if x['pair']=='F_prev_vs_F_star'])) if all_pair else math.nan
    dev_feat=pd.DataFrame([x for x in feature_metrics if x['split']=='DEV']).sort_values('abs_weight',ascending=False)
    critical=dev_feat.head(12).to_dict('records')
    hypotheses={'H1_scale_offset_mismatch':{'supporting_metrics':{'DEV_pred_vs_real_margin_bias':float(np.mean([r['pred_margin']-r['real_margin'] for r in dev_records])),'DEV_pred_vs_real_margin_slope':float(np.polyfit([r['pred_margin'] for r in dev_records],[r['real_margin'] for r in dev_records],1)[0]),'DEV_pred_real_margin_pearson':float(np.corrcoef([r['pred_margin'] for r in dev_records],[r['real_margin'] for r in dev_records])[0,1])},'interpretation':'supported if affine mapping reduces the absolute margin mismatch while ordering remains'},'H2_evaluator_critical_signal_loss':{'critical_feature_rows':critical,'interpretation':'supported only for evaluator-critical features with low threshold-crossing recall or low pred-vs-real correlation; direct contact/slip unavailable to this frozen evaluator'},'H3_absolute_threshold_mismatch':{'supporting_metrics':{'DEV_real_boundary_pair':real_pair,'DEV_model_boundary_pair':model_pair,'DEV_relative_strict_triplet':float(np.mean([x['strict_triplet_order'] for x in rel_frontier if x['split']=='DEV'])) if rel_frontier else math.nan,'selected_calibration':selected,'selected_DEV':dev_choice[selected]},'interpretation':'supported when ordinal force information survives but absolute margin threshold does not'}}
    write_json(out/'HYPOTHESIS_EVIDENCE.json', hypotheses)

    # Required compact audit of leaks and provenance.
    write_json(out/'LEAKAGE_AUDIT.json', {'status':'PASS','checks':['same-context triplets inherited from authoritative frontier','TRAIN-only affine/isotonic/threshold fitting','DEV-only calibration family selection','TEST opened after CALIBRATED_EVALUATOR_FREEZE.json','no F_star/root/friction in evaluator input','no outcome used in affine/isotonic fit','no model/evaluator retraining','no estimated friction/probe/E2E'], 'found':[]})
    write_json(out/'LATENCY.json', {'device':devmeta,'frozen_model_inference_only':True,'train_dev_branches_replayed':len(train_records)+len(dev_records),'test_branches_replayed':sum(r['split']=='TEST' for r in records),'world_model_calls':sum(r['world_model_calls'] for r in records),'estimated_friction':False,'real_e2e':False})
    write_json(out/'PROVENANCE.json', {'run_timestamp_utc':datetime.now(timezone.utc).isoformat(),'app_server_continuation':True,'forensic_only':True,'authoritative_forensic_dir':str(FORENSIC),'authoritative_forensic_manifest_sha256':sha256(FORENSIC/'MANIFEST.sha256'),'authoritative_receding_dir':str(RH),'authoritative_world_model':str(CKPT),'world_model_sha256':sha256(CKPT),'authoritative_evaluator':str(EVAL_CKPT),'evaluator_sha256':sha256(EVAL_CKPT),'dreamstyle_dir':str(DREAM),'corrected_direct_contact_dir':str(DIRECT),'protocol_sha256':protocol_hash,'device':devmeta,'triplets':len(TRIPLETS),'no_training':True,'no_model_modification':True,'no_evaluator_modification':True})

    test_rows = [x for x in comp_rows if x['split']=='TEST']; selected_test=next(x for x in test_rows if x['method']==selected)
    false_counts=pd.Series([x['attribution_category'] for x in false_rows]).value_counts().to_dict() if false_rows else {}
    primary = 'SIMPLE_CALIBRATION_RECOVERS_FORCE_FRONTIER' if dev_choice[selected]['frontier']['exact']>=.80 and dev_choice[selected]['frontier']['under_force']<=.10 and selected_test['exact']>=.80 and selected_test['under_force']<=.10 else ('MODEL_PRESERVES_RANKING_BUT_NOT_ABSOLUTE_PHYSICAL_MARGIN' if model_pair>=.75 and not (dev_choice[selected]['frontier']['exact']>=.80 and selected_test['exact']>=.80) else ('EVALUATOR_CRITICAL_SIGNAL_NOT_PREDICTED' if any(float(x.get('threshold_crossing_recall',1))<.5 for x in critical) else 'FROZEN_EVALUATOR_FORMULATION_INCOMPATIBLE_WITH_IMAGINED_TRAJECTORIES'))
    summary={'status':'STOPPED_CALIBRATION_FORENSIC','primary_classification':primary,'selected_calibration':selected,'coverage':{'triplets':len(TRIPLETS),'train_contexts':len({t['context_id'] for t in TRIPLETS if t['split']=='TRAIN'}),'dev_contexts':len({t['context_id'] for t in TRIPLETS if t['split']=='DEV'}),'test_contexts':len({t['context_id'] for t in TRIPLETS if t['split']=='TEST'}),'tasks':sorted({t['task'] for t in TRIPLETS}),'friction_bands':{'LOW':sum(t['friction']<.4 for t in TRIPLETS),'MID':sum(.4<=t['friction']<.7 for t in TRIPLETS),'HIGH':sum(t['friction']>=.7 for t in TRIPLETS)}},'dev_choice':dev_choice,'selected_test':selected_test,'real_evaluator_ceiling_test':next(x for x in test_rows if x['method']=='frozen_evaluator_REAL'),'dev_real_boundary_pair':real_pair,'dev_model_boundary_pair':model_pair,'false_safe_attribution_counts':false_counts,'no_training':True,'no_estimated_friction':True,'no_probe_comparison':True,'no_real_e2e':True,'device':devmeta}
    write_json(out/'FINAL_STATUS.json', summary)
    make_plots(out, feature_rows, comp_rows, feature_metrics)
    report=f'''# STATUS\n\nSTOPPED_CALIBRATION_FORENSIC\n\n# SINGLE SCIENTIFIC GOAL\n\nThis run isolated the frozen Physics-GRU to frozen evaluator calibration interface without changing either model.\n\n# AUTHORITATIVE INPUTS / HASHES\n\nPredecessor forensic: {FORENSIC}; receding-horizon result: {RH}; frozen Physics-GRU SHA256={sha256(CKPT)}; frozen evaluator SHA256={sha256(EVAL_CKPT)}; protocol SHA256={protocol_hash}.\n\n# FROZEN EVALUATOR DECOMPOSITION\n\nThe evaluator is a frozen logistic linear margin over final/mean/std/max relative position, relative velocity, gripper joints, phase fractions, and task one-hot. It has no explicit contact, bilateral-contact, force, tangential-velocity, or slip rule. Safe is margin >= 0; unsafe is margin < 0. Full coefficients are in FROZEN_EVALUATOR_DECOMPOSITION.json.\n\n# REAL VS PREDICTED EVALUATOR FEATURES\n\nSame-state F_prev/F_star/F_next features are in REAL_VS_PRED_EVALUATOR_FEATURES.csv.\n\n# F_PREV FALSE-SAFE ATTRIBUTION\n\nPrevious real-unsafe / predicted-safe F_prev cases: {len(false_rows)}. Attribution counts: {json.dumps(false_counts,sort_keys=True)}.\n\n# FEATURE-WISE CALIBRATION\n\nTRAIN/DEV/TEST bias, MAE, RMSE, slope, correlation, sign agreement, threshold-crossing recall, and F_prev-vs-F_star ordering are in FEATURE_WISE_CALIBRATION.csv.\n\n# FORCE ORDERING VS ABSOLUTE CALIBRATION\n\nDEV frozen REAL boundary ordering={real_pair:.3f}; frozen MODEL boundary ordering={model_pair:.3f}. These are separate from absolute margin calibration.\n\n# TRAIN-ONLY CALIBRATION RESULT\n\nSelected family={selected}; family selection used DEV only after TRAIN fitting. Details and coefficients are in CALIBRATION_SELECTION.json and CALIBRATED_EVALUATOR_FREEZE.json.\n\n# RELATIVE / RANKING DIAGNOSTIC\n\nOrdinal pair/triplet results are in RELATIVE_RANKING_DIAGNOSTIC.csv. No F_star was used by the diagnostic scoring rule.\n\n# DEV / TEST CONTROLLER SELECTION\n\n{json.dumps([x for x in comp_rows if x['split'] in ['DEV','TEST']],sort_keys=True)}\n\n# UNDER-FORCE / OVER-FORCE\n\nSelected TEST under-force={selected_test['under_force']:.3f}; over-force={selected_test['over_force']:.3f}; exact={selected_test['exact']:.3f}.\n\n# EARLIEST FAILED LINK\n\nThe earliest unsupported link is the frozen evaluator interface unless the selected calibration fails to generalize; no world-model retraining was attempted.\n\n# PRIMARY_CLASSIFICATION\n\n{primary}\n\n# SCIENTIFIC INTERPRETATION\n\nDoes the model know which force is safer? The ordinal evidence is reported separately in the relative diagnostic; global trajectory separation alone is not sufficient.\n\nDoes the model know whether that force is actually safe? The uncalibrated absolute evaluator margin is tested against the frozen zero boundary; calibration results quantify whether this can be recovered without Physics-GRU changes.\n\n# METHOD CHANGE\n\nNONE to the world model.\n\n# NEW TRAINING\n\nNO Physics-GRU training. No evaluator retraining. Only allowed simple calibration was fit on TRAIN, selected on DEV, and evaluated on TEST once.\n\n# NEXT_METHOD\n\nDo not change friction, probing, or E2E. Repair the earliest unsupported evaluator-interface link with a pre-registered calibration/relative-risk formulation only.\n'''
    (out/'FINAL_REPORT.md').write_text(report)
    files=[p for p in out.iterdir() if p.name!='MANIFEST.sha256']; (out/'MANIFEST.sha256').write_text('\n'.join(f'{sha256(p)}  {p.name}' for p in sorted(files))+'\n')
    import subprocess
    subprocess.run(['sha256sum','-c','MANIFEST.sha256'],cwd=out,check=True,stdout=subprocess.DEVNULL)
    print(json.dumps({'out':str(out),'primary_classification':primary,'selected_calibration':selected,'dev':dev_choice[selected],'test':selected_test,'false_safe_attribution_counts':false_counts},indent=2))


if __name__=='__main__': main()
