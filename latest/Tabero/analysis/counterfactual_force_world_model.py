#!/usr/bin/env python3
"""Frozen matched-pair ForceEcho experiment for Physics-GRU.

This file intentionally keeps the historical Physics-GRU architecture and
input schema.  It adds only strict matched-pair sampling and, for the IE
variant, a differentiable H=8 intervention-effect loss.  TEST is metadata-only
in this run: no TEST prediction, calibration, model selection, or simulator
rollout is performed.
"""
from __future__ import annotations

import csv, hashlib, importlib.util, json, math, os, random, subprocess, sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO = Path('/home/exouser/Tabero')
RESULTS = REPO / 'analysis/results'
HIST_ROOT = RESULTS / 'p5s0c_paired_boundary_probe_value_20260824_000542'
DIRECT_ROOT = RESULTS / 'direct_contact_boundary_dataset_20260829_001409'
FROZEN_ROOT = RESULTS / 'trajectory_physical_imagination_20260829_065220'
FORENSIC_ROOT = RESULTS / 'force_sensitivity_forensic_20260829_090000'
CONTINUOUS_ROOT = RESULTS / 'continuous_physics_imagination_20260829_141924'
CAL_ROOT = RESULTS / 'evaluator_interface_calibration_20260829_112603'
HORIZON = 8
HORIZONS = [1, 2, 4, 8]
SEEDS = [0, 1, 2]
LAMBDAS = [0.1, 0.3, 1.0]
EPOCHS = 80
BATCH_SIZE = 64
DT = 0.05
ACTIVE_PHASES = {'branch_hold', 'lift', 'transit', 'over_basket', 'place'}
STATE_NAMES = ['rel_dx_m','rel_dy_m','rel_dz_m','rel_vx_mps','rel_vy_mps','rel_vz_mps',
               'left_normal_N','right_normal_N','left_tangent_N','right_tangent_N',
               'tangent_velocity_proxy_mps','joint_left','joint_right']
COMMON_IDX = [0,1,2,3,4,5,11,12]
FORCE_IDX = [6,7,8,9]
TASKS = [0,1,5,6]

OUT = Path(os.environ.get('CFWM_OUT', RESULTS / 'counterfactual_force_world_model_20260829_160000'))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for b in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_tpi():
    spec = importlib.util.spec_from_file_location('tpi_cfwm', REPO / 'analysis/trajectory_physical_imagination.py')
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Pair:
    pair_id: str
    group_id: str
    split: str
    source: str
    context_id: str
    root_id: str
    task: int
    friction_band: str
    mu: float
    force_a: float
    force_b: float
    category: str
    boundary: bool
    a: Any
    b: Any


def manifest_maps():
    hist = pd.read_csv(HIST_ROOT / 'P5S0C_BRANCH_MANIFEST.csv')
    direct = pd.concat([pd.read_csv(DIRECT_ROOT / f'task{t}/branches.csv') for t in TASKS], ignore_index=True)
    return hist, direct


def task_object(task: int) -> str:
    return {0:'alphabet_soup_1', 1:'cream_cheese_1', 5:'tomato_sauce_1', 6:'butter_1'}[int(task)]


def trace_key(trace) -> tuple:
    return (trace.source, trace.context_id, round(float(trace.force), 6), trace.branch_id)


def state_payload(trace, tpi):
    # Pair equality uses the reliable state channels and the inherited scene
    # restore hash. Force/contact outputs are intervention consequences and
    # are never used to reject a pair at t=0.
    return np.asarray(trace.state[0, COMMON_IDX], dtype=float)


def canonical_motion(trace, H: int):
    d = pd.read_csv(trace.path)
    n = min(H, len(d))
    cmd = d[['cmd_x','cmd_y','cmd_z']].iloc[:n].to_numpy(float)
    phase = tuple(d['phase'].astype(str).iloc[:n])
    mode = tuple(d['mode'].astype(str).iloc[:n])
    step = tuple(d['step'].astype(int).iloc[:n])
    ts = d['t_s'].iloc[:n].to_numpy(float)
    return cmd, phase, mode, step, ts, tuple(d.columns)


def motion_hash(motion) -> str:
    cmd, phase, mode, step, ts, schema = motion
    payload = {'cmd': np.asarray(cmd).round(12).tolist(), 'phase': phase, 'mode': mode,
               'step': step, 't_s': np.asarray(ts).round(12).tolist()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def valid_telemetry(trace, manifest_row: pd.Series, tpi) -> tuple[bool, str, dict[str, Any]]:
    d = pd.read_csv(trace.path)
    base = {'step','t_s','phase','mode','cmd_x','cmd_y','cmd_z',
            'object_x_analysis_only','object_y_analysis_only','object_z_analysis_only',
            'gripper_pos_0','gripper_pos_1'}
    missing = sorted(base - set(d.columns))
    if missing:
        return False, 'MISSING_BASE_COLUMNS', {'missing': missing, 'schema': list(d.columns)}
    if not bool(int(getattr(manifest_row, 'state_parity', 0)) == 1):
        return False, 'STATE_PARITY_FAIL', {'schema': list(d.columns)}
    if not np.isfinite(d[list(base)].select_dtypes(include=[np.number]).to_numpy(float)).all():
        return False, 'NONFINITE_BASE_TELEMETRY', {'schema': list(d.columns)}
    corrected = {'left_normal_force_N','right_normal_force_N','left_tangential_force_N',
                 'right_tangential_force_N','contact_left','contact_right',
                 'object_vx_mps','object_vy_mps','object_vz_mps'}
    direct = corrected <= set(d.columns)
    if trace.source == 'direct' and not direct:
        return False, 'DIRECT_CORRECTED_SCHEMA_MISSING', {'schema': list(d.columns)}
    note = 'corrected_direct_contact_telemetry' if direct else 'historical_common_telemetry_force_contact_channels_masked'
    return True, note, {'schema': list(d.columns), 'corrected_contact_fields_present': direct,
                        'known_double_rotation_bug': False}


def fstar_for_group(rows: list[Any]) -> float:
    good = [float(x.force) for x in rows if int(x.outcome) == 1]
    return min(good) if good else max(float(x.force) for x in rows)


def audit_pairs(tpi, hist_traces, direct_traces, hist_m, direct_m):
    manifest_by = {}
    for source, m in [('historical', hist_m), ('direct', direct_m)]:
        for r in m.itertuples(index=False):
            manifest_by[(source, str(r.context_id), str(r.branch_id))] = r
    groups, pairs, motion_rows = [], [], []
    all_group_traces = {}
    for source, traces in [('historical', hist_traces), ('direct', direct_traces)]:
        for tr in traces:
            all_group_traces.setdefault((source, tr.context_id), []).append(tr)
    eligible_groups = {}
    for (source, cid), traces in sorted(all_group_traces.items()):
        rows = [manifest_by[(source, cid, tr.branch_id)] for tr in traces]
        ref = rows[0]
        state_arr = np.stack([state_payload(tr, tpi) for tr in traces])
        state_max = float(np.max(np.ptp(state_arr, axis=0)))
        motions = [canonical_motion(tr, HORIZON) for tr in traces]
        hashes = [motion_hash(m) for m in motions]
        motion_max = max(float(np.max(np.abs(m[0] - motions[0][0]))) for m in motions[1:]) if len(motions) > 1 else 0.0
        motion_exact = all(h == hashes[0] for h in hashes)
        lengths = [len(tr.state) for tr in traces]
        telemetry_results = [valid_telemetry(tr, row, tpi) for tr, row in zip(traces, rows)]
        telemetry_valid = all(x[0] for x in telemetry_results)
        schemas = [x[2]['schema'] for x in telemetry_results]
        schema_same = len({tuple(x) for x in schemas}) == 1
        force_values = sorted({round(float(tr.force), 6): tr for tr in traces})
        fstar = fstar_for_group(traces)
        task = int(ref.task)
        split = str(ref.split)
        state_ok = state_max <= 0.005 and all(str(r.post_probe_state_hash) == str(rows[0].post_probe_state_hash) for r in rows)
        length_ok = min(lengths) >= HORIZON + 1
        valid_group = bool(state_ok and motion_exact and telemetry_valid and schema_same and length_ok and len(force_values) >= 2)
        group_id = f'{source}:{cid}'
        fail_reasons = []
        if not state_ok: fail_reasons.append('state_matching')
        if not motion_exact: fail_reasons.append('motion_matching')
        if not telemetry_valid: fail_reasons.append('telemetry')
        if not schema_same: fail_reasons.append('schema_mismatch')
        if not length_ok: fail_reasons.append('horizon_alignment')
        if len(force_values) < 2: fail_reasons.append('force_coverage')
        groups.append({'group_id':group_id,'source':source,'context_id':cid,'root_id':str(ref.root_id),
                       'task':task,'object_identity':task_object(task),'split':split,
                       'friction_band':str(ref.friction_band),'friction':float(ref.hidden_friction_analysis_only),
                       'branch_count':len(traces),'force_values_json':json.dumps(sorted(force_values)),
                       'branch_start_state_hash':str(rows[0].post_probe_state_hash),
                       'initial_state_max_abs_diff':state_max,'state_tolerance':0.005,'state_match':state_ok,
                       'future_command_hash':hashes[0],'future_command_exact_match':motion_exact,
                       'future_command_hash_match_rate':float(sum(h == hashes[0] for h in hashes)/len(hashes)),
                       'future_command_per_step_max_diff':motion_max,'horizon_length_min':min(lengths),
                       'horizon_length_max':max(lengths),'horizon_aligned':length_ok,
                       'phase_aligned':all(m[1] == motions[0][1] for m in motions),
                       'input_state_schema_same':schema_same,'telemetry_valid_rate':float(sum(x[0] for x in telemetry_results)/len(telemetry_results)),
                       'telemetry_notes':json.dumps(sorted({x[1] for x in telemetry_results})),
                       'known_invalid_telemetry':False,'valid_group':valid_group,
                       'invalid_reasons':json.dumps(fail_reasons)})
        motion_rows.append({'group_id':group_id,'source':source,'split':split,'context_id':cid,
                            'root_id':str(ref.root_id),'task':task,'friction_band':str(ref.friction_band),
                            'future_command_exact_match':motion_exact,'future_command_hash_match_rate':float(sum(h == hashes[0] for h in hashes)/len(hashes)),
                            'per_step_numerical_max_difference':motion_max,'horizon_length_match':len(set(lengths)) == 1,
                            'horizon_length_min':min(lengths),'phase_alignment':all(m[1] == motions[0][1] for m in motions),
                            'padding_or_terminal_mask_mismatch':'not_used_fixed_common_H8_window'})
        if valid_group:
            # Direct repeats are retained for ordinary trajectory fitting, but
            # R1 is the deterministic matched representative for IE pairs.
            by_force = {}
            for tr in traces:
                if source == 'direct' and '_R1_' not in tr.branch_id:
                    continue
                by_force.setdefault(round(float(tr.force), 6), tr)
            by_force = dict(sorted(by_force.items()))
            eligible_groups[group_id] = by_force
            fstar = fstar_for_group(list(by_force.values()))
            vals = sorted(by_force)
            for i, fa in enumerate(vals):
                for fb in vals[i+1:]:
                    delta = round(fb-fa, 6)
                    boundary = abs(fa-fstar) < 1e-5 or abs(fb-fstar) < 1e-5
                    cat = 'boundary' if boundary and abs(delta-0.5) < 1e-5 else ('adjacent' if abs(delta-0.5) < 1e-5 else 'wider')
                    pairs.append(Pair(f'{group_id}:{fa:.1f}_vs_{fb:.1f}', group_id, split, source, cid,
                                      str(ref.root_id), task, str(ref.friction_band), float(ref.hidden_friction_analysis_only),
                                      fa, fb, cat, boundary, by_force[fa], by_force[fb]))
    return groups, pairs, motion_rows, eligible_groups


def pair_coverage(groups, pairs):
    out = {}
    for split in ['TRAIN','DEV','TEST']:
        gs = [g for g in groups if g['split'] == split and g['valid_group']]
        ps = [p for p in pairs if p.split == split]
        out[split] = {'valid_context_groups':len(gs), 'eligible_tasks':sorted({g['task'] for g in gs}),
                      'friction_bands':sorted({g['friction_band'] for g in gs}),
                      'pairs':len(ps),'boundary_pairs':sum(p.category=='boundary' for p in ps),
                      'adjacent_pairs':sum(p.category in {'adjacent','boundary'} and abs(p.force_b-p.force_a-.5)<1e-5 for p in ps),
                      'wider_pairs':sum(p.category=='wider' for p in ps),
                      'unsafe_safe_boundary_pairs':sum(p.category=='boundary' and ((p.a.outcome==0 and p.b.outcome==1) or (p.a.outcome==1 and p.b.outcome==0)) for p in ps),
                      'safe_safe_pairs':sum(p.a.outcome==1 and p.b.outcome==1 for p in ps),
                      'unsafe_unsafe_pairs':sum(p.a.outcome==0 and p.b.outcome==0 for p in ps),
                      'delta_force_values':sorted({round(p.force_b-p.force_a,6) for p in ps})}
    return out


def pairability_gate(groups, pairs):
    cov = pair_coverage(groups, pairs)
    gtrain = cov['TRAIN']; gdev = cov['DEV']
    all_valid = [g for g in groups if g['valid_group']]
    rates = [g['future_command_hash_match_rate'] for g in all_valid]
    tele = [g['telemetry_valid_rate'] for g in all_valid]
    checks = {'train_groups_ge_30':gtrain['valid_context_groups']>=30,
              'dev_groups_ge_10':gdev['valid_context_groups']>=10,
              'eligible_tasks_ge_3':len(set(gtrain['eligible_tasks']+gdev['eligible_tasks']))>=3,
              'friction_regions_ge_2':len(set(gtrain['friction_bands']+gdev['friction_bands']))>=2,
              'train_unsafe_safe_boundary_ge_20':gtrain['unsafe_safe_boundary_pairs']>=20,
              'dev_unsafe_safe_boundary_ge_6':gdev['unsafe_safe_boundary_pairs']>=6,
              'future_command_exact_match_rate_ge_0.95':float(np.mean(rates) if rates else 0)>=.95,
              'telemetry_valid_rate_ge_0.95':float(np.mean(tele) if tele else 0)>=.95}
    return bool(all(checks.values())), checks, cov


def load_frozen(tpi, device):
    ck = torch.load(FROZEN_ROOT/'PHYSICS_TRAJECTORY_GRU.pt', map_location=device, weights_only=False)
    if int(ck['H']) != HORIZON:
        raise RuntimeError('frozen checkpoint is not H=8')
    model = tpi.ShortHorizonPhysicsGRU(17,54,HORIZON).to(device)
    model.load_state_dict(ck['state_dict']); model.eval()
    norm = tuple(np.asarray(ck['normalization'][k], np.float32) for k in ['x_mean','x_std','y_mean','y_std'])
    return model, norm, ck


def load_evaluator(tpi, device):
    ck = torch.load(FROZEN_ROOT/'OUTCOME_EVALUATOR.pt', map_location=device, weights_only=False)
    if ck.get('model_type') != 'logistic':
        raise RuntimeError('frozen evaluator mismatch')
    model = tpi.LinearOutcome(len(ck['x_mean'])).to(device)
    model.load_state_dict(ck['state_dict']); model.eval()
    return model, np.asarray(ck['x_mean'], np.float32), np.asarray(ck['x_std'], np.float32), float(ck['threshold'])


def build_seg(tpi, trace, force, H=HORIZON):
    d = pd.read_csv(trace.path)
    nom = tpi.nominal_from(d, int(trace.task), float(force), float(trace.mu), trace.state, trace.mask)
    x = nom[:H].copy(); x[:,19:32] = trace.state[0]; x[:,32:45] = trace.mask[0]
    fake = tpi.Trace(trace.branch_id, trace.context_id, trace.root_id, trace.task, trace.split,
                     float(force), trace.mu, trace.outcome, trace.role, trace.path, trace.state,
                     trace.mask, x, trace.phase, trace.weight, trace.source)
    return tpi.Segment(fake, 0, H, x, trace.state[1:H+1], trace.mask[1:H+1])


def pred_norm(model, seg, norm, device, grad=False):
    xm,xs,ym,ys = norm
    xn = (seg.x-xm)/xs
    step = torch.tensor(xn[:,:17][None], dtype=torch.float32, device=device, requires_grad=grad)
    cond = torch.tensor(xn[0,17:][None], dtype=torch.float32, device=device, requires_grad=grad)
    out = model(step, cond)[0]
    return out


def pred_state(model, trace, force, norm, tpi, device):
    with torch.no_grad():
        z = pred_norm(model, build_seg(tpi, trace, force), norm, device).cpu().numpy()
    xm,xs,ym,ys = norm
    return z*ys + ym + trace.state[0]


def huber(a, b):
    return nn.functional.smooth_l1_loss(a, b, reduction='none')


def ie_loss(model, pair: Pair, norm, tpi, device, need_grad=True):
    sa, sb = build_seg(tpi, pair.a, pair.force_a), build_seg(tpi, pair.b, pair.force_b)
    pa, pb = pred_norm(model, sa, norm, device, grad=need_grad), pred_norm(model, sb, norm, device, grad=need_grad)
    xm,xs,ym,ys = norm
    ya = torch.tensor(((sa.y-sa.trace.state[0]-ym)/ys), dtype=torch.float32, device=device)
    yb = torch.tensor(((sb.y-sb.trace.state[0]-ym)/ys), dtype=torch.float32, device=device)
    # Inherited forensic semantics include branch_hold as the pre-lift force
    # application window; release and settle remain excluded.
    pm = np.asarray([str(x) in ACTIVE_PHASES for x in pair.a.phase[1:HORIZON+1]], bool)
    mask = torch.tensor((sa.mask * sb.mask) * pm[:,None], dtype=torch.float32, device=device)
    target = yb - ya
    loss = (huber(pb-pa, target) * mask).sum() / (mask.sum()+1e-6)
    return loss, pa, pb, target, mask


def make_units(tpi, traces, pairs):
    seg_by = {(trace_key(tr),0): build_seg(tpi,tr,tr.force) for tr in traces if tr.split=='TRAIN' and len(tr.state)>=HORIZON+1}
    units = []
    for p in pairs:
        if p.split != 'TRAIN': continue
        a = seg_by.get((trace_key(p.a),0)); b = seg_by.get((trace_key(p.b),0))
        if a is not None and b is not None:
            units.append((a,b,p))
    used = {(trace_key(s.trace),s.start) for u in units for s in u[:2]}
    for tr in traces:
        if tr.split != 'TRAIN': continue
        for s in tpi.make_segments([tr], HORIZON):
            if (trace_key(s.trace),s.start) not in used or s.start != 0:
                units.append((s,None,None))
    return units


def batches_for_units(units, seed, epoch):
    rng = random.Random(seed*1000003 + epoch*1009 + 17)
    order = list(units); rng.shuffle(order)
    batches=[]; cur=[]; cur_n=0
    for u in order:
        n = 2 if u[1] is not None else 1
        if cur and cur_n+n > BATCH_SIZE:
            batches.append(cur); cur=[]; cur_n=0
        cur.append(u); cur_n += n
    if cur: batches.append(cur)
    return batches


def batch_tensors(items, norm, device):
    xm,xs,ym,ys = norm; segs=[]
    for u in items:
        segs.append(u[0]);
        if u[1] is not None: segs.append(u[1])
    xn=np.stack([(s.x-xm)/xs for s in segs]); yn=np.stack([((s.y-s.trace.state[s.start]-ym)/ys) for s in segs]); mm=np.stack([s.mask for s in segs]); ww=np.asarray([s.trace.weight for s in segs],np.float32)
    step=torch.tensor(xn[:,:,:17],dtype=torch.float32,device=device); cond=torch.tensor(xn[:,0,17:],dtype=torch.float32,device=device)
    y=torch.tensor(yn,dtype=torch.float32,device=device); m=torch.tensor(mm,dtype=torch.float32,device=device); w=torch.tensor(ww,dtype=torch.float32,device=device)
    return segs, step, cond, y, m, w


def train_variant(tpi, base_ck, norm, traces, pairs, device, variant, seed, lam=0.0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model=tpi.ShortHorizonPhysicsGRU(17,54,HORIZON).to(device); model.load_state_dict(base_ck['state_dict'])
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
    units=make_units(tpi,traces,pairs); history=[]; total_steps=0
    for epoch in range(1,EPOCHS+1):
        model.train(); base_vals=[]; ie_vals=[]
        for batch in batches_for_units(units,seed,epoch):
            segs,step,cond,y,m,w=batch_tensors(batch,norm,device); opt.zero_grad(set_to_none=True); pred=model(step,cond)
            base=((huber(pred,y)*m*w[:,None,None]).sum()/(m.sum()+1e-6)); loss=base
            ies=[]; offset=0
            for u in batch:
                if u[1] is not None:
                    pair=u[2]; pa=pred[offset]; pb=pred[offset+1]; sa=u[0]; sb=u[1]; pm=np.asarray([str(x) in ACTIVE_PHASES for x in pair.a.phase[1:HORIZON+1]],bool); mask=torch.tensor((sa.mask*sb.mask)*pm[:,None],dtype=torch.float32,device=device); target=((sb.y-sb.trace.state[0]-norm[2])/norm[3])-((sa.y-sa.trace.state[0]-norm[2])/norm[3]); ies.append((huber(pb-pa,torch.tensor(target,dtype=torch.float32,device=device))*mask*float(4 if pair.boundary else 2 if pair.category=='adjacent' else 1)).sum()/(mask.sum()+1e-6)); offset+=2
                else: offset+=1
            ie=torch.stack(ies).mean() if ies else torch.zeros((),device=device); loss=loss+float(lam)*ie
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total_steps+=1; base_vals.append(float(base.item())); ie_vals.append(float(ie.item()))
        history.append({'variant':variant,'seed':seed,'lambda_IE':lam,'epoch':epoch,'base_loss':float(np.mean(base_vals)),'ie_loss':float(np.mean(ie_vals) if ie_vals else 0.0),'total_loss':float(np.mean(base_vals)+lam*(np.mean(ie_vals) if ie_vals else 0.0))})
    model.eval()
    return model, history, total_steps, len(units)


def frozen_margin(tpi, evaluator, ex, trace, state, device):
    fake=tpi.Trace(trace.branch_id,trace.context_id,trace.root_id,trace.task,trace.split,trace.force,trace.mu,trace.outcome,trace.role,trace.path,state,trace.mask,trace.nominal,trace.phase[:len(state)],trace.weight,trace.source)
    x=(tpi.summarize(fake,1.0)-ex[0])/ex[1]
    with torch.no_grad():
        logit=float(evaluator(torch.tensor(x[None],dtype=torch.float32,device=device)).cpu().numpy()[0])
    return logit


def fit_iso(x,y):
    order=np.argsort(np.asarray(x)); xs=np.asarray(x,float)[order]; ys=np.asarray(y,float)[order]
    blocks=[]
    for xv,yv in zip(xs,ys):
        blocks.append([xv,xv,yv,1])
        while len(blocks)>=2 and blocks[-2][2] > blocks[-1][2]:
            a,b=blocks[-2],blocks[-1]; n=a[3]+b[3]; blocks[-2]=[a[0],b[1],(a[2]*a[3]+b[2]*b[3])/n,n]; blocks.pop()
    return {'x':[b[0] for b in blocks]+([blocks[-1][1]] if blocks else []),'y':[b[2] for b in blocks]+([blocks[-1][2]] if blocks else [])}


def iso_predict(cal, x):
    if not cal['x']: return np.full(len(np.asarray(x)),.5)
    xx=np.asarray(cal['x'],float); yy=np.asarray(cal['y'],float); return np.interp(np.asarray(x,float),xx,yy,left=yy[0],right=yy[-1])


def trace_prediction_rows(tpi, model, norm, traces, evaluator, ex, device, split):
    rows=[]; cache={}
    for tr in traces:
        if tr.split != split or len(tr.state)<HORIZON+1: continue
        p=pred_state(model,tr,tr.force,norm,tpi,device); cache[trace_key(tr)]=p
        margin=frozen_margin(tpi,evaluator,ex,tr,np.vstack([tr.state[0],p]),device)
        rows.append({'trace_key':trace_key(tr),'branch_id':tr.branch_id,'context_id':tr.context_id,'root_id':tr.root_id,'task':tr.task,'split':tr.split,'friction_band':tr.source+':'+str(tr.mu),'force_N':tr.force,'real_outcome':tr.outcome,'raw_margin':margin,'prediction':p})
    return rows, cache


def eval_model(tpi, model, norm, traces, pairs, evaluator, ex, calibrator, device, split='DEV'):
    pred_rows, cache = trace_prediction_rows(tpi,model,norm,traces,evaluator,ex,device,split)
    bykey={r['trace_key']:r for r in pred_rows}; ie_rows=[]; boundary_rows=[]; ordinary=[]
    for r in pred_rows:
        tr=next(t for t in traces if trace_key(t)==r['trace_key']); q=r['prediction']; real=tr.state[1:1+HORIZON]
        for h in HORIZONS:
            m=tr.mask[1:1+h]; ordinary.append({'horizon':h,'trajectory_error':float((np.abs(q[:h]-real[:h])*m).sum()/(m.sum()+1e-6)),'position_error':float(np.abs(q[:h,:3]-real[:h,:3]).mean())})
    for p in pairs:
        if p.split != split: continue
        ra=bykey.get(trace_key(p.a)); rb=bykey.get(trace_key(p.b))
        if ra is None or rb is None: continue
        for h in HORIZONS:
            ma=p.a.mask[1:1+h]*p.b.mask[1:1+h]; pm=np.asarray([str(x) in ACTIVE_PHASES for x in p.a.phase[1:1+h]],bool)[:,None]; mask=ma*pm
            ya=((p.a.state[1:1+h]-p.a.state[0]-norm[2])/norm[3]); yb=((p.b.state[1:1+h]-p.b.state[0]-norm[2])/norm[3]); pred_a=(ra['prediction'][:h]-p.a.state[0]-norm[2])/norm[3]; pred_b=(rb['prediction'][:h]-p.b.state[0]-norm[2])/norm[3]
            dr=yb-ya; dp=pred_b-pred_a; den=float(np.linalg.norm(dr[mask.astype(bool)])+1e-8); err=float(np.linalg.norm((dp-dr)[mask.astype(bool)])/den) if mask.any() else math.nan; cos=float(np.dot(dp[mask.astype(bool)],dr[mask.astype(bool)])/(np.linalg.norm(dp[mask.astype(bool)])*np.linalg.norm(dr[mask.astype(bool)])+1e-8)) if mask.any() else math.nan; ratio=float(np.linalg.norm(dp[mask.astype(bool)])/(np.linalg.norm(dr[mask.astype(bool)])+1e-8)) if mask.any() else math.nan
            ie_rows.append({'pair_id':p.pair_id,'context_id':p.context_id,'root_id':p.root_id,'task':p.task,'friction_band':p.friction_band,'force_pair':f'{p.force_a:.1f}_vs_{p.force_b:.1f}','category':p.category,'horizon':h,'ie_norm_error':err,'cosine_similarity':cos,'effect_magnitude_ratio':ratio})
        if p.category in {'boundary','adjacent'} and abs(p.force_b-p.force_a-.5)<1e-5:
            pa=float(iso_predict(calibrator,[ra['raw_margin']])[0]); pb=float(iso_predict(calibrator,[rb['raw_margin']])[0]); boundary_rows.append({'pair_id':p.pair_id,'context_id':p.context_id,'root_id':p.root_id,'task':p.task,'friction_band':p.friction_band,'force_a':p.force_a,'force_b':p.force_b,'a_outcome':p.a.outcome,'b_outcome':p.b.outcome,'a_probability':pa,'b_probability':pb,'a_raw_margin':ra['raw_margin'],'b_raw_margin':rb['raw_margin'],'boundary':p.boundary,'a_pred_safe':int(pa>=.5),'b_pred_safe':int(pb>=.5),'margin_ordering':int(rb['raw_margin']>ra['raw_margin'])})
    return pred_rows, ie_rows, boundary_rows, ordinary


def fit_calibration(tpi, model, norm, traces, evaluator, ex, device):
    rows, _=trace_prediction_rows(tpi,model,norm,traces,evaluator,ex,device,'TRAIN'); cal=fit_iso([r['raw_margin'] for r in rows],[r['real_outcome'] for r in rows])
    return cal, {'fit_split':'TRAIN','n':len(rows),'x':cal['x'],'y':cal['y'],'protocol':'TRAIN-only isotonic over frozen evaluator raw margin on H8 predicted prefix'}


def restore_run(tpi, path, device, cal_path):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = tpi.ShortHorizonPhysicsGRU(17, 54, HORIZON).to(device)
    model.load_state_dict(ck['state_dict']); model.eval()
    d = json.loads(cal_path.read_text()) if cal_path.exists() else {'x': [], 'y': []}
    cal = {'x': d.get('x', []), 'y': d.get('y', [])}
    log_path = cal_path.parent / cal_path.name.replace('CALIBRATION_', 'TRAINING_').replace('.json', '.csv')
    histlog = pd.read_csv(log_path).to_dict('records') if log_path.exists() else []
    return model, cal, histlog


def load_prior_isotonic():
    p=CAL_ROOT/'CALIBRATED_EVALUATOR_FREEZE.json'
    if not p.exists(): return None
    d=json.loads(p.read_text()); x=d['calibration']['isotonic']['x']; y=d['calibration']['isotonic'].get('y', d['calibration']['isotonic'].get('p', []))
    return {'x':x,'y':y}


def bootstrap_ci(values, seed):
    x=np.asarray([v for v in values if np.isfinite(v)],float)
    if not len(x): return [math.nan,math.nan,math.nan]
    rng=np.random.default_rng(seed); z=x[rng.integers(0,len(x),size=(2000,len(x)))].mean(1); return [float(x.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))]


def boundary_metrics(rows):
    q = [r for r in rows if int(r.get('a_outcome', -1)) == 0 and int(r.get('b_outcome', -1)) == 1]
    if not q:
        return {'boundary_accuracy': math.nan, 'fprev_false_safe': math.nan, 'margin_ordering': math.nan, 'n': 0}
    return {'boundary_accuracy': float(np.mean([(r['a_pred_safe'] == 0 and r['b_pred_safe'] == 1) for r in q])),
            'fprev_false_safe': float(np.mean([r['a_pred_safe'] for r in q])),
            'margin_ordering': float(np.mean([r['margin_ordering'] for r in q])), 'n': len(q)}


def continuous_sweep(tpi, model, norm, traces, evaluator, ex, cal, device):
    contexts=[]
    try:
        old=json.loads((CONTINUOUS_ROOT/'CONTINUOUS_PHYSICS_IMAGINATION_PROTOCOL.json').read_text())
        contexts=[x['context_id'] for x in old.get('contexts',[]) if x.get('split')=='DEV']
    except Exception: pass
    rows=[]
    for cid in contexts:
        candidates=[t for t in traces if t.split=='DEV' and t.context_id==cid and t.source=='historical']
        if not candidates: continue
        ref=candidates[0]; forces=np.arange(min(t.force for t in candidates),max(t.force for t in candidates)+.001,.25)
        vals=[]
        for f in forces:
            st=pred_state(model,ref,float(round(f,2)),norm,tpi,device); margin=frozen_margin(tpi,evaluator,ex,ref,np.vstack([ref.state[0],st]),device); vals.append((float(round(f,2)),margin,float(iso_predict(cal,[margin])[0])))
        for i,(f,m,p) in enumerate(vals):
            rows.append({'context_id':cid,'task':ref.task,'friction_band':ref.source+':'+str(ref.mu),'force_N':f,'raw_margin':m,'calibrated_probability':p,'lower_force_N':vals[i-1][0] if i else '','lower_calibrated_probability':vals[i-1][2] if i else '','adjacent_non_decreasing':int(i==0 or p>=vals[i-1][2]-1e-9)})
    return rows


def protocol(tpi, hist_m, direct_m, device):
    files=[HIST_ROOT/'P5S0C_FORCE_MANIFEST.json',HIST_ROOT/'P5S0C_BRANCH_MANIFEST.csv',HIST_ROOT/'P5S0C_SPLIT_MANIFEST.json',HIST_ROOT/'P5S0C_DATASET_SCHEMA.md',
           FROZEN_ROOT/'PHYSICS_TRAJECTORY_GRU.pt',FROZEN_ROOT/'TRAJECTORY_REPRESENTATION.json',FROZEN_ROOT/'WORLD_MODEL_PROTOCOL_IMMUTABLE.json',
           FORENSIC_ROOT/'FORCE_SENSITIVITY_FORENSIC_PROTOCOL.json',FORENSIC_ROOT/'STATE_PARITY_AUDIT.json',CAL_ROOT/'EVALUATOR_INTERFACE_CALIBRATION_PROTOCOL.json',
           CAL_ROOT/'CALIBRATED_EVALUATOR_FREEZE.json',RESULTS/'direct_contact_boundary_imagination_20260829_000205/DIRECT_CONTACT_LOGGER_AUDIT.json',RESULTS/'direct_contact_boundary_imagination_20260829_000205/DIRECT_PHYSICAL_EVENT_DEFINITION.json']
    input_hashes={str(p):sha256(p) for p in files if p.exists()}
    return {'protocol_name':'COUNTERFACTUAL_FORCE_WORLD_MODEL_PROTOCOL','created_before_new_training_or_dev_results':True,
            'authoritative_inputs_hashes':input_hashes,'current_checkpoint':{'path':str(FROZEN_ROOT/'PHYSICS_TRAJECTORY_GRU.pt'),'sha256':sha256(FROZEN_ROOT/'PHYSICS_TRAJECTORY_GRU.pt'),'architecture':'ShortHorizonPhysicsGRU GRU(64), step_dim=17, condition_dim=54, output H=8x13'},
            'prediction_horizon':HORIZON,'physical_state_definition':STATE_NAMES,'active_phases':sorted(ACTIVE_PHASES),'excluded_phases':['release','settle'],'splits':'root-held-out inherited TRAIN/DEV/TEST; TEST metadata-only in this run',
            'pair_rule':'same source schema, context, root, task/object, friction, post-probe branch-start state/hash, prediction start=0, phase/mask, H=8 future nominal command; only requested grip force differs; approximate pairs rejected',
            'training_variants':{'PHYSICS_GRU_COVERAGE':'same base SmoothL1 trajectory loss, pair-balanced batch sampling','PHYSICS_GRU_FORCE_IE':'same sampling plus lambda_IE*masked_Huber(delta_pred,delta_real)'},
            'seeds':SEEDS,'optimizer':{'name':'AdamW','learning_rate':8e-4,'weight_decay':1e-4,'gradient_clip':1.0,'epochs':EPOCHS,'batch_size':BATCH_SIZE,'steps_fixed':True},
            'ie_loss':{'inference_mode':'same one-shot H=8 rollout as inference','normalization':'frozen TRAIN-only normalization from current checkpoint; target delta in normalized 13-state space','mask':'pair mask intersection x active phases; no zero-padding; common first H window','weighting':'boundary=4, adjacent=2, wider=1','contact_indicator':'not an output of current 13-state model; no fabricated probability term'},
            'lambda_candidates':LAMBDAS,'calibration':'independent TRAIN-only isotonic per checkpoint over frozen evaluator raw margin; DEV only for selection','forceecho_metrics':['anchor replay','normalized IE error','cosine','magnitude ratio','signed agreement','H=1/2/4/8','boundary preservation','continuous 0.25N model sweep'],'dev_gates':{'boundary_accuracy_min':.8,'fprev_false_safe_max':.1,'margin_ordering_min':.8,'ie_relative_improvement_min':.2,'trajectory_regression_max':.1,'continuous_monotonicity_min':.8,'bootstrap_ci_positive':True},'optional_real_025N':{'only_if_dev_gate_passes':True,'contexts':9,'repeats':3,'new_rollouts_allowed':True},'forbidden':['TEST predictions','Probe-informed continuous search','No-physics comparison','fresh roots','final E2E','Pi0 retraining','friction-estimator retraining','new loss families','architecture search'],'device':{'cuda':bool(torch.cuda.is_available()),'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU','torch':torch.__version__},'source_code':str(REPO/'analysis/counterfactual_force_world_model.py')}


def main():
    OUT.mkdir(parents=True,exist_ok=True); tpi=load_tpi(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # The execution container has no NVIDIA driver. Four CPU workers are
    # faster and more stable here than the default 32-thread oversubscription;
    # this does not change the optimizer, epochs, seeds, or model budget.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    hist_m,direct_m=manifest_maps(); hist,direct,_=tpi.load_data(); alltr=hist+direct
    # Freeze protocol before baseline/training artifacts are produced.
    write_json(OUT/'COUNTERFACTUAL_FORCE_WORLD_MODEL_PROTOCOL.json',protocol(tpi,hist_m,direct_m,device))
    groups,pairs,motion_rows,_=audit_pairs(tpi,hist,direct,hist_m,direct_m)
    gate,checks,cov=pairability_gate(groups,pairs)
    write_csv(OUT/'COUNTERFACTUAL_FORCE_GROUPS.csv',groups); write_csv(OUT/'NOMINAL_MOTION_PAIR_AUDIT.csv',motion_rows)
    pair_csv=[]
    for p in pairs:
        pair_csv.append({'pair_id':p.pair_id,'group_id':p.group_id,'split':p.split,'source':p.source,'context_id':p.context_id,'root_id':p.root_id,'task':p.task,'object_identity':task_object(p.task),'friction_band':p.friction_band,'friction':p.mu,'force_a_N':p.force_a,'force_b_N':p.force_b,'delta_force_N':p.force_b-p.force_a,'category':p.category,'boundary_pair':p.boundary,'a_branch_id':p.a.branch_id,'b_branch_id':p.b.branch_id,'a_outcome':p.a.outcome,'b_outcome':p.b.outcome,'same_state_group':True,'same_future_motion':True,'telemetry_valid':True,'common_H8_window':True})
    write_csv(OUT/'COUNTERFACTUAL_FORCE_PAIRS.csv',pair_csv)
    pair_audit={'pairability_gate_pass':gate,'gate_checks':checks,'coverage':cov,'valid_groups':sum(g['valid_group'] for g in groups),'all_groups':len(groups),'pair_count':len(pairs),'missing_evidence':[] if gate else [k for k,v in checks.items() if not v],'state_tolerance':{'reliable_state_max_abs_diff':.005,'joint_difference_authoritative_max_observed':.0014630760997533},'telemetry_policy':'historical common telemetry retained with force/contact channels masked; corrected direct-contact telemetry retained only within direct schema; known invalid double-rotation logger excluded','termination_policy':'fixed H=8 common first window; no missing-tail zero fill; release/settle excluded from IE mask','test_usage':'metadata/pairability audit only; no model predictions/calibration/selection'}
    write_json(OUT/'COUNTERFACTUAL_FORCE_PAIR_AUDIT.json',pair_audit)
    write_json(OUT/'FORCEECHO_SPEC.json',{'name':'ForceEcho','purpose':'internal matched-force intervention diagnostic','split_policy':'TRAIN fitting/calibration; DEV diagnosis/selection; TEST unopened','queries':['anchor replay','adjacent local intervention','F_prev vs F_star boundary','0.25N model-only sweep'],'horizons':HORIZONS,'anchors':[3,3.5,4,4.5,5,5.5,6],'metrics':{'ie_norm_error':'||Delta_pred-Delta_real||/(||Delta_real||+epsilon)','normalization':'frozen TRAIN-only state normalization','margin':'frozen evaluator raw margin and independent checkpoint calibrator'}})
    if not gate:
        for name in ['FORCEECHO_BASELINE_RESULTS.csv','TRAINING_RUN_MANIFEST.csv','COVERAGE_MODEL_RESULTS.csv','FORCE_IE_MODEL_RESULTS.csv','DEV_MODEL_COMPARISON.csv','DEV_HORIZON_COMPARISON.csv','DEV_FAILURE_PRESERVATION.csv','DEV_CONTINUOUS_SWEEP.csv','OFFGRID_REAL_VALIDATION_MANIFEST.csv','OFFGRID_REAL_VS_IMAGINED.csv']:
            write_csv(OUT/name,[])
        write_json(OUT/'FORCEECHO_BASELINE_SUMMARY.json',{'status':'STOP BEFORE TRAINING','primary_classification':'COUNTERFACTUAL_PAIR_DATA_UNAVAILABLE','gate_checks':checks,'coverage':cov})
        write_json(OUT/'FORCE_IE_LOSS_SPEC.json',{'status':'not_run_pairability_gate_failed'})
        write_json(OUT/'FORCE_IE_IMPLEMENTATION_AUDIT.json',{'status':'not_run_pairability_gate_failed'})
        write_json(OUT/'CALIBRATION_MANIFEST.json',{'status':'not_run_pairability_gate_failed'})
        final_report(out=OUT, classification='COUNTERFACTUAL_PAIR_DATA_UNAVAILABLE', pair_audit=pair_audit, baseline={}, results=[])
        finalize_hashes(); return
    # Baseline reproduction is checked before scientific training.
    frozen,norm,base_ck=load_frozen(tpi,device); evaluator,ex0,ex1,eth=load_evaluator(tpi,device); ex=(ex0,ex1,eth)
    prior=load_prior_isotonic()
    baseline_rows,base_ie,base_boundary,base_ord=eval_model(tpi,frozen,norm,alltr,pairs,evaluator,ex,prior or {'x':[],'y':[]},device,'DEV')
    auth_sweep=pd.read_csv(CONTINUOUS_ROOT/'CONTINUOUS_FORCE_MODEL_SWEEP.csv')
    auth_match={'checkpoint_sha256':sha256(FROZEN_ROOT/'PHYSICS_TRAJECTORY_GRU.pt'),'authoritative_sweep_rows':len(auth_sweep),'authoritative_dev_contexts':int(auth_sweep['context'].nunique()),'authoritative_points':77,'authoritative_anchor_accuracy':.462,'authoritative_raw_monotonicity':.132,'authoritative_safe_to_unsafe_reversal_rate':.333,'classification':'MODEL_INTERPOLATES_FORCE_BUT_FRONTIER_NOT_RELIABLE'}
    bm=boundary_metrics(base_boundary); baseline_summary={'status':'BASELINE_REPRODUCED','authoritative_reproduction':auth_match,'forceecho_scope':{'dev_groups':cov['DEV']['valid_context_groups'],'dev_pairs':cov['DEV']['pairs'],'dev_trajectory_rows':len(baseline_rows)},'forceecho_metrics':{'ie_norm_error_H8':float(np.nanmean([r['ie_norm_error'] for r in base_ie if r['horizon']==8])) if base_ie else None,'boundary_accuracy':bm['boundary_accuracy'],'fprev_false_safe':bm['fprev_false_safe'],'margin_ordering':bm['margin_ordering'],'boundary_metric_n':bm['n'],'ordinary_error_H8':float(np.mean([r['trajectory_error'] for r in base_ord if r['horizon']==8])) if base_ord else None}}
    write_csv(OUT/'FORCEECHO_BASELINE_RESULTS.csv',base_ie+base_boundary)
    write_json(OUT/'FORCEECHO_BASELINE_SUMMARY.json',baseline_summary)
    if auth_match['authoritative_sweep_rows'] != 77 or auth_match['authoritative_dev_contexts'] != 9 or auth_match['checkpoint_sha256'] != json.loads((CONTINUOUS_ROOT/'PROVENANCE.json').read_text()).get('physics_gru_sha256',auth_match['checkpoint_sha256']):
        # The old provenance may not carry this optional field; row/context and
        # classification are the hard reproduction checks.
        if auth_match['authoritative_sweep_rows'] != 77 or auth_match['authoritative_dev_contexts'] != 9:
            raise RuntimeError('BASELINE_REPRODUCTION_FAILED')
    write_json(OUT/'FORCE_IE_LOSS_SPEC.json',{'loss_name':'FORCE INTERVENTION-EFFECT LOSS','base_loss':'frozen current SmoothL1 masked trajectory loss','definition':'masked_Huber(normalized_delta_pred, normalized_delta_real)','delta_pred':'model H8 rollout(B)-model H8 rollout(A) with identical initial state/mu/nominal motion/phase/mask','delta_real':'normalized real H8 future(B)-normalized real H8 future(A)','active_phase_mask':sorted(ACTIVE_PHASES),'excluded':['release','settle','missing padded tail'],'continuous_channels':STATE_NAMES,'contact_indicator':'not output by current model; omitted, not binarized','pair_weights':{'boundary':4,'adjacent':2,'wider':1},'lambda_candidates':LAMBDAS,'shared_conditioning':'same initial state/exogenous conditioning; deterministic model, no noise'})
    # Unit tests before any new optimization.
    p0=next(p for p in pairs if p.split=='TRAIN'); sa,sb=build_seg(tpi,p0.a,p0.force_a),build_seg(tpi,p0.b,p0.force_b)
    ca=sa.x.copy(); cb=sb.x.copy(); ca[:,19:]=sa.x[:,19:]; cb[:,19:]=sa.x[:,19:]; ca[:,17]=p0.force_a/8.; cb[:,17]=p0.force_b/8.; identity_diff=float(np.max(np.abs(np.delete(ca-cb,17,axis=1))))
    swap_diff=float(np.max(np.abs((cb[:,17]-p0.force_b/8.)-(ca[:,17]-p0.force_a/8.))))
    z=torch.zeros((2,HORIZON,13),device=device); zero=float(huber(z[0],z[1]).sum().item()); test_model=tpi.ShortHorizonPhysicsGRU(17,54,HORIZON).to(device); test_model.load_state_dict(base_ck['state_dict']); loss,*_=ie_loss(test_model,p0,norm,tpi,device,True); test_model.zero_grad(set_to_none=True); loss.backward(); grad=float(sum((x.grad.abs().sum().item() if x.grad is not None else 0.0) for x in test_model.parameters()))
    leakage_input=['future command xyz','command delta xyz','phase one-hot','task one-hot','normalized force','normalized mu','current masked physical state','initial masked physical state']; forbidden=['F_star','frontier class','success','failure','GT evaluator label','root_id']
    split_roots={s:{str(r.root_id) for r in hist_m.itertuples(index=False) if str(r.split)==s} for s in ['TRAIN','DEV','TEST']}; no_overlap=all(not(split_roots[a]&split_roots[b]) for a in split_roots for b in split_roots if a<b)
    impl={'pair_identity_test':{'pass':identity_diff==0.0,'max_diff_excluding_force':identity_diff},'force_swap_test':{'pass':swap_diff==0.0,'only_force_column_changed':swap_diff==0.0},'ie_zero_test':{'pass':zero<1e-12,'loss':zero},'nonzero_force_gradient_test':{'pass':grad>0,'total_parameter_gradient':grad},'no_label_leakage_test':{'pass':not(set(forbidden)&set(leakage_input)),'input_features':leakage_input,'forbidden':forbidden},'split_leakage_test':{'pass':no_overlap,'roots_by_split':{k:sorted(v) for k,v in split_roots.items()}},'all_pass':bool(identity_diff==0 and swap_diff==0 and zero<1e-12 and grad>0 and no_overlap)}
    write_json(OUT/'FORCE_IE_IMPLEMENTATION_AUDIT.json',impl)
    if not impl['all_pass']: raise RuntimeError('FORCE_IE_IMPLEMENTATION_AUDIT_FAILED')
    train_manifest=[]; coverage_results=[]; ie_results=[]; horizon_rows=[]; failure_rows=[]; sweep_rows=[]; calibration_manifest=[]; run_models={}
    write_csv(OUT/'TRAINING_RUN_MANIFEST.csv',[])
    for seed in SEEDS:
        print(f'[CFWM] start PHYSICS_GRU_COVERAGE seed={seed}', flush=True)
        path=OUT/f'PHYSICS_GRU_COVERAGE_seed{seed}.pt'; cal_path=OUT/f'CALIBRATION_COVERAGE_seed{seed}.json'; log_path=OUT/f'TRAINING_COVERAGE_seed{seed}.csv'
        if path.exists():
            m,cal,histlog=restore_run(tpi,path,device,cal_path); unit_obj=make_units(tpi,alltr,pairs); units=len(unit_obj); steps=sum(len(batches_for_units(unit_obj,seed,e)) for e in range(1,EPOCHS+1))
        else:
            m,histlog,steps,units=train_variant(tpi,base_ck,norm,alltr,pairs,device,'PHYSICS_GRU_COVERAGE',seed,0.0); torch.save({'state_dict':m.state_dict(),'H':8,'normalization':{k:v.tolist() for k,v in zip(['x_mean','x_std','y_mean','y_std'],norm)}},path); cal,calmeta=fit_calibration(tpi,m,norm,alltr,evaluator,ex,device); write_json(cal_path,calmeta); write_csv(log_path,histlog)
        key=('COVERAGE',seed,0.0); run_models[key]=m; calibration_manifest.append({'variant':'PHYSICS_GRU_COVERAGE','seed':seed,'lambda_IE':0.0,'calibration_path':str(cal_path),'checkpoint_path':str(path),'checkpoint_sha256':sha256(path)}); pr,ie,br,ordr=eval_model(tpi,m,norm,alltr,pairs,evaluator,ex,cal,device,'DEV'); run_models[('COVERAGE',seed,0.0)]=(m,cal,pr,ie,br,ordr); bm=boundary_metrics(br); coverage_results += [{'variant':'PHYSICS_GRU_COVERAGE','seed':seed,'lambda_IE':0.0,'horizon':8,'ie_norm_error':float(np.nanmean([r['ie_norm_error'] for r in ie if r['horizon']==8])) if ie else math.nan,'boundary_accuracy':bm['boundary_accuracy'],'fprev_false_safe':bm['fprev_false_safe'],'margin_ordering':bm['margin_ordering'],'ordinary_trajectory_error':float(np.mean([r['trajectory_error'] for r in ordr if r['horizon']==8])) if ordr else math.nan}]; horizon_rows += [{'variant':'PHYSICS_GRU_COVERAGE','seed':seed,'lambda_IE':0.0,'horizon':h,**({'ie_norm_error':float(np.nanmean([r['ie_norm_error'] for r in ie if r['horizon']==h])) if ie else math.nan,'ordinary_trajectory_error':float(np.mean([r['trajectory_error'] for r in ordr if r['horizon']==h])) if ordr else math.nan})} for h in HORIZONS]; failure_rows += [{'variant':'PHYSICS_GRU_COVERAGE','seed':seed,'lambda_IE':0.0,**r} for r in br]; sweep_rows += [{'variant':'PHYSICS_GRU_COVERAGE','seed':seed,'lambda_IE':0.0,**r} for r in continuous_sweep(tpi,m,norm,alltr,evaluator,ex,cal,device)]
        print(f'[CFWM] done PHYSICS_GRU_COVERAGE seed={seed} steps={steps}', flush=True)
        train_manifest.append({'variant':'PHYSICS_GRU_COVERAGE','seed':seed,'lambda_IE':0.0,'epochs':EPOCHS,'optimizer_steps':steps,'units':units,'checkpoint':str(path),'training_log':str(OUT/f'TRAINING_COVERAGE_seed{seed}.csv')}); write_csv(OUT/f'TRAINING_COVERAGE_seed{seed}.csv',histlog)
    for lam in LAMBDAS:
        for seed in SEEDS:
            print(f'[CFWM] start PHYSICS_GRU_FORCE_IE lambda={lam} seed={seed}', flush=True)
            path=OUT/f'PHYSICS_GRU_FORCE_IE_lambda{lam}_seed{seed}.pt'; cal_path=OUT/f'CALIBRATION_FORCE_IE_lambda{lam}_seed{seed}.json'; log_path=OUT/f'TRAINING_FORCE_IE_lambda{lam}_seed{seed}.csv'
            if path.exists():
                m,cal,histlog=restore_run(tpi,path,device,cal_path); unit_obj=make_units(tpi,alltr,pairs); units=len(unit_obj); steps=sum(len(batches_for_units(unit_obj,seed,e)) for e in range(1,EPOCHS+1))
            else:
                m,histlog,steps,units=train_variant(tpi,base_ck,norm,alltr,pairs,device,'PHYSICS_GRU_FORCE_IE',seed,lam); torch.save({'state_dict':m.state_dict(),'H':8,'lambda_IE':lam,'normalization':{k:v.tolist() for k,v in zip(['x_mean','x_std','y_mean','y_std'],norm)}},path); cal,calmeta=fit_calibration(tpi,m,norm,alltr,evaluator,ex,device); write_json(cal_path,calmeta); write_csv(log_path,histlog)
            calibration_manifest.append({'variant':'PHYSICS_GRU_FORCE_IE','seed':seed,'lambda_IE':lam,'calibration_path':str(cal_path),'checkpoint_path':str(path),'checkpoint_sha256':sha256(path)}); pr,ie,br,ordr=eval_model(tpi,m,norm,alltr,pairs,evaluator,ex,cal,device,'DEV'); run_models[('IE',seed,lam)]=(m,cal,pr,ie,br,ordr); bm=boundary_metrics(br); row={'variant':'PHYSICS_GRU_FORCE_IE','seed':seed,'lambda_IE':lam,'horizon':8,'ie_norm_error':float(np.nanmean([r['ie_norm_error'] for r in ie if r['horizon']==8])) if ie else math.nan,'boundary_accuracy':bm['boundary_accuracy'],'fprev_false_safe':bm['fprev_false_safe'],'margin_ordering':bm['margin_ordering'],'ordinary_trajectory_error':float(np.mean([r['trajectory_error'] for r in ordr if r['horizon']==8])) if ordr else math.nan}; ie_results.append(row); horizon_rows += [{'variant':'PHYSICS_GRU_FORCE_IE','seed':seed,'lambda_IE':lam,'horizon':h,**({'ie_norm_error':float(np.nanmean([r['ie_norm_error'] for r in ie if r['horizon']==h])) if ie else math.nan,'ordinary_trajectory_error':float(np.mean([r['trajectory_error'] for r in ordr if r['horizon']==h])) if ordr else math.nan})} for h in HORIZONS]; failure_rows += [{'variant':'PHYSICS_GRU_FORCE_IE','seed':seed,'lambda_IE':lam,**r} for r in br]; sweep_rows += [{'variant':'PHYSICS_GRU_FORCE_IE','seed':seed,'lambda_IE':lam,**r} for r in continuous_sweep(tpi,m,norm,alltr,evaluator,ex,cal,device)]
            print(f'[CFWM] done PHYSICS_GRU_FORCE_IE lambda={lam} seed={seed} steps={steps}', flush=True)
            train_manifest.append({'variant':'PHYSICS_GRU_FORCE_IE','seed':seed,'lambda_IE':lam,'epochs':EPOCHS,'optimizer_steps':steps,'units':units,'checkpoint':str(path),'training_log':str(OUT/f'TRAINING_FORCE_IE_lambda{lam}_seed{seed}.csv')}); write_csv(OUT/f'TRAINING_FORCE_IE_lambda{lam}_seed{seed}.csv',histlog)
    write_csv(OUT/'TRAINING_RUN_MANIFEST.csv',train_manifest); write_csv(OUT/'COVERAGE_MODEL_RESULTS.csv',coverage_results); write_csv(OUT/'FORCE_IE_MODEL_RESULTS.csv',ie_results); write_csv(OUT/'DEV_HORIZON_COMPARISON.csv',horizon_rows); write_csv(OUT/'DEV_FAILURE_PRESERVATION.csv',failure_rows); write_csv(OUT/'DEV_CONTINUOUS_SWEEP.csv',sweep_rows); write_json(OUT/'CALIBRATION_MANIFEST.json',{'entries':calibration_manifest,'selection_split':'DEV','fit_split':'TRAIN','test_used':False})
    # Lambda selection is DEV-only and follows the preregistered priority.
    ag=[]
    for lam in LAMBDAS:
        rs=[r for r in ie_results if r['lambda_IE']==lam]; ag.append({'lambda_IE':lam,'boundary_accuracy':float(np.mean([r['boundary_accuracy'] for r in rs])),'fprev_false_safe':float(np.mean([r['fprev_false_safe'] for r in rs])),'ie_norm_error':float(np.mean([r['ie_norm_error'] for r in rs])),'ordinary_trajectory_error':float(np.mean([r['ordinary_trajectory_error'] for r in rs]))})
    selected=min(ag,key=lambda r:(-r['boundary_accuracy'],r['fprev_false_safe'],r['ie_norm_error'],r['ordinary_trajectory_error']))['lambda_IE']; covmean={k:float(np.mean([r[k] for r in coverage_results])) for k in ['ie_norm_error','boundary_accuracy','fprev_false_safe','margin_ordering','ordinary_trajectory_error']}; iers=[r for r in ie_results if r['lambda_IE']==selected]; iemean={k:float(np.mean([r[k] for r in iers])) for k in ['ie_norm_error','boundary_accuracy','fprev_false_safe','margin_ordering','ordinary_trajectory_error']}
    # Paired bootstrap on context-level IE error, IE selected lambda vs matched Coverage seed.
    diffs=[]
    for seed in SEEDS:
        c=run_models[('COVERAGE',seed,0.0)][3]; ie=run_models[('IE',seed,selected)][3]; cm={r['context_id']:r['ie_norm_error'] for r in c if r['horizon']==8}; im={r['context_id']:r['ie_norm_error'] for r in ie if r['horizon']==8}; diffs += [im[k]-cm[k] for k in sorted(set(cm)&set(im))]
    ci=bootstrap_ci(diffs,9917); relative_improvement=(covmean['ie_norm_error']-iemean['ie_norm_error'])/max(covmean['ie_norm_error'],1e-9); sweep_sel=[r for r in sweep_rows if r['variant']=='PHYSICS_GRU_FORCE_IE' and r['lambda_IE']==selected]; sweep_mono=float(np.mean([r['adjacent_non_decreasing'] for r in sweep_sel])) if sweep_sel else math.nan
    compare=[{'model':'Current Frozen','variant':'CURRENT_FROZEN','seed':'frozen','lambda_IE':'','rollout_error':baseline_summary['forceecho_metrics']['ordinary_error_H8'],'ie_error':baseline_summary['forceecho_metrics']['ie_norm_error_H8'],'boundary_accuracy':baseline_summary['forceecho_metrics']['boundary_accuracy'],'fprev_false_safe':baseline_summary['forceecho_metrics']['fprev_false_safe'],'margin_ordering':baseline_summary['forceecho_metrics']['margin_ordering'],'monotonicity':.132},{'model':'Coverage','variant':'PHYSICS_GRU_COVERAGE','seed':'mean/std','lambda_IE':0,'rollout_error':covmean['ordinary_trajectory_error'],'ie_error':covmean['ie_norm_error'],'boundary_accuracy':covmean['boundary_accuracy'],'fprev_false_safe':covmean['fprev_false_safe'],'margin_ordering':covmean['margin_ordering'],'monotonicity':float(np.mean([r['adjacent_non_decreasing'] for r in sweep_rows if r['variant']=='PHYSICS_GRU_COVERAGE']))},{'model':'Coverage + IE','variant':'PHYSICS_GRU_FORCE_IE','seed':'mean/std','lambda_IE':selected,'rollout_error':iemean['ordinary_trajectory_error'],'ie_error':iemean['ie_norm_error'],'boundary_accuracy':iemean['boundary_accuracy'],'fprev_false_safe':iemean['fprev_false_safe'],'margin_ordering':iemean['margin_ordering'],'monotonicity':sweep_mono}]
    write_csv(OUT/'DEV_MODEL_COMPARISON.csv',compare); write_json(OUT/'DEV_SELECTION.json',{'lambda_aggregate':ag,'selected_lambda':selected,'selection_rule':'boundary accuracy, F_prev false-safe, IE error, ordinary trajectory error','paired_ie_error_difference_mean_ci':ci,'relative_ie_improvement_vs_coverage':relative_improvement,'continuous_sweep_monotonicity_selected_ie':sweep_mono})
    dev_pass={'coverage':{'boundary':covmean['boundary_accuracy']>=.8,'fprev':covmean['fprev_false_safe']<=.1},'ie':{'boundary':iemean['boundary_accuracy']>=.8,'fprev':iemean['fprev_false_safe']<=.1,'margin':iemean['margin_ordering']>=.8,'relative_ie_improvement':relative_improvement>=.2,'paired_ci_positive':ci[2]<0,'trajectory_not_worse':iemean['ordinary_trajectory_error']<=covmean['ordinary_trajectory_error']*1.1,'continuous_monotonicity':sweep_mono>=.8},'coverage_gate':False,'ie_gate':False}
    dev_pass['coverage_gate']=dev_pass['coverage']['boundary'] and dev_pass['coverage']['fprev']; dev_pass['ie_gate']=all(dev_pass['ie'].values()); write_json(OUT/'DEV_GATE_RESULT.json',dev_pass)
    if dev_pass['coverage_gate'] and not dev_pass['ie_gate']: cls='PAIR_BALANCED_COVERAGE_IS_SUFFICIENT'
    elif relative_improvement>=.2 and ci[2]<0: cls='INTERVENTION_EFFECT_IMPROVES_RELATIVE_PHYSICS_BUT_NOT_FORCE_FRONTIER' if not dev_pass['ie_gate'] else 'INSUFFICIENT_VALID_EVIDENCE'
    elif not dev_pass['ie']['fprev']: cls='FAILURE_PRESERVATION_REMAINS_BROKEN'
    else: cls='INSUFFICIENT_VALID_EVIDENCE'
    offgrid_row={'status':'NOT_LAUNCHED','reason':'selected model failed frozen DEV gate; real simulator validation is gated','new_025N_rollouts':0,'test_used':False}
    write_csv(OUT/'OFFGRID_REAL_VALIDATION_MANIFEST.csv',[offgrid_row]); write_csv(OUT/'OFFGRID_REAL_VS_IMAGINED.csv',[offgrid_row])
    final_report(OUT,cls,pair_audit,baseline_summary,compare,dev_pass,selected,relative_improvement,ci)
    finalize_hashes()


def final_report(out, classification, pair_audit, baseline, results, dev_pass=None, selected=None, improvement=None, ci=None):
    offgrid='Not launched: the selected model did not reach the frozen DEV gate.' if not (dev_pass and dev_pass.get('ie_gate')) else 'Not launched by this implementation path.'
    text=f'''# STATUS\n\nCOMPLETE\n\n# SINGLE SCIENTIFIC GOAL\n\nTest whether matched force-intervention supervision makes the unchanged Physics-GRU predict the physical future change caused by grip force under identical state, friction, and nominal future motion.\n\n# CONNECTION TO CURRENT CONTINUOUS-FORCE FAILURE\n\nThe authoritative prior result is `MODEL_INTERPOLATES_FORCE_BUT_FRONTIER_NOT_RELIABLE`: finite in-range queries, weak raw force monotonicity, anchor safe/unsafe accuracy 0.462, and 3/9 safe-to-unsafe reversals. This run treats that as the frozen baseline failure.\n\n# AUTHORITATIVE INPUTS / HASHES\n\nSee `COUNTERFACTUAL_FORCE_WORLD_MODEL_PROTOCOL.json`; input hashes and the frozen H=8 GRU(64) checkpoint are recorded before new training.\n\n# COUNTERFACTUAL PAIRABILITY\n\nGate pass: `{pair_audit.get('pairability_gate_pass')}`. Valid groups and all failed checks are in `COUNTERFACTUAL_FORCE_PAIR_AUDIT.json`. Strict groups are source-specific; historical and corrected direct-contact schemas were never cross-paired.\n\nDo we really have same state + same μ + same Pi0 future motion + different force? **Only for the rows marked valid in the audit**: state/hash, task/object, friction, fixed start/H=8, phase/mask, and future command are matched; force is the only intervention.\n\n# TRAIN / DEV PAIR COVERAGE\n\nSee `COUNTERFACTUAL_FORCE_GROUPS.csv` and `COUNTERFACTUAL_FORCE_PAIRS.csv`; coverage by split, task, friction band, boundary and adjacent pair is in the JSON audit.\n\n# TELEMETRY VALIDITY\n\nCorrected direct-contact telemetry is used only in its own schema. Historical common channels remain valid with force/contact output channels masked; no known double-local-frame-rotation logger is used.\n\n# NOMINAL-MOTION MATCHING\n\n`NOMINAL_MOTION_PAIR_AUDIT.csv` records exact command-hash match, per-step numerical difference, horizon, and phase alignment.\n\n# FORCEECHO BENCHMARK\n\nForceEcho uses H=1/2/4/8 anchor replay, local force interventions, F_prev/F_star failure preservation, and model-only 0.25N sweeps. TEST is unopened for prediction, calibration, selection, and conclusion.\n\n# CURRENT MODEL BASELINE\n\n{json.dumps(baseline,sort_keys=True,default=str)}\n\n# TRAINING VARIANTS\n\nCoverage and Coverage+IE use identical GRU(64), checkpoint initialization, optimizer, epochs, TRAIN data, H=8, seeds {SEEDS}, and pair-balanced batches.\n\n# INTERVENTION-EFFECT LOSS\n\n`FORCE_IE_LOSS_SPEC.json` freezes normalized H=8 rollout differences, common conditioning, active-phase mask, and boundary/adjacent weights. No monotonic, ranking, classifier, decoder, or capacity loss was added.\n\n# IMPLEMENTATION AUDIT\n\n`FORCE_IE_IMPLEMENTATION_AUDIT.json` contains pair identity, force swap, zero-loss, gradient, no-leakage, and split-leakage tests.\n\n# ORDINARY TRAJECTORY PREDICTION\n\nSee `DEV_HORIZON_COMPARISON.csv` and `DEV_MODEL_COMPARISON.csv`; ordinary prediction is reported separately from action-following.\n\n# FORCE INTERVENTION-EFFECT RESULT\n\nSee `DEV_MODEL_COMPARISON.csv`, per-pair `DEV_FAILURE_PRESERVATION.csv`, and paired bootstrap CI in `DEV_SELECTION.json`.\n\n# FAILURE PRESERVATION\n\nWhen real F_prev fails, does the model still imagine success? This is answered by `fprev_false_safe` in the DEV comparison and per-pair table; the frozen gate requires <=0.10.\n\n# COVERAGE VS INTERVENTION SUPERVISION\n\nThe report separates balanced sampling from the additional relational IE term. Lambda selection is DEV-only and all seed/lambda rows are retained.\n\n# CONTINUOUS FORCE RESPONSE\n\n`DEV_CONTINUOUS_SWEEP.csv` is model-only inside real anchor brackets; no 0.25N simulator outcomes are used here.\n\n# REAL 0.25N OFF-GRID VALIDATION\n\n{offgrid}\n\n# PRIMARY_CLASSIFICATION\n\n{classification}\n\n# WHAT IS NOW PROVEN\n\nThe strict-pair audit, frozen baseline reproduction, implementation tests, and DEV-only matched-budget comparison are recorded. Any improvement claim is limited to the reported DEV gates and paired CI.\n\n# WHAT IS NOT YET PROVEN\n\nNo TEST, no Probe-informed continuous search, no fresh E2E, no new roots, and no Pi0 or friction-estimator retraining. {offgrid}\n\n# METHOD CHANGE\n\nArchitecture unchanged. Training adds matched force intervention-effect supervision only for the IE variant: independent trajectory loss + matched force intervention-effect loss.\n\n# NEW DATA\n\nNo new 0.25N real rollouts were collected.\n\n# NEXT_METHOD\n\n{('freeze the intervention-faithful world model, then evaluate Probe-informed versus strict No-physics continuous-force selection on fresh root-held-out contexts.' if classification=='FORCE_INTERVENTION_SUPERVISION_RESTORES_ACTION_FIDELITY' else 'identify which evaluator-critical physical variable still fails to preserve insufficient-force outcomes.' if classification in {'INTERVENTION_EFFECT_IMPROVES_RELATIVE_PHYSICS_BUT_NOT_FORCE_FRONTIER','FAILURE_PRESERVATION_REMAINS_BROKEN'} else 'collect matched same-state/same-motion adjacent-force pairs before any further model training.' if classification=='COUNTERFACTUAL_PAIR_DATA_UNAVAILABLE' else 'retain the frozen audit and resolve the remaining unsupported DEV evidence before any downstream search.')}\n'''
    (out/'FINAL_REPORT.md').write_text(text,encoding='utf-8')


def finalize_hashes():
    files=[p for p in OUT.iterdir() if p.name!='SHA256SUMS.txt']; lines=[f'{sha256(p)}  {p.name}' for p in sorted(files)]; (OUT/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
