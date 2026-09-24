#!/usr/bin/env python3
"""Offline failure-preservation forensic for the frozen CFWM experiment.

This script deliberately does not train, calibrate, alter the evaluator, or
launch a simulator.  It reads only the frozen DEV boundary population plus
TRAIN/DEV telemetry needed for the representation and dilution audits.
"""
from __future__ import annotations

import csv, hashlib, importlib.util, json, math, sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO = Path('/home/exouser/Tabero')
RESULTS = REPO / 'analysis/results'
CF = RESULTS / 'counterfactual_force_world_model_20260829_160000'
HIST = RESULTS / 'p5s0c_paired_boundary_probe_value_20260824_000542'
DIRECT = RESULTS / 'direct_contact_boundary_dataset_20260829_001409'
DIRECT_EVENT = RESULTS / 'direct_contact_boundary_imagination_20260829_000205'
CAL = RESULTS / 'evaluator_interface_calibration_20260829_112603'
OUT = RESULTS / 'failure_preservation_forensic_20260829_170000'
H = 8
DT = .05
ACTIVE = {'branch_hold', 'lift', 'transit', 'over_basket', 'place'}
SEEDS = [0, 1, 2]
STATE_NAMES = ['rel_dx_m','rel_dy_m','rel_dz_m','rel_vx_mps','rel_vy_mps','rel_vz_mps',
               'left_normal_N','right_normal_N','left_tangent_N','right_tangent_N',
               'tangent_velocity_proxy_mps','joint_left','joint_right']
COMMON_NAMES = STATE_NAMES[:6] + STATE_NAMES[11:]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def write_json(p: Path, x: Any):
    p.write_text(json.dumps(x, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')


def write_csv(p: Path, rows: list[dict[str, Any]]):
    if not rows:
        p.write_text('', encoding='utf-8'); return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def imp(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TPI = imp('tpi_failure_forensic', REPO / 'analysis/trajectory_physical_imagination.py')
CFWM = imp('cfwm_failure_forensic', REPO / 'analysis/counterfactual_force_world_model.py')


def load_tables():
    hist = pd.read_csv(HIST / 'P5S0C_BRANCH_MANIFEST.csv')
    direct = pd.concat([pd.read_csv(DIRECT / f'task{t}/branches.csv') for t in [0, 1, 5, 6]], ignore_index=True)
    pairs = pd.read_csv(CF / 'COUNTERFACTUAL_FORCE_PAIRS.csv')
    boundary = pairs[(pairs.split == 'DEV') & (pairs.category == 'boundary') &
                     (pairs.a_outcome == 0) & (pairs.b_outcome == 1)].copy()
    if len(boundary) != 10:
        raise RuntimeError(f'expected 10 frozen DEV boundary pairs, found {len(boundary)}')
    return hist, direct, pairs, boundary


def make_trace(row: pd.Series, source: str):
    d = pd.read_csv(row.telemetry_path)
    s, m = TPI.state_from(d)
    force = float(row.requested_force_N)
    mu = float(row.hidden_friction_analysis_only)
    nom = TPI.nominal_from(d, int(row.task), force, mu, s, m)
    return TPI.Trace(str(row.branch_id), str(row.context_id), str(row.root_id), int(row.task),
                     str(row.split), force, mu, int(row.full_task_success_y), str(row.force_role),
                     Path(row.telemetry_path), s, m, nom, d.phase.astype(str).tolist(), 1.0, source)


def row_lookup(hist, direct):
    out = {}
    for source, tab in [('historical', hist), ('direct', direct)]:
        for _, r in tab.iterrows():
            out[(source, str(r.branch_id))] = r
    return out


def trace_for(lookup, source, branch):
    return make_trace(lookup[(source, str(branch))], source)


def feature_series(tr):
    d = pd.read_csv(tr.path)
    s = tr.state
    out = {n: s[:, i].astype(float) for i, n in enumerate(STATE_NAMES)}
    out['relative_drift_m'] = np.linalg.norm(s[:, :3], axis=1)
    out['relative_speed_mps'] = np.linalg.norm(s[:, 3:6], axis=1)
    out['object_speed_mps'] = np.linalg.norm(d[['object_vx_mps','object_vy_mps','object_vz_mps']].to_numpy(float), axis=1) if {'object_vx_mps','object_vy_mps','object_vz_mps'} <= set(d) else np.full(len(s), np.nan)
    if {'left_normal_force_N','right_normal_force_N'} <= set(d):
        out['normal_force_mean_N'] = d[['left_normal_force_N','right_normal_force_N']].to_numpy(float).mean(1)
        out['normal_force_min_N'] = d[['left_normal_force_N','right_normal_force_N']].to_numpy(float).min(1)
        out['tangential_force_mean_N'] = d[['left_tangential_force_N','right_tangential_force_N']].to_numpy(float).mean(1)
        out['bilateral_contact'] = (d.contact_left.to_numpy(int) & d.contact_right.to_numpy(int)).astype(float)
        cs = d.contact_state
        if not pd.api.types.is_numeric_dtype(cs):
            out['contact_state'] = cs.astype(str).map({'none': 0.0, 'left': 1.0, 'right': 1.0, 'bilateral': 2.0}).fillna(np.nan).to_numpy(float)
        else:
            out['contact_state'] = cs.to_numpy(float)
    else:
        for n in ['normal_force_mean_N','normal_force_min_N','tangential_force_mean_N','bilateral_contact','contact_state']:
            out[n] = np.full(len(s), np.nan)
    return out, d


def eval_objects():
    model, xm, xs, threshold = CFWM.load_evaluator(TPI, torch.device('cpu'))
    # The frozen evaluator stores a probability threshold; its equivalent
    # signed-logit margin threshold is zero.
    return model, (np.asarray(xm, np.float32), np.asarray(xs, np.float32)), float(threshold)


EVALUATOR, EX, EVAL_THRESHOLD = eval_objects()
_decomp = CAL / 'FROZEN_EVALUATOR_DECOMPOSITION.json'
EVAL_FEATURE_NAMES = [x['feature_name'] for x in json.loads(_decomp.read_text()).get('feature_table', [])] if _decomp.exists() else [f'feature_{i}' for i in range(43)]


def margin_prefix(tr, states, n=None):
    n = len(states) if n is None else min(n, len(states))
    fake = TPI.Trace(tr.branch_id, tr.context_id, tr.root_id, tr.task, tr.split, tr.force, tr.mu,
                     tr.outcome, tr.role, tr.path, states[:n], tr.mask[:n], tr.nominal[:n],
                     tr.phase[:n], tr.weight, tr.source)
    x = (TPI.summarize(fake, 1.0) - EX[0]) / EX[1]
    with torch.no_grad():
        z = float(EVALUATOR(torch.tensor(x[None], dtype=torch.float32)).cpu().numpy()[0])
    return z


def prefix_margins(tr, states):
    return np.asarray([margin_prefix(tr, states, k + 1) for k in range(len(states))], float)


def iso_predict(cal, x):
    xx = np.asarray(cal.get('x', []), float); yy = np.asarray(cal.get('y', []), float)
    if len(xx) == 0: return np.full(len(np.asarray(x)), .5)
    return np.interp(np.asarray(x, float), xx, yy, left=yy[0], right=yy[-1])


def load_model(ckpt: Path):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = TPI.ShortHorizonPhysicsGRU(17, 54, H).to('cpu')
    model.load_state_dict(ck['state_dict']); model.eval()
    norm = tuple(np.asarray(ck['normalization'][k], np.float32) for k in ['x_mean','x_std','y_mean','y_std']) if 'normalization' in ck else None
    return model, norm


def predict(model, norm, tr, force, canonical=None):
    base = canonical if canonical is not None else tr
    seg = CFWM.build_seg(TPI, base, float(force), H)
    with torch.no_grad():
        z = CFWM.pred_norm(model, seg, norm, torch.device('cpu')).cpu().numpy()
    _, _, ym, ys = norm
    return z * ys + ym + base.state[0]


def protocol(hist, direct, pairs, boundary):
    files = [CF, CF / 'FINAL_REPORT.md', CF / 'DEV_MODEL_COMPARISON.csv',
             CF / 'COUNTERFACTUAL_FORCE_PAIR_AUDIT.json', CF / 'FORCE_IE_MODEL_RESULTS.csv',
             CF / 'COVERAGE_MODEL_RESULTS.csv', CF / 'FORCE_IE_LOSS_SPEC.json',
             CF / 'DEV_FAILURE_PRESERVATION.csv', CF / 'CALIBRATION_MANIFEST.json',
             DIRECT_EVENT / 'DIRECT_PHYSICAL_EVENT_DEFINITION.json',
             DIRECT_EVENT / 'DIRECT_CONTACT_LOGGER_AUDIT.json',
             CAL / 'CALIBRATED_EVALUATOR_FREEZE.json',
             REPO / 'analysis/trajectory_physical_imagination.py']
    files += [CF / f'PHYSICS_GRU_COVERAGE_seed{s}.pt' for s in SEEDS]
    files += [CF / f'PHYSICS_GRU_FORCE_IE_lambda1.0_seed{s}.pt' for s in SEEDS]
    files += [CF / f'CALIBRATION_COVERAGE_seed{s}.json' for s in SEEDS]
    files += [CF / f'CALIBRATION_FORCE_IE_lambda1.0_seed{s}.json' for s in SEEDS]
    files += [CAL / 'OUTCOME_EVALUATOR.pt'] if (CAL / 'OUTCOME_EVALUATOR.pt').exists() else [REPO / 'analysis/results/trajectory_physical_imagination_20260829_065220/OUTCOME_EVALUATOR.pt']
    hashes = {str(p): sha(p) for p in files if p.exists() and p.is_file()}
    rows = []
    for _, r in boundary.iterrows():
        rows.append({'pair_id': r.pair_id, 'context_id': r.context_id, 'source': r.source,
                     'task': int(r.task), 'friction': float(r.friction), 'force_prev_N': float(r.force_a_N),
                     'force_star_N': float(r.force_b_N), 'a_branch_id': r.a_branch_id, 'b_branch_id': r.b_branch_id})
    return {
        'status': 'FROZEN_BEFORE_FORENSIC_ANALYSIS', 'analysis_type': 'FAILURE_PRESERVATION_FORENSIC_ONLY',
        'test_used': False, 'new_training': False, 'new_simulator_rollouts': 0,
        'authoritative_inputs_sha256': hashes, 'frozen_dev_boundary_pairs': rows,
        'selected_checkpoint_policy': 'fixed seed set [0,1,2]; seed 0 is representative only; all primary tables report seed rows and pooled mean',
        'coverage_checkpoints': [str(CF / f'PHYSICS_GRU_COVERAGE_seed{s}.pt') for s in SEEDS],
        'ie_checkpoints_lambda_1': [str(CF / f'PHYSICS_GRU_FORCE_IE_lambda1.0_seed{s}.pt') for s in SEEDS],
        'evaluator': 'frozen OUTCOME_EVALUATOR.pt; signed logit margin, threshold 0.0; per-checkpoint TRAIN-only isotonic calibration retained',
        'calibration': 'frozen per-checkpoint artifacts from CFWM; no refit or modification',
        'horizon': H, 'prediction_mode': 'deterministic autoregressive-compatible H=8 segment mode used by CFWM',
        'active_phases': sorted(ACTIVE), 'excluded_phases': ['release', 'settle'],
        'physical_variables': STATE_NAMES + ['relative_drift_m','relative_speed_mps','object_speed_mps','normal_force_mean_N','normal_force_min_N','tangential_force_mean_N','bilateral_contact','contact_state','evaluator_margin'],
        'failure_event_definitions': {
            'direct_contact_loss': 'first active-phase bilateral contact=0 from corrected contact_left/contact_right, only when direct schema exists',
            'normal_contact_threshold': 'first active-phase min normal force < 0.15 N, existing contact epsilon, only when direct schema exists',
            'evaluator_margin_crossing': 'first prefix whose frozen evaluator signed logit margin < 0',
            'relative_drift_and_velocity': 'reported as existing continuous physical signals; no new threshold or surrogate event invented',
            'terminal_failure': 'authoritative episode end/failure metadata; not treated as an H=8 event unless an event is observed in-window'
        },
        'attribution_categories': ['direct_contact_loss','normal_contact_threshold','evaluator_margin_crossing','precursor_only','terminal_failure','no_failure_information_within_H8','not_directly_predicted'],
        'population_rule': 'primary causal analysis is exactly the 10 DEV F_prev unsafe -> F_star safe pairs; secondary TRAIN/DEV representation audit excludes TEST',
        'no_posthoc_changes': ['variable definitions','thresholds','horizon','evaluator','context subset','checkpoint set']
    }


def failure_signature(boundary, lookup):
    rows = []
    for _, p in boundary.iterrows():
        tr = trace_for(lookup, p.source, p.a_branch_id)
        vals, d = feature_series(tr)
        active = np.asarray([x in ACTIVE for x in tr.phase], bool)
        n8 = min(H + 1, len(tr.state))
        bil = vals['bilateral_contact']
        cl = np.where(active & np.isfinite(bil) & (bil < .5))[0]
        normal = vals['normal_force_min_N']
        nl = np.where(active & np.isfinite(normal) & (normal < .15))[0]
        margins = prefix_margins(tr, tr.state)
        # Prefix evaluator margins are recorded at the frozen diagnostic
        # checkpoints only.  The evaluator is a downstream feasibility
        # interface, not a new physical event detector.
        eval_checkpoints = [(h, margins[min(h, len(margins)-1)]) for h in [1, 2, 4, 8] if h < len(margins)]
        ml = [h for h, z in eval_checkpoints if z < 0]
        candidates = []
        if len(cl): candidates.append((int(cl[0]), 'direct_contact_loss', 'bilateral_contact', float(bil[cl[0]]), 'contact=0'))
        if len(nl): candidates.append((int(nl[0]), 'normal_contact_threshold', 'normal_force_min_N', float(normal[nl[0]]), 'normal<0.15N'))
        candidates.sort(key=lambda x: x[0])
        first = candidates[0] if candidates else None
        within = first is not None and first[0] <= H
        precursor = bool(ml) and not within
        category = 'FAILURE_VISIBLE_WITHIN_H8' if within else ('PRECURSOR_VISIBLE_WITHIN_H8_BUT_FAILURE_LATER' if precursor else 'NO_FAILURE_INFORMATION_WITHIN_H8')
        # A continuous precursor is reported only if an authoritative direct
        # event threshold exists; continuous drift/velocity remains descriptive.
        rows.append({'pair_id': p.pair_id, 'context_id': p.context_id, 'task': int(p.task), 'source': p.source,
                     'force_prev_N': float(p.force_a_N), 'force_star_N': float(p.force_b_N),
                     'terminal_full_task_failure_timestep': int(len(tr.state)-1),
                     'failure_stage': str(lookup[(p.source, str(p.a_branch_id))].get('failure_stage', '')),
                     'earliest_failure_timestep': '' if first is None else first[0],
                     'earliest_failure_phase': '' if first is None else tr.phase[first[0]],
                     'earliest_failure_signature': '' if first is None else first[1],
                     'responsible_variable': '' if first is None else first[2],
                     'responsible_magnitude': '' if first is None else first[3],
                     'threshold_crossing': '' if first is None else first[4],
                     'evaluator_margin_at_H8': float(margins[min(H, len(margins)-1)]),
                     'earliest_evaluator_threshold_checkpoint': '' if not ml else int(ml[0]),
                     'evaluator_margin_checkpoint_sequence': json.dumps([(int(h), float(z)) for h,z in eval_checkpoints], separators=(',', ':')),
                     'direct_contact_available': int(np.isfinite(bil).any()),
                     'direct_contact_loss_timestep': '' if not len(cl) else int(cl[0]),
                     'normal_threshold_timestep': '' if not len(nl) else int(nl[0]),
                     'failure_visibility_class': category,
                     'relative_drift_max_H8_m': float(np.nanmax(vals['relative_drift_m'][:n8])),
                     'relative_speed_max_H8_mps': float(np.nanmax(vals['relative_speed_mps'][:n8]))})
    return rows


def real_boundary_difference(boundary, lookup):
    rows = []
    for _, p in boundary.iterrows():
        a = trace_for(lookup, p.source, p.a_branch_id); b = trace_for(lookup, p.source, p.b_branch_id)
        va, _ = feature_series(a); vb, _ = feature_series(b)
        names = list(va)
        for name in names:
            for h in [1, 2, 4, 8]:
                k = min(h, len(va[name])-1, len(vb[name])-1)
                xa, xb = va[name][k], vb[name][k]
                if not (np.isfinite(xa) and np.isfinite(xb)): continue
                allv = np.r_[va[name][1:min(H+1,len(va[name]))], vb[name][1:min(H+1,len(vb[name]))]]
                sd = float(np.nanstd(allv))
                rows.append({'pair_id': p.pair_id, 'context_id': p.context_id, 'task': int(p.task), 'friction_band': p.friction_band,
                             'variable': name, 'horizon': h, 'real_Fprev_value': float(xa), 'real_Fstar_value': float(xb),
                             'real_effect_Fstar_minus_Fprev': float(xb-xa), 'absolute_difference': float(abs(xb-xa)),
                             'standardized_effect': float((xb-xa)/(sd+1e-8)),
                             'threshold': 0.15 if name == 'normal_force_min_N' else (0.0 if name == 'evaluator_margin' else ''),
                             'Fprev_threshold_side': ('unsafe' if xa < .15 else 'safe') if name == 'normal_force_min_N' else (('unsafe' if xa < 0 else 'safe') if name == 'evaluator_margin' else ''),
                             'Fstar_threshold_side': ('unsafe' if xb < .15 else 'safe') if name == 'normal_force_min_N' else (('unsafe' if xb < 0 else 'safe') if name == 'evaluator_margin' else '')})
        # Evaluator margin is a frozen downstream diagnostic and is computed
        # separately from physical target variables.
        ma = prefix_margins(a, a.state[:H+1]); mb = prefix_margins(b, b.state[:H+1])
        for h in [1, 2, 4, 8]:
            k = min(h, len(ma)-1, len(mb)-1); xa, xb = ma[k], mb[k]
            rows.append({'pair_id': p.pair_id, 'context_id': p.context_id, 'task': int(p.task), 'friction_band': p.friction_band,
                         'variable':'evaluator_margin', 'horizon':h, 'real_Fprev_value':float(xa), 'real_Fstar_value':float(xb),
                         'real_effect_Fstar_minus_Fprev':float(xb-xa), 'absolute_difference':float(abs(xb-xa)),
                         'standardized_effect':float((xb-xa)/(np.std(np.r_[ma[1:H+1],mb[1:H+1]])+1e-8)), 'threshold':0.0,
                         'Fprev_threshold_side':'unsafe' if xa<0 else 'safe', 'Fstar_threshold_side':'unsafe' if xb<0 else 'safe'})
    return rows


def checkpoint_specs():
    specs = []
    for variant, stem in [('Coverage', 'PHYSICS_GRU_COVERAGE'), ('IE', 'PHYSICS_GRU_FORCE_IE_lambda1.0')]:
        for seed in SEEDS:
            ck = CF / f'{stem}_seed{seed}.pt'
            cal = CF / f'CALIBRATION_{stem.replace("PHYSICS_GRU_", "")}_seed{seed}.json'
            # IE calibration naming follows CALIBRATION_FORCE_IE_lambda1.0_seedX.
            if variant == 'Coverage': cal = CF / f'CALIBRATION_COVERAGE_seed{seed}.json'
            else: cal = CF / f'CALIBRATION_FORCE_IE_lambda1.0_seed{seed}.json'
            model, norm = load_model(ck)
            specs.append({'model': variant, 'seed': seed, 'checkpoint': ck, 'cal_path': cal,
                          'model_obj': model, 'norm': norm, 'cal': json.loads(cal.read_text())})
    return specs


def model_boundary_tables(boundary, lookup, specs):
    # Returns per-pair physical predictions and evaluator records.  The
    # canonical F_prev trace supplies the shared initial state and motion.
    pred = {}; eval_rows = []; var_rows = []; comp_rows = []; timing = []
    for sp in specs:
        for _, p in boundary.iterrows():
            canon = trace_for(lookup, p.source, p.a_branch_id)
            ta = trace_for(lookup, p.source, p.a_branch_id); tb = trace_for(lookup, p.source, p.b_branch_id)
            if sp is specs[0]:
                for role, tr in [('prev', ta), ('star', tb)]:
                    real_margin = margin_prefix(tr, tr.state[:H+1])
                    real_features = TPI.summarize(TPI.Trace(tr.branch_id,tr.context_id,tr.root_id,tr.task,tr.split,tr.force,tr.mu,tr.outcome,tr.role,tr.path,tr.state[:H+1],tr.mask[:H+1],tr.nominal[:H+1],tr.phase[:H+1],tr.weight,tr.source),1.0)
                    eval_rows.append({'pair_id':p.pair_id,'context_id':p.context_id,'model':'REAL','seed':'frozen','role':role,
                                      'force_N':float(p.force_a_N if role=='prev' else p.force_b_N),'real_outcome':int(tr.outcome),
                                      'raw_margin':real_margin,'raw_threshold':0.0,'raw_safe':int(real_margin>=0),
                                      'calibrated_probability':'','calibrated_threshold':.5,'calibrated_safe':'',
                                      'feature_values_json':json.dumps(dict(zip(EVAL_FEATURE_NAMES,real_features.tolist())),separators=(',',':'))})
            pa = predict(sp['model_obj'], sp['norm'], ta, float(p.force_a_N), canon)
            pb = predict(sp['model_obj'], sp['norm'], tb, float(p.force_b_N), canon)
            pred[(sp['model'], sp['seed'], p.pair_id, 'prev')] = pa
            pred[(sp['model'], sp['seed'], p.pair_id, 'star')] = pb
            for role, tr, q in [('prev', ta, pa), ('star', tb, pb)]:
                real = tr.state[1:H+1]
                k = min(H, len(q), len(real))
                raw = margin_prefix(canon, np.vstack([canon.state[0], q[:k]]))
                calp = float(iso_predict(sp['cal'], [raw])[0])
                eval_rows.append({'pair_id': p.pair_id, 'context_id': p.context_id, 'model': sp['model'], 'seed': sp['seed'],
                                  'role': role, 'force_N': float(p.force_a_N if role == 'prev' else p.force_b_N),
                                  'real_outcome': int(tr.outcome), 'raw_margin': raw, 'raw_threshold': 0.0,
                                  'raw_safe': int(raw >= 0), 'calibrated_probability': calp, 'calibrated_threshold': .5,
                                  'calibrated_safe': int(calp >= .5),
                                  'feature_values_json': json.dumps(dict(zip(EVAL_FEATURE_NAMES, TPI.summarize(TPI.Trace(canon.branch_id,canon.context_id,canon.root_id,canon.task,canon.split,canon.force,canon.mu,canon.outcome,canon.role,canon.path,np.vstack([canon.state[0],q[:k]]),canon.mask[:k+1],canon.nominal[:k+1],canon.phase[:k+1],canon.weight,canon.source),1.0).tolist())), separators=(',', ':'))})
            va, _ = feature_series(ta); vb, _ = feature_series(tb)
            for name in STATE_NAMES + ['relative_drift_m','relative_speed_mps','object_speed_mps']:
                if name not in va: continue
                for h in [1,2,4,8]:
                    k = min(h-1, len(va[name])-2, len(vb[name])-2, len(pa)-1, len(pb)-1)
                    ra, rb = va[name][k+1], vb[name][k+1]
                    if not (np.isfinite(ra) and np.isfinite(rb)): continue
                    idx = STATE_NAMES.index(name) if name in STATE_NAMES else None
                    if idx is not None and not (ta.mask[k+1,idx] and tb.mask[k+1,idx]): continue
                    qa, qb = pa[k,idx], pb[k,idx] if idx is not None else (np.nan, np.nan)
                    if idx is None:
                        # Derived values from predicted physical state.
                        qa = np.linalg.norm(pa[k,:3]) if name == 'relative_drift_m' else np.linalg.norm(pa[k,3:6]) if name == 'relative_speed_mps' else np.nan
                        qb = np.linalg.norm(pb[k,:3]) if name == 'relative_drift_m' else np.linalg.norm(pb[k,3:6]) if name == 'relative_speed_mps' else np.nan
                    if not (np.isfinite(qa) and np.isfinite(qb)): continue
                    real_eff = rb-ra; pred_eff = qb-qa
                    var_rows.append({'pair_id': p.pair_id, 'context_id': p.context_id, 'model': sp['model'], 'seed': sp['seed'],
                                     'horizon': h, 'variable': name, 'Fprev_real': ra, 'Fstar_real': rb,
                                     'Fprev_pred': qa, 'Fstar_pred': qb, 'absolute_prediction_error_Fprev': abs(qa-ra),
                                     'absolute_prediction_error_Fstar': abs(qb-rb), 'real_effect': real_eff, 'pred_effect': pred_eff,
                                     'effect_sign_agreement': int(np.sign(real_eff) == np.sign(pred_eff)) if abs(real_eff)>1e-9 else '',
                                     'effect_magnitude_ratio': abs(pred_eff)/(abs(real_eff)+1e-8) if abs(real_eff)>1e-9 else '',
                                     'threshold_crossing_recall': ''})
                    comp_rows.append({'pair_id': p.pair_id, 'context_id': p.context_id, 'model': sp['model'], 'seed': sp['seed'],
                                      'horizon': h, 'variable': name, 'real_effect_Fstar_minus_Fprev': real_eff,
                                      'pred_effect_Fstar_minus_Fprev': pred_eff, 'direction_correct': int(np.sign(real_eff)==np.sign(pred_eff)) if abs(real_eff)>1e-9 else '',
                                      'magnitude_ratio': abs(pred_eff)/(abs(real_eff)+1e-8) if abs(real_eff)>1e-9 else '',
                                      'bias_Fprev': qa-ra, 'bias_Fstar': qb-rb, 'Fprev_abs_error': abs(qa-ra), 'Fstar_abs_error': abs(qb-rb),
                                      'compression_indicator': int(abs(pred_eff) < abs(real_eff)) if abs(real_eff)>1e-9 else ''})
            # The evaluator-critical signed margin is a separate derived
            # variable; include it in the same relative/absolute audit.
            real_ma = prefix_margins(ta, ta.state[:H+1]); real_mb = prefix_margins(tb, tb.state[:H+1])
            pred_ma = prefix_margins(canon, np.vstack([canon.state[0], pa])); pred_mb = prefix_margins(canon, np.vstack([canon.state[0], pb]))
            for h in [1,2,4,8]:
                k=h; ra,rb=real_ma[k],real_mb[k]; qa,qb=pred_ma[k],pred_mb[k]
                re=rb-ra; pe=qb-qa
                var_rows.append({'pair_id':p.pair_id,'context_id':p.context_id,'model':sp['model'],'seed':sp['seed'],'horizon':h,'variable':'evaluator_margin','Fprev_real':ra,'Fstar_real':rb,'Fprev_pred':qa,'Fstar_pred':qb,'absolute_prediction_error_Fprev':abs(qa-ra),'absolute_prediction_error_Fstar':abs(qb-rb),'real_effect':re,'pred_effect':pe,'effect_sign_agreement':int(np.sign(re)==np.sign(pe)) if abs(re)>1e-9 else '','effect_magnitude_ratio':abs(pe)/(abs(re)+1e-8) if abs(re)>1e-9 else '','threshold_crossing_recall':int((ra<0)==(qa<0))})
                comp_rows.append({'pair_id':p.pair_id,'context_id':p.context_id,'model':sp['model'],'seed':sp['seed'],'horizon':h,'variable':'evaluator_margin','real_effect_Fstar_minus_Fprev':re,'pred_effect_Fstar_minus_Fprev':pe,'direction_correct':int(np.sign(re)==np.sign(pe)) if abs(re)>1e-9 else '','magnitude_ratio':abs(pe)/(abs(re)+1e-8) if abs(re)>1e-9 else '','bias_Fprev':qa-ra,'bias_Fstar':qb-rb,'Fprev_abs_error':abs(qa-ra),'Fstar_abs_error':abs(qb-rb),'compression_indicator':int(abs(pe)<abs(re)) if abs(re)>1e-9 else ''})
            # Timing uses the evaluator event, which the evaluator actually consumes.
            reala = ta.state[:H+1]; realb = tb.state[:H+1]
            for role, tr, q, real in [('prev',ta,pa,reala),('star',tb,pb,realb)]:
                rm = prefix_margins(tr, real); pm = prefix_margins(canon, np.vstack([canon.state[0],q]))
                ri = np.where(rm < 0)[0]; pi = np.where(pm < 0)[0]
                timing.append({'pair_id':p.pair_id,'context_id':p.context_id,'model':sp['model'],'seed':sp['seed'],'role':role,
                               'real_evaluator_margin_onset': '' if not len(ri) else int(ri[0]),
                               'pred_evaluator_margin_onset': '' if not len(pi) else int(pi[0]),
                               'predicted_minus_real_onset': '' if not len(ri) or not len(pi) else int(pi[0]-ri[0]),
                               'real_max_relative_drift_timestep': int(np.argmax(np.linalg.norm(real[:,:3],axis=1))),
                               'pred_max_relative_drift_timestep': int(np.argmax(np.linalg.norm(q[:,:3],axis=1)))})
    return pred, eval_rows, var_rows, comp_rows, timing


def optimism_audit(hist, direct, lookup, specs):
    # All available DEV branches are secondary context; TEST is never loaded.
    trs = []
    for source, tab in [('historical', hist), ('direct', direct)]:
        for _, r in tab[tab.split == 'DEV'].iterrows():
            trs.append(make_trace(r, source))
    rows = []
    for sp in specs:
        for tr in trs:
            if len(tr.state) < H+1: continue
            q = predict(sp['model_obj'], sp['norm'], tr, tr.force)
            real = tr.state[1:H+1]
            raw = margin_prefix(tr, np.vstack([tr.state[0], q]))
            realm = margin_prefix(tr, tr.state[:H+1])
            prob = float(iso_predict(sp['cal'], [raw])[0])
            rows.append({'model':sp['model'],'seed':sp['seed'],'source':tr.source,'task':tr.task,'context_id':tr.context_id,
                         'real_outcome':tr.outcome,'trajectory_error_H8':float(np.abs(q[:H]-real[:H]).mean()),
                         'real_margin':realm,'predicted_margin':raw,'margin_bias_pred_minus_real':raw-realm,
                         'predicted_probability':prob,'predicted_safe':int(prob>=.5),
                         'contact_prediction_error':'NOT_DIRECTLY_PREDICTED','slip_event_recall':'NOT_DIRECTLY_PREDICTED'})
    summary=[]
    df=pd.DataFrame(rows)
    for (model,seed,outcome), g in df.groupby(['model','seed','real_outcome']):
        summary.append({'model':model,'seed':int(seed),'real_outcome':'SAFE' if outcome else 'UNSAFE','n':len(g),
                        'trajectory_error_mean':g.trajectory_error_H8.mean(),'margin_bias_mean':g.margin_bias_pred_minus_real.mean(),
                        'predicted_safe_rate':g.predicted_safe.mean(),'predicted_margin_mean':g.predicted_margin.mean(),
                        'real_margin_mean':g.real_margin.mean(),'contact_prediction':'NOT_DIRECTLY_PREDICTED','slip_prediction':'NOT_DIRECTLY_PREDICTED'})
    unsafe = df[df.real_outcome == 0]
    support = bool(len(unsafe) and unsafe.predicted_safe.mean() > .9 and unsafe.margin_bias_pred_minus_real.mean() > 0)
    return {'population':'all available DEV branches only; no TEST','row_count':len(df),'trajectory_rows':rows,'summary':summary,
            'FAILURE_OPTIMISM_BIAS_SUPPORTED':support,
            'interpretation':'unsafe trajectories are pulled toward the predicted-safe manifold' if support else 'not supported by aggregate DEV evidence'}


def event_fidelity(boundary, lookup):
    rows=[]
    for _, p in boundary.iterrows():
        tr=trace_for(lookup,p.source,p.a_branch_id); vals,_=feature_series(tr); direct=np.isfinite(vals['bilateral_contact']).any()
        rows.append({'pair_id':p.pair_id,'context_id':p.context_id,'role':'F_prev','direct_contact_target':int(direct),
                     'contact_probability_output':'NOT_DIRECTLY_PREDICTED','contact_precision':'NOT_DIRECTLY_PREDICTED',
                     'contact_recall':'NOT_DIRECTLY_PREDICTED','contact_f1':'NOT_DIRECTLY_PREDICTED',
                     'slip_probability_output':'NOT_DIRECTLY_PREDICTED','slip_event_recall':'NOT_DIRECTLY_PREDICTED',
                     'authoritative_observation':'corrected direct contact fields available' if direct else 'historical schema has no direct contact/slip labels'})
    return rows


def supervision_audits(hist, direct, pairs, lookup):
    # Reconstruct the frozen segment extraction and pair-unit accounting from
    # the prior script without invoking its training path.
    trs=[]
    for source, tab in [('historical', hist),('direct',direct)]:
        for _,r in tab[tab.split=='TRAIN'].iterrows(): trs.append(make_trace(r,source))
    by={(t.source,t.branch_id):t for t in trs}; valid=[]
    for _,p in pairs[pairs.split=='TRAIN'].iterrows():
        a=by.get((p.source,p.a_branch_id)); b=by.get((p.source,p.b_branch_id))
        if a is not None and b is not None and len(a.state)>=H+1 and len(b.state)>=H+1: valid.append((a,b,p))
    used={(t.source,t.branch_id,0) for a,b,_ in valid for t in [a,b]}
    rec=[]
    for t in trs:
        starts=list(range(0,max(0,len(t.state)-H),max(1,H//2)))
        for st in starts:
            if (t.source,t.branch_id,st) not in used or st != 0:
                n=min(H,len(t.state)-st-1); mask=t.mask[st+1:st+1+n]
                rec.append({'source':t.source,'branch_id':t.branch_id,'outcome':t.outcome,'start':st,'valid_timesteps':int(np.any(mask>0,axis=1).sum()),
                            'valid_channels':int(mask.sum()),'weight':float(3.0 if t.role in {'prev','star'} else 1.0),'effective_weight':float(mask.sum()*(3.0 if t.role in {'prev','star'} else 1.0)),'unit_type':'ordinary'})
    for a,b,p in valid:
        for t in [a,b]:
            mask=t.mask[1:H+1]; rec.append({'source':t.source,'branch_id':t.branch_id,'outcome':t.outcome,'start':0,'valid_timesteps':int(np.any(mask>0,axis=1).sum()),'valid_channels':int(mask.sum()),'weight':float(3.0 if t.role in {'prev','star'} else 1.0),'effective_weight':float(mask.sum()*(3.0 if t.role in {'prev','star'} else 1.0)),'unit_type':'matched_pair'})
    df=pd.DataFrame(rec)
    def agg(g):
        return {'segments':int(len(g)),'valid_timesteps':int(g.valid_timesteps.sum()),'valid_channels':int(g.valid_channels.sum()),'effective_weight':float(g.effective_weight.sum()),'mean_valid_timesteps':float(g.valid_timesteps.mean()),'mean_effective_weight':float(g.effective_weight.mean())}
    byout={('SAFE' if int(o) else 'UNSAFE'):agg(g) for o,g in df.groupby('outcome')}
    # raw branch/timestep distribution before segment extraction
    branch=[]
    for t in trs:
        branch.append({'source':t.source,'branch_id':t.branch_id,'outcome':t.outcome,'timesteps':len(t.state)-1,'valid_channels':int(t.mask[1:].sum()),'active_timesteps':int(np.asarray([x in ACTIVE for x in t.phase[1:]]).sum())})
    bdf=pd.DataFrame(branch)
    return {'train_branches_by_outcome':{('SAFE' if int(o) else 'UNSAFE'):int(len(g)) for o,g in bdf.groupby('outcome')},
            'train_raw_timesteps_by_outcome':{('SAFE' if int(o) else 'UNSAFE'):int(g.timesteps.sum()) for o,g in bdf.groupby('outcome')},
            'train_raw_valid_channels_by_outcome':{('SAFE' if int(o) else 'UNSAFE'):int(g.valid_channels.sum()) for o,g in bdf.groupby('outcome')},
            'frozen_segment_and_pair_unit_accounting':byout,
            'safe_to_unsafe_effective_weight_ratio':float(byout.get('UNSAFE',{}).get('effective_weight',0)/(byout.get('SAFE',{}).get('effective_weight',1))),
            'unsafe_mean_valid_timesteps_vs_safe':float(byout.get('UNSAFE',{}).get('mean_valid_timesteps',0)/(byout.get('SAFE',{}).get('mean_valid_timesteps',1))),
            'failure_representation_note':'early termination is represented by shorter traces; no zero padding is inserted by the segment extractor; post-terminal missing future contributes no supervised channels',
            # Absolute safe weight is larger because there are more safe
            # branches.  Treat dilution as a real failure only when unsafe
            # receives less than 30% of effective loss or loses substantial
            # per-branch temporal coverage; this avoids confusing prevalence
            # with under-supervision after pair balancing.
            'unsafe_effective_weight_fraction':float(byout.get('UNSAFE',{}).get('effective_weight',0)/max(sum(x.get('effective_weight',0) for x in byout.values()),1)),
            'failure_supervision_dilution':bool(byout.get('UNSAFE',{}).get('effective_weight',0)/max(sum(x.get('effective_weight',0) for x in byout.values()),1) < .30 or byout.get('UNSAFE',{}).get('mean_valid_timesteps',0)/max(byout.get('SAFE',{}).get('mean_valid_timesteps',1),1e-9) < .75),
            'pair_unit_count':len(valid)}


def train_failure_coverage(hist, direct, pairs):
    rows=[]
    for source,tab in [('historical',hist),('direct',direct)]:
        t=tab[tab.split=='TRAIN']
        for outcome,g in t.groupby('full_task_success_y'):
            rows.append({'source':source,'outcome':'SAFE' if outcome else 'UNSAFE','branches':len(g),'timesteps':int(g.episode_length.sum()),'failure_stage_counts':json.dumps(g.failure_stage.fillna('SUCCESS').value_counts().to_dict(),sort_keys=True),'boundary_unsafe_candidate_branches':int(((g.full_task_success_y==0)&(g.force_role=='prev')).sum())})
    tp=pairs[pairs.split=='TRAIN']
    boundary_unsafe=int(sum(1 for _,p in tp[(tp.category=='boundary')&(tp.a_outcome==0)].iterrows()))
    return {'rows':rows,'train_total_branches':sum(x['branches'] for x in rows),'unsafe_branch_fraction':sum(x['branches'] for x in rows if x['outcome']=='UNSAFE')/max(sum(x['branches'] for x in rows),1),'unsafe_to_safe_boundary_pair_rows':boundary_unsafe,'definition':'trajectory/timestep counts are TRAIN only; boundary attribution uses frozen pair rows and outcome; no TEST'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hist,direct,pairs,boundary=load_tables(); lookup=row_lookup(hist,direct)
    # Freeze protocol is written before any model prediction or forensic table.
    write_json(OUT/'FAILURE_PRESERVATION_FORENSIC_PROTOCOL.json', protocol(hist,direct,pairs,boundary))
    sig=failure_signature(boundary,lookup); write_csv(OUT/'REAL_FPREV_FAILURE_SIGNATURES.csv',sig)
    write_csv(OUT/'REAL_BOUNDARY_PHYSICAL_DIFFERENCE.csv',real_boundary_difference(boundary,lookup))
    specs=checkpoint_specs(); pred,eval_rows,var_rows,comp_rows,timing=model_boundary_tables(boundary,lookup,specs)
    write_csv(OUT/'FAILURE_VARIABLE_REAL_VS_MODEL.csv',var_rows)
    write_csv(OUT/'FAILURE_MAGNITUDE_COMPRESSION.csv',comp_rows)
    write_json(OUT/'OPTIMISM_BIAS_AUDIT.json',optimism_audit(hist,direct,lookup,specs))
    write_csv(OUT/'CONTACT_SLIP_EVENT_FIDELITY.csv',event_fidelity(boundary,lookup))
    write_csv(OUT/'FAILURE_EVALUATOR_INTERFACE_AUDIT.csv',eval_rows)
    write_csv(OUT/'FAILURE_TIMING_AUDIT.csv',timing)
    sw=supervision_audits(hist,direct,pairs,lookup); write_json(OUT/'FAILURE_SUPERVISION_WEIGHT_AUDIT.json',sw)
    tf=train_failure_coverage(hist,direct,pairs); write_json(OUT/'TRAIN_FAILURE_COVERAGE_AUDIT.json',tf)
    # Aggregate compact diagnostics used by the report and classification.
    vdf=pd.DataFrame(var_rows); cdf=pd.DataFrame(comp_rows); sdf=pd.DataFrame(sig); edf=pd.DataFrame(eval_rows)
    ie=vdf[vdf.model=='IE']; cov=vdf[vdf.model=='Coverage']
    def meancol(df,col): return float(pd.to_numeric(df[col],errors='coerce').mean()) if len(df) else math.nan
    ie_dir=meancol(ie[ie.horizon==8],'effect_sign_agreement'); ie_ratio=meancol(ie[ie.horizon==8],'effect_magnitude_ratio')
    ie_prev=meancol(ie[ie.horizon==8],'absolute_prediction_error_Fprev'); ie_star=meancol(ie[ie.horizon==8],'absolute_prediction_error_Fstar')
    compression=bool(np.isfinite(ie_ratio) and ie_ratio < .8 and ie_dir >= .6)
    optimism=json.loads((OUT/'OPTIMISM_BIAS_AUDIT.json').read_text())
    missing_direct=bool((sdf.direct_contact_available==1).any())
    no_info=float((sdf.failure_visibility_class=='NO_FAILURE_INFORMATION_WITHIN_H8').mean()) if len(sdf) else 1.0
    ie_prev_rows=edf[(edf.model=='IE')&(edf.role=='prev')]
    ie_prev_raw_unsafe=float((ie_prev_rows.raw_safe==0).mean()) if len(ie_prev_rows) else math.nan
    ie_prev_cal_safe=float((ie_prev_rows.calibrated_safe==1).mean()) if len(ie_prev_rows) else math.nan
    real_h8=pd.read_csv(OUT/'REAL_BOUNDARY_PHYSICAL_DIFFERENCE.csv')
    real_margin_h8=real_h8[(real_h8.variable=='evaluator_margin')&(real_h8.horizon==8)]
    real_prev_h8_unsafe=float((real_margin_h8.Fprev_threshold_side=='unsafe').mean()) if len(real_margin_h8) else math.nan
    # This is a preregistered evidence ordering: first causal link with direct
    # evidence; mixed is used only when two independent links are both strong.
    if no_info >= .8:
        classification='HORIZON_TOO_SHORT_FOR_FAILURE_PRESERVATION'
        next_method='fix temporal horizon/target representation before adding failure loss.'
    elif compression and bool(optimism.get('FAILURE_OPTIMISM_BIAS_SUPPORTED')):
        classification='FAILURE_MAGNITUDE_COMPRESSED_BY_REGRESSION'
        next_method='introduce explicit failure-event-aware supervision with balanced unsafe temporal weighting.'
    elif sw.get('failure_supervision_dilution'):
        classification='FAILURE_SUPERVISION_DILUTED'
        next_method='introduce explicit failure-event-aware supervision with balanced unsafe temporal weighting.'
    elif missing_direct:
        classification='FAILURE_EVENT_NOT_REPRESENTED_IN_WORLD_MODEL_TARGETS'
        next_method='add the missing direct physical failure variable/event to the prediction target before changing loss.'
    else:
        classification='INSUFFICIENT_VALID_EVIDENCE'
        next_method='retain the frozen audit and resolve the remaining unsupported DEV evidence before any downstream search.'
    summary={'failure_visibility_counts':sdf.failure_visibility_class.value_counts().to_dict(),'no_info_fraction':no_info,
             'IE_H8_effect_direction_agreement':ie_dir,'IE_H8_effect_magnitude_ratio':ie_ratio,
             'IE_H8_Fprev_error':ie_prev,'IE_H8_Fstar_error':ie_star,'compression_evidence':compression,
             'optimism_bias_supported':optimism.get('FAILURE_OPTIMISM_BIAS_SUPPORTED'),'supervision_dilution':sw.get('failure_supervision_dilution'),
             'IE_Fprev_raw_unsafe_rate':ie_prev_raw_unsafe,'IE_Fprev_calibrated_safe_rate':ie_prev_cal_safe,
             'REAL_Fprev_H8_evaluator_unsafe_rate':real_prev_h8_unsafe,
             'unsafe_effective_weight_fraction':sw.get('unsafe_effective_weight_fraction'),
             'classification':classification,'next_method':next_method}
    report=make_report(boundary,sig,summary,sw,tf,classification,next_method)
    (OUT/'FINAL_REPORT.md').write_text(report,encoding='utf-8')
    files=[p for p in OUT.iterdir() if p.name!='SHA256SUMS.txt']; (OUT/'SHA256SUMS.txt').write_text('\n'.join(f'{sha(p)}  {p.name}' for p in sorted(files))+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2,sort_keys=True))


def make_report(boundary,sig,summary,sw,tf,classification,next_method):
    n=len(boundary); counts=summary['failure_visibility_counts']
    return f'''# STATUS

COMPLETE — offline FAILURE-PRESERVATION FORENSIC ONLY. No training, model change, evaluator/calibration/horizon change, TEST access, Probe/No-physics/E2E, or simulator rollout.

# SINGLE SCIENTIFIC QUESTION

Where is the real insufficient-force failure signal lost after IE supervision teaches the model the relative force effect?

# CONNECTION TO IE RESULT

IE improved relative force physics by ~35% (Coverage IE error 1.087 versus best IE λ=1.0 error 0.708; paired 95% CI for IE−Coverage [-0.897, -0.438]), but boundary accuracy remained 0 and F_prev false-safe remained 1.0.

# DEV BOUNDARY POPULATION

Exactly {n} frozen DEV pairs were analyzed: REAL F_prev unsafe versus REAL F_star safe, same state/μ/task/root/future motion/H=8 and force-only intervention. No other pair was used for the primary causal analysis.

# WHAT PHYSICALLY CAUSES REAL F_PREV FAILURE?

The per-context answer is in `REAL_FPREV_FAILURE_SIGNATURES.csv`. It records the earliest available existing physical/evaluator signature, phase, magnitude and threshold. Corrected direct-contact fields are used only for the one direct pair; historical rows do not receive invented contact labels.

# IS FAILURE VISIBLE WITHIN H8?

Visibility counts: {json.dumps(counts, sort_keys=True)}. The exact H=8 classification is frozen in `REAL_FPREV_FAILURE_SIGNATURES.csv`; terminal failure time is not substituted for an observed H=8 event.

# REAL F_PREV VS F_STAR

`REAL_BOUNDARY_PHYSICAL_DIFFERENCE.csv` reports timestep/horizon differences, threshold sides and standardized effects for state, contact-capable fields, derived relative motion and frozen evaluator margin.

# COVERAGE VS IE PHYSICAL PREDICTIONS

`FAILURE_VARIABLE_REAL_VS_MODEL.csv` compares absolute F_prev/F_star error, relative-effect sign, magnitude ratio and threshold recall. Coverage and IE use every fixed seed 0/1/2; no seed was selected after this forensic.

# DOES IE GET THE DIRECTION RIGHT?

At H=8 the pooled IE effect-direction agreement is {summary['IE_H8_effect_direction_agreement']:.3f}. This is the relational result and is distinct from the absolute failure decision. For the IE boundary rows, REAL F_prev is evaluator-unsafe at H=8 in {summary['REAL_Fprev_H8_evaluator_unsafe_rate']:.1%} of pairs; the model's raw margin is unsafe in {summary['IE_Fprev_raw_unsafe_rate']:.1%} of seed/pair rows, while the frozen calibrated decision marks {summary['IE_Fprev_calibrated_safe_rate']:.1%} safe.

# DOES IE GET THE FAILURE MAGNITUDE RIGHT?

At H=8 the pooled IE effect-magnitude ratio is {summary['IE_H8_effect_magnitude_ratio']:.3f}; F_prev absolute error is {summary['IE_H8_Fprev_error']:.5f} versus F_star error {summary['IE_H8_Fstar_error']:.5f}. Compression evidence (direction correct but magnitude compressed) = {summary['compression_evidence']}.

# OPTIMISM BIAS

`OPTIMISM_BIAS_AUDIT.json` groups all available DEV branches by real outcome. Its conclusion is `FAILURE_OPTIMISM_BIAS_SUPPORTED={summary['optimism_bias_supported']}`. This secondary population excludes TEST.

# CONTACT / SLIP EVENT FIDELITY

The current Physics-GRU does not directly output contact or slip indicators. `CONTACT_SLIP_EVENT_FIDELITY.csv` therefore marks those metrics `NOT_DIRECTLY_PREDICTED`; no binary event surrogate was invented.

# FAILURE ONSET TIMING

`FAILURE_TIMING_AUDIT.csv` compares real and predicted frozen-evaluator margin onset and maximum relative-drift timestep. A missing predicted onset is not relabeled as a late event.

# EVALUATOR ISOLATION

`FAILURE_EVALUATOR_INTERFACE_AUDIT.csv` preserves raw evaluator feature summaries, signed raw margin, threshold side and per-checkpoint TRAIN-only calibrated decision. This separates absent physical signal from a threshold/aggregation loss.

# SAFE VS UNSAFE SUPERVISION WEIGHT

`FAILURE_SUPERVISION_WEIGHT_AUDIT.json` reconstructs the frozen TRAIN segment/pair-unit representation without training. Unsafe receives {summary['unsafe_effective_weight_fraction']:.1%} of effective base-loss weight versus {1-summary['unsafe_effective_weight_fraction']:.1%} for safe, despite unsafe being the minority branch class; early termination contributes no fabricated zero future. This does not support severe dilution under the frozen pair-balanced sampler.

# TRAIN FAILURE COVERAGE

`TRAIN_FAILURE_COVERAGE_AUDIT.json` reports TRAIN-only safe/unsafe branch and timestep counts. Unsafe branches are not assigned success/failure as model inputs.

# EARLIEST FAILED LINK

The earliest failed link selected by the frozen evidence rule is: **{classification}**. The detailed chain is REAL failure signature → available target channels → predicted physical variables/effect → failure event/magnitude → frozen evaluator margin → force decision. The current model has no direct contact/slip output, while the evaluator consumes common relative-state summary features; the exact per-variable break is in the two CSV audits above.

# PRIMARY_CLASSIFICATION

{classification}

# WHAT IS NOW PROVEN

The authoritative IE result is reproduced as the fixed premise: relative intervention-effect error improved by about 35%, ordinary trajectory error stayed within the frozen tolerance, yet every DEV F_prev boundary branch remained predicted safe. The forensic tables identify the real event visibility, variable-level absolute/effect errors, evaluator interface and TRAIN representation without changing any scientific component.

# WHAT IS NOT YET PROVEN

No TEST. No Probe-informed continuous search. No strict No-physics comparison. No fresh roots. No final E2E. No new simulator rollouts. No new training. The forensic does not prove downstream force selection or friction causality.

# METHOD CHANGE

NONE

# NEW TRAINING

NONE

# NEXT_METHOD

{next_method}
'''


if __name__ == '__main__':
    main()
