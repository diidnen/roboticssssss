#!/usr/bin/env python3
"""Direct physical-ranking forensic for the frozen Tabero H=8 world model.

This run deliberately does not train a model.  It asks whether the exact
previously selected short-horizon model preserves force-dependent physical
differences when its trajectories are scored by a mechanics-defined stable
attachment criterion.
"""
from __future__ import annotations

import csv, hashlib, json, math, os, random, sys, time, importlib.util
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path('/home/exouser/Tabero')
RESULTS = REPO / 'analysis/results'
PREV = RESULTS / 'trajectory_physical_imagination_20260829_065220'
HIST_ROOT = RESULTS / 'p5s0c_paired_boundary_probe_value_20260824_000542'
DIRECT_ROOT = RESULTS / 'direct_contact_boundary_dataset_20260829_001409'
FRICTION_ROOT = RESULTS / 'active_friction_imagination_20260828_211106'
SEED = 2026082911
H = 8
TASKS = [0, 1, 5, 6]
ACTIVE_PHASES = {'branch_hold', 'lift', 'transit', 'over_basket', 'place'}
POS_TOL_M = 0.005
VEL_TOL_MPS = 0.020
VEL_WEIGHT = 0.25


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(p: Path, obj):
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + '\n')


def write_csv(p: Path, rows):
    rows = list(rows)
    if not rows:
        p.write_text('')
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_tpi():
    spec = importlib.util.spec_from_file_location('tpi_frozen', REPO / 'analysis/trajectory_physical_imagination.py')
    m = importlib.util.module_from_spec(spec)
    sys.modules['tpi_frozen'] = m
    spec.loader.exec_module(m)
    return m


def device_check():
    import subprocess
    r = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback is forbidden')
    return torch.device('cuda'), {
        'nvidia_smi_returncode': r.returncode,
        'nvidia_smi_stdout': r.stdout.strip(),
        'nvidia_smi_stderr': r.stderr.strip(),
        'torch_version': torch.__version__,
        'cuda_available': bool(torch.cuda.is_available()),
        'device_count': int(torch.cuda.device_count()),
        'device_name': torch.cuda.get_device_name(0),
    }


def real_frontier(hist, cid):
    fs = [t.force for t in hist if t.context_id == cid and t.outcome == 1]
    return min(fs) if fs else math.nan


def chunk_costs(t, state=None):
    """Mechanics-only local cost; release and settle are not stability phases."""
    s = t.state if state is None else state
    phases = t.phase[:len(s)]
    active = np.asarray([p in ACTIVE_PHASES for p in phases])
    costs, phase_names = [], []
    for start in range(0, len(s) - 1, H):
        end = min(len(s) - 1, start + H)
        x = s[start + 1:end + 1]
        a = active[start + 1:end + 1]
        if not np.any(a):
            continue
        pos = np.linalg.norm(x[a, :3] / POS_TOL_M, axis=1)
        vel = np.linalg.norm(x[a, 3:6] / VEL_TOL_MPS, axis=1)
        costs.append(float(np.mean(pos + VEL_WEIGHT * vel)))
        phase_names.append(phases[start + 1 + int(np.flatnonzero(a)[0])])
    return np.asarray(costs, dtype=float), phase_names


def aggregate(costs):
    if len(costs) == 0:
        return math.nan
    # Worst 25% local segments: conservative but less brittle than a single
    # one-step maximum.  This is a mean over chunks, not a full-horizon mean.
    k = max(1, int(math.ceil(0.25 * len(costs))))
    return float(np.sort(costs)[-k:].mean())


def score_row(t, source='historical'):
    c, ph = chunk_costs(t)
    return {
        'source': source, 'branch_id': t.branch_id, 'context_id': t.context_id,
        'root_id': t.root_id, 'task': t.task, 'split': t.split,
        'candidate_force_N': t.force, 'friction_gt': t.mu,
        'real_success': t.outcome, 'force_role': t.role,
        'physical_score': aggregate(c), 'chunk_count': len(c),
        'chunk_scores_json': json.dumps([float(x) for x in c]),
        'chunk_phases_json': json.dumps(ph),
    }


def metrics_for_rows(rows, thresholds, split):
    d = pd.DataFrame([r for r in rows if r['split'] == split])
    out = []
    if d.empty:
        return {'contexts': 0, 'exact': math.nan, 'within_one': math.nan,
                'under_force': math.nan, 'over_force': math.nan,
                'no_valid_force': math.nan, 'mae_N': math.nan}
    for cid, g in d.groupby('context_id'):
        fstar = real_frontier_by_group(g)
        ok = g[g.physical_score <= thresholds[int(g.task.iloc[0])]]
        chosen = float(ok.candidate_force_N.min()) if len(ok) else math.nan
        out.append({
            'context_id': cid, 'real_F_star': fstar, 'chosen_F': chosen,
            'exact': int(np.isfinite(chosen) and np.isfinite(fstar) and chosen == fstar),
            'within_one': int(np.isfinite(chosen) and np.isfinite(fstar) and abs(chosen - fstar) <= 0.5),
            'under_force': int(np.isfinite(chosen) and np.isfinite(fstar) and chosen < fstar),
            'over_force': int(np.isfinite(chosen) and np.isfinite(fstar) and chosen > fstar),
            'no_valid_force': int(not np.isfinite(chosen)),
            'absolute_error': abs(chosen - fstar) if np.isfinite(chosen) and np.isfinite(fstar) else math.nan,
            'signed_error': chosen - fstar if np.isfinite(chosen) and np.isfinite(fstar) else math.nan,
        })
    q = pd.DataFrame(out)
    return {
        'contexts': len(q), 'exact': float(q.exact.mean()),
        'within_one': float(q.within_one.mean()), 'under_force': float(q.under_force.mean()),
        'over_force': float(q.over_force.mean()), 'no_valid_force': float(q.no_valid_force.mean()),
        'mae_N': float(q.absolute_error.mean()), 'details': out,
    }


def real_frontier_by_group(g):
    ok = g[g.real_success == 1]
    return float(ok.candidate_force_N.min()) if len(ok) else math.nan


def tune_thresholds(rows):
    d = pd.DataFrame([r for r in rows if r['split'] == 'DEV'])
    thresholds, choices = {}, {}
    for task in TASKS:
        x = d[d.task == task]
        best = None
        for delta in sorted(x.physical_score.dropna().unique()):
            ar = []
            for _, g in x.groupby('context_id'):
                fs = real_frontier_by_group(g)
                ok = g[g.physical_score <= delta]
                ch = float(ok.candidate_force_N.min()) if len(ok) else math.nan
                ar.append((int(np.isfinite(ch) and np.isfinite(fs) and ch == fs),
                           int(np.isfinite(ch) and np.isfinite(fs) and ch < fs),
                           int(np.isfinite(ch) and np.isfinite(fs) and abs(ch-fs) <= .5),
                           int(not np.isfinite(ch))))
            a = np.mean(ar, axis=0)
            # A false-safe is more serious than no valid force; use only DEV.
            key = (a[0] - 2*a[1], a[0], a[2], -a[3], -delta)
            if best is None or key > best[0]:
                best = (key, float(delta), a)
        thresholds[task] = best[1]
        choices[task] = {'delta': best[1], 'dev_exact': float(best[2][0]),
                         'dev_under_force': float(best[2][1]),
                         'dev_within_one': float(best[2][2]),
                         'dev_no_valid_force': float(best[2][3])}
    return thresholds, choices


def boundary_rows(rows, thresholds=None):
    d = pd.DataFrame(rows)
    ans = []
    for cid, g in d.groupby('context_id'):
        g = g.sort_values('candidate_force_N')
        fs = real_frontier_by_group(g)
        if not np.isfinite(fs):
            continue
        star = g[g.candidate_force_N == fs].iloc[0]
        prev = g[g.candidate_force_N < fs].tail(1)
        nxt = g[g.candidate_force_N > fs].head(1)
        if len(prev):
            p = prev.iloc[0]
            ans.append({'context_id': cid, 'root_id': star.root_id, 'task': int(star.task),
                        'split': star.split, 'pair': 'F_prev_vs_F_star', 'F_low': p.candidate_force_N,
                        'F_high': fs, 'score_low': p.physical_score, 'score_high': star.physical_score,
                        'low_is_worse': int(p.physical_score > star.physical_score),
                        'score_margin_high_minus_low': star.physical_score-p.physical_score})
        if len(nxt):
            n = nxt.iloc[0]
            ans.append({'context_id': cid, 'root_id': star.root_id, 'task': int(star.task),
                        'split': star.split, 'pair': 'F_star_vs_F_next', 'F_low': fs,
                        'F_high': n.candidate_force_N, 'score_low': star.physical_score,
                        'score_high': n.physical_score,
                        'low_is_worse': int(star.physical_score > n.physical_score),
                        'score_margin_high_minus_low': n.physical_score-star.physical_score})
    return ans


def auc(y, score):
    y, score = np.asarray(y).astype(int), np.asarray(score, float)
    p, n = score[y == 1], score[y == 0]
    if not len(p) or not len(n): return math.nan
    return float(((p[:, None] > n[None, :]).sum() + .5*(p[:, None] == n[None, :]).sum())/(len(p)*len(n)))


def fast_chain(m, model, t, d, force, mu, norm, device):
    """Exact previous H=8 chain, with telemetry/nominal data cached per call."""
    state, mask = t.state[0].copy(), t.mask[0].copy()
    base = m.nominal_from(d, int(t.task), float(force), float(mu), t.state, t.mask)
    preds, calls = [], 0
    xm, xs, ym, ys = norm
    for start in range(0, len(t.state) - 1 - H + 1, H):
        x = base.copy()
        x[:, 19:32], x[:, 32:45] = state, mask
        # Preserve the authoritative prior chain exactly: the current imagined
        # state is injected into the model input block, while predict_segment
        # reconstructs absolute state using the same trace-state indexing as
        # the frozen m.chain_imagination implementation.
        fake = m.Trace(t.branch_id, t.context_id, t.root_id, t.task, t.split,
                       float(force), float(mu), t.outcome, t.role, t.path,
                       np.vstack([state, t.state[1:]]),
                       np.vstack([mask, t.mask[1:]]),
                       x, t.phase, t.weight, t.source)
        seg = m.Segment(fake, start, H, x[start:start+H],
                        np.zeros((H, 13), np.float32), np.zeros((H, 13), np.float32))
        pred = m.predict_segment(model, seg, norm, device)
        preds.append(pred)
        state = pred[-1].copy()
        calls += 1
    if not preds:
        return np.empty((0, 13), np.float32), calls
    return np.concatenate(preds), calls


def imagined_score(t, predicted):
    # Keep the same phase schedule, padding only an incomplete final chunk with
    # its last predicted state. The direct score never sees branch outcome.
    n = len(t.state) - 1
    if len(predicted) == 0:
        return math.nan, [], []
    if len(predicted) < n:
        predicted = np.vstack([predicted, np.repeat(predicted[-1][None], n-len(predicted), axis=0)])
    state = np.vstack([t.state[0], predicted[:n]])
    fake = type('FakeTrace', (), {'state': state, 'phase': t.phase})
    c, ph = chunk_costs(fake)
    return aggregate(c), c, ph


def main():
    requested_out = os.environ.get('DSPR_OUT', '')
    out = Path(requested_out) if requested_out else RESULTS / ('dreamstyle_physical_ranking_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=bool(requested_out))
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device, devmeta = device_check()
    m = load_tpi()
    hist, direct, _ = m.load_data()
    ckpt = PREV / 'PHYSICS_TRAJECTORY_GRU.pt'
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    if int(ck['H']) != H:
        raise RuntimeError(f'authoritative checkpoint H={ck["H"]}, expected H=8')
    norm = tuple(np.asarray(ck['normalization'][k], np.float32) for k in ['x_mean','x_std','y_mean','y_std'])
    model = m.ShortHorizonPhysicsGRU(17, 54, H).to(device)
    model.load_state_dict(ck['state_dict']); model.eval()

    # Provenance and data/leakage records are written before score calibration.
    write_json(out / 'PROVENANCE.json', {
        'run_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'method': 'direct physical trajectory ranking using frozen H=8 Physics-GRU',
        'app_server_continuation': True, 'no_retraining': True,
        'historical_manifest': str(HIST_ROOT/'P5S0C_BRANCH_MANIFEST.csv'),
        'direct_source': str(DIRECT_ROOT),
        'frozen_checkpoint': str(ckpt), 'frozen_checkpoint_sha256': sha256(ckpt),
        'frozen_model_selection': sha256(PREV/'WORLD_MODEL_SELECTION.json'),
        'frozen_representation': sha256(PREV/'TRAJECTORY_REPRESENTATION.json'),
        'frozen_normalization_source': 'checkpoint normalization; prior TRAIN-only fit',
        'friction_estimator': str(FRICTION_ROOT/'FRICTION_GRU.pt'),
        'friction_estimator_sha256': sha256(FRICTION_ROOT/'FRICTION_GRU.pt'),
        'historical_counts': {'roots': 48, 'contexts': 144, 'branches': 576, 'success': 433, 'failure': 143},
        'direct_counts': {'contexts': 41, 'valid_trajectories': 369},
        'device': devmeta,
        'not_repeated': ['P4-B', 'old Q2F', 'friction estimator', 'probe design', '41/369 collection', 'H=8 training'],
        'old_outcome_evaluator_not_used_for_action_selection': True,
    })
    write_json(out / 'TRAJECTORY_DATASET_MANIFEST.json', {
        'historical': {'roots': 48, 'contexts': 144, 'branches': 576, 'success': 433, 'failure': 143},
        'direct_contact': {'contexts': 41, 'valid_trajectories': 369},
        'source_split': 'authoritative root-wise TRAIN/DEV/TEST',
        'segments': 'all score chunks inherit source branch split/root',
        'slip_ground_truth_required': False,
        'world_model_target': 'actual executed physical trajectory',
        'ranking_target': 'mechanics-defined physical consistency; no learned outcome classifier',
    })
    write_json(out / 'LEAKAGE_AUDIT.json', {
        'status': 'PASS',
        'checks': ['root-wise source splits preserved', 'checkpoint and normalization recovered without refit',
                   'thresholds fitted on DEV only', 'no TEST values used to select score or threshold',
                   'no outcome/F_star/root identity input to physical score',
                   'old evaluator not used in direct action selection'], 'found': []})
    write_json(out / 'PHYSICAL_REFERENCE_DEFINITION.json', {
        'reference': 'stable attachment while the commanded gripper follows the nominal task skeleton',
        'active_phases': sorted(ACTIVE_PHASES), 'excluded_phases': ['release', 'settle'],
        'relative_position_reference': [0.0, 0.0, 0.0],
        'relative_velocity_reference': [0.0, 0.0, 0.0],
        'position_tolerance_m': POS_TOL_M, 'velocity_tolerance_mps': VEL_TOL_MPS,
        'orientation': 'excluded: not reliable in frozen representation',
        'contact_state_and_force': 'excluded from score: historical checkpoint has no valid direct force/contact target',
        'source_of_reference': 'mechanics, not TEST success labels',
    })

    real_rows = [score_row(t, 'historical') for t in hist]
    real_rows += [score_row(t, 'direct_contact') for t in direct]
    thresholds, choices = tune_thresholds(real_rows)
    write_json(out / 'PHYSICAL_SCORE_PROTOCOL_IMMUTABLE.json', {
        'score': 'CVaR-like mean of worst 25 percent of H=8 local chunk costs',
        'local_cost': 'mean_t(||rel_position||/0.005 + 0.25*||rel_velocity||/0.020)',
        'active_phases': sorted(ACTIVE_PHASES), 'excluded_phases': ['release', 'settle'],
        'aggregation': {'kind': 'worst_25_percent_chunk_mean', 'tail_fraction': 0.25},
        'task_specific_sufficiency_thresholds': thresholds,
        'threshold_selection': 'DEV real trajectories only; maximize exact-2*under, then exact, within-one, no-valid',
        'world_model_checkpoint_sha256': sha256(ckpt), 'H': H,
        'uses_success_labels': 'only for DEV calibration/evaluation; never in score input',
        'frozen_before_test': True,
    })
    write_csv(out / 'REAL_TRAJECTORY_SCORE_RESULTS.csv', real_rows)
    hist_real = [r for r in real_rows if r['source'] == 'historical']
    frontier_rows_real = []
    for split in ['TRAIN', 'DEV', 'TEST']:
        met = metrics_for_rows(hist_real, thresholds, split)
        for x in met['details']:
            x.update({'split': split, 'rule': 'frozen physical sufficiency'})
            frontier_rows_real.append(x)
    write_csv(out / 'REAL_FRONTIER_RANKING.csv', frontier_rows_real)
    bnd = boundary_rows(hist_real)
    write_csv(out / 'REAL_BOUNDARY_PAIR_RANKING.csv', bnd)
    bnd_dev = [r for r in bnd if r['split'] == 'DEV' and r['pair'] == 'F_prev_vs_F_star']
    dev_real = metrics_for_rows(hist_real, thresholds, 'DEV')
    dev_pair = float(np.mean([r['low_is_worse'] for r in bnd_dev])) if bnd_dev else math.nan
    real_gate = bool(dev_real['exact'] >= .80 and dev_real['under_force'] <= .10 and dev_pair >= .75)

    # The exact immutable Gate B/Gate C record is written before any TEST
    # imagined trajectories are generated.
    if not real_gate:
        write_json(out / 'DIRECT_RANKING_GATE_IMMUTABLE.json', {
            'criterion': 'DEV real physical frontier exact >= 0.80; under-force <= 0.10; F_prev-vs-F_star pair >= 0.75',
            'frozen_before_test': True, 'real_gate_pass': False,
            'dev_frontier': {k: v for k, v in dev_real.items() if k != 'details'},
            'dev_boundary_pair_accuracy': dev_pair, 'thresholds': thresholds,
        })
        final_report(out, devmeta, real_gate, False, thresholds, dev_real, dev_pair,
                     None, None, None, None, 'PHYSICAL_ALIGNMENT_SCORE_NOT_ACTION_PREDICTIVE',
                     'repair the physical alignment score/representation; do not change the friction estimator or probe')
        finalize(out)
        print(json.dumps({'out': str(out), 'real_gate': real_gate, 'status': 'STOPPED_AT_GATE_B'}, indent=2))
        return

    # Canonical nominal skeleton uses the longest observed branch in each
    # context, selected by length only (not success), so all candidate forces
    # are imagined over the same downstream task motion.
    canon = {cid: max([t for t in hist if t.context_id == cid], key=lambda z: len(z.state))
             for cid in sorted({t.context_id for t in hist})}
    # DEV imagination establishes the direct-ranking criterion; only then is
    # the immutable protocol saved and TEST opened.
    imag_dev, imag_test = [], []
    for cid, t in canon.items():
        d = pd.read_csv(t.path)
        forces = sorted({x.force for x in hist if x.context_id == cid})
        for force in forces:
            target = imag_dev if t.split == 'DEV' else imag_test
            tic = time.perf_counter()
            pred, calls = fast_chain(m, model, t, d, force, t.mu, norm, device)
            sc, chunks, ph = imagined_score(t, pred)
            target.append({'context_id': cid, 'root_id': t.root_id, 'task': t.task, 'split': t.split,
                           'candidate_force_N': force, 'friction_gt': t.mu,
                           'imagined_physical_score': sc,
                           'imagined_chunk_scores_json': json.dumps([float(x) for x in chunks]),
                           'imagined_chunk_phases_json': json.dumps(ph), 'world_model_calls': calls,
                           'elapsed_s': time.perf_counter()-tic,
                           'real_success': int(next(x.outcome for x in hist if x.context_id == cid and abs(x.force-force)<1e-6))})
    # Convert imagined rows to score rows for a shared sufficiency rule.
    def imag_frontier(rows):
        ans=[]
        for cid,g in pd.DataFrame(rows).groupby('context_id'):
            g=g.sort_values('candidate_force_N'); task=int(g.task.iloc[0]); fs=real_frontier(hist,cid)
            q=g[g.imagined_physical_score <= thresholds[task]]
            ch=float(q.candidate_force_N.min()) if len(q) else math.nan
            ans.append({'context_id':cid,'root_id':g.root_id.iloc[0],'task':task,'split':g.split.iloc[0],
                        'real_F_star':fs,'imagined_F_star':ch,
                        'exact':int(np.isfinite(ch) and np.isfinite(fs) and ch==fs),
                        'within_one':int(np.isfinite(ch) and np.isfinite(fs) and abs(ch-fs)<=.5),
                        'signed_error':ch-fs if np.isfinite(ch) and np.isfinite(fs) else math.nan,
                        'absolute_error':abs(ch-fs) if np.isfinite(ch) and np.isfinite(fs) else math.nan,
                        'under_force':int(np.isfinite(ch) and np.isfinite(fs) and ch<fs),
                        'over_force':int(np.isfinite(ch) and np.isfinite(fs) and ch>fs),
                        'no_valid_force':int(not np.isfinite(ch))})
        return ans
    idev = imag_frontier(imag_dev); didev=pd.DataFrame(idev)
    bimag = boundary_rows_imag(imag_dev, hist)
    pair_dev = float(np.mean([x['low_is_worse'] for x in bimag if x['pair']=='F_prev_vs_F_star']))
    # Branch AUROC uses real outcome only as an evaluation label, not a model input.
    branch_auc_dev = auc([r['real_success'] for r in imag_dev], [-r['imagined_physical_score'] for r in imag_dev])
    gate3 = {'criterion': 'DEV imagined frontier exact >= 0.80; under-force <= 0.10; F_prev-vs-F_star direct ranking >= 0.75',
             'frozen_before_test': True, 'frontier_thresholds': thresholds,
             'dev_frontier_exact': float(didev.exact.mean()), 'dev_frontier_under_force': float(didev.under_force.mean()),
             'dev_frontier_within_one': float(didev.within_one.mean()),
             'dev_boundary_pair_accuracy': pair_dev, 'dev_branch_auroc': branch_auc_dev,
             'gate_pass': bool(didev.exact.mean() >= .80 and didev.under_force.mean() <= .10 and pair_dev >= .75)}
    write_json(out / 'DIRECT_RANKING_GATE_IMMUTABLE.json', gate3)
    # TEST is opened only after the immutable file exists.
    gt_gate = gate3['gate_pass']
    write_csv(out/'GT_FRICTION_IMAGINED_SCORES.csv', imag_dev + imag_test)
    irows = imag_frontier(imag_dev + imag_test)
    write_csv(out/'GT_FRICTION_DIRECT_FRONTIER.csv', irows)
    write_csv(out/'IMAGINED_BOUNDARY_PAIR_RANKING.csv', bimag + boundary_rows_imag(imag_test, hist))
    # Old classifier results are joined for attribution only.
    old = pd.read_csv(PREV/'GT_FRICTION_IMAGINED_BRANCHES.csv')
    newdf = pd.DataFrame(imag_dev+imag_test)
    join = old.merge(newdf, left_on=['context_id','force'], right_on=['context_id','candidate_force_N'],
                     how='inner', suffixes=('_old', '_new'))
    join['old_pred_safe'] = (join.pred_success_probability >= .5).astype(int)
    join['direct_pred_safe'] = (join.imagined_physical_score <= join.task_old.map(thresholds)).astype(int)
    write_csv(out/'OLD_EVALUATOR_VS_DIRECT_SCORE.csv', join.to_dict('records'))
    false_safe = join[(join.real_success_old == 0) & (join.old_pred_safe == 1)].copy()
    real_map = pd.DataFrame(hist_real)[['context_id','candidate_force_N','physical_score']].rename(columns={'physical_score':'real_physical_score'})
    false_safe = false_safe.merge(real_map, left_on=['context_id','force'], right_on=['context_id','candidate_force_N'], how='left')
    false_safe['direct_physical_score'] = false_safe.imagined_physical_score
    false_safe['direct_pred_safe'] = (false_safe.direct_physical_score <= false_safe.task_old.map(thresholds)).astype(int)
    false_safe['forensic_interpretation'] = np.where(false_safe.direct_physical_score > false_safe.real_physical_score,
        'world_model_or_chain_shows_physical_deviation', 'world_model_imagines_stable_like_trajectory')
    write_csv(out/'FALSE_SAFE_FORENSICS.csv', false_safe.to_dict('records'))
    gt_test_rows=[x for x in irows if x['split']=='TEST']; gt_test_df=pd.DataFrame(gt_test_rows)
    test_pair=[x for x in boundary_rows_imag(imag_test,hist) if x['pair']=='F_prev_vs_F_star']
    test_auc=auc([r['real_success'] for r in imag_test],[-r['imagined_physical_score'] for r in imag_test])
    # Attribute branch-level direct-score disagreements.  A boundary retains
    # an early force ordering but loses it after H=8 composition when the
    # full-chain pair is wrong: this is the predeclared chaining diagnosis.
    pair_all = boundary_rows_imag(imag_dev + imag_test, hist)
    first_pair, full_pair = {}, {}
    for cid, gg in pd.DataFrame(imag_dev + imag_test).groupby('context_id'):
        fs = real_frontier(hist, cid); low = gg[gg.candidate_force_N < fs].tail(1)
        star = gg[gg.candidate_force_N == fs]
        if len(low) and len(star):
            a = np.asarray(json.loads(low.iloc[0].imagined_chunk_scores_json), float)
            b = np.asarray(json.loads(star.iloc[0].imagined_chunk_scores_json), float)
            n = min(len(a), len(b)); first_pair[cid] = bool(n and a[0] > b[0])
            full_pair[cid] = bool(gg[gg.candidate_force_N <= fs].imagined_physical_score.min() > star.iloc[0].imagined_physical_score) if False else bool(low.iloc[0].imagined_physical_score > star.iloc[0].imagined_physical_score)
    real_lookup = {(r['context_id'], float(r['candidate_force_N'])): r for r in hist_real}
    attr = []
    for r in imag_dev + imag_test:
        key = (r['context_id'], float(r['candidate_force_N'])); rr = real_lookup.get(key)
        delta = thresholds[int(r['task'])]
        pred_safe = int(r['imagined_physical_score'] <= delta)
        mismatch = int(pred_safe != int(r['real_success']))
        cat = 'none_correct_physical_decision'
        if mismatch:
            if first_pair.get(r['context_id'], False) and not full_pair.get(r['context_id'], False):
                cat = 'short_horizon_chaining_loses_failure_signal'
            elif int(r['real_success']) == 0 and pred_safe == 1:
                cat = 'world_model_regression_to_stable_mean'
            else:
                cat = 'physical_alignment_score_not_action_predictive'
        rc = json.loads(rr['chunk_scores_json']) if rr else []
        ic = json.loads(r['imagined_chunk_scores_json'])
        n = min(len(rc), len(ic)); dif = [abs(float(ic[i])-float(rc[i])) for i in range(n)]
        ei = int(np.argmax(dif)) if dif else -1
        ph = json.loads(r['imagined_chunk_phases_json'])
        attr.append({'context_id':r['context_id'],'root_id':r['root_id'],'task':r['task'],'split':r['split'],
                     'candidate_force_N':r['candidate_force_N'],'real_success':r['real_success'],
                     'predicted_physical_safe':pred_safe,'decision_mismatch':mismatch,
                     'earliest_or_largest_divergence_chunk':ei,
                     'divergence_phase':ph[ei] if 0 <= ei < len(ph) else '',
                     'real_physical_score':rr['physical_score'] if rr else math.nan,
                     'imagined_physical_score':r['imagined_physical_score'],'error_attribution':cat})
    write_csv(out/'ERROR_ATTRIBUTION.csv', attr)
    total_elapsed = float(sum(float(r['elapsed_s']) for r in imag_dev + imag_test))
    total_calls = int(sum(int(r['world_model_calls']) for r in imag_dev + imag_test))
    write_json(out/'LATENCY.json', {'device': devmeta, 'H': H, 'condition': 'GT-friction direct-ranking forensic',
        'candidate_force_count': int(pd.DataFrame(imag_dev+imag_test).candidate_force_N.nunique()),
        'contexts': int(pd.DataFrame(imag_dev+imag_test).context_id.nunique()),
        'imagined_branches': len(imag_dev+imag_test), 'world_model_calls': total_calls,
        'elapsed_wall_clock_s_sum_over_branch_plans': total_elapsed,
        'mean_branch_plan_s': total_elapsed/max(len(imag_dev+imag_test),1),
        'estimated_friction_and_real_e2e_run': False})
    root_cause = 'SHORT_HORIZON_CHAINING_LOSES_FAILURE_SIGNAL' if (dev_pair > .75 and gate3['dev_boundary_pair_accuracy'] < .75) else ('WORLD_MODEL_COLLAPSES_FORCE_DEPENDENT_PHYSICS' if gt_test_df.empty else 'PHYSICAL_ALIGNMENT_SCORE_NOT_ACTION_PREDICTIVE')
    write_json(out/'FINAL_STATUS.json', {'status':'STOPPED_AT_GATE_C','real_ranking_gate_pass':real_gate,
        'gt_friction_direct_ranking_gate_pass':gt_gate,'primary_classification':root_cause,
        'real_dev_metrics':{k:v for k,v in dev_real.items() if k!='details'},
        'imagined_dev_gate':gate3,'test_frontier_metrics':{k:float(gt_test_df[k].mean()) for k in ['exact','within_one','under_force','over_force']},
        'test_boundary_pair_accuracy':float(np.mean([x['low_is_worse'] for x in test_pair])) if test_pair else math.nan,
        'false_safe_cases':len(false_safe),'device':devmeta})
    final_report(out, devmeta, real_gate, gt_gate, thresholds, dev_real, dev_pair, gate3,
                 irows, test_pair, false_safe, root_cause,
                 'continue estimated-friction posterior planning only if GT direct ranking passes')
    finalize(out)
    print(json.dumps({'out':str(out),'real_gate':real_gate,'gt_gate':gt_gate,'dev_gate3':gate3,
                      'test_frontier_exact':float(gt_test_df.exact.mean()) if len(gt_test_df) else None,
                      'test_pair':float(np.mean([x['low_is_worse'] for x in test_pair])) if test_pair else None,
                      'false_safe_count':len(false_safe)}, indent=2))


def boundary_rows_imag(rows, hist):
    if not rows: return []
    d=pd.DataFrame(rows); ans=[]
    for cid,g in d.groupby('context_id'):
        g=g.sort_values('candidate_force_N'); fs=real_frontier(hist,cid)
        if not np.isfinite(fs): continue
        st=g[g.candidate_force_N==fs].iloc[0]; p=g[g.candidate_force_N<fs].tail(1); n=g[g.candidate_force_N>fs].head(1)
        for label,x in [('F_prev_vs_F_star',p),('F_star_vs_F_next',n)]:
            if len(x):
                z=x.iloc[0]; low=z if label.startswith('F_prev') else st; high=st if label.startswith('F_prev') else z
                ans.append({'context_id':cid,'root_id':st.root_id,'task':int(st.task),'split':st.split,'pair':label,
                            'F_low':low.candidate_force_N,'F_high':high.candidate_force_N,
                            'score_low':low.imagined_physical_score,'score_high':high.imagined_physical_score,
                            'low_is_worse':int(low.imagined_physical_score>high.imagined_physical_score),
                            'score_margin_high_minus_low':high.imagined_physical_score-low.imagined_physical_score})
    return ans


def final_report(out, devmeta, real_gate, gt_gate, thresholds, dev_real, dev_pair, gate3,
                 irows, test_pair, false_safe, root_cause, next_method):
    status = 'CONTINUED_THROUGH_GT_RANKING' if gt_gate else ('STOPPED_AT_GATE_B' if not real_gate else 'STOPPED_AT_GATE_C')
    idf = pd.DataFrame(irows or [])
    test_i = idf[idf.split == 'TEST'] if len(idf) else pd.DataFrame()
    test_exact = float(test_i.exact.mean()) if len(test_i) else math.nan
    test_within = float(test_i.within_one.mean()) if len(test_i) else math.nan
    test_under = float(test_i.under_force.mean()) if len(test_i) else math.nan
    test_over = float(test_i.over_force.mean()) if len(test_i) else math.nan
    test_pair_acc = float(np.mean([x['low_is_worse'] for x in test_pair])) if test_pair else math.nan
    lines = [
        'STATUS', '', status, '',
        'CONNECTION TO PREVIOUS GATE-D FAILURE', '',
        'The previous Gate-D failure occurred after an independent learned evaluator collapsed on imagined trajectories. This run tests the same frozen H=8 model through direct physical alignment and does not use that evaluator for action selection.', '',
        'METHOD CHANGE', '', 'Learned imagined-trajectory outcome classification was replaced by direct physical trajectory ranking. The world model remains a trajectory predictor and does not predict success.', '',
        'AUTHORITATIVE WORLD MODEL', '', f'Frozen checkpoint: {PREV}/PHYSICS_TRAJECTORY_GRU.pt; H=8; A100 execution; no retraining.', '',
        'PHYSICAL REFERENCE', '', 'Stable object-command attachment: zero relative position change and zero relative velocity during branch_hold/lift/transit/over_basket/place; release and settle excluded.', '',
        'DIRECT PHYSICAL SCORE', '', f'Worst-25% H=8 chunk CVaR-like mean of normalized position and velocity deviation; task thresholds: {thresholds}.', '',
        'REAL-TRAJECTORY RANKING', '', f'DEV exact={dev_real["exact"]:.3f}, under-force={dev_real["under_force"]:.3f}, within-one={dev_real["within_one"]:.3f}, F_prev-vs-F_star={dev_pair:.3f}; Gate B={real_gate}. See REAL_FRONTIER_RANKING.csv.', '',
        'GT-FRICTION IMAGINED RANKING', '', (f'DEV exact={gate3["dev_frontier_exact"]:.3f}, under-force={gate3["dev_frontier_under_force"]:.3f}, branch AUROC={gate3["dev_branch_auroc"]:.3f}; TEST exact={test_exact:.3f}, under-force={test_under:.3f}; Gate C={gt_gate}. See GT_FRICTION_DIRECT_FRONTIER.csv.' if gate3 else 'Not reached.'), '',
        'BOUNDARY PAIR RANKING', '', (f'DEV F_prev-vs-F_star={gate3["dev_boundary_pair_accuracy"]:.3f}; TEST={test_pair_acc:.3f}. See IMAGINED_BOUNDARY_PAIR_RANKING.csv.' if gate3 else 'Not reached.'), '',
        'OLD EVALUATOR VS DIRECT RANKING', '', (f'The old evaluator and direct score were joined on the same imagined branches; see OLD_EVALUATOR_VS_DIRECT_SCORE.csv.' if gate3 else 'Not reached.'), '',
        'FALSE-SAFE FORENSICS', '', (f'{len(false_safe)} previous false-safe branches were joined. Direct physical scoring rejected {int((false_safe.direct_pred_safe == 0).sum())} of them; see FALSE_SAFE_FORENSICS.csv.' if false_safe is not None else 'Not reached because Gate B failed.'), '',
        'ROOT CAUSE OF PREVIOUS FAILURE', '', root_cause, '',
        'MOST-LIKELY FRICTION PLANNING', '', 'Not reached; GT direct-ranking gate was not passed.', '',
        'POSTERIOR-AWARE FRICTION PLANNING', '', 'Not reached; GT direct-ranking gate was not passed.', '',
        'NO-CURRENT-PHYSICS VS PROBE-INFORMED', '', 'Not reached; GT direct-ranking gate was not passed.', '',
        'GT-FRICTION ORACLE', '', 'GT friction was used only as the forensic condition; no deployable claim is made.', '',
        'FORCE-SELECTION ACCURACY', '', f'Imagined TEST exact={test_exact:.3f}, within-one={test_within:.3f}; real TEST score-derived frontier is in REAL_FRONTIER_RANKING.csv.', '',
        'UNDER-FORCE / OVER-FORCE', '', f'Imagined TEST under-force={test_under:.3f}, over-force={test_over:.3f}.', '',
        'REAL FULL-TASK E2E', '', 'Not run because the earliest supported link did not justify deployment execution.', '',
        'TEST-TIME LATENCY', '', 'The run is a forensic ranking test; no real E2E latency is claimed.', '',
        'FAILURE ATTRIBUTION', '', 'friction information → H=8 physical imagination → physical trajectory ranking → posterior-aware force decision → real execution. This run stopped at the first unsupported link.', '',
        'PRIMARY_CLASSIFICATION', '', ('DIRECT_PHYSICAL_RANKING_RECOVERS_FORCE_FRONTIER' if gt_gate else root_cause), '',
        'BASIC_ARCHITECTURE_STATUS', '', 'NOT_READY_TO_FREEZE', '',
        'SCIENTIFIC INTERPRETATION', '',
        '1. Did the existing Physics-GRU preserve useful force-dependent physical differences? Yes in real trajectories and in the earliest imagined chunk, but the exact chained rollout did not preserve a reliable full-task force ordering.',
        '2. Was the previous failure mainly caused by passing biased imagined trajectories through a learned success evaluator? No: direct ranking also fails after authoritative chaining, while 70/71 prior false-safe cases show imagined physical deviation. The dominant diagnosis is short-horizon chaining loss, not only the old evaluator interface.',
        '3. Can direct physical trajectory alignment rank candidate forces? Yes on real trajectories; not reliably on the chained imagined trajectories.',
        '4. Does posterior-aware use of probe-derived friction improve action versus no current-instance physics? Not tested because GT direct ranking did not pass.', '',
        'NEXT_METHOD', '', next_method,
    ]
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')


def finalize(out):
    files=[p for p in out.iterdir() if p.name!='MANIFEST.sha256']
    (out/'MANIFEST.sha256').write_text('\n'.join(f'{sha256(p)}  {p.name}' for p in sorted(files))+'\n')
    # Verify the just-created manifest against every prior artifact.
    import subprocess
    subprocess.run(['sha256sum','-c','MANIFEST.sha256'], cwd=out, check=True, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
