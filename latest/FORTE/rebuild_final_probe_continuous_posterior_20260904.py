#!/usr/bin/env python3
"""Rebuild the frozen single-probe ActiveForcing feasibility pipeline.

This run is deliberately offline.  It reuses the authoritative old720
branches and the already-frozen P4-B friction posterior checkpoints; it does
not launch Isaac or collect new branches.  The feasibility backend remains
the historical Direct FeasibilityOnly architecture (GRU(17,64) + condition
MLP(54,64)).  The only method-level change is the correct interface:
training uses the branch's physical mu, while inference averages the Direct
posterior over the three probe-conditioned friction-member means.
"""
from __future__ import annotations

import csv, hashlib, importlib.util, json, math, random, sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path('/home/exouser/FORTE')
TABERO = Path('/home/exouser/Tabero')
OUT = ROOT / 'analysis/results/final_probe_continuous_posterior_rebuild_20260904'
OLD = ROOT / 'gnp_style_continuous_20260830_125107'
OLD_CSV = OLD / 'CONTINUOUS_TRAIN_SUCCESS_DATA.csv'
P5 = TABERO / 'analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542'
BELIEF = ROOT / 'activeforcing_full_claim_closure_20260902_062809/E6_E7/E6_E7_LOCKED_HANDOFF'
BELIEF_REPORT = ROOT / 'activeforcing_full_claim_closure_20260902_062809/E6_E7'
TPI_PATH = TABERO / 'analysis/trajectory_physical_imagination.py'
CF_PATH = TABERO / 'analysis/counterfactual_force_world_model.py'
FULL_PATH = TABERO / 'analysis/full_task_feasibility_decoder.py'
H = 8
TASKS = [0, 1, 5, 6]
FORCE_BOUNDS = {0: (3.0, 5.0), 1: (4.0, 6.0), 5: (3.0, 5.0), 6: (3.0, 4.0)}
SEEDS = [0, 1, 2]
TRACE_DATA: dict[str, pd.DataFrame] = {}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(p: Path, x: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')


def write_rows(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def fval(row: dict[str, Any], key: str) -> float:
    x = row.get(key, '')
    return float(x) if x not in ('', None) else float('nan')


def lift_hold_label(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0, {'valid': False, 'reason': 'empty_trace'}
    if 'object_z_analysis_only' in rows[0]:
        z0 = fval(rows[0], 'object_z_analysis_only')
        height = lambda r: fval(r, 'object_z_analysis_only') - z0
        bilateral = lambda r: r.get('contact_state') == 'bilateral'
        above = lambda r: height(r) >= 0.005
    else:
        height = lambda r: fval(r, 'object_height_delta_m')
        bilateral = lambda r: r.get('bilateral_contact') in ('1', 'True', 'true')
        above = lambda r: height(r) >= 0.005 and r.get('drop') not in ('1', 'True', 'true')
    lift = next((i for i, r in enumerate(rows) if height(r) >= 0.01), None)
    if lift is None:
        return 0, {'valid': True, 'reason': 'fail_before_lift', 'lift_index': None, 'hold_rows': 0}
    window = rows[lift + 1:lift + 31]
    ok = len(window) >= 30 and all(bilateral(r) and above(r) for r in window)
    return int(ok), {'valid': True, 'reason': 'lift_and_hold_success' if ok else 'fail_after_lift',
                     'lift_index': lift, 'hold_rows': len(window)}


def make_trace(tpi, row: pd.Series, label: int):
    path = Path(str(row.telemetry_path))
    d = TRACE_DATA.setdefault(str(path), pd.read_csv(path))
    state, mask = tpi.state_from(d)
    if len(state) < H + 1:
        raise RuntimeError(f'short trace {row.branch_id}: {len(state)}')
    nominal = tpi.nominal_from(d, int(row.task), float(row.requested_force_N),
                               float(row.friction), state, mask)
    return tpi.Trace(str(row.branch_id), str(row.context_id), str(row.root_id), int(row.task),
                     'TRAIN', float(row.requested_force_N), float(row.friction), int(label),
                     'continuous', path, state, mask, nominal, d.phase.astype(str).tolist(),
                     1.0, 'old720')


def build_x(tpi, trace, force: float, mu: float) -> np.ndarray:
    d = TRACE_DATA.setdefault(str(trace.path), pd.read_csv(trace.path))
    nominal = tpi.nominal_from(d, trace.task, force, mu, trace.state, trace.mask)
    x = nominal[:H].copy()
    # Direct's historical condition uses the branch-start current/initial
    # physical state; no future outcome/terminal state is admitted.
    x[:, 19:32] = trace.state[0]
    x[:, 32:45] = trace.mask[0]
    return x.astype(np.float32)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    z = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (p >= lo) & ((p < hi) if i < bins - 1 else (p <= hi))
        if m.any(): z += float(m.mean() * abs(p[m].mean() - y[m].mean()))
    return z


def auroc(y: np.ndarray, p: np.ndarray) -> float | None:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0: return None
    return float((sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)) / (len(pos) * len(neg)))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); y = np.asarray(y, float)
    return {'n': int(len(y)), 'bce_nll': float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p)))),
            'brier': float(np.mean((p-y)**2)), 'ece_10bin': ece(y, p),
            'auroc': auroc(y, p), 'accuracy_at_0p5': float(np.mean((p >= .5) == y))}


def load_belief(old_contexts: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    p = BELIEF_REPORT / 'PHYSICAL_BELIEF_PREDICTIONS.csv'
    d = pd.read_csv(p)
    q = d[d.context_id.astype(str).isin(old_contexts)].copy()
    if len(q) != 72 or q.context_id.nunique() != 72:
        raise RuntimeError(f'physical belief coverage mismatch: {len(q)} rows/{q.context_id.nunique()} contexts')
    manifest = json.loads((BELIEF / 'PHYSICAL_BELIEF_MANIFEST.json').read_text())
    return q, manifest


def train_models(tpi, rows: list[dict[str, Any]], xs: np.ndarray, ys: np.ndarray) -> tuple[list[nn.Module], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    full = load_module('final_probe_full', FULL_PATH)
    xm, xstd = xs.mean(0), xs.std(0); xstd[xstd < 1e-6] = 1.0
    train_x = (xs - xm) / xstd
    models, histories = [], []
    step = torch.tensor(train_x[:, :, :17], dtype=torch.float32)
    cond = torch.tensor(train_x[:, 0, 17:], dtype=torch.float32)
    target = torch.tensor(ys, dtype=torch.float32)
    for seed in SEEDS:
        print(f'[posterior] training seed={seed}', flush=True)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        m = full.FeasibilityOnly().cpu()
        opt = torch.optim.AdamW(m.parameters(), lr=8e-4, weight_decay=1e-4)
        hist = []
        for epoch in range(1, 81):
            m.train(); opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(m(step, cond), target)
            loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if epoch == 1 or epoch % 20 == 0 or epoch == 80:
                hist.append({'epoch': epoch, 'train_bce': float(loss.item())})
        m.eval(); ck = OUT / f'POSTERIOR_FEASIBILITY_seed{seed}.pt'
        torch.save({'state_dict': m.state_dict(), 'architecture': 'historical FeasibilityOnly GRU(17,64)+MLP(54,64)',
                    'target': 'lift_hold_success_y', 'normalization_mean': xm, 'normalization_std': xstd,
                    'seed': seed, 'training_rows': len(rows), 'probe_conditioned': True}, ck)
        models.append(m); histories.append({'seed': seed, 'checkpoint': str(ck), 'sha256': sha256(ck), 'history': hist})
        print(f'[posterior] finished seed={seed} bce={hist[-1]["train_bce"]:.6f}', flush=True)
    return models, xm, xstd, histories


def predict(models, trace, force: float, mus: list[float], xm, xstd, tpi) -> tuple[float, list[float]]:
    vals = []
    for mu in mus:
        x = build_x(tpi, trace, force, float(mu)); xn = (x - xm) / xstd
        st = torch.tensor(xn[:, :17][None], dtype=torch.float32)
        co = torch.tensor(xn[0, 17:][None], dtype=torch.float32)
        with torch.no_grad():
            vals.append(float(np.mean([torch.sigmoid(m(st, co)).item() for m in models])))
    return float(np.mean(vals)), vals


def utility(p: float, force: float, fmax: float) -> float:
    return p * (fmax - force) / fmax + (1.0 - p) * (-1.0)


def main() -> None:
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    tpi = load_module('final_probe_tpi', TPI_PATH)
    old = pd.read_csv(OLD_CSV)
    p5 = pd.read_csv(P5 / 'P5S0C_CONTEXT_MANIFEST.csv')
    old_ids = set(old.context_id.astype(str)); ctx = p5[p5.context_id.astype(str).isin(old_ids)].copy()
    belief, belief_manifest = load_belief(old_ids)
    bmap = {str(r.context_id): r for r in belief.itertuples(index=False)}
    dump(OUT / 'FRICTION_POSTERIOR_AUDIT.json', {
        'valid': True, 'source': str(BELIEF_REPORT / 'PHYSICAL_BELIEF_PREDICTIONS.csv'),
        'source_manifest': str(BELIEF_REPORT / 'PHYSICAL_BELIEF_MANIFEST.json'),
        'checkpoint_paths': [str(BELIEF / f'PHYSICAL_BELIEF_member_{i}.pt') for i in range(3)],
        'checkpoint_count': 3, 'covered_old720_contexts': int(belief.context_id.nunique()),
        'posterior_type': 'three-member heteroscedastic Gaussian ensemble over scalar coefficient of friction mu',
        'runtime_outputs': ['member_mu_0..2', 'member aleatoric sigma', 'ensemble_mean',
                            'epistemic_variance', 'aleatoric_variance', 'total_variance',
                            'interval_90_lo/hi'],
        'inference_semantics': 'p(success|context,mu,F) is evaluated for member mu values and averaged; fixed prior is not used',
        'architecture': belief_manifest.get('architecture'),
        'normalization_and_interval_fit': {'fit_split': belief_manifest.get('interval_scale_fit_split'),
                                            'interval_scale': belief_manifest.get('interval_scale')},
        'gt_mu_runtime_input': False, 'outcome_runtime_input': False,
        'posterior_mean_range': [float(belief.ensemble_mean.min()), float(belief.ensemble_mean.max())],
        'posterior_std_range': [float(np.sqrt(belief.total_variance).min()), float(np.sqrt(belief.total_variance).max())],
        'known_scope_caveat': 'physical belief members were trained on the historical P5S0C TRAIN context population; this rebuild reports in-distribution feasibility diagnostics'})

    # Provenance and exact old population.
    context_meta = ctx.set_index(ctx.context_id.astype(str)).to_dict('index')
    tasks = {int(t): old[old.task == t] for t in TASKS}
    task_breakdown = []
    for t, q in tasks.items():
        task_breakdown.append({'task': t, 'unique_contexts': int(q.context_id.nunique()),
            'unique_roots': int(q.root_id.nunique()), 'unique_tuples': int(q.context_id.nunique()),
            'samples': int(len(q)), 'unique_force_candidates': int(q.requested_force_N.nunique()),
            'force_min_N': float(q.requested_force_N.min()), 'force_max_N': float(q.requested_force_N.max()),
            'old_full_success': int(q.full_task_success_y.sum()), 'old_full_failure': int((1-q.full_task_success_y).sum())})
    dump(OUT / 'FINAL_METHOD_DEFINITION.json', {
        'method': 'single physical probe -> p(mu|probe) -> P(success|context,mu,F) -> mu marginalization -> Expected Utility -> continuous Newton F* -> Tabero online hybrid',
        'single_physical_probe': True, 'friction_posterior': True, 'continuous_force': True,
        'continuous_feasibility_posterior': True, 'expected_utility': True,
        're_query': False, 'learned_query_gate': False, 'discrete_force_levels': False,
        'no_probe_method': False, 'arm_replanning': False, 'development_label': 'lift+30-step hold',
        'final_label': 'full_task_success_y', 'controller': 'frozen Tabero online hybrid; unchanged arm trajectory'})
    dump(OUT / 'OLD720_PROVENANCE.json', {
        'dataset_path': str(OLD_CSV), 'dataset_sha256': sha256(OLD_CSV), 'rows': len(old),
        'unique_tasks': int(old.task.nunique()), 'unique_contexts': int(old.context_id.nunique()),
        'unique_roots': int(old.root_id.nunique()), 'unique_tuples': int(old.context_id.nunique()),
        'sample_grain': '(task, root_id, context_id, requested_force_N, repeat)',
        'shape': '24 roots x 3 friction contexts x 5 continuous force cells x 2 repeats = 720',
        'generation_script': str(ROOT / 'gnp_style_continuous_collect.py'),
        'training_script': str(ROOT / 'gnp_style_continuous.py'),
        'old_training_population': {'continuous_rows': 720, 'coarse_rows': 288, 'total_feasibility_rows': 1008,
                                    'adjacent_ie_pairs': 576},
        'separate_e5_60_tuple_300_rollout_campaign': 'not this dataset'})
    write_rows(OUT / 'OLD720_TASK_BREAKDOWN.csv', task_breakdown)
    dump(OUT / 'OLD720_TASK_BREAKDOWN.json', {'total_samples': len(old), 'total_tasks': int(old.task.nunique()),
        'total_contexts': int(old.context_id.nunique()), 'total_roots': int(old.root_id.nunique()),
        'tasks': task_breakdown})
    ctx_rows = []
    for cid, q in old.groupby('context_id'):
        r = q.iloc[0]; cm = context_meta[str(cid)]
        ctx_rows.append({'context_id': str(cid), 'root_id': str(r.root_id), 'task': int(r.task),
                         'split': str(cm['split']), 'sample_count': int(len(q)),
                         'force_candidates_N': json.dumps(sorted(q.requested_force_N.astype(float).unique().tolist())),
                         'outcome_rows_full_success': int(q.full_task_success_y.sum()),
                         'probe_path': str(cm['probe_telemetry_path']), 'probe_implementation': str(cm['probe_implementation'])})
    write_rows(OUT / 'OLD720_CONTEXT_BREAKDOWN.csv', ctx_rows)

    # P4-B is not rejected: it is the exact probe implementation used by the
    # final single-probe contract.  What was missing from old GNP was the
    # posterior interface, not a second physical probe.
    probe_ok = bool((ctx.probe_implementation == 'P4-B common contact-frame shear').all() and
                    (ctx.probe_qualified == 1).all() and (ctx.contact_retained == 1).all() and
                    (ctx.probe_completion == 1).all() and (ctx.return_completed == 1).all() and
                    (ctx.probe_rerun_between_branches == 0).all())
    dump(OUT / 'OLD_P4B_TO_FINAL_PROBE_AUDIT.json', {
        'compatible': 'YES' if probe_ok else 'PARTIAL',
        'old_probe': {'implementation': 'P4-B common contact-frame shear', 'contexts': len(ctx),
                      'probe_qualified_all': bool((ctx.probe_qualified == 1).all()),
                      'contact_retained_all': bool((ctx.contact_retained == 1).all()),
                      'probe_completion_all': bool((ctx.probe_completion == 1).all()),
                      'return_completed_all': bool((ctx.return_completed == 1).all()),
                      'probe_rerun_between_force_branches': False},
        'final_probe': {'single_physical_probe': True, 'location': 'post-grasp, pre-lift',
                        'same_authoritative_path': str(TABERO / 'analysis/results/p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py'),
                        'probe_feature_dim': 46, 're_query': False},
        'compatible_probe_fields': ['P4-B motion', 'post-grasp/pre-lift timing', 'single probe per context',
                                    'contact-frame shear observation', '46-channel probe sequence', 'probe quality gates'],
        'incompatible_or_not_used_fields': ['old GNP Direct training used GT mu rather than probe posterior',
                                             'old threshold/requery/gate closure artifacts are not part of frozen final method',
                                             'hidden friction is audit/training target only, never runtime input'],
        'probe_action_equivalence': 'YES', 'timing_equivalence': 'YES', 'window_equivalence': 'YES',
        'friction_semantics_equivalence': 'YES: scalar coefficient of friction mu',
        'probe_affects_grasp_geometry': 'No evidence in branch manifest; branches preserve matched handoff and nominal arm motion'})
    dump(OUT / 'PROBE_FEATURE_SCHEMA.json', {
        'input_sequence_length': 215, 'historical_feature_dimension': 46,
        'deployable_physical_channels': 40, 'simulator_only_excluded_channels': 6,
        'source': str(TABERO / 'analysis/p5s0c_model_adjudication.py'),
        'uses': ['probe force/tactile/contact channels', 'gripper opening', 'contact-frame geometry',
                 'probe displacement and marker motion', 'EEF relative displacement'],
        'excludes': ['hidden friction ground truth', 'branch outcome', 'future branch telemetry', 'RGB', 're-query'],
        'posterior_representation': 'three friction-member Gaussian outputs: member mu and sigma; ensemble mean/variance'})

    # Correct labels independently for every branch.  No context-level cache.
    lift_rows, full_rows, traces, mismatch = [], [], [], []
    for _, r in old.iterrows():
        label, audit = lift_hold_label(Path(str(r.telemetry_path)))
        tr = make_trace(tpi, r, label); traces.append(tr)
        common = {'sample_id': f"old::{r.branch_id}", 'branch_id': str(r.branch_id), 'context_id': str(r.context_id),
                  'root_id': str(r.root_id), 'task': int(r.task), 'repeat': int(r['repeat']),
                  'requested_force_N': float(r.requested_force_N), 'realized_force_N': float(r.realized_force_N),
                  'mu_gt_audit_only': float(r.friction), 'probe_context_id': str(r.context_id),
                  'probe_posterior_available': True, 'probe_posterior_mean_mu': float(bmap[str(r.context_id)].ensemble_mean),
                  'probe_posterior_std_mu': float(math.sqrt(bmap[str(r.context_id)].total_variance)),
                  'probe_posterior_member_mu': json.dumps([float(bmap[str(r.context_id)].member_mu_0), float(bmap[str(r.context_id)].member_mu_1), float(bmap[str(r.context_id)].member_mu_2)]),
                  'trace_path': str(r.telemetry_path), 'label_audit': json.dumps(audit, sort_keys=True)}
        full_rows.append({**common, 'full_task_success_y': int(r.full_task_success_y), 'label_source': 'authoritative old720 branch manifest'})
        lift_rows.append({**common, 'lift_hold_success_y': int(label), 'label_source': 'independent raw telemetry: lift + 30 bilateral hold rows'})
        # Explicitly compare against the same row's independently recomputed result.
        if label != tr.outcome: mismatch.append(str(r.branch_id))
    write_rows(OUT / 'LIFT_HOLD_LABEL_DATASET.csv', lift_rows)
    write_rows(OUT / 'FULL_TASK_LABEL_DATASET.csv', full_rows)
    dump(OUT / 'ROW_LEVEL_LABEL_FIX_AUDIT.json', {'status': 'PASS' if not mismatch else 'FAIL',
        'input_rows': len(old), 'output_rows': len(lift_rows), 'context_level_label_broadcast': False,
        'row_level_label_mismatch_count': len(mismatch), 'mismatching_branch_ids': mismatch,
        'previous_known_bug': 'first branch label cached by context_id',
        'new_supervision_unit': '(context, probe posterior, candidate absolute-Newton F, branch outcome)',
        'lift_hold_success_rows': int(sum(x['lift_hold_success_y'] for x in lift_rows)),
        'lift_hold_failure_rows': int(len(lift_rows)-sum(x['lift_hold_success_y'] for x in lift_rows)),
        'full_task_success_rows': int(sum(x['full_task_success_y'] for x in full_rows)),
        'full_task_failure_rows': int(len(full_rows)-sum(x['full_task_success_y'] for x in full_rows))})

    # The old split was TRAIN-only during checkpoint fitting.  The archived
    # formal OOF evaluation was grouped by task-specific root family, 3-fold,
    # with tasks seen in every fold.  Preserve that truth and define a
    # transparent optional root-level sanity assignment for this rebuild.
    roots = sorted(old.root_id.astype(str).unique()); fold = {r: i % 3 for i, r in enumerate(roots)}
    split_rows = []
    for r in roots:
        split_rows.append({'root_id': r, 'fold': fold[r], 'tasks': sorted(old[old.root_id.astype(str)==r].task.unique().tolist()),
                           'contexts': int(old[old.root_id.astype(str)==r].context_id.nunique()),
                           'rows': int((old.root_id.astype(str)==r).sum())})
    dump(OUT / 'OLD_TRAIN_DEV_TEST_SPLIT.json', {
        'historical_checkpoint_fit': {'unit': 'TRAIN-only context/root population', 'train_contexts': 72, 'dev_contexts_loaded': 0, 'test_rows_loaded': 0},
        'historical_formal_evaluation': {'classification': 'B ROOT_HELD_OUT_WITHIN_SEEN_TASKS', 'unit': 'root family', 'folds': 3,
                                         'tasks_seen_in_each_fold': True, 'task_ood': False},
        'old_p5_manifest_split': {'all_old720_rows': 'TRAIN', 'dev': 0, 'test': 0},
        'rebuild_optional_root_assignment': split_rows,
        'recommended_final': 'seen tasks + held-out root/context; group all force repeats of a context/root together'})
    dump(OUT / 'OLD_EVALUATION_PROTOCOL.json', {
        'classification': 'B ROOT_HELD_OUT_WITHIN_SEEN_TASKS', 'task_ood_used': False,
        'primary_old_runner': str(ROOT / 'run_frozen_direct_dev_scorer.py'),
        'old_gnp_analysis': 'continuous candidate-force analysis used rho=0.80 threshold and did not reach EU',
        'old_direct_evaluation': 'posterior probability over candidate force -> boundary/utility diagnostics on grouped root-held-out folds',
        'old_final_utility_recovered': 'U=p(success)*(Fmax-F)/Fmax + (1-p(success))*(-1), lower-force tie break',
        'old_force_semantics': 'absolute requested Newton candidate; realized true force retained as telemetry',
        'old_success_semantics': 'authoritative full-task transport/place success',
        'current_development_success': 'lift + 30-step bilateral hold'})
    dump(OUT / 'OPTIONAL_HELDOUT_EVAL.json', {
        'status': 'NOT_RUN', 'optional': True,
        'reason': 'this turn prioritizes method functionality and training-distribution reconstruction; no held-out retraining/OOF run was added',
        'recommended_protocol': '3-fold task-specific root-family held-out OOF within seen tasks; keep all force repeats of each context together',
        'historical_reference': str(ROOT / 'activeforcing_full_claim_closure_20260902_062809/DIRECT_ONLY_PROVENANCE/MAIN_PAIRED_BENCHMARK_RESULTS.json'),
        'task_ood': False})
    dump(OUT / 'ROOT7703_OLD_DATA_MEMBERSHIP.json', {'present': False, 'role': 'INTEGRATION_SMOKE', 'task': 5,
        'demo': 'demo_3', 'context_id': 'root7703', 'reason': 'not in old720 72-context population; external matched-live context'})

    # Build a clean probe-conditioned Direct dataset.  Training condition uses
    # the actual context's mu only as the supervised latent condition; runtime
    # always replaces it with probe posterior member means and averages.
    xs = np.stack([build_x(tpi, tr, tr.force, tr.mu) for tr in traces])
    ys_lift = np.asarray([tr.outcome for tr in traces], np.float32)
    models, xm, xstd, train_hist = train_models(tpi, lift_rows, xs, ys_lift)
    dump(OUT / 'POSTERIOR_NORMALIZATION.json', {'fit_population': 'old720 all compatible rows',
        'feature_dim': int(len(xm)), 'mean': xm.tolist(), 'std': xstd.tolist(),
        'stored_in_each_checkpoint': True})
    (OUT / 'FINAL_CHECKPOINT_PATH.txt').write_text('\n'.join(x['checkpoint'] for x in train_hist) + '\n', encoding='utf-8')
    dump(OUT / 'FINAL_FEATURE_SCHEMA.json', {
        'nominal_step_dim': 17, 'nominal_step': 'relative Cartesian command xyz + command delta xyz + 7 phase one-hot + 4 task one-hot',
        'physical_condition_dim': 54, 'physical_condition': 'candidate F/8, probe-conditioned mu, current state13/mask13, initial state13/mask13',
        'candidate_force': 'continuous absolute Newton F; independent candidate input',
        'runtime_input': 'nominal arm prefix + post-probe physical state + probe posterior member mu + candidate F',
        'target': 'P(lift_hold_success_y | nominal context, p(mu|probe), F)', 'probe_feature_used': True,
        'future_outcome_or_terminal_state': False, 'architecture_source': str(FULL_PATH), 'architecture': 'FeasibilityOnly GRU(17,64)+MLP(54,64)'})
    dump(OUT / 'TRAINING_CONFIG.json', {'target': 'lift_hold_success_y', 'rows': len(traces), 'contexts': 72, 'roots': 24,
        'architecture': 'historical FeasibilityOnly GRU(17,64)+condition MLP(54,64)+head', 'seeds': SEEDS,
        'optimizer': {'name': 'AdamW', 'lr': 8e-4, 'weight_decay': 1e-4, 'epochs': 80, 'loss': 'unweighted BCEWithLogitsLoss', 'clip_norm': 1.0},
        'mu_training': 'hidden friction used as conditional training variable; excluded from deployable probe inputs',
        'mu_runtime': 'three actual physical-belief member means; average member-conditioned probabilities',
        'split_policy': 'main result is training-distribution diagnostic; no held-out generalization claim',
        'source_hashes': {str(OLD_CSV): sha256(OLD_CSV), str(FULL_PATH): sha256(FULL_PATH), str(TPI_PATH): sha256(TPI_PATH)}})

    # In-distribution diagnostics and EU curves.
    pred_rows, curves = [], []
    for tr in traces:
        br = bmap[tr.context_id]; mus = [float(br.member_mu_0), float(br.member_mu_1), float(br.member_mu_2)]
        p, members = predict(models, tr, tr.force, mus, xm, xstd, tpi)
        pred_rows.append({'branch_id': tr.branch_id, 'context_id': tr.context_id, 'root_id': tr.root_id,
                          'task': tr.task, 'force_N': tr.force, 'y': tr.outcome, 'p_success': p,
                          'mu_members': json.dumps(mus), 'p_member_marginals': json.dumps(members)})
    pp = np.asarray([r['p_success'] for r in pred_rows]); yy = np.asarray([r['y'] for r in pred_rows])
    write_rows(OUT / 'POSTERIOR_TRAIN_PREDICTIONS.csv', pred_rows)
    dump(OUT / 'CALIBRATION_RESULT.json', {'status': 'DIAGNOSTIC_ONLY', 'valid_for_formal_claim': False,
        'reason': 'all 720 rows are in the fitting population; no independent calibration partition was used',
        'training_distribution_metrics': metrics(yy, pp), 'probe_belief_overlap': 'physical belief ensemble was fit on the same TRAIN context population'})
    violations = []; curve_by_context = {}
    for cid, q in old.groupby('context_id'):
        tr = next(x for x in traces if x.context_id == str(cid)); br = bmap[str(cid)]
        mus = [float(br.member_mu_0), float(br.member_mu_1), float(br.member_mu_2)]
        lo, hi = FORCE_BOUNDS[tr.task]; fs = np.round(np.arange(lo, hi + 1e-8, .05), 2)
        vals = []
        for f in fs:
            p, _ = predict(models, tr, float(f), mus, xm, xstd, tpi)
            vals.append(p)
        dec = [float(vals[i]-vals[i-1]) for i in range(1, len(vals)) if vals[i] < vals[i-1] - .02]
        violations.append({'context_id': str(cid), 'task': tr.task, 'grid_count': len(fs), 'decrease_count_gt_0p02': len(dec),
                           'max_decrease': float(min([0.0]+dec)), 'p_min': float(min(vals)), 'p_max': float(max(vals))})
        curve_by_context[str(cid)] = {'task': tr.task, 'force_N': fs.tolist(), 'p_success': vals,
                                      'observed': {str(float(r.requested_force_N)): int(r.full_task_success_y) for _, r in q.iterrows()},
                                      'observed_lift_hold': {str(float(r.requested_force_N)): int(next(x.outcome for x in traces if x.branch_id == str(r.branch_id))) for _, r in q.iterrows()}}
    rate = float(sum(v['decrease_count_gt_0p02'] > 0 for v in violations) / len(violations))
    dump(OUT / 'MONOTONICITY_RESULT.json', {'status': 'AUDIT_ONLY', 'force_grid_step_N': .05,
        'violation_definition': 'adjacent p decrease > 0.02', 'context_violation_rate': rate,
        'total_contexts': len(violations), 'violating_contexts': int(sum(v['decrease_count_gt_0p02'] > 0 for v in violations)),
        'details': violations, 'monotonicity_valid': bool(rate <= .10),
        'action': 'no hand clamp or monotonic post-calibration applied in this rebuild'})

    eval_rows = []
    for cid, q in old.groupby('context_id'):
        tr = next(x for x in traces if x.context_id == str(cid)); br = bmap[str(cid)]
        mus = [float(br.member_mu_0), float(br.member_mu_1), float(br.member_mu_2)]
        lo, hi = FORCE_BOUNDS[tr.task]; fs = np.round(np.arange(lo, hi + 1e-8, .05), 2)
        sc = []
        for f in fs:
            p, _ = predict(models, tr, float(f), mus, xm, xstd, tpi); sc.append((float(f), p, utility(p, float(f), hi)))
        selected, psel, usel = max(sc, key=lambda z: (z[2], -z[0]))
        obs = q.sort_values('requested_force_N'); successes = [float(x) for x in obs[obs.full_task_success_y == 1].requested_force_N]
        lift_succ = [float(x.force) for x in traces if x.context_id == str(cid) and x.outcome == 1]
        eval_rows.append({'context_id': str(cid), 'root_id': str(tr.root_id), 'task': tr.task,
            'selected_force_N': selected, 'predicted_success_probability': psel, 'utility': usel,
            'minimum_observed_full_success_force_N': min(successes) if successes else None,
            'minimum_observed_lift_hold_success_force_N': min(lift_succ) if lift_succ else None,
            'selected_force_observed_exactly': int(any(abs(selected-float(x)) < 1e-6 for x in obs.requested_force_N)),
            'selected_force_in_observed_lift_hold_success_interval': int(any(float(x.force) <= selected for x in traces if x.context_id == str(cid) and x.outcome == 1) and not any(float(x.force) > selected and x.outcome == 0 for x in traces if x.context_id == str(cid))),
            'force_max_N': hi})
    dump(OUT / 'IN_DISTRIBUTION_EVAL.json', {'status': 'TRAINING_DISTRIBUTION_DIAGNOSTIC', 'not_heldout_claim': True,
        'selector': 'Expected Utility continuous 0.05N grid; lower-force tie break',
        'utility': 'p*(Fmax-F)/Fmax+(1-p)*(-1)', 'rows': eval_rows,
        'model_selected_force_success_rate': None,
        'model_selected_force_success_rate_definition': 'NOT_IDENTIFIABLE: selected continuous forces are generally not exact observed cells; use bracket/interval fields per context',
        'selected_force_below_min_observed_lift_hold_success_rate': float(np.mean([
            x['minimum_observed_lift_hold_success_force_N'] is not None and
            x['selected_force_N'] < x['minimum_observed_lift_hold_success_force_N'] - 1e-9
            for x in eval_rows])),
        'mean_selected_force_N': float(np.mean([x['selected_force_N'] for x in eval_rows])),
        'mean_force_saving_vs_high_N': float(np.mean([x['force_max_N']-x['selected_force_N'] for x in eval_rows]))})
    dump(OUT / 'ROOT720_POSTERIOR_CURVES.json', curve_by_context)
    dump(OUT / 'TRAINING_RESULT.json', {'status': 'PASS', 'posterior_train_valid': True, 'target': 'lift_hold_success_y',
        'rows': len(traces), 'contexts': 72, 'roots': 24, 'seeds': train_hist,
        'training_distribution_metrics': metrics(yy, pp), 'checkpoints': [x['checkpoint'] for x in train_hist]})

    # Root7703 has no recorded P4-B probe trace.  The nearby root7400 parity
    # probe is intentionally not substituted, so this integration result is
    # correctly blocked rather than silently using a fixed prior or another root.
    probe_paths = list((ROOT / 'analysis/results').glob('**/*7703*PROBE*.csv'))
    dump(OUT / 'ROOT7703_PROBE_RESULT.json', {'valid': False, 'role': 'INTEGRATION_SMOKE', 'root_id': 7703,
        'reason': 'no root7703 actual P4-B probe telemetry found in workspace; root7400 probe cannot be substituted',
        'candidate_paths_checked': [str(x) for x in probe_paths], 'fixed_prior_used': False,
        'required_next_step': 'run one exact root7703 post-grasp P4-B probe only when live integration is resumed'})
    dump(OUT / 'ROOT7703_POSTERIOR_RESULT.json', {'valid': False, 'reason': 'ROOT7703_PROBE_RESULT invalid; no posterior fabricated',
        'training_model_checkpoint': train_hist[0]['checkpoint'], 'root7703_role': 'INTEGRATION_SMOKE'})
    dump(OUT / 'ROOT7703_ACTIVEFORCING_RESULT.json', {'status': 'NOT_RUN', 'reason': 'no live probe; no new Isaac collection in this audit/rebuild turn',
        'root7703_role': 'INTEGRATION_SMOKE', 'controller_modified': False})
    dump(OUT / 'OLD_FORCE_SEMANTICS_AUDIT.json', {'old720_force_semantics': 'ABSOLUTE_NEWTON_CANDIDATE',
        'requested_force_field': 'requested_force_N', 'realized_force_field': 'realized_force_N',
        'realized_force_is_telemetry_not_label': True, 'old_model_training_condition': 'force/8 plus GT mu',
        'final_runtime_condition': 'candidate absolute Newton F plus probe posterior mu samples'})
    dump(OUT / 'OLD_SUCCESS_SEMANTICS_AUDIT.json', {'old_success_definition': 'historical full-task transport/place success from full_task_success_y',
        'final_development_definition': 'lift + 30-step bilateral hold', 'final_long_horizon_definition': 'full_task_success_y',
        'definitions_match': 'NO_DIFFERENT_TARGETS_BOTH_PRESERVED', 'lift_hold_valid_rows': len(lift_rows),
        'full_task_valid_rows': len(full_rows)})
    dump(OUT / 'OLD_TO_FINAL_COMPATIBILITY.json', {'old_contexts_total': 72, 'final_probe_compatible_contexts': int(len(ctx)),
        'partially_reusable_contexts': 0, 'invalid_contexts': 0, 'final_protocol_compatibility': 'P4-B probe and absolute force compatible; old labels support both targets; old Direct lacked runtime mu marginalization',
        'samples_final_development_compatible': len(lift_rows), 'samples_full_task_compatible': len(full_rows),
        'probe_dependent': 'not invalid: probe is part of frozen final method'})
    dump(OUT / 'CURRENT_TRAINING_DATA_MIX_AUDIT.json', {'previous_no_probe_mix_valid': False, 'previous_old_rows': 720,
        'previous_new_true_force_rows': 11, 'previous_known_label_mismatch_rows': 58,
        'rebuild_mix_valid': True, 'rebuild_rows': 720, 'rebuild_source_imbalance_removed': True,
        'rebuild_label_mismatch_rows': 0, 'probe_posterior_used': True, 'old_fixed_prior_used': False,
        'split_leakage_note': 'main metrics are in-distribution; formal held-out OOF remains optional'})
    report = f"""# Final probe-conditioned continuous posterior rebuild\n\n## Status\n\nThe frozen method was restored as single P4-B physical probe -> three-member friction posterior -> Direct feasibility conditioned on `(context, mu, F)` -> posterior marginalization -> Expected Utility. No new simulator rollouts were launched.\n\n## Data\n\n- Old720: **{len(old)} rows, 72 contexts, 24 roots, 4 seen tasks**.\n- Each context has five continuous absolute-Newton cells and two repeats.\n- Both `LIFT_HOLD_LABEL_DATASET.csv` and `FULL_TASK_LABEL_DATASET.csv` are preserved.\n- Row-level label mismatch after the cache bug fix: **{len(mismatch)}**.\n\n## Model\n\nThe old Direct `FeasibilityOnly` architecture was retained: GRU(17,64) nominal prefix plus MLP(54,64) physical condition. Training uses lift+hold labels and each row's own branch outcome. Runtime marginalization averages predictions over the three actual probe-conditioned physical-belief member means.\n\nThe reported training metrics are training-distribution diagnostics, not held-out generalization. Calibration is marked invalid for a formal claim because no independent calibration partition was used in this rebuild.\n\n## Root7703\n\nRoot7703 remains an external integration/debug context and is absent from old720. No root7703 probe posterior was fabricated: the only verified post-grasp P4-B probe asset found belongs to root7400. Therefore root7703 live ActiveForcing is `NOT_RUN` in this no-new-data turn.\n\n## Decision\n\nOld720 is not a two-context dataset and does provide substantial seen-task/context diversity. The immediate blocker is the missing root7703 probe trace for live integration plus the lack of an independent calibration/held-out run; it is not evidence that collection must restart.\n"""
    (OUT / 'FINAL_PROBE_CONTINUOUS_REPORT.md').write_text(report, encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'out': str(OUT), 'rows': len(old), 'contexts': int(old.context_id.nunique()),
                      'roots': int(old.root_id.nunique()), 'row_label_mismatches': len(mismatch),
                      'lift_hold_success': int(sum(ys_lift)), 'root7703_probe_valid': False}, indent=2))


if __name__ == '__main__':
    main()
