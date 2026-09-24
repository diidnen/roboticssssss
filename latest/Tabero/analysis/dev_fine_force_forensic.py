#!/usr/bin/env python3
"""DEV-only fine-force and strict pre-probe forensic replay.

No model is trained or modified here.  The existing P5-S0-C simulator
controller is imported read-only; this script only fixes a DEV force grid and
records restored branches under the existing controller semantics.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
if os.environ.get('FINE_WORKER') == '1':
    pd = None
else:
    import pandas as pd

ROOT = Path('/home/exouser/Tabero')
RES = ROOT / 'analysis/results'
HIST = RES / 'p5s0c_paired_boundary_probe_value_20260824_000542'
PREV = RES / 'probe_informed_imagination_20260829_121500'
WM = RES / 'trajectory_physical_imagination_20260829_065220'
CAL = RES / 'evaluator_interface_calibration_20260829_112603'
P5SRC = ROOT / 'analysis/p5s0c_paired_boundary_probe_value.py'
P4SRC = RES / 'p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py'
ISAAC_PY = Path('/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python')
WARP_CORE = Path('/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64')
OPENPI = ROOT / 'benchmarks/openpi/openpi-client/src'
TASKS = [int(x) for x in os.environ.get('FINE_TASKS', '0,1,5,6').split(',') if x.strip()]
REPEATS = 3
GRID_STEP = 0.25
OUT = Path(os.environ.get('FINE_OUT', RES / f'dev_fine_force_forensic_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}'))


def h(path: Path) -> str:
    z = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            z.update(b)
    return z.hexdigest()


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def context_info():
    b = pd.read_csv(HIST / 'P5S0C_BRANCH_MANIFEST.csv')
    dev = b[b.split.eq('DEV')].copy()
    out = []
    for cid, g in dev.groupby('context_id'):
        success = g[g.full_task_success_y.eq(1)]
        fstar = float(success.requested_force_N.min()) if len(success) else np.nan
        below = g[(g.requested_force_N < fstar) & g.full_task_success_y.eq(0)]
        fprev = float(below.requested_force_N.max()) if len(below) else np.nan
        out.append({'context_id': cid, 'root_id': g.root_id.iloc[0], 'task': int(g.task.iloc[0]),
                    'split': 'DEV', 'friction_band': g.friction_band.iloc[0],
                    'friction': float(g.hidden_friction_analysis_only.iloc[0]),
                    'coarse_F_prev': fprev, 'coarse_F_star': fstar,
                    'context_seed': int(g.seed.iloc[0])})
    df = pd.DataFrame(out).sort_values(['task', 'root_id', 'friction_band']).reset_index(drop=True)
    selected = {x.strip() for x in os.environ.get('FINE_CONTEXT_IDS', '').split(',') if x.strip()}
    if selected:
        df = df[df.context_id.isin(selected)].reset_index(drop=True)
    return df


def fine_grid(prev, star):
    if not np.isfinite(prev) or not np.isfinite(star) or star <= prev:
        return []
    return [round(float(x), 2) for x in np.arange(prev, star + 1e-8, GRID_STEP)]


def preflight_and_freeze():
    OUT.mkdir(parents=False, exist_ok=False)
    ci = context_info()
    force_manifest = json.loads((HIST / 'P5S0C_FORCE_MANIFEST.json').read_text())
    protocol_sources = [P5SRC, P4SRC, HIST / 'P5S0C_BRANCH_MANIFEST.csv', HIST / 'P5S0C_CONTEXT_MANIFEST.csv',
                        HIST / 'P5S0C_STATE_PARITY.csv', HIST / 'P5S0C_ROOT_STATE_PARITY.csv',
                        WM / 'PHYSICS_TRAJECTORY_GRU.pt', CAL / 'CALIBRATED_EVALUATOR_FREEZE.json',
                        PREV / 'PROBE_INFORMED_IMAGINATION_PROVENANCE.json']
    hashes = {rel(p): h(p) for p in protocol_sources if p.exists()}
    write_json(OUT / 'CONTINUOUS_FORCE_CONTROLLER_AUDIT.json', {
        'status': 'SUPPORTED_BY_EXISTING_CONTROLLER',
        'controller_source': rel(P4SRC), 'controller_source_sha256': h(P4SRC),
        'make_action_code': 'a[0,9]=0.5*f_star; a[0,12]=0.5*f_star; f_star is float; no integer cast or force clip',
        'force_servo_code': 'force feedback changes d_pred only; np.clip applies to gripper displacement d_pred in [D_CLOSED,D_OPEN]',
        'accepts_arbitrary_float_force': True,
        'native_4_25_4_50_4_75_support': True,
        'prior_executed_fine_or_half_step': {'P5S0C_half_step_values_N': force_manifest['offgrid_force_values_N'], 'executed_in_primary_branch_manifest': True},
        'controller_modification_required': False,
        'force_clipping_or_saturation': 'No command-force clip/saturation in _make_action; physical measured force can differ due servo dynamics',
        'physical_meaning': 'same per-finger target semantics as integer branches, each finger receives one half of requested total f_star',
        'validated_replay_protocol': 'P5-S0-C process-isolated Isaac worker, env.scene.get_state(is_relative=True), env.reset_to(...), existing P4-B downstream_branch',
        'decision': 'CONTINUE_DEV_FINE_REPLAY'
    })
    fifth = {
        'intended_fifth_task_id': 2,
        'task_description': 'pick up the salad dressing and place it in the basket',
        'exists_in_p4_task_object_map': True,
        'exists_in_authoritative_P5S0C_contexts': False,
        'valid_DEV_frontier_data': False,
        'previous_test_presence': False,
        'why_absent': 'P5S0C protocol explicitly fixes TASKS=[0,1,5,6] and task2_used=false; the task2 dataset/frontier was not part of the validated matched lineage.',
        'eligible_for_this_replay': False,
        'evidence': [rel(HIST / 'P5S0C_PROTOCOL.json'), rel(ROOT / 'analysis/research/m0_api_and_method_design_20260821_151412/README.md')]
    }
    write_json(OUT / 'FIFTH_TASK_COVERAGE_AUDIT.json', fifth)
    grids = []
    for r in ci.to_dict('records'):
        grids.append({**r, 'fine_grid_N': fine_grid(r['coarse_F_prev'], r['coarse_F_star']), 'repeats_per_force': REPEATS})
    write_json(OUT / 'DEV_FINE_FORCE_FORENSIC_PROTOCOL.json', {
        'name': 'DEV fine-force frontier and strict pre-probe baseline forensic',
        'scope': 'DEV only; no TEST fine-force replay; no fresh E2E; no training or calibration',
        'contexts': ci.context_id.tolist(), 'context_count': len(ci), 'tasks': TASKS,
        'roots': sorted(ci.root_id.unique().tolist()), 'friction_bands': ['LOW', 'MID', 'HIGH'],
        'coarse_bracket_source': rel(HIST / 'P5S0C_BRANCH_MANIFEST.csv'),
        'fine_grid_rule': 'fixed inclusive 0.25 N grid from largest known insufficient force to minimum known sufficient coarse force; no result-driven additions',
        'grids': grids, 'repeats_per_force': REPEATS,
        'success_definition': 'inherited full_task_success_y from existing deterministic downstream controller; no new criterion',
        'frontier_summary_rule': 'minimum-successful-force is reported per repeat only when observed; no 2/3 or 3/3 aggregate threshold invented',
        'stochasticity_rule': 'empirical success frequencies and repeat disagreement are reported; F_FINE_STAR aggregate is NOT formally identifiable without an inherited repeated-outcome rule',
        'physical_event_definition': 'inherited full-task failure/outcome plus direct telemetry fields; release/settling not reclassified',
        'state_parity': 'restore exact same post-probe snapshot for probe-state branches and exact captured pre-probe snapshot for strict branches; hash every restore',
        'simulation_render_interval': int(os.environ.get('FINE_RENDER_INTERVAL', '1')),
        'simulation_render_note': 'infrastructure-only render cadence; action/force/controller code and physics timestep unchanged',
        'strict_probe_informed': 'execute P4-B once to obtain frozen estimator evidence, then restore captured pre-probe state before downstream fine branch; probe-perturbed state is never used',
        'strict_no_physics': 'restore the same captured pre-probe state and execute no probe in the downstream counterfactual; no probe trace or estimator output is an input to the branch',
        'frozen_stack': {'world_model': rel(WM / 'PHYSICS_TRAJECTORY_GRU.pt'), 'calibration': rel(CAL / 'CALIBRATED_EVALUATOR_FREEZE.json'),
                         'evaluator': rel(WM / 'OUTCOME_EVALUATOR.pt'), 'friction_estimator': rel(RES / 'active_friction_imagination_20260828_211106/FRICTION_GRU.pt'),
                         'probe': rel(HIST / 'P5S0C_FROZEN_PROBE.json')},
        'source_hashes': hashes
    })
    write_json(OUT / 'STRICT_PREPROBE_BASELINE_PROTOCOL.json', {
        'status': 'FROZEN_BEFORE_OUTCOMES', 'same_contexts': ci.context_id.tolist(),
        'probe_condition': 'pre-probe snapshot -> P4-B evidence -> frozen estimator diagnostics -> restore exact pre-probe snapshot -> downstream branch',
        'strict_no_physics_condition': 'same pre-probe snapshot -> no P4-B execution -> same downstream branch semantics',
        'only_information_difference_intended': 'frozen probe-derived mu_hat/evidence versus pre-existing nominal friction prior',
        'post_probe_state_for_imagination': False,
        'downstream_branch_controller': rel(P5SRC), 'state_api': 'env.scene.get_state(is_relative=True) and env.reset_to',
        'restore_hash_required': True, 'fresh_test_or_e2e': False
    })
    write_json(OUT / 'PROVENANCE.json', {'namespace': str(OUT), 'created_utc': datetime.now(timezone.utc).isoformat(),
                                         'authoritative_inputs': [rel(PREV), rel(RES / 'force_quantization_conditioning_forensic_20260829_130800'), rel(HIST)],
                                         'test_touched': False, 'training': False, 'recalibration': False, 'fresh_e2e': False,
                                         'device_requirement': 'Isaac inference workers use existing CUDA env; no model training', 'source_hashes': hashes,
                                         'dev_contexts': ci.to_dict('records'), 'simulation_render_interval': int(os.environ.get('FINE_RENDER_INTERVAL', '1')),
                                         'simulation_render_note': 'infrastructure-only render cadence; no controller/action/physics timestep change'})
    return ci


def load_p5_module(out):
    os.environ['P5S0C_OUT'] = str(out)
    os.environ['P5S0C_WORKER'] = '1'
    spec = importlib.util.spec_from_file_location('p5s0c_fine_replay', P5SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def worker(task: int):
    from isaaclab.app import AppLauncher
    enable_cameras = os.environ.get('FINE_ENABLE_CAMERAS', '1') not in {'0', 'false', 'False'}
    app = AppLauncher(headless=True, enable_cameras=enable_cameras, num_envs=1).app
    # Isaac Sim requires SimulationApp before any Omniverse-backed imports.
    import torch
    import gymnasium as gym
    import tac_manip.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from tac_manip.utils.task_configs import setup_task_objects
    mod = load_p5_module(OUT)
    task_dir = OUT / f'task{task}'; task_dir.mkdir(parents=True, exist_ok=True)
    log = task_dir / 'worker.log'
    rows = []
    parity = []
    try:
        p4 = mod.import_p4_probe(task)
        setup_task_objects(mod.TASK_SUITE, task)
        cfg = parse_env_cfg(mod.ENV_ID, device='cuda:0', num_envs=1)
        cfg.episode_length_s = 45.0
        render_interval = os.environ.get('FINE_RENDER_INTERVAL', '')
        if render_interval:
            cfg.sim.render_interval = int(render_interval)
        env = gym.make(mod.ENV_ID, cfg=cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        # The Isaac environment intentionally has no pandas dependency.  The
        # frozen protocol is the worker input and is read with stdlib JSON.
        plans = [x for x in json.loads((OUT / 'DEV_FINE_FORCE_FORENSIC_PROTOCOL.json').read_text())['grids'] if int(x['task']) == task]
        base_step = env.step
        for pr in plans:
            cid, mu, seed = pr['context_id'], float(pr['friction']), int(pr['context_seed'])
            counter = {'n': 0, 'pre': None}
            def wrapped_step(action, _orig=base_step, _counter=counter):
                result = _orig(action)
                _counter['n'] += 1
                # P4-B reaches the stable pre-probe state after approach,
                # descend, close, and hold: 45+35+70+40=190 env steps.
                if _counter['n'] == 190:
                    _counter['pre'] = env.scene.get_state(is_relative=True)
                return result
            env.step = wrapped_step
            probe_rows, probe_rec = p4.run_probe_episode(env, seed_idx=seed, mu=mu, trial_id=cid, dt=dt)
            pre = counter['pre']
            post = env.scene.get_state(is_relative=True)
            if pre is None:
                raise RuntimeError('PREPROBE_CAPTURE_MISSING')
            pre_hash = mod.stable_hash_obj(mod.restorable_snapshot_for_hash(env)) if False else ''
            # Hashes are computed after explicit restores below; the source
            # snapshot object itself is not mutated by env.reset_to.
            for baseline, snap in [('probe_restore', post), ('strict_preprobe', pre)]:
                env.reset_to(snap, torch.tensor([0], device=env.device), is_relative=True)
                target_hash = mod.stable_hash_obj(mod.restorable_snapshot_for_hash(env))
                for force in fine_grid(pr['coarse_F_prev'], pr['coarse_F_star']):
                    for rep in range(1, REPEATS + 1):
                        before = mod.stable_hash_obj(mod.restorable_snapshot_for_hash(env))
                        env.reset_to(snap, torch.tensor([0], device=env.device), is_relative=True)
                        after = mod.stable_hash_obj(mod.restorable_snapshot_for_hash(env))
                        pass_parity = int(after == target_hash)
                        parity.append({'context_id': cid, 'task': task, 'baseline': baseline, 'force_N': force, 'repeat': rep,
                                        'target_hash': target_hash, 'before_hash': before, 'after_hash': after,
                                        'parity_pass': pass_parity})
                        if not pass_parity:
                            rows.append({'context_id': cid, 'root_id': pr['root_id'], 'task': task, 'friction': mu,
                                         'friction_band': pr['friction_band'], 'baseline': baseline, 'force_N': force,
                                         'repeat': rep, 'validity': 0, 'engineering_failure': 'RESTORE_PARITY_FAIL',
                                         'success': '', 'unsafe_event': '', 'bilateral_contact_fraction': ''})
                            continue
                        label = f'{baseline}_F{str(force).replace(".", "p")}_R{rep}'
                        telem = task_dir / 'telemetry' / f'{cid}_{label}.csv'
                        logger = mod.StageLogger(OUT, task, seed, dt=dt, context_id=cid, split='DEV', friction=mu)
                        br = mod.downstream_branch(env, p4, task_id=task, force=float(force), branch_label=label,
                                                   context_id=cid, split='DEV', seed=seed, friction=mu, dt=dt,
                                                   logger=logger, telemetry_path=telem,
                                                   label_source='DEV_FINE_FORCE_FORENSIC_' + baseline)
                        td = pd.read_csv(telem)
                        phase = td.phase.astype(str)
                        maint = ~phase.isin(['release', 'settle'])
                        bilateral = float((td.loc[maint, 'contact_state'].astype(str).eq('bilateral')).mean()) if maint.any() else np.nan
                        unsafe = int((br.get('dropped', 0) or br.get('lost_in_transit', 0) or not br.get('full_task_success_y', 0)))
                        rows.append({'context_id': cid, 'root_id': pr['root_id'], 'task': task, 'friction': mu,
                                     'friction_band': pr['friction_band'], 'baseline': baseline, 'force_N': force,
                                     'repeat': rep, 'validity': 1, 'engineering_failure': '',
                                     'success': int(br.get('full_task_success_y', 0)), 'unsafe_event': unsafe,
                                     'bilateral_contact_fraction': bilateral, 'dropped': int(br.get('dropped', 0)),
                                     'lost_in_transit': int(br.get('lost_in_transit', 0)), 'failure_reason': br.get('failure_reason', ''),
                                     'steps': int(br.get('steps', 0)), 'telemetry_path': str(telem),
                                     'probe_execution_for_context': 1, 'preprobe_snapshot_captured': 1,
                                     'target_state_kind': baseline})
            env.step = base_step
            # Persist after each context so an infrastructure interruption is
            # recoverable without silently reusing partial output.
            write_rows(task_dir / 'fine_branches.csv', rows)
            write_rows(task_dir / 'restore_parity.csv', parity)
        env.close()
        write_json(task_dir / 'result.json', {'task': task, 'status': 'PASS', 'rows': len(rows), 'parity_rows': len(parity),
                                              'valid_rows': sum(int(x.get('validity', 0)) for x in rows), 'scientific_retries': 0})
    except Exception as exc:
        write_json(task_dir / 'error.json', {'task': task, 'status': 'ENGINEERING_FAILURE', 'error': repr(exc)})
        raise
    finally:
        try: app.close()
        except Exception: pass


def orchestrate(ci):
    records = []
    for task in TASKS:
        env = os.environ.copy()
        env.update({'PYTHONNOUSERSITE': '1', 'PYTHONPATH': os.pathsep.join([str(WARP_CORE), str(ROOT), str(OPENPI)]),
                    'OMNI_KIT_ACCEPT_EULA': 'YES', 'ACCEPT_EULA': 'Y', 'FINE_OUT': str(OUT),
                    'FINE_WORKER': '1', 'P5S0C_TASK_ID': str(task), 'P5S0C_OUT': str(OUT), 'P5S0C_WORKER': '1',
                    'TABERO_ROOT': str(ROOT), 'HDF5_TRAJ_SOURCE_DIR': str(ROOT / 'benchmarks/datasets/libero/assembled_hdf5'),
                    'LIBERO_CONFIG_DIR': str(ROOT / 'benchmarks/datasets/libero/config'),
                    'LIBERO_ASSETS_DATA_DIR': str(ROOT / 'benchmarks/datasets/libero/USD')})
        log = OUT / 'logs' / f'task{task}.log'; log.parent.mkdir(exist_ok=True)
        start = time.time()
        with log.open('w', encoding='utf-8') as f:
            p = subprocess.run([str(ISAAC_PY), '-u', str(Path(__file__).resolve()), '--worker', str(task)], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        records.append({'task': task, 'returncode': p.returncode, 'elapsed_wall_s': time.time() - start, 'log': str(log),
                        'engineering_failure': p.returncode != 0})
        write_json(OUT / 'WORKER_RECORDS.json', records)
        if p.returncode != 0:
            raise RuntimeError(f'ENGINEERING_FAILURE_TASK_{task}')


def collect_and_summarize(ci):
    parts, pars = [], []
    for task in TASKS:
        td = OUT / f'task{task}'
        if (td / 'fine_branches.csv').exists() and (td / 'fine_branches.csv').stat().st_size:
            parts.append(pd.read_csv(td / 'fine_branches.csv'))
        if (td / 'restore_parity.csv').exists() and (td / 'restore_parity.csv').stat().st_size:
            pars.append(pd.read_csv(td / 'restore_parity.csv'))
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    p = pd.concat(pars, ignore_index=True) if pars else pd.DataFrame()
    write_rows(OUT / 'DEV_FINE_FORCE_FRONTIER.csv', d.to_dict('records') if len(d) else [])
    write_rows(OUT / 'RESTORE_PARITY.csv', p.to_dict('records') if len(p) else [])
    summaries = []
    for r in ci.to_dict('records'):
        for baseline in ['probe_restore', 'strict_preprobe']:
            g = d[(d.context_id == r['context_id']) & (d.baseline == baseline)] if len(d) else pd.DataFrame()
            for force, q in g.groupby('force_N') if len(g) else []:
                summaries.append({**r, 'baseline': baseline, 'fine_force_N': force, 'n_valid': int(q.validity.sum()),
                                  'n_repeats': int(len(q)), 'success_frequency': float(q.success.mean()) if len(q) else np.nan,
                                  'unsafe_event_frequency': float(q.unsafe_event.mean()) if len(q) else np.nan,
                                  'bilateral_contact_fraction_mean': float(q.bilateral_contact_fraction.mean()) if len(q) else np.nan,
                                  'repeat_outcome_disagreement': int(q.success.nunique() > 1) if len(q) else 1})
    s = pd.DataFrame(summaries)
    write_rows(OUT / 'DEV_FINE_FORCE_FRONTIER_SUMMARY.csv', s.to_dict('records') if len(s) else [])
    return d, s, p


def finalize(ci, d, s, p):
    # No aggregate fine F* is declared unless the inherited protocol defines a
    # repeated-outcome rule.  The existing protocol only defines minimum
    # successful force for a single deterministic branch.
    pair_rows = []
    for cid, g in ci.groupby('context_id') if False else []:
        pass
    for task, g in ci.groupby('task'):
        pass
    for root, g in ci.groupby('root_id'):
        for _, r in g.iterrows():
            pass
    # Cross-friction same-root/task comparison is not present in DEV because
    # each root_id is one task.  Report same-task DEV roots by band pair.
    info = ci.copy()
    for task, g in info.groupby('task'):
        for a_name, b_name in [('LOW', 'MID'), ('MID', 'HIGH'), ('LOW', 'HIGH')]:
            a = g[g.friction_band.eq(a_name)]; b = g[g.friction_band.eq(b_name)]
            for _, ra in a.iterrows():
                for _, rb in b.iterrows():
                    if abs(ra.friction - rb.friction) > 0.40:  # descriptive band pairing, not used for selection
                        continue
                    pair_rows.append({'task': task, 'context_a': ra.context_id, 'context_b': rb.context_id,
                                      'root_a': ra.root_id, 'root_b': rb.root_id, 'friction_a': ra.friction,
                                      'friction_b': rb.friction, 'coarse_F_star_a': ra.coarse_F_star,
                                      'coarse_F_star_b': rb.coarse_F_star,
                                      'coarse_F_star_diff': int(ra.coarse_F_star != rb.coarse_F_star),
                                      'fine_frontier_status': 'NOT_FORMALLY_IDENTIFIABLE_WITHOUT_REPEATED_RULE',
                                      'fine_F_star_a': '', 'fine_F_star_b': '', 'fine_F_star_diff': '',
                                      'classification': 'D_NOT_IDENTIFIABLE'})
    write_rows(OUT / 'FRICTION_FINE_FRONTIER_PAIRS.csv', pair_rows)
    masking = {'status': 'NOT_IDENTIFIABLE', 'reason': 'Existing P5S0C protocol defines single deterministic minimum successful force; no repeated-outcome aggregation rule was inherited.',
               'continuous_force_controller_supported': True, 'fine_frontier_rows': int(len(d)),
               'coarse_frontier_contexts': int(len(ci)), 'quantization_masking_rate': None,
               'quantization_masking_rate_reason': 'Cannot compare F*_fine differences without a formal repeated frontier rule.',
               'observed_fine_success_curves': True, 'no_test_tuning': True}
    write_json(OUT / 'FORCE_QUANTIZATION_MASKING_ANALYSIS.json', masking)
    # The strict baseline is physically replayed from the captured pre-probe
    # state, but force-selection model inputs are not redefined in this run.
    # Preserve a truthful comparison artifact at the outcome-curve level.
    comp = []
    if len(s):
        for (cid, force), g in s.groupby(['context_id', 'fine_force_N']):
            q = g.set_index('baseline')
            row = {'context_id': cid, 'fine_force_N': force}
            for b in ['probe_restore', 'strict_preprobe']:
                if b in q.index:
                    row[b + '_success_frequency'] = q.loc[b, 'success_frequency']
                    row[b + '_unsafe_event_frequency'] = q.loc[b, 'unsafe_event_frequency']
                    row[b + '_n_repeats'] = q.loc[b, 'n_repeats']
            row['matched_force_curve_comparison_only'] = True
            row['frozen_imagination_action_selection'] = 'NOT_REDEFINED_FOR_LIVE_PREPROBE_TENSOR'
            comp.append(row)
    write_rows(OUT / 'PROBE_VS_STRICT_NOPHYSICS.csv', comp)
    legacy = json.loads((PREV / 'PROBE_INFORMED_IMAGINATION_PROVENANCE.json').read_text())
    write_rows(OUT / 'LEGACY_VS_STRICT_NOPHYSICS.csv', [{'metric': 'legacy_baseline_definition', 'legacy': 'same branch-start physical state, no probe trace, prior [0.30,0.56,0.92]',
                                                         'strict': 'captured pre-probe state, no probe execution in counterfactual branch',
                                                         'comparison_status': 'strict physical branches completed; selected-force comparison not formally redefined'}])
    write_rows(OUT / 'OFFGRID_MODEL_DIAGNOSTIC.csv', [])
    report = f'''# STATUS

STOPPED_AT_EARLIEST_UNSUPPORTED_LINK — DEV fine-force physical replay completed where valid; formal fine-frontier and frozen-stack strict action-selection comparison are not fully identifiable under the inherited protocol.

# SINGLE SCIENTIFIC GOAL

Test whether friction moves the true minimum force inside coarse bins and whether a strict pre-probe no-physics baseline differs from the legacy shared-state baseline.

# AUTHORITATIVE INPUTS / HASHES

Frozen checkpoint, calibration, evaluator, P4-B probe, P5-S0-C branch data, root splits, and controller source hashes are in PROVENANCE.json and DEV_FINE_FORCE_FORENSIC_PROTOCOL.json.

# CONTINUOUS FORCE CONTROLLER VALIDITY

PASS. The existing `_make_action` accepts arbitrary float force, writes half to each finger, and does not integerize or clip force. Prior P5S0C half-step forces were executed. No controller implementation was changed.

# DEV POPULATION

DEV contexts={len(ci)}, tasks={TASKS}, with two DEV roots per task and LOW/MID/HIGH friction. Each fixed coarse bracket was tested at 0.25N spacing with {REPEATS} restore repeats where the worker completed. Valid branch rows={len(d)}; restore records={len(p)}.

# FIFTH TASK STATUS

Task 2 is the intended fifth task (salad dressing), but it is absent from the authoritative P5S0C lineage: the protocol explicitly uses tasks 0/1/5/6, `task2_used=false`, and no valid DEV frontier exists. It was correctly excluded; no artificial task was added.

# COARSE FORCE FRONTIER

Coarse brackets were inherited exactly from P5S0C; no frontier search or candidate-set change was performed.

# FINE FORCE FRONTIER

Empirical success frequency, unsafe-event frequency, bilateral-contact fraction, and repeat disagreement are in DEV_FINE_FORCE_FRONTIER_SUMMARY.csv. The inherited criterion is a single deterministic minimum successful force and does not define a repeated-outcome aggregation rule.

# FRICTION → FINE MINIMUM FORCE

Not formally identifiable as F*_fine differences because no 2/3 or 3/3 rule was authorized. FRICTION_FINE_FRONTIER_PAIRS.csv records the fixed DEV pair coverage without inventing a fine frontier label.

# QUANTIZATION MASKING

QUANTIZATION_MASKING_RATE=NOT_IDENTIFIABLE. Continuous force is controller-valid, but the existing data/protocol do not support a scientifically defined repeated fine frontier; therefore no claim that quantization is the main cause is made.

# FRONTIER STOCHASTICITY

Repeat frequencies and disagreement flags are recorded per force. Non-monotone or mixed repeats are not forced into a monotone F*_fine.

# STRICT PRE-PROBE BASELINE

The replay captured the state immediately after the existing P4-B approach/descend/close/hold sequence and before the probe, then restored that exact snapshot for strict branches. The strict branch itself did not execute a probe. The probe-state branch restored the post-probe snapshot. Restore parity is recorded explicitly.

# LEGACY VS STRICT NO-PHYSICS

The strict physical branch curves are available, but a valid selected-force comparison was not redefined because the frozen imagination stack’s live pre-probe tensor construction was not part of the inherited protocol. No legacy action metric was overwritten.

# PROBE VS STRICT NO-PHYSICS

PROBE_VS_STRICT_NOPHYSICS.csv is a matched physical force-curve comparison, not a new controller-selection method. It does not claim a force-selection improvement.

# OFF-GRID PHYSICS-GRU DIAGNOSTIC

Not run as a formal claim. No off-grid model performance is used to label physical fine frontiers.

# PRIMARY_CLASSIFICATION

INSUFFICIENT_VALID_EVIDENCE

# WHAT IS NOW PROVEN

The existing controller natively supports arbitrary continuous force commands, and DEV same-state fine-force physical branches were executed under the frozen simulator/controller semantics where valid. Strict pre-probe state restoration was implemented as a separately logged counterfactual branch.

# WHAT IS NOT YET PROVEN

The amount of within-bin continuous F* motion, quantization masking rate, and a strict pre-probe selected-action advantage are not proven. No TEST fine replay or fresh E2E was run.

# IMPLICATION FOR FINAL METHOD

Continuous/fine force is technically supported and scientifically worth a dedicated protocol, but this run does not upgrade it to the formal method. A strict pre-probe no-physics baseline should be required for a clean causal comparison.

# METHOD CHANGE

NONE

# NEW TRAINING

NONE. Physics-GRU, friction estimator, isotonic calibration, evaluator, Pi0, and candidate-set definitions were frozen.

# NEXT_METHOD

Define and preregister a repeated-outcome fine-frontier rule on DEV, then rerun only the validated DEV fine-force replay before any method upgrade.
'''
    (OUT / 'FINAL_REPORT.md').write_text(report, encoding='utf-8')
    files = [x for x in OUT.iterdir() if x.name != 'SHA256SUMS.txt']
    (OUT / 'SHA256SUMS.txt').write_text('\n'.join(f'{h(x)}  {x.name}' for x in sorted(files)) + '\n', encoding='utf-8')


def main():
    if '--worker' in sys.argv:
        task = int(sys.argv[sys.argv.index('--worker') + 1])
        worker(task)
        return
    if OUT.exists() and (OUT / 'DEV_FINE_FORCE_FORENSIC_PROTOCOL.json').exists():
        ci = context_info()
    else:
        ci = preflight_and_freeze()
    orchestrate(ci)
    d, s, p = collect_and_summarize(ci)
    finalize(ci, d, s, p)
    print(json.dumps({'out': str(OUT), 'status': 'INSUFFICIENT_VALID_EVIDENCE', 'dev_contexts': len(ci),
                      'rows': len(d), 'parity_rows': len(p)}, indent=2))


if __name__ == '__main__':
    main()
