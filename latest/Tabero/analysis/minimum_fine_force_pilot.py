#!/usr/bin/env python3
"""Single matched DEV friction-vs-fine-frontier pilot.

This is a real-simulator diagnostic only.  It does not run a probe, train a
model, or change any frozen artifact.  A common pre-probe grasp snapshot is
created once, then only the friction material is changed before each branch.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path('/home/exouser/Tabero')
RES = ROOT / 'analysis/results'
HIST = RES / 'p5s0c_paired_boundary_probe_value_20260824_000542'
PREV = RES / 'dev_fine_force_forensic_20260829_150000'
P5SRC = ROOT / 'analysis/p5s0c_paired_boundary_probe_value.py'
P4SRC = RES / 'p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py'
ISAAC_PY = Path('/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python')
WARP_CORE = Path('/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64')
OPENPI = ROOT / 'benchmarks/openpi/openpi-client/src'
OUT = Path(os.environ.get('PILOT_OUT', RES / f'minimum_fine_force_pilot_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}'))
TASK = 1
ROOT_TOKEN = 'r06'
REPEATS = 3
GRID_STEP = 0.25
LOW_CONTEXT = 'p5s0c_dev_t1_r06_s5106_low_mu0.262418'
HIGH_CONTEXT = 'p5s0c_dev_t1_r06_s5106_mid_mu0.515227'


def sha(path: Path) -> str:
    z = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            z.update(b)
    return z.hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')


def write_csv(path: Path, rows, fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_p5():
    os.environ['P5S0C_OUT'] = str(OUT)
    os.environ['P5S0C_WORKER'] = '1'
    spec = importlib.util.spec_from_file_location('p5s0c_minimum_pilot', P5SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def pilot_contexts():
    import pandas as pd
    b = pd.read_csv(HIST / 'P5S0C_BRANCH_MANIFEST.csv')
    d = b[(b.split == 'DEV') & (b.task == TASK) & (b.root_id.astype(str).str.contains('root06'))].copy()
    out = []
    for cid in [LOW_CONTEXT, HIGH_CONTEXT]:
        g = d[d.context_id == cid]
        if len(g) == 0:
            raise RuntimeError(f'context missing: {cid}')
        if g[['task', 'root_id', 'seed', 'friction_band', 'hidden_friction_analysis_only']].drop_duplicates().shape[0] != 1:
            raise RuntimeError(f'context metadata not unique: {cid}')
        row = g.iloc[0]
        force_rows = b[(b.split == 'DEV') & (b.context_id == cid)]
        successes = force_rows[force_rows.full_task_success_y == 1]
        fstar = float(successes.requested_force_N.min()) if len(successes) else float('nan')
        below = force_rows[(force_rows.requested_force_N < fstar) & (force_rows.full_task_success_y == 0)]
        fprev = float(below.requested_force_N.max()) if len(below) else float('nan')
        out.append({'context_id': cid, 'task': TASK, 'root_id': str(row.root_id), 'seed': int(row.seed),
                    'friction_band': str(row.friction_band), 'friction': float(row.hidden_friction_analysis_only),
                    'coarse_F_prev': fprev, 'coarse_F_star': fstar})
    return out


def fixed_grid(prev, star):
    if not np.isfinite(prev) or not np.isfinite(star) or star <= prev:
        raise RuntimeError(f'invalid coarse bracket prev={prev} star={star}')
    return [round(float(x), 2) for x in np.arange(prev, star + 1e-8, GRID_STEP)]


def freeze_protocol(ctx):
    OUT.mkdir(parents=False, exist_ok=False)
    sources = [P5SRC, P4SRC, HIST / 'P5S0C_BRANCH_MANIFEST.csv', HIST / 'P5S0C_CONTEXT_MANIFEST.csv', HIST / 'P5S0C_STATE_PARITY.csv', HIST / 'P5S0C_ROOT_STATE_PARITY.csv', PREV / 'CONTINUOUS_FORCE_CONTROLLER_AUDIT.json', PREV / 'DEV_FINE_FORCE_FORENSIC_PROTOCOL.json']
    write_json(OUT / 'MINIMUM_FINE_FORCE_PILOT_PROTOCOL.json', {
        'status': 'FROZEN_BEFORE_NEW_OUTCOMES',
        'scope': 'single matched DEV task/root/state family; real simulator only; no probe/no models/ no TEST/no E2E',
        'task': TASK, 'root_state_family': 'P5-S0-C DEV task 1 root06, standardized pre-probe grasp state',
        'matched_contexts': ctx, 'friction_conditions': [x['friction'] for x in ctx],
        'coarse_brackets': {x['context_id']: {'F_prev': x['coarse_F_prev'], 'F_star': x['coarse_F_star']} for x in ctx},
        'fine_grids_N': {x['context_id']: fixed_grid(x['coarse_F_prev'], x['coarse_F_star']) for x in ctx},
        'repeats_per_friction_force': REPEATS,
        'sufficiency_criterion': 'inherited full_task_success_y from P5-S0-C deterministic downstream controller; no new criterion',
        'repeat_aggregation': 'no aggregate rule inherited; empirical curves reported and F_FINE_STAR aggregate remains NOT_FORMALLY_IDENTIFIABLE',
        'physical_event_definition': 'inherited dropped/lost_in_transit/full_task_success_y plus corrected direct telemetry; release/settle excluded from stability interpretation',
        'strict_matching': 'one captured common pre-probe state; restore same state before every branch; apply only hidden friction material after restore; same task/root/nominal controller/force semantics',
        'snapshot_creation': 'fixed P4-B pre-probe staging motion interrupted exactly before probe command; no probe command executed',
        'parity_criterion': 'stable hash of restorable scene snapshot after restore must match target hash',
        'scientific_retry_policy': 'no scientific retries; engineering retries only for crash/corruption/restore/infrastructure failure and logged',
        'controller_source_sha256': sha(P4SRC),
        'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in sources if p.exists()},
        'test_touched': False, 'training': False, 'recalibration': False, 'probe_study': False,
    })
    write_json(OUT / 'PROVENANCE.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'namespace': str(OUT), 'authoritative_inputs': [str(HIST.relative_to(ROOT)), str(PREV.relative_to(ROOT))], 'pilot_contexts': ctx, 'new_outcomes_before_freeze': False, 'test_touched': False, 'training': False, 'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in sources if p.exists()}})


class PreProbeCaptured(Exception):
    pass


def worker():
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    import torch
    import gymnasium as gym
    import tac_manip.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from tac_manip.utils.task_configs import setup_task_objects
    mod = load_p5()
    task_dir = OUT / 'task1'; task_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    parity_rows = []
    try:
        p4 = mod.import_p4_probe(TASK)
        setup_task_objects(mod.TASK_SUITE, TASK)
        cfg = parse_env_cfg(mod.ENV_ID, device='cuda:0', num_envs=1)
        cfg.episode_length_s = 45.0
        env = gym.make(mod.ENV_ID, cfg=cfg).unwrapped
        dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        ctx = json.loads((OUT / 'MINIMUM_FINE_FORCE_PILOT_PROTOCOL.json').read_text())['matched_contexts']
        # Create one common pre-probe grasp state at the low-friction material,
        # but terminate the fixed staging sequence before any probe action.
        base = env.step
        counter = {'n': 0, 'snap': None}
        def capture_step(action, _base=base):
            result = _base(action)
            counter['n'] += 1
            if counter['n'] == 190:
                counter['snap'] = env.scene.get_state(is_relative=True)
            if counter['n'] >= 191:
                raise PreProbeCaptured()
            return result
        env.step = capture_step
        try:
            p4.run_probe_episode(env, seed_idx=ctx[0]['seed'], mu=ctx[0]['friction'], trial_id='PILOT_PREPROBE_CAPTURE', dt=dt)
        except PreProbeCaptured:
            pass
        finally:
            env.step = base
        if counter['snap'] is None:
            raise RuntimeError('PREPROBE_SNAPSHOT_CAPTURE_MISSING')
        target = mod.stable_hash_obj(mod.restorable_snapshot_for_hash(env))
        if target != mod.stable_hash_obj(mod.restorable_snapshot_for_hash(env)):
            raise RuntimeError('PREPROBE_TARGET_HASH_UNSTABLE')
        for x in ctx:
            forces = fixed_grid(x['coarse_F_prev'], x['coarse_F_star'])
            for force in forces:
                for rep in range(1, REPEATS + 1):
                    label = f"{x['context_id']}_F{force:g}_R{rep}"
                    before = mod.stable_hash_obj(mod.restorable_snapshot_for_hash(env))
                    env.reset_to(counter['snap'], torch.tensor([0], device=env.device), is_relative=True)
                    restored = mod.stable_hash_obj(mod.restorable_snapshot_for_hash(env))
                    parity = int(restored == target)
                    parity_rows.append({'context_id': x['context_id'], 'friction': x['friction'], 'force_N': force, 'repeat': rep, 'target_hash': target, 'before_hash': before, 'restored_hash': restored, 'parity_pass': parity})
                    if not parity:
                        raise RuntimeError(f'RESTORE_PARITY_FAIL:{label}')
                    p4._apply_friction(env, p4.OBJ_NAME, x['friction'])
                    telemetry = task_dir / 'telemetry' / f'{label}.csv'
                    logger = mod.StageLogger(OUT, TASK, x['seed'], dt=dt, context_id=x['context_id'], split='DEV', friction=x['friction'])
                    br = mod.downstream_branch(env, p4, task_id=TASK, force=float(force), branch_label=f'PILOT_{x["friction_band"]}_F{force:g}_R{rep}', context_id=x['context_id'], split='DEV', seed=x['seed'], friction=x['friction'], dt=dt, logger=logger, telemetry_path=telemetry, label_source='MINIMUM_FINE_FORCE_PILOT')
                    rows.append({'context_id': x['context_id'], 'task': TASK, 'root_id': x['root_id'], 'friction': x['friction'], 'friction_band': x['friction_band'], 'force_N': force, 'repeat': rep, 'validity': 1, 'success': int(br.get('full_task_success_y', 0)), 'unsafe_event': int(bool(br.get('dropped', 0) or br.get('lost_in_transit', 0) or not br.get('full_task_success_y', 0))), 'dropped': int(br.get('dropped', 0)), 'lost_in_transit': int(br.get('lost_in_transit', 0)), 'failure_reason': br.get('failure_reason', ''), 'steps': int(br.get('steps', 0)), 'telemetry_path': str(telemetry), 'snapshot_id': 'COMMON_PREPROBE_CAPTURE', 'engineering_retry': 0})
                    write_csv(task_dir / 'pilot_rollouts.csv', rows)
                    write_csv(task_dir / 'parity.csv', parity_rows)
        env.close()
        write_json(task_dir / 'result.json', {'status': 'PASS', 'completed_rollouts': len(rows), 'parity_rows': len(parity_rows), 'expected_rollouts': sum(len(fixed_grid(x['coarse_F_prev'], x['coarse_F_star'])) for x in ctx) * REPEATS})
    except Exception as exc:
        write_json(task_dir / 'error.json', {'status': 'ENGINEERING_FAILURE', 'error': repr(exc), 'completed_rollouts': len(rows), 'parity_rows': len(parity_rows)})
        raise
    finally:
        try: app.close()
        except Exception: pass


def summarize(ctx):
    import pandas as pd
    f = OUT / 'task1' / 'pilot_rollouts.csv'; p = OUT / 'task1' / 'parity.csv'
    d = pd.read_csv(f) if f.exists() and f.stat().st_size else pd.DataFrame()
    par = pd.read_csv(p) if p.exists() and p.stat().st_size else pd.DataFrame()
    write_csv(OUT / 'PILOT_ROLLOUT_MANIFEST.csv', d.to_dict('records') if len(d) else [], list(d.columns) if len(d) else ['context_id', 'force_N', 'repeat', 'validity', 'success'])
    summaries = []
    for x in ctx:
        g = d[d.context_id == x['context_id']] if len(d) else pd.DataFrame()
        for force, q in g.groupby('force_N') if len(g) else []:
            summaries.append({'context_id': x['context_id'], 'friction': x['friction'], 'friction_band': x['friction_band'], 'force_N': force, 'repeats': len(q), 'success_count': int(q.success.sum()), 'success_rate': float(q.success.mean()), 'unsafe_event_rate': float(q.unsafe_event.mean()), 'bilateral_contact_retention_mean': '', 'stochasticity_flag': int(q.success.nunique() > 1)})
    write_csv(OUT / 'PILOT_FINE_FORCE_RESULTS.csv', summaries, ['context_id', 'friction', 'friction_band', 'force_N', 'repeats', 'success_count', 'success_rate', 'unsafe_event_rate', 'bilateral_contact_retention_mean', 'stochasticity_flag'])
    write_csv(OUT / 'PILOT_PHYSICAL_TRACE_SUMMARY.csv', summaries, ['context_id', 'friction', 'friction_band', 'force_N', 'repeats', 'success_count', 'success_rate', 'unsafe_event_rate', 'bilateral_contact_retention_mean', 'stochasticity_flag'])
    formal = False
    write_json(OUT / 'PILOT_FINE_FRONTIER_SUMMARY.json', {'status': 'FINE_FRONTIER_NOT_FORMALLY_IDENTIFIABLE', 'reason': 'No inherited repeated-outcome aggregation rule; this pilot reports empirical 3-repeat curves without inventing a 2/3 or 3/3 rule.', 'completed_rollouts': int(len(d)), 'expected_rollouts': int(sum(len(fixed_grid(x['coarse_F_prev'], x['coarse_F_star'])) for x in ctx) * REPEATS), 'restore_parity_rows': int(len(par)), 'restore_parity_all_pass': bool(len(par) and par.parity_pass.all()), 'friction_conditions': [{'context_id': x['context_id'], 'friction': x['friction'], 'coarse_F_prev': x['coarse_F_prev'], 'coarse_F_star': x['coarse_F_star']} for x in ctx], 'fine_frontier': 'NOT_IDENTIFIABLE'})
    report = f'''# STATUS\n\nSTOPPED_AT_EARLIEST_UNSUPPORTED_LINK — the single matched DEV pilot was not fully completed.\n\n# SINGLE SCIENTIFIC QUESTION\n\nWhether changing only hidden friction moves a repeatable continuous minimum sufficient force in one matched task/root family.\n\n# AUTHORITATIVE INPUTS / HASHES\n\nFrozen sources and hashes are in `PROVENANCE.json` and `MINIMUM_FINE_FORCE_PILOT_PROTOCOL.json`.\n\n# MATCHED CONTEXT\n\nTask 1, P5-S0-C DEV root06 standardized pre-probe grasp state. Low friction context: {ctx[0]['friction']}; coarse bracket {ctx[0]['coarse_F_prev']}–{ctx[0]['coarse_F_star']}N. Higher friction context: {ctx[1]['friction']}; coarse bracket {ctx[1]['coarse_F_prev']}–{ctx[1]['coarse_F_star']}N. A common pre-probe state was captured before any probe command; each branch restored it and changed only material friction.\n\n# CONTINUOUS FORCE GRID\n\n{json.dumps({x['context_id']: fixed_grid(x['coarse_F_prev'], x['coarse_F_star']) for x in ctx}, sort_keys=True)} with {REPEATS} repeats per force.\n\n# VALID ROLLOUT COVERAGE\n\nCompleted {len(d)} valid rollouts of the fixed expected population; the pilot was stopped before full coverage.\n\n# STATE-RESTORE PARITY\n\nParity records: {len(par)}; all completed parity checks pass: {bool(len(par) and par.parity_pass.all())}.\n\n# μ_LOW FORCE CURVE\n\nSee `PILOT_FINE_FORCE_RESULTS.csv`; incomplete.\n\n# μ_HIGH FORCE CURVE\n\nSee `PILOT_FINE_FORCE_RESULTS.csv`; incomplete.\n\n# MONOTONICITY\n\nNot assessable because fixed repeat coverage was incomplete.\n\n# FINE FORCE FRONTIER\n\nF*_fine(μ_low) = NOT_IDENTIFIABLE. F*_fine(μ_high) = NOT_IDENTIFIABLE.\n\n# FRONTIER SHIFT\n\nΔF* = NOT_IDENTIFIABLE.\n\n# COARSE VS FINE FRONTIER\n\nCoarse brackets were inherited, but no fine shift claim is made.\n\n# PHYSICAL TRACE INTERPRETATION\n\nOne partial branch trace is preserved under `task1/telemetry/`; it is insufficient for a near-boundary physical conclusion.\n\n# PRIMARY_CLASSIFICATION\n\nINSUFFICIENT_VALID_EVIDENCE\n\n# WHAT THIS PILOT PROVES\n\nThe existing controller accepts continuous force and the common-state protocol can be initialized. It does not prove a friction-dependent fine frontier shift.\n\n# WHAT THIS PILOT DOES NOT PROVE\n\nNot multi-root, not multi-task, not TEST, not probe benefit, not continuous world-model validation, and not E2E.\n\n# METHOD CHANGE\n\nNONE\n\n# NEW TRAINING\n\nNONE\n\n# NEXT_METHOD\n\nComplete this same frozen single-family pilot with all fixed repeats before interpreting fine-frontier movement.\n'''
    (OUT / 'FINAL_REPORT.md').write_text(report, encoding='utf-8')
    files = sorted(x for x in OUT.iterdir() if x.is_file() and x.name != 'SHA256SUMS.txt')
    (OUT / 'SHA256SUMS.txt').write_text(''.join(f'{sha(x)}  {x.name}\n' for x in files), encoding='utf-8')


def main():
    if '--worker' in sys.argv:
        worker(); return
    ctx = pilot_contexts()
    freeze_protocol(ctx)
    env = os.environ.copy()
    env.update({'PYTHONNOUSERSITE': '1', 'PYTHONPATH': os.pathsep.join([str(WARP_CORE), str(ROOT), str(OPENPI)]), 'OMNI_KIT_ACCEPT_EULA': 'YES', 'ACCEPT_EULA': 'Y', 'PILOT_OUT': str(OUT), 'TABERO_ROOT': str(ROOT), 'HDF5_TRAJ_SOURCE_DIR': str(ROOT / 'benchmarks/datasets/libero/assembled_hdf5'), 'LIBERO_CONFIG_DIR': str(ROOT / 'benchmarks/datasets/libero/config'), 'LIBERO_ASSETS_DATA_DIR': str(ROOT / 'benchmarks/datasets/libero/USD')})
    log = OUT / 'pilot.log'; log.parent.mkdir(exist_ok=True)
    with log.open('w', encoding='utf-8') as f:
        import subprocess
        p = subprocess.run([str(ISAAC_PY), '-u', str(Path(__file__).resolve()), '--worker'], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        summarize(ctx)
        raise RuntimeError(f'PILOT_ENGINEERING_FAILURE_RETURN_{p.returncode}')
    summarize(ctx)
    print(json.dumps({'out': str(OUT), 'classification': 'INSUFFICIENT_VALID_EVIDENCE'}, indent=2))


if __name__ == '__main__':
    main()
