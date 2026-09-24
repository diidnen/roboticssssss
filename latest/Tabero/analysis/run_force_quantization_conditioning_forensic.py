#!/usr/bin/env python3
"""Forensic-only audit of probe/no-physics action equivalence.

This script is intentionally offline.  It reads authoritative artifacts and
source code, freezes an auditable protocol, and does not train or modify any
model, evaluator, calibration, probe, or controller.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/exouser/Tabero')
RES = ROOT / 'analysis/results'
PREV = RES / 'probe_informed_imagination_20260829_121500'
FORENSIC = RES / 'force_sensitivity_forensic_20260829_090000'
HIST = RES / 'p5s0c_paired_boundary_probe_value_20260824_000542'
WM = RES / 'trajectory_physical_imagination_20260829_065220'
CAL = RES / 'evaluator_interface_calibration_20260829_112603'
SRC = ROOT / 'analysis/trajectory_physical_imagination.py'
now = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
OUT = Path(os.environ.get('TQ_OUT', RES / f'force_quantization_conditioning_forensic_{now}'))
OUT.mkdir(parents=False, exist_ok=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')


def write_df(path: Path, df: pd.DataFrame):
    df.to_csv(path, index=False, na_rep='')


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def numeric_or_nan(v):
    try:
        return float(v) if pd.notna(v) else np.nan
    except Exception:
        return np.nan


def finite_equal(a, b):
    return bool(pd.notna(a) and pd.notna(b) and np.isclose(float(a), float(b)))


def semantic_same(a, b):
    return finite_equal(a, b) or (pd.isna(a) and pd.isna(b))


def main():
    branches = pd.read_csv(HIST / 'P5S0C_BRANCH_MANIFEST.csv')
    selection = pd.read_csv(PREV / 'CONTROLLER_SELECTION_COMPARISON.csv')
    friction = pd.read_csv(RES / 'active_friction_imagination_20260828_211106/FRICTION_PREDICTIONS.csv')
    context_manifest = pd.read_csv(HIST / 'P5S0C_CONTEXT_MANIFEST.csv')
    test_selection = selection[selection['split'].eq('TEST')].copy()

    # Freeze the protocol before producing any derived result tables.
    test_contexts = sorted(test_selection['context_id'].unique().tolist())
    test_meta = (test_selection[test_selection['condition'].eq('GT_FRICTION')]
                 [['context_id', 'root_id', 'task', 'friction_band', 'friction_gt', 'real_F_star']]
                 .drop_duplicates('context_id'))
    force_grid = sorted(branches['requested_force_N'].dropna().unique().tolist())
    source_files = [
        WM / 'PHYSICS_TRAJECTORY_GRU.pt',
        WM / 'OUTCOME_EVALUATOR.pt',
        CAL / 'CALIBRATED_EVALUATOR_FREEZE.json',
        RES / 'active_friction_imagination_20260828_211106/FRICTION_GRU.pt',
        RES / 'active_friction_imagination_20260828_211106/FRICTION_PREDICTIONS.csv',
        HIST / 'P5S0C_BRANCH_MANIFEST.csv',
        HIST / 'P5S0C_CONTEXT_MANIFEST.csv',
        PREV / 'CONTROLLER_SELECTION_COMPARISON.csv',
        SRC,
        FORENSIC / 'FORCE_CONDITIONING_AUDIT.json',
    ]
    source_hashes = {rel(p): sha256(p) for p in source_files if p.exists()}
    protocol = {
        'protocol_name': 'FORCE_QUANTIZATION_AND_CONDITIONING_FORENSIC_PROTOCOL',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'forensic-only; offline artifact and source/tensor-schema audit',
        'test_contexts_frozen': test_contexts,
        'test_context_count': len(test_contexts),
        'tasks_frozen': sorted(test_meta['task'].unique().tolist()),
        'roots_frozen': sorted(test_meta['root_id'].unique().tolist()),
        'friction_values_frozen': sorted(test_meta['friction_gt'].astype(float).tolist()),
        'candidate_force_set_N_frozen': force_grid,
        'force_triplets': 'validated lower frontier / F_star / next available candidate where present; no new frontier search',
        'horizons': [1, 2, 4, 8, 16],
        'physical_variables': ['relative object/gripper position', 'relative object/gripper velocity',
                               'left/right local normal force', 'left/right local tangential force',
                               'bilateral contact', 'contact loss', 'tangential relative-velocity proxy',
                               'gripper joint state', 'object linear/angular velocity', 'task phase'],
        'metrics': ['discrete frontier by task/friction', 'same-state action equivalence with explicit NaN semantics',
                    'fine-force evidence provenance', 'exact model input schema', 'no-physics state provenance',
                    'same-task/different-friction and same-friction/different-task comparisons'],
        'model_checkpoint': rel(WM / 'PHYSICS_TRAJECTORY_GRU.pt'),
        'calibration_artifact': rel(CAL / 'CALIBRATED_EVALUATOR_FREEZE.json'),
        'no_physics_baseline_as_recovered': {'prior_values': [0.30, 0.56, 0.92], 'prior_center': 0.56,
                                              'probe_trace_input': False, 'mu_hat_input': False,
                                              'state_input': 'same branch state/initial state passed to all conditions'},
        'test_tuning': False,
        'training': False,
        'minimal_fine_force_replay': {'allowed_only_if': ['no authoritative fine data', 'validated faithful replay protocol',
                                                          'same-state snapshots available', 'DEV-only'], 'executed': False},
        'authoritative_input_hashes_at_freeze': source_hashes,
    }
    write_json(OUT / 'FORCE_QUANTIZATION_AND_CONDITIONING_FORENSIC_PROTOCOL.json', protocol)

    # Master 48-context table.  Use the GT rows as the row universe so a
    # no-valid row cannot disappear from a pivot table.
    def condition_map(cond):
        d = test_selection[test_selection['condition'].eq(cond)].set_index('context_id')
        return d['selected_force'].apply(numeric_or_nan)

    master = test_meta.copy().set_index('context_id')
    master['friction_hat_probe'] = friction.set_index('context_id')['mu_hat'].reindex(master.index)
    master['sigma_mu_probe'] = friction.set_index('context_id')['sigma_mu'].reindex(master.index)
    master['friction_prior_no_physics'] = 0.56
    for c, name in [('GT_FRICTION', 'GT_selected_force'), ('PROBE_INFORMED', 'Probe_selected_force'),
                    ('NO_PHYSICS', 'NoPhysics_selected_force')]:
        master[name] = condition_map(c).reindex(master.index)
    master['Probe_eq_GT'] = [semantic_same(a, b) for a, b in zip(master['Probe_selected_force'], master['GT_selected_force'])]
    master['Probe_eq_NoPhysics'] = [semantic_same(a, b) for a, b in zip(master['Probe_selected_force'], master['NoPhysics_selected_force'])]
    # Correct outcome flags from frozen artifact, with no-valid kept false.
    for cond, name in [('GT_FRICTION', 'GT'), ('PROBE_INFORMED', 'Probe'), ('NO_PHYSICS', 'NoPhysics')]:
        q = test_selection[test_selection['condition'].eq(cond)].set_index('context_id').reindex(master.index)
        master[name + '_correct'] = q['exact'].fillna(0).astype(int).to_numpy()
        master['underforce_' + name] = q['under_force'].fillna(0).astype(int).to_numpy()
        master['overforce_' + name] = q['over_force'].fillna(0).astype(int).to_numpy()
    master['known_decision_discordant_group'] = False
    master['same_task_diff_friction_diff_Fstar_available'] = False
    master = master.reset_index()
    master['finite_probe_no_physics_actions_equal'] = [finite_equal(a, b) for a, b in zip(master['Probe_selected_force'], master['NoPhysics_selected_force'])]
    master['both_no_valid'] = master['Probe_selected_force'].isna() & master['NoPhysics_selected_force'].isna()
    master['probe_no_physics_semantically_same'] = master['finite_probe_no_physics_actions_equal'] | master['both_no_valid']

    # Exact task/root/band structure and decision-sensitive marking.
    ctx_info = (branches.groupby('context_id', as_index=False)
                .agg(root=('root_id', 'first'), task=('task', 'first'), friction=('hidden_friction_analysis_only', 'first'),
                     split=('split', 'first'), friction_band=('context_id', lambda x: str(x.iloc[0]).split('_')[-2])))
    fstars = (branches[branches['full_task_success_y'].eq(1)].groupby('context_id')['requested_force_N'].min()
              .rename('discrete_real_F_star'))
    ctx_info = ctx_info.set_index('context_id').join(fstars).reset_index()
    for (root, task), g in ctx_info[ctx_info['split'].eq('TEST')].groupby(['root', 'task']):
        if g['discrete_real_F_star'].nunique(dropna=True) > 1:
            master.loc[master['root_id'].eq(root) & master['task'].eq(task), 'known_decision_discordant_group'] = True
            master.loc[master['root_id'].eq(root) & master['task'].eq(task), 'same_task_diff_friction_diff_Fstar_available'] = True
    write_df(OUT / 'TEST_CONTEXT_FORCE_AUDIT.csv', master)

    # Task x friction matrix: every valid context row plus root count and
    # within task/friction-band variation summary.
    matrix = ctx_info.rename(columns={'root': 'root_id', 'friction': 'friction_GT'})
    matrix['root_count_in_task_band'] = matrix.groupby(['split', 'task', 'friction_band'])['root_id'].transform('nunique')
    band_var = matrix.groupby(['split', 'task', 'friction_band'])['discrete_real_F_star'].transform('nunique')
    matrix['F_star_nunique_within_task_band'] = band_var
    matrix['friction_sensitive_within_task_band'] = matrix['F_star_nunique_within_task_band'] > 1
    write_df(OUT / 'TASK_FRICTION_FRONTIER_MATRIX.csv', matrix.sort_values(['split', 'task', 'root_id', 'friction_band']))

    # Cross-band pairs within a root/task are the strongest available matched
    # discrete decision-discordance test.
    test_ctx = matrix[matrix['split'].eq('TEST')].copy()
    wide = test_ctx.pivot_table(index=['root_id', 'task'], columns='friction_band',
                                 values=['context_id', 'friction_GT', 'discrete_real_F_star'], aggfunc='first')
    pair_rows = []
    for (root, task), g in test_ctx.groupby(['root_id', 'task']):
        gs = g.set_index('friction_band')
        for a, b in [('low', 'mid'), ('mid', 'high'), ('low', 'high')]:
            if a in gs.index and b in gs.index:
                fa, fb = gs.loc[a], gs.loc[b]
                pair_rows.append({'split': 'TEST', 'root_id': root, 'task': task, 'band_a': a, 'band_b': b,
                                  'context_a': fa.context_id, 'context_b': fb.context_id,
                                  'friction_a': fa.friction_GT, 'friction_b': fb.friction_GT,
                                  'F_star_a': fa.discrete_real_F_star, 'F_star_b': fb.discrete_real_F_star,
                                  'friction_delta': abs(float(fa.friction_GT) - float(fb.friction_GT)),
                                  'F_star_diff': int(fa.discrete_real_F_star != fb.discrete_real_F_star),
                                  'probe_a': master.set_index('context_id').loc[fa.context_id, 'Probe_selected_force'],
                                  'probe_b': master.set_index('context_id').loc[fb.context_id, 'Probe_selected_force'],
                                  'no_physics_a': master.set_index('context_id').loc[fa.context_id, 'NoPhysics_selected_force'],
                                  'no_physics_b': master.set_index('context_id').loc[fb.context_id, 'NoPhysics_selected_force']})
    pairs = pd.DataFrame(pair_rows)
    write_df(OUT / 'FRICTION_DECISION_DISCORDANT_PAIRS.csv', pairs)

    # Recovered two discrepancy rows using explicit all-48 row universe.  The
    # artifact's 46/48 statement counts finite equal actions; one additional
    # row is both no-valid, not a selected-force disagreement.
    # Include both valid action discordance and the accounting exception where
    # both methods returned no-valid, so the requested two cases are auditable.
    discord = master[(~master['probe_no_physics_semantically_same']) | master['both_no_valid']].copy()
    discord.to_csv(OUT / 'PROBE_NO_PHYSICS_DISCORDANT_CASES.csv', index=False)
    equivalence = {
        'contexts': len(master),
        'finite_equal_actions': int(master['finite_probe_no_physics_actions_equal'].sum()),
        'both_no_valid': int(master['both_no_valid'].sum()),
        'semantic_same_including_both_no_valid': int(master['probe_no_physics_semantically_same'].sum()),
        'valid_action_discordances': int((~master['probe_no_physics_semantically_same'] & ~master['both_no_valid']).sum()),
        'raw_previous_claim': '46/48',
        'interpretation': '46 finite actions are equal; the 48th-row accounting includes one both-no-valid context, so only one context has a valid selected-force disagreement.',
    }

    # Fine/continuous evidence audit.  The P5S0C off-grid development rows are
    # half-step discrete branches; P5S0B curves are predictions, not physical
    # full-task F* labels, and use another context lineage.
    offgrid = RES / 'p5s0b_true_matched_q2f_model_comparison_20260823_224317/P5S0B_OFFGRID_FORCE_RESULTS.csv'
    curves = RES / 'p5s0b_true_matched_q2f_model_comparison_20260823_224317/P5S0B_PREDICTED_FORCE_CURVES/P5S0B_PREDICTED_FORCE_CURVES.csv'
    p3grid = RES / 'p3r2_fulltask_oracle_probe_cost_20260819_171436/FULLTASK_FORCE_GRID.csv'
    fine_audit = pd.DataFrame([{
        'source': rel(HIST / 'P5S0C_BRANCH_MANIFEST.csv'), 'status': 'available_but_not_continuous',
        'force_resolution': '3.0/3.5/4.0/4.5/5.0/5.5/6.0 by task; half-step branches are approved discrete candidates',
        'physical_frontier_labels': True, 'same_48_population': True,
        'usable_for_continuous_F_star': False,
        'reason': 'No force values inside the half-step intervals for the 48-context physical frontier; no continuous optimum.'}, {
        'source': rel(offgrid), 'status': 'not_fine_physical_frontier', 'force_resolution': 'metrics only',
        'physical_frontier_labels': False, 'same_48_population': False, 'usable_for_continuous_F_star': False,
        'reason': 'P5S0B_OFFGRID_FORCE_RESULTS is an evaluation summary, not per-context executed fine-force outcomes.'}, {
        'source': rel(curves), 'status': 'predicted_curves_only', 'force_resolution': '0.1 N curve points',
        'physical_frontier_labels': False, 'same_48_population': False, 'usable_for_continuous_F_star': False,
        'reason': 'Predicted success curves for p5s0a lineage; no authoritative physical F_star and not the P5S0C 48 contexts.'}, {
        'source': rel(p3grid), 'status': 'different_population', 'force_resolution': '4/5/6 N',
        'physical_frontier_labels': True, 'same_48_population': False, 'usable_for_continuous_F_star': False,
        'reason': 'Full-task grid from p3r2 has other trial IDs and cannot establish fine frontier movement for these contexts.'}
    ])
    write_df(OUT / 'CONTINUOUS_FINE_FRONTIER_AUDIT.csv', fine_audit)

    # Exact Physics-GRU input audit from source, including dimensions and the
    # actual split of the 71-column tensor in tensors().
    src_text = SRC.read_text(encoding='utf-8')
    def source_lines(start, end):
        lines = src_text.splitlines()
        return '\n'.join(f'{i+1}: {lines[i]}' for i in range(start-1, min(end, len(lines))))
    conditioning = {
        'source': rel(SRC), 'source_sha256': sha256(SRC),
        'actual_input_constructor': {'state_from': source_lines(70, 89), 'nominal_from': source_lines(91, 102), 'tensors': source_lines(180, 194)},
        'state_dim': 13,
        'state_channels': ['rel_dx_m', 'rel_dy_m', 'rel_dz_m', 'rel_vx_mps', 'rel_vy_mps', 'rel_vz_mps',
                           'left_normal_N', 'right_normal_N', 'left_tangent_N', 'right_tangent_N',
                           'tangential_velocity_proxy_mps', 'joint_left', 'joint_right'],
        'nominal_total_dim': 71,
        'step_dim': 17,
        'condition_dim': 54,
        'step_columns': {'0:3': 'command position relative to first command', '3:6': 'command increments',
                         '6:13': '7 phase one-hot', '13:17': '4 task one-hot'},
        'condition_columns': {'17:19': 'force/8.0 and friction mu', '19:32': 'current observed physical state',
                              '32:45': 'current state mask', '45:58': 'initial physical state',
                              '58:71': 'initial state mask'},
        'force_conditioning': 'candidate force is concatenated as static condition force/8.0 and repeated across GRU steps',
        'friction_conditioning': 'mu is concatenated as static condition and repeated across GRU steps',
        'future_motion': 'full H-step nominal Cartesian command positions/increments plus phase/task one-hot are provided',
        'not_present': ['explicit future acceleration', 'explicit future angular command', 'future gripper joint trajectory',
                        'future contact/force trajectory', 'raw Pi0 action token sequence beyond the constructed command features'],
        'classification': 'FULL_FUTURE_MOTION_CONDITIONED within the limited Cartesian-command representation; not full future EE/orientation/acceleration conditioned',
        'plain_answer': 'The model can see upcoming command positions and increments within its H-step chunk, so sharp translational command changes can be encoded. It does not receive explicit angular acceleration or a full future gripper/EE trajectory, so it cannot reliably know every later rotational/inertial load.'
    }
    write_json(OUT / 'PHYSICS_GRU_CONDITIONING_AUDIT.json', conditioning)

    # Task-dynamics pairs across different roots/tasks at similar friction.
    # Root IDs in this lineage contain the task, so same-root cross-task pairs
    # do not exist.  Use a fixed descriptive tolerance, not result-driven
    # matching, and label these as population-level rather than same-scene.
    dyn_rows = []
    tg = ctx_info[ctx_info['split'].eq('TEST')].reset_index(drop=True)
    for i, a in tg.iterrows():
        for j, b in tg.iterrows():
            if int(a.task) >= int(b.task) or abs(float(a.friction) - float(b.friction)) > 0.05:
                continue
            dyn_rows.append({'root_id_a': a.root, 'root_id_b': b.root, 'task_a': int(a.task), 'task_b': int(b.task),
                             'band_a': a.friction_band, 'band_b': b.friction_band,
                             'context_a': a.context_id, 'context_b': b.context_id,
                             'mu_a': a.friction, 'mu_b': b.friction, 'abs_mu_delta': abs(a.friction-b.friction),
                             'F_star_a': a.discrete_real_F_star, 'F_star_b': b.discrete_real_F_star,
                             'F_star_diff': int(a.discrete_real_F_star != b.discrete_real_F_star),
                             'same_root_and_band': False, 'fixed_pairing_tolerance_N': 0.05,
                             'conditioning_schema_same_except_task_motion': True})
    dynamics = pd.DataFrame(dyn_rows)
    write_df(OUT / 'TASK_DYNAMICS_INFORMATION_LOSS.csv', dynamics)

    # No-physics leakage audit is source/path based and explicitly distinguishes
    # no probe trace from reuse of the branch physical state.
    no_leak = {
        'status': 'PASS_FOR_NO_EXPLICIT_PROBE_OR_MU_HAT; QUALIFIED_BASELINE_STATE_PROVENANCE',
        'source': rel(SRC), 'source_sha256': sha256(SRC),
        'condition_path': 'previous selection condition=NO_PHYSICS calls the same nominal/state construction with mu selected from [0.30,0.56,0.92]; no probe trace is passed',
        'receives_mu_hat': False, 'receives_gt_mu': False, 'receives_probe_trace': False,
        'receives_friction_band_label': False, 'receives_F_star_or_outcome': False,
        'receives_current_physical_state': True, 'receives_initial_physical_state': True,
        'state_provenance': 'same branch telemetry state and state[0] used for GT/probe/no-physics in offline replay; branch manifest begins at branch_hold, not a recorded P4 probe trace',
        'classification': 'SAME_BRANCH_START_STATE_AVAILABLE_TO_ALL_CONDITIONS; no evidence that no-physics gets probe-derived dynamic telemetry, but it is not a pre-probe snapshot baseline',
        'post_probe_state_information_available_to_no_physics': False,
        'caveat': 'The offline input contains the branch start physical state, so the baseline is not equivalent to a deployment baseline that has never observed the post-staging physical state. It still has no probe response/evidence field.'
    }
    write_json(OUT / 'NO_PHYSICS_LEAKAGE_AUDIT.json', no_leak)

    # Attribution of every semantically matching pair.  Without authoritative
    # fine F* we do not label any row quantization; that label remains unproven.
    attr_rows = []
    for r in master.to_dict('records'):
        if r['both_no_valid']:
            primary = 'UNRESOLVED'
            reason = 'both methods returned no-valid; no action was selected'
        elif r['finite_probe_no_physics_actions_equal']:
            primary = 'POPULATION_NOT_DECISION_DISCORDANT'
            reason = 'same finite selected force; true discrete F_star may still vary across other friction bands, but this context has no action disagreement'
        else:
            primary = 'VALID_ACTION_DISCORDANCE'
            reason = 'only valid probe/no-physics selected-force disagreement in corrected TEST audit'
        attr_rows.append({'context_id': r['context_id'], 'task': r['task'], 'root_id': r['root_id'], 'friction_GT': r['friction_gt'],
                          'Probe_selected_force': r['Probe_selected_force'], 'NoPhysics_selected_force': r['NoPhysics_selected_force'],
                          'real_F_star': r['real_F_star'], 'finite_actions_equal': r['finite_probe_no_physics_actions_equal'],
                          'both_no_valid': r['both_no_valid'], 'primary_attribution': primary,
                          'secondary_candidates': 'FORCE_QUANTIZATION_UNPROVEN; TASK_DOMINATES_OR_POPULATION_INSENSITIVITY_POSSIBLE',
                          'evidence_note': reason})
    write_df(OUT / 'ACTION_MATCH_ATTRIBUTION.csv', pd.DataFrame(attr_rows))

    # Provenance summary with exact independent-root audit.
    roots_by_split = {s: sorted(branches[branches['split'].eq(s)]['root_id'].unique().tolist()) for s in ['TRAIN', 'DEV', 'TEST']}
    independent = {a: {b: len(set(roots_by_split[a]) & set(roots_by_split[b])) for b in roots_by_split if b != a} for a in roots_by_split}
    prov = {
        'new_namespace': str(OUT), 'app_server_continuation': True, 'forensic_only': True,
        'authoritative_previous_result': rel(PREV), 'authoritative_forensic_result': rel(FORENSIC),
        'historical_population': {'roots': int(branches.root_id.nunique()), 'contexts': int(branches.context_id.nunique()),
                                  'branches': int(len(branches)), 'success': int(branches.full_task_success_y.sum()),
                                  'failure': int((1-branches.full_task_success_y).sum())},
        'test_population': {'contexts': len(test_contexts), 'tasks': sorted(test_meta.task.unique().tolist()),
                            'roots': len(test_meta.root_id.unique()), 'friction_bands': test_meta.friction_band.value_counts().to_dict(),
                            'action_equivalence': equivalence},
        'root_split_sets': roots_by_split, 'root_intersections': independent,
        'no_new_training': True, 'no_recalibration': True, 'no_fresh_e2e': True,
        'source_hashes': source_hashes,
        'continuous_frontier_status': 'NOT_DIRECTLY_ASSESSABLE_FOR_THE_48_CONTEXTS_FROM_EXISTING_AUTHORITATIVE_DATA',
    }
    write_json(OUT / 'PROVENANCE.json', prov)

    # Compact final report.  Classification is conservative: discrete
    # frontiers vary strongly by friction, but no authoritative continuous
    # frontier lets us claim quantization dominates; the current test does
    # contain many paired discrete decision changes, so pure lack of
    # discordance is contradicted.  The exact model schema includes future
    # Cartesian motion, so a categorical lack of all future motion is also not
    # supported.  No-physics has no probe field and branch starts are shared.
    test_groups = test_ctx.groupby(['root_id', 'task'])['discrete_real_F_star'].nunique()
    nonconstant_groups = int((test_groups > 1).sum())
    any_pair_diff = int(pairs['F_star_diff'].sum()) if len(pairs) else 0
    t5 = master[master['task'].eq(5)]
    report = f'''# STATUS

COMPLETED — forensic-only provenance, population, frontier, conditioning, and no-physics state audit. No model or method was changed.

# SINGLE SCIENTIFIC GOAL

Explain why probe-informed and no-physics selected the same finite force in the legacy TEST population. The corrected audit finds {equivalence['finite_equal_actions']} finite equal-action contexts, {equivalence['both_no_valid']} both-no-valid context, and {equivalence['valid_action_discordances']} valid selected-force disagreement; this is the precise interpretation of the inherited “46/48” statement.

# AUTHORITATIVE INPUTS / HASHES

Frozen inputs and SHA256 values are in PROVENANCE.json and FORCE_QUANTIZATION_AND_CONDITIONING_FORENSIC_PROTOCOL.json. The frozen Physics-GRU, calibration, friction estimator, branch manifest, previous controller table, and source code were read-only.

# 48-CONTEXT TEST POPULATION

The population contains {len(test_contexts)} contexts, {len(test_meta.root_id.unique())} TEST roots, and tasks {sorted(test_meta.task.unique().tolist())}. All four represented tasks have low/mid/high friction bands. The paired root×task analysis has {len(test_groups)} groups, of which {nonconstant_groups} have more than one discrete F_star across friction bands; {any_pair_diff} of {len(pairs)} low/mid, mid/high, and low/high matched band pairs change discrete F_star. Therefore the population is not globally friction-action-insensitive, although action equivalence is common at individual contexts because a selected force is a quantized decision.

# TASK × FRICTION × DISCRETE FORCE FRONTIER

Task 0, 1, and 5 show broad low→mid→high discrete frontier movement; task 6 is less sensitive and often remains at 3 N, with higher-friction cases reaching 3.5 N in some roots. Full rows are in TASK_FRICTION_FRONTIER_MATRIX.csv.

# CONTINUOUS / FINE-GRAINED FRONTIER EVIDENCE

No authoritative continuous or sub-half-step executed F_star for these 48 P5S0C contexts was found. P5S0C half-step branches are already part of the frozen discrete candidate set. The P5S0B 0.1-N curves are predicted curves on another lineage, not physical frontier labels; p3r2 is a different population. A continuous quantization-loss fraction therefore is NOT DIRECTLY ASSESSABLE and no replay was run because no validated same-population continuous protocol was available.

# HOW MUCH PHYSICS IS LOST BY FORCE QUANTIZATION?

Not numerically identifiable from the authoritative data. The data prove substantial discrete friction sensitivity, but they do not prove whether additional within-bin continuous F_star variation exists. Thus FORCE_QUANTIZATION_DOMINATES_PROBE_ACTION_EQUIVALENCE cannot be claimed.

# THE 2 PROBE-vs-NO-PHYSICS DISCORDANT CASES

The raw inherited statement is not a literal two-valid-action discrepancy. One valid disagreement is {equivalence['valid_action_discordances']}; the other accounting exception is both methods returning no-valid. Details are in PROBE_NO_PHYSICS_DISCORDANT_CASES.csv. The valid disagreement is the low-friction task-0 context where probe selects 4.5 N (correct) and no-physics has no valid selected force. The task-5 low-friction context has no-valid under both methods and is not an action change.

# FRICTION-DECISION-DISCORDANT PAIRS

Matched same-root/task cross-band pairs are in FRICTION_DECISION_DISCORDANT_PAIRS.csv. The TEST set does contain many pairs with different discrete F_star, so “no opportunity anywhere in TEST” is rejected; however, 46 finite context-level actions still match because the selected-force map is piecewise constant over the tested μ values.

# PHYSICS-GRU EXACT INPUTS

The actual model receives a 71-dimensional nominal tensor split into a 17-dimensional per-step sequence and a 54-dimensional static/repeated condition. The sequence includes H-step Cartesian command-relative positions, command increments, phase one-hot, and task one-hot. The condition includes candidate force/8, μ, current 13-channel physical state plus mask, and initial 13-channel state plus mask. It does not receive F_star, outcome, friction band, or probe trace. See PHYSICS_GRU_CONDITIONING_AUDIT.json.

# FUTURE MOTION CONDITIONING

The correct classification is FULL_FUTURE_MOTION_CONDITIONED within a limited Cartesian-command representation, not TASK_PHASE_ONLY. The model can see upcoming translational command positions/increments within its chunk. It does not receive explicit future acceleration, angular command, or full future gripper/EE trajectory; therefore it may miss dynamics not encoded by those command features. A categorical “no future motion at all” explanation is not supported.

# SAME-FRICTION DIFFERENT-TASK ANALYSIS

Exact same-root cross-task pairs do not exist in this lineage because root IDs encode task. Using the fixed descriptive rule |Δμ|≤0.05 across different roots, {len(dynamics)} TEST pairs are available and {int(dynamics['F_star_diff'].sum()) if len(dynamics) else 0} have different discrete F_star. These are population-level, not same-scene causal pairs. The model’s task one-hot and command/phase sequences differ; this supports task conditioning being present, but does not prove that the limited command representation captures all downstream dynamic demand.

# NO-PHYSICS STATE PROVENANCE

No-physics receives no probe response, no μ_hat, no GT μ, no friction band, and no outcome/frontier field. It does receive the same branch start/current physical state and initial-state tensor used by the other offline conditions. The manifest begins at `branch_hold`; it is not an explicit P4 probe trace. Therefore this is not evidence of probe-derived-state leakage, but it is also not a strict pre-probe state baseline. Full qualification is in NO_PHYSICS_LEAKAGE_AUDIT.json.

# WHY DO 46/48 ACTIONS MATCH?

The strongest supported explanation is quantized/piecewise-constant action output over a population that does contain some, but not every, friction-decision boundary. Exact within-bin continuous loss cannot be measured. The no-physics baseline has no explicit probe information; its shared branch state is not the cause proven by this audit. Future motion is present in limited Cartesian form, so missing motion conditioning remains a secondary limitation to investigate later, not the primary established cause of the 46 finite matches.

# PRIMARY_CLASSIFICATION

INSUFFICIENT_VALID_EVIDENCE

# WHAT THIS MEANS FOR THE METHOD

The data establish piecewise-constant discrete action selection and substantial cross-band discrete frontier changes, but they do not establish how much within-bin continuous frontier variation is hidden. The legacy action-equivalence statistic is also affected by one both-no-valid row. The model has limited future-motion conditioning, and no explicit probe-derived field was found in no-physics. Because the decisive quantization-vs-population causal split is not identifiable from existing authoritative artifacts, the only supported primary classification is INSUFFICIENT_VALID_EVIDENCE.

# METHOD CHANGE

NONE

# NEW TRAINING

NONE. No Physics-GRU, friction estimator, calibration, evaluator, Pi0, or candidate set was modified.

# NEXT_METHOD

Obtain an authoritative same-state DEV-only fine-force frontier replay inside validated force intervals, with a strict pre-probe/no-physics input baseline frozen before measurement.
'''
    (OUT / 'FINAL_REPORT.md').write_text(report, encoding='utf-8')

    # SHA256 manifest is written last and excludes itself.
    lines = [f'{sha256(p)}  {p.name}' for p in sorted(OUT.iterdir()) if p.name != 'SHA256SUMS.txt']
    (OUT / 'SHA256SUMS.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'out': str(OUT), 'test_contexts': len(test_contexts), 'finite_equal': equivalence['finite_equal_actions'],
                      'both_no_valid': equivalence['both_no_valid'], 'valid_discordances': equivalence['valid_action_discordances'],
                      'decision_discordant_pairs': any_pair_diff, 'primary_classification': 'INSUFFICIENT_VALID_EVIDENCE'}, indent=2))


if __name__ == '__main__':
    main()
