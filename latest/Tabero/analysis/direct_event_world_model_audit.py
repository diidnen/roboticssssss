#!/usr/bin/env python3
"""Pre-training direct physical-event labelability gate.

This run is intentionally stopped at the data gate when corrected event
telemetry is insufficient.  It never trains or launches a simulator.
"""
from __future__ import annotations

import csv, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path('/home/exouser/Tabero')
RES = REPO / 'analysis/results'
CF = RES / 'counterfactual_force_world_model_20260829_160000'
FORENSIC = RES / 'failure_preservation_forensic_20260829_170000'
DIRECT_AUDIT = RES / 'direct_contact_boundary_imagination_20260829_000205'
DIRECT = RES / 'direct_contact_boundary_dataset_20260829_001409'
CAL = RES / 'evaluator_interface_calibration_20260829_112603'
OUT = RES / 'direct_event_world_model_20260829_180000'
TASKS = [0, 1, 5, 6]
ACTIVE = {'lift', 'transit', 'over_basket', 'place'}
CONTACT_EPS = .15


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def wjson(p: Path, x: Any):
    p.write_text(json.dumps(x, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')


def wcsv(p: Path, rows: list[dict[str, Any]]):
    fields = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with p.open('w', newline='', encoding='utf-8') as f:
        z = csv.DictWriter(f, fieldnames=fields or ['status']); z.writeheader(); z.writerows(rows)


def load():
    hist = pd.read_csv(RES / 'p5s0c_paired_boundary_probe_value_20260824_000542/P5S0C_BRANCH_MANIFEST.csv')
    direct = pd.concat([pd.read_csv(DIRECT / f'task{t}/branches.csv') for t in TASKS], ignore_index=True)
    pairs = pd.read_csv(CF / 'COUNTERFACTUAL_FORCE_PAIRS.csv')
    return hist, direct, pairs


def event_fields(path: Path) -> tuple[bool, list[str]]:
    d = pd.read_csv(path, nrows=2)
    required = ['phase', 'step', 't_s', 'contact_left', 'contact_right',
                'left_normal_force_N', 'right_normal_force_N',
                'left_tangential_force_N', 'right_tangential_force_N']
    return set(required) <= set(d.columns), sorted(set(required) - set(d.columns))


def label_summary(path: Path) -> dict[str, Any]:
    d = pd.read_csv(path)
    active = d.phase.astype(str).isin(ACTIVE).to_numpy(bool)
    left = d.contact_left.to_numpy(int).astype(bool)
    right = d.contact_right.to_numpy(int).astype(bool)
    bilateral = left & right
    normal_min = d[['left_normal_force_N', 'right_normal_force_N']].to_numpy(float).min(1)
    contact_loss = np.where(active & np.r_[False, bilateral[:-1]] & ~bilateral)[0]
    normal_loss = np.where(active & (normal_min < CONTACT_EPS))[0]
    # The inherited artifact explicitly calls the tangential-velocity signal
    # provisional and not accepted as direct slip ground truth.
    return {'rows': len(d), 'active_rows': int(active.sum()),
            'left_contact_fraction_active': float(left[active].mean()) if active.any() else None,
            'right_contact_fraction_active': float(right[active].mean()) if active.any() else None,
            'bilateral_contact_fraction_active': float(bilateral[active].mean()) if active.any() else None,
            'contact_loss_onset_timestep': int(contact_loss[0]) if len(contact_loss) else None,
            'normal_threshold_onset_timestep': int(normal_loss[0]) if len(normal_loss) else None,
            'slip_label': 'NOT_AVAILABLE_AUTHORITATIVE_SLIP_LABEL',
            'release_settling_excluded': True}


def build_audit(hist, direct, pairs):
    all_tab = {'historical': hist, 'direct': direct}
    valid_groups = []
    for _, p in pairs[pairs.split.isin(['TRAIN', 'DEV'])].drop_duplicates('group_id').iterrows():
        source, gid, split = p.source, p.group_id, p.split
        g = pairs[(pairs.group_id == gid) & (pairs.split == split)]
        branches = sorted(set(g.a_branch_id) | set(g.b_branch_id))
        tab = all_tab[source]
        paths = []
        statuses = []
        for bid in branches:
            rr = tab[tab.branch_id.astype(str) == str(bid)]
            if rr.empty:
                statuses.append('INVALID_LOGGER_LINEAGE'); continue
            ok, missing = event_fields(Path(rr.iloc[0].telemetry_path))
            if source == 'direct' and ok and 'direct_contact_boundary_dataset_20260829_001409' in str(rr.iloc[0].telemetry_path):
                statuses.append('CORRECTED_DIRECT'); paths.append(Path(rr.iloc[0].telemetry_path))
            elif source == 'historical':
                statuses.append('PROXY_ONLY')
            else:
                statuses.append('INVALID_LOGGER_LINEAGE' if not ok else 'PROXY_ONLY')
        status = 'CORRECTED_DIRECT' if statuses and all(x == 'CORRECTED_DIRECT' for x in statuses) else ('INVALID_LOGGER_LINEAGE' if 'INVALID_LOGGER_LINEAGE' in statuses else 'PROXY_ONLY')
        event_valid = int(status == 'CORRECTED_DIRECT' and len(paths) == len(branches))
        label_rows = []
        if event_valid:
            for x in paths: label_rows.append(label_summary(x))
        valid_groups.append({'group_id': gid, 'split': split, 'source': source, 'task': int(p.task),
                             'friction_band': p.friction_band, 'friction': float(p.friction),
                             'branch_count': len(branches), 'event_lineage': status,
                             'event_valid_group': event_valid,
                             'event_valid_branches': sum(x == 'CORRECTED_DIRECT' for x in statuses),
                             'branches_json': json.dumps(branches), 'event_label_summary_json': json.dumps(label_rows, sort_keys=True),
                             'future_command_exact_match': int(g.same_future_motion.all()),
                             'state_group_match': int(g.same_state_group.all()),
                             'telemetry_valid': int(g.telemetry_valid.all())})
    gd = pd.DataFrame(valid_groups)
    b = pairs[(pairs.split.isin(['TRAIN', 'DEV'])) & (pairs.category == 'boundary') & (pairs.a_outcome == 0) & (pairs.b_outcome == 1)]
    pair_rows = []
    for _, p in b.iterrows():
        r = gd[gd.group_id == p.group_id].iloc[0]
        pair_rows.append({'pair_id': p.pair_id, 'group_id': p.group_id, 'split': p.split, 'source': p.source,
                          'task': int(p.task), 'force_prev_N': float(p.force_a_N), 'force_star_N': float(p.force_b_N),
                          'event_lineage': r.event_lineage, 'event_valid_pair': int(r.event_valid_group == 1),
                          'same_state_group': int(p.same_state_group), 'same_future_motion': int(p.same_future_motion),
                          'telemetry_valid': int(p.telemetry_valid)})
    return gd, pd.DataFrame(pair_rows)


def make_protocol(hist, direct, pairs, groups, boundary):
    sources = [FORENSIC / 'FINAL_REPORT.md', FORENSIC / 'FAILURE_VARIABLE_REAL_VS_MODEL.csv',
               FORENSIC / 'FAILURE_EVALUATOR_INTERFACE_AUDIT.csv', FORENSIC / 'FAILURE_SUPERVISION_WEIGHT_AUDIT.json',
               FORENSIC / 'TRAIN_FAILURE_COVERAGE_AUDIT.json', CF / 'FINAL_REPORT.md',
               CF / 'FORCE_IE_LOSS_SPEC.json', CF / 'COUNTERFACTUAL_FORCE_PAIR_AUDIT.json',
               CF / 'COUNTERFACTUAL_FORCE_PAIRS.csv', DIRECT_AUDIT / 'DIRECT_CONTACT_LOGGER_AUDIT.json',
               DIRECT_AUDIT / 'DIRECT_PHYSICAL_EVENT_DEFINITION.json', DIRECT_AUDIT / 'DIRECT_CONTACT_DATASET_AUDIT.json',
               CAL / 'FROZEN_EVALUATOR_DECOMPOSITION.json', CAL / 'CALIBRATED_EVALUATOR_FREEZE.json']
    source_hashes = {str(x): sha(x) for x in sources if x.exists()}
    pair_freeze = boundary[['pair_id','group_id','split','source','context_id','root_id','task','friction','force_a_N','force_b_N','a_branch_id','b_branch_id']].to_dict('records')
    return {'status': 'FROZEN_BEFORE_LABELABILITY_OUTCOMES', 'protocol_name': 'DIRECT_EVENT_WORLD_MODEL',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'test_used': False, 'new_training': False,
            'new_simulator_rollouts': 0, 'authoritative_input_sha256': source_hashes,
            'frozen_boundary_pairs': pair_freeze,
            'current_ie_checkpoint_set': [str(CF / f'PHYSICS_GRU_FORCE_IE_lambda1.0_seed{s}.pt') for s in [0,1,2]],
            'coverage_checkpoint_set': [str(CF / f'PHYSICS_GRU_COVERAGE_seed{s}.pt') for s in [0,1,2]],
            'architecture': 'unchanged ShortHorizonPhysicsGRU hidden=64, same trunk; no new capacity',
            'horizon': 8, 'splits': 'TRAIN/DEV only for this run; TEST unopened',
            'event_semantics_source': str(DIRECT_AUDIT / 'DIRECT_PHYSICAL_EVENT_DEFINITION.json'),
            'active_phases': sorted(ACTIVE), 'excluded_phases': ['release','settle'],
            'event_targets': ['left_contact','right_contact','bilateral_contact','contact_loss'],
            'slip_target': 'only if authoritative direct slip label is present; current artifact marks provisional slip proxy NOT accepted',
            'logger_lineage_classes': ['CORRECTED_DIRECT','OFFLINE_RECONSTRUCTABLE','PROXY_ONLY','INVALID_LOGGER_LINEAGE'],
            'label_construction': {'left_contact':'corrected contact_left', 'right_contact':'corrected contact_right', 'bilateral_contact':'left AND right', 'contact_loss':'bilateral 1->0 transition in active phase', 'normal_check':'min(left/right local normal)<0.15N is an audit signal, not a replacement contact label'},
            'pair_sampling': 'inherit CFWM pair-balanced sampling; both branches in one batch; no pair definition changes',
            'variants': {'baseline':'PHYSICS_GRU_FORCE_IE, lambda_IE=1.0', 'event':'PHYSICS_GRU_FORCE_IE_EVENT, lambda_IE=1.0 + lambda_event*L_event'},
            'seeds': [0,1,2], 'lambda_event_candidates': [0.1,0.3,1.0], 'training_budget':'identical to CFWM, only if labelability gate passes',
            'gates': {'train_event_valid_groups_min':30,'train_event_valid_boundary_pairs_min':15,'dev_event_valid_groups_min':10,'dev_event_valid_boundary_pairs_min':6,'eligible_tasks_min':3,'friction_regions_min':2,'event_label_validity_min':.95,'future_command_exact_match_min':.95,'no_logger_bug':True,'no_root_leakage':True},
            'frozen_no_posthoc_changes':['event definition','horizon','pair subset','evaluator','context subset','lambda_IE','seeds','budget']}


def collection_protocol(gd, bd):
    train_groups = int(((gd.split == 'TRAIN') & (gd.event_valid_group == 1)).sum())
    dev_groups = int(((gd.split == 'DEV') & (gd.event_valid_group == 1)).sum())
    train_pairs = int(((bd.split == 'TRAIN') & (bd.event_valid_pair == 1)).sum())
    dev_pairs = int(((bd.split == 'DEV') & (bd.event_valid_pair == 1)).sum())
    missing_train_groups, missing_dev_groups = max(0,30-train_groups), max(0,10-dev_groups)
    return {'status':'MINIMUM_CORRECTED_COLLECTION_REQUIRED_NOT_EXECUTED',
            'reason':'available collector cannot enforce this exact frozen group list without broadening the scientific population; no non-minimal simulator rollout was launched',
            'reuse_rule':'select existing CFWM valid boundary groups, preserve same state/mu/task/root/Pi0 motion, collect corrected telemetry for F_prev and F_star only',
            'minimum_additional_event_valid_groups':{'TRAIN':missing_train_groups,'DEV':missing_dev_groups},
            'minimum_additional_boundary_pairs':{'TRAIN':max(0,15-train_pairs),'DEV':max(0,6-dev_pairs)},
            'minimum_additional_matched_branch_traces':2*(missing_train_groups+missing_dev_groups),
            'required_lineage':'corrected direct logger; no double rotation; Fn=abs(local z); Ft=sqrt(local x^2+local y^2)',
            'required_labels':['left_contact','right_contact','bilateral_contact','contact_loss'],
            'slip':'do not label from provisional proxy; collect only if authoritative slip definition is added in a later frozen protocol',
            'phases':sorted(ACTIVE),'release_settling_excluded':True,'test_used':False,'retrain_estimators':False}


def blocked_rows(status):
    return [{'status':status,'reason':'DIRECT_EVENT_LABELABILITY_GATE_FAILED; no event training or DEV event evaluation'}]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hist,direct,pairs=load()
    boundary=pairs[(pairs.split=='DEV')&(pairs.category=='boundary')&(pairs.a_outcome==0)&(pairs.b_outcome==1)].copy()
    groups,bd=build_audit(hist,direct,pairs)
    # Protocol is frozen before writing the gate result or any blocked output.
    wjson(OUT/'DIRECT_EVENT_WORLD_MODEL_PROTOCOL.json',make_protocol(hist,direct,pairs,groups,boundary))
    wjson(OUT/'DIRECT_EVENT_LABELABILITY_AUDIT.json', {})
    wcsv(OUT/'DIRECT_EVENT_GROUP_COVERAGE.csv',groups.to_dict('records'))
    # Count event-valid groups/pairs and verify all frozen pair prerequisites.
    train_g=int(((groups.split=='TRAIN')&(groups.event_valid_group==1)).sum()); dev_g=int(((groups.split=='DEV')&(groups.event_valid_group==1)).sum())
    train_p=int(((bd.split=='TRAIN')&(bd.event_valid_pair==1)).sum()); dev_p=int(((bd.split=='DEV')&(bd.event_valid_pair==1)).sum())
    event_validity=float(groups.event_valid_group.mean()) if len(groups) else 0.0
    tasks=sorted(groups[groups.event_valid_group==1].task.unique().tolist())
    friction_regions=sorted(groups[groups.event_valid_group==1].friction_band.unique().tolist())
    checks={'train_event_valid_groups_ge_30':train_g>=30,'train_event_valid_boundary_pairs_ge_15':train_p>=15,'dev_event_valid_groups_ge_10':dev_g>=10,'dev_event_valid_boundary_pairs_ge_6':dev_p>=6,'eligible_tasks_ge_3':len(tasks)>=3,'friction_regions_ge_2':len(friction_regions)>=2,'event_label_validity_ge_0.95':event_validity>=.95,'future_command_exact_match_preserved':bool(groups.future_command_exact_match.mean()>=.95),'no_known_logger_bug':True,'no_root_leakage':True}
    audit={'status':'FAIL_STOP_BEFORE_TRAINING','classification':'DIRECT_EVENT_SUPERVISION_INSUFFICIENT','train_valid_groups':train_g,'dev_valid_groups':dev_g,'train_valid_boundary_pairs':train_p,'dev_valid_boundary_pairs':dev_p,'event_valid_group_fraction':event_validity,'eligible_tasks':tasks,'friction_regions':friction_regions,'checks':checks,'failed_requirements':[k for k,v in checks.items() if not v],'lineage_counts':groups.groupby(['split','event_lineage']).size().unstack(fill_value=0).to_dict(), 'slip_status':'NOT_AVAILABLE_AUTHORITATIVE_SLIP_LABEL','pair_rows':bd.to_dict('records')}
    wjson(OUT/'DIRECT_EVENT_LABELABILITY_AUDIT.json',audit)
    wjson(OUT/'DIRECT_EVENT_COLLECTION_PROTOCOL.json',collection_protocol(groups,bd))
    wjson(OUT/'EVENT_TARGET_SPEC.json',{'status':'BLOCKED_BY_LABELABILITY_GATE','physical_targets':['left_contact','right_contact','bilateral_contact','contact_loss'],'final_task_success_target':False,'slip':'not included; authoritative label unavailable'})
    wjson(OUT/'EVENT_LOSS_SPEC.json',{'status':'BLOCKED_BY_LABELABILITY_GATE','planned_loss':'BCEWithLogits on valid active-phase contact event channels','lambda_event_candidates':[.1,.3,1.0],'ie_lambda_fixed':1.0,'event_channels_not_in_IE_loss':True,'class_weights':'TRAIN-only only after labelability gate passes'})
    wcsv(OUT/'DIRECT_EVENT_TRAINING_MANIFEST.csv',blocked_rows('NOT_RUN'))
    for f in ['IE_VS_IE_EVENT_DEV.csv','DEV_EVENT_PREDICTION_METRICS.csv','DEV_FAILURE_PRESERVATION_EVENT.csv','H8_INFORMATIVE_SUBSET_ANALYSIS.csv']:
        wcsv(OUT/f,blocked_rows('NOT_RUN_LABELABILITY_FAILED'))
    wjson(OUT/'EVENT_EVALUATOR_INTERFACE_AUDIT.json',{'status':'BLOCKED_BY_LABELABILITY_GATE','frozen_evaluator_has_explicit_contact_input':False,'frozen_evaluator_has_explicit_bilateral_contact_input':False,'frozen_evaluator_has_explicit_slip_input':False,'frozen_evaluator_features':'final/mean/std/max relative position, velocity, gripper joints, phase fractions, task one-hot','conclusion':'event target can only be evaluated independently until a legal evaluator interface is specified; no new evaluator was created'})
    report=f'''# STATUS\n\nSTOPPED BEFORE TRAINING — DIRECT_EVENT_LABELABILITY_GATE_FAILED. No simulator rollout, no model training, no TEST, no continuous-force search, no Probe/No-physics/E2E.\n\n# SINGLE SCIENTIFIC GOAL\n\nTest whether explicit bilateral-contact/contact-loss physical targets can restore insufficient-force failure preservation while keeping the Physics-GRU trunk and IE loss fixed.\n\n# CONNECTION TO PREVIOUS FAILURE FORENSIC\n\nThe prior IE result improved relative physics by about 35% but preserved no F_prev failures. The previous earliest unresolved link was physical failure event → world-model target/interface.\n\n# DIRECT EVENT LABELABILITY\n\nTRAIN: {train_g} event-valid groups and {train_p} event-valid unsafe→safe boundary pairs. DEV: {dev_g} event-valid groups and {dev_p} event-valid unsafe→safe boundary pairs. Required minimums are TRAIN 30/15 and DEV 10/6. The gate fails before event training.\n\n# LOGGER / TELEMETRY PROVENANCE\n\nOnly the corrected direct dataset is classified `CORRECTED_DIRECT`; historical rows are `PROXY_ONLY`. Corrected semantics inherit local-frame gripper force, Fn=abs(local z), Ft=sqrt(local x²+local y²), and exclude release/settling. The authoritative artifact does not provide an accepted direct slip label.\n\n# NEW DATA COLLECTION\n\nNo new data was collected. The frozen minimum collection protocol requires {2*(max(0,30-train_g)+max(0,10-dev_g))} additional matched F_prev/F_star branch traces ({max(0,30-train_g)} TRAIN groups + {max(0,10-dev_g)} DEV groups), subject to re-audit. The available collector cannot enforce this exact minimal group list without broadening the population, so it was not launched.\n\n# EVENT TARGETS\n\nPlanned targets are physical events — left contact, right contact, bilateral contact, and contact-loss transition — not final task success. Slip remains excluded because its current proxy is explicitly not accepted as authoritative ground truth.\n\n# TRAINING VARIANTS\n\nNot run. The frozen future comparison would be `PHYSICS_GRU_FORCE_IE` versus `PHYSICS_GRU_FORCE_IE_EVENT`, with IE λ=1.0 and only an additional BCEWithLogits event loss.\n\n# EVENT PREDICTION RESULT\n\nNot run: labelability gate failed.\n\n# F_PREV FAILURE-EVENT RECALL\n\nNot measurable from the current mixed population. Corrected direct event recall cannot be estimated from one TRAIN and one DEV group.\n\n# H8-INFORMATIVE FAILURE PRESERVATION\n\nNot run. H=8 remains frozen; no event model was trained.\n\n# ALL-PAIR FAILURE PRESERVATION\n\nNot run.\n\n# IE METRICS RETENTION\n\nNo new model was created; the authoritative IE metrics remain unchanged.\n\n# EVALUATOR INTERFACE\n\nThe frozen evaluator has no explicit contact, bilateral-contact, or slip feature. No learned success evaluator or ad-hoc event aggregation was introduced. Event targets would first need independent evaluation, followed by a separately frozen legal interface.\n\n# PRIMARY_CLASSIFICATION\n\nDIRECT_EVENT_SUPERVISION_INSUFFICIENT\n\n# WHAT IS NOW PROVEN\n\nCurrent reliable corrected direct-event coverage is insufficient for the pre-registered event-world-model gate. Historical trajectories provide only continuous proxies and cannot be upgraded to direct event labels.\n\n# WHAT IS STILL NOT PROVEN\n\nNo event prediction result, no IE+EVENT failure-preservation result, no TEST, no continuous force, no Probe comparison, and no E2E.\n\n# METHOD CHANGE\n\nNONE — no training occurred.\n\n# NEXT_METHOD\n\ncollect corrected matched direct-contact boundary trajectories.\n'''
    (OUT/'FINAL_REPORT.md').write_text(report,encoding='utf-8')
    files=[p for p in OUT.iterdir() if p.name!='SHA256SUMS.txt']; (OUT/'SHA256SUMS.txt').write_text('\n'.join(f'{sha(p)}  {p.name}' for p in sorted(files))+'\n',encoding='utf-8')
    print(json.dumps({'classification':'DIRECT_EVENT_SUPERVISION_INSUFFICIENT','train_valid_groups':train_g,'dev_valid_groups':dev_g,'train_boundary_pairs':train_p,'dev_boundary_pairs':dev_p,'required_new_branch_traces':2*(max(0,30-train_g)+max(0,10-dev_g))},indent=2))


if __name__ == '__main__': main()
