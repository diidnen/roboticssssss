"""Read-only source reconciliation; writes derived audit artifacts only."""
import csv
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import run_full_horizon_matched_pilot_20260905 as run
from activeforcing_execution_snapshot import max_difference

OUT=run.OUT
HISTORY=run.HISTORY_ROOT
PARENT=HISTORY.parent
BANDS=('high','mid','low')
FIELDS=('full_task_success_y','lift_success','place_success','dropped')


def read(p):return json.loads(Path(p).read_text())
def write(name,data):run.write(OUT/name,data)
def csvread(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def csvwrite(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def mark_early_pilot():
    early=HISTORY/'guard_fixed'
    rows=[]
    for p in sorted(early.glob('branches/*/*/RESULT.json')):
        r=read(p);native=csvread(p.parent/'NATIVE_BRANCH_TRACE_DIAGNOSTIC.csv')
        counts=dict(Counter(t['phase'] for t in native))
        rows.append({'context':r['context'],'candidate_F':r['candidate_F'],'repeat':r['repeat'],
            'raw_legacy_success':r['full_task_success_y'],'steps':r['steps'],
            'final_phase':native[-1]['phase'],'all_phases_completed':counts==run.EXPECTED_PHASE_STEPS,
            'phase_counts':counts,'raw_result_path':str(p),'raw_result_sha256':run.sha256(p)})
    assert len(rows)==12 and not any(r['all_phases_completed'] for r in rows)
    run.write(early/'EARLY_TERMINATION_ELIGIBILITY_AUDIT.json',{
        'status':'EARLY_TERMINATION_DIAGNOSTIC_ONLY','formal_full_task_labels_eligible':False,
        'reason':'Success terminated/reset at transit step 156 before place/release/settle',
        'raw_artifacts_preserved':True,'rows':rows})
    text='''# Early-termination diagnostic only

All 12 branches ended at step 156 during transit. They passed the original loose task-goal/any-basket-contact condition, but did not execute the complete place/release/settle schedule. Their raw `full_task_success_y=1` fields are historical runner outputs, **not qualified complete-task labels**. None may enter formal feasibility training.

Original artifacts are retained unchanged. See EARLY_TERMINATION_ELIGIBILITY_AUDIT.json for per-file hashes and phase counts, and ../full_horizon/ for the corrected full-horizon pilot. No physical success probability or force boundary is inferred from this diagnostic.
'''
    (early/'MATCHED_PILOT_REPORT.md').write_text(text)
    return rows


def reconcile():
    run.protocol()
    if not (OUT/'PILOT_COMPLETE.json').exists():raise RuntimeError('Full pilot incomplete')
    rows=[];phases=[];traces={};checks=[];parity=[];feature_max=0.
    for band in BANDS:
        ref=run.reference_dir(band)
        x=np.load(ref/'FROZEN_PREACTION_FEATURES.npy')
        assert x.shape==(9,8,71)
        # Same posterior member across candidates: only explicit F/8 at index17 differs.
        for mu in range(3):
            for f_index in (1,2):
                diff=np.delete(x[3*f_index+mu]-x[mu],17,axis=-1)
                feature_max=max(feature_max,float(np.max(np.abs(diff))))
                np.testing.assert_array_equal(diff,np.zeros_like(diff))
                np.testing.assert_array_equal(x[3*f_index+mu,:,17],np.full(8,(3+f_index)/8))
        snapshot=torch.load(ref/'DECISION_STATE.pt',map_location='cpu',weights_only=False)
        assert snapshot['candidate_actions_already_executed']==0
        obj=csvread(ref/'RAW_PROBE.csv')[-1]['object_id']
        z0=float(snapshot['state']['rigid_object'][obj]['root_pose'][0,2])
        for f,repeat in [(3.,0),(4.,0),(5.,0),(4.,1)]:
            job=run.job_dir('branch',band,f,repeat)
            r=read(job/'RESULT.json');trace=read(job/'BRANCH_TRACE.json')
            native=csvread(job/'NATIVE_BRANCH_TRACE_DIAGNOSTIC.csv')
            counts=dict(Counter(t['phase'] for t in native))
            assert len(trace)==len(native)==r['steps']
            assert r['decision_snapshot_sha256']==run.sha256(ref/'DECISION_STATE.pt')
            assert r['reference_feature_sha256']==run.sha256(ref/'FROZEN_PREACTION_FEATURES.npy')
            np.testing.assert_array_equal(np.load(job/'FROZEN_PREACTION_FEATURES.npy'),x)
            assert read(job/'PREFIX_RECONSTRUCTION_GATE.json')['passed']
            assert read(job/'FEATURE_GATE.json')['passed']
            assert read(job/'DECISION_FEATURES.json')['no_candidate_executed_before_extraction']
            early_trace=read(HISTORY/'guard_fixed'/'branches'/band/f'F{f:g}_R{repeat}'/'BRANCH_TRACE.json')
            # Last old trace is the pre-reset terminal state, not reset observation.
            old_prefix_comparison=run.base.compare_trace(early_trace,trace[:len(early_trace)])
            old_action_diff=max_difference([t['action'] for t in early_trace],
                [t['action'] for t in trace[:len(early_trace)]])
            if not old_prefix_comparison['passed'] or old_action_diff!=0:
                raise AssertionError('Changed pre-termination dynamics compared with old pilot')
            # Independent evaluator does not call the runner's label helper.
            lift=int(max(t['object_position_m'][2]-z0 for t in trace)>=.03)
            complete=counts=={'branch_hold':20,'lift':50,'transit':110,'over_basket':30,'place':40,'release':50,'settle':50}
            place=int(complete and trace[-1]['terminations']['success'] and trace[-1]['basket_contact_force_N']>.05)
            dropped=int(any(t['terminations']['object_1_dropped'] for t in trace))
            label=int(lift and place and not dropped)
            assert (label,lift,place,dropped)==tuple(r[k] for k in FIELDS)
            for t in trace:
                if t['captured_before_auto_reset']:
                    assert any(v for k,v in t['terminations'].items() if k!='success')
            settle=[t for t,n in zip(trace,native) if n['phase']=='settle']
            row={'context':band.upper(),'candidate_F':f,'repeat':repeat,
                'full_task_success_y':label,'lift_success':lift,'place_success':place,'dropped':dropped,
                'full_task_label_validated':False,'label_status':'FROZEN_CONTACT_CRITERION_DIAGNOSTIC_ONLY_PENDING_GEOMETRY',
                'steps':len(trace),'all_phases_completed':complete,'final_phase':native[-1]['phase'],
                'first_goal_step':r['first_task_goal_step'],'final_goal':trace[-1]['terminations']['success'],
                'final_basket_contact_N':trace[-1]['basket_contact_force_N'],
                'final_normal_left_N':trace[-1]['normal_force_N'][0],
                'final_normal_right_N':trace[-1]['normal_force_N'][1],
                'final_object_linear_speed_m_s':float(np.linalg.norm(trace[-1]['object_velocity'][:3])),
                'settle_basket_contact_fraction':float(np.mean([t['basket_contact_force_N']>.05 for t in settle])) if settle else None,
                'settle_no_finger_contact_fraction':float(np.mean([max(t['normal_force_N'])<=.15 for t in settle])) if settle else None,
                'settle_object_position_range_m':float(np.max(np.ptp(np.array([t['object_position_m'] for t in settle]),axis=0))) if settle else None,
                'max_lift_m':max(t['object_position_m'][2]-z0 for t in trace),
                'terminal_reset_count':r['terminal_reset_count'],
                'native_legacy_label':r['native_runner_label'],
                'native_legacy_label_agrees':r['native_label_agrees'],
                'source':str(job/'RESULT.json'),'source_sha256':run.sha256(job/'RESULT.json')}
            rows.append(row);traces[band,f,repeat]=trace
            checks.append({'context':band,'candidate_F':f,'repeat':repeat,'independent_label_matches':True,
                'source_sha256':row['source_sha256'],'feature_parity_exact':True,'post_action_input':False,
                'old_pilot_prefix_dynamics_comparison':old_prefix_comparison,
                'old_pilot_prefix_action_max_diff':old_action_diff})
            for phase in counts:
                pairs=[(t,n) for t,n in zip(trace,native) if n['phase']==phase]
                normal=np.array([t['normal_force_N'] for t,n in pairs]);bi=2*np.min(normal,axis=1)
                phases.append({'context':band.upper(),'candidate_F':f,'repeat':repeat,'phase':phase,'steps':len(pairs),
                    'mean_left_normal_N':float(normal[:,0].mean()),'mean_right_normal_N':float(normal[:,1].mean()),
                    'mean_bilateral_normal_N':float(bi.mean()),'min_bilateral_normal_N':float(bi.min()),
                    'mean_effective_controller_target_N':float(np.mean([float(n['F_target_eff_n']) for t,n in pairs])),
                    'mean_predicted_finger_command_m':float(np.mean([float(n['gripper_aperture_pred']) for t,n in pairs])),
                    'mean_actual_per_finger_position_m':float(np.mean([float(n['gripper_aperture_actual']) for t,n in pairs])),
                    'zero_joint_target_fraction':float(np.mean([float(n['gripper_aperture_cmd'])<=1e-8 for t,n in pairs])),
                    'max_basket_contact_N':max(t['basket_contact_force_N'] for t,n in pairs),
                    'basket_contact_fraction':float(np.mean([t['basket_contact_force_N']>.05 for t,n in pairs])),
                    'raw_goal_satisfied_fraction':float(np.mean([t['terminations']['success'] for t,n in pairs]))})
        a,b=traces[band,4.,0],traces[band,4.,1]
        comparison=run.base.compare_trace(a,b)
        action_diff=max_difference([t['action'] for t in a],[t['action'] for t in b])
        terminal_diff=max_difference([t['terminations'] for t in a],[t['terminations'] for t in b])
        basket_diff=max_difference([t['basket_contact_force_N'] for t in a],[t['basket_contact_force_N'] for t in b])
        same=all(next(r for r in rows if r['context']==band.upper() and r['candidate_F']==4 and r['repeat']==0)[k]==
            next(r for r in rows if r['context']==band.upper() and r['candidate_F']==4 and r['repeat']==1)[k] for k in FIELDS)
        passed=bool(comparison['passed'] and comparison.get('episode_step_equal') and action_diff==0 and terminal_diff==0 and basket_diff==0 and same)
        parity.append({'context':band,'passed':passed,'steps':len(a),'trace_comparison':comparison,
            'action_max_diff':action_diff,'terminal_max_diff':terminal_diff,'basket_contact_max_diff':basket_diff,'labels_equal':same})
    return rows,phases,checks,parity,feature_max


def main():
    early=mark_early_pilot()
    rows,phases,checks,parity,feature_max=reconcile()
    geometry_path=HISTORY/'placement_geometry_diagnostic/GEOMETRY_AUDIT.json'
    geometry=read(geometry_path)
    geometry_cases=[item['case'] for item in geometry['cases']]
    assert all(item['replay_gate_passed'] for item in geometry_cases)
    csvwrite(OUT/'EMPIRICAL_FORCE_SUCCESS_TABLE.csv',rows)
    checked_cases={(c['context'],c['candidate_F']) for c in geometry_cases}
    reviewed=[]
    for row in rows:
        if row['repeat']!=0:continue
        verified=(row['context'],row['candidate_F']) in checked_cases
        reviewed.append({'context':row['context'],'candidate_F':row['candidate_F'],
            'full_task_success_y_verified':0 if verified else '',
            'lift_success':row['lift_success'],'place_success_verified':0 if verified else '',
            'dropped':row['dropped'],'raw_contact_criterion_y':row['full_task_success_y'],
            'review_status':'OUTSIDE_BASKET_CONFIRMED_BY_FINAL_POSE_AND_IMAGE' if verified else 'NOT_GEOMETRICALLY_ADJUDICATED',
            'formal_training_label':False})
    csvwrite(OUT/'GEOMETRY_REVIEWED_EMPIRICAL_TABLE.csv',reviewed)
    csvwrite(OUT/'PHASE_FORCE_DIAGNOSTICS.csv',phases)
    write('LABEL_AUDIT.json',checks);write('FULL_TASK_RECONSTRUCTION_PARITY.json',parity)
    # Record unchanged evaluator/config source paths separately; these hashes
    # are audit-time provenance, not retroactively claimed pre-run freeze keys.
    evaluator_sources=[
        run.smoke.TABERO/'source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/terminations.py',
        run.smoke.TABERO/'source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/franka_libero_env_cfg.py',
        run.smoke.TABERO/'source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/franka_tactile_libero_env_cfg.py',
        run.smoke.TABERO/'benchmarks/datasets/libero/config/libero_object.json']
    write('EVALUATOR_SOURCE_AUDIT.json',{'recorded_at':'post-run audit; not claimed as pre-run frozen',
        'source_hashes':{str(p):run.sha256(p) for p in evaluator_sources},
        'raw_goal_alone_not_used':True,
        'final_goal_true_without_final_contact_count':sum(r['final_goal'] and r['final_basket_contact_N']<=.05 for r in rows),
        'label_definition_not_containment_proof':True,
        'dropped_semantics':'environment object_1_dropped: object height below -0.05m; not a generic slip detector'})
    gate=all(p['passed'] for p in parity)
    representative=[r for r in rows if r['repeat']==0]
    boundary={b:len({r['full_task_success_y'] for r in representative if r['context']==b.upper()})==2 for b in BANDS}
    monotone={b:all(a<=c for a,c in zip(
        [r['full_task_success_y'] for r in representative if r['context']==b.upper()],
        [r['full_task_success_y'] for r in representative if r['context']==b.upper()][1:])) for b in BANDS}
    minima={b:min([r['candidate_F'] for r in representative if r['context']==b.upper() and r['full_task_success_y']] or [99.]) for b in BANDS}
    probes={b:read(run.reference_dir(b)/'PROBE_RESULT.json') for b in BANDS}
    posterior={b:read(run.reference_dir(b)/'POSTERIOR.json') for b in BANDS}
    integration=read(PARENT/'POSTERIOR_INTEGRATION_DIAGNOSTIC.json')
    starts=list(PARENT.rglob('STARTED.json'))
    # Initial direct restore diagnostics use the same marker; count all attempts, not just accepted data.
    completed=sum((p.parent/'RAW_PROBE.csv').exists() for p in starts)
    blockers=['Confirmed false-positive place label: HIGH-4N object remains outside basket but contact criterion labels success',
        'Basket moves 10.3-10.6cm during checked branches while downstream trajectory retains the pre-branch basket target; task path/placement execution needs repair',
        'Formal posterior integration is unvalidated: three member means omit sigma uncertainty',
        'One task0 root and three friction contexts do not validate task1/task5/task6']
    if not gate:blockers.append('Complete same-force execution reconstruction gate failed')
    if not any(boundary.values()):blockers.append('No within-context force-success boundary observed at unchanged 3/4/5N on these contexts')
    if not all(monotone.values()):blockers.append('Pilot contains non-monotone full-task outcomes; placement/contact dynamics need qualification before a monotone force-boundary claim')
    status={'FINAL_STATUS':'EXECUTION_PARITY_PASSED_PLACEMENT_PATH_AND_LABEL_BUG_CONFIRMED' if gate else 'FULL_HORIZON_PILOT_RECONSTRUCTION_FAILED',
        'PROBE_CONTRACT_FIXED':True,'POSTERIOR_INPUT_CONTRACT_FIXED':True,
        'CURRENT_PROBE_STEPS_HIGH_MID_LOW':[probes[b]['total_steps'] for b in BANDS],
        'CURRENT_PROBE_OUTWARD_STEPS_HIGH_MID_LOW':[probes[b]['outward_steps'] for b in BANDS],
        'CURRENT_POSTERIOR_HIGH_MID_LOW':[posterior[b]['mean'] for b in BANDS],
        'CURRENT_POSTERIOR_SUPPORT':{b:posterior[b]['member_means'] for b in BANDS},
        'FORMAL_CONTINUOUS_POSTERIOR_VALIDATED':False,
        'DECISION_STATE_DEFINED':True,'TEMPORAL_FEATURE_PARITY':True,
        'CANDIDATE_PREACTION_STATE_EQUALITY':True,'MAX_FEATURE_DIFF':feature_max,
        'EXECUTION_CONTRACT_PARITY':gate,'DIRECT_SNAPSHOT_RESTORE_VALIDATED':False,
        'RESTORE_METHOD':'FRESH_PROCESS_EXACT_PREFIX_RECONSTRUCTION',
        'EXECUTION_PARITY_SCOPE':'tested isolated full-horizon pilot contract, not an automatic production-runtime migration',
        'PRODUCTION_RUNTIME_SWITCHED':False,
        'HANDOFF_COMMAND_CONTINUITY_FIXED':True,'FULL_HORIZON_SUCCESS_EVALUATION_FIXED':True,
        'FULL_TASK_LABEL_CONTRACT_FIXED':False,'DOWNSTREAM_PLACEMENT_CONTRACT_FIXED':False,
        'PILOT_PHYSICS_EXECUTED':True,'PILOT_CONTEXTS':[b.upper() for b in BANDS],
        'PILOT_CANDIDATE_FORCES':[3.,4.,5.],'PILOT_BRANCH_COUNT':12,'UNIQUE_CONTEXT_FORCE_CELLS':9,
        'FULL_TASK_REPEAT_FORCE':4.,'EARLY_TERMINATION_DIAGNOSTIC_BRANCHES':len(early),
        'GEOMETRY_DIAGNOSTIC_REPLAY_BRANCHES':len(geometry_cases),
        'TOTAL_PHYSICAL_PREFIX_ATTEMPTS_THIS_TURN':len(starts),
        'TOTAL_COMPLETED_PHYSICAL_PREFIXES_THIS_TURN':completed,
        'TOTAL_INCOMPLETE_PHYSICAL_PREFIXES_THIS_TURN':len(starts)-completed,
        'FORCE_BOUNDARY_EXISTS':'NOT_ESTABLISHED_PLACE_PROXY_INVALID',
        'CONTACT_CRITERION_FORCE_BOUNDARY_EXISTS':bool(gate and any(boundary.values())),
        'FORCE_BOUNDARY_INTERPRETATION':'Raw contact-criterion variation is observed; genuine full-task boundary is not yet qualified',
        'PLACE_LABEL_GEOMETRY_VALIDATED':False,
        'PLACE_LABEL_FALSE_POSITIVE_CONFIRMED':True,
        'GEOMETRY_CHECKED_CASES_OUTSIDE_BASKET':['HIGH-3N','HIGH-4N','LOW-5N'],
        'WITHIN_CONTEXT_FORCE_BOUNDARY':boundary,
        'EMPIRICAL_FORCE_SUCCESS_MONOTONE':monotone,
        'CONTEXT_DEPENDENT_FORCE_BOUNDARY':'NOT_ESTABLISHED_PLACE_PROXY_INVALID',
        'CONTACT_CRITERION_CONTEXT_DEPENDENT_BOUNDARY':bool(gate and len(set(minima.values()))>1),
        'MIN_SUCCESSFUL_TESTED_FORCE':{b:(None if v==99. else v) for b,v in minima.items()},
        'READY_FOR_FULL_DATA_COLLECTION':False,'FULL_COLLECTION_STARTED':False,
        'FEASIBILITY_RETRAINED':False,'PHYSICAL_BELIEF_RETRAINED_THIS_TURN':False,
        'UTILITY_CHANGED':False,'CONTROLLER_CHANGED':False,'PROBE_CHANGED_THIS_TURN':False,
        'POSTERIOR_INTERFACE_CHANGED':False,'RUNTIME_FORCE_RESCALED':False,
        'OLD720_LABELS_USED':False,'POST_ACTION_STATE_USED_AS_INPUT':False,
        'PHYSICAL_PLANNER_BENCHMARK_EXECUTED':False,
        'POSTERIOR_TEST_MEAN_VARIANCE_FRACTION_OMITTED':integration['summary']['TEST']['variance_fraction_omitted_by_member_mean_only_support'],
        'CURRENT_BLOCKERS':blockers}
    write('FINAL_STATUS.json',status)
    write('PLACEMENT_ROOT_CAUSE_AUDIT.json',{
        'geometry_audit_path':str(geometry_path),'geometry_audit_sha256':run.sha256(geometry_path),
        'checked_cases':geometry_cases,
        'image_review':'All three retained agentview images show the target object outside the basket',
        'confirmed_false_positive':'HIGH-4N: contact criterion=1, object outside basket',
        'rejected_intermediate_hypothesis':'LOW-5N was near the OLD basket position, not inside the moved basket',
        'branch_target_source':str(run.smoke.P5_PATH),'branch_target_source_sha256':run.sha256(run.smoke.P5_PATH),
        'target_construction':'downstream_branch reads basket_b once before phases and does not refresh it after basket displacement',
        'geometry_inspection_does_not_relabel_unchecked_rows':True,
        'qualified_full_task_boundary':False,'formal_labels_admitted':False,
        'next_required_scope':'repair nominal transport/placement clearance and target-state handling; define geometric place label before another bounded pilot; keep force controller/utility/probe frozen'})
    write('PILOT_LABEL_MANIFEST.json',{'formal_training_dataset_eligible':False,'pilot_only':True,'place_label_geometry_validated':False,
        'label_definition':'lift (rise >= .03 m) AND place (complete phases + final original goal + final basket contact > .05 N) AND NOT dropped',
        'protocol':str(OUT/'PROTOCOL.json'),'protocol_sha256':run.sha256(OUT/'PROTOCOL.json'),
        'restoration_method':status['RESTORE_METHOD'],'complete_execution_reconstruction_validated':gate,
        'labels':rows,'original_label_records_not_overwritten':True})
    belief=read(run.base.BELIEF)
    for checkpoint in belief['checkpoints']:
        assert run.sha256(checkpoint['path'])==checkpoint['sha256']
    write('ARTIFACT_PROVENANCE.json',{'run_protocol_sha256':run.sha256(OUT/'PROTOCOL.json'),
        'audit_script':str(Path(__file__).resolve()),'audit_script_sha256':run.sha256(Path(__file__)),
        'belief_manifest':str(run.base.BELIEF),'belief_manifest_sha256':run.sha256(run.base.BELIEF),
        'unchanged_belief_checkpoints':belief['checkpoints'],
        'source_hashes_verified':True,'production_runtime_switched':False,
        'empirical_table_sha256':run.sha256(OUT/'EMPIRICAL_FORCE_SUCCESS_TABLE.csv'),
        'phase_diagnostics_sha256':run.sha256(OUT/'PHASE_FORCE_DIAGNOSTICS.csv'),
        'label_audit_sha256':run.sha256(OUT/'LABEL_AUDIT.json')})
    tests=subprocess.run([sys.executable,'-m','unittest','-v','test_full_horizon_execution_contract.py',
        'test_activeforcing_command_handoff.py','test_activeforcing_execution_snapshot.py',
        'test_activeforcing_decision_state.py','test_activeforcing_probe_friction_contract.py',
        'test_current_contract_belief.py'],cwd=run.ROOT,capture_output=True,text=True)
    (OUT/'TEST_RESULTS.txt').write_text(tests.stdout+tests.stderr)
    if tests.returncode:raise RuntimeError('Regression suite failed')
    table='\n'.join(f"| {r['context']} | {r['candidate_F']:g} | {r['repeat']} | {r['full_task_success_y']} | {r['lift_success']} | {r['place_success']} | {r['dropped']} | {r['steps']} |" for r in rows)
    report=f'''# Pre-action matched full-horizon pilot

## Result

{status['FINAL_STATUS']}. Full collection remains **NO**. This is a task0 execution/label diagnostic, not a trained feasibility model or AF-vs-fixed benchmark. The table below reports the **frozen final-contact operational criterion only**, NOT geometry-validated full-task ground truth. All labels remain quarantined.

| Context | Setpoint | Repeat | Full-task | Lift | Place | Dropped | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

Within-context contact-criterion variation observed: {boundary}. Lowest passing tested setpoint under that criterion: {status['MIN_SUCCESSFUL_TESTED_FORCE']}. Nine unique context-force cells on one root; 4N has a deterministic replay, not an independent success-rate replicate. This is **not a qualified force-success boundary**: geometric/visual checks confirmed HIGH-4N is outside the basket despite label=1. No raw labels are rewritten after seeing the outcomes.

## Confirmed downstream placement defect

Three additional diagnostic-only replays (HIGH-3N, HIGH-4N, LOW-5N) match their complete previous action/state trajectories exactly and save final basket/object poses and camera images. The basket shifts by {', '.join(str(round(r['basket_xy_displacement_m']*100,3))+' cm' for r in geometry_cases)} in these cases. The branch runner constructs its basket target only once before the trajectory, so subsequent placement is aimed at the old location. All three inspected objects are outside the basket in the saved images; HIGH-4N merely contacts the outside wall and receives a false positive under the contact criterion.

The initial suspicion that LOW-5N might be inside the basket with missing force readback was rejected: it is near the **old** basket location. The final basket pose was essential. Geometry uses final poses plus original USD assets; outer-bounds checks are a diagnostic, not a newly fitted containment label. See PLACEMENT_ROOT_CAUSE_AUDIT.json and ../placement_geometry_diagnostic/GEOMETRY_AUDIT.json. Other six unique cells were not visually adjudicated and are not silently declared physically successful.

[HIGH-4N falsely labelled successful](../placement_geometry_diagnostic/branches/high/F4_R0/FINAL_agentview_cam.png), [HIGH-3N](../placement_geometry_diagnostic/branches/high/F3_R0/FINAL_agentview_cam.png), [LOW-5N](../placement_geometry_diagnostic/branches/low/F5_R0/FINAL_agentview_cam.png).

## Fixed causal defects

1. The post-probe gripper handoff used measured joint position .0222525354 m instead of the last command .0028 m, causing a +.0194525355 m command jump and complete contact loss in the initial HIGH hold20 audit. Preserving the real last command recovers contact (HIGH mean left/right normal 2.236725/3.555427 N). No D_CLOSED hardcode or force scaling.
2. Original task success reset the environment at transit step156. All 12 old pilot “success” rows are quarantined as EARLY_TERMINATION_DIAGNOSTIC_ONLY. The new wrapper defers **only success termination** so all prescribed 350 steps, including release/settle, execute. Drop/time-out still stop immediately. Original raw success terms, geometry thresholds and basket-contact threshold are retained.

Place now requires the complete phase schedule, the original goal at the final state and final basket contact >.05 N. Full-task remains lift AND place AND NOT dropped. This is an explicit evaluation-time correction, not a claim that termination semantics were unchanged. Release is scheduled, but no new unregistered stability-duration threshold was invented. Final finger forces and object speed are reported in the CSV.

The environment's `object_1_dropped` term means object height below -.05m; it does not classify every loss of grasp or floor landing as a drop. A floor landing can therefore correctly appear as place=0, dropped=0. Raw goal flags are insufficient on their own (some remain true with zero final basket contact); the explicit final-contact check is independently required. This inherited goal/contact criterion is not a geometric proof of full containment inside the basket.

## Pre-action and execution evidence

All candidates share immutable reference features and posterior, extracted before any candidate action. Across F=3/4/5, only feature17=F/8 changes for a fixed mu member: maximum non-candidate feature difference {feature_max}. Every worker verifies the full action prefix, 58D physical evidence and exposed decision state exactly before executing candidate F.

For all new branches, the first156 steps match the earlier pilot state/action traces exactly. Thus the correction does not alter pre-termination physical dynamics; it exposes the previously unexecuted place/release/settle outcome.

Literal snapshot restoration remains **FAIL**: exposed states match initially but same-action trajectories differ after stepping, so hidden PhysX state is not claimed restored. The validated alternative is **fresh-process exact prefix reconstruction**; same-force full-horizon replay gate: {gate}. Raw state/action/terminal comparisons are in FULL_TASK_RECONSTRUCTION_PARITY.json. This is not described as one simulator snapshot perfectly restored without replay. Every physical probe replay is counted: {len(starts)} attempts, {completed} complete prefixes; this includes failed harness guards and the earlier diagnostic runs. No probe replay observation is used to replace the reference planner input.

## Probability and force semantics

Current HIGH/MID/LOW mean supports: {status['CURRENT_POSTERIOR_HIGH_MID_LOW']}, outward probe steps {status['CURRENT_PROBE_OUTWARD_STEPS_HIGH_MID_LOW']}. Input discrimination is demonstrated for these cases; strict friction ordering is not a forced model-selection condition. The unchanged planner still uses three member means and omits test-average {100*status['POSTERIOR_TEST_MEAN_VARIANCE_FRACTION_OMITTED']:.2f}% of predictive variance. Full continuous posterior deployment is **not validated**; sigma integration and positive-support treatment must be explicitly resolved, not silently truncated.

Setpoint is not exact instantaneous Newton force. PHASE_FORCE_DIAGNOSTICS.csv separates requested F, existing effective controller targets, per-finger true normal forces and actual joint positions. Source filenames/legacy CSV columns containing “aperture” may be per-finger values; the derived columns label this explicitly. No controller gain, utility, force support, probe, posterior interface or model weights changed this turn.

These corrections are implemented and tested in the isolated pilot runner. The old production entrypoint/checkpoint manifest has not been silently migrated; execution-parity claims apply to the tested new contract only. ARTIFACT_PROVENANCE.json verifies unchanged belief checkpoint hashes and derived audit artifacts.

## Blockers and next gate

{' ; '.join(blockers)}. No full720 collection, no old720 labels, no feasibility training, no calibration/recipe search, no AF-vs-fixed physical benchmark. If force outcomes do not vary, inspect existing phase/contact/command traces before requesting any expansion of context or execution scope.

Source provenance is frozen in PROTOCOL.json and BRANCH_CODE_PROVENANCE.json. LABEL_AUDIT.json independently reconstructs labels from raw pre-reset observations and phase counts. TEST_RESULTS.txt records the regression suite. READ_ONLY_AUDIT.ipynb is the rerunnable analytical companion.
'''
    (OUT/'FULL_HORIZON_MATCHED_PILOT_REPORT.md').write_text(report)
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
