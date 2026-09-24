"""Audit matched pilot outcomes and independently reconcile terminal labels."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import run_matched_history_reconstruction_20260905 as run
from activeforcing_execution_snapshot import max_difference

ROOT=run.ROOT;HISTORY=run.HISTORY_ROOT;OUT=HISTORY/'guard_fixed';PARENT=HISTORY.parent
run.OUT=OUT


def read(path):return json.loads(Path(path).read_text())
def write(name,value):run.write(OUT/name,value)
def csvout(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    protocol=run.protocol()
    if not (OUT/'PILOT_COMPLETE.json').exists():raise RuntimeError('Pilot not complete; cannot synthesize missing outcomes')
    initial=read(PARENT/'high/RESTORE_GATE.json')
    continuity=read(PARENT/'command_continuity/high/RESTORE_GATE.json')
    integration=read(PARENT/'POSTERIOR_INTEGRATION_DIAGNOSTIC.json')
    results=[];label_audit=[];trace_by={};parity=[]
    for band in ('high','mid','low'):
        reference=run.reference_dir(band)
        verification=read(HISTORY/'verification'/band/'RESULT.json')
        assert verification['passed']
        for force,repeat in [(3.,0),(4.,0),(5.,0),(4.,1)]:
            job=run.job_dir('branch',band,force,repeat)
            r=read(job/'RESULT.json');trace=read(job/'BRANCH_TRACE.json');trace_by[(band,force,repeat)]=trace
            frozen=read(job/'PREFIX_RECONSTRUCTION_GATE.json');features=read(job/'FEATURE_GATE.json')
            assert frozen['passed'] and features['passed'] and r['decision_snapshot_sha256']==run.sha256(reference/'DECISION_STATE.pt')
            assert r['reference_feature_sha256']==run.sha256(reference/'FROZEN_PREACTION_FEATURES.npy')
            with (reference/'RAW_PROBE.csv').open() as f:probe=list(csv.DictReader(f))
            # Independent lift computation uses saved pre-action object state.
            import torch
            snapshot=torch.load(reference/'DECISION_STATE.pt',map_location='cpu',weights_only=False)
            obj=probe[-1]['object_id'];z0=float(snapshot['state']['rigid_object'][obj]['root_pose'][0,2])
            lift=int(any(t['object_position_m'][2]-z0>=.03 for t in trace))
            place=int(any(t['basket_contact_force_N']>.05 for t in trace))
            dropped=int(any(t['terminations'].get('object_1_dropped',False) for t in trace))
            y=int(lift and place and not dropped)
            assert (lift,place,dropped,y)==tuple(r[k] for k in ('lift_success','place_success','dropped','full_task_success_y'))
            action=np.array([t['action'][0] for t in trace])
            tracking=action[:,9]+action[:,12]>0
            forces=np.array([t['normal_force_N'] for t in trace]);bilateral=2*np.min(forces,axis=1)
            kept=bilateral[tracking]
            summary={'context':band.upper(),'candidate_F':force,'repeat':repeat,
                **{k:r[k] for k in ('full_task_success_y','lift_success','place_success','dropped')},
                'steps':len(trace),'pre_reset_terminal_captures':sum(t['captured_before_auto_reset'] for t in trace),
                'environment_success':int(any(t['terminations'].get('success',False) for t in trace)),
                'max_object_lift_m':max(t['object_position_m'][2]-z0 for t in trace),
                'max_basket_contact_N':max(t['basket_contact_force_N'] for t in trace),
                'tracking_mean_bilateral_normal_N':float(kept.mean()) if len(kept) else None,
                'tracking_contact_loss_fraction':float((kept<=.2).mean()) if len(kept) else None,
                'native_runner_label':r['native_runner_label'],'native_label_agrees':r['native_label_agrees'],
                'reference_snapshot_sha256':r['decision_snapshot_sha256'],
                'reference_feature_sha256':r['reference_feature_sha256']}
            results.append(summary);label_audit.append({'context':band,'force':force,'repeat':repeat,
                'independent_label_matches':True,'label_source':'CURRENT_MATCHED_FULL_TASK_BRANCH',
                'post_action_features_used':False,'old720_labels_used':False})
        a=trace_by[(band,4.,0)];b=trace_by[(band,4.,1)]
        compare=run.base.compare_trace(a,b)
        actions_error=max_difference([t['action'] for t in a],[t['action'] for t in b])
        outcomes=[next(r for r in results if r['context']==band.upper() and r['candidate_F']==4 and r['repeat']==k) for k in (0,1)]
        same=all(outcomes[0][k]==outcomes[1][k] for k in ('full_task_success_y','lift_success','place_success','dropped'))
        passed=bool(compare['passed'] and compare.get('episode_step_equal') and actions_error==0 and same)
        parity.append({'context':band,'passed':passed,'trace_comparison':compare,'actions_max_difference':actions_error,'same_outcome':same})
    csvout('EMPIRICAL_FORCE_SUCCESS_TABLE.csv',results);write('LABEL_AUDIT.json',label_audit)
    write('FULL_TASK_RECONSTRUCTION_PARITY.json',parity)
    full_gate=all(p['passed'] for p in parity)
    certificates=[]
    for r in results:
        band=r['context'].lower();job=run.job_dir('branch',band,r['candidate_F'],r['repeat'])
        raw_result=read(job/'RESULT.json')
        certificates.append({**raw_result,
            'raw_result_path':str(job/'RESULT.json'),'raw_result_sha256':run.sha256(job/'RESULT.json'),
            'restore_execution_verified':next(x['passed'] for x in parity if x['context']==band),
            'restoration_method':'FRESH_PROCESS_EXACT_PREFIX_RECONSTRUCTION',
            'verification_certificate':str(OUT/'FULL_TASK_RECONSTRUCTION_PARITY.json'),
            'label_admitted_for_training':False,
            'label_quarantine_reason':'Pilot-only evidence; formal posterior and dataset contract unresolved'})
    write('PILOT_LABEL_MANIFEST.json',{'labels':certificates,'formal_training_dataset_eligible':False,
        'old720_labels_used':False,'post_action_state_used_as_input':False})
    representative=[r for r in results if r['repeat']==0]
    boundary={band:len({r['full_task_success_y'] for r in representative if r['context']==band})==2 for band in ('HIGH','MID','LOW')}
    minima={band:min([r['candidate_F'] for r in representative if r['context']==band and r['full_task_success_y']] or [float('inf')]) for band in ('HIGH','MID','LOW')}
    finite_minima={band:(None if not np.isfinite(v) else v) for band,v in minima.items()}
    context_boundary=len(set(minima.values()))>1
    current={b:read(run.reference_dir(b)/'POSTERIOR.json') for b in ('high','mid','low')}
    probes={b:read(run.reference_dir(b)/'PROBE_RESULT.json') for b in ('high','mid','low')}
    n_history=len(list(HISTORY.glob('references/*/STARTED.json')))+len(list(HISTORY.glob('verification/*/STARTED.json')))+len(list(OUT.glob('branches/*/*/STARTED.json')))
    failed_prefix_attempts=len(list(HISTORY.glob('branches/*/*/STARTED.json')))
    failed_prefix_completed=len(list(HISTORY.glob('branches/*/*/RAW_PROBE.csv')))
    status={'FINAL_STATUS':'MATCHED_PILOT_COMPLETE' if full_gate else 'PILOT_COMPLETE_BUT_RECONSTRUCTION_GATE_FAILED',
        'PROBE_CONTRACT_FIXED':True,'POSTERIOR_INPUT_CONTRACT_FIXED':True,
        'CURRENT_PROBE_OUTWARD_STEPS_HIGH_MID_LOW':[probes[b]['outward_steps'] for b in ('high','mid','low')],
        'CURRENT_POSTERIOR_MEAN_HIGH_MID_LOW':[current[b]['mean'] for b in ('high','mid','low')],
        'HANDOFF_COMMAND_CONTINUITY_FIXED':True,'DECISION_STATE_DEFINED':True,
        'TEMPORAL_FEATURE_PARITY':True,'CANDIDATE_PREACTION_STATE_EQUALITY':True,
        'DIRECT_SNAPSHOT_RESTORE_VALIDATED':False,
        'EXECUTION_CONTRACT_PARITY':full_gate,
        'RESTORE_METHOD':'FRESH_PROCESS_EXACT_PREFIX_RECONSTRUCTION',
        'PILOT_PHYSICS_EXECUTED':True,'PILOT_CONTEXTS':['HIGH','MID','LOW'],'PILOT_CANDIDATE_FORCES':[3.,4.,5.],
        'PILOT_BRANCH_COUNT':len(results),'FULL_TASK_REPEAT_FORCE':4.,'UNIQUE_CONTEXT_FORCE_CELLS':9,
        'INITIAL_RESTORE_DIAGNOSTIC_PHYSICAL_PREFIXES':2,'HISTORY_REFERENCE_RECONSTRUCTION_PREFIXES':n_history,
        'TOTAL_PHYSICAL_PREFIXES_THIS_TURN':2+n_history+failed_prefix_completed,
        'TOTAL_PHYSICAL_PREFIX_ATTEMPTS_THIS_TURN':2+n_history+failed_prefix_attempts,
        'HARNESS_GUARD_FAILURE_PREFIXES_COMPLETED':failed_prefix_completed,
        'HARNESS_GUARD_FAILURE_PREFIXES_INTERRUPTED':failed_prefix_attempts-failed_prefix_completed,
        'FORCE_BOUNDARY_EXISTS':bool(full_gate and any(boundary.values())),
        'WITHIN_CONTEXT_FORCE_BOUNDARY':boundary,'MIN_SUCCESSFUL_TESTED_SETPOINT':finite_minima,
        'CONTEXT_DEPENDENT_FORCE_BOUNDARY':bool(full_gate and context_boundary),
        'FORMAL_CONTINUOUS_POSTERIOR_VALIDATED':False,'READY_FOR_FULL_DATA_COLLECTION':False,
        'PHYSICAL_BELIEF_RETRAINED_THIS_TURN':False,'FEASIBILITY_RETRAINED':False,
        'UTILITY_CHANGED':False,'CONTROLLER_CHANGED':False,'PROBE_CHANGED':False,
        'POSTERIOR_INTERFACE_CHANGED':False,'RUNTIME_FORCE_RESCALED':False,
        'POST_ACTION_INPUT_USED':False,'OLD720_LABELS_USED':False,
        'NATIVE_RUNNER_LABEL_DISAGREEMENT_COUNT':sum(not r['native_label_agrees'] for r in results),
        'POSTERIOR_VARIANCE_OMITTED_TEST_MEAN':integration['summary']['TEST']['variance_fraction_omitted_by_member_mean_only_support'],
        'CURRENT_BLOCKERS':['formal posterior integration/support contract not yet corrected and validated',
            'task0 pilot covers one root and three contexts; tasks1/5/6 unvalidated']+
            ([] if full_gate else ['same-force full-task reconstruction mismatch'])+
            ([] if any(boundary.values()) else ['no within-context force-success boundary observed on this pilot']),
        'FULL_COLLECTION_STARTED':False,'PHYSICAL_PLANNER_BENCHMARK_EXECUTED':False}
    write('FINAL_STATUS.json',status)
    tests=subprocess.run([sys.executable,'-m','unittest','-v','test_activeforcing_command_handoff.py',
        'test_activeforcing_execution_snapshot.py','test_activeforcing_decision_state.py',
        'test_activeforcing_probe_friction_contract.py','test_current_contract_belief.py'],cwd=ROOT,capture_output=True,text=True)
    (OUT/'TEST_RESULTS.txt').write_text(tests.stdout+tests.stderr)
    if tests.returncode:raise RuntimeError('Regression suite failed')
    table='\n'.join(f"| {r['context']} | {r['candidate_F']:g} | {r['repeat']} | {r['full_task_success_y']} | {r['lift_success']} | {r['place_success']} | {r['dropped']} |" for r in results)
    report=f'''# Current-contract matched feasibility pilot

## Result

Status: {status['FINAL_STATUS']}. Full-data collection remains **NO**. This is an empirical task0 pilot, not an ActiveForcing selector benchmark, held-out feasibility model, or final continuous-posterior validation.

| Context | Candidate setpoint | Repeat | Full-task | Lift | Place | Dropped |
|---|---:|---:|---:|---:|---:|---:|
{table}

Within-context mixed force outcomes: {boundary}. Minimum successful tested setpoint: {finite_minima}; null means no tested force succeeded, not an estimated threshold beyond 5. Every 4N cell has two repeats; 3N/5N have one. Deterministic replays do not provide independent success-rate estimates. One root, three friction contexts, nine unique context-force cells.

## Causal fixes and failed checks retained

The old handoff replaced a last probe action command of .0028 m by measured joint position .0222525354 m, a .0194525355 m command jump. All 20 subsequent HIGH hold steps lost bilateral contact. Preserving the real last command kept mean left/right normal patch forces 2.236725/3.555427 N. This is a command-continuity fix, not force rescaling or a D_CLOSED hardcode. The controller source and gains did not change.

Direct exposed-state restore remains FAIL. Scene, controller, target, sensor/cache and environment values match exactly before execution, but trajectories differ after stepping. There is no claim that hidden PhysX solver state was restored. Both failed diagnostic runs and their sources are retained.

The replacement restoration mechanism is explicitly **fresh-process exact pre-action history reconstruction**. Each replay must match all reference probe actions, all 58 observable features, and exposed decision state before executing candidate F. The reference features and posterior are immutable and reused. No candidate-specific post-action state enters model inputs. Three hold20 checks pass, and the full-task 4N repeat gate is {full_gate}. All {n_history} successful-stage history construction/replay prefixes and the 2 earlier direct-restore diagnostic prefixes are counted as real physical work. In addition, {failed_prefix_completed} completed prefixes were stopped before any candidate by a Python3.10/3.11 ast.unparse text-hash incompatibility and {failed_prefix_attempts-failed_prefix_completed} prefixes were interrupted. The guard now verifies original source and normalized semantic AST hashes across both interpreters; archived failures are not replaced or used as labels. It is not described as a single physical probe with literal scene-snapshot restoration.

## Label provenance and execution

New labels only: lift is any object rise at least .03 m from the reference decision state; place follows the unchanged current P5 criterion of basket-contact force above .05 N; dropped is any object_1_dropped terminal event. Full-task = lift AND place AND NOT dropped. Place here is the current runner's basket-contact criterion, not an additional stable-placement/release-duration criterion. Environment success is reported separately in the CSV. Terminal values are captured before auto-reset; no reset state is used as the preceding outcome. Native runner label disagreements: {status['NATIVE_RUNNER_LABEL_DISAGREEMENT_COUNT']}.

The existing branch phase schedule, servo, force range [3,5], arm path and release actions are unchanged. An AST-checked one-node adaptation replaces only gripper initialization by the checked probe command. Source code and hashes are in BRANCH_CODE_PROVENANCE.json and PROTOCOL.json. The first candidate action is executed only after feature extraction. Each branch starts from its reconstructed pre-action state, never from a preceding force branch.

## Posterior remains a separate blocker

The current input contract and mean discrimination pass. However, the unchanged three-member-mean support omits test-average {100*status['POSTERIOR_VARIANCE_OMITTED_TEST_MEAN']:.2f}% of the trained Gaussian mixture's predictive variance. Incorporating sigma also introduces negative-mu mass under the untruncated Gaussian model. No support truncation, calibration or deployment-interface change was silently applied. This pilot does not validate formal posterior integration.

## Limits and next gate

Remaining blockers: {'; '.join(status['CURRENT_BLOCKERS'])}. No new feasibility model, full dataset, utility change, force scaling or AF-vs-fixed planner benchmark was run. The mixed-unit historical audit key joint_position_m contains seven arm angles in radians and two finger positions in metres; equality comparisons do not reinterpret those units. Velocity vectors contain linear and angular components. The report does not use these mixed keys as scalar physical displacement claims.

Artifacts: EMPIRICAL_FORCE_SUCCESS_TABLE.csv, LABEL_AUDIT.json, FULL_TASK_RECONSTRUCTION_PARITY.json, FINAL_STATUS.json, PROTOCOL.json, BRANCH_CODE_PROVENANCE.json, TEST_RESULTS.txt, READ_ONLY_AUDIT.ipynb. Per-job raw probe/patch data, snapshots, frozen features, action and terminal traces are retained.
'''
    (OUT/'MATCHED_PILOT_REPORT.md').write_text(report)
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
