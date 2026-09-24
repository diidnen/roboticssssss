"""Freeze the completed three-probe diagnosis and its scope boundaries."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('/home/exouser/FORTE')
OUT=ROOT/'analysis/results/probe_only_instrumented_20260905'

def main():
    results=json.loads((OUT/'PROCESS_RESULTS.json').read_text())
    assert len(results)==3 and all(r['returncode']==0 for r in results)
    freeze=json.loads((OUT/'LAUNCH_MANIFEST.json').read_text())
    for p,h in freeze['frozen_sources'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    assert hashlib.sha256((ROOT/'probe_only_instrumented_20260905.py').read_bytes()).hexdigest()==freeze['runner_sha256']
    summary=pd.read_csv(OUT/'PROBE_ONLY_SUMMARY.csv')
    selection=pd.read_csv(OUT/'NEW_PROBE_OFFLINE_SELECTION.csv')
    assert len(summary)==3 and (summary.instrumentation_errors==0).all()
    assert (summary.old_smoke_raw46_max_error==0).all()
    assert (summary.max_non_object_contact_world_error==0).all()
    oldp=ROOT/'analysis/results/e1_verified_recovery_revision_20260905/CHAINED_PARITY.json'
    old=json.loads(oldp.read_text());assert old['passed']
    e1source=ROOT/'analysis/results/ACTIVEFORCING_E1_FINAL_UTILITY_MAIN_TABLE_20260902_052942/E1_SELECTED_EPISODES.csv'
    history=pd.read_csv(e1source)
    af=history[history.method=='ActiveForcing-1Q Utility']
    status={'FINAL_STATUS':'HISTORICAL_INFERENCE_REPRODUCED_CURRENT_TRANSFER_UNRESOLVED','authorized_physical_probes_completed':3,
        'physical_task_branches':0,'physics_processes_exited':True,'existing_policy_servers_untouched':True,
        'E1_exact_selected_replay_cases':144,'E1_posterior_ablation_replay_cases':144,'E1_mean_selected_setpoint':old['point_mean'],
        'E1_historical_result_source':str(e1source),'three_new_probes_match_previous_smoke_raw46_exactly':True,
        'object_static_and_dynamic_friction_readback_correct':True,'non_target_contact_contamination_detected':False,
        'force_coordinate_rotation_error_max':float(summary.max_local_coordinate_difference.max()),
        'all_three_probes_stop_after_one_outward_step':bool((summary.outward_steps==1).all()),
        'current_context_sensitive_selection_recovered':False,'final_18_branch_rerun_allowed':False,
        'controller_changed':False,'probe_parameters_changed':False,'utility_changed':False,'training_performed':False,
        'unresolved':'Physical probe statistics differ from historical training. Clean feasibility state features were trained on post-action branch row 1, not current predecision observation.',
        'next_step_requires_scope_change':'A clean feasibility rebuild using the exact E1 predecision feature contract is a possible next diagnostic step, but training remains unapproved and would not by itself certify current physical-belief transfer.'}
    (OUT/'FINAL_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    artifacts=[]
    for band in ['high','mid','low']:
        for filename in ['RAW_PROBE.csv','CONTACT_READBACK.json','RESULT.json','POST_PROBE_STATE.pt']:
            p=OUT/'jobs'/band/filename
            artifacts.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
    (OUT/'PROBE_ARTIFACT_HASHES.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    lines=['# Authorized probe-only diagnosis — final record','',
        '## Outcome','',
        'The user-authorized high/mid/low probes ran once each in three fresh Isaac processes. All completed, all exited; zero task branches and zero training runs. Existing policy services were neither used for a rollout nor modified.',
        'Historical E1 inference remains exactly reproduced (144 primary + 144 posterior selections). This is distinct from reproducing prospective physical success under a different runtime.',
        '', '## Exact measurements','',
        '| Context | Applied static/dynamic μ | Outward steps | First outward ft/fn | Current hold mean ft | Historical hold mean ft |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.band} | {r.readback_static_mu:.10f} / {r.readback_dynamic_mu:.10f} | {r.outward_steps} | {r.first_outward_rho:.9f} | {r.hold_mean_ft:.9f} | {r.historical_hold_mean_ft:.9f} |')
    lines+=['','All three stop with `shear_ratio_cap`; frozen cap=.08. Object mass=.10000000149 kg in all three contexts.',
        'New raw46 features equal the corresponding previous-smoke repeat1 inputs exactly, maximum error=0. This verifies instrumentation did not alter the observed probe trajectory; it does not validate the transferred estimator.',
        'All-contact world forces and object-filtered world forces are identical across the observed probe steps: maximum difference=0. The gripper-local policy forces agree with independent object-filtered local forces to approximately 2e-6 N. Contact override and authoritative inner force loop remain off.',
        'Therefore missing object-friction assignment, non-target contact contamination and a large force-coordinate conversion error are excluded for these runs. Readback does not by itself reconstruct the entire historical contact model or explain the changed physical force statistics.',
        '', '## Offline inference on the new probes (diagnostic only)','',
        '| Context | E1 primary point | E1 posterior ablation | Clean lift+hold | Existing clean full-task |',
        '|---|---:|---:|---:|---:|']
    for band in ['high','mid','low']:
        vals=selection[selection.band==band].set_index('model').selected_setpoint
        lines.append(f'| {band} | {vals["E1_PRIMARY_POINT"]:.2f} | {vals["E1_POSTERIOR"]:.2f} | {vals["CURRENT_CLEAN_LIFTHOLD"]:.2f} | {vals["EXISTING_CLEAN_FULLTASK"]:.2f} |')
    lines+=['','These are not authorized physical commands. The current probe distribution still does not satisfy the historical estimator-transfer gate; variable numbers alone do not establish reliable adaptation.',
        '', '## What is fixed versus unresolved','',
        'Recovered and checked: original checkpoint/preprocessing/selector chain; exact historical variable selections; current feature adapter; honest uncertainty reporting; input-support refusal gate. Controller, utility and probe parameters are unchanged.',
        'Unresolved: the current probe produces substantially different tangential loading than historical training. The current clean feasibility models also use a different target/data/observation contract: their condition was fitted to branch row1 after applying a candidate action, while deployment observes the predecision post-probe state. Label cleanliness does not remove that mismatch.',
        'A monotonic setpoint-to-force mapping alone is insufficient to prove that a historical numerical success-probability curve transfers unchanged to the new execution state. No input rescaling, threshold tuning, synthetic probe padding or oracle substitution has been deployed.',
        '', '## Remaining boundary','',
        'The three-probe exception is now fully consumed. The original 18-branch gates remain closed. There is no remaining unrun probe in this authorized diagnostic batch.',
        'A possible next offline step is rebuilding clean feasibility with the exact E1 predecision feature contract; this requires explicit permission to train and would address only the feasibility-input mismatch, not automatically the physical-belief transfer. It is not a guarantee of recovering good current physical results. No such training has started.',
        '', '## Artifacts','',
        '`PROBE_ONLY_SUMMARY.csv`, `PER_STEP_CONTACT_DIAGNOSIS.csv`, `NEW_PROBE_POSTERIORS.json`, `NEW_PROBE_OFFLINE_SELECTION.csv`, `NEW_PROBE_OFFLINE_CURVES.csv`, per-context raw/JSON/snapshots, and SHA256 inventory in `PROBE_ARTIFACT_HASHES.json`.',
        'Earlier historical replay: `/home/exouser/FORTE/analysis/results/e1_verified_recovery_revision_20260905/`.',
        'Training observation-time and raw-input audits: `/home/exouser/FORTE/analysis/results/physical_belief_transfer_root_cause_20260905/`.']
    (OUT/'PROBE_ONLY_DIAGNOSIS_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(status,indent=2))

if __name__=='__main__':main()
