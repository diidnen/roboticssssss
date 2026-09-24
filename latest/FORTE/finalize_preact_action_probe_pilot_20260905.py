"""Finalize current pre-action diagnostic evidence; NEVER launches physics."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home/exouser/FORTE')
OUT=ROOT/'analysis/results/preaction_decision_contract_20260905'
sys.path.insert(0,str(ROOT))
from activeforcing_decision_state import collection_gate


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value): (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    cases={b:json.loads((OUT/'corrected_friction_stop'/b/'RESULT.json').read_text()) for b in ['high','mid','low']}
    contracts={b:json.loads((OUT/'corrected_friction_stop'/b/'EXECUTION_CONTRACT.json').read_text()) for b in cases}
    tested_source=OUT/'TESTED_CURRENT_PROBE_SOURCE.py'
    assert all(c['p4_sha256']==sha(tested_source) for c in contracts.values())
    assert all(r['task_branches']==0 for r in cases.values())
    assert all(r['decision_feature_parity']['MAX_FEATURE_DIFF']==0 for r in cases.values())
    # One root / three cases can diagnose, not validate belief generalization.
    gate=collection_gate(probe_validated=True,posterior_validated=False,decision_parity=True,
        execution_restore_validated=False,empirical_boundary_validated=False)
    gate['probe_validated_scope']='stop-signal mechanism checked on three task0 contexts, not population validation'
    write('COLLECTION_GATE.json',gate)
    test=subprocess.run([sys.executable,'-m','unittest','-v','test_activeforcing_decision_state.py',
                         'test_activeforcing_probe_friction_contract.py'],cwd=ROOT,capture_output=True,text=True)
    (OUT/'TEST_RESULTS.txt').write_text(test.stdout+test.stderr)
    assert test.returncode==0
    status={
        'FINAL_STATUS':'STOP_SIGNAL_BUG_REPAIRED_AND_PHYSICALLY_TESTED; BELIEF_TRANSFER_BLOCKED; NO_FEASIBILITY_COLLECTION',
        'PROBE_CONTRACT_FIXED':'PARTIAL: corrected stop signal tested; full probe/posterior contract NOT qualified',
        'POSTERIOR_INPUT_CONTRACT_FIXED':'NO: causal inputs verified, frozen estimator transfer still invalid',
        'CURRENT_PROBE_STEPS_HIGH_MID_LOW':[cases[b]['outward_steps'] for b in cases],
        'CURRENT_PROBE_TOTAL_ROWS_HIGH_MID_LOW':[cases[b]['steps'] for b in cases],
        'CURRENT_POSTERIOR_HIGH_MID_LOW':[cases[b]['posterior_mean'] for b in cases],
        'CURRENT_POSTERIOR_SUPPORT':{b:cases[b]['posterior_support'] for b in cases},
        'DECISION_STATE_DEFINED':True,'TEMPORAL_FEATURE_PARITY':True,
        'EXECUTION_CONTRACT_PARITY':'NOT_YET_VALIDATED: no matched branch; PhysX warm-start/controller restoration not certified',
        'CANDIDATE_PREACTION_STATE_EQUALITY':True,'MAX_FEATURE_DIFF':0.,
        'PILOT_PHYSICS_EXECUTED':False,'PILOT_CONTEXTS':0,'PILOT_CANDIDATE_FORCES':[],
        'PLANNED_PILOT_CONTEXTS':list(cases),'PLANNED_PILOT_CANDIDATE_FORCES':[3.,4.,5.],
        'PROBE_ONLY_PHYSICS_EXECUTED':True,'PROBE_ONLY_RUNS':6,
        'FORCE_BOUNDARY_EXISTS':'UNKNOWN_NOT_TESTED','CONTEXT_DEPENDENT_FORCE_BOUNDARY':'UNKNOWN_NOT_TESTED',
        'READY_FOR_FULL_DATA_COLLECTION':False,'NEW_FEASIBILITY_TRAINING_ROWS':0,
        'OLD720_LABELS_USED':False,'PHYSICAL_BELIEF_WEIGHTS_CHANGED':False,'PHYSICAL_BELIEF_INTERFACE_CHANGED':False,
        'CONTROLLER_CHANGED':False,'UTILITY_CHANGED':False,'FORCE_DOMAIN_CHANGED':False,
        'RUNTIME_FORCE_RESCALED':False,'TESTS_PASSED':17,
        'BLOCKER':'New physical probe responses differ, but frozen E6 posterior remains non-discriminative; root-held-out current-contract belief validation is missing',
        'NEXT_ACTION':'Proposed root-grouped probe-only corpus and belief retraining; user direction requested before expanding scope. No task branches or feasibility labels yet.'}
    write('FINAL_STATUS.json',status)
    empirical=[]
    for b in cases:
        for f in [3.,4.,5.]:
            empirical.append(dict(context=b,candidate_F=f,full_task_success=None,lift_success=None,
                place_success=None,dropped=None,status='NOT_EXECUTED_POSTERIOR_GATE_CLOSED'))
    with (OUT/'EMPIRICAL_FORCE_SUCCESS_TABLE.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(empirical[0]));writer.writeheader();writer.writerows(empirical)
    first_step=[]
    for b in cases:
        rb=json.loads((OUT/'corrected_friction_stop'/b/'CONTACT_PATCH_READBACK.json').read_text())
        first=next(x for x in rb if x['termination_signal']['phase']=='probe_out')
        first_step.append((b,first['termination_signal']['incremental_directional_friction_ratio']))
    table='\n'.join(f"| {b.upper()} | {cases[b]['outward_steps']} | {cases[b]['steps']} | {cases[b]['posterior_mean']:.9f} | {dict(first_step)[b]:.6f} | {cases[b]['probe_record']['stop_trigger']} |" for b in cases)
    report=f'''# Current pre-action contract: probe fix and gated pilot status

## Result

The normal-projection-as-shear stop bug was isolated and a corrected, opt-in current probe was physically tested. HIGH/MID/LOW naturally took 2/5/8 outward steps. The frozen E6 belief still does not discriminate these three contexts reliably. No full-task pilot or new feasibility training was started.

| Context | Outward steps | Total steps | Frozen E6 mean μ | First-step true-friction increment / normal | Stop reason |
|---|---:|---:|---:|---:|---|
{table}

The first-step comparison uses identical 0.2-mm commanded outward displacements. It supports different probe responses on these three cases, NOT a validated population friction estimator or a force-success boundary.

## Causal diagnosis

1. IsaacLab ContactSensor `net_forces_w` and `force_matrix_w` contain **normal contact forces only**, not the friction component. Source: `/media/volume/newdata/exouser/tabero/IsaacLab23/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_data.py`, lines 53–87.
2. Historical P4 computes its nominal shear from the local XY projection of these normal-force vectors. Contact-surface tilt can therefore trigger its cap without probe-induced friction. Current high contact-patch normals have world vertical components approximately 0.271/0.277 (about 16°), explaining the projected component. This does NOT identify why the historical/current physical contact geometry differs.
3. We read `get_contact_data` and `get_friction_data` separately, using physics dt=1/60 s and correct pair start/count ranges. Normal patch sums reproduce the old object-filtered sensor within 2.4e-7 N. The independent readback runs reproduce previous 46-D policy inputs exactly on all three contexts; sensor instrumentation did not change those traces.
4. The corrected stop signal is `abs(dot(sum(friction_world)-last10hold_baseline, probe_direction_world)) / (2*min(sum(normal_patch_magnitudes_per_finger)))`. No sensor value is scaled or rewritten. Original cap=.08, impulse budget=.012, .2-mm increment, 2-mm displacement cap, contact-loss and relative-normal guards remain. The normalization denominator is measured normal force, not a train-distribution adjustment.
5. The unchanged E6 raw adapter does not ingest the newly recorded true-friction patch signal. The posterior already clusters at roughly .713/.669/.698 at the end of hold, before outward probing. Correct stopping alone leaves final means .712/.669/.692. These are diagnostic old-model predictions, not validated new physical posteriors. Prefix outputs during approach/close are also diagnostic only; the old estimator was not validated for every intermediate phase.

## Implemented pre-action contract

`probe → return/hold → DECISION_STATE → save snapshot → extract X → hypothetical F`

Six freshly captured decisions have zero candidate actions before extraction. Raw observation, scene state and snapshot joints/velocity are aligned. Commands are generated as the upcoming eight branch-hold commands at decision time; no branch row supplies a state or command prefix.

`activeforcing_decision_state.py` uses the canonical feature builder for runtime and matched training rows, protects snapshot integrity, rejects extraction after candidate execution, and rejects historical/ambiguous label sources. For 3/4/5 and all three posterior supports, only feature index 17 (F/8) may change when F changes: maximum difference outside that explicit condition is zero. Mu marginalization remains external.

17 tests passed. Full simulator/solver/controller restoration parity is **not** established by a scene hash. Snapshots include additional controller tensors and RNG for investigation; they are not advertised as complete warm-start restore certificates. This remains a separate branch gate.

## Empirical table and collection gate

`EMPIRICAL_FORCE_SUCCESS_TABLE.csv` contains nine planned rows, all outcomes blank and explicitly NOT_EXECUTED. Unknown is not failure. No current force-success boundary has been measured in this turn. Full-task labels, when authorized and qualified, must be `lift_success AND place_success AND NOT dropped` and must come from a new same-decision branch, never old720.

READY_FOR_FULL_DATA_COLLECTION = NO.

The next necessary work is current-contract physical-belief development/validation using a root-grouped probe-only corpus. Three debugging contexts are not sufficient to train or certify it; no three-point mapping, oracle μ substitution or calibration trick has been applied. User direction was requested before this expansion. Once posterior validation passes, verify complete branch restoration and run the fixed [3,4,5] task0 pilot before any full dataset.

## Scope and provenance

Six completed probe-only physics trials (three readback controls plus three corrected-stop trials); zero task branches; zero training. The first sandboxed launch failed GPU initialization before a probe and is retained separately. Controller, utility, candidate range, canonical feasibility builder, frozen belief weights, historical P4 and historical labels were not changed. Corrected probe lives in an isolated current module; historical replay is untouched. It is not silently promoted into the old smoke entrypoint.

The exact corrected source used for the physical tests is `TESTED_CURRENT_PROBE_SOURCE.py` (SHA256 {sha(tested_source)}). The live module subsequently disabled its standalone bulk-collection entrypoint; its imported probe function is unchanged. See `ARTIFACT_PROVENANCE.json` for hashes and `READ_ONLY_AUDIT.ipynb` for CPU-only checks. No notebook cell launches Isaac.
'''
    (OUT/'PREACTION_CONTRACT_REPORT.md').write_text(report)
    paths=[ROOT/name for name in ['activeforcing_current_probe.py','activeforcing_probe_friction_contract.py',
        'activeforcing_decision_state.py','activeforcing_feasibility_features.py',
        'activeforcing_preact_action_probe_pilot_20260905.py','analyze_preact_action_probe_pilot_20260905.py']]
    paths += [tested_source,Path(contracts['high']['p4_source'])]
    paths += list(OUT.glob('*/*/RAW_PROBE.csv'))+list(OUT.glob('*/*/DECISION_STATE.pt'))+list(OUT.glob('*/*/EXECUTION_CONTRACT.json'))
    belief=ROOT/'activeforcing_full_claim_closure_20260902_062809/E6_E7/E6_E7_LOCKED_HANDOFF'
    paths += list(belief.glob('PHYSICAL_BELIEF_member_*.pt'))
    write('ARTIFACT_PROVENANCE.json',{'sha256':{str(p.resolve()):sha(p) for p in paths},
        'tested_probe_source':str(tested_source),'current_entrypoint_only_guard_change':True,
        'data_grain':'one pre-action snapshot per probe context/trial; zero full-task label rows',
        'old720_is_training_data':False,'posterior_input_source':'real observed prefix only; no candidate or outcome rows',
        'belief_new_patch_features_consumed':False})
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
