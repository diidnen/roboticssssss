"""Finalize the task0 belief candidate and report its limited qualification."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from train_current_contract_belief_20260905 import OUT,ROOT,write,csvout,sha


def main():
    metrics=json.loads((OUT/'TRAINING_METRICS.json').read_text())
    comparison=json.loads((OUT/'OLD_VS_NEW_METRICS.json').read_text())
    manifest=json.loads((OUT/'PHYSICAL_BELIEF_CANDIDATE_MANIFEST.json').read_text())
    dataset=json.loads((OUT/'DATASET_PROVENANCE.json').read_text())
    diagnostic=json.loads((OUT/'TASK0_HIGH_MID_LOW_POSTERIORS.json').read_text())
    protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    with (OUT/'PHYSICAL_BELIEF_PREDICTIONS.csv').open() as stream:
        predictions=list(csv.DictReader(stream))
    support_diagnostic={}
    for split in ['VAL','TEST','DIAGNOSTIC']:
        examples=[r for r in predictions if r['split']==split]
        inside=sum(min(float(r[f'mu_{i}']) for i in range(3))<=float(r['target_mu'])<=
                   max(float(r[f'mu_{i}']) for i in range(3)) for r in examples)
        support_diagnostic[split]={'count':len(examples),'truth_inside_member_mean_span':inside,
            'fraction':inside/len(examples),
            'gaussian_mixture_90_interval_mean_width':metrics[split]['ensemble']['intervals']['0.9']['mean_width']}
    write('POSTERIOR_SUPPORT_DIAGNOSTIC.json',{'splits':support_diagnostic,
        'interpretation':'Member-mean span is descriptive, not a nominal credible interval. Gaussian mixture coverage uses sigma heads, which the unchanged planner does not integrate.',
        'posthoc_diagnostic_only':True,'used_for_checkpoint_selection':False,
        'final_continuous_posterior_validated':False,'posterior_interface_changed':False})
    models=manifest['checkpoints'];test=metrics['TEST']['ensemble'];val=metrics['VAL']['ensemble']
    qualified=metrics['eligibility']['task0_candidate_qualified']
    result=subprocess.run([sys.executable,'-m','unittest','-v','test_current_contract_belief.py',
        'test_activeforcing_decision_state.py','test_activeforcing_probe_friction_contract.py'],cwd=ROOT,capture_output=True,text=True)
    (OUT/'TEST_RESULTS.txt').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Regression tests failed')
    status={'FINAL_STATUS':'TASK0_CURRENT_CONTRACT_BELIEF_CANDIDATE_QUALIFIED' if qualified else 'TRAINED_BUT_QUALIFICATION_FAILED',
        'PHYSICAL_BELIEF_RETRAINED':True,'NEW_PROBE_PHYSICS_RUNS':33,'DIAGNOSTIC_PROBES_REUSED':3,
        'TASKS':[0],'ROOTS':12,'ROOT_SPLIT_LEAKAGE':False,
        'TRAIN_CONTEXTS':metrics['TRAIN']['ensemble']['count'],'VAL_CONTEXTS':val['count'],
        'TEST_CONTEXTS':test['count'],'DIAGNOSTIC_CONTEXTS':3,
        'TRAIN_ROOTS':protocol['root_split']['TRAIN'],'VAL_ROOTS':protocol['root_split']['VAL'],
        'TEST_ROOTS':protocol['root_split']['TEST'],'DIAGNOSTIC_ROOTS':protocol['root_split']['DIAGNOSTIC'],
        'FEATURE_DIM':58,'MODEL_ARCHITECTURE':protocol['architecture'],'MODEL_SEEDS':[0,1,2],
        'CHECKPOINT_SELECTION':protocol['checkpoint_rule'],'CALIBRATION':'NONE',
        'VAL_MAE':val['MAE'],'VAL_RMSE':val['RMSE'],'VAL_GAUSSIAN_MIXTURE_NLL':val['GAUSSIAN_MIXTURE_NLL'],
        'TEST_MAE':test['MAE'],'TEST_RMSE':test['RMSE'],'TEST_GAUSSIAN_MIXTURE_NLL':test['GAUSSIAN_MIXTURE_NLL'],
        'TEST_SPEARMAN':test['SPEARMAN'],'TEST_90_INTERVAL_COVERAGE':test['intervals']['0.9']['coverage'],
        'TEST_COMPLETE_ROOT_ORDER_FRACTION':metrics['TEST']['root_order']['ordered_fraction'],
        'OLD_E6_TEST_MAE':comparison['TEST']['OLD_E6']['MAE'],
        'OLD_E6_TEST_NLL':comparison['TEST']['OLD_E6']['GAUSSIAN_MIXTURE_NLL'],
        'CURRENT_HIGH_MID_LOW_POSTERIOR':[diagnostic[b]['mean'] for b in ['HIGH','MID','LOW']],
        'CURRENT_HIGH_MID_LOW_SUPPORT':{b:diagnostic[b]['member_means'] for b in ['HIGH','MID','LOW']},
        'TASK0_BELIEF_CANDIDATE_QUALIFIED':qualified,'FOUR_TASK_BELIEF_VALIDATED':False,
        'FINAL_CONTINUOUS_POSTERIOR_VALIDATED':False,
        'TEST_TRUTH_INSIDE_MEMBER_MEAN_SPAN':support_diagnostic['TEST'],
        'POSTERIOR_INTERFACE_CHANGED':False,'PLANNER_SIGMA_INTEGRATION_ADDED':False,
        'TASK_BRANCHES_EXECUTED':0,'FEASIBILITY_LABELS_COLLECTED':0,'FEASIBILITY_MODEL_RETRAINED':False,
        'READY_FOR_FULL_FEASIBILITY_COLLECTION':False,'CONTROLLER_CHANGED':False,
        'UTILITY_CHANGED':False,'PROBE_CHANGED_DURING_THIS_COLLECTION':False,'RUNTIME_FORCE_RESCALED':False,
        'CHECKPOINTS':models,'MANIFEST':str(OUT/'PHYSICAL_BELIEF_CANDIDATE_MANIFEST.json'),
        'NEXT_ACTION':'Review task0 candidate; separately verify full snapshot/controller restore and run gated 3/4/5 matched full-task pilot' if qualified else 'Inspect failing held-out qualification checks; do not run feasibility branches or tune against TEST',
        'CURRENT_BLOCKERS':['four-task generalization untested','full execution-state restore unverified','force-success boundary unmeasured',
            'planner uses narrow member-mean supports; Gaussian sigma-head coverage is not planner posterior validation']+
            ([] if qualified else ['predeclared task0 belief qualification failed'])}
    write('FINAL_STATUS.json',status)
    rows=[]
    for split in ['VAL','TEST']:
        for name,values in comparison[split].items():rows.append({'split':split,'model':name,
            **{k:values[k] for k in ['count','MAE','RMSE','GAUSSIAN_MIXTURE_NLL','SPEARMAN']},
            'coverage90':values['intervals']['0.9']['coverage']})
    csvout('MODEL_COMPARISON.csv',rows)
    table='\n'.join(f"| {r['split']} | {r['model']} | {r['MAE']:.6f} | {r['RMSE']:.6f} | {r['GAUSSIAN_MIXTURE_NLL']:.6f} | {r['coverage90']:.3f} |" for r in rows)
    diag_table='\n'.join(f"| {b} | {diagnostic[b]['mean']:.6f} | {', '.join(f'{m:.6f}' for m in diagnostic[b]['member_means'])} |" for b in ['HIGH','MID','LOW'])
    report=f'''# Current-contract task0 physical belief

## Result

Task0 candidate qualification: **{'PASS' if qualified else 'FAIL'}** under the rule frozen before acquisition/training. This is not a four-task final physical belief and does not establish any feasibility force-success boundary.

33 new probe-only physics runs; 3 previously collected corrected-probe cases reused only for diagnostic inference. No task branches, feasibility labels, utility changes, controller changes, or force scaling.

| Split | Model | MAE | RMSE | Gaussian-mixture NLL | 90% interval coverage |
|---|---|---:|---:|---:|---:|
{table}

Gaussian NLL is a continuous-density score, not Bernoulli BCE; negative values are valid. The frozen TRAIN Gaussian prior comparison and all three seeds' mean/std are in TRAINING_METRICS.json. No calibration parameters were fitted.

## Fixed diagnostic contexts (root00; never used to fit or select)

| Context | New mean μ | Three member means |
|---|---:|---|
{diag_table}

These are predictions on saved real corrected-probe observations, not oracle friction substitution and not three-point calibration. The probe remains the corrected friction-patch protocol (2/5/8 outward steps for these existing cases).

## Data and split

12 original task0 root families, 3 friction contexts each. TRAIN roots5101–5105, VAL5106–5107, TEST5108–5111, diagnostic5100. Admitted contexts TRAIN/VAL/TEST: {status['TRAIN_CONTEXTS']}/{status['VAL_CONTEXTS']}/{status['TEST_CONTEXTS']}. All members of a root remain in one split. Labels are explicitly configured object μ, checked against simulator material readback; labels and scenario IDs/bands do not enter the network.

Coverage is limited to the original task0 grasp configuration and friction bands [.20,.30], [.45,.60], [.90,1.00]. It does not test intermediate gaps, new geometries or other tasks. Four independent TEST roots are a small test set; rank/coverage fractions are diagnostics, not strong population guarantees. Environment construction retains the existing creation-seed behavior; source/state artifacts are saved, but bitwise future physical replay is not claimed.

## Features and recipe

58 raw observable channels: 46 existing observable proxies, explicitly marked as normal-vector projections where applicable, plus true world-frame friction vectors for both fingers, summed normal patch magnitudes, actual aperture and finger joints, and the real incremental friction stopping ratio. The new channels are measured data, not scale-adjusted old inputs. Physics dt=1/60 s; no candidate action appears in the sequence.

Architecture unchanged apart from input width: Linear(58,16)+ReLU → GRU(16,16) → μ/logσ heads. Three seeds0/1/2, 80 optimizer steps each, root-bootstrap TRAIN sampling, AdamW lr=.001/weight_decay=.0001, gradient clip1, Gaussian NLL+.05MAE. Normalization uses unpadded TRAIN observations only. Each seed's checkpoint is selected by lowest VAL NLL; all three selected seeds are equally weighted. TEST labels are loaded by the trainer only after CHECKPOINT_SELECTION_LOCK.json is written. There was no recipe search or diagnostic-force-based selection.

## Runtime contract and uncertainty

Online and saved-data feature paths are identical, label/private-state poisoning tests pass, and future rows cannot alter an existing feature prefix. A completed probe and retained bilateral contact are required for decision-time use; interim predictions are diagnostic only. Full simulator/controller restore is still a separate unpassed gate.

The planning interface is unchanged: three empirical support points at member μ values. σ heads provide Gaussian-mixture uncertainty diagnostics; they are **not silently integrated into the planner**. This result must not be described as a newly validated full continuous-posterior planner. CurrentPhysicalBelief loads absolute checkpoint paths with SHA checks and refuses unqualified non-diagnostic use. The old runtime/belief checkpoints remain untouched; this is an explicit candidate manifest, not an automatic physics deployment.

A post-lock support diagnostic makes this distinction concrete: only {support_diagnostic['TEST']['truth_inside_member_mean_span']}/{support_diagnostic['TEST']['count']} TEST truths fall between the minimum and maximum of the three member means (VAL {support_diagnostic['VAL']['truth_inside_member_mean_span']}/{support_diagnostic['VAL']['count']}; diagnostic {support_diagnostic['DIAGNOSTIC']['truth_inside_member_mean_span']}/{support_diagnostic['DIAGNOSTIC']['count']}). That span is not a nominal credible interval. The Gaussian-mixture 90% interval has mean width {support_diagnostic['TEST']['gaussian_mixture_90_interval_mean_width']:.6f} and uses σ heads absent from the unchanged planner's integration. Better mean estimation therefore does not establish calibrated planner uncertainty. The frozen task0 candidate criterion is unchanged, but final continuous-posterior validation remains **NO**. See POSTERIOR_SUPPORT_DIAGNOSTIC.json; this posthoc observation was not used to select or retrain checkpoints.

## Next gate

{'The task0 candidate passes the frozen probability/admission/order checks. Next, inspect its provenance, validate complete snapshot/controller restoration, and request the matched 3/4/5 full-task pilot. Do not jump to full feasibility collection.' if qualified else 'At least one frozen qualification check failed. Keep the candidate diagnostic-only; inspect TRAINING_METRICS.json eligibility and do not tune against TEST or proceed to feasibility physics.'}

Artifacts: PROTOCOL.json, SPLIT_MANIFEST.json, FEATURE_SCHEMA.json, DATASET_PROVENANCE.json, PHYSICAL_BELIEF_CANDIDATE_MANIFEST.json, TRAINING_METRICS.json, MODEL_COMPARISON.csv, PHYSICAL_BELIEF_PREDICTIONS.csv, TASK0_DIAGNOSTIC_PER_STEP_POSTERIOR.csv, PREFIX_AND_RUNTIME_PARITY.csv, POSTERIOR_SUPPORT_DIAGNOSTIC.json, SAVED_DECISION_INFERENCE.json, TEST_RESULTS.txt, READ_ONLY_AUDIT.ipynb.
'''
    (OUT/'CURRENT_CONTRACT_BELIEF_REPORT.md').write_text(report)
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
