#!/usr/bin/env python3
"""Complete non-computational audit artifacts without rerunning TEST."""
import hashlib, json
from pathlib import Path
import pandas as pd

OUT=Path('/home/exouser/Tabero/analysis/results/trajectory_physical_imagination_20260829_065220')
branches=pd.read_csv(OUT/'GT_FRICTION_IMAGINED_BRANCHES.csv')
frontier=pd.read_csv(OUT/'GT_FRICTION_FRONTIER.csv')
gate=json.loads((OUT/'NEW_GATE3_PROTOCOL_IMMUTABLE.json').read_text())
threshold=float(gate['frontier_threshold'])
rows=[]
for r in branches.itertuples(index=False):
    pred=int(float(r.pred_success_probability)>=threshold)
    err=int(pred != int(r.real_success))
    if not err:
        category='none'
    elif pred and not int(r.real_success):
        category='evaluator_misclassification_false_safe_after_imagined_trajectory'
    else:
        category='evaluator_misclassification_false_unsafe_after_imagined_trajectory'
    rows.append({'context_id':r.context_id,'root_id':r.root_id,'task':int(r.task),'split':r.split,'force':float(r.force),'real_success':int(r.real_success),'predicted_success_probability':float(r.pred_success_probability),'predicted_success':pred,'outcome_error':err,'earliest_observed_error':category,'diagnostic_basis':'imagined branch probability versus held-out full-task outcome; no slip label used'})
pd.DataFrame(rows).to_csv(OUT/'ERROR_ATTRIBUTION.csv',index=False)
calls=branches.world_model_calls.astype(float)
lat={'status':'PARTIAL_MEASUREMENT','measurement_scope':'GT-friction imagined branches already evaluated; no TEST rerun','candidate_force_count_total':int(len(branches)),'candidate_force_count_by_split':branches.groupby('split').size().to_dict(),'friction_hypotheses_per_branch':1,'trajectory_horizon_steps':int(json.loads((OUT/'WORLD_MODEL_SELECTION.json').read_text())['selected_H']),'world_model_calls_total':int(calls.sum()),'world_model_calls_mean_per_branch':float(calls.mean()),'wall_clock_planning_time_s':None,'wall_clock_reason':'the completed runner did not bracket GT frontier calls with a timer; value intentionally left null rather than inferred','real_task_duration_s':None,'real_e2e_run':False}
(OUT/'LATENCY.json').write_text(json.dumps(lat,indent=2,sort_keys=True)+'\n')
rep=json.loads((OUT/'TRAJECTORY_REPRESENTATION.json').read_text())
rep['contact_state_status']='not a separate world-model output; direct contact_left/right retained in source/audit only, while direct per-finger normal/tangential forces are supervised in the 13-channel model'
rep['contact_state_prediction_f1']='not evaluated for this new head; latest reused Physics-GRU v3 diagnostic remains macro F1 approximately 0.922'
rep['slip']='not a target; tangential velocity proxy is not direct slip ground truth'
(OUT/'TRAJECTORY_REPRESENTATION.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
report=(OUT/'FINAL_REPORT.md').read_text()
report=report.replace('REAL VS IMAGINED FORCE FRONTIER\n\nTEST frontier metrics are in GT_FRICTION_FRONTIER.csv; error attribution remains recorded per context in the frontier file.', 'REAL VS IMAGINED FORCE FRONTIER\n\nTEST frontier metrics are in GT_FRICTION_FRONTIER.csv; per-imagined-branch earliest observed error attribution is in ERROR_ATTRIBUTION.csv.')
report=report.replace('TEST-TIME IMAGINATION LATENCY\n\nWorld-model call counts are recorded in GT_FRICTION_IMAGINED_BRANCHES.csv; no real E2E latency is claimed.', 'TEST-TIME IMAGINATION LATENCY\n\nWorld-model call counts and the missing wall-clock measurement are recorded honestly in LATENCY.json; no real E2E latency is claimed.')
(OUT/'FINAL_REPORT.md').write_text(report)
files=[p for p in OUT.iterdir() if p.name!='MANIFEST.sha256']
(OUT/'MANIFEST.sha256').write_text('\n'.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}' for p in sorted(files))+'\n')
print(json.dumps({'error_attribution_rows':len(rows),'latency':lat,'manifest_entries':len(files)},indent=2))
