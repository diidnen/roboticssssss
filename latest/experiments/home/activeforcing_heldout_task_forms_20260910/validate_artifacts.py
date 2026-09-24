"""Read-only scientific QA, with derived receipts written beside the report."""
from pathlib import Path
from datetime import datetime,timezone
import json,csv,hashlib,itertools
P=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
required=['CURRENT_FEASIBILITY_INPUT_AUDIT.md','EXISTING_GENERALIZATION_AUDIT.md','TASK_FORM_SCOPE.md','ALL_AVAILABLE_TASKS.csv','TASK_FORM_TAXONOMY.md','TASK_FORM_DIVERSITY_MATRIX.csv','FROZEN_VLA_NEW_TASK_QUALIFICATION.md','FINAL_HELDOUT_TASK_FORM_SET.json','TASK_FORM_OVERVIEW_CONTACT_SHEET.png','HELDOUT_TASK_FORM_CLAIM_AUDIT.json','FINAL_HELDOUT_TASK_FORM_GENERALIZATION_REPORT.md','TERMINAL_SUMMARY.txt']
assert all((P/f).is_file() for f in required)
inventory=list(csv.DictReader((P/'ALL_AVAILABLE_TASKS.csv').open()))
diversity=list(csv.DictReader((P/'TASK_FORM_DIVERSITY_MATRIX.csv').open()))
assert len(inventory)==len(diversity)==842
features=list(csv.DictReader((P/'CURRENT_FEASIBILITY_FEATURES.csv').open()))
assert len(features)==64
assert sum(x['explicit_task_identity']=='YES' for x in features)==4
manifest=P/'FINAL_HELDOUT_TASK_FORM_SET.json'
assert sha(manifest)==(P/'FINAL_HELDOUT_TASK_FORM_SET_SHA256.txt').read_text().split()[0]
final=read(manifest);assert not final['tasks'] and final['formal_branches']==0 and final['AF_calls_on_new_tasks']==0
assert all(v['status'] in ['SUPPORTED','MIXED','NOT_SUPPORTED','NOT_TESTED'] for v in read(P/'HELDOUT_TASK_FORM_CLAIM_AUDIT.json').values())
text=(P/'FINAL_HELDOUT_TASK_FORM_GENERALIZATION_REPORT.md').read_text()
assert all(f'# {i}. ' in text for i in range(1,18))
source_manifest=read('/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1/CONTINUOUS_FRICTION_RUNTIME_MANIFEST.json')
source_checks=[dict(path=path,sha256=expected,match=sha(path)==expected) for path,expected in source_manifest['source_hashes'].items()]
assert all(c['match'] for c in source_checks)
q=[];rpc_total=0
for result in sorted((P/'qualification').glob('*/QUALIFICATION_RESULT.json')):
 r=read(result);assert r['valid_outcome'] and r['AF_calls']==0 and r['method']=='FIXED_5'
 trace=[json.loads(x) for x in (result.parent/'TRACE.jsonl').read_text().splitlines()]
 assert len(trace)==r['steps']==200
 receipts=sorted((result.parent/'RPC').glob('*.json'));assert len(receipts)==r['rpc_count']==20
 for receipt in receipts:
  rpc=read(receipt)
  assert rpc['checkpoint_sha256']==source_manifest['vla_checkpoint_sha256']
  assert rpc['model_inference_called'] and rpc['observation_origin']=='LIVE_CURRENT_CONTROL_STEP'
  assert sha(receipt.with_suffix('.npz'))==rpc['artifact_sha256']
  assert rpc['instruction']==r['instruction']
 rpc_total+=len(receipts)
 longest=max((sum(1 for _ in g) for yes,g in itertools.groupby(bool(x['target_object_bilateral_contact'] and x['contact_opposition']) for x in trace) if yes),default=0)
 assert longest<4 and not r['established_grasp']
 q.append(dict(context=r['plan']['id'],bilateral_contact_steps=sum(x['target_object_bilateral_contact'] for x in trace),max_consecutive_bilateral_steps=longest,native_success_steps=sum(x['native_success'] for x in trace),observed_failure=r['failure']))
assert len(q)==6
for s in read(P/'HELDOUT_TASK_FORM_VISUAL_AUDIT/SOURCE_MANIFEST.json'):
 assert sha(s['source'])==s['sha256']==sha(P/s['artifact'])
qa=dict(status='PASS_WITH_SCIENTIFIC_LIMITATIONS',checked_utc=datetime.now(timezone.utc).isoformat(),inventory_rows=len(inventory),feature_channels=len(features),qualification_valid_episodes=len(q),verified_online_RPCs=rpc_total,formal_AF_branches=0,source_checks=source_checks,contact_screen_diagnostics=q,final_manifest_sha256=sha(manifest),limitations=['Native-reset screen does not test supplied-grasp downstream competence.','No final benchmark admitted; held-out task-form AF performance remains NOT_TESTED.','Native form intent is not measured downstream loading diversity.'])
(P/'evidence/FINAL_ARTIFACT_QA.json').write_text(json.dumps(qa,indent=2)+'\n')
(P/'ARTIFACT_SHA256.json').write_text(json.dumps({f:sha(P/f) for f in required},indent=2)+'\n')
print(json.dumps({k:v for k,v in qa.items() if k not in ['source_checks','contact_screen_diagnostics']},indent=2))
