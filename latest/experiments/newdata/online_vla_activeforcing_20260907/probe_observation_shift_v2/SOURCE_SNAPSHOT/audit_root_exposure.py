"""Conservative read-only exposure inventory; does not allocate final roots."""
import json,re,subprocess,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
B=Path('/home/exouser/FORTE/analysis/results')
sources=[B/'current_multitask58_trained_v2_20260905/FROZEN_INPUT_PLAN.json',
    B/'current_matched_stage1_648_v1_20260906/STAGE_MANIFEST.json',
    B/'icra_final_experiment_package_v1_20260907/FINAL_FRESH_ROOT_PLAN.json']
groups={};proof={}
for p in sources:
    d=json.loads(p.read_text());proof[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    if p.name=='FROZEN_INPUT_PLAN.json':groups['physical_belief_all_splits']=sorted(map(int,d['root_splits']))
    elif p.name=='STAGE_MANIFEST.json':groups['feasibility_all_splits']=sorted({r for rs in d['selected_roots'].values() for r in rs})
    else:
        groups['previous_scripted_final_roots']=d['roots'];groups['previous_scripted_ablation_roots']=d['burned_ablation_roots']
groups['online_vla_qualification']=[5100]
locations=[str(B),'/home/exouser/Tabero/analysis/results','/media/volume/newdata/exouser/online_vla_activeforcing_20260907']
locations=[p for p in locations if Path(p).exists()]
res=subprocess.run(['rg','--files','--hidden',*locations],capture_output=True,text=True,check=True)
paths=res.stdout.splitlines();pattern=re.compile(r'(?:^|[/_])(?:root|r)([0-9]+)(?=[_/.-]|$)',re.I)
seen={}
for path in paths:
    for match in pattern.finditer(path):
        root=int(match[1]);seen.setdefault(root,[])
        if len(seen[root])<3:seen[root].append(path)
excluded=set(seen)
for roots in groups.values():excluded.update(roots)
report=dict(role='PRELIMINARY_EXPOSURE_INVENTORY_NOT_FINAL_ROOT_FREEZE',known_exposure_groups=groups,
    scanned_locations=locations,scanned_file_count=len(paths),path_inventory_sha256=hashlib.sha256(res.stdout.encode()).hexdigest(),
    source_manifest_sha256=proof,excluded_root_ids=sorted(excluded),example_path_evidence=seen,
    final_roots_selected=[],fresh_root_physics_started=0,
    limitations=['Path naming and known model/split manifests do not prove all historical exposure absent.',
        'Before final freeze, search chosen identifiers and seed/root fields in source records, training metadata and runtime logs.',
        'Previous scripted test roots are conservatively burned even when not used in model fitting.'])
with (H/'ROOT_EXPOSURE_AUDIT.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({'known_groups':groups,'excluded_count':len(excluded),'files_scanned':len(paths)}))
