"""Inventory task definitions; UNKNOWN is intentional, never an inferred rollout metric."""
from pathlib import Path
import ast,csv,hashlib,json,re
from collections import Counter
OUT=Path(__file__).resolve().parent
TAB=Path('/home/exouser/Tabero')
SRC=TAB/'source/tac_manip/tac_manip/tasks/manipulation'
paths=(OUT/'evidence/task_source_paths.txt').read_text().splitlines()
train={'alphabet_soup_1','cream_cheese_1','tomato_sauce_1','butter_1'}
def write(n,x): (OUT/n).write_text(json.dumps(x,indent=2)+'\n')
def csvwrite(n,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/n).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def classify(name,objects,goals,suite):
    if len(objects)>1 or (any(g.get('operation') for g in goals) and any(g.get('relationship') for g in goals)) or ' and close ' in name or ' and put ' in name or 'both ' in name:
        return 'OUTSIDE_MULTI_GRASP','multiple objects or manipulation modes; complete task cannot be one maintained grasp'
    if name.startswith('push '):return 'OUTSIDE_NON_GRASP_PUSH','surface contact exists, but pushing is not maintained-grasp dragging'
    if name.startswith('turn on'):return 'FORM_F_UNCONFIRMED','knob operation; grasp/control mode unverified'
    if name.startswith('open '):return 'FORM_D','articulated drawer pulling; one handle grasp is plausible, not qualified'
    if 'back compartment' in name:return 'FORM_C_CANDIDATE','book-to-caddy constrained placement; actual alignment/loading must be verified'
    if 'wine bottle on the rack' in name:return 'FORM_C_CANDIDATE','rack placement; reorientation not established by current position-only evaluator'
    if 'in the top drawer' in name and name.startswith('pick up'):return 'FORM_D_CANDIDATE','bowl extraction from drawer; novelty depends on extraction occurring after handoff'
    return 'FORM_A','lift/transport/place; semantic or target changes alone do not establish a new form'
rows=[];config_rows=[]
for p in sorted((TAB/'benchmarks/datasets/libero/config').glob('*.json')):
    x=json.loads(p.read_text());suite=x['task_suite_name']
    for index,t in enumerate(x['tasks']):
        task=t['task_id'];name=t['language_instruction'];objs=t['obj_of_interest'];goals=t['goals']
        if not objs:objs=[g['target'] for g in goals if 'operation' in g]
        form,reason=classify(name,t['obj_of_interest'],goals,suite)
        trained=suite=='libero_object' and task in [0,1,5,6]
        overlap=bool(set(objs)&train)
        category='TRAINING_TASK' if trained else ('OUTSIDE_CURRENT_SCOPE' if form.startswith('OUTSIDE') else ('SAME_FORM_NEW_OBJECT' if suite=='libero_object' else 'NEW_TASK_SAME_FORM') if form=='FORM_A' else 'HELDOUT_TASK_FORM_CANDIDATE' if overlap else 'HELDOUT_TASK_FORM_AND_OBJECT_CANDIDATE')
        demos=[s for s in paths if re.search(rf'/{suite}_task{task}_.*\.hdf5$',s)]
        # Fixture names are mapped by SceneConfig, so this checks object assets only.
        missing=[v['type'] for v in t['objects'].values() if not (TAB/'benchmarks/datasets/libero/USD'/v['type']/(v['type']+'.usd')).exists()]
        r=dict(task_id=f'{suite}:{task}',task_name=t['task_name'],language_instruction=name,
            manipulated_object=';'.join(objs),target_object_or_receptacle=';'.join(t['targets']) or ';'.join(g['target'] for g in goals),
            existing_grasp_snapshot='YES: original-family runtime references' if trained else 'NOT_FOUND_FOR_CURRENT_RUNTIME',
            native_terminal_evaluator=str(SRC/'libero/mdp/terminations.py')+'::libero_goals_reached',
            known_VLA_support='VERIFIED_CURRENT_ONLINE' if trained else 'CONFIGURED_EVAL_TASK_NOT_CURRENT_QUALIFIED' if suite=='libero_object' and task!=4 else 'UNVERIFIED_FROZEN_CHECKPOINT',
            training_usage='CURRENT_FEASIBILITY_TRAIN' if trained else 'ABSENT_CURRENT_FEASIBILITY_TRAIN; frozen VLA exact dataset membership not proved',
            existing_rollouts=';'.join(demos) or 'NO_MATCHED_HDF5_FOUND; broader filename-search ledger retained',
            requires_lift='YES_TASK_DESCRIPTION' if form=='FORM_A' or 'book' in name else 'UNKNOWN',
            transport_distance='UNMEASURED',lateral_motion='UNKNOWN',vertical_motion='UNKNOWN',wrist_rotation='UNMEASURED',object_reorientation='UNVERIFIED',
            surface_contact='YES_PUSH' if name.startswith('push ') else 'UNKNOWN',environment_contact='YES_TERMINAL_TARGET_OR_ARTICULATION',
            insertion='CANDIDATE' if form=='FORM_C_CANDIDATE' else 'NO_EVIDENCE',extraction='CANDIDATE' if form.startswith('FORM_D') else 'NO_EVIDENCE',
            constrained_alignment='CANDIDATE' if form=='FORM_C_CANDIDATE' else 'NOT_ESTABLISHED',release_required='TASK_SEMANTICS_ONLY_NATIVE_GOAL_DOES_NOT_CHECK' if 'put ' in name or 'place ' in name else 'UNKNOWN',
            regrasp_required='YES_OR_HAND_MODE_CHANGE' if form=='OUTSIDE_MULTI_GRASP' else 'UNVERIFIED',task_form=form,
            form_in_feasibility_training='YES' if form=='FORM_A' else 'NO_IDENTIFIED',only_object_new='YES' if category=='SAME_FORM_NEW_OBJECT' else 'NO',
            downstream_loading_genuinely_new='UNVERIFIED' if form!='FORM_A' else 'NO_ESTABLISHED_DIFFERENCE',candidate_category=category,object_overlap_with_training=overlap,
            source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_array_index=index,
            config_index_matches_id=task==index,missing_object_usd_assets=';'.join(missing),classification_evidence=reason,
            evaluator_goals_json=json.dumps(goals,sort_keys=True),availability_tier='NATIVE_TABERO_CONFIG')
        rows.append(r);config_rows.append(r)

# Concrete native registered task semantics; controller/sensor variants are not new tasks.
for key,obj,target,form,reason in [
    ('peg_insert','peg_8mm','hole_8mm','FORM_C','peg/hole geometry and registered native insertion task'),
    ('gear_mesh','medium_gear','gear_base','FORM_C','registered gear meshing task'),
    ('nut_thread','nut_m16','bolt_m16','FORM_B_C_CANDIDATE','threading needs orientation/contact; repeated grasp necessity unverified'),
    ('open_drawer','drawer_handle','cabinet','FORM_D','native drawer joint terminal evaluator'),
    ('put_into_and_close_drawer','object_and_drawer','cabinet','OUTSIDE_MULTI_GRASP','object placement then drawer closure changes manipulation mode')]:
    factory=key in ['peg_insert','gear_mesh','nut_thread'];source=SRC/('factory/factory_env_cfg.py' if factory else 'articulated/open_drawer_env_cfg.py')
    rows.append(dict(task_id='native:'+key,task_name=key,language_instruction='NOT_DEFINED_IN_NATIVE_TASK_CONFIG',manipulated_object=obj,target_object_or_receptacle=target,
        existing_grasp_snapshot='NOT_FOUND',native_terminal_evaluator=str(SRC/('factory/mdp/terminations.py' if factory else 'articulated/mdp/terminations.py')),
        known_VLA_support='UNVERIFIED; native task has different action/observation contract',training_usage='ABSENT_CURRENT_FEASIBILITY_TRAIN',existing_rollouts='NO_CURRENT_CHECKPOINT_MATCHED_ROLLOUT_FOUND',
        **{k:'UNKNOWN' for k in ['requires_lift','transport_distance','lateral_motion','vertical_motion','wrist_rotation','object_reorientation','surface_contact','environment_contact','release_required','regrasp_required']},
        insertion='YES_CONFIG' if factory else 'NO',extraction='YES_CONFIG' if key=='open_drawer' else 'NO',constrained_alignment='YES_CONFIG' if factory else 'UNKNOWN',task_form=form,
        form_in_feasibility_training='NO',only_object_new='NO',downstream_loading_genuinely_new='CONFIG_INTENT; unverified under frozen VLA',candidate_category='OUTSIDE_CURRENT_SCOPE' if form.startswith('OUTSIDE') else 'HELDOUT_TASK_FORM_AND_OBJECT_CANDIDATE',object_overlap_with_training=False,source=str(source),classification_evidence=reason,availability_tier='NATIVE_OTHER_CONTROLLER'))

# xhumanoid entries are task identifiers only; no supported setup branch beyond env variables.
tree=ast.parse((TAB/'source/tac_manip/tac_manip/utils/task_configs.py').read_text());values={}
for n in tree.body:
    if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id.startswith('xhumanoid_'):values[n.targets[0].id]=ast.literal_eval(n.value)
for task,instruction in values['xhumanoid_task_dict'].items():
    objs=values['xhumanoid_object_name_dict'][task]
    rows.append(dict(task_id='xhumanoid:'+str(task),task_name=instruction,language_instruction=instruction,manipulated_object=objs[0],target_object_or_receptacle=';'.join(objs[1:]),
        task_form='FORM_A',candidate_category='NEW_TASK_SAME_FORM',form_in_feasibility_training='YES',only_object_new='NO',downstream_loading_genuinely_new='NO_EVIDENCE',
        availability_tier='IDENTIFIERS_ONLY',known_VLA_support='UNVERIFIED',training_usage='ABSENT_CURRENT_FEASIBILITY_TRAIN',existing_grasp_snapshot='NOT_FOUND',native_terminal_evaluator='NOT_RESOLVED',existing_rollouts='NOT_FOUND',source=str(TAB/'source/tac_manip/tac_manip/utils/task_configs.py')))

# Include all locally found LIBERO BDDL variants as external definitions, deduplicated by suite/stem.
known={(r['task_id'].split(':')[0],r['task_name']) for r in config_rows}
external={}
for s in paths:
    p=Path(s)
    if p.suffix!='.bddl':continue
    key=(p.parent.name,p.stem)
    external.setdefault(key,[]).append(s)
for (suite,name),sources in sorted(external.items()):
    if (suite,name) in known:continue
    text=Path(sources[0]).read_text();m=re.search(r'\(:language\s+(.*?)\)',text,re.S);instruction=' '.join(m.group(1).split()) if m else name.replace('_',' ')
    form,reason=classify(instruction,[],[],suite)
    rows.append(dict(task_id='bddl:'+suite+':'+name,task_name=name,language_instruction=instruction,
        manipulated_object='SEE_BDDL; not mapped to current Tabero assets',target_object_or_receptacle='SEE_BDDL',existing_grasp_snapshot='NOT_FOUND_FOR_CURRENT_RUNTIME',
        native_terminal_evaluator='LIBERO BDDL goal predicates; different simulator, not current evaluator',known_VLA_support='UNVERIFIED_CURRENT_CHECKPOINT',training_usage='ABSENT_CURRENT_FEASIBILITY_TRAIN',
        existing_rollouts='OTHER_PROJECT_DEFINITIONS; not current-checkpoint evidence',task_form=form,form_in_feasibility_training='YES' if form=='FORM_A' else 'NO_IDENTIFIED',only_object_new='UNVERIFIED',downstream_loading_genuinely_new='UNVERIFIED',candidate_category='EXTERNAL_DEFINITION_REQUIRES_RUNTIME_AUDIT',
        source=';'.join(sources),classification_evidence='PROVISIONAL_LANGUAGE_SCREEN_ONLY: '+reason,availability_tier='EXTERNAL_BDDL_NOT_TABERO_PORT'))
required='task_id task_name language_instruction manipulated_object target_object_or_receptacle existing_grasp_snapshot native_terminal_evaluator known_VLA_support training_usage existing_rollouts requires_lift transport_distance lateral_motion vertical_motion wrist_rotation object_reorientation surface_contact environment_contact insertion extraction constrained_alignment release_required regrasp_required'.split()
for r in rows:
    for key in required:r.setdefault(key,'UNKNOWN_NOT_MEASURED')
csvwrite('ALL_AVAILABLE_TASKS.csv',rows)
csvwrite('TASK_FORM_DIVERSITY_MATRIX.csv',[{k:r.get(k,'UNKNOWN') for k in ['task_id','availability_tier','task_form','form_in_feasibility_training','only_object_new','object_overlap_with_training','downstream_loading_genuinely_new','candidate_category','classification_evidence']} for r in rows])
stats=dict(inventory_rows=len(rows),native_tabero_config_rows=len(config_rows),native_other_task_semantics=5,identifier_only_tasks=6,
    external_bddl_rows=sum(r['availability_tier']=='EXTERNAL_BDDL_NOT_TABERO_PORT' for r in rows),
    tiers=dict(Counter(r['availability_tier'] for r in rows)),native_form_counts=dict(Counter(r['task_form'] for r in rows if not r['availability_tier'].startswith('EXTERNAL'))),
    configured_evaluation_tasks=json.loads((TAB/'benchmarks/datasets/tabero/config/tabero_tasks.json').read_text()),
    all_scalar_motion_quantities_unmeasured=True,formal_candidate_selection_done=False)
write('evidence/TASK_INVENTORY_COUNTS.json',stats);print(json.dumps(stats,indent=2))
