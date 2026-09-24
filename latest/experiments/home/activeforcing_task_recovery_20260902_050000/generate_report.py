from pathlib import Path
import csv
import math
import statistics
from collections import defaultdict, Counter

OUT = Path('/home/exouser/activeforcing_task_recovery_20260902_050000')
ARCHIVE = Path('/home/exouser/FORTE/gnp_style_continuous_20260830_125107')
DATA = ARCHIVE / 'CONTINUOUS_TRAIN_SUCCESS_DATA.csv'
TELEMETRY = ARCHIVE / 'collection_long2/P5S0C_BRANCH_TELEMETRY'

TASKS = {
    '0': dict(name='alphabet_soup_1', object='alphabet soup', role='basic hidden-friction adaptation; short basket pick-place; breadth anchor', support='3.0–5.0 N', qualification='current 720 archive; B2-R2 physics-decision-positive', caveat='post-lift failure cause unavailable'),
    '1': dict(name='cream_cheese_1', object='cream cheese', role='basic adaptation plus exact-force critical; short basket pick-place; strongest grasp-force qualification', support='4.0–6.0 N', qualification='current 720 archive; P6G0 Track A replicated', caveat='140/180 terminal labels reconstructed; post-lift cause unavailable'),
    '5': dict(name='tomato_sauce_1', object='tomato sauce', role='basic hidden-friction adaptation; short basket pick-place; breadth anchor', support='3.0–5.0 N', qualification='current 720 archive; B2-R2 physics-decision-positive', caveat='post-lift failure cause unavailable'),
    '6': dict(name='butter_1', object='butter', role='narrow-force / delayed-failure stress case; short basket pick-place; breadth anchor', support='3.0–4.0 N', qualification='current 720 archive; P6G0 Track A replicated', caveat='narrow support and post-lift cause unavailable'),
}
SUPPORTS = {'0': (3.0, 5.0), '1': (4.0, 6.0), '5': (3.0, 5.0), '6': (3.0, 4.0)}
POST = {'transit', 'over_basket', 'place', 'release', 'settle'}

def read_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def f(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def i(v):
    try: return int(float(v))
    except (TypeError, ValueError): return 0

def write_csv(name, rows, fields):
    with (OUT / name).open('w', newline='') as g:
        w = csv.DictWriter(g, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def write(name, text):
    (OUT / name).write_text(text.strip() + '\n')

rows = read_rows(DATA)
phase_cache = {}
for r in rows:
    p = Path(r['telemetry_path'])
    phases = set()
    if p.exists():
        with p.open(newline='') as g:
            for tr in csv.DictReader(g):
                if tr.get('phase'): phases.add(tr['phase'])
    phase_cache[r['telemetry_path']] = phases
    r['_phases'] = phases
    r['_success'] = i(r['full_task_success_y'])
    r['_force'] = f(r['requested_force_N'])
    r['_post'] = bool(phases & POST)

by_task = defaultdict(list)
by_context = defaultdict(list)
for r in rows:
    by_task[r['task']].append(r); by_context[r['context_id']].append(r)

def first_success(rs):
    good = [r['_force'] for r in rs if r['_success'] and r['_force'] is not None]
    return min(good) if good else None

def monotonic_violations(rs):
    levels = defaultdict(list)
    for r in rs: levels[round(r['_force'], 10)].append(r['_success'])
    vals = [(x, statistics.mean(y)) for x, y in sorted(levels.items())]
    return sum(1 for a, b in zip(vals, vals[1:]) if a[1] > b[1] + 1e-9)

diag = []
for task in sorted(by_task, key=int):
    trs = by_task[task]
    contexts = sorted({r['context_id'] for r in trs})
    roots = sorted({r['root_id'] for r in trs})
    bands = sorted({r['friction_band'] for r in trs})
    by_band = defaultdict(list)
    for r in trs: by_band[r['friction_band']].append(r)
    band_sr = {b: statistics.mean(r['_success'] for r in x) for b, x in by_band.items()}
    context_both = sum(1 for c in contexts if {r['_success'] for r in by_context[c]} == {0, 1})
    # Contexts are the unit for a local force frontier; two repeats make this an empirical diagnostic, not a reliable threshold.
    frontiers = [first_success(by_context[c]) for c in contexts]
    frontiers = [x for x in frontiers if x is not None]
    context_viol = sum(monotonic_violations(by_context[c]) for c in contexts)
    post_fail = sum(1 for r in trs if not r['_success'] and r['_post'])
    diag.append({
        'grain':'task', 'task':task, 'task_name':TASKS[task]['name'], 'n_rows':len(trs), 'n_contexts':len(contexts), 'n_root_families':len(roots),
        'friction_bands':'|'.join(bands), 'friction_min':f'{min(f(r["friction"]) for r in trs):.6f}', 'friction_max':f'{max(f(r["friction"]) for r in trs):.6f}',
        'force_support_N':TASKS[task]['support'], 'force_min_observed':f'{min(r["_force"] for r in trs):.6f}', 'force_max_observed':f'{max(r["_force"] for r in trs):.6f}',
        'full_success_rate':f'{statistics.mean(r["_success"] for r in trs):.6f}', 'contexts_with_both_success_and_failure':context_both,
        'context_both_rate':f'{context_both/len(contexts):.6f}', 'empirical_first_success_median_N':f'{statistics.median(frontiers):.6f}' if frontiers else '',
        'empirical_first_success_mean_N':f'{statistics.mean(frontiers):.6f}' if frontiers else '', 'adjacent_force_nonmonotonicity_count':context_viol,
        'post_lift_failure_candidates':post_fail, 'archive_under_force_rows':'123 total; task attribution not recovered',
        'label_quality':TASKS[task]['caveat'], 'interpretation':'force/friction sensitivity is observable in archive but exact causal failure stage is not fully labeled'
    })
    for c in contexts:
        cr = by_context[c]
        good = [r['_force'] for r in cr if r['_success'] and r['_force'] is not None]
        fail = [r['_force'] for r in cr if not r['_success'] and r['_force'] is not None]
        diag.append({'grain':'context','task':task,'task_name':TASKS[task]['name'],'n_rows':len(cr),'n_contexts':1,'n_root_families':1,
            'friction_bands':cr[0]['friction_band'],'friction_min':cr[0]['friction'],'friction_max':cr[0]['friction'],'force_support_N':TASKS[task]['support'],
            'force_min_observed':f'{min(r["_force"] for r in cr):.6f}','force_max_observed':f'{max(r["_force"] for r in cr):.6f}',
            'full_success_rate':f'{statistics.mean(r["_success"] for r in cr):.6f}','contexts_with_both_success_and_failure':int({r['_success'] for r in cr}=={0,1}),
            'context_both_rate':int({r['_success'] for r in cr}=={0,1}),'empirical_first_success_median_N':f'{min(good):.6f}' if good else '',
            'empirical_first_success_mean_N':f'{max(fail):.6f}' if fail else '','adjacent_force_nonmonotonicity_count':monotonic_violations(cr),
            'post_lift_failure_candidates':sum(1 for r in cr if not r['_success'] and r['_post']),'archive_under_force_rows':'', 'label_quality':TASKS[task]['caveat'],'interpretation':c})

diag_fields = ['grain','task','task_name','n_rows','n_contexts','n_root_families','friction_bands','friction_min','friction_max','force_support_N','force_min_observed','force_max_observed','full_success_rate','contexts_with_both_success_and_failure','context_both_rate','empirical_first_success_median_N','empirical_first_success_mean_N','adjacent_force_nonmonotonicity_count','post_lift_failure_candidates','archive_under_force_rows','label_quality','interpretation']
write_csv('EXISTING_720_TASK_DIAGNOSTICS.csv', diag, diag_fields)
write('EXISTING_720_TASK_DIAGNOSTICS.md', '''# Existing 720-task diagnostics

The raw archive has 720 rows: 180 per task, 18 contexts per task, and 6 root families per task. The archive is valid/state-parity clean, but each context uses continuous force draws rather than the exact frozen 0.25 N grid. The diagnostics CSV includes task and context grain.

`full_success_rate` is the observed archive label rate, not the later GT-Direct OOF rate. `contexts_with_both_success_and_failure` is a force-sensitivity diagnostic under two repeats per context, not a causal threshold estimate. `post_lift_failure_candidates` means full failure with telemetry reaching a post-lift phase; it does not prove grip loss, transport failure, or placement failure. The closure taxonomy reports 123 `UNDER_FORCE` rows overall, but task attribution was not recovered and is intentionally not fabricated.
''')

current = []
for t in sorted(TASKS, key=int):
    a = next(x for x in diag if x['grain']=='task' and x['task']==t)
    current.append({'task_id':t,'task_name':TASKS[t]['name'],'object':TASKS[t]['object'],'instruction':f'pick up the {TASKS[t]["object"]} and place it in the basket','suite':'libero_object','scene_or_fixture':'floor / basket_1','available_branch_count':a['n_rows'],'available_context_count':a['n_contexts'],'available_root_count':a['n_root_families'],'friction_levels':'LOW/MID/HIGH; continuous context draws','force_levels':TASKS[t]['support'],'label_quality':TASKS[t]['caveat'],'nominal_vla_behavior':'frozen VLA reaches canonical pre-probe state; exact current all-task reset-to-end π0 chain not fully verified','manipulation_stages':'approach/contact → close/grasp → lift → transit → over_basket → place → release → settle','success_criterion':'official object is in basket; frozen full-task evaluator label','known_failure_stages':f'{a["post_lift_failure_candidates"]} post-lift-phase/full-failure candidates; exact drop/place cause unavailable','primary_scientific_role':TASKS[t]['role'],'current_paper_role':'hidden-friction core for E1/E2 and conditional E3; insufficient alone for semantic coverage or valid LocalLift contrast','evidence_status':TASKS[t]['qualification']})
write_csv('AUTHORITATIVE_CURRENT_TASK_MAP.csv', current, list(current[0]))

write('AUTHORITATIVE_CURRENT_TASK_MAP.md', f'''# Authoritative current task map

The authoritative current archive is `FORTE/gnp_style_continuous_20260830_125107/CONTINUOUS_TRAIN_SUCCESS_DATA.csv`: 720 valid rows, 72 contexts, 24 task-specific root families, four tasks, five continuous equal-width force strata per context, and two repeats per stratum. The archive is a hidden-friction core, not a broad manipulation benchmark.

All four current tasks are short `LIBERO-Object` food-object → basket pick-and-place tasks. Their primary role is basic hidden-friction/force adaptation. None is a clean primary example of long post-lift transport, turning/acceleration, constrained placement, mass sensitivity, or friction+mass interaction. Telemetry exposes phases through `settle`, but the full-failure cause after lift is not authoritatively labeled; therefore “transport failure” is an observable downstream candidate, not a confirmed grip-loss diagnosis.

The archive SHA-256 recorded by the closure audit is `80bb811e79796a9c292517ae37d2e224710a228ab8afd1e5be38ab9c2e5857c9`. Exact frozen 0.25 N Direct grid evaluation and a true zero-query run are still data gaps: the archive uses continuous context-specific draws and its offline no-information row is Query-Ignored.
''')

hist_rows = [
 {'category':'A_IMPLEMENTED_AND_DATA_AVAILABLE','design':'P5-S0-C / current continuous archive','tasks':'0,1,5,6','status':'COMPLETED_ARCHIVE','evidence':'FORTE/gnp_style_continuous_20260830_125107; 720 rows','scientific_use':'E1/E2; conditional E3'},
 {'category':'A_IMPLEMENTED_AND_DATA_AVAILABLE','design':'B2-R2 nine-task LIBERO-Object oracle breadth','tasks':'0,1,2,3,5,6,7,8,9','status':'COMPLETED_SCREENING','evidence':'Tabero analysis/results/b2r2_tabero_task_breadth_20260821_073931','scientific_use':'task qualification only; no learned/probe claim'},
 {'category':'A_IMPLEMENTED_AND_DATA_AVAILABLE','design':'P6G0 explicit grasp-position + hidden COM screening','tasks':'0,1,2,5,6','status':'COMPLETED_SCREENING','evidence':'Tabero analysis/results/p6g0_grasp_force_physics_benchmark_20260824_182251','scientific_use':'t1/t6 joint grasp-force qualification; t0/t2/t5 not promoted'},
 {'category':'B_IMPLEMENTED_BUT_NOT_CURRENTLY_USED','design':'P5-S0-D legacy fresh E2E Q2F','tasks':'0,1,5,6','status':'PASS_LEGACY','evidence':'Tabero analysis/results/p5s0d_fresh_e2e_q2f_20260824_090205','scientific_use':'historical end-to-end context; not current Direct E5'},
 {'category':'B_IMPLEMENTED_BUT_NOT_CURRENTLY_USED','design':'P6G0 t0/t2/t5 candidate generator','tasks':'0,2,5','status':'NOT_PROMOTED','evidence':'P6G0/P6G0R1 reports','scientific_use':'candidate evidence only; generator not general/reliable'},
 {'category':'C_PARTIALLY_IMPLEMENTED','design':'LocalLift matched diagnostic','tasks':'0,1,5,6','status':'FAILED_SCIENTIFICALLY','evidence':'FULLTASK_LOCALLIFT_MATCHED_REPORT.md; 0 negative LocalLift labels','scientific_use':'cannot compare LocalLift'},
 {'category':'C_PARTIALLY_IMPLEMENTED','design':'current Direct fresh all-task reset-to-end E2E','tasks':'0,1,5,6','status':'IN_PROGRESS','evidence':'FRESH_E2E_REPORT.md / TABLE_FRESH_E2E.csv','scientific_use':'E5 gap'},
 {'category':'C_PARTIALLY_IMPLEMENTED','design':'exact frozen 0.25 N force grid','tasks':'0,1,5,6','status':'NOT_ARCHIVED','evidence':'ACTIVEFORCING_FINAL_DATA_AUDIT.md','scientific_use':'fresh matched collection required'},
 {'category':'D_PLANNED_ONLY','design':'long post-lift transport','tasks':'new qualification needed','status':'PLANNED','evidence':'TASK_BENCHMARK_RECOVERY.md','scientific_use':'Group B gap'},
 {'category':'D_PLANNED_ONLY','design':'turning / acceleration stress','tasks':'new qualification needed','status':'PLANNED','evidence':'TASK_BENCHMARK_RECOVERY.md; root_scaling decision','scientific_use':'Group C gap'},
 {'category':'D_PLANNED_ONLY','design':'precise constrained placement','tasks':'new qualification needed','status':'PLANNED','evidence':'TASK_BENCHMARK_RECOVERY.md','scientific_use':'Group D gap'},
 {'category':'D_PLANNED_ONLY','design':'mass extension and 3×3 friction×mass joint','tasks':'new qualification needed','status':'PLANNED','evidence':'LIBERO_PHYSICS_CONFIGURATION.zh-CN.md; no current mass core','scientific_use':'mass claim gap'},
 {'category':'D_PLANNED_ONLY','design':'LIBERO-PRO perturbation evaluation','tasks':'base task required','status':'PLANNED','evidence':'official LIBERO-PRO repository/paper','scientific_use':'generalization extension'},
 {'category':'E_DISCUSSED_BUT_NO_ACTIVEFORCING_DATA','design':'articulated drawer/cabinet/microwave variants','tasks':'LIBERO-10/Goal candidates','status':'ASSETS_EXIST_BUT_NOT_QUALIFIED','evidence':'Tabero task configs and LIBERO inventories','scientific_use':'candidate only; no ActiveForcing force frontier'},
 {'category':'F_NEWLY_PROPOSED_NOW','design':'LongTransportBasket wrapper with delayed post-lift checkpoints','tasks':'reuse qualified object task or LIBERO-10 t7','status':'PROPOSAL_ONLY','evidence':'this recovery report','scientific_use':'minimum Group B addition'},
 {'category':'F_NEWLY_PROPOSED_NOW','design':'TurnAccelRoute + constrained endpoint wrapper','tasks':'reuse qualified object task or LIBERO-10 t5/t8','status':'PROPOSAL_ONLY','evidence':'this recovery report','scientific_use':'minimum Group C/D addition'},
]
write_csv('HISTORICAL_TASK_IDEAS.csv', hist_rows, list(hist_rows[0]))
write('HISTORICAL_EXPERIMENT_DESIGN_RECOVERY.md', '''# Historical experiment-design recovery

The recoverable lineage is documented in current FORTE/Tabero reports, manifests, result tables, and git history. Both repositories are dirty and were treated as read-only. Current git history is shallow: it does not recover a deleted, authoritative benchmark implementation beyond the documents and result directories listed below.

Implemented evidence includes the four-task P5-S0-C/current continuous archive, the B2-R2 nine-task oracle breadth screen, P6G0 grasp-position/force screening, the legacy P5-S0-D fresh Q2F E2E run, and current Direct/probe-conditioned OOF artifacts. Partially implemented designs include the degenerate LocalLift diagnostic, missing exact 0.25 N grid, and incomplete current all-task Direct E5 chain. Planned-only designs are long transport, turning/acceleration, precise placement, mass and joint friction×mass, and LIBERO-PRO perturbations.

The key recovery decision is to preserve historical artifacts as evidence with explicit status, rather than promote every named task or method into the current benchmark. P6G0 is useful for t1/t6 qualification but does not prove π0.5/ACT grasp modes, query selection, DreamTrajectory, or LIBERO-PRO. The four current tasks remain the hidden-friction core; new tasks require a fresh qualification gate.
''')

roles = []
for t in sorted(TASKS, key=int):
    roles.append({'task_id':t,'task_name':TASKS[t]['name'],'group_A_basic_hidden_friction':'PRIMARY','group_B_long_transport':'SECONDARY_CANDIDATE_ONLY','group_C_turning_acceleration':'NOT_COVERED','group_D_precise_placement':'NOT_COVERED','group_E_mass_sensitive':'NOT_COVERED','group_F_joint_friction_mass':'NOT_COVERED','group_G_breadth_transfer':'SECONDARY_WITHIN_OBJECT_FAMILY','physics_varied_currently':'object friction only','why_keep':TASKS[t]['role'],'minimum_new_design_needed':'one qualified long/delayed-transport task and one qualified turning/constrained-placement task; separate mass×friction task for mass claims','do_not_claim':'semantic zero-shot, LocalLift comparison, mass sensitivity, or fresh current Direct all-task E2E until new matched evidence'})
write_csv('CURRENT_TASK_SCIENTIFIC_ROLE.csv', roles, list(roles[0]))

inventory = [
 ('libero_object','0','alphabet soup → basket','food object short pick-place','LOW','current core; friction-positive'),
 ('libero_object','1','cream cheese → basket','food object short pick-place','HIGH','current core; P6G0 t1 force/grasp positive'),
 ('libero_object','2','salad dressing → basket','food object short pick-place','MEDIUM','B2-R2 positive; not current archive; new qualification'),
 ('libero_object','5','tomato sauce → basket','food object short pick-place','LOW','current core; friction-positive'),
 ('libero_object','6','butter → basket','food object short pick-place','HIGH','current core; P6G0 t6 and narrow-force stress'),
 ('libero_10','5','book → caddy back compartment','longer / constrained placement','HIGH','exact instruction recovered from official task list; requires frozen-VLA reach and force qualification'),
 ('libero_10','7','alphabet soup + tomato sauce → basket','multi-object long transport','HIGH','best minimal semantic extension; delayed-failure candidate; requires new asset'),
 ('libero_10','8','two moka pots → stove','multi-object contact-rich transport','MEDIUM','force/turning candidate; feasibility gap in B2-R2; requires qualification'),
 ('libero_10','2','turn on stove + moka pot on it','turning / contact-rich placement','MEDIUM','articulated fixture and mass limitations; candidate only'),
 ('libero_10','9','mugs → microwave + close','precise placement / articulated fixture','MEDIUM','candidate only; exact force sensitivity unknown'),
 ('libero_spatial','position variant','black bowl → plate under spatial relation','position generalization','LOW','good breadth control; not a substitute for delayed transport'),
 ('libero_goal','goal variant','bowl/wine bottle/plate/cabinet goal variants','goal/semantic generalization','LOW','verify exact task-map ID and π0 compatibility before use'),
 ('libero-pro','perturbation axis','object / position / semantic / task / environment over a qualified base','generalization wrapper','UNKNOWN','official perturbation framework; requires new asset and one-axis-at-a-time evaluation'),
]
inv_rows = [{'suite':a,'task_id':b,'candidate_instruction':c,'candidate_role':d,'force_sensitivity_prior':e,'status_and_evidence':g,'current_pi0_compatibility_if_known':'UNKNOWN' if a not in ('libero_object',) else ('CURRENT' if b in ('0','1','5','6') else 'NOT_CURRENT_ARCHIVE'),'requires_new_asset':'NO' if a=='libero_object' and b in ('0','1','5','6') else 'YES','mass_support_caveat':'articulated fixtures may not support mass config; confirm asset metadata' if a in ('libero_10','libero_goal') else 'object mass not varied in current archive'} for a,b,c,d,e,g in inventory]
write_csv('LIBERO_LIBEROPRO_TASK_INVENTORY.csv', inv_rows, list(inv_rows[0]))

short = [
 {'priority':'P0','candidate':'LIBERO-10 task 7','suite':'libero_10','role':'Long Transport / delayed downstream','why':'two food objects into basket extends post-lift horizon while retaining familiar basket semantics','minimum_protocol':'fresh frozen-VLA reach; post-lift checkpoint labels; same P4-B query and force grid; 3 friction bands × force frontier','risk':'multi-object policy reach and order effects','recommendation':'PRIMARY new task if frozen VLA reaches; otherwise use a controlled LongTransportBasket wrapper on task 0 or 1'},
 {'priority':'P0','candidate':'LIBERO-10 task 5','suite':'libero_10','role':'Precise Placement','why':'book into caddy back compartment supplies a rigid, constrained endpoint without an articulated fixture','minimum_protocol':'fresh reach qualification; endpoint containment/contact labels; force and placement tolerance sweep','risk':'semantic shift from food/basket and exact map instruction must be verified','recommendation':'PRIMARY Group D candidate / reserve if task 7 fails reach qualification'},
 {'priority':'P1','candidate':'TurnAccelRoute wrapper','suite':'ActiveForcing extension','role':'Turning / acceleration','why':'controlled route with turn and acceleration segments isolates dynamic transport from object identity','minimum_protocol':'same object, fixed friction; pre-register route curvature, peak acceleration, speed; log measured force and slip/drop','risk':'new asset/controller semantics; no task ID yet','recommendation':'minimum design for Group C; must be separately qualified'},
 {'priority':'P1','candidate':'LIBERO-10 task 8','suite':'libero_10','role':'Contact-rich multi-object / turning proxy','why':'two moka pots on stove creates repeated transport/contact demands','minimum_protocol':'new current asset; fixture contact labels; one-axis physics perturbation first','risk':'B2-R2 found no feasible low-friction frontier; stove fixture and mass caveats','recommendation':'secondary candidate, not minimum until feasibility is repaired'},
 {'priority':'P2','candidate':'3×3 friction×mass on qualified task','suite':'ActiveForcing extension','role':'Mass / joint physics','why':'directly tests mass sensitivity and friction–mass interaction','minimum_protocol':'low/mid/high mass × low/mid/high friction; all metadata logged; fixed geometry/appearance/initial state','risk':'mass values and inertial effects need pre-registration; no current mass archive','recommendation':'required for mass claims, but not a replacement for Group B–D task coverage'},
]
write_csv('ADDITIONAL_TASK_SHORTLIST.csv', short, list(short[0]))
write('ADDITIONAL_TASK_SHORTLIST.md', '''# Additional task shortlist

No new task is promoted as already qualified. The minimum breadth addition is one long/delayed-transport task and one turning/constrained-placement design. The primary concrete candidate is LIBERO-10 task 7 (two food objects into a basket), with LIBERO-10 task 5 (book into a caddy compartment) as the precise-placement reserve. A controlled `TurnAccelRoute` wrapper is the cleanest way to isolate turning/acceleration; LIBERO-10 task 8 is a contact-rich proxy but currently carries a feasibility and articulated-fixture risk.

For mass claims, add a separate one-axis mass sweep and then a 3×3 friction×mass design on a task that passes the force-frontier and evaluator gates. Do not infer mass sensitivity from the current archive: current physics variation is object friction only.

Official references: [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO), [LIBERO datasets](https://libero-project.github.io/datasets), [LIBERO paper](https://arxiv.org/abs/2306.03310), [LIBERO-PRO repository](https://github.com/RLinf/LIBERO-PRO), and [LIBERO-PRO paper](https://arxiv.org/abs/2510.03827). LIBERO-PRO's object, position, semantic, task, and environment perturbations should be applied one axis at a time over a qualified base task.
''')

claims = [
 ('friction claim','fixed P4-B trace carries friction-predictive information','0,1,5,6 current archive; grouped-root OOF','SUPPORTED_WITH_SCOPE','continuous friction archive; no universal transfer claim'),
 ('force-selection claim','probe-conditioned Direct selection reduces force error/utility cost','current archive-compatible OOF','PARTIALLY_SUPPORTED','exact frozen 0.25 N grid and true no-query need fresh matched run'),
 ('delayed-failure claim','post-lift failures are a meaningful downstream stress signal','current telemetry + 145 local-lift/full-failure candidates','PARTIALLY_SUPPORTED','stage reached is observed; grip loss/drop/place cause is unavailable'),
 ('LocalLift claim','LocalLift is a valid alternative/superior label','current archive','NOT_SUPPORTED','all 720 labels local-lift positive; no negative class'),
 ('mass claim','method adapts to mass','none current','NOT_SUPPORTED','mass not varied in 720 core'),
 ('joint claim','method identifies friction×mass interaction','none current','NOT_SUPPORTED','requires 3×3 joint collection'),
 ('breadth claim','semantic transfer across tasks/suites','0,1,5,6 only','PARTIALLY_SUPPORTED','same short basket family; no clean semantic zero-shot'),
 ('current E5 claim','fresh current Direct reset-to-end all-task E2E','current closure','NOT_READY','E5 report remains IN_PROGRESS'),
]
write_csv('FRICTION_CLAIM_TO_TASK_MATRIX.csv',[{'claim':a,'claim_text':b,'task_scope':c,'status':d,'evidence_boundary':e,'needed_next_evidence':'fresh matched E2E' if d in ('PARTIALLY_SUPPORTED','NOT_READY') else 'none'} for a,b,c,d,e in claims],['claim','claim_text','task_scope','status','evidence_boundary','needed_next_evidence'])
mass_claims = [
 ('mass sensitivity','none current','NEW LongTransportBasket or qualified task','low/mid/high mass, friction fixed','not tested in current archive'),
 ('friction×mass interaction','none current','same qualified task','3×3 mu×mass factorial','pre-register mass/inertia and log all physics'),
 ('turning under load','none current','TurnAccelRoute / LIBERO-10 t8','mass × route acceleration after base qualification','separate object friction from load effects'),
 ('delayed transport failure','current 145 candidates, cause unknown','LIBERO-10 t7 or wrapper','post-lift checkpoints + force frontier','do not label as confirmed drop'),
]
write_csv('MASS_CLAIM_TO_TASK_MATRIX.csv',[{'claim':a,'current_evidence':b,'recommended_task':c,'design':d,'status_and_caveat':e} for a,b,c,d,e in mass_claims],['claim','current_evidence','recommended_task','design','status_and_caveat'])
write('MASS_TASK_RECOMMENDATIONS_FOR_AGENT_B.md', '''# Mass-task recommendations for Agent B

1. Qualify one base task first: frozen VLA reach, exact pre-probe state, force frontier, full-task labels, and post-lift checkpoint labels must all pass.
2. Run mass-only low/mid/high while holding object friction, geometry, appearance, initial state, controller, route, and query policy fixed. The actual mass values must be extracted from asset metadata and pre-registered; no absolute kg values are recovered in this lane.
3. Then run a 3×3 factorial: friction LOW/MID/HIGH × mass LOW/MID/HIGH. Record requested and realized force, measured force, acceleration/turning, contact state, slip/drop/placement events, and full success.
4. Treat mass as a real dynamical intervention: report inertia/load effects, not just a change in an object label. Articulated fixtures such as stove/microwave/cabinet may not support the same mass configuration, so use a rigid object/receptacle task for the first mass result.

Recommended order: current task 1 or 6 as a narrow-force stress control → `LIBERO-10 task 7`/LongTransportBasket for delayed transport → 3×3 joint physics. This is a recommendation for new collection, not an existing result.
''')
write('JOINT_PHYSICS_TASK_DESIGN.md', '''# Joint friction×mass task design

Use a qualified rigid-object basket task first (task 1 or task 6 as controls, then LongTransportBasket if it passes). Do not start with a stove, microwave, cabinet, or other articulated fixture until its mass-configuration support is audited.

Run LOW/MID/HIGH object friction × LOW/MID/HIGH object mass. Extract actual asset mass/inertia metadata and pre-register values; this lane does not recover authoritative kg values. Hold geometry, appearance, initial state, route, controller, horizon, query trace, and force candidates fixed. Record requested/realized/measured force, acceleration, contact state, slip/drop/placement event, and full success.

Report the 3×3 response surface, main effects, interaction contrast, force utility, and stage-specific failure rates. A mass effect requires a mass-only change under fixed friction; a joint effect requires an interaction contrast with uncertainty and reproducible labels. Do not infer either from the current 720 rows.
''')

baseline_rows=[]
E2_METRICS = {
 'GT-Direct': {'0':'SR 1.0000; mean force 4.272 N; under .0000; excess .576; utility .1455','1':'SR .9444; mean force 5.120 N; under .0000; excess .347; utility .0900','5':'SR 1.0000; mean force 4.285 N; under .0000; excess .588; utility .1429','6':'SR .8611; mean force 3.221 N; under .1389; excess .102; utility .0221'},
 'NoProbe-Direct': {'0':'SR 1.0000; mean force 4.375 N; under .0000; excess .678; utility .1250','1':'SR .9444; mean force 5.807 N; under .0000; excess 1.075; utility -.0245','5':'SR .9444; mean force 4.509 N; under .0556; excess .811; utility .0365','6':'SR .8889; mean force 3.216 N; under .1111; excess .097; utility .0579'},
 'ProbeRich-Direct': {'0':'SR 1.0000; mean force 4.194 N; under .0000; excess .498; utility .1611','1':'SR .9444; mean force 5.042 N; under .0000; excess .265; utility .1030','5':'SR .8889; mean force 4.012 N; under .1111; excess .315; utility .0623','6':'SR .8611; mean force 3.124 N; under .1389; excess .005; utility .0463'},
 'ProbeScalar-Direct': {'0':'SR 1.0000; mean force 4.289 N; under .0000; excess .592; utility .1422','1':'SR .9444; mean force 5.218 N; under .0000; excess .451; utility .0736','5':'SR .9444; mean force 4.136 N; under .0556; excess .439; utility .0993','6':'SR .8611; mean force 3.163 N; under .1389; excess .044; utility .0365'},
}
for t in sorted(TASKS, key=int):
    for b, desc, status in [('GT-Direct','ground-truth friction / oracle force selection','ARCHIVE_OOF'),('NoProbe-Direct','query-ignored / no physical information','ARCHIVE_OOF'),('ProbeRich-Direct','fixed P4-B trace, richer probe scalarization','ARCHIVE_OOF'),('ProbeScalar-Direct','fixed P4-B trace, scalar probe','ARCHIVE_OOF'),('π0-Default','frozen VLA default path','NOT_MATCHED_CURRENT'),('Tabero-Neutral','legacy neutral execution path','REDUNDANT_OR_NOT_MATCHED')]:
        baseline_rows.append({'task_id':t,'baseline':b,'description':desc,'data_status':status,'force_policy':'task-specific continuous support in archive' if 'Direct' in b else 'not comparable','full_task_metric':E2_METRICS[b][t] if b in E2_METRICS else 'not available as matched current all-task','use_in_final_matrix':'yes' if b in ('GT-Direct','NoProbe-Direct','ProbeRich-Direct','ProbeScalar-Direct') else 'diagnostic/context only','caveat':'exact 0.25 N frozen grid not archived' if 'Direct' in b else 'same checkpoint/neutral path or incomplete current chain'});
write_csv('BASELINE_TASK_MATRIX.csv', baseline_rows, list(baseline_rows[0]))

exp = [
 ('E0','Data/telemetry integrity','0,1,5,6','current archive','archive row validity, parity, hashes, trajectory phase coverage','720/720 valid; 720/720 parity; task1 reconstruction caveat','complete'),
 ('E1','Friction identification','0,1,5,6','GT friction, fixed P4-B probe, learned friction model','OOF friction error, calibration, rank correlation, grouped-root split','current archive supports grouped-root OOF information claim','complete'),
 ('E2','Full-task force feasibility','0,1,5,6','GT-Direct, NoProbe-Direct, ProbeRich-Direct, ProbeScalar-Direct','SR, mean force, under/excess, utility, force tracking','144 paired OOF rows per method; continuous draws','complete_archive_compatible'),
 ('E3','LocalLift vs FullTask','0,1,5,6','FullTask-Direct and LocalLift diagnostic','SR and label agreement by stage','LocalLift degenerate; not identifiable','blocked_by_label_design'),
 ('E4','Task breadth qualification','0,1,5,6 + candidates','fixed low / robust / oracle screening','nonzero friction and force sensitivity, delayed failure, reach rate','B2-R2 and P6G0 historical screening only','partial'),
 ('E5','Fresh current end-to-end','0,1,5,6','π0-Default, Tabero-Neutral, current Direct/query methods','reset-to-end SR, force, query cost, failure stage','current all-task matched run incomplete; legacy P5-S0-D is not substitute','required_next'),
 ('E6','Long transport / delayed failure','LIBERO-10 t7 or LongTransportBasket','fixed robust, current Direct, query methods','checkpoint retention, drop/slip/place cause, SR, force utility','new task and stage-aware evaluator required','required_new_task'),
 ('E7','Turning/acceleration + constrained placement','TurnAccelRoute + LIBERO-10 t5 reserve','fixed robust, current Direct, query methods','route/acceleration force, endpoint tolerance, SR, stage cause','new design/qualification required','required_new_task'),
 ('E8','Mass and joint physics','qualified rigid base task','mass-only then 3×3 friction×mass; same baselines','mass adaptation, interaction, load/turning failure, SR/force utility','no current mass evidence','required_new_collection'),
]
write_csv('FINAL_E0_E8_EXPERIMENT_MATRIX.csv',[{'stage':a,'purpose':b,'tasks':c,'baselines_or_methods':d,'metrics':e,'current_evidence':f,'status':g} for a,b,c,d,e,f,g in exp],['stage','purpose','tasks','baselines_or_methods','metrics','current_evidence','status'])
write('FINAL_E0_E8_EXPERIMENT_MATRIX.md', '''# Final E0–E8 experiment matrix

| Stage | Scientific question | Task set | Status |
|---|---|---|---|
| E0 | Is the archive/telemetry valid? | current four | Complete; 720 valid/parity rows |
| E1 | Does the fixed interaction trace identify friction? | current four | Complete, grouped-root OOF |
| E2 | Does force selection improve full-task utility? | current four | Archive-compatible OOF complete; exact grid gap |
| E3 | Is LocalLift a valid reduced target? | current four | Not identifiable; all LocalLift labels positive |
| E4 | Which tasks qualify for breadth? | current four + candidates | Partial; historical screens, no new current Direct qualification |
| E5 | Does the current frozen π0 → query → Direct chain work end-to-end? | current four | Required fresh matched run; incomplete |
| E6 | Does adaptation survive long/delayed transport? | LIBERO-10 t7 or wrapper | New task required |
| E7 | Does adaptation survive turning and precise placement? | TurnAccelRoute + LIBERO-10 t5 reserve | New design required |
| E8 | Does it adapt to mass and friction×mass interaction? | qualified rigid base | New mass-only and 3×3 collection required |

E2 report metrics must retain SR, mean selected force, under-force/excess-force, force tracking error, utility, and query cost. E6–E8 add stage-specific labels and physics metadata so downstream failures are not inferred from a phase reached alone.
''')

write('FINAL_EXPERIMENT_ROLLOUT_BUDGET.md', '''# Final experiment rollout budget

Budget units below are branch counts for planning, not claims about completed data.

| Block | Design | Recommended minimum | Notes |
|---|---|---:|---|
| Current E5 | 4 tasks × 6 roots × 3 friction bands × 5 exact 0.25 N force cells × 2 repeats × 4 methods | 2,880 branches | Replace continuous draws with the frozen grid; add π0/default and neutral only if truly distinct and matched |
| E6 | 1 new long-transport task × 6 roots × 3 bands × 5 force cells × 2 repeats × 4 methods | 720 branches | Add checkpoint labels; 6 roots is a minimum qualification set |
| E7 | 1 turning/constrained task × 6 roots × 3 bands × 5 force cells × 2 repeats × 4 methods | 720 branches | Keep route/acceleration fixed across methods |
| E8 mass-only | 1 qualified rigid task × 6 roots × 3 masses × 3 bands × 5 force cells × 2 repeats × 4 methods | 2,160 branches | Can be staged: mass-only before full joint design |
| E8 joint | 1 qualified rigid task × 6 roots × 3 masses × 3 friction bands × 2 repeats × 4 methods | 432 branches | Reuse exact force cells only if budget allows; otherwise pre-register a reduced frontier |

The measured sequential reference in the FORTE acceleration decision is approximately 0.035754 completed execution branches/s with one environment and one worker, before overheads and reruns. A one-week single-worker envelope is therefore about 21.6k raw branches, but the safe plan should reserve at least 25% for failed qualification, telemetry, and reruns. Do not enable vectorization or alter physics/controller semantics without a separately audited comparison.
''')

write('PAPER_EXPERIMENT_SECTION_BLUEPRINT.md', '''# Paper experiment-section blueprint

1. **Benchmark and protocol.** Define the four current LIBERO-Object food-object→basket tasks, grouped-root split, 72 contexts, 24 task-specific root families, five continuous equal-width friction-force strata, two repeats, and the fixed P4-B trace. State that the archive is a hidden-friction core.
2. **Baselines.** Compare GT-Direct, Query-Ignored/NoProbe-Direct, ProbeRich-Direct, and ProbeScalar-Direct on archive-compatible OOF. Describe π0-Default and Tabero-Neutral only as context/legacy baselines until a matched current E5 chain exists.
3. **Metrics.** Report full-task SR, selected/realized force, force tracking error, under-force, excess-force, utility, and query cost. Keep continuous-grid results distinct from exact frozen 0.25 N results.
4. **Task-role and failure limits.** Explain that all four tasks share the same short basket semantic; post-lift phase is observable, but exact drop/place cause is not. LocalLift is not a valid contrast because every current label is positive.
5. **Breadth and new tasks.** Present E4 as qualification, then E6/E7/E8 as required new evidence for long transport, turning/placement, and mass/joint claims. Do not call these completed.
6. **Generalization.** If LIBERO-PRO is used, apply one perturbation axis at a time and report base-task compatibility separately from physics adaptation.

Use [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and [LIBERO-PRO](https://github.com/RLinf/LIBERO-PRO) as benchmark references; cite the corresponding papers ([LIBERO](https://arxiv.org/abs/2306.03310), [LIBERO-PRO](https://arxiv.org/abs/2510.03827)).
''')

write('PAPER_TABLE_FIGURE_PLAN.md', '''# Paper table and figure plan

| Artifact | Content | Source/status |
|---|---|---|
| Table 1 | Current task map and scientific roles | current archive; complete |
| Table 2 | E2 baseline SR/force/utility/query metrics by task | archive-compatible OOF; complete |
| Table 3 | E0–E8 matrix and evidence status | this recovery; E5–E8 planned |
| Table 4 | Claim-to-evidence matrix | explicit supported/partial/not-supported boundaries |
| Figure 1 | Force frontier and friction bands by current task | 720 archive; empirical diagnostic |
| Figure 2 | Probe → force selection → full-task outcome pipeline | protocol schematic; distinguish observed vs planned |
| Figure 3 | Failure-stage observability: pre-lift vs post-lift candidate, with unknown cause bucket | current telemetry; no invented drop labels |
| Figure 4 | Proposed E6–E8 task taxonomy and factorial physics design | planned design, clearly marked |

Do not plot LocalLift superiority: the matched label is degenerate. Do not plot exact frozen-grid comparisons from the continuous archive as if they were exact grid results.
''')

write_csv('FINAL_CLAIM_TO_EVIDENCE_MATRIX.csv',[{'claim':a,'claim_text':b,'evidence':c,'status':d,'boundary':e,'next_test':('fresh E5 matched all-task run' if d=='NOT_READY' else 'new task/evaluator' if d=='NOT_SUPPORTED' else 'optional extension' if d=='PARTIALLY_SUPPORTED' else 'none')} for a,b,c,d,e in claims],['claim','claim_text','evidence','status','boundary','next_test'])
write('FINAL_CLAIM_TO_EVIDENCE_MATRIX.md', '''# Final claim-to-evidence matrix

The current evidence supports a scoped friction-identification claim and an archive-compatible full-task Direct feasibility result. It only partially supports probe-conditioned selection as a final end-to-end claim because the exact frozen grid, true no-query condition, and fresh current all-task E2E are incomplete. It does not support LocalLift comparison, mass adaptation, joint friction×mass interaction, or semantic zero-shot transfer. These are explicit evidence boundaries, not missing prose.
''')

write('TASK_DESIGN_HANDOFF_TO_MAIN.md', '''# Task-design handoff to Main

Final status: `ACTIVEFORCING_FINAL_BENCHMARK_DESIGN_READY`.

Keep tasks 0/1/5/6 as the current hidden-friction core. Use E1/E2 archive-compatible OOF for the scoped friction/force result. Do not claim LocalLift, mass, semantic zero-shot, exact 0.25 N grid, or current fresh all-task E2E. E5 requires a fresh matched run. Minimum breadth additions are E6 long/delayed transport and E7 turning/constrained placement; E8 is a separate mass/joint physics collection.
''')
write('TASK_DESIGN_HANDOFF_TO_MASS.md', '''# Task-design handoff to Mass

Mass claims are not supported by the current 720 archive: only object friction varies. First qualify a rigid-object base task, then run mass-only low/mid/high, followed by a 3×3 friction×mass factorial. Preserve geometry, appearance, initial state, route, controller, and query protocol; log mass/inertia and measured force. Avoid articulated fixtures for the first mass result unless asset support is verified.
''')
write('TASK_DESIGN_HANDOFF_TO_E6E7.md', '''# Task-design handoff to E6/E7

E6: qualify LIBERO-10 task 7 (two food objects to basket) or a controlled LongTransportBasket wrapper; add post-lift checkpoints and cause-specific slip/drop/place labels.

E7: implement a controlled TurnAccelRoute and qualify LIBERO-10 task 5 as the precise-placement reserve. Pre-register route curvature, speed, acceleration, endpoint tolerance, and force grid. Keep all four current tasks as short-task controls. Neither E6 nor E7 is complete in the current evidence.
''')
write_csv('FULLTASK_LOCALLIFT_TASK_SHORTLIST.csv', [
 {'task_id':t,'fulltask_label_available':'YES','locallift_label_available':'YES_BUT_DEGENERATE','local_lift_positive_rows':180,'local_lift_negative_rows':0,'fulltask_success_rows':sum(r['_success'] for r in by_task[t]),'fulltask_failure_rows':sum(1-r['_success'] for r in by_task[t]),'post_lift_full_failure_candidates':sum(1 for r in by_task[t] if r['_post'] and not r['_success']),'recommendation':'Use FullTask only; redesign LocalLift with negative examples and stage-cause labels before comparison'} for t in sorted(TASKS,key=int)], ['task_id','fulltask_label_available','locallift_label_available','local_lift_positive_rows','local_lift_negative_rows','fulltask_success_rows','fulltask_failure_rows','post_lift_full_failure_candidates','recommendation'])
write('FINAL_E2E_TASK_RECOMMENDATIONS.md', '''# Final E2E task recommendations

The current final E2E recommendation is a staged collection plan, not a claim that collection is complete.

1. Re-run current tasks 0/1/5/6 with the exact frozen 0.25 N candidate grid, true no-query condition, and a verified reset-to-end frozen π0/VLA chain. Keep the legacy P5-S0-D result labeled legacy.
2. Promote one new E6 task only after reach, pre-probe parity, force frontier, full-task success, and post-lift checkpoint labels pass. Prefer LIBERO-10 task 7; fall back to a LongTransportBasket wrapper over a current task.
3. Promote one E7 design only after route curvature/acceleration and endpoint tolerance are pre-registered. Use TurnAccelRoute plus LIBERO-10 task 5 as precise-placement reserve.
4. Run E8 mass-only and 3×3 friction×mass only on a qualified rigid base task.

The four current tasks are controls for the new designs, not substitutes for them. LocalLift must be redesigned with negative labels and cause-specific event annotations before it can re-enter the E2E matrix.
''')

summary = []
for t in sorted(TASKS, key=int):
    a = next(x for x in diag if x['grain']=='task' and x['task']==t)
    summary.append(f"| {t} | {TASKS[t]['name']} | {a['n_rows']} | {a['n_contexts']} | {a['full_success_rate']} | {a['contexts_with_both_success_and_failure']} | {a['post_lift_failure_candidates']} | {TASKS[t]['role']} |")
write('ACTIVEFORCING_FINAL_BENCHMARK_DESIGN_REPORT.md', f'''# ActiveForcing final benchmark design report

## Status

`ACTIVEFORCING_FINAL_BENCHMARK_DESIGN_READY`

## Executive answer

The current authoritative benchmark has four tasks: alphabet soup, cream cheese, tomato sauce, and butter, each picked and placed in a basket. They are a coherent hidden-friction/force-adaptation core, but they are not four distinct manipulation classes. The minimum new benchmark design is one qualified long/delayed-transport task and one qualified turning/constrained-placement task; a separate mass-only plus 3×3 friction×mass collection is required for mass claims.

## Current task map and diagnostics

| ID | Task | Rows | Contexts | Full SR | Mixed-outcome contexts | Post-lift/full-failure candidates | Scientific role |
|---|---|---:|---:|---:|---:|---:|---|
{chr(10).join(summary)}

The archive contains 720 valid rows, 72 contexts, 24 task-specific root families, five continuous equal-width force strata per context, and two repeats. Current force supports are task 0: 3–5 N, task 1: 4–6 N, task 5: 3–5 N, and task 6: 3–4 N. Current physics variation is object friction; mass is not varied.

The most important diagnostic is label scope. A post-lift phase is observable, and 145 rows are local-lift-positive/full-task-negative candidates, but the archive does not authoritatively distinguish grip loss, drop, transport, or placement failure for every row. The existing 123-row under-force and 145-row transport-candidate counts are archive-derived diagnostic buckets, not causal ground truth. LocalLift is unusable as a matched comparison because all 720 labels are positive.

## Historical design recovery

Recoverable designs include the current P5-S0-C continuous archive, B2-R2 nine-task oracle breadth, P6G0 t1/t6 grasp-force screening, and the legacy P5-S0-D fresh Q2F E2E run. The latter is useful historical context but is not current Direct all-task E5 evidence. Long transport, turning/acceleration, precise placement, mass, joint friction×mass, and LIBERO-PRO perturbations remain planned or require new qualification.

## Final benchmark taxonomy

Groups A–G are represented as follows: A (basic hidden-friction adaptation) is covered by all four; B (long transport) is not covered as a distinct semantic; C (turning/acceleration) is not covered; D (precise placement) is not covered; E (mass-sensitive) is not covered; F (joint friction+mass) is not covered; G (within-family breadth) is only partially covered. Therefore the four current tasks should remain the core, while the claim surface must stay narrower than the eventual benchmark taxonomy.

## Minimum additions

1. E6: one long/delayed transport task, preferably LIBERO-10 task 7 (two food objects into a basket) or a controlled LongTransportBasket wrapper if frozen-VLA reach fails.
2. E7: one turning/acceleration plus constrained-placement design. Use a controlled TurnAccelRoute and qualify LIBERO-10 task 5 (book to caddy back compartment) as the rigid, precise-placement reserve.
3. E8: after a rigid base task passes qualification, run mass-only low/mid/high and then a 3×3 friction×mass factorial. This is required for mass claims and is not interchangeable with E6/E7.

## Baselines and metrics

The archive-compatible matrix is GT-Direct, Query-Ignored/NoProbe-Direct, ProbeRich-Direct, and ProbeScalar-Direct. Report full-task success, selected/realized force, force-tracking error, under-force, excess-force, utility, and query cost. Exact frozen 0.25 N grid and true zero-query results require fresh matched collection. π0-Default and Tabero-Neutral are not currently safe as matched current baselines: the available report finds a redundant/neutral path and an incomplete fresh current chain.

## E0–E8 decision matrix

E0 integrity and E1 friction identification are complete. E2 is complete for archive-compatible continuous OOF. E3 is blocked by degenerate LocalLift labels. E4 is partial breadth qualification. E5 requires fresh current all-task reset-to-end E2E. E6 long transport, E7 turning/placement, and E8 mass/joint physics require new collections and stage-aware labels.

## Reproducibility and sources

All generated files in this directory are independent analysis outputs. Existing FORTE and Tabero files were read only. The benchmark inventory is grounded in the [official LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO), [LIBERO datasets](https://libero-project.github.io/datasets), [LIBERO paper](https://arxiv.org/abs/2306.03310), [official LIBERO-PRO repository](https://github.com/RLinf/LIBERO-PRO), and [LIBERO-PRO paper](https://arxiv.org/abs/2510.03827).

See `EXISTING_720_TASK_DIAGNOSTICS.csv` for machine-readable diagnostics, `FINAL_E0_E8_EXPERIMENT_MATRIX.csv` for the final stage matrix, and the three handoff files for execution ownership.
''')

print(f'WROTE {len(list(OUT.iterdir()))} files to {OUT}')
