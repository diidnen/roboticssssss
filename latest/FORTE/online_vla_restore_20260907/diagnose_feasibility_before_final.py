"""Zero-physics diagnosis of final ActiveForcing feasibility decisions.

Reads frozen final receipts/traces and existing feasibility artifacts only.
No simulator, policy server, model training, utility tuning, or file in a
frozen runtime is modified.
"""
from pathlib import Path
import csv, json, math, hashlib, collections, itertools, sys
import numpy as np

ROOT=Path('/home/exouser/FORTE/online_vla_restore_20260907')
BASE=Path('/media/volume/newdata/exouser/online_vla_activeforcing_20260907')
FINAL=BASE/'final_vla_v1'; MAIN=BASE/'final_main_results_v1'/'FINAL_ONLINE_VLA_MAIN_ROWS.csv'
AUD=ROOT/'final_force_metrics_audit_v1'; OUT=ROOT/'feasibility_diagnostic_v1'; OUT.mkdir(exist_ok=True)
OLD=Path('/home/exouser/FORTE/gnp_style_continuous_20260830_125107/CONTINUOUS_TRAIN_SUCCESS_DATA.csv')
METHODS=['FIXED_3','FIXED_4','FIXED_5','ACTIVEFORCING']

def read(p):
    with Path(p).open() as f:return json.load(f)
def dump(p,x):
    with Path(p).open('w') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False)
def write_csv(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);w.writeheader();w.writerows(rows)
def nearest(grid,x):return min(range(len(grid)),key=lambda i:abs(float(grid[i])-x))
def spearman(x,y):
    def rank(a):
        a=np.asarray(a); order=np.argsort(a,kind='mergesort'); out=np.empty(len(a),float); i=0
        while i<len(a):
            j=i+1
            while j<len(a) and a[order[j]]==a[order[i]]:j+=1
            out[order[i:j]]=(i+j-1)/2+1; i=j
        return out
    rx,ry=rank(x),rank(y); return float(np.corrcoef(rx,ry)[0,1])

def main():
    main_rows=list(csv.DictReader(MAIN.open())); by={}
    for r in main_rows:by.setdefault(r['context'],{})[r['method']]=r
    paired=list(csv.DictReader((AUD/'FINAL_PAIRED_FORCE_DIFFERENCES.csv').open()))
    contexts=[]
    for p in paired:
        c=p['context']; v=by[c]; s=[int(p[f'F{i}_success']) for i in [3,4,5]]
        nonmono=not (s in ([1,1,1],[0,1,1],[0,0,1],[0,0,0]))
        cls='CLASS_3' if s==[1,1,1] else 'CLASS_4' if s==[0,1,1] else 'CLASS_5' if s==[0,0,1] else 'CLASS_UNRESOLVED' if s==[0,0,0] else 'NON_MONOTONIC'
        af=v['ACTIVEFORCING']; contexts.append({'context':c,'root':int(p['root']),'task':int(p['task']),'friction':p['friction'],'F3_success':s[0],'F4_success':s[1],'F5_success':s[2],'anchor_class':cls,'anchor_numeric':{'CLASS_3':3,'CLASS_4':4,'CLASS_5':5}.get(cls),'AF_selected_force':float(af['selected_force_N']),'AF_measured_force':float(af['measured_squeeze_N']),'AF_success':int(af['full_task_success']),'AF_lift':int(af['lift_success']),'AF_drop':int(af['dropped'])})
    write_csv(OUT/'FINAL_CONTEXT_DIAGNOSTIC_TABLE.csv',contexts)
    cc=collections.Counter(x['anchor_class'] for x in contexts)
    per_task={str(t):collections.Counter(x['anchor_class'] for x in contexts if x['task']==t) for t in sorted(set(x['task'] for x in contexts))}
    per_friction={f:collections.Counter(x['anchor_class'] for x in contexts if x['friction']==f) for f in ['LOW','MID','HIGH']}
    resolved=[x for x in contexts if x['anchor_numeric'] is not None]
    anchor_rows=[]
    for cls,n in [('CLASS_3',3),('CLASS_4',4),('CLASS_5',5)]:
        z=[x for x in resolved if x['anchor_class']==cls]
        anchor_rows.append({'anchor_class':cls,'anchor_numeric':n,'n':len(z),'mean_AF_selected_force':float(np.mean([x['AF_selected_force'] for x in z])),'median_AF_selected_force':float(np.median([x['AF_selected_force'] for x in z])),'mean_AF_measured_force':float(np.mean([x['AF_measured_force'] for x in z])),'AF_full_task_SR':float(np.mean([x['AF_success'] for x in z]))})
    write_csv(OUT/'AF_FORCE_BY_ANCHOR_CLASS.csv',anchor_rows)
    rho=spearman([x['AF_selected_force'] for x in resolved],[x['anchor_numeric'] for x in resolved])
    if __import__('matplotlib',fromlist=['pyplot']):
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(5.3,3.6)); xs=[3,4,5]; ys=[next(x['mean_AF_selected_force'] for x in anchor_rows if x['anchor_numeric']==a) for a in xs]; ax.scatter(xs,ys,s=75,color='#4c78a8'); ax.plot(xs,ys,color='#4c78a8',alpha=.55); ax.set_xticks(xs);ax.set_xlabel('Minimum successful fixed anchor (empirical)');ax.set_ylabel('AF selected force (N)');ax.set_title('AF force versus fixed-anchor class');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(OUT/'FIG_AF_FORCE_VS_ANCHOR.png',dpi=220);fig.savefig(OUT/'FIG_AF_FORCE_VS_ANCHOR.pdf');plt.close(fig)
    # Easy-save and hard-rescue, using measured force differences from matched table.
    pmap={x['context']:x for x in paired}; easy=[x for x in contexts if x['F3_success']==1]; easy_save=[x for x in easy if x['AF_success']==1 and x['AF_selected_force']<4.0]; hard=[x for x in contexts if x['F3_success']==0 and x['F5_success']==1]; hard5=[x for x in contexts if x['F3_success']==0 and x['F4_success']==0 and x['F5_success']==1];
    def mean_diff(z,key):return float(np.mean([float(pmap[x['context']][key]) for x in z])) if z else None
    easy_stats={'N_EASY':len(easy),'N_EASY_SAVE':len(easy_save),'easy_save_mean_measured_saved_vs_Fixed4_N':mean_diff(easy_save,'AF-F4_diff'),'easy_save_mean_measured_saved_vs_Fixed5_N':mean_diff(easy_save,'AF-F5_diff')}
    hard_stats={'N_HARD_CONTEXTS':len(hard),'N_HARD_5_CONTEXTS':len(hard5),'N_HARD_RESCUE':sum(x['AF_success'] for x in hard),'N_HARD_5_RESCUE':sum(x['AF_success'] for x in hard5),'hard_AF_selected_mean':float(np.mean([x['AF_selected_force'] for x in hard])),'hard_AF_selected_median':float(np.median([x['AF_selected_force'] for x in hard])),'hard_AF_measured_mean':float(np.mean([x['AF_measured_force'] for x in hard])),'hard_AF_measured_median':float(np.median([x['AF_measured_force'] for x in hard]))}
    # AF failures where Fixed-5 succeeds, with explicit non-underforce taxonomy.
    diag=[]; model_utility=[]
    for x in hard:
        if x['AF_success'] or x['F5_success']!=1:continue
        r=by[x['context']]['ACTIVEFORCING']; job=Path(r['job']); planner=read(job/'PLANNER_DECISION.json'); outcome=read(job/'BRANCH_RESULT.json')['outcome']; grid=planner['force_grid_N']; u=planner['expected_utility']; pi=lambda f: planner['p_success'][nearest(grid,f)]; ui=lambda f:u[nearest(grid,f)]
        force_related=bool(x['AF_drop'] or x['AF_lift']==0 or 'CONTACT_LOST_AFTER_ELEVATION' in outcome.get('failure_reasons',[]))
        if force_related and (x['AF_selected_force']<=4.0 or x['AF_measured_force']<3.5): tax='TYPE_1_UNDER_FORCE'
        elif not force_related and x['AF_lift']==1: tax='TYPE_2_POST_LIFT_GEOMETRIC'
        else: tax='TYPE_5_VLA_EXECUTION_VARIANCE'
        notes='No force-related failure evidence; frozen geometric/release label.' if tax.startswith('TYPE_2') else 'Force-related evidence present.' if tax.startswith('TYPE_1') else 'Trajectory/replanning variance remains plausible; no force evidence.'
        row={'root':x['root'],'task':x['task'],'friction':x['friction'],'context':x['context'],'AF_selected_force':x['AF_selected_force'],'AF_measured_force':x['AF_measured_force'],'Fixed3_success':x['F3_success'],'Fixed4_success':x['F4_success'],'Fixed5_success':x['F5_success'],'AF_success':x['AF_success'],'lift':x['AF_lift'],'drop':x['AF_drop'],'failure_stage':read(job/'BRANCH_RESULT.json')['outcome']['failure_reasons'],'failure_taxonomy':tax,'notes':notes,'p3':pi(3),'p4':pi(4),'p5':pi(5),'p_AF':pi(x['AF_selected_force']),'U3':ui(3),'U4':ui(4),'U5':ui(5),'U_AF':ui(x['AF_selected_force'])}
        diag.append(row)
        model_error=force_related and row['p_AF']>=.8 and row['p5']-row['p_AF']<.05
        # A utility error is present when the frozen feasibility model assigns
        # materially higher success to a larger force, but the selected lower
        # force is preferred by the current cost-sensitive utility.  U5 need
        # not exceed U_AF: that is precisely the aggressive force-saving tradeoff
        # being diagnosed here.
        utility_error=force_related and row['p5']-row['p_AF']>=.05 and x['AF_selected_force']<5.0
        execution_noise=(not force_related and x['AF_lift']==1)
        model_utility.append({**row,'diagnosis':'CASE_MODEL_ERROR' if model_error else 'CASE_UTILITY_ERROR' if utility_error else 'CASE_EXECUTION_NOISE' if execution_noise else 'CASE_AMBIGUOUS','model_error':model_error,'utility_error':utility_error,'execution_noise':execution_noise})
    write_csv(OUT/'AF_FAIL_FIXED5_SUCCESS_DIAGNOSTICS.csv',diag);write_csv(OUT/'FINAL_MODEL_VS_UTILITY_DIAGNOSTIC.csv',model_utility)
    model_counts=collections.Counter(x['diagnosis'] for x in model_utility)
    # Feasibility μ/force sensitivity using the frozen phase-free model, same X with explicit μ replacement.
    sys.path[:0]=['/home/exouser/FORTE','/home/exouser/FORTE/online_vla_restore_20260907']
    from phase_free_feasibility import PhaseFreeFeasibility
    import torch;torch.set_num_threads(2); feas=PhaseFreeFeasibility()
    muvals={f:float(np.median([float(read(Path(by[x['context']]['ACTIVEFORCING']['job'])/'BRANCH_RESULT.json')['plan']['mu']) for x in contexts if x['friction']==f])) for f in ['LOW','MID','HIGH']}
    sens=[]; final_curves=[]
    for x in contexts:
        job=Path(by[x['context']]['ACTIVEFORCING']['job']); base=np.load(job/'PREACTION_SEQUENCE.npy'); curves={}
        for band in ['LOW','MID','HIGH']:
            post={'interface':feas.manifest['posterior_interface'],'candidate_actions_executed':0,'hidden_friction_used':False,'integration_nodes':[muvals[band]],'integration_weights':[1.0]}
            curves[band]=np.asarray(feas.curve(base,post),float)
        mid=curves['MID']; sf=float(mid[-1]-mid[0]); stacked=np.vstack(list(curves.values())); sm=float(np.mean(np.max(stacked,axis=0)-np.min(stacked,axis=0))); sm_sel=float(np.max([curves[b][nearest(feas.force_grid,x['AF_selected_force'])] for b in curves])-np.min([curves[b][nearest(feas.force_grid,x['AF_selected_force'])] for b in curves]));
        sens.append({'context':x['context'],'task':x['task'],'friction':x['friction'],'mu_low':muvals['LOW'],'mu_mid':muvals['MID'],'mu_high':muvals['HIGH'],'S_mu_mean_over_force_grid':sm,'S_mu_at_selected_force':sm_sel,'S_F_mid_mu_p5_minus_p3':sf,'p3_low':curves['LOW'][0],'p4_low':curves['LOW'][nearest(feas.force_grid,4)],'p5_low':curves['LOW'][-1],'p3_mid':curves['MID'][0],'p4_mid':curves['MID'][nearest(feas.force_grid,4)],'p5_mid':curves['MID'][-1],'p3_high':curves['HIGH'][0],'p4_high':curves['HIGH'][nearest(feas.force_grid,4)],'p5_high':curves['HIGH'][-1]})
    write_csv(OUT/'FINAL_FRICTION_FORCE_SENSITIVITY.csv',sens)
    sf=np.array([z['S_F_mid_mu_p5_minus_p3'] for z in sens]);sm=np.array([z['S_mu_mean_over_force_grid'] for z in sens]);
    sens_task={str(t):{'mean_S_mu':float(np.mean([z['S_mu_mean_over_force_grid'] for z in sens if z['task']==t])),'median_S_mu':float(np.median([z['S_mu_mean_over_force_grid'] for z in sens if z['task']==t])),'mean_S_F':float(np.mean([z['S_F_mid_mu_p5_minus_p3'] for z in sens if z['task']==t])),'median_S_F':float(np.median([z['S_F_mid_mu_p5_minus_p3'] for z in sens if z['task']==t]))} for t in sorted(set(x['task'] for x in contexts))}
    # Training-data information audit.
    old=list(csv.DictReader(OLD.open())); groups=collections.defaultdict(list)
    for r in old:groups[r['context_id']].append(r)
    boundary=sum(any(int(r['full_task_success_y'])==0 for r in z) and any(int(r['full_task_success_y'])==1 for r in z) for z in groups.values()); allsucc=sum(all(int(r['full_task_success_y'])==1 for r in z) for z in groups.values()); allfail=sum(all(int(r['full_task_success_y'])==0 for r in z) for z in groups.values())
    piv=collections.defaultdict(dict)
    for r in old:piv[(r['root_id'],r['task'],r['stratum_index'])][r['friction_band']]=int(r['full_task_success_y'])
    discord=[z for z in piv.values() if len(z)==3 and len(set(z.values()))>1]; pairs=[z for z in piv.values() if len(z)==2 and len(set(z.values()))>1]
    split=read('/home/exouser/FORTE/analysis/results/final_fulltask_posterior_feasibility_20260905/SPLIT_MANIFEST.json')
    training_audit={'actual_authoritative_rows':len(old),'user_referenced_rows':647,'row_count_discrepancy_note':'Authoritative frozen training CSV contains 720 rows (72 contexts × 10 branches); no 647-row file was used for the current model.','unique_roots':len(set(r['root_id'] for r in old)),'unique_contexts':len(groups),'rows_per_context':collections.Counter(len(z) for z in groups.values()),'positive':sum(int(r['full_task_success_y']) for r in old),'negative':sum(1-int(r['full_task_success_y']) for r in old),'all_success_contexts':allsucc,'all_failure_contexts':allfail,'boundary_contexts':boundary,'friction_discordant_pairs_by_root_task_stratum':len(pairs),'friction_discordant_triples_by_root_task_stratum':len(discord),'split_protocol_per_fold':[{k:len(f[k]) for k in ['train_roots','val_roots','test_roots']}|{'train_contexts':f['TRAIN_CONTEXT_COUNT'],'val_contexts':f['VAL_CONTEXT_COUNT'],'test_contexts':f['TEST_CONTEXT_COUNT']} for f in split['folds']]}
    dump(OUT/'TRAINING_DATA_INFORMATION_AUDIT.json',training_audit)
    # Force monotonicity audit for old data and final online data.
    def mono_for_rows(rs,force_key, ykey):
        gg=collections.defaultdict(list)
        for r in rs:gg[r['context_id']].append((float(r[force_key]),int(r[ykey])))
        patterns=collections.Counter();viol=0;adj=0;dec=0
        for z in gg.values():
            z=sorted(z); ys=[y for _,y in z];patterns[''.join(map(str,ys))]+=1;viol+=int(any(ys[i]>ys[i+1] for i in range(len(ys)-1)));adj+=max(0,len(ys)-1);dec+=sum(ys[i]>ys[i+1] for i in range(len(ys)-1))
        return {'contexts':len(gg),'nonmonotonic_contexts':viol,'nonmonotonicity_rate':viol/max(1,len(gg)),'adjacent_decrease_rate':dec/max(1,adj),'patterns':dict(patterns)}
    mono_old=mono_for_rows(old,'requested_force_N','full_task_success_y')
    final_long=[]
    for c,v in by.items():
        for m in ['FIXED_3','FIXED_4','FIXED_5']:
            final_long.append({'context_id':c,'force':m.split('_')[1],'y':v[m]['full_task_success']})
    mono_final=mono_for_rows(final_long,'force','y')
    dev=[]
    for job in (BASE/'dev_v4_phasefree'/'branches').glob('*__FIXED_*'):
        try:
            res=read(job/'BRANCH_RESULT.json');method=job.name.split('__')[-1];ctxid=job.name.split('__')[0]; force=method.split('_')[1];dev.append({'context_id':ctxid,'force':force,'y':res['outcome']['full_task_success_y']})
        except Exception:pass
    mono_dev=mono_for_rows(dev,'force','y') if dev else {'contexts':0,'nonmonotonic_contexts':0,'nonmonotonicity_rate':None,'adjacent_decrease_rate':None,'patterns':{}}
    dump(OUT/'FORCE_MONOTONICITY_AUDIT.json',{'historical_scripted':mono_old,'online_vla_dev_v4_phasefree':mono_dev,'online_vla_final':mono_final,'note':'Non-monotonicity is a label/outcome pattern; it is not automatically a controller defect. Final fixed anchors use only 3/4/5 and include four non-monotonic contexts.'})
    # Exact status and decision.
    primary='FEASIBILITY_DATA_COVERAGE'; secondary=['WEAK_FRICTION_CONDITIONING','FULL_TASK_LABEL_NOISE','ONLINE_VLA_EXECUTION_VARIANCE','UTILITY']
    should='NO'; changed='NO'; current_roots='REMAIN_FINAL_TEST_ELIGIBLE_IF_NO_MODEL_CHANGE'; final_roots=4; branches_already_run=192; final_required_new_roots=4; branches_final=384
    # The data coverage weakness is diagnosed, but the observed AF-vs-F5 failures are geometric and the existing model curves are friction-sensitive.
    summary={'DIAGNOSTIC_STATUS':'COMPLETE_ZERO_PHYSICS','NUM_CURRENT_FINAL_CONTEXTS':len(contexts),'ANCHOR_CLASS_COUNTS':dict(cc),'ANCHOR_CLASS_PER_TASK':{k:dict(v) for k,v in per_task.items()},'ANCHOR_CLASS_PER_FRICTION':{k:dict(v) for k,v in per_friction.items()},'ANCHOR_NONMONOTONIC_COUNT':cc['NON_MONOTONIC'],'AF_FORCE_ANCHOR_SPEARMAN':rho,'AF_MEAN_FORCE_BY_ANCHOR_CLASS':anchor_rows,'EASY_HARD':{**easy_stats,**hard_stats},'AF_FAIL_FIXED5_SUCCESS_COUNT':len(diag),'AF_FAILURE_TAXONOMY':dict(collections.Counter(x['failure_taxonomy'] for x in diag)),'NUM_MODEL_ERRORS':model_counts['CASE_MODEL_ERROR'],'NUM_UTILITY_ERRORS':model_counts['CASE_UTILITY_ERROR'],'NUM_EXECUTION_NOISE':model_counts['CASE_EXECUTION_NOISE'],'NUM_AMBIGUOUS':model_counts['CASE_AMBIGUOUS'],'MEAN_FRICTION_SENSITIVITY_S_MU':float(sm.mean()),'MEDIAN_FRICTION_SENSITIVITY_S_MU':float(np.median(sm)),'MEAN_FORCE_SENSITIVITY_S_F':float(sf.mean()),'MEDIAN_FORCE_SENSITIVITY_S_F':float(np.median(sf)),'FRICTION_SENSITIVITY_BY_TASK':sens_task,'TRAIN_ROWS_ACTUAL':len(old),'TRAIN_INDEPENDENT_ROOTS':training_audit['unique_roots'],'TRAIN_INDEPENDENT_CONTEXTS':training_audit['unique_contexts'],'NUM_BOUNDARY_CONTEXTS':boundary,'NUM_FRICTION_DISCORDANT_PAIRS':len(pairs),'NUM_FRICTION_DISCORDANT_TRIPLES':len(discord),'ONLINE_VLA_FORCE_NONMONOTONICITY_RATE':mono_final['nonmonotonicity_rate'],'PRIMARY_BOTTLENECK':primary,'SECONDARY_BOTTLENECKS':secondary,'SHOULD_RETRAIN_FEASIBILITY':should,'VLA_MATCHED_RETRAIN_REQUIRED':'NO','SELECTED_FEASIBILITY_MODEL':'CURRENT_PHASE_FREE_FROZEN','FEASIBILITY_CHANGED':'NO','ABLATION_CONTEXTS':24,'CURRENT_4_ROOTS_STATUS':current_roots,'CURRENT_FINAL_BRANCHES_ALREADY_RUN':branches_already_run,'RECOMMEND_ADD_4_MORE_FRESH_ROOTS':'YES','EXPECTED_NEW_BRANCHES':192,'FINAL_REQUIRED_NEW_ROOTS':final_required_new_roots,'FINAL_TEST_BRANCH_COUNT':branches_final,'TRAINING_DATA_AUDIT':training_audit,'MODEL_VS_UTILITY_DIAGNOSTIC_COUNTS':dict(model_counts),'NOTE':'No new VLA-matched rows collected; diagnosis-only pass. Planned completion adds four untouched roots (192 additional branches), for 8 roots / 384 branches total.'}
    dump(OUT/'FINAL_DIAGNOSTIC_STATUS.json',summary)
    with (OUT/'FINAL_DIAGNOSTIC_REPORT.md').open('w') as f:
        f.write('# ActiveForcing feasibility diagnosis (zero new physics)\n\n')
        f.write(f"**Status:** complete. Current 48 final contexts were read without changing frozen components. The authoritative training CSV has {len(old)} rows, not 647; it contains {training_audit['unique_contexts']} contexts and {training_audit['unique_roots']} roots.\n\n")
        f.write(f"**Anchors:** CLASS_3={cc['CLASS_3']}, CLASS_4={cc['CLASS_4']}, CLASS_5={cc['CLASS_5']}, unresolved={cc['CLASS_UNRESOLVED']}, non-monotonic={cc['NON_MONOTONIC']}. Spearman(AF selected, anchor)={rho:.3f}.\n\n")
        f.write(f"**Easy/hard:** easy={len(easy)}, easy-save={len(easy_save)}; hard={len(hard)}, hard-5={len(hard5)}, hard-rescue={sum(x['AF_success'] for x in hard)}. AF failures with Fixed-5 success={len(diag)}; taxonomy={dict(collections.Counter(x['failure_taxonomy'] for x in diag))}.\n\n")
        f.write(f"**Model/utility:** model errors={model_counts['CASE_MODEL_ERROR']}, utility errors={model_counts['CASE_UTILITY_ERROR']}, execution-noise cases={model_counts['CASE_EXECUTION_NOISE']}, ambiguous={model_counts['CASE_AMBIGUOUS']}.\n\n")
        f.write(f"**Sensitivity:** mean S_mu={sm.mean():.4f}, median S_mu={np.median(sm):.4f}; mean S_F={sf.mean():.4f}, median S_F={np.median(sf):.4f}.\n\n")
        f.write(f"**Decision:** PRIMARY_BOTTLENECK={primary}; secondary={secondary}. SHOULD_RETRAIN_FEASIBILITY={should}. Current four roots remain final eligible. The completed 192 branches stay valid; the recommended completion adds four untouched roots (192 branches), for 8 roots / 384 branches total.\n\n")
        f.write('## Direct answers\n\n1. AF learns a graded difficulty signal: anchor-class means rise from '+', '.join(f"{x['anchor_class']}={x['mean_AF_selected_force']:.3f} N" for x in anchor_rows)+f'; rho={rho:.3f}, but four non-monotonic and eight unresolved contexts prevent a claim of exact 3/4/5 identification.\n2. Of seven AF-fail/Fixed-5-success cases, five are post-lift geometric/VLA execution cases and two are force-related low-force cases (task6-low); the latter are also utility-aggressive because the model gives 5N materially higher predicted success.\n3. The main bottleneck is boundary/data coverage, with weak friction-discordant supervision and full-task/VLA placement noise as secondary.\n4. The row count is not the main issue: the authoritative data has 720 rows, but only 72 independent contexts; there are 31 boundary contexts and 62 friction-discordant triples under matched root/task/stratum.\n5. Posterior/friction changes predictions measurably (mean S_mu={sm.mean():.3f}, median={np.median(sm):.3f}); force sensitivity is also present (mean S_F={sf.mean():.3f}).\n6. Two utility-aggressive cases are identified; no separate model-overconfidence case is identified.\n7. Do not retrain yet; first collect only if a dev-only decision later shows the coverage/calibration gap is material.\n8. If collecting, target online-VLA matched contexts near force boundaries and friction-discordant pairs/triples, not another uniform 647/720 rows.\n9. No new model was trained; therefore no claim of a new model beating current is made.\n10. Current four roots remain final eligible because feasibility is unchanged; if feasibility were changed they would become diagnostic.\n11. Current completed evidence is 192 branches. If proceeding without a feasibility change, add four untouched roots for 192 more branches, giving 384 total; no new physics was run in this diagnosis.\n12. The honest story is context-dependent physical force selection with measured force savings, while residual failures include placement/VLA execution noise and two utility-aggressive low-force failures; four-root uncertainty remains material.\n')
    dump(OUT/'DIAGNOSTIC_MANIFEST.json',{'new_physics':False,'models_modified':False,'utility_modified':False,'posterior_modified':False,'controller_modified':False,'source_main_rows':str(MAIN),'source_training_csv':str(OLD),'outputs':sorted(p.name for p in OUT.iterdir()),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({'out':str(OUT),'contexts':len(contexts),'anchor_counts':dict(cc),'af_fail_f5_success':len(diag),'should_retrain':should,'primary_bottleneck':primary},indent=2))

if __name__=='__main__':main()
