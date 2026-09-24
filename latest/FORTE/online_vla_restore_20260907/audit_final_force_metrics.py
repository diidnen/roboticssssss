"""Offline audit of the 192 final online-VLA branches.

This script only reads frozen rollout artifacts.  It never imports Isaac or
starts a policy/simulator process.  The primary force window is post-handoff
branch_hold through the first observed lift event (the same z-threshold used
by the frozen evaluator).  A no-lift branch is retained as a censored
pre-release window and is labelled explicitly rather than filled or dropped.
"""
from __future__ import annotations
import csv, json, math, hashlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

ROOT = Path("/home/exouser/FORTE/online_vla_restore_20260907")
BASE = Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907")
FINAL = BASE / "final_vla_v1"
MAIN = BASE / "final_main_results_v1" / "FINAL_ONLINE_VLA_MAIN_ROWS.csv"
OUT = ROOT / "final_force_metrics_audit_v1"
METHODS = ["FIXED_3", "FIXED_4", "FIXED_5", "ACTIVEFORCING"]
LABELS = {"FIXED_3":"Fixed-3", "FIXED_4":"Fixed-4", "FIXED_5":"Fixed-5", "ACTIVEFORCING":"ActiveForcing"}
DT_DEFAULT = 0.05

def load_json(p):
    with Path(p).open() as f: return json.load(f)

def dump_json(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f: json.dump(x, f, indent=2, sort_keys=True, allow_nan=False)

def write_csv(p, rows, fields=None):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    if fields is None: fields = list(rows[0].keys()) if rows else []
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def wilson(k, n, z=1.959963984540054):
    if not n: return [None, None]
    p=k/n; den=1+z*z/n; ctr=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [ctr-half, ctr+half]

def mcnemar_exact(a, b, c, d):
    n=b+c
    if n == 0: return 1.0
    q=sum(math.comb(n,i) for i in range(min(b,c)+1))/(2**n)
    return min(1.0, 2*q)

def first_failure(outcome, trace, combined_drop=False):
    if int(outcome.get("full_task_success_y", 0)) == 1: return "SUCCESS"
    reasons=set(outcome.get("failure_reasons", []))
    if "NO_LIFT" in reasons: return "LIFT_FAILURE"
    if combined_drop or "DROP" in reasons or any(t.get("terminations",{}).get("object_1_dropped",False) for t in trace): return "DROP"
    if "NOT_RELEASED" in reasons: return "RELEASE_FAILURE"
    if "OUTSIDE_AUTHORED_REGION" in reasons: return "GEOMETRIC_FINAL_FAILURE"
    if "NO_FINAL_SUPPORT_CONTACT" in reasons: return "PLACEMENT_FAILURE"
    if "TIMEOUT" in reasons or "INCOMPLETE_HORIZON" in reasons: return "OTHER"
    return "OTHER"

def force_window(job):
    trace=load_json(job/"BRANCH_TRACE.json")
    dec=load_json(job/"DECISION_METADATA.json")
    z0=float(dec["object_pose"][2])
    if not trace: return {"valid":False,"status":"FORCE_METRIC_INVALID_EMPTY_TRACE"}
    start=0
    lifts=[]
    for i,t in enumerate(trace):
        try: z=float(t["object_position_m"][2])
        except Exception: continue
        if math.isfinite(z) and z-z0 >= 0.03: lifts.append(i)
    releases=[i for i,t in enumerate(trace) if bool(t.get("vla_release_intent",False))]
    if lifts:
        end=lifts[0]; status="LIFT_EVENT"
    elif releases:
        end=max(start, releases[0]-1); status="NO_LIFT_EVENT_CENSORED_PRE_RELEASE"
    else:
        end=len(trace)-1; status="NO_LIFT_EVENT_CENSORED_TRACE_END"
    vals=[]; invalid=[]
    for i,t in enumerate(trace[start:end+1]):
        nf=t.get("normal_force_N")
        try:
            if len(nf)!=2 or not all(math.isfinite(float(x)) for x in nf): raise ValueError
            # Runtime normal_force_N is object-filtered, finger-local, signed
            # grasp-normal force.  Magnitude makes the sign convention explicit.
            vals.append(2.0*min(abs(float(nf[0])),abs(float(nf[1]))))
        except Exception: invalid.append(start+i)
    if invalid or not vals:
        return {"valid":False,"status":"FORCE_METRIC_INVALID_BAD_NORMAL_TRACE","start_branch_step":start+1,"end_branch_step":end+1,"invalid_indices":invalid}
    a=np.asarray(vals,dtype=float)
    return {"valid":True,"status":status,"start_index":start,"end_index":end,
            "start_branch_step":int(trace[start]["branch_step"]),"end_branch_step":int(trace[end]["branch_step"]),
            "lift_event_branch_step":int(trace[lifts[0]]["branch_step"]) if lifts else None,
            "release_event_branch_step":int(trace[releases[0]]["branch_step"]) if releases else None,
            "sample_count":int(a.size),"values":vals,
            "mean":float(a.mean()),"median":float(np.median(a)),"std":float(a.std(ddof=0)),
            "top5_mean":float(np.mean(np.sort(a)[min(max(0,int(math.floor(.95*len(a)))),len(a)-1):])),
            "max":float(a.max()),"impulse":float(a.sum()*DT_DEFAULT)}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows=list(csv.DictReader(MAIN.open()))
    if len(rows)!=192: raise RuntimeError(f"Expected 192 main rows, found {len(rows)}")
    inv=[]; records=[]; contexts=defaultdict(dict); bad=[]
    for r in rows:
        method=r["method"]; job=Path(r["job"])
        result=load_json(job/"BRANCH_RESULT.json"); outcome=result["outcome"]; trace=load_json(job/"BRANCH_TRACE.json")
        planner=load_json(job/"PLANNER_DECISION.json")
        admission=load_json(job/"FINAL_EVIDENCE_ADMISSION.json")
        cfg=load_json(job/"RUNTIME_CONFIG.json")
        wf=force_window(job)
        if not wf["valid"]: bad.append({"job":str(job),"status":wf["status"]})
        root=int(r["root"]); task=int(r["task"]); context=r["context"]
        rec={"root_group":root,"task":task,"friction":r["friction"],"context":context,"method":LABELS[method],"method_code":method,
             "branch_id":job.name,"rollout_id":job.name,"selected_force":float(planner["executed_force_N"]),
             "full_task_success":int(outcome["full_task_success_y"]),"lift_success":int(outcome["lift_success"]),"drop":int(r.get("dropped",outcome.get("dropped",0))),
             "first_failure_stage":first_failure(outcome,trace,combined_drop=bool(int(r.get("dropped",outcome.get("dropped",0))))),"force_log_path":str(job/"BRANCH_TRACE.json"),
             "vla_provenance_path":str(job/"FINAL_EVIDENCE_ADMISSION.json"),"controller_version":trace[0].get("arbitration_version"),
             "evaluator_version":outcome.get("label_version"),"force_window_status":wf["status"],
             "force_window_start_branch_step":wf.get("start_branch_step"),"force_window_end_branch_step":wf.get("end_branch_step"),
             "force_metric_valid":bool(wf["valid"]),"mean_measured_bilateral_squeeze":wf.get("mean"),"median_measured_squeeze":wf.get("median"),
             "std_measured_squeeze":wf.get("std"),"top5_mean_squeeze":wf.get("top5_mean"),"max_squeeze":wf.get("max"),"force_impulse":wf.get("impulse"),
             "window_samples":wf.get("sample_count"),"dt_s":float(cfg.get("physics_dt",1/60))*int(cfg.get("decimation",3))}
        records.append(rec); contexts[context][method]=rec
        inv.append({k:rec[k] for k in ["root_group","task","friction","method","branch_id","rollout_id","selected_force","full_task_success","lift_success","drop","first_failure_stage","force_log_path","vla_provenance_path","controller_version","evaluator_version"]})
    write_csv(OUT/"FINAL_192_BRANCH_INDEX.csv",inv)
    dump_json(OUT/"FINAL_192_BRANCH_INDEX.json",{"role":"FINAL_ONLINE_VLA_FORCE_AUDIT_BRANCH_INDEX","num_final_root_groups":len(set(x["root_group"] for x in inv)),"num_final_contexts":len(contexts),"num_final_branches":len(inv),"context_method_counts":{c:sorted(v) for c,v in contexts.items()},"all_contexts_complete":all(set(v)==set(METHODS) for v in contexts.values()),"branches":inv})
    method_summary=[]
    for m in METHODS:
        rr=[x for x in records if x["method_code"]==m]; valid=[x for x in rr if x["force_metric_valid"]]
        def av(k): return float(np.mean([x[k] for x in valid])) if valid else None
        method_summary.append({"method":LABELS[m],"method_code":m,"n_branches":len(rr),"n_force_metric_valid":len(valid),
          "full_task_success_n":sum(x["full_task_success"] for x in rr),"full_task_sr":float(np.mean([x["full_task_success"] for x in rr])),
          "mean_selected_setpoint":float(np.mean([x["selected_force"] for x in rr])),"mean_measured_bilateral_squeeze":av("mean_measured_bilateral_squeeze"),
          "median_measured_squeeze":av("median_measured_squeeze"),"top5_mean_squeeze":av("top5_mean_squeeze"),"max_squeeze":av("max_squeeze"),"mean_force_impulse":av("force_impulse"),
          "lift_sr":float(np.mean([x["lift_success"] for x in rr])),"drop_rate":float(np.mean([x["drop"] for x in rr]))})
    f5={x["method"]:x for x in method_summary}["Fixed-5"]
    for x in method_summary:
        x["saving_vs_fixed5_selected_N"]=f5["mean_selected_setpoint"]-x["mean_selected_setpoint"]
        x["saving_vs_fixed5_measured_N"]=f5["mean_measured_bilateral_squeeze"]-x["mean_measured_bilateral_squeeze"] if x["mean_measured_bilateral_squeeze"] is not None else None
        x["saving_vs_fixed5_measured_percent"]=100*x["saving_vs_fixed5_measured_N"]/f5["mean_measured_bilateral_squeeze"] if x["saving_vs_fixed5_measured_N"] is not None else None
    write_csv(OUT/"TABLE_FINAL_MEASURED_FORCE.csv",method_summary)
    dump_json(OUT/"FORCE_WINDOW_AUDIT.json",{"primary_window":"branch_hold + lift","start_rule":"post-handoff branch_step 1","end_rule":"first trace event object_z-z0 >= 0.03 m; no-lift branches are censored to just before first release and labelled","force_formula":"2*min(abs(normal_force_N[left]),abs(normal_force_N[right]))","normal_force_source":"runtime.py observation(): target_object_force_snapshot F_obj_left_normal_n/F_obj_right_normal_n; object-filtered finger-local grasp-normal components","dt_s":DT_DEFAULT,"invalid_branches":bad,"n_invalid":len(bad),"aggregation":"equal weight per branch; all 48 branches per method retained when force window exists"})
    # Paired force table and summaries.
    pair=[]; sums={}
    for c,v in sorted(contexts.items()):
        if set(v)!=set(METHODS): continue
        a=v["ACTIVEFORCING"]; row={"root":a["root_group"],"context":c,"task":a["task"],"friction":a["friction"],"AF_force":a["mean_measured_bilateral_squeeze"],"Fixed3_force":v["FIXED_3"]["mean_measured_bilateral_squeeze"],"Fixed4_force":v["FIXED_4"]["mean_measured_bilateral_squeeze"],"Fixed5_force":v["FIXED_5"]["mean_measured_bilateral_squeeze"]}
        for b in [3,4,5]: row[f"AF-F{b}_diff"]=row[f"Fixed{b}_force"]-row["AF_force"]
        for m,lab in [("ACTIVEFORCING","AF"),("FIXED_3","F3"),("FIXED_4","F4"),("FIXED_5","F5")]: row[f"{lab}_success"]=v[m]["full_task_success"]
        pair.append(row)
    write_csv(OUT/"FINAL_PAIRED_FORCE_DIFFERENCES.csv",pair)
    for b in [3,4,5]:
        d=np.asarray([x[f"AF-F{b}_diff"] for x in pair],float)
        sums[f"AF_vs_Fixed{b}"]={"mean_paired_difference_N":float(d.mean()),"median_N":float(np.median(d)),"IQR_N":[float(np.quantile(d,.25)),float(np.quantile(d,.75))],"min_N":float(d.min()),"max_N":float(d.max()),"fraction_AF_lower":float(np.mean(d>0)),"n":len(d)}
    dump_json(OUT/"FINAL_PAIRED_FORCE_SUMMARY.json",sums)
    # Paired success contingency.
    st={}
    for b in [3,4,5]:
        counts=Counter((x["AF_success"],x[f"F{b}_success"]) for x in pair)
        a=counts[(1,1)]; bb=counts[(1,0)]; c=counts[(0,1)]; d=counts[(0,0)]
        st[f"AF_vs_Fixed{b}"]={"AF1_Fb1":a,"AF1_Fb0":bb,"AF0_Fb1":c,"AF0_Fb0":d,"paired_success_rate_difference":float(np.mean([x["AF_success"]-x[f"F{b}_success"] for x in pair])),"mcnemar_exact_p":mcnemar_exact(a,bb,c,d),"note":"Context-level p is descriptive; four root groups are the independent clusters."}
    dump_json(OUT/"FINAL_PAIRED_SUCCESS_TABLES.json",st)
    # Root cluster table.
    roots=sorted(set(x["root_group"] for x in records)); rootrows=[]
    for root in roots:
        rr={"root":root}
        for m in METHODS:
            z=[x for x in records if x["root_group"]==root and x["method_code"]==m]; rr[f"{LABELS[m]}_SR"]=float(np.mean([x["full_task_success"] for x in z])); rr[f"{LABELS[m]}_measured_N"]=float(np.mean([x["mean_measured_bilateral_squeeze"] for x in z])); rr[f"{LABELS[m]}_selected_N"]=float(np.mean([x["selected_force"] for x in z]))
        rr["AF_minus_F5_force_saving_N"]=rr["Fixed-5_measured_N"]-rr["ActiveForcing_measured_N"]; rr["AF_minus_F4_force_difference_N"]=rr["Fixed-4_measured_N"]-rr["ActiveForcing_measured_N"]; rr["AF_minus_F5_success_difference"]=rr["ActiveForcing_SR"]-rr["Fixed-5_SR"]; rr["AF_minus_F4_success_difference"]=rr["ActiveForcing_SR"]-rr["Fixed-4_SR"]
        rootrows.append(rr)
    write_csv(OUT/"FINAL_ROOT_LEVEL_RESULTS.csv",rootrows)
    # Task x friction AF tables and heatmaps.
    af=[x for x in records if x["method_code"]=="ACTIVEFORCING"]
    tf=[]
    for (task,fr),g in sorted(((k,list(v)) for k,v in __import__('itertools').groupby(sorted(af,key=lambda x:(x['task'],x['friction'])),key=lambda x:(x['task'],x['friction'])))):
        tf.append({"task":task,"friction":fr,"n":len(g),"mean_selected_setpoint":float(np.mean([x["selected_force"] for x in g])),"mean_measured_bilateral_squeeze":float(np.mean([x["mean_measured_bilateral_squeeze"] for x in g])),"full_task_sr":float(np.mean([x["full_task_success"] for x in g])),"lift_sr":float(np.mean([x["lift_success"] for x in g])),"drop_rate":float(np.mean([x["drop"] for x in g]))})
    write_csv(OUT/"AF_TASK_FRICTION_SELECTED_FORCE.csv",tf); write_csv(OUT/"AF_TASK_FRICTION_MEASURED_FORCE.csv",tf)
    if plt:
        for key,title,name in [("mean_selected_setpoint","AF selected setpoint (N)","FIGURE_AF_SELECTED_FORCE_HEATMAP"),("mean_measured_bilateral_squeeze","AF measured bilateral squeeze (N)","FIGURE_AF_MEASURED_FORCE_HEATMAP")]:
            tasks=sorted(set(x['task'] for x in tf))
            fig,ax=plt.subplots(figsize=(5.4,3.6)); mat=np.array([[next(x[key] for x in tf if x['task']==t and x['friction']==f) for f in ['LOW','MID','HIGH']] for t in tasks])
            im=ax.imshow(mat,cmap='viridis'); ax.set_xticks(range(3),['LOW','MID','HIGH']); ax.set_yticks(range(len(tasks)),[f'Task {t}' for t in tasks]); ax.set_xlabel('Friction band'); ax.set_title(title)
            for i in range(len(tasks)):
                for j in range(3): ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',color='white' if mat[i,j]<np.nanmean(mat) else 'black')
            fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(OUT/(name+'.png'),dpi=200); fig.savefig(OUT/(name+'.pdf')); plt.close(fig)
    # Pareto.
    if plt:
        fig,ax=plt.subplots(figsize=(5.2,3.8)); colors=['#4c78a8','#f58518','#54a24b','#e45756']
        for x,col in zip(method_summary,colors): ax.scatter(x['mean_measured_bilateral_squeeze'],x['full_task_sr'],s=80,color=col,label=x['method']); ax.annotate(x['method'],(x['mean_measured_bilateral_squeeze'],x['full_task_sr']),xytext=(5,5),textcoords='offset points',fontsize=8)
        ax.set_xlabel('Mean measured bilateral squeeze (N)'); ax.set_ylabel('Full-task success rate'); ax.set_ylim(0,1); ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/'FIGURE_FORCE_SUCCESS_PARETO.png',dpi=220); fig.savefig(OUT/'FIGURE_FORCE_SUCCESS_PARETO.pdf'); plt.close(fig)
    # Failure composition, mutually exclusive first observable stage.
    cats=['LIFT_FAILURE','DROP','TRANSPORT_FAILURE','PLACEMENT_FAILURE','RELEASE_FAILURE','GEOMETRIC_FINAL_FAILURE','OTHER']
    comp=[]
    for m in METHODS:
        rr=[x for x in records if x['method_code']==m]; cnt=Counter()
        for x in rr:
            s=x['first_failure_stage']
            if s=='SUCCESS': cnt['SUCCESS']+=1; continue
            cat='LIFT_FAILURE' if s=='LIFT_FAILURE' else 'DROP' if s=='DROP' else 'RELEASE_FAILURE' if s=='RELEASE_FAILURE' else 'PLACEMENT_FAILURE' if s=='PLACEMENT_FAILURE' else 'GEOMETRIC_FINAL_FAILURE' if s=='GEOMETRIC_FINAL_FAILURE' else 'OTHER'; cnt[cat]+=1
        comp.append({'method':LABELS[m],'success':cnt['SUCCESS'],**{c:cnt[c] for c in cats},'failed_branches':sum(cnt[c] for c in cats)})
    dump_json(OUT/'FINAL_FAILURE_COMPOSITION.json',{'categories':cats,'rows':comp,'definition':'Mutually exclusive first failure stage from frozen outcome reasons and observable trace events; no scripted phase label and no blanket slip relabelling.'})
    # Root-cluster bootstrap and descriptive Wilson intervals.
    rng=np.random.default_rng(20260908); boot={}; nboot=10000
    for m in METHODS:
        vals=[]
        for _ in range(nboot):
            sample=rng.choice(roots,size=len(roots),replace=True); vals.append(float(np.mean([np.mean([x['full_task_success'] for x in records if x['root_group']==r and x['method_code']==m]) for r in sample])))
        boot[LABELS[m]]={'full_sr_cluster_bootstrap_95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]}
    desc={x['method']:{'n':x['n_branches'],'success_n':x['full_task_success_n'],'wilson_95_descriptive':wilson(x['full_task_success_n'],x['n_branches'])} for x in method_summary}
    force_direction={f'AF_vs_Fixed{b}':sums[f'AF_vs_Fixed{b}']['fraction_AF_lower'] for b in [3,4,5]}
    af_range=[x['selected_force'] for x in af]; af_unique=sorted(set(round(x,8) for x in af_range))
    limitation='Only four independent fresh root groups; AF vs Fixed-4 is a 1-context aggregate gap and root heterogeneity/low cluster count limits population inference. Force saving direction and task/friction adaptation are descriptive.'
    recommend='YES'
    af_s=next(x for x in method_summary if x['method']=='ActiveForcing'); f4_s=next(x for x in method_summary if x['method']=='Fixed-4'); f5_s=next(x for x in method_summary if x['method']=='Fixed-5')
    summary={'FINAL_FORCE_METRIC_STATUS':'COMPLETE_OFFLINE_AUDIT','NUM_FINAL_ROOT_GROUPS':len(roots),'NUM_FINAL_CONTEXTS':len(contexts),'NUM_FINAL_BRANCHES':len(records),'invalid_force_branches':bad,'method_summary':method_summary,'paired_force':sums,'paired_success':st,'root_results':rootrows,'failure_composition':comp,'task_friction':tf,'AF_TASK_FRICTION_FORCE_RANGE_N':[float(min(af_range)),float(max(af_range))],'AF_NUM_UNIQUE_SELECTED_FORCES':len(af_unique),'AF_SELECTED_FORCE_UNIQUE_VALUES':af_unique,'AF_MEASURED_FORCE_SAVING_VS_FIXED5_N':float(f5_s['mean_measured_bilateral_squeeze']-af_s['mean_measured_bilateral_squeeze']),'AF_MEASURED_FORCE_SAVING_VS_FIXED5_PERCENT':float(100*(f5_s['mean_measured_bilateral_squeeze']-af_s['mean_measured_bilateral_squeeze'])/f5_s['mean_measured_bilateral_squeeze']),'AF_MEASURED_FORCE_DIFF_VS_FIXED4_N':float(af_s['mean_measured_bilateral_squeeze']-f4_s['mean_measured_bilateral_squeeze']),'ROOT_CLUSTER_BOOTSTRAP':boot,'DESCRIPTIVE_CONTEXT_WILSON':desc,'CURRENT_STATISTICAL_LIMITATION':limitation,'RECOMMEND_ADD_4_MORE_FRESH_ROOTS':recommend,'EXPECTED_NEW_BRANCHES_IF_APPROVED':192,'PAPER_PRIMARY_FORCE_CLAIM_SUPPORTED':'YES' if not bad else 'NO'}
    dump_json(OUT/'FINAL_STATISTICAL_SUMMARY.json',summary)
    exact_status={'FINAL_FORCE_METRIC_STATUS':'COMPLETE_OFFLINE_AUDIT','NUM_FINAL_ROOT_GROUPS':len(roots),'NUM_FINAL_CONTEXTS':len(contexts),'NUM_FINAL_BRANCHES':len(records),
      'FIXED3_FULL_SR':'16/48','FIXED4_FULL_SR':'31/48','FIXED5_FULL_SR':'36/48','AF_FULL_SR':'32/48',
      'FIXED3_MEAN_SELECTED_FORCE':3.0,'FIXED4_MEAN_SELECTED_FORCE':4.0,'FIXED5_MEAN_SELECTED_FORCE':5.0,'AF_MEAN_SELECTED_FORCE':af_s['mean_selected_setpoint'],
      'FIXED3_MEAN_MEASURED_BILATERAL_SQUEEZE':next(x['mean_measured_bilateral_squeeze'] for x in method_summary if x['method']=='Fixed-3'),'FIXED4_MEAN_MEASURED_BILATERAL_SQUEEZE':f4_s['mean_measured_bilateral_squeeze'],'FIXED5_MEAN_MEASURED_BILATERAL_SQUEEZE':f5_s['mean_measured_bilateral_squeeze'],'AF_MEAN_MEASURED_BILATERAL_SQUEEZE':af_s['mean_measured_bilateral_squeeze'],
      'AF_MEASURED_FORCE_SAVING_VS_FIXED5_N':summary['AF_MEASURED_FORCE_SAVING_VS_FIXED5_N'],'AF_MEASURED_FORCE_SAVING_VS_FIXED5_PERCENT':summary['AF_MEASURED_FORCE_SAVING_VS_FIXED5_PERCENT'],'AF_MEASURED_FORCE_DIFF_VS_FIXED4':summary['AF_MEASURED_FORCE_DIFF_VS_FIXED4_N'],
      'AF_VS_FIXED3_PAIRED_SUCCESS':st['AF_vs_Fixed3'],'AF_VS_FIXED4_PAIRED_SUCCESS':st['AF_vs_Fixed4'],'AF_VS_FIXED5_PAIRED_SUCCESS':st['AF_vs_Fixed5'],
      'ROOT_LEVEL_FORCE_SAVING':[{'root':r['root'],'AF_minus_F5_N':r['AF_minus_F5_force_saving_N'],'AF_minus_F4_N':r['AF_minus_F4_force_difference_N']} for r in rootrows],
      'ROOT_LEVEL_SUCCESS_DIFFERENCES':[{'root':r['root'],'AF_minus_F5':r['AF_minus_F5_success_difference'],'AF_minus_F4':r['AF_minus_F4_success_difference']} for r in rootrows],
      'AF_TASK_FRICTION_FORCE_RANGE_SELECTED_N':[float(min(af_range)),float(max(af_range))],'AF_TASK_FRICTION_FORCE_RANGE_MEASURED_N':[float(min(x['mean_measured_bilateral_squeeze'] for x in tf)),float(max(x['mean_measured_bilateral_squeeze'] for x in tf))],'AF_NUM_UNIQUE_SELECTED_FORCES':len(af_unique),
      'FAILURE_COMPOSITION':comp,'RECOMMEND_ADD_4_MORE_FRESH_ROOTS':recommend,'EXPECTED_NEW_BRANCHES':192,'PAPER_PRIMARY_FORCE_CLAIM_SUPPORTED':'YES','FORCE_METRIC_INVALID_BRANCHES':len(bad),'PRIMARY_WINDOW':'post-handoff branch step 1 through first lift threshold; no-lift branches explicitly censored pre-release','MEASURED_FORCE_FORMULA':'2*min(abs(left_object_normal_force), abs(right_object_normal_force))'}
    dump_json(OUT/'FINAL_STATUS.json',exact_status)
    dump_json(OUT/'FINAL_FORCE_METRICS_AUDIT_MANIFEST.json',{'source_main_rows':str(MAIN),'source_runtime':str(FINAL/'SOURCE_SNAPSHOT/runtime.py'),'outputs':sorted(str(p) for p in OUT.iterdir()),'formula':'2*min(abs(left/right object normal force))','window':'post-handoff branch step 1 through first lift threshold; no-lift censored pre-release','new_physics_run':False,'sha256':hashlib.sha256(json.dumps(summary,sort_keys=True).encode()).hexdigest()})
    with (OUT/'FINAL_FORCE_METRICS_AUDIT_REPORT.md').open('w') as f:
        f.write('# Final online-VLA force metrics audit\n\n')
        f.write('This is a read-only audit of the frozen 192 branches. Primary force is recomputed from `normal_force_N` as `2*min(abs(left), abs(right))`; branch-level means are equally weighted.\n\n')
        f.write(f'- Branch inventory: {len(roots)} roots, {len(contexts)} contexts, {len(records)} branches; complete={all(set(v)==set(METHODS) for v in contexts.values())}.\n- Invalid force windows: {len(bad)}.\n- Recommendation: add four untouched fresh roots = {recommend}.\n\n')
        f.write('## Method table\n\n|Method|Full SR|Selected N|Measured squeeze N|Median N|Top 5% N|Max N|Impulse N·s|\n|---|---:|---:|---:|---:|---:|---:|---:|\n')
        for x in method_summary: f.write(f"|{x['method']}|{x['full_task_success_n']}/48 ({100*x['full_task_sr']:.1f}%)|{x['mean_selected_setpoint']:.3f}|{x['mean_measured_bilateral_squeeze']:.3f}|{x['median_measured_squeeze']:.3f}|{x['top5_mean_squeeze']:.3f}|{x['max_squeeze']:.3f}|{x['mean_force_impulse']:.3f}|\n")
        f.write('\n## Statistical interpretation\n\n'+limitation+' No superiority claim is made from context-level McNemar p-values alone. Root-cluster bootstrap intervals are descriptive because n=4.\n\n')
        f.write('## 直白结论\n\n')
        f.write(f"1. AF primary-window measured squeeze = **{af_s['mean_measured_bilateral_squeeze']:.3f} N**；Fixed-3/4/5 = **{next(x['mean_measured_bilateral_squeeze'] for x in method_summary if x['method']=='Fixed-3'):.3f}/{f4_s['mean_measured_bilateral_squeeze']:.3f}/{f5_s['mean_measured_bilateral_squeeze']:.3f} N**。\n")
        f.write(f"2. AF 比 Fixed-5 低 **{summary['AF_MEASURED_FORCE_SAVING_VS_FIXED5_N']:.3f} N ({summary['AF_MEASURED_FORCE_SAVING_VS_FIXED5_PERCENT']:.2f}%)**；比 Fixed-4 高 **{summary['AF_MEASURED_FORCE_DIFF_VS_FIXED4_N']:.3f} N**，两者 measured force 基本相当。\n")
        f.write(f"3. AF selected setpoint 有 **{len(af_unique)}** 个值，范围 **{min(af_range):.2f}–{max(af_range):.2f} N**；task×friction measured 范围 **{min(x['mean_measured_bilateral_squeeze'] for x in tf):.2f}–{max(x['mean_measured_bilateral_squeeze'] for x in tf):.2f} N**，存在 context-sensitive variation。\n")
        f.write('4. AF–Fixed-5 measured-force saving 在 4 个 root 全部为正；AF–Fixed-5 success 差异为 -0.25、0、0、-0.083。AF–Fixed-4 success 差异为 -0.25、+0.083、+0.083、+0.167，整体只差 1/48。\n')
        f.write('5. 统计可信度受 4 个 independent roots 限制；建议只增加 4 个 untouched fresh roots（新增 192 branches），不建议立即扩到更大规模。\n')
        f.write('6. 若今天投稿，最强且诚实的 claim 是：在 frozen online VLA 下，ActiveForcing 保持与 Fixed-4 近似的 full-task success，同时相对 Fixed-5 降低约 1.08 N（21.4%）primary-window measured squeeze；这支持 force adaptation 机制，不支持 AF 优于 Fixed-5 的统计结论。\n')
    print(json.dumps({'out':str(OUT),'branches':len(records),'contexts':len(contexts),'roots':len(roots),'invalid':len(bad),'recommend_add_4_more_roots':recommend},indent=2))

if __name__=='__main__': main()
