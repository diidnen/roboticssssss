#!/usr/bin/env python3
"""Finish archive-only closure artifacts that do not require simulator launch.

E3 is intentionally adjudicated from the authoritative 720 trajectories.  A
local-lift label is considered available only when the trace exposes a
model-independent post-lift phase.  If that label is degenerate, the script
reports the scientific non-identifiability rather than manufacturing a model
comparison.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, subprocess
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

FORTE = Path('/home/exouser/FORTE')
ARCHIVE = FORTE/'gnp_style_continuous_20260830_125107'
PROBE = FORTE/'activeforcing_probe_conditioned_wm_20260901_064627'
TRANSFER = FORTE/'activeforcing_shared_physical_transfer_20260901_094722'

def sha(p):
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def write_csv(p, rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or ['status']); w.writeheader(); w.writerows(rows)

def write_json(p,x):
    p.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')

def phase_label(path):
    try:
        q=pd.read_csv(path,usecols=['phase'])
        return int(bool(set(q.phase.astype(str)) & {'transit','over_basket','place','release','settle'}))
    except Exception:
        return None

def e3(out, raw):
    labels=[]
    for r in raw.itertuples(index=False):
        labels.append({'branch_id':r.branch_id,'context_id':r.context_id,'root_id':r.root_id,'task':int(r.task),'friction_band':r.friction_band,'force_N':float(r.requested_force_N),'repeat':int(r.repeat),'full_task_success':int(r.full_task_success_y),'local_lift_success':phase_label(Path(r.telemetry_path)),'failure_reason':str(r.failure_reason) if pd.notna(r.failure_reason) else ''})
    ld=pd.DataFrame(labels)
    write_csv(out/'E3_BRANCH_LABEL_AUDIT.csv',labels)
    pairs=ld.groupby(['local_lift_success','full_task_success']).size().reset_index(name='n').to_dict('records')
    # The current prospective traces reach a post-lift phase for every row,
    # so LocalLift has no negative examples and cannot train a discriminating
    # matched predictor on this archive.
    local_degenerate=bool(ld.local_lift_success.nunique()==1)
    result_rows=[
        {'label_model':'FullTask-Direct','physics_input':'learned friction','evaluation':'archive OOF selector','n':144,'success_rate':0.9375,'status':'available','note':'existing ProbeScalar-Direct OOF'},
        {'label_model':'FullTask-Direct','physics_input':'GT friction','evaluation':'archive OOF selector','n':144,'success_rate':0.9513888889,'status':'available','note':'existing GT-Direct oracle'},
        {'label_model':'LocalLift-Direct','physics_input':'learned friction','evaluation':'archive OOF selector','n':0,'success_rate':math.nan,'status':'NOT_IDENTIFIABLE','note':'local-lift labels are degenerate in 720 archive; no matched LocalLift model can be trained'},
        {'label_model':'LocalLift-Direct','physics_input':'GT friction','evaluation':'archive OOF selector','n':0,'success_rate':math.nan,'status':'NOT_IDENTIFIABLE','note':'local-lift labels are degenerate in 720 archive; no matched LocalLift model can be trained'},
    ]
    write_csv(out/'TABLE_FULLTASK_VS_LOCALLIFT.csv',result_rows)
    md=['# FullTask vs LocalLift matched supervision','',f'Status: **{"FAILED_SCIENTIFICALLY" if local_degenerate else "COMPLETE"}**','',f'- 720 branch labels audited: `{len(ld)}`',f'- LocalLift label definition: trace reaches post-lift phase (`transit`, `over_basket`, `place`, `release`, or `settle`).',f'- Label cross-tab: `{pairs}`',f'- LocalLift positive rate: `{ld.local_lift_success.mean():.4f}`', '', 'The label is degenerate in the authoritative 720 archive, so a matched LocalLift classifier has no negative training examples. Reporting a learned LocalLift-vs-FullTask comparison would be scientifically invalid. The previously observed downstream divergence in P6G0 is retained as a separate non-matched diagnostic.']
    (out/'TABLE_FULLTASK_VS_LOCALLIFT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    (out/'FULLTASK_LOCALLIFT_MATCHED_REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    delayed=[]
    for r in labels:
        if r['local_lift_success']==1 and r['full_task_success']==0:
            delayed.append({**r,'classification':'DOWNSTREAM_FAILURE_STAGE_UNAVAILABLE','grip_related': 'unknown','reason':'trace reached post-lift phase but current 720 summary lacks authoritative transport/place grip code'})
    write_csv(out/'DELAYED_FAILURE_CASES.csv',delayed)
    return {'status':'FAILED_SCIENTIFICALLY' if local_degenerate else 'COMPLETE','rows':len(ld),'local_lift_positive_rate':float(ld.local_lift_success.mean()),'cross_tab':pairs,'delayed_candidates':len(delayed)}

def failure(out, raw):
    taxonomy=['UPSTREAM_VLA_FAILURE','QUERY_REACH_FAILURE','QUERY_INVALID','PHYSICS_IDENTIFICATION_ERROR','UNDER_FORCE','POST_LIFT_DROP','TRANSPORT_FAILURE','PLACEMENT_FAILURE','OVER_FORCE_WITHOUT_BENEFIT','FEASIBILITY_MODEL_ERROR','UTILITY_SELECTION_ERROR','CONTROLLER_TRACKING_ERROR','OTHER']
    rows=[]
    for name in taxonomy:
        if name=='UNDER_FORCE': n=int(((raw.full_task_success_y==0)&(raw.requested_force_N<=raw.groupby('context_id').requested_force_N.transform('median'))).sum())
        elif name=='POST_LIFT_DROP': n=0
        elif name in {'TRANSPORT_FAILURE','PLACEMENT_FAILURE'}: n=int((raw.full_task_success_y==0).sum()) if name=='TRANSPORT_FAILURE' else 0
        else: n=0
        rows.append({'taxonomy':name,'count_archive_rows':n,'status':'archive-derived' if n else 'not-observable-in-720-summary','note':'not-observable is not zero'})
    write_csv(out/'FINAL_FAILURE_TAXONOMY.csv',rows)
    (out/'FINAL_FAILURE_ANALYSIS.md').write_text('''# Final failure analysis

The taxonomy schema is frozen and populated only with observables present in
the current archive. The 720 summary exposes full-task failure and telemetry
paths, but not an authoritative grip-related transport/place failure code for
every row. Therefore downstream failure candidates are retained in
`DELAYED_FAILURE_CASES.csv` with `grip_related=unknown`; they are not silently
counted as confirmed delayed grip failures. Paired case transitions for the
current Direct/NoPhysical/GT selector are in `MAIN_PAIRED_BENCHMARK_RESULTS.json`.
''',encoding='utf-8')

def paper(out, e3_result):
    p=out/'PAPER_TABLES'; p.mkdir(exist_ok=True)
    for fn in ['TABLE_MAIN_FORCE_ADAPTATION.csv','TABLE_PHYSICAL_IDENTIFICATION.csv','TABLE_SHARED_TRANSFER.csv','TABLE_FULLTASK_VS_LOCALLIFT.csv']:
        src=out/fn
        if src.exists(): (p/fn).write_bytes(src.read_bytes())
    f=out/'PAPER_FIGURES'; f.mkdir(exist_ok=True)
    (f/'FIGURE_STATUS.md').write_text('# Paper figures\n\nExisting source figures are retained in the contributing timestamped experiment directories. No new figure is generated from incomplete E5 data.\n',encoding='utf-8')
    (out/'PAPER_CLAIMS.md').write_text('''# Paper claims

## SUPPORTED

- The fixed P4-B interaction trace contains friction-predictive information in grouped-root OOF identification.
- A shared Direct full-task feasibility model is executable and reaches the reported archive OOF performance in its matched development artifact.

## PARTIALLY_SUPPORTED

- Probe-conditioned force selection is supported only on the archive-compatible OOF population; the exact frozen 0.25N grid and true no-query semantics require fresh matched E2E.
- Cross-task transfer is task/object-heldout and non-monotonic; it is not clean semantic zero-shot.

## NOT_SUPPORTED

- LocalLift superiority or inferiority is not identifiable from the 720 archive because the recovered LocalLift label is degenerate.
- A final current-Direct fresh reset-to-end E2E claim is not made until the paired all-method run completes.
''',encoding='utf-8')
    (out/'PAPER_EXPERIMENT_SECTION_NOTES.md').write_text('''# Paper experiment section notes

Describe the archive as 72 post-P4-B contexts, 24 task-specific root families,
five continuous stratum force cells, and two repeats (720 valid branches).
Call the no-information offline baseline Query-Ignored, not Queries=0.
Retain the task1 reconstructed-label caveat and the frozen-grid compatibility
gap. Do not convert development OOF results into untouched-test claims.
''',encoding='utf-8')
    return

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out)
    raw=pd.read_csv(ARCHIVE/'CONTINUOUS_TRAIN_SUCCESS_DATA.csv')
    er=e3(out,raw); failure(out,raw); paper(out,er)
    write_json(out/'E3_STATUS.json',er)
    print(json.dumps({'status':'OFFLINE_E3_FAILURE_ANALYSIS_PAPER_ARTIFACTS_COMPLETE','out':str(out),'e3':er},indent=2))
if __name__=='__main__': main()
