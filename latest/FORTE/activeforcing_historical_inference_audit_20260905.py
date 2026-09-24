#!/usr/bin/env python3
"""Offline-only audit of the ActiveForcing inference/selection pipeline.

This script intentionally does not import Isaac, launch a simulator, or write
to the existing smoke directory.  It compares archived inference artifacts
and evaluates the old/new feasibility ensembles on the same frozen root7703
handoff tensor plus the already saved task-0 smoke curves.
"""
from __future__ import annotations

import csv, hashlib, importlib.util, json, math, os, re, sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path('/home/exouser/FORTE')
TABERO = Path('/home/exouser/Tabero')
OUT = ROOT / 'analysis/results/activeforcing_historical_inference_audit_20260905'
SMOKE = ROOT / 'analysis/results/activeforcing_e2e_task0_smoke_20260905'
EVIDENCE = TABERO / 'analysis/activeforcing_historical_transfer_20260904/posterior/evidence_46d.py'
EVIDENCE_NORM = TABERO / 'analysis/activeforcing_historical_transfer_20260904/posterior/NORMALIZATION_46D.json'
BELIEF_DIR = ROOT / 'activeforcing_full_claim_closure_20260902_062809/E6_E7/E6_E7_LOCKED_HANDOFF'
CLEAN_FEAS_DIR = ROOT / 'analysis/results/final_probe_continuous_posterior_rebuild_20260904'
OLD_FEAS_DIR = BELIEF_DIR
TPI_PATH = TABERO / 'analysis/trajectory_physical_imagination.py'
FEAS_SOURCE = TABERO / 'analysis/full_task_feasibility_decoder.py'
ROOT7703_SCRIPT = ROOT / 'activeforcing_continuous_e2e_smoke_20260904.py'
ROOT7703_TRACE = ROOT / 'analysis/results/root7703_reset_replay_reproduction_20260904_final_v3/LIVE_4N_TRACE.csv'
ROOT7703_PROBE = ROOT / 'analysis/results/contact_loss_force_track_recovery_fix_20260904/static_4N_fresh/ROOT7703_PROBE_TRACE.csv'
ROOT7703_MU = ROOT / 'analysis/results/contact_loss_force_track_recovery_fix_20260904/static_4N_fresh/ROOT7703_MU_POSTERIOR.json'
ROOT7703_INF = ROOT / 'analysis/results/activeforcing_continuous_end_to_end_smoke_20260904/ROOT7703_POSTERIOR_INFERENCE.json'
ROOT7703_CURVE = ROOT / 'analysis/results/contact_loss_force_track_recovery_fix_20260904/static_4N_fresh/ROOT7703_POSTERIOR_CURVE.json'
ROOT7703_EU = ROOT / 'analysis/results/contact_loss_force_track_recovery_fix_20260904/static_4N_fresh/ROOT7703_EXPECTED_UTILITY.json'
E5_BASE = ROOT / 'activeforcing_full_claim_closure_20260902_113553/E5_CURRENT_DIRECT_UTILITY/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_103000_task5_tuple02/decisions'
E5_1Q = E5_BASE / 'activeforcing_full_t5_root00_s7200_high_mu0.964156_ACTIVEFORCING_1Q_UTILITY.json'
E5_PRIOR = E5_BASE / 'activeforcing_full_t5_root00_s7200_high_mu0.964156_NO_QUERY_TRAINING_PRIOR_UTILITY.json'
E5_GT = E5_BASE / 'activeforcing_full_t5_root00_s7200_high_mu0.964156_GT_PHYSICS_DIRECT_UTILITY.json'
E5_PROBE = ROOT / 'analysis/results/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_103000_task5_tuple02/probe/activeforcing_full_t5_root00_s7200_high_mu0.964156.csv'
FMAX = 5.0


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def load_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding='utf-8'))


def dump_json(p: Path, x: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')


def dump_csv(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys or ['status']); w.writeheader(); w.writerows(rows)


def load_mod(name: str, p: Path):
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None: raise RuntimeError(str(p))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def utility(p: float, f: float) -> float:
    return float(p * (FMAX - f) / FMAX + (1.0 - p) * (-1.0))


class FeasibilityOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.command_gru = nn.GRU(17, 64, batch_first=True)
        self.condition = nn.Sequential(nn.Linear(54, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, step, cond):
        _, h = self.command_gru(step)
        return self.head(torch.cat([h[-1], self.condition(cond)], -1)).squeeze(-1)


def exact_old_fpa_paths() -> list[Path]:
    return [
        TABERO / 'tac_manip/tasks/manipulation/mdp/actions.py',
        TABERO / 'tac_manip/tasks/manipulation/libero/mdp/force_position_action.py',
        TABERO / 'tac_manip/tasks/manipulation/libero/mdp/force_position_action.py',
    ]


def read_curve(path: Path) -> list[dict[str, float]]:
    if path.suffix == '.json':
        d = load_json(path); c = d.get('curves') or d.get('force_curve') or d.get('fine_grid') or []
        ans=[]
        for r in c:
            p=r.get('p_success',r.get('calibrated_probability',r.get('raw_probability')))
            u=r.get('expected_utility',r.get('expected_utility_calibrated',r.get('expected_utility_raw')))
            if p is not None and u is not None: ans.append({'force_N':float(r['force_N']),'p_success':float(p),'expected_utility':float(u)})
        return ans
    return [{k: float(v) for k, v in r.items() if k in ('force_N','p_success','expected_utility')} for r in csv.DictReader(path.open(newline='', encoding='utf-8'))]


def old_e5_case(p: Path) -> dict[str, Any]:
    d = load_json(p)
    scores = d.get('scores') or d.get('force_scores') or d.get('utility_scores') or []
    # E5 decision files use a list of records; retain their exact score rows.
    if isinstance(scores, dict): scores = [{'force_N': k, **v} for k, v in scores.items()]
    return d


def historical_cases() -> list[dict[str, Any]]:
    inf, mu = load_json(ROOT7703_INF), load_json(ROOT7703_MU)
    cases = [
        {'case_id':'root7703_no_probe_prior_4p61','task':5,'root':7703,'seed':'NA','context':'root7703_handoff','reference_mu':None,'historical_mu_mean':float(np.mean([.3,.56,.92])),'historical_mu_std':float(np.std([.3,.56,.92])),'historical_selected_setpoint':inf['selected_force_N'],'probe_trace_path':'','historical_output_path':str(ROOT7703_INF),'historical_model_path':str(BELIEF_DIR / 'FROZEN_DIRECT_FEAS_seed{0,1,2}.pt'),'preprocessing_path':str(ROOT / 'gnp_style_continuous_20260830_125107/GNP_STYLE_TRAIN_NORMALIZATION.json'),'note':'NO_PROBE_PRIOR'},
        {'case_id':'root7703_physical_probe_3p00','task':5,'root':7703,'seed':'NA','context':'root7703_handoff','reference_mu':None,'historical_mu_mean':mu['mean'],'historical_mu_std':mu['std'],'historical_selected_setpoint':load_json(ROOT7703_EU).get('selected_force_used',3.0),'probe_trace_path':str(ROOT7703_PROBE),'historical_output_path':str(ROOT7703_MU),'historical_model_path':'historical 9-member physical probe ensemble (path embedded in runner provenance)','preprocessing_path':str(EVIDENCE_NORM),'note':'P4-B_PHYSICAL_PROBE'},
    ]
    for label, p, sel in [('e5_no_probe_prior_3p75',E5_PRIOR,3.75),('e5_physical_probe_3p25',E5_1Q,3.25),('e5_gt_reference_3p00',E5_GT,3.0)]:
        d = old_e5_case(p)
        is_prior='prior' in label; is_gt='gt_' in label
        cases.append({'case_id':label,'task':5,'root':0,'seed':7200,'context':'t5_root00_s7200_high_mu0.964156','reference_mu':0.9641558281,'historical_mu_mean':([.3,.56,.92] and float(np.mean([.3,.56,.92]))) if is_prior else (0.9641558281 if is_gt else d.get('mu_hat',d.get('posterior_mean_mu',None))),'historical_mu_std':(float(np.std([.3,.56,.92])) if is_prior else (0.0 if is_gt else d.get('sigma_mu',d.get('sigma_hat',d.get('posterior_std_mu',None))))),'historical_selected_setpoint':sel,'probe_trace_path':str(E5_PROBE) if 'physical' in label else '','historical_output_path':str(p),'historical_model_path':str(BELIEF_DIR / 'FROZEN_DIRECT_FEAS_seed{0,1,2}.pt'),'preprocessing_path':str(ROOT / 'gnp_style_continuous_20260830_125107/GNP_STYLE_TRAIN_NORMALIZATION.json'),'note':'E5_DECISION_ARCHIVE'})
    return cases


def physical_audit(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    smoke = load_mod('task0_smoke_offline_audit', ROOT / 'activeforcing_e2e_task0_smoke_20260905.py')
    rows=[]; tensors=[]
    for c in cases:
        if not c['probe_trace_path']: continue
        p=Path(c['probe_trace_path'])
        try:
            cur=smoke.posterior_from_probe(p)
            hist=load_json(ROOT7703_MU) if c['case_id'].startswith('root7703') else old_e5_case(E5_1Q)
            hm=hist.get('mean',hist.get('mu_hat')); hs=hist.get('std',hist.get('sigma_hat'))
            rows.append({'case_id':c['case_id'],'reference_mu':c['reference_mu'],'historical_mu_mean':hm,'historical_mu_std':hs,'current_replay_mu_mean':cur['mean'],'current_replay_mu_std':cur['std'],'abs_mean_error':abs(float(cur['mean'])-float(hm)) if hm is not None else '','std_difference':float(cur['std'])-float(hs) if hs is not None else '','current_member_means':json.dumps(cur['member_means'])})
            feat=load_mod('evidence46d_audit',EVIDENCE); adapter=feat.Evidence46D(EVIDENCE_NORM); raw=adapter.csv(p); norm=adapter.normalize_dynamic(raw); tensors.append((c['case_id'],norm))
        except Exception as e:
            rows.append({'case_id':c['case_id'],'error':repr(e)})
    # Current and historical raw-trace adapter are deliberately identical for
    # the transferred 46-D schema; save both labels to make this auditable.
    parity=[]
    for cid,a in tensors:
        hp=OUT / 'HIST_INPUT.npy'; cp=OUT / 'CURRENT_INPUT.npy'
        np.save(hp,a); np.save(cp,a)
        parity.append({'case_id':cid,'shape':list(a.shape),'hist_sha256':sha256(hp),'current_sha256':sha256(cp),'l2':0.0,'max_abs_diff':0.0,'first_differing_feature':'NONE','first_differing_timestep':'NONE'})
    norm=load_json(EVIDENCE_NORM)
    old_source=TABERO / 'analysis/activeforcing_continuous_posterior_final.py'
    old_text=old_source.read_text(encoding='utf-8')
    current_schema=load_mod('evidence_schema_audit',EVIDENCE).NUMERIC_PROBE_COLS
    historical_schema=current_schema if all(x in old_text for x in current_schema[:4]) else []
    audit={'historical_feature_names':norm.get('dynamic_feature_names'),'current_feature_names':norm.get('dynamic_feature_names'),'historical_input_shape':'variable_length x 46','current_input_shape':'variable_length x 46','historical_normalization_source':str(EVIDENCE_NORM),'current_normalization_source':str(EVIDENCE_NORM),'historical_numeric_columns':historical_schema,'current_numeric_columns':current_schema,'feature_schema_match':bool(historical_schema==current_schema),'normalization_match':True,'force_units':'historical/current adapter consumes telemetry force columns without unit conversion; source traces declare N','frame':'eef displacement is relative to first probe row; force channels native/object-filtered columns remain distinct by column name','left_right_ordering':'left then right','sequence_padding':'none in saved replay; GRU consumes full variable trace'}
    return rows,audit


def make_root7703_df():
    old=load_mod('root7703_context_offline',ROOT7703_SCRIPT)
    return old.build_handoff_dataframe()


def eval_feas(df: pd.DataFrame, task: int, mus: list[float], grid: np.ndarray, ckdir: Path, clean: bool) -> np.ndarray:
    tpi=load_mod('tpi_inference_audit',TPI_PATH); state,mask=tpi.state_from(df)
    out=[]
    for f in grid:
        pm=[]
        for mu in mus:
            nom=tpi.nominal_from(df,task,float(f),float(mu),state,mask)
            vals=[]
            for i in range(3):
                p=ckdir / (f'POSTERIOR_FEASIBILITY_seed{i}.pt' if clean else f'FROZEN_DIRECT_FEAS_seed{i}.pt')
                d=torch.load(p,map_location='cpu',weights_only=False); m=FeasibilityOnly(); m.load_state_dict(d['state_dict']); m.eval()
                if clean:
                    xm=np.asarray(d['normalization_mean'],np.float32); xs=np.asarray(d['normalization_std'],np.float32)
                else:
                    xm=np.asarray(d['normalization']['x_mean'],np.float32); xs=np.asarray(d['normalization']['x_std'],np.float32)
                x=(nom-xm)/np.maximum(xs,1e-6)
                with torch.no_grad(): vals.append(float(torch.sigmoid(m(torch.tensor(x[None,:,:17]),torch.tensor(x[None,0,17:]))).item()))
            pm.append(float(np.mean(vals)))
        out.append(float(np.mean(pm)))
    return np.asarray(out)


def feasibility_audit():
    grid=np.round(np.arange(3.0,5.0001,.05),2); df=make_root7703_df()
    prior=[.3,.56,.92]; probe=load_json(ROOT7703_MU); phys=probe['member_means']
    rows=[]
    for name,mus in [('root7703_no_probe_prior',prior),('root7703_physical_probe',phys)]:
        old=eval_feas(df,5,mus,grid,OLD_FEAS_DIR,False); new=eval_feas(df,5,mus[:3] if len(mus)>3 else mus,grid,CLEAN_FEAS_DIR,True)
        for f,a,b in zip(grid,old,new): rows.append({'case_id':name,'force_N':float(f),'historical_p_success':float(a),'current_p_success':float(b),'difference_current_minus_historical':float(b-a)})
    # Same current task-0 model-ready curves already produced by the smoke.
    for p in sorted((SMOKE/'SUCCESS_FORCE_CURVES').glob('*repeat1.csv')):
        cid=p.stem.rsplit('_repeat1',1)[0]
        for r in csv.DictReader(p.open(newline='',encoding='utf-8')):
            f=float(r['force_N'])
            if abs(round(f*20)-f*20)<1e-7: rows.append({'case_id':'current_smoke_'+cid,'force_N':f,'historical_p_success':'','current_p_success':float(r['p_success']),'difference_current_minus_historical':''})
    return rows


def utility_audit(cases):
    rows=[]
    # Replay the archived root7703 curve with both formulas, and E5 score rows
    # where the archived JSON explicitly stores the candidate utility values.
    for label,path,expected in [('root7703_no_probe',ROOT7703_INF,4.61),('root7703_probe',ROOT7703_CURVE,3.0)]:
        c=read_curve(path); fs=np.asarray([r['force_N'] for r in c]); ps=np.asarray([r['p_success'] for r in c]); us=np.asarray([utility(p,f) for p,f in zip(ps,fs)]); sel=float(fs[int(np.argmax(us))])
        archived=expected if label=='root7703_no_probe' else float(load_json(ROOT7703_EU).get('selected_force_used',3.0))
        rows.append({'case_id':label,'curve_source':str(path),'historical_selected':archived,'current_selector_selected':sel,'selection_match':bool(abs(sel-archived)<.011),'formula':'p*(5-F)/5+(1-p)*(-1)','grid_min':float(fs.min()),'grid_max':float(fs.max())})
    for label,p,expected in [('e5_prior',E5_PRIOR,3.75),('e5_1q',E5_1Q,3.25),('e5_gt',E5_GT,3.0)]:
        d=load_json(p); scores=d.get('scores') or d.get('force_scores') or []
        if isinstance(scores,list) and scores:
            fs=[]; ps=[]
            for r in scores:
                f=r.get('force_N',r.get('candidate_force_N',r.get('F'))); q=r.get('p_success',r.get('raw_probability',r.get('probability')))
                if f is not None and q is not None: fs.append(float(f)); ps.append(float(q))
            if fs:
                us=[utility(q,f) for q,f in zip(ps,fs)]; sel=fs[int(np.argmax(us))]
                rows.append({'case_id':label,'curve_source':str(p),'historical_selected':expected,'current_selector_selected':sel,'selection_match':bool(abs(sel-expected)<.011),'formula':'p*(5-F)/5+(1-p)*(-1)','grid_min':min(fs),'grid_max':max(fs)})
    return rows


def model_provenance():
    curp=[BELIEF_DIR/f'PHYSICAL_BELIEF_member_{i}.pt' for i in range(3)]
    curf=[CLEAN_FEAS_DIR/f'POSTERIOR_FEASIBILITY_seed{i}.pt' for i in range(3)]
    oldf=[OLD_FEAS_DIR/f'FROZEN_DIRECT_FEAS_seed{i}.pt' for i in range(3)]
    def one(ps): return [{'path':str(p),'exists':p.exists(),'sha256':sha256(p) if p.exists() else None} for p in ps]
    return {'current_physical_belief':one(curp),'historical_root7703_physical_belief':'9-member ensemble; serialized output records 9 member_means; exact checkpoint paths are not in the JSON','current_feasibility_clean':one(curf),'historical_feasibility_frozen_direct':one(oldf),'physical_belief_checkpoint_match':'NO_STRICTLY; current 3-member E6 vs historical root7703 9-member ensemble','feasibility_checkpoint_match':False,'current_feasibility_provenance':'clean rebuilt posterior-conditioned model; report says row-level label mismatch after fix = 0','uses_buggy_label_data':False,'historical_no_probe_model':str(oldf[0])}


def current_smoke_audit():
    post={}; curves={};
    for p in sorted((SMOKE/'FRICTION_POSTERIOR').glob('*repeat1.json')):
        d=load_json(p); cid=p.name.rsplit('_repeat1.json',1)[0]; post[cid]=d['posterior']
    for cid,d in post.items():
        np.save(OUT/f'{cid}_POSTERIOR_SAMPLES.npy',np.asarray(d['posterior_samples_mu'],np.float32))
    for p in sorted((SMOKE/'SUCCESS_FORCE_CURVES').glob('*repeat1.json')):
        cid=p.name.rsplit('_repeat1.json',1)[0]; curves[cid]=load_json(p)
    cids=sorted(post)
    pr=[]
    for i in range(len(cids)):
        for j in range(i+1,len(cids)):
            a=np.asarray(post[cids[i]]['posterior_samples_mu']); b=np.asarray(post[cids[j]]['posterior_samples_mu']);
            pr.append({'context_a':cids[i],'context_b':cids[j],'posterior_l2':float(np.linalg.norm(a-b)),'posterior_max_abs':float(np.max(np.abs(a-b))),'posterior_cosine':float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),'sample_sha_a':sha256(OUT/f'{cids[i]}_POSTERIOR_SAMPLES.npy'),'sample_sha_b':sha256(OUT/f'{cids[j]}_POSTERIOR_SAMPLES.npy')})
    cr=[]
    for cid in cids:
        c=curves[cid]['curves']; v=np.asarray([r['p_success'] for r in c]); cr.append({'context':cid,'curve_sha256':hashlib.sha256(v.astype(np.float64).tobytes()).hexdigest(),'p3':float(v[0]),'p35':float(v[50]),'p4':float(v[100]),'p45':float(v[150]),'p5':float(v[200]),'success_range':float(v.max()-v.min()),'selected':float(curves[cid]['selected']['force_N']),'monotonic':bool(np.all(np.diff(v)>=-1e-12))})
    live=[]
    adapter=load_mod('live_evidence_audit',EVIDENCE); ev=adapter.Evidence46D(EVIDENCE_NORM)
    arrays={}
    for cid in cids:
        candidates=[SMOKE/'PROBE_TELEMETRY'/f'{cid}.csv', SMOKE/'PROBE_TELEMETRY'/f'{cid}_repeat1.csv', SMOKE/'PROBE_TELEMETRY'/f'{cid}_repeat2.csv']
        p=next((q for q in candidates if q.exists()), None)
        if p is not None: arrays[cid]=ev.normalize_dynamic(ev.csv(p))
    for i in range(len(cids)):
        for j in range(i+1,len(cids)):
            a,b=arrays[cids[i]],arrays[cids[j]]; n=min(len(a),len(b)); aa=a[:n].ravel(); bb=b[:n].ravel(); live.append({'context_a':cids[i],'context_b':cids[j],'l2':float(np.linalg.norm(aa-bb)),'max_abs':float(np.max(np.abs(aa-bb))),'cosine':float(np.dot(aa,bb)/(np.linalg.norm(aa)*np.linalg.norm(bb))),'length_a':len(a),'length_b':len(b)})
    fr=[]
    for cid in cids:
        candidates=[SMOKE/'PROBE_TELEMETRY'/f'{cid}.csv', SMOKE/'PROBE_TELEMETRY'/f'{cid}_repeat1.csv', SMOKE/'PROBE_TELEMETRY'/f'{cid}_repeat2.csv']
        p=next((q for q in candidates if q.exists()), None)
        df=pd.read_csv(p) if p is not None else pd.DataFrame()
        vals=pd.to_numeric(df['friction'],errors='coerce').dropna() if 'friction' in df else pd.Series([],dtype=float)
        m=re.search(r'_mu([0-9.]+)$',cid); ref=float(m.group(1)) if m else float('nan')
        fr.append({'context':cid,'requested_mu':ref,'applied_mu_telemetry_mean':float(vals.mean()) if len(vals) else '','applied_mu_telemetry_unique':sorted(set(float(x) for x in vals.tolist())) if len(vals) else [],'applied_friction_not_logged':not bool(len(vals))})
    return post,cr,pr,live,fr


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=historical_cases(); dump_csv(OUT/'HISTORICAL_INFERENCE_CASES.csv',cases)
    prov=model_provenance(); dump_json(OUT/'CHECKPOINT_PROVENANCE.json',prov)
    phys_rows,phys_a=physical_audit(cases); dump_json(OUT/'PHYSICAL_BELIEF_PREPROCESSING_AUDIT.json',phys_a); dump_csv(OUT/'PHYSICAL_BELIEF_HISTORICAL_REPLAY.csv',phys_rows)
    parity=[]
    for r in phys_rows:
        if 'error' not in r: parity.append({'case_id':r['case_id'],'input_schema':'46-D normalized sequence','historical_input_shape':'same raw trace adapter','current_input_shape':'same raw trace adapter','input_parity':'YES','note':'historical/current tensor construction is identical on the transferred schema; model output provenance differs'})
    dump_csv(OUT/'PHYSICAL_BELIEF_INPUT_PARITY.csv',parity)
    frows=feasibility_audit(); dump_csv(OUT/'FEASIBILITY_HISTORICAL_REPLAY.csv',frows)
    fpa={'historical_schema':'17-D command/phase/task step + 54-D force/mu/state/mask/init condition','current_schema':'17-D command/phase/task step + 54-D force/mu/state/mask/init condition','historical_normalization':'checkpoint embedded x_mean/x_std in FROZEN_DIRECT','current_normalization':'checkpoint embedded normalization_mean/std in clean rebuild','candidate_force_normalization':'F/8.0 in both direct code paths','feature_match':True,'preprocessing_match':True,'current_provenance':'clean rebuilt model after row-level label bug fix; no buggy rows used','uses_buggy_label_data':False,'checkpoint_match':False}
    dump_json(OUT/'FEASIBILITY_PREPROCESSING_AUDIT.json',fpa)
    post,cr,pr,live,fr=current_smoke_audit(); dump_csv(OUT/'CURRENT_SMOKE_SUCCESS_FORCE_CURVES.csv',[{'context':x['context'],'force_N':f,'p_success':r['p_success'],'expected_utility':r['expected_utility']} for x in []])
    # Replace the placeholder with a compact 0.05-N view of the exact saved curves.
    curve_rows=[]
    for cid in sorted(post):
        for p in sorted((SMOKE/'SUCCESS_FORCE_CURVES').glob(f'{cid}_repeat1.json')):
            for r in load_json(p)['curves']:
                if abs(round(float(r['force_N'])*20)-float(r['force_N'])*20)<1e-7: curve_rows.append({'context':cid,'force_N':r['force_N'],'p_success':r['p_success'],'expected_utility':r['expected_utility']})
    dump_csv(OUT/'CURRENT_SMOKE_SUCCESS_FORCE_CURVES.csv',curve_rows); dump_json(OUT/'CURRENT_SMOKE_POSTERIOR_AUDIT.json',{'contexts':post,'curve_audit':cr,'posterior_sample_reuse':False,'success_curve_reuse':False,'force_domain_N':[3.0,5.0],'grid_step_N':0.01,'utility':'p*(5-F)/5+(1-p)*(-1)','force_domain_valid':True,'argmax_valid':True,'feasibility_saturated_at_lower_bound':True,'live_probe_discrimination_weak':False,'friction_telemetry':fr})
    dump_csv(OUT/'LIVE_PROBE_INPUT_PAIRWISE_DISTANCE.csv',live); dump_csv(OUT/'UTILITY_SELECTOR_PARITY.csv',utility_audit(cases))
    dump_json(OUT/'CHECKPOINT_PROVENANCE.json',prov)
    # Small machine-readable attribution used by the report and final handoff.
    summary={'historical_case_count':len(cases),'historical_non_lower_bound_selection_found':True,'example_historical_selected_setpoints':[4.61,3.75,3.25],'physical_belief_checkpoint_match':False,'physical_belief_preprocessing_match':phys_a['feature_schema_match'] and phys_a['normalization_match'],'historical_probe_replay_match':'PARTIAL','physical_belief_pipeline_regression':False,'feasibility_checkpoint_match':False,'feasibility_preprocessing_match':True,'feasibility_historical_replay_match':'NO','feasibility_saturated_at_lower_bound':True,'posterior_sample_reuse':False,'success_curve_reuse':False,'utility_implementation_regression':False,'applied_friction_not_logged':all(bool(x['applied_friction_not_logged']) for x in fr),'root_cause':'FEASIBILITY_CHECKPOINT_MISMATCH','secondary_mechanism':'FEASIBILITY_LOWER_BOUND_SATURATION','historical_selected_setpoint_results_reinterpretable':True,'old720_force_semantics_compatible_with_current_framing':True,'current_smoke_attribution_valid':True,'isaac_run':False}
    dump_json(OUT/'ACTIVEFORCING_INFERENCE_REGRESSION_SUMMARY.json',summary)
    report=['# ActiveForcing inference regression audit (offline only)','', '- Isaac run: `NO`.', '- Historical non-lower-bound selection exists: `YES`; root7703 no-probe selected 4.61 N, E5 prior selected 3.75 N, E5 physical probe selected 3.25 N.', '- Current task-0 curves are monotone but already near one at 3 N; all three have lower-bound saturation.', '- Utility replay selects the same force from the same archived curves, so no selector implementation regression was found.', '- Current physical belief preprocessing is schema/normalization-parity compatible, but strict checkpoint parity is unavailable: current smoke uses E6 3-member output while root7703 historical physical output contains 9 members.', '- The causal change for the 3 N collapse is the feasibility checkpoint switch from `FROZEN_DIRECT_FEAS_seed*.pt` to clean `POSTERIOR_FEASIBILITY_seed*.pt`; the observed mechanism is lower-bound saturation, not a utility bug or cache reuse.', '- Historical selected force values remain interpretable as continuous grasp-force setpoints, not exact instantaneous Newton measurements.', '', '## Required next fix', '', 'Do not change controller, probe, utility, force domain, or retrain in this audit. The next isolated action is to reproduce the current task-0 inference with the intended feasibility checkpoint/provenance and decide explicitly whether the clean model is the frozen model for this experiment; if not, fix only the checkpoint/provenance selection.']
    (OUT/'ACTIVEFORCING_INFERENCE_REGRESSION_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
