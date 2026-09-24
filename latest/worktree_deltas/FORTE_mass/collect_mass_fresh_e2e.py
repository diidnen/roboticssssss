#!/usr/bin/env python3
"""Fresh-root paired E2E mass evaluation for task2.

Every row executes reset -> frozen P4-B query -> mass estimate/policy choice ->
one selected-force structured task branch. Train utilities are read only from
the formal TRAIN split; fresh E2E roots are never used to fit the identifier.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, importlib.util, json, os, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np

TABERO=Path("/home/exouser/Tabero"); P4_PATH=Path("/home/exouser/FORTE_mass/frozen_p4_mass_probe.py"); STRUCTURED_PATH=Path("/home/exouser/FORTE_mass/qualify_mass_structure.py"); TASK=2; OBJ="salad_dressing_1"
BANDS={"LOW":.05,"MID":.10,"HIGH":.20}; FORCES=[.5,1.,1.5,2.5,4.]; POLICIES=["Default","Fixed-Max","NoQuery-Prior","ActiveForcing-Mass","GT-Mass + Direct"]

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m
def jsonable(v):
    try:
        import torch
        if isinstance(v,torch.Tensor): return v.detach().cpu().numpy().tolist()
    except Exception: pass
    if isinstance(v,dict): return {str(k):jsonable(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [jsonable(x) for x in v]
    if isinstance(v,(np.integer,np.floating,np.bool_)): return v.item()
    return v
def add_mass(p4,obj):
    orig=p4._apply_friction; state={}
    def apply(env,name,mu):
        out=orig(env,name,mu); import torch; view=env.scene[obj].root_physx_view
        if not state: state.update(mass=view.get_masses().clone(),inertia=view.get_inertias().clone(),nominal=float(view.get_masses()[0,0].item()))
        ratio=float(os.environ["MASS_QUERY_CURRENT_KG"])/state["nominal"]; ids=torch.arange(1,dtype=torch.int32); view.set_masses(state["mass"]*ratio,ids); view.set_inertias(state["inertia"]*ratio,ids); return out
    p4._apply_friction=apply; return state
def feat(c):
    try:q=json.loads(c["query_record"])
    except Exception:q={}
    def f(*ks):
        for k in ks:
            try:return float(q.get(k,0))
            except Exception:pass
        return 0.
    return np.array([f("f_meas_mean","f_meas_mean_N"),f("normal_force_peak","normal_force_peak_N"),float(c.get("query_history_rows",0))])
def fit(rows):
    x=np.asarray([feat(r) for r in rows]); y=np.asarray([float(r["mass_kg"]) for r in rows]); mu=x.mean(0); sd=x.std(0); sd[sd<1e-8]=1; z=(x-mu)/sd; X=np.c_[np.ones(len(z)),z]; w=np.linalg.solve(X.T@X+.01*np.eye(X.shape[1]),X.T@y); return lambda q: float(np.c_[np.ones(1), (np.asarray(q)-mu)/sd]@w)
def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields or ["status"]);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);ap.add_argument("--formal-train",type=Path,required=True);ap.add_argument("--roots",type=int,nargs="+",default=[8400,8401]);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    train_contexts=list(csv.DictReader((a.formal_train/"M3_TASK2_STRUCTURED_FORMAL_CONTEXTS.csv").open(newline="",encoding="utf-8"))); train_contexts=[r for r in train_contexts if int(r["query_valid"])]
    train_branches=list(csv.DictReader((a.formal_train/"M3_TASK2_STRUCTURED_FORMAL_BRANCHES.csv").open(newline="",encoding="utf-8")))
    predict=fit(train_contexts); utility={}
    for b in BANDS:
        vals=[]
        for f in FORCES:
            rr=[r for r in train_branches if r["mass_band"]==b and abs(float(r["requested_force_N"])-f)<1e-8]
            vals.append((np.mean([float(r["full_task_success_y"]) for r in rr]) if rr else 0)-.002*f,f)
        utility[b]=max(vals)[1]
    os.environ.update({"OMNI_KIT_ACCEPT_EULA":"YES","ACCEPT_EULA":"Y","P4_TASK_ID":"2","P4_VARIANT":"P4B","P4_OUT":str(a.out/"P4B_IMPORT"),"P4_RESUME":"0","P4_EPISODE_S":"45","HDF5_TRAJ_SOURCE_DIR":str(TABERO/"benchmarks/datasets/libero/assembled_hdf5"),"LIBERO_CONFIG_DIR":str(TABERO/"benchmarks/datasets/libero/config"),"LIBERO_ASSETS_DATA_DIR":str(TABERO/"benchmarks/datasets/libero/USD")})
    protocol={"status":"RUNNING","task_id":2,"fresh_roots":a.roots,"policies":POLICIES,"query":"frozen_P4B","mass_bands_kg":BANDS,"candidate_forces_N":FORCES,"identifier_fit":"formal TRAIN contexts only","downstream_outcome_used_by_identifier":False}; pf=a.out/"MASS_FRESH_E2E_PROTOCOL.json";pf.write_text(json.dumps(protocol,indent=2)+"\n"); rows=[]; exit_code=3
    try:
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,enable_cameras=False,num_envs=1).app
        import gymnasium as gym, torch
        import tac_manip.tasks # noqa
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        setup_task_objects("libero_object",2);cfg=parse_env_cfg("Isaac-Libero-Franka-Hybrid-Tactile-v0",device="cuda:0",num_envs=1);cfg.episode_length_s=45.;env=gym.make("Isaac-Libero-Franka-Hybrid-Tactile-v0",cfg=cfg).unwrapped;dt=float(env.cfg.sim.dt)*int(env.cfg.decimation)
        p4=load(P4_PATH,"fresh_e2e_p4"); structured=load(STRUCTURED_PATH,"fresh_e2e_structured"); state=add_mass(p4,OBJ)
        for root in a.roots:
            for band,mass in BANDS.items():
                os.environ["MASS_QUERY_CURRENT_KG"]=str(mass);cid=f"fresh_t2_root{root}_{band.lower()}";env.reset(seed=int(root));init=copy.deepcopy(env.scene.get_state(is_relative=True));ih=hashlib.sha256(json.dumps(jsonable(init),sort_keys=True).encode()).hexdigest();probe,rec=p4.run_probe_episode(env,seed_idx=int(root),mu=.5,trial_id=cid,dt=dt);post=copy.deepcopy(env.scene.get_state(is_relative=True));valid=int(rec.get("probe_failure",0)==0 and rec.get("contact_lost_probe",0)==0 and rec.get("dropped",0)==0);pm=predict(feat({"query_record":json.dumps(rec),"query_history_rows":len(probe)}));est=min(BANDS,key=lambda b:abs(BANDS[b]-pm));picks={"Default":1.,"Fixed-Max":4.,"NoQuery-Prior":utility["MID"],"ActiveForcing-Mass":utility[est],"GT-Mass + Direct":utility[band]}
                for policy,force in picks.items():
                    oracle_force = utility[band]
                    if not valid: br={"lift_success":0,"transport_retention":0,"place_success":0,"full_task_success_y":0,"dropped":0,"requested_force_N":force,"failure_stage":"query_invalid","measured_force_mean_N":0.}
                    else:
                        env.reset_to(copy.deepcopy(post),torch.tensor([0],device=env.device),is_relative=True);br=structured.structured_branch(env,p4,task=2,force=force,context_id=cid,seed=int(root),band=band,dt=dt,out_path=a.out/"telemetry"/f"{cid}_{policy.replace(' ','_')}.csv")
                    br.update({"context_id":cid,"root_seed":root,"mass_band":band,"mass_kg":mass,"friction":.5,"policy":policy,"selected_force_N":force,"oracle_force_reference_N":oracle_force,"under_force":int(force<oracle_force),"excess_force":int(force>oracle_force),"pred_mass_kg":pm,"estimated_band":est,"query_state_reach":valid,"query_validity":valid,"conditional_success":float(br.get("full_task_success_y",0)) if valid else 0.,"e2e_success":int(float(br.get("full_task_success_y",0))>0 and valid),"query_duration_s":len(probe)*dt,"initial_state_hash":ih,"post_query_state_hash":hashlib.sha256(json.dumps(jsonable(post),sort_keys=True).encode()).hexdigest(),"query_record":json.dumps(rec,sort_keys=True)});rows.append(br)
                write(a.out/"MASS_FRESH_E2E_ROWS.csv",rows)
        protocol.update({"status":"COMPLETED","rows":len(rows),"contexts":len(a.roots)*len(BANDS),"dt_s":dt,"nominal_mass_kg":state.get("nominal"),"source_sha256":hashlib.sha256(P4_PATH.read_bytes()).hexdigest()});exit_code=0
    except Exception as exc: protocol.update({"status":"ERROR","error":repr(exc)})
    pf.write_text(json.dumps(protocol,indent=2)+"\n");sys.stdout.flush();sys.stderr.flush();os._exit(exit_code)
if __name__=="__main__":main()
