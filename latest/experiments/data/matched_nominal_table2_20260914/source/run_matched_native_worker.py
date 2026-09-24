#!/usr/bin/env python3
"""One matched post-query Native-VLA branch for the Table-II contexts."""
from __future__ import annotations

import argparse, csv, hashlib, importlib.util, json, os, random, sys, time, traceback
from collections import deque
from pathlib import Path
import numpy as np

FORTE=Path("/home/exouser/FORTE")
TABERO=Path("/home/exouser/Tabero")
FINAL=Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_vla_v1")
CONFIRM=Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/confirmatory_v1")
SOURCE=FINAL/"SOURCE_SNAPSHOT"
HORIZON=350; REPLAN_STEPS=10; OPEN_INTENT_THRESHOLD_M=.039; CONTACT_THRESHOLD_N=.15

def clean(x):
    if hasattr(x,"detach"): return clean(x.detach().cpu().numpy())
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,Path): return str(x)
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    return x

def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as f: json.dump(clean(value),f,indent=2,sort_keys=True,allow_nan=False); f.write("\n")

def read(path): return json.loads(Path(path).read_text())

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def identical(a,b):
    import torch
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor): return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray) and isinstance(b,np.ndarray): return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict) and isinstance(b,dict): return a.keys()==b.keys() and all(identical(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)): return len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
    return a==b

def native_outcome(base,trace):
    opened=len(trace)>=20 and all(t["vla_release_intent"] for t in trace[-20:])
    unheld=bool(base.get("unheld_last20")); inside=bool(base.get("inside_last50")); support=bool(base.get("final_support_contact"))
    lift=bool(base.get("lift_success")); dropped=bool(base.get("dropped"))
    timeout=any(t["terminations"].get("time_out",False) for t in trace)
    reset=any(t["physical_reset_requested"] for t in trace); complete=len(trace)==HORIZON
    flags={"NO_LIFT":not lift,"DROP":dropped,"TIMEOUT":timeout,"RESET":reset,"INCOMPLETE_HORIZON":not complete,
        "OUTSIDE_AUTHORED_REGION":not inside,"NOT_RELEASED":not(opened and unheld),"NO_FINAL_SUPPORT_CONTACT":not support}
    base.update(label_version="MATCHED_POSTQUERY_NATIVE_OPEN_INTENT_V1",full_task_success_y=int(not any(flags.values())),
        place_success=int(inside and opened and unheld and support),failure_reasons=[k for k,v in flags.items() if v],
        opened_last20=opened,native_release_semantics="raw pi0 aperture >=0.039; no gripper rewrite")
    return base

def rollout(env,p4,p5,plan,job,prepared,runtime):
    import torch, full_task_label as legacy_label
    geom=legacy_label.geom
    taskgeo=next(x for x in read(FORTE/"analysis/results/current_runtime_recovery_v2_20260905/TASK_GEOMETRY_MANIFEST.json")["tasks"] if x["task"]==plan["task"])
    if sha(taskgeo["label_geometry"])!=taskgeo["label_sha256"]: raise RuntimeError("geometry changed")
    with np.load(taskgeo["label_geometry"]) as f: vertices=f["vertices"]; region=geom.Region(f["basket_from_site"],f["half_size"])
    client,build,buffer,chunk,proof=prepared
    traces=[]; terminals=[]; error=None; z0=float(env.scene[plan["object"]].data.root_pos_w[0,2])
    original_compute,original_reset=env.termination_manager.compute,env._reset_idx
    def compute():
        original_compute(); failures=torch.zeros_like(env.termination_manager.terminated)
        for name in env.termination_manager.active_terms:
            if name!="success" and not env.termination_manager.get_term_cfg(name).time_out: failures.logical_or_(env.termination_manager.get_term(name))
        env.termination_manager._terminated_buf.copy_(failures); return env.termination_manager.dones
    def reset(ids):
        if len(ids): terminals.append(runtime.observation(env,p4,p5,plan,geom))
    env.termination_manager.compute,env._reset_idx=compute,reset
    try:
        for step in range(1,HORIZON+1):
            if step>1 and (step-1)%REPLAN_STEPS==0: chunk,proof=runtime.request_chunk(env,plan,client,build,buffer,step)
            index=(step-1)%REPLAN_STEPS; raw=np.ascontiguousarray(chunk[index,:13],dtype=np.float32)
            before=int(env.episode_length_buf[0]); env.step(torch.from_numpy(raw.copy()).reshape(1,13).to(env.device))
            rec=terminals[-1] if terminals else runtime.observation(env,p4,p5,plan,geom)
            release=bool(raw[6]>=OPEN_INTENT_THRESHOLD_M)
            row={"branch_step":step,"request_id":proof["request_id"],"chunk_index":index,"action":raw.tolist(),
                "action_sha256":hashlib.sha256(raw.tobytes()).hexdigest(),"raw_vla_arm_command":raw[:6].tolist(),
                "raw_vla_gripper_command":float(raw[6]),"final_arm_command":raw[:6].tolist(),
                "final_gripper_command":float(raw[6]),"vla_release_intent":release,"selected_force_setpoint":None,
                "active_force_setpoint":None,"vla_gripper_override_after_af_handoff":False,
                "arbitration_version":"NONE_NATIVE_POSTPROCESSED_PI0_ACTION","physical_reset_requested":bool(terminals),**rec}
            traces.append(row)
            with (job/"ACTION_TRACE.jsonl").open("a") as f: f.write(json.dumps(clean(row),allow_nan=False)+"\n"); f.flush()
            if terminals: break
            if int(env.episode_length_buf[0])!=before+1: raise RuntimeError("unexpected episode clock")
    except Exception as exc: error={"error":repr(exc),"traceback":traceback.format_exc()}
    finally: env.termination_manager.compute,env._reset_idx=original_compute,original_reset
    write(job/"BRANCH_TRACE.json",traces)
    outcome=runtime.evaluate(traces,vertices,region,geom,z0,error=error)
    if outcome.get("label_valid"): outcome=native_outcome(outcome,traces)
    active=[t for t in traces if not t["vla_release_intent"]]
    contact=[t["measured_bilateral_squeeze"] for t in active if min(t["normal_force_N"])>=CONTACT_THRESHOLD_N]
    outcome.update(contact_conditional_squeeze_N=float(np.mean(contact)) if contact else None,
        nonrelease_samples=len(active),bilateral_contact_samples=len(contact),contact_fraction=len(contact)/len(active) if active else None)
    result={"plan":plan,"method":"MATCHED_NATIVE_VLA","steps":len(traces),"rpc_count":client.calls,"outcome":outcome,"error":error,
        "commanded_force_N":None,"probe_evidence_used_for_action":False,"probe_prefix_shared":True,"action_override_used":False,
        "online_vla_verified":bool(traces) and error is None and client.calls==(len(traces)+REPLAN_STEPS-1)//REPLAN_STEPS,
        "checkpoint_sha256":client.metadata["checkpoint_sha256"]}
    write(job/"BRANCH_RESULT.json",result)
    if error: raise RuntimeError(error["error"])
    return result

def run(args):
    out=Path(args.out); job=Path(args.job); plan=read(args.plan); ref=Path(args.reference)
    sys.path[:0]=[str(SOURCE),str(FORTE),str(FORTE/"analysis/results/current_runtime_recovery_v2_20260905"),
        str(FORTE/"analysis/results/current_runtime_core_snapshot_v2_20260905"),
        str(FORTE/"analysis/results/current_runtime_sensor_repair_v3_candidate_20260905"),
        str(FORTE/"analysis/results/current_runtime_branch_execution_v6_20260905"),str(TABERO),str(TABERO/"analysis"),
        str(TABERO/"benchmarks/openpi/openpi-client/src")]
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=True,num_envs=1).app
    try:
        import torch, current_runtime_core as core, measurement_hooks
        from activeforcing_execution_snapshot import capture
        from branch_execution import SNAPSHOT_GROUPS
        spec=importlib.util.spec_from_file_location("matched_native_runtime",SOURCE/"runtime.py")
        runtime=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runtime;spec.loader.exec_module(runtime)
        def decision(env,p4,p5,liveplan,livejob):
            saved=torch.load(job/"DECISION_STATE.pt",map_location="cpu",weights_only=False)
            reference=torch.load(ref/"DECISION_STATE.pt",map_location="cpu",weights_only=False)
            live=capture(env)
            eq={k:identical(saved[k],live[k]) and identical(saved[k],reference[k]) for k in SNAPSHOT_GROUPS}
            eq["probe_rows"]=list(csv.DictReader((job/"RAW_PROBE.csv").open()))==list(csv.DictReader((ref/"RAW_PROBE.csv").open()))
            eq["probe_records"]=identical(read(job/"CONTACT_PATCH_READBACK.json"),read(ref/"CONTACT_PATCH_READBACK.json"))
            write(job/"POSTQUERY_MATCH_GATE.json",{"passed":all(eq.values()),"checks":eq,"reference":str(ref)})
            if not all(eq.values()): raise RuntimeError("post-query state does not match Table-II reference")
            prepared=runtime.first_chunk(env,plan,job,args.port,ref)
            return rollout(env,p4,p5,plan,job,prepared,runtime)
        rc=core.run_probe(plan,job,measurement_hooks=measurement_hooks,on_decision=decision,runtime_manifest_sha256=args.protocol_sha256)
        branch=read(job/"BRANCH_RESULT.json") if (job/"BRANCH_RESULT.json").exists() else {}
        valid=rc==0 and branch.get("online_vla_verified") and branch.get("outcome",{}).get("label_valid")
        write(job/"WORKER_COMPLETION.json",{"logical_success":bool(valid),"method":"MATCHED_NATIVE_VLA"})
        return 0 if valid else 2
    finally: app.close()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--job",required=True)
    p.add_argument("--plan",required=True);p.add_argument("--reference",required=True);p.add_argument("--protocol-sha256",required=True)
    p.add_argument("--port",type=int,default=18895);raise SystemExit(run(p.parse_args()))
