#!/usr/bin/env python3
"""Wait for the incumbent experiment, then run all 96 nominal contexts."""
import json, os, signal, socket, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path("/media/volume/data/exouser/nominal_table2_formal_20260913")
SOURCE=ROOT/"source"; CONTEXTS=ROOT/"contexts"; LOGS=ROOT/"logs"
FINAL=Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_vla_v1")
CONF=Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/confirmatory_v1")
ISAAC=Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
SERVER_PY=Path("/media/volume/newdata/exouser/softvtbench/openpi-venv/bin/python")
SERVER_SRC=Path("/home/exouser/Tabero/analysis/results/task_form_coverage_20260910/runtime/frozen_vla_server/SOURCE_SNAPSHOT")
VTLA=Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA")
PORT=18895

def now(): return datetime.now(timezone.utc).isoformat()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    os.replace(tmp,path)
def pgrep(pattern):
    r=subprocess.run(["pgrep","-f",pattern],capture_output=True,text=True)
    return [int(x) for x in r.stdout.split() if int(x)!=os.getpid()]
def port_ready():
    try:
        with socket.create_connection(("127.0.0.1",PORT),timeout=1): return True
    except OSError: return False

def aggregate():
    subprocess.run([sys.executable,str(SOURCE/"aggregate_nominal.py"),"--root",str(ROOT)],check=False)

def main():
    ROOT.mkdir(parents=True,exist_ok=True); CONTEXTS.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)
    write(ROOT/"QUEUE_STATUS.json",{"status":"waiting_for_incumbent_stage_i","updated_utc":now()})
    pattern="af_motion_diversity_study_20260913/run_stage_i"
    while pgrep(pattern):
        time.sleep(30)
    # The incumbent RoboTwin policy server belongs exclusively to the now-ended queue.
    for pid in pgrep("activeforcing_robotwin_taskforms_20260911/.*/setup_policy_server.py"):
        try: os.kill(pid,signal.SIGTERM)
        except ProcessLookupError: pass
    deadline=time.time()+180
    while pgrep("activeforcing_robotwin_taskforms_20260911/.*/setup_policy_server.py") and time.time()<deadline:
        time.sleep(5)

    plans=[]
    for path in (FINAL/"FINAL_FRESH_ROOT_PLAN.json",CONF/"FINAL_FRESH_ROOT_PLAN.json"):
        plans.extend(json.loads(path.read_text())["contexts"])
    if len(plans)!=96 or len({x["id"] for x in plans})!=96:
        raise RuntimeError("formal context manifest is not exactly 96 unique contexts")
    write(ROOT/"FROZEN_NOMINAL_PLAN.json",{"created_utc":now(),"contexts":plans,
        "roots":list(range(170040,170048)),"tasks":[0,1,5,6],"bands":["LOW","MID","HIGH"],
        "protocol":"natural-reset Native VLA; no probe/AF/action override"})

    env=os.environ.copy()
    env.update({"PYTHONNOUSERSITE":"1","XLA_PYTHON_CLIENT_PREALLOCATE":"false","OMP_NUM_THREADS":"4",
        "PYTHONPATH":os.pathsep.join([
            str(SERVER_SRC),
            "/media/volume/newdata/exouser/tabero/uv-cache/git-v0/checkouts/b2400a7a62d6a7cf/b883328/src",
            str(VTLA/"src"),str(VTLA/"packages/openpi-client/src")])})
    server_dir=ROOT/"server"
    if server_dir.exists() and not (server_dir/"SERVER_READY.json").exists():
        server_dir.rename(ROOT/f"server_incomplete_{int(time.time())}")
    elif server_dir.exists():
        server_dir=ROOT/f"server_{int(time.time())}"
    with (LOGS/"server.log").open("a") as log:
        server=subprocess.Popen([str(SERVER_PY),"-u",str(SERVER_SRC/"server.py"),
            "--out",str(server_dir),"--port",str(PORT)],cwd=VTLA,env=env,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/"SERVER_PROCESS.json",{"pid":server.pid,"started_utc":now(),"out":str(server_dir)})
    deadline=time.time()+900
    while not port_ready():
        if server.poll() is not None: raise RuntimeError(f"policy server exited {server.returncode}")
        if time.time()>deadline: raise TimeoutError("policy server readiness timeout")
        time.sleep(5)

    done=failed=0
    try:
        for index,plan in enumerate(plans):
            job=CONTEXTS/(plan["id"]+"__NOMINAL_VLA")
            if (job/"RESULT.json").exists():
                done+=1; continue
            if job.exists(): job.rename(CONTEXTS/(job.name+f"__engineering_incomplete_{int(time.time())}"))
            write(ROOT/"QUEUE_STATUS.json",{"status":"running","completed_contexts":done,
                "engineering_failures":failed,"planned_contexts":96,"current_context":plan["id"],
                "index":index,"updated_utc":now()})
            cmd=[str(ISAAC),"-u",str(SOURCE/"run_nominal_context.py"),"--out",str(job),
                "--port",str(PORT),"--task",str(plan["task"]),"--root",str(plan["root"]),
                "--band",plan["band"],"--friction",str(plan["mu"])]
            with (LOGS/(plan["id"]+".log")).open("w") as log:
                result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode==0 and (job/"RESULT.json").exists(): done+=1
            else: failed+=1
            aggregate()
        status="complete" if done==96 and failed==0 else "complete_with_engineering_failures"
        write(ROOT/"QUEUE_STATUS.json",{"status":status,"completed_contexts":done,
            "engineering_failures":failed,"planned_contexts":96,"updated_utc":now()})
    finally:
        if server.poll() is None:
            server.terminate()
            try: server.wait(timeout=60)
            except subprocess.TimeoutExpired: server.kill()
        aggregate()

if __name__=="__main__":
    try: main()
    except Exception as e:
        write(ROOT/"QUEUE_STATUS.json",{"status":"engineering_failure","error":repr(e),
            "traceback":traceback.format_exc(),"updated_utc":now()})
        raise
