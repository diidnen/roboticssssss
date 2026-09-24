#!/usr/bin/env python3
import hashlib,json,os,signal,socket,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path("/media/volume/data/exouser/matched_nominal_table2_20260914");SRC=ROOT/"source";JOBS=ROOT/"contexts";LOGS=ROOT/"logs"
FINAL=Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_vla_v1")
CONF=Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/confirmatory_v1")
ISAAC=Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
SERVERPY=Path("/media/volume/newdata/exouser/softvtbench/openpi-venv/bin/python")
SERVERSRC=Path("/home/exouser/Tabero/analysis/results/task_form_coverage_20260910/runtime/frozen_vla_server/SOURCE_SNAPSHOT")
VTLA=Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA");PORT=18895
def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".tmp");q.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n");os.replace(q,p)
def ready():
 try:
  with socket.create_connection(("127.0.0.1",PORT),1):return True
 except OSError:return False
def pids(pattern):
 r=subprocess.run(["pgrep","-f",pattern],capture_output=True,text=True)
 return [int(x) for x in r.stdout.split() if int(x)!=os.getpid()]
def main():
 ROOT.mkdir(parents=True,exist_ok=True);SRC.mkdir(exist_ok=True);JOBS.mkdir(exist_ok=True);LOGS.mkdir(exist_ok=True)
 protocol={"method":"MATCHED_NATIVE_VLA","contexts":96,"roots":list(range(170040,170048)),"tasks":[0,1,5,6],
  "start":"exact deterministic Table-II post-query state","probe_prefix":"shared but evidence ignored by Native policy",
  "execution":"raw postprocessed pi0 arm+gripper action; no force/gripper override","horizon":350,
  "checkpoint_sha256":"0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17"}
 write(ROOT/"FROZEN_PROTOCOL.json",protocol);protocol_sha=hashlib.sha256((ROOT/"FROZEN_PROTOCOL.json").read_bytes()).hexdigest()
 plans=[]
 for base in (FINAL,CONF): plans.extend(json.loads((base/"FINAL_FRESH_ROOT_PLAN.json").read_text())["contexts"])
 if len(plans)!=96 or len({p["id"] for p in plans})!=96:raise RuntimeError("not 96 unique frozen contexts")
 write(ROOT/"FROZEN_PLAN.json",{"contexts":plans,"protocol_sha256":protocol_sha})
 # Soft GPU gate: run in parallel with Fixed sweep when headroom remains.
 # Do NOT kill the robotwin :6001 server — Matched uses its own :18895 server.
 def gpu_busy():
  try:
   out=subprocess.check_output([
    "nvidia-smi","--query-gpu=utilization.gpu,memory.free","--format=csv,noheader,nounits"
   ],text=True).strip().split(",")
   util=float(out[0]); free_mib=float(out[1])
   return util>=95.0 or free_mib<6000.0, util, free_mib
  except Exception as e:
   return False, None, None
 while True:
  busy,util,free_mib=gpu_busy()
  done_n=sum(1 for p in plans if (JOBS/(p["id"]+"__MATCHED_NATIVE_VLA")/"BRANCH_RESULT.json").exists())
  if not busy: break
  write(ROOT/"QUEUE_STATUS.json",{"status":"waiting_for_gpu_headroom","completed_contexts":done_n,
   "engineering_failures_excluded":True,"planned_contexts":96,"gpu_util_pct":util,"memory_free_mib":free_mib,
   "gate":"util<95 and free_mib>=6000","updated_utc":now()})
  time.sleep(30)
 write(ROOT/"QUEUE_STATUS.json",{"status":"starting_parallel_with_fixed_sweep","completed_contexts":sum(1 for p in plans if (JOBS/(p["id"]+"__MATCHED_NATIVE_VLA")/"BRANCH_RESULT.json").exists()),
  "planned_contexts":96,"port":PORT,"updated_utc":now()})
 env=os.environ.copy();env.update(PYTHONNOUSERSITE="1",XLA_PYTHON_CLIENT_PREALLOCATE="false",OMP_NUM_THREADS="4",
  OMNI_KIT_ACCEPT_EULA="YES",ACCEPT_EULA="Y",TABERO_ROOT="/home/exouser/Tabero",
  HDF5_TRAJ_SOURCE_DIR="/home/exouser/Tabero/benchmarks/datasets/libero/assembled_hdf5",
  LIBERO_CONFIG_DIR="/home/exouser/Tabero/benchmarks/datasets/libero/config",
  LIBERO_ASSETS_DATA_DIR="/home/exouser/Tabero/benchmarks/datasets/libero/USD",
  PYTHONPATH=os.pathsep.join([str(SERVERSRC),
   "/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64",
   "/media/volume/newdata/exouser/tabero/uv-cache/git-v0/checkouts/b2400a7a62d6a7cf/b883328/src",
   str(VTLA/"src"),str(VTLA/"packages/openpi-client/src"),"/home/exouser/FORTE","/home/exouser/Tabero",
   "/home/exouser/Tabero/benchmarks/openpi/openpi-client/src"]))
 serverdir=ROOT/"server";serverdir=serverdir if not serverdir.exists() else ROOT/f"server_{int(time.time())}"
 with (LOGS/"server.log").open("a") as f:server=subprocess.Popen([str(SERVERPY),"-u",str(SERVERSRC/"server.py"),"--out",str(serverdir),"--port",str(PORT)],cwd=VTLA,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 deadline=time.time()+900
 while not ready():
  if server.poll() is not None:raise RuntimeError(f"server exit {server.returncode}")
  if time.time()>deadline:raise TimeoutError("server timeout")
  time.sleep(5)
 done=failed=0
 try:
  for i,plan in enumerate(plans):
   job=JOBS/(plan["id"]+"__MATCHED_NATIVE_VLA")
   if (job/"BRANCH_RESULT.json").exists():done+=1;continue
   if job.exists():job.rename(JOBS/(job.name+f"__engineering_incomplete_{int(time.time())}"))
   planfile=ROOT/"plans"/(plan["id"]+".json");write(planfile,plan)
   base=FINAL if plan["root"]<=170043 else CONF;ref=base/"references"/plan["id"]
   write(ROOT/"QUEUE_STATUS.json",{"status":"running","completed_contexts":done,"engineering_failures":failed,"current_context":plan["id"],"planned_contexts":96,"updated_utc":now()})
   cmd=[str(ISAAC),"-u",str(SRC/"run_matched_native_worker.py"),"--out",str(ROOT),"--job",str(job),"--plan",str(planfile),"--reference",str(ref),"--protocol-sha256",protocol_sha,"--port",str(PORT)]
   with (LOGS/(plan["id"]+".log")).open("w") as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
   if r.returncode==0 and (job/"BRANCH_RESULT.json").exists():done+=1
   else:failed+=1
   subprocess.run([sys.executable,str(SRC/"aggregate.py")],check=False)
  write(ROOT/"QUEUE_STATUS.json",{"status":"complete" if done==96 and failed==0 else "complete_with_engineering_failures","completed_contexts":done,"engineering_failures":failed,"planned_contexts":96,"updated_utc":now()})
 finally:
  if server.poll() is None:
   server.terminate()
   try:server.wait(60)
   except subprocess.TimeoutExpired:server.kill()
  subprocess.run([sys.executable,str(SRC/"aggregate.py")],check=False)
if __name__=="__main__":
 try:main()
 except Exception as e:
  write(ROOT/"QUEUE_STATUS.json",{"status":"engineering_failure","error":repr(e),"traceback":traceback.format_exc(),"updated_utc":now()});raise
