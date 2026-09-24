#!/usr/bin/env bash
set -euo pipefail

E5_ROOT=/home/exouser/FORTE/analysis/results/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_053006
P1_ROOT=/home/exouser/FORTE/agent_lanes/E3_FULLTASK_VS_LOCALLIFT/P1_NOMINAL_DEV_20260903
LANE=/home/exouser/E3_E6_E7_LANES/E3_FULLTASK_VS_LOCALLIFT
ART=/home/exouser/FORTE_e3/ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000
LOG=$P1_ROOT/P1_POST_P0_COORDINATOR.log
SCHEDULER_PID=3494903
SERVER_PID=

mkdir -p $P1_ROOT
exec > >(tee -a $LOG) 2>&1
echo P1_POST_P0_COORDINATOR_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

cleanup() {
  if test -n "$SERVER_PID" && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo stopping coordinator-owned candidate server pid=$SERVER_PID
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for i in 1 2 3 4 5; do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 2
    done
  fi
}
trap cleanup EXIT

while true; do
  marker=$E5_ROOT/E5_CURRENT_UTILITY_FRESH_RESET_TO_END_E2E_COMPLETE
  alive=0
  pgrep -af '[c]ore_gpu_scheduler.py' >/dev/null 2>&1 && alive=1
  pgrep -af '[r]un_e5_fresh_utility.py' >/dev/null 2>&1 && alive=1
  pgrep -af '[r]un_mass_fresh_e2e_pi0.py' >/dev/null 2>&1 && alive=1
  complete=0
  grep -q '"event": "gpu_collection_complete"' $E5_ROOT/CORE_GPU_SCHEDULER.jsonl 2>/dev/null && complete=1
  if test -f $marker && test $alive -eq 0 && test $complete -eq 1; then
    break
  fi
  sleep 30
done

SCHEDULER_PID=$(/usr/bin/python3 - $E5_ROOT/CORE_GPU_SCHEDULER.jsonl <<'PY'
import json, sys
starts=[]
for line in open(sys.argv[1], encoding='utf-8'):
    try:
        d=json.loads(line)
    except json.JSONDecodeError:
        continue
    if d.get('event') == 'scheduler_start' and isinstance(d.get('pid'), int):
        starts.append(d)
if not starts:
    raise SystemExit('no scheduler_start evidence')
print(max(starts, key=lambda d: d.get('utc', ''))['pid'])
PY
)

EVIDENCE=$P1_ROOT/E5_NATURAL_EXIT_$SCHEDULER_PID.json
if test ! -e $EVIDENCE; then
  /usr/bin/python3 - $E5_ROOT $EVIDENCE $SCHEDULER_PID <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
root, out, pid = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
log = root / "CORE_GPU_SCHEDULER.jsonl"
events = []
for line in log.read_text(encoding="utf-8").splitlines():
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(item, dict):
        events.append(item)
starts = [e for e in events if e.get("event") == "scheduler_start" and e.get("pid") == pid]
complete = [e for e in events if e.get("event") == "gpu_collection_complete"]
if not starts or not complete:
    raise SystemExit("refusing evidence without scheduler start and completion event")
record = {
    "schema": "E5_SCHEDULER_NATURAL_EXIT_EVIDENCE_V1",
    "scheduler_pid": pid,
    "exit_kind": "NATURAL",
    "exit_code": 0,
    "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "completion_event_utc": complete[-1].get("utc"),
    "evidence_basis": "P0 completion marker present; gpu_collection_complete observed; scheduler and E5/Mass workers absent at observation",
    "scheduler_log": str(log),
}
out.parent.mkdir(parents=True, exist_ok=True)
fd, tmp = tempfile.mkstemp(prefix="." + out.name + ".", dir=out.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, out)
finally:
    if os.path.exists(tmp):
        os.unlink(tmp)
print(json.dumps(record, sort_keys=True))
PY
fi

CANDIDATE_GATE=$P1_ROOT/E3_DYNAMIC_GATE_CANDIDATE_SERVER.json
if test ! -e $CANDIDATE_GATE; then
  $LANE/prepare_dynamic_gate_after_e5_exit.sh \
    --operation E3_TASK5_CANDIDATE_SERVER_START \
    --natural-exit-evidence $EVIDENCE \
    --output $CANDIDATE_GATE
fi

SERVER_RUN=$P1_ROOT/CANDIDATE_SERVER_RUN
SERVER_LOG=$P1_ROOT/CANDIDATE_SERVER.log
if ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq '(^|:)18883$'; then
  E3_ROOT_GO=YES E3_CORE_COORDINATOR_GO=YES E3_FINAL_GATE_GO=YES \
    setsid bash $ART/launch_task5_onboarded_candidate_server.sh \
      $CANDIDATE_GATE $SERVER_RUN >$SERVER_LOG 2>&1 &
  SERVER_PID=$!
fi
for i in $(seq 1 90); do
  ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq '(^|:)18883$' && break
  test -z "$SERVER_PID" || kill -0 "$SERVER_PID" 2>/dev/null || { echo candidate server exited; exit 2; }
  sleep 2
done
ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq '(^|:)18883$' || { echo candidate server timeout; exit 2; }

for root in 7600 7601 7602 7603 7604; do
  gate=$P1_ROOT/E3_DYNAMIC_GATE_root$root.json
  if test ! -e $gate; then
    $LANE/prepare_dynamic_gate_after_e5_exit.sh \
      --operation E3_TASK5_NOMINAL_DEV_CELL \
      --root-seed $root \
      --natural-exit-evidence $EVIDENCE \
      --output $gate
  fi
  E3_ROOT_GO=YES E3_CORE_COORDINATOR_GO=YES E3_FINAL_GATE_GO=YES \
    bash $ART/launch_task5_post_onboarding_nominal_dev_cell.sh \
      $gate $P1_ROOT $root
done

NOMINAL_GATE=$P1_ROOT/E3_TASK5_POST_ONBOARDING_NOMINAL_DEV_GATE.json
FREEZE=$P1_ROOT/TASK5_ONBOARDED_PI0_FINAL_FREEZE.json
if test ! -e $NOMINAL_GATE; then
  PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
    /media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python \
      $ART/analyze_task5_post_onboarding_nominal_dev.py \
      $P1_ROOT \
      /media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_5DEMO_CANDIDATE_LOCK_20260902/CANDIDATE_LOCK.json || true
fi

if test -f $NOMINAL_GATE && /usr/bin/python3 - $NOMINAL_GATE <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("gate_pass") is True else 1)
PY
then
  if test ! -e $FREEZE; then
    PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
      /media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python \
        $ART/promote_task5_onboarded_candidate_after_nominal_dev.py \
        /media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_5DEMO_CANDIDATE_LOCK_20260902/CANDIDATE_LOCK.json \
        $NOMINAL_GATE $FREEZE
  fi
else
  echo P1 nominal DEV did not pass; no final freeze or matched training authorized
fi

# User-authorized P1 simplification: once the candidate server and old nominal
# capability qualification are complete, collect the new matched tuple-only
# protocol before this coordinator exits.  This keeps the P3 watcher from
# starting while the simplified P1 lane is using the GPU.  The legacy P4-B
# branch path is not invoked or imported.
SIMPLIFIED_P1=$LANE/run_p1_simplified_collection.sh
if test -x "$SIMPLIFIED_P1"; then
  if ! bash "$SIMPLIFIED_P1"; then
    echo "P1 simplified collection incomplete; preserving artifacts and continuing coordinator cleanup"
  fi
fi

# Keep nominal capability qualification separate from formal matched-label
# closure. The current repository has no admissible post-freeze official
# nominal-VLA branch adapter for the registered long-horizon comparison.
P1_ADAPTER=/home/exouser/E3_E6_E7_LANES/E3_FULLTASK_VS_LOCALLIFT/p1_frozen_pi0_official_evaluator_adapter_contract.py
P1_STATUS=$P1_ROOT/P1_SCIENTIFIC_CLOSURE_STATUS.json
if test ! -e "$P1_STATUS"; then
  /usr/bin/python3 - "$P1_STATUS" "$NOMINAL_GATE" "$FREEZE" "$P1_ADAPTER" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

out, nominal, freeze, adapter = map(Path, sys.argv[1:])

def sha(path):
    if not path.is_file():
        return "MISSING"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()

record = {
    "schema": "P1_SCIENTIFIC_CLOSURE_STATUS_V1",
    "status": "INCONCLUSIVE_FORMAL_COMPARISON_NOT_ADMISSIBLE",
    "nominal_dev_gate": json.loads(nominal.read_text()) if nominal.is_file() else {"status": "NOT_RUN"},
    "final_freeze": str(freeze) if freeze.is_file() else None,
    "formal_matched_comparison": "NOT_RUN",
    "formal_branch_manifest": "ABSENT",
    "adapter_contract": str(adapter),
    "adapter_contract_sha256": sha(adapter),
    "reason": "No admissible post-freeze official nominal-VLA branch adapter currently exists for libero_10/task5; legacy B5/P5S0C paths have task, trajectory, and evaluator incompatibilities and remain excluded.",
    "test_used": False,
    "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}
out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_name("." + out.name + ".tmp")
tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
os.replace(tmp, out)
print(json.dumps(record, sort_keys=True))
PY
fi
echo P1_POST_P0_COORDINATOR_DONE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
