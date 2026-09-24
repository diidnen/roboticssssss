#!/usr/bin/env bash
set -euo pipefail

LANE=/home/exouser/E3_E6_E7_LANES/E7_CONTINUOUS_FORCE_PLANNING
P1_SCRIPT=/home/exouser/activeforcing_p1_nominal_after_p0.sh
LOG=$LANE/P3_POST_P1_COORDINATOR.log
exec > >(tee -a $LOG) 2>&1
echo P3_POST_P1_COORDINATOR_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

while true; do
  p1=0
  pgrep -af '[a]ctiveforcing_p1_nominal_after_p0.sh' >/dev/null 2>&1 && p1=1
  e5=0
  pgrep -af '[r]un_e5_fresh_utility.py' >/dev/null 2>&1 && e5=1
  mass=0
  pgrep -af '[r]un_mass_fresh_e2e_pi0.py' >/dev/null 2>&1 && mass=1
  e7=0
  pgrep -af '[a]ctiveforcing_e7_offgrid_rollout.py --execute' >/dev/null 2>&1 && e7=1
  if test $p1 -eq 0 && test $e5 -eq 0 && test $mass -eq 0 && test $e7 -eq 0; then break; fi
  sleep 30
done

cd $LANE
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/e7_cpu_gate_audit.py > $LANE/E7_CPU_GATE_REFRESH_AFTER_P1.json
/usr/bin/python3 - $LANE/E7_CPU_GATE_REFRESH_AFTER_P1.json <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding='utf-8'))
if d.get('safe_to_launch_unchanged_8_rollout_gate') is not True:
    raise SystemExit('P3 gate remains closed after fresh resource audit')
PY

bash scripts/run_e7_fresh_exact_float_small_block_after_clear.sh
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
p=Path('E7_OFFGRID_DEV_EXECUTION_AUDIT.json')
if not p.is_file() or json.loads(p.read_text()).get('status') != 'PASS':
    raise SystemExit('P3 exact-float 8/8 gate did not PASS; full matrix remains closed')
PY

PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/e7_cpu_gate_audit.py > $LANE/E7_CPU_GATE_REFRESH_BEFORE_FULL.json
/usr/bin/python3 - $LANE/E7_CPU_GATE_REFRESH_BEFORE_FULL.json <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding='utf-8'))
if d.get('safe_to_launch_unchanged_8_rollout_gate') is not True:
    raise SystemExit('P3 full matrix resource gate failed')
PY

RUN_NAME=isaac_full_2x2_matrix_fresh_$(date -u +%Y%m%d_%H%M%S)
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/activeforcing_e7_offgrid_rollout.py \
  --execute --gpu-preflight-ok \
  --output $LANE \
  --manifest-name E7_MATCHED_FULL_DEV_ROLLOUT_MANIFEST.json \
  --run-name $RUN_NAME \
  --timeout-s 7200
echo P3_POST_P1_COORDINATOR_DONE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

