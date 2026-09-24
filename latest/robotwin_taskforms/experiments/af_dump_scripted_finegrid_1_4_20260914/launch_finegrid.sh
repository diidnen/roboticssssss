#!/bin/bash
set -euo pipefail
HERE=/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/experiments/af_dump_scripted_finegrid_1_4_20260914
QUEUE_PY=/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/venv_robotwin/bin/python3
if pgrep -f "$HERE/run_scripted_queue.py" >/dev/null; then
  echo already live >&2; exit 1
fi
setsid env PYTHONUNBUFFERED=1 AF_SCRIPTED_PARALLEL="${AF_SCRIPTED_PARALLEL:-3}" \
  "$QUEUE_PY" -u "$HERE/run_scripted_queue.py" \
  > "$HERE/queue.log" 2>&1 < /dev/null &
echo "{\"pid\": $!, \"started_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"parallel\": ${AF_SCRIPTED_PARALLEL:-3}}" > "$HERE/QUEUE_PID.json"
sleep 2
ps -p $! -o pid,etimes,cmd || true
