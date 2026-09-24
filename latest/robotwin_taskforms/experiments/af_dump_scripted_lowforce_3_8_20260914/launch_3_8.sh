#!/bin/bash
set -euo pipefail
BASE=/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911
HERE=/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/experiments/af_dump_scripted_lowforce_3_8_20260914
QUEUE_PY=$BASE/venv_robotwin/bin/python3
if pgrep -f "$HERE/run_scripted_queue.py" >/dev/null; then
  echo "3-8N queue already live" >&2; exit 1
fi
setsid env PYTHONUNBUFFERED=1 AF_SCRIPTED_PARALLEL="${AF_SCRIPTED_PARALLEL:-3}" \
  "$QUEUE_PY" -u "$HERE/run_scripted_queue.py" \
  > "$HERE/queue.log" 2>&1 < /dev/null &
queue_pid=$!
echo "{\"pid\": $queue_pid, \"started_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"parallel\": ${AF_SCRIPTED_PARALLEL:-3}}" > "$HERE/QUEUE_PID.json"
sleep 2
ps -p "$queue_pid" -o pid,etimes,cmd || true
tail -n 15 "$HERE/queue.log" || true
