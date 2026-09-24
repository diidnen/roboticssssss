#!/usr/bin/env python3
"""Stop the incumbent Stage-I process group after 16 complete contexts."""
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

PID = 847675
STATUS = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/experiments/af_motion_diversity_study_20260913/STAGE_I_STATUS.json")
RECEIPT = Path("/media/volume/data/exouser/nominal_table2_formal_20260913/STAGE_I_STOP_AT_16.json")

while True:
    if not Path(f"/proc/{PID}").exists():
        state = {"status": "process_already_gone"}
        break
    try:
        current = json.loads(STATUS.read_text())
    except Exception:
        time.sleep(1)
        continue
    if int(current.get("completed_contexts", 0)) >= 16:
        # The runner is a session/process-group leader.  Stopping the group also
        # prevents context 17 from continuing if it was spawned between polls.
        os.killpg(PID, signal.SIGTERM)
        state = {"status": "sigterm_sent", "pid": PID, "observed": current}
        break
    time.sleep(1)

state["requested_utc"] = datetime.now(timezone.utc).isoformat()
RECEIPT.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
