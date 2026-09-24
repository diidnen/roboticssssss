#!/usr/bin/env python3
"""Launch the persistent challenge supervisor without shell redirection."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "activeforcing_final_experiment_20260901_045000"
log = OUT / "collection_challenge" / "logs" / "supervisor.log"
log.parent.mkdir(parents=True, exist_ok=True)
with log.open("a") as fh:
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "run_force_critical_campaign.py")],
        cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
    )
(OUT / "FORCE_CRITICAL_SUPERVISOR_PID.json").write_text(json.dumps({
    "pid": proc.pid, "control": "read FORCE_CRITICAL_COLLECTION_SUPERVISOR.json; SIGTERM only when no simulator branch is executing",
    "log": str(log), "model_queries": 0, "untouched_TEST_read": False,
}, indent=2, sort_keys=True) + "\n")
print(json.dumps({"status": "STARTED", "pid": proc.pid, "log": str(log)}, indent=2))
