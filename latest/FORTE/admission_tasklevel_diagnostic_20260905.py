#!/usr/bin/env python3
"""Run a non-branching task-level P4-B admission diagnostic.

This is deliberately isolated from the frozen validation output.  It uses the
same current wrapper and P4-B, but places all three task-1 contexts in one
worker, matching the historical collection process topology.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path("/home/exouser/FORTE")
OUT_NAME = os.environ.get("ADMISSION_OUT_NAME", "ADMISSION_DIAGNOSTIC_TASKLEVEL_CREATION_SEED")
OUT = ROOT / "analysis/results/current_runtime_setpoint_mapping_validation_20260905" / OUT_NAME
WRAPPER = ROOT / "current4task_low_force_e3.py"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
TABERO = Path("/home/exouser/Tabero")
HISTORICAL_TABERO = Path("/media/volume/newdata/exouser/Tabero_e3lh")
WARP = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA")
OPENPI = Path("/media/volume/newdata/exouser/openpi/src")
MANIFEST = ROOT / "analysis/results/current_runtime_setpoint_mapping_validation_20260905/VALIDATION_CONTEXT_MANIFEST.json"


def main() -> None:
    target = json.loads(MANIFEST.read_text())
    contexts = [x for x in target["selected_contexts"] if int(x["task"]) == 1]
    ids = [x["context_id"] for x in contexts]
    OUT.mkdir(parents=True, exist_ok=True)
    worker_manifest = OUT / "WORKER_TARGET_MANIFEST.json"
    worker_manifest.write_text(json.dumps({
        "manifest_name": "TASKLEVEL_ADMISSION_ONLY_DIAGNOSTIC",
        "contexts": {cid: [] for cid in ids},
        "expected_contexts": len(ids),
        "expected_branches": 0,
    }, indent=2) + "\n")
    env = os.environ.copy()
    code_root = HISTORICAL_TABERO if os.environ.get("ADMISSION_USE_HISTORICAL_CODE") == "1" else TABERO
    env.update({
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join([str(WARP), str(code_root), str(TABERO), str(OPENPI)]),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y",
        "TABERO_ROOT": str(TABERO),
        "P5S0C_OUT": str(OUT), "P5S0C_WORKER": "1", "P5S0C_TASK_ID": "1",
        "P5S0C_TARGET_MANIFEST": str(worker_manifest),
        "P5S0C_CONTEXT_IDS": ",".join(ids), "P5S0C_SKIP_REPLAY": "1",
        "CONTINUOUS_STRICT_PREPROBE_WRAPPER": "1",
        "HDF5_TRAJ_SOURCE_DIR": str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        "LIBERO_CONFIG_DIR": str(TABERO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(TABERO / "benchmarks/datasets/libero/USD"),
    })
    if os.environ.get("ADMISSION_CREATION_SEED"):
        env["P5S0C_ENV_CREATION_SEED"] = os.environ["ADMISSION_CREATION_SEED"]
    log = OUT / "isaac_worker.log"
    with log.open("w") as fh:
        proc = subprocess.run(
            [str(ISAAC_PY), "-u", str(WRAPPER), "--worker"],
            cwd=TABERO, env=env, stdout=fh, stderr=subprocess.STDOUT,
            timeout=900,
        )
    (OUT / "RUN_RESULT.json").write_text(json.dumps({
        "returncode": proc.returncode, "contexts": ids, "log": str(log),
    }, indent=2) + "\n")
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
