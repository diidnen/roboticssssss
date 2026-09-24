#!/usr/bin/env python3
"""Start the already-validated frozen JAX pi0 server in a MASS log namespace."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
VTLA = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA")
SERVER = Path("/home/exouser/FORTE/online_vla_restore_20260907/server_v3_frozen/SOURCE_SNAPSHOT/server.py")
COMMON = SERVER.with_name("common.py")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18885); args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy(); env.update(PYTHONPATH=os.pathsep.join([
        "/media/volume/newdata/exouser/tabero/uv-cache/archive-v0/FiR5so0BgJNw-JER",
        "/media/volume/newdata/exouser/softvtbench/openpi-venv/lib/python3.11/site-packages",
        str(VTLA / "src"), str(VTLA / "packages/openpi-client/src"),
        "/home/exouser/Tabero/benchmarks/openpi/openpi-client/src"]),
        XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="4")
    command = [str(VTLA / ".venv/bin/python"), "-u", str(SERVER), "--out", str(args.out), "--port", str(args.port)]
    with (args.out / "SERVER.log").open("x") as log:
        process = subprocess.Popen(command, cwd=VTLA, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    with (args.out / "PROCESS.json").open("x") as stream:
        json.dump({"pid": process.pid, "command": command,
                   "source_hashes": {str(SERVER): sha(SERVER), str(COMMON): sha(COMMON)}}, stream, indent=2)
        stream.write("\n")
    print(process.pid)


if __name__ == "__main__": main()
