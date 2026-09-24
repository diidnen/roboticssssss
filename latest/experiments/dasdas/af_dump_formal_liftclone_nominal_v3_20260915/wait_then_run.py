"""Wait for the live AF lift-clone queue, then start Nominal on the same prefix."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
AF = Path("/media/volume/dasdas/exouser/af_dump_formal_liftclone_v3_20260915")
BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
PYTHON = BASE / "venv_robotwin/bin/python3"
QUEUE = HERE / "run_nominal_queue.py"
POLL_S = 60


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def server_ready() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 6001), timeout=2):
            return True
    except OSError:
        return False


def af_queue_alive() -> bool:
    try:
        output = subprocess.check_output(["pgrep", "-f", "af_dump_formal_liftclone_v3_20260915/run_af_queue.py"], text=True)
    except subprocess.CalledProcessError:
        return False
    return any(line.strip() for line in output.splitlines())


def af_complete() -> bool:
    if not (AF / "STATUS.json").exists() or not (AF / "FINAL_RESULTS.json").exists():
        return False
    status = read(AF / "STATUS.json")
    return status.get("status") == "complete" and status.get("completed_contexts") == 24


def main() -> None:
    lock_file = (HERE / "LIFTCLONE_NOMINAL_V3_WAITER.lock").open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A live lift-clone Nominal waiter already holds the lock") from error
    if (HERE / "STATUS.json").exists() and read(HERE / "STATUS.json").get("status") == "complete":
        write(HERE / "WAITER_STATUS.json", {"status": "already_complete", "updated_utc": now()})
        return
    write(
        HERE / "WAITER_STATUS.json",
        {
            "status": "waiting_for_af",
            "af_experiment": str(AF),
            "updated_utc": now(),
        },
    )
    while not af_complete():
        if not server_ready():
            write(HERE / "WAITER_STATUS.json", {"status": "blocked_pi0_down", "updated_utc": now()})
            raise RuntimeError("Frozen pi0 died before Nominal could start")
        if not af_queue_alive() and not af_complete():
            write(
                HERE / "WAITER_STATUS.json",
                {
                    "status": "blocked_af_died",
                    "af_status": read(AF / "STATUS.json") if (AF / "STATUS.json").exists() else None,
                    "updated_utc": now(),
                },
            )
            raise RuntimeError("AF lift-clone queue died before complete")
        write(
            HERE / "WAITER_STATUS.json",
            {
                "status": "waiting_for_af",
                "af_status": read(AF / "STATUS.json"),
                "updated_utc": now(),
            },
        )
        time.sleep(POLL_S)
    if not server_ready():
        write(HERE / "WAITER_STATUS.json", {"status": "blocked_pi0_down", "updated_utc": now()})
        raise RuntimeError("Frozen pi0 is not listening after AF complete")
    write(
        HERE / "WAITER_STATUS.json",
        {
            "status": "starting_nominal",
            "af_status": read(AF / "STATUS.json"),
            "updated_utc": now(),
        },
    )
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    with (HERE / "waiter_queue.log").open("a") as log:
        process = subprocess.Popen(
            [str(PYTHON), "-u", str(QUEUE)],
            cwd=str(HERE),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        write(
            HERE / "WAITER_STATUS.json",
            {
                "status": "nominal_running",
                "nominal_queue_pid": process.pid,
                "updated_utc": now(),
            },
        )
        code = process.wait()
    write(
        HERE / "WAITER_STATUS.json",
        {
            "status": "complete" if code == 0 else "nominal_failed",
            "nominal_queue_exit": code,
            "updated_utc": now(),
        },
    )
    if code != 0:
        raise RuntimeError("Nominal lift-clone queue failed")


if __name__ == "__main__":
    main()
