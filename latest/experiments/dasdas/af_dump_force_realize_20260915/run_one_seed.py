"""One frozen seed: 200014 × official dump μ × v4 20-force grid."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
REPO = BASE / "RoboTwin"
PYTHON = BASE / "venv_robotwin/bin/python3"
INFER = HERE / "run_sweep_context.py"
FORCES = [round(0.25 * i, 2) for i in range(1, 21)]
FRICTIONS = [0.425, 0.575, 0.85]
SEED = 200014


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def status(name: str, **fields) -> None:
    payload = {"status": name, "updated_utc": now()}
    payload.update(fields)
    write(HERE / "STATUS.json", payload)


def main() -> None:
    contexts = [
        {
            "id": f"force_realize_mu{mu:.3f}_seed{SEED}",
            "seed": SEED,
            "friction": mu,
            "forces_N": FORCES,
            "role": "force_realize_sanity",
            "mu_eff_average": 0.5 * (0.3 + mu),
        }
        for mu in FRICTIONS
    ]
    write(HERE / "PLAN.json", {"contexts": contexts, "n": len(contexts), "forces_N": FORCES, "seed": SEED})
    raw = HERE / "main_raw"
    raw.mkdir(exist_ok=True)
    status("running", n=len(contexts), done=0)
    for index, context in enumerate(contexts):
        job = raw / context["id"] / "job"
        if (job / "FORCE_REALIZE_CONTEXT_RESULT.json").exists():
            continue
        job_parent = raw / context["id"]
        job_parent.mkdir(parents=True, exist_ok=True)
        write(job_parent / "CONTEXT.json", {"context": context})
        if job.exists():
            shutil.rmtree(job)
        log = job_parent / "worker.log"
        command = [str(PYTHON), str(INFER), "--context", str(job_parent / "CONTEXT.json"), "--out", str(job)]
        with log.open("w") as stream:
            process = subprocess.run(
                command,
                cwd=str(REPO),
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        write(job_parent / "EXIT.json", {"returncode": process.returncode, "finished_utc": now()})
        if process.returncode != 0:
            status("failed", context=context["id"], returncode=process.returncode)
            raise SystemExit(process.returncode)
        status("running", n=len(contexts), done=index + 1, last=context["id"])
    status("complete", n=len(contexts), done=len(contexts))


if __name__ == "__main__":
    main()
