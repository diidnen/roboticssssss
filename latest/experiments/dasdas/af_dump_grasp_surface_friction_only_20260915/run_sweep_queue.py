"""Queue for grasp-surface-only dump sanity (200014 + 200003 negative control)."""
from __future__ import annotations

import hashlib
import json
import os
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
SANITY_SEED = 200014
CONTROL_SEED = 200003


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
    contexts = []
    for seed, role in ((SANITY_SEED, "sanity"), (CONTROL_SEED, "negative_control_wedging")):
        for mu in FRICTIONS:
            contexts.append(
                {
                    "id": f"grasp_surface_mu{mu:.3f}_seed{seed}",
                    "seed": seed,
                    "friction": mu,
                    "forces_N": FORCES,
                    "role": role,
                    "mu_eff_average": 0.5 * (0.3 + mu),
                }
            )
    write(HERE / "PLAN.json", {"contexts": contexts, "n": len(contexts), "forces_N": FORCES})
    raw = HERE / "main_raw"
    raw.mkdir(exist_ok=True)
    status("running", n=len(contexts), done=0)
    for index, context in enumerate(contexts):
        job = raw / context["id"] / "job"
        if (job / "GRASP_SURFACE_CONTEXT_RESULT.json").exists():
            continue
        job_parent = raw / context["id"]
        job_parent.mkdir(parents=True, exist_ok=True)
        ctx_path = job_parent / "CONTEXT.json"
        write(ctx_path, {"context": context})
        if job.exists():
            import shutil

            shutil.rmtree(job)
        log = job_parent / "worker.log"
        command = [str(PYTHON), str(INFER), "--context", str(ctx_path), "--out", str(job)]
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
