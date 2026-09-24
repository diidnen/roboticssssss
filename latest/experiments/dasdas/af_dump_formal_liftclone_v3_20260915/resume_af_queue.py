"""Resume AF lift-clone after a prefix failure. Does not change frozen sources."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_af_queue import (
    aggregate,
    now,
    read,
    run_one,
    server_ready,
    verify_plan,
    write,
)

SKIP_PATH = HERE / "PREFIX_FAILURES.json"


def skipped_ids() -> set[str]:
    if not SKIP_PATH.exists():
        return set()
    return {row["id"] for row in read(SKIP_PATH)}


def main() -> None:
    lock_file = (HERE / "LIFTCLONE_V3_FORMAL_QUEUE.lock").open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A live lift-clone v3 formal queue already holds the lock") from error
    if not (HERE / "SMOKE_PASS.json").exists():
        raise RuntimeError("Smoke has not passed; use run_af_queue.py")
    protocol, _smoke, contexts = verify_plan()
    skip = skipped_ids()
    unknown = skip - {row["id"] for row in contexts}
    if unknown:
        raise ValueError("PREFIX_FAILURES.json has unknown context ids: " + str(sorted(unknown)))
    nominal_by_id = read(HERE / "plan/REUSED_NOMINAL_OUTCOMES.json")
    if not server_ready():
        raise RuntimeError("Frozen pi0 policy server is not listening on port 6001")
    records = []
    executed = 0
    for index, context in enumerate(contexts):
        verify_plan()
        if context["id"] in skip:
            write(
                HERE / "STATUS.json",
                {
                    "status": "main_running",
                    "completed_contexts": executed,
                    "completed_rollouts": executed,
                    "skipped_prefix_failures": len(skip),
                    "current_context": context["id"],
                    "current_action": "skip_archived_prefix_failure",
                    "updated_utc": now(),
                },
            )
            continue
        write(
            HERE / "STATUS.json",
            {
                "status": "main_running",
                "completed_contexts": executed,
                "completed_rollouts": executed,
                "skipped_prefix_failures": len(skip),
                "current_context": context["id"],
                "updated_utc": now(),
            },
        )
        records.append(run_one(context, "main"))
        executed += 1
    if len(records) != len(contexts) - len(skip):
        raise RuntimeError("Resume record count does not match planned minus skipped")
    final = aggregate(records, nominal_by_id)
    final.update(
        planned_contexts=len(contexts),
        skipped_prefix_failures=read(SKIP_PATH) if SKIP_PATH.exists() else [],
        claim_boundary=(
            protocol["claim_boundary"]
            + "; one prefix failure archived and not retried "
            "(dump shear query lost contact after 12N grasp)"
        ),
    )
    write(HERE / "FINAL_RESULTS.json", final)
    write(
        HERE / "STATUS.json",
        {
            "status": "complete",
            "completed_contexts": len(contexts),
            "completed_rollouts": len(records),
            "skipped_prefix_failures": len(skip),
            "current_context": None,
            "updated_utc": now(),
        },
    )
    print(json.dumps(final["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
