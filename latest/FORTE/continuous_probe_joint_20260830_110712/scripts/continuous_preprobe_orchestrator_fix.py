#!/usr/bin/env python3
"""Engineering-only race-free launcher for the frozen continuous collector.

The original launcher let four task processes rewrite one identical manifest.
This wrapper writes it once, then calls the frozen collector's unchanged
``run_one``/``--worker`` path.  It does not alter contexts, forces, repeats,
state capture, controller execution, success labels, or retry policy.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


FROZEN = Path(__file__).with_name("continuous_preprobe_collect.py")


def load():
    spec = importlib.util.spec_from_file_location("frozen_continuous_collector", FROZEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def task_run(args) -> None:
    mod = load(); out = Path(args.out)
    manifest = out / "CONTINUOUS_STRICT_PREPROBE_TARGET_MANIFEST.json"
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    ids = [c["context_id"] for c in protocol["dev_context_population"] if int(c["task"]) == args.task]
    rec = mod.run_one(out, manifest, args.task, ids, args.timeout_s)
    mod.write_json(out / f"TASK{args.task}_COLLECTOR_RECORD.json", rec)
    if rec["returncode"] != 0:
        raise SystemExit(1)


def orchestrate(args) -> None:
    mod = load(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = mod.build_manifest(Path(args.protocol), out)
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    by_task = defaultdict(list)
    for c in protocol["dev_context_population"]: by_task[int(c["task"])].append(c["context_id"])
    procs = []
    for task in sorted(by_task):
        procs.append((task, subprocess.Popen([
            sys.executable, "-u", str(Path(__file__).resolve()), "--task", str(task),
            "--protocol", str(Path(args.protocol)), "--out", str(out), "--timeout-s", str(args.timeout_s),
        ], cwd=Path(__file__).resolve().parent)))
    records = []
    for task, proc in procs:
        proc.wait(); rp = out / f"TASK{task}_COLLECTOR_RECORD.json"
        records.append(json.loads(rp.read_text()) if rp.exists() else {"task":task,"returncode":proc.returncode,"error":"record missing"})
    mod.write_json(out / "CONTINUOUS_COLLECTION_RUN_MANIFEST.json", {
        "status": "COMPLETE" if all(int(r.get("returncode",1)) == 0 for r in records) else "ENGINEERING_FAILURE",
        "protocol": str(Path(args.protocol)), "protocol_sha256": mod.sha256(Path(args.protocol)),
        "target_manifest": str(manifest), "target_manifest_sha256": mod.sha256(manifest),
        "expected_contexts": len(protocol["dev_context_population"]),
        "expected_branches": sum(len(v) for v in json.loads(manifest.read_text())["contexts"].values()),
        "workers": records, "scientific_failures_retried": 0,
        "engineering_correction": "manifest written once before task launch; frozen worker unchanged",
    })
    mod.write_csv(out / "CONTINUOUS_COLLECTION_RUN_MANIFEST.csv", records)
    if any(int(r.get("returncode",1)) != 0 for r in records): raise SystemExit(1)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--protocol",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--timeout-s",type=int,default=10800); ap.add_argument("--task",type=int)
    args=ap.parse_args(); task_run(args) if args.task is not None else orchestrate(args)


if __name__ == "__main__": main()
