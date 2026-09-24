#!/usr/bin/env python3
"""Execute an already-frozen MASS online queue without changing its order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mass_online_launch import launch


def read(path): return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True)
    parser.add_argument("--plan", required=True); parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--limit", type=int); parser.add_argument("--port", type=int, default=18885)
    args = parser.parse_args(); plan = read(args.plan)
    queue = plan["execution_queue"][:args.limit] if args.limit is not None else plan["execution_queue"]
    for position, entry in enumerate(queue, 1):
        launch(Path(args.out), Path(args.plan), Path(args.runtime_manifest), entry["context_index"], entry["method"], args.port)
        print(json.dumps({"queue_position": position, "queue_total": len(queue), **entry}), flush=True)


if __name__ == "__main__": main()
