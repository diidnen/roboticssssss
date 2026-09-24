#!/usr/bin/env python3
"""Probe the frozen WebSocket server metadata without simulator physics."""

import argparse
import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--port", type=int, default=18885)
    args = parser.parse_args()

    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    rows = []
    for index in range(args.attempts):
        started = time.perf_counter()
        try:
            client = WebsocketClientPolicy("127.0.0.1", args.port)
            metadata = client.get_server_metadata()
            client._ws.close()
            rows.append(
                {
                    "attempt": index + 1,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "success": True,
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                    "server_pid": metadata.get("pid"),
                    "checkpoint": metadata.get("checkpoint"),
                    "checkpoint_sha256": metadata.get("checkpoint_sha256"),
                    "backend": metadata.get("backend"),
                    "downstream_action_source": metadata.get("downstream_action_source"),
                    "error": "",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "attempt": index + 1,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "success": False,
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                    "server_pid": None,
                    "checkpoint": None,
                    "checkpoint_sha256": None,
                    "backend": None,
                    "downstream_action_source": None,
                    "error": repr(error),
                }
            )
        time.sleep(0.1)

    args.output.write_text(json.dumps({"port": args.port, "rows": rows}, indent=2) + "\n")
    successes = sum(row["success"] for row in rows)
    print(f"POLICY_SERVER_SUCCESS_COUNT={successes}")
    print(f"POLICY_SERVER_FAILURE_COUNT={len(rows) - successes}")


if __name__ == "__main__":
    main()
