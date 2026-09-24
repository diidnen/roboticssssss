#!/usr/bin/env python3
"""Exercise a minimal CUDA allocation without using NVML."""

import argparse
import csv
import datetime as dt
import gc
import time
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--attempts", type=int, default=10)
    args = parser.parse_args()

    import torch

    rows = []
    for index in range(args.attempts):
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            available = bool(torch.cuda.is_available())
            count = int(torch.cuda.device_count())
            name = torch.cuda.get_device_name(0) if count else ""
            before_free, before_total = torch.cuda.mem_get_info(0)
            tensor = torch.arange(4096, dtype=torch.float32, device="cuda:0")
            result = (tensor * 2.0 + 1.0).sum()
            torch.cuda.synchronize(0)
            value = float(result.item())
            del result, tensor
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(0)
            after_free, after_total = torch.cuda.mem_get_info(0)
            rows.append(
                {
                    "attempt": index + 1,
                    "timestamp_utc": timestamp,
                    "environment": args.environment,
                    "torch_version": torch.__version__,
                    "cuda_available": available,
                    "device_count": count,
                    "device_name": name,
                    "free_bytes_before": before_free,
                    "total_bytes_before": before_total,
                    "free_bytes_after": after_free,
                    "total_bytes_after": after_total,
                    "operation_result": value,
                    "success": True,
                    "error": "",
                }
            )
        except Exception:
            rows.append(
                {
                    "attempt": index + 1,
                    "timestamp_utc": timestamp,
                    "environment": args.environment,
                    "torch_version": getattr(torch, "__version__", "unknown"),
                    "cuda_available": False,
                    "device_count": 0,
                    "device_name": "",
                    "free_bytes_before": "",
                    "total_bytes_before": "",
                    "free_bytes_after": "",
                    "total_bytes_after": "",
                    "operation_result": "",
                    "success": False,
                    "error": traceback.format_exc().strip().replace("\n", " | "),
                }
            )
        time.sleep(0.1)

    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    successes = sum(bool(row["success"]) for row in rows)
    print(f"CUDA_TEST_SUCCESS_COUNT={successes}")
    print(f"CUDA_TEST_FAILURE_COUNT={len(rows) - successes}")


if __name__ == "__main__":
    main()
