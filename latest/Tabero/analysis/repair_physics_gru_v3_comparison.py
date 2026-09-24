#!/usr/bin/env python3
"""Repair only the offline v1/v2 common-TEST comparison in a frozen v3 run."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
from pathlib import Path


REPO = Path("/home/exouser/Tabero")
OUT = REPO / "analysis/results/physics_gru_v3_direct_contact_20260829_055635"


def load_v3():
    spec = importlib.util.spec_from_file_location("physics_gru_v3_repair", REPO / "analysis/physics_gru_v3_direct_contact.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    mod = load_v3()
    traces, _, _ = mod.load_traces()
    test = [t for t in traces if t.split == "TEST"]
    old_traj, old_events = mod.old_model_common_test(test, mod.torch.device("cpu"))

    traj_path = OUT / "TRAJECTORY_METRICS.csv"
    with traj_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if not str(r.get("status", "")).startswith("UNAVAILABLE:")]
    mod.write_csv(traj_path, rows + old_traj)

    event_path = OUT / "DIRECT_EVENT_METRICS.csv"
    with event_path.open(newline="", encoding="utf-8") as f:
        event_rows = list(csv.DictReader(f))
    mod.write_csv(event_path, event_rows + old_events)

    # Refresh provenance because the v3 source now includes the import fix.
    mod.write_json(OUT / "REUSE_AND_PROVENANCE.json", mod.provenance(OUT, traces))

    report_path = OUT / "FINAL_REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    old = "On the common direct TEST windows, v3 full-rollout relative-position MAE was 0.012652 m and relative-velocity MAE 0.042128 m/s. `TRAJECTORY_METRICS.csv` contains the constant baseline and common-window v1/v2 kinematic comparison; v1/v2 have no comparable direct-force outputs."
    new = "On the common direct TEST windows, v3 full-rollout relative-position MAE was 0.012652 m and relative-velocity MAE 0.042128 m/s; v1 was 0.010842 m / 0.007345 m/s and v2 was 0.059141 m / 0.024666 m/s. `TRAJECTORY_METRICS.csv` contains the common-window comparison; v1/v2 have no comparable direct-force outputs."
    report_path.write_text(report.replace(old, new), encoding="utf-8")

    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "MANIFEST.sha256")
    (OUT / "MANIFEST.sha256").write_text("".join(f"{digest(p)}  {p.name}\n" for p in files), encoding="utf-8")
    print({"trajectory_rows_added": len(old_traj), "event_rows_added": len(old_events), "out": str(OUT)})


if __name__ == "__main__":
    main()
