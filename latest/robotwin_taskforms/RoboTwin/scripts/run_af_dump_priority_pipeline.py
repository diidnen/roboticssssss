#!/usr/bin/env python3
"""Run the single-task priority pipeline for dump_bin_bigbin."""

from __future__ import annotations

import os

os.environ["AF_PRIORITY_TASK"] = "dump_bin_bigbin"
os.environ.setdefault(
    "AF_PRIORITY_EXPERIMENT",
    "/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/experiments/af_dump_bin_bigbin_forcegrid324_v1",
)

from run_af_handover_priority_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
