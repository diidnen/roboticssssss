#!/usr/bin/env python3
"""Minimal deterministic executor for this notebook when Jupyter is unavailable."""

from __future__ import annotations

import contextlib
import io
import json
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
path = HERE / "UTILITY_FORCE_SCALE_ANALYSIS.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))
namespace: dict = {"__name__": "__notebook__"}
count = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    count += 1
    source = "".join(cell["source"])
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            exec(compile(source, f"{path.name}:cell-{count}", "exec"), namespace)
        cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": capture.getvalue().splitlines(keepends=True)}]
    except Exception as exc:
        cell["outputs"] = [{
            "ename": type(exc).__name__, "evalue": str(exc), "output_type": "error",
            "traceback": traceback.format_exc().splitlines(),
        }]
        path.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
        raise
    cell["execution_count"] = count
nb["metadata"]["execution"] = {
    "engine": "standard-library sequential executor (Jupyter packages unavailable)",
    "status": "completed",
    "code_cells": count,
}
path.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
print(f"executed {count} code cells top-to-bottom")
