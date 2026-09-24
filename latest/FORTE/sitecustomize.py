"""Compatibility shim for the installed LeRobot package used by OpenPI.

The frozen Tabero-VTLA checkout imports the historical ``lerobot.datasets``
namespace, while this server environment exposes the same implementation under
``lerobot.common.datasets``.  This shim only aliases imports; it does not alter
the π0 model or data.
"""
from __future__ import annotations

import importlib
import sys

try:
    datasets = importlib.import_module("lerobot.common.datasets")
    dataset = importlib.import_module("lerobot.common.datasets.lerobot_dataset")
    sys.modules.setdefault("lerobot.datasets", datasets)
    sys.modules.setdefault("lerobot.datasets.lerobot_dataset", dataset)
except Exception:
    pass
