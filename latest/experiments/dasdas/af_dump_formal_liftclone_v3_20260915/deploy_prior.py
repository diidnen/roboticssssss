"""Friction-agnostic quadrature for lift-clone select().

v3 trained with point μ in the hold-chunk. Dump shear query is not v4 P4
evidence, so this prior does not load original-P4 belief rows and does not
insert hidden true friction.
"""
from __future__ import annotations

import numpy as np

from liftstyle_runtime import POSTERIOR_INTERFACE


def deploy_prior_posterior() -> dict:
    nodes = np.linspace(0.30, 0.85, 32)
    weights = np.ones(len(nodes), dtype=float) / len(nodes)
    return {
        "interface": POSTERIOR_INTERFACE,
        "candidate_actions_executed": 0,
        "hidden_friction_used": False,
        "integration_nodes": nodes.astype(float).tolist(),
        "integration_weights": weights.tolist(),
        "derived_summary_only": True,
        "source": "LIFTCLONE_FRICTION_AGNOSTIC_QUADRATURE_NOT_V4_P4_BELIEF",
    }
