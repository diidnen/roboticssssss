"""Recompute actual ablation force decision and verify online arm provenance."""
import sys
from pathlib import Path
import numpy as np
from common import ROOT,read,sha
sys.path.insert(0,str(ROOT))
from audit_rollout import audit
from audit_geometric_label import audit as geometry
from online_ablation_feasibility import AblationFeasibility,METHODS

def review(job):
    job=Path(job);p=audit(job);g=geometry(job);d=read(job/'PLANNER_DECISION.json')
    if d['method'] not in METHODS:raise RuntimeError('Not a frozen ablation variant')
    m=AblationFeasibility(d['method'],device='cpu')
    x=np.load(job/'PREACTION_SEQUENCE.npy');q=read(job/'PREACTION_POSTERIOR.json')
    actual=m.select(x,q)
    if not p['passed'] or not g['passed']:raise RuntimeError('Invalid online ablation execution')
    if d['executed_force_N']!=actual['selected_force_N'] or d['selected_force_N']!=actual['selected_force_N']:
        raise RuntimeError('Executed ablation force does not match frozen variant')
    for key in ('force_grid_N','used_integration_nodes','used_integration_weights','scientific_label','model_target'):
        if actual[key]!=d[key]:raise RuntimeError('Ablation semantics differ: '+key)
    if not np.allclose(actual['p_success'],d['p_success'],rtol=0,atol=1e-7):
        raise RuntimeError('Ablation probability not reproducible')
    if d['phase_representation']!='NONE' or not d['probe_executed'] or d['legal_no_probe_pathway']:
        raise RuntimeError('Invalid phase/NoProbe claim')
    return dict(passed=True,provenance=p,geometry=g,variant=d['method'],
        selected_force=d['executed_force_N'],decision_sha256=sha(job/'PLANNER_DECISION.json'),
        scientific_label=d['scientific_label'],actual_full_task_success=p['outcome']['full_task_success_y'])
