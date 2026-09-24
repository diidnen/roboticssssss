"""Saved-probe support and uncertainty reporting, without changing inference.

Sequence parity is a conservative evidence gate, not a physical safety law.
It does not declare every shorter adaptive probe invalid in principle.
"""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np

E6_MANIFEST=Path('/home/exouser/FORTE/activeforcing_full_claim_closure_20260902_062809/E6_E7/PHYSICAL_BELIEF_MANIFEST.json')
VERIFIED_PHASE_COUNTS={'approach':45,'descend':35,'close':70,'hold':40,'probe_out':10,'probe_back':10,'probe_hold':5}


def annotate(probe_path, posterior):
    with Path(probe_path).open() as f:
        rows=list(csv.DictReader(f))
    counts=dict(Counter(r['probe_phase'] for r in rows))
    reasons=[]
    if counts!=VERIFIED_PHASE_COUNTS:
        reasons.append('PROBE_SEQUENCE_NOT_IN_VERIFIED_E1_E6_SUPPORT')
    mus=np.asarray(posterior['member_means'],float)
    logs=np.asarray(posterior['member_log_sigma'],float)
    if len(mus)!=3 or logs.shape!=mus.shape or not np.isfinite(mus).all() or not np.isfinite(logs).all():
        raise ValueError('Expected three finite physical-belief member outputs')
    if np.any(mus<=0):reasons.append('NONPOSITIVE_MEMBER_MU')
    scale=float(json.loads(E6_MANIFEST.read_text())['interval_scale'])
    epi=float(np.var(mus,ddof=1));alea=float(np.mean(np.exp(2*logs)))
    total=(epi+alea)*scale**2
    result=dict(posterior)
    result.update(std_semantics='epistemic sample SD only; retained for backwards compatibility',
        epistemic_variance=epi,epistemic_std=float(np.sqrt(epi)),aleatoric_variance=alea,
        calibrated_total_variance=total,calibrated_total_std=float(np.sqrt(total)),
        uncertainty_interval_scale=scale,uncertainty_calibration_source=str(E6_MANIFEST),
        feasibility_integration_semantics='unchanged empirical support at member means; sigma heads NOT integrated',
        input_contract={'verified_sequence_support':not reasons,'reasons':reasons,'observed_phase_counts':counts,
                        'reference':'96 E6 TRAIN/DEV raw probes replayed; 72 E1 raw probes replayed',
                        'scope':'support-parity gate, not a claim that shorter P4-B is inherently unsafe'},
        final_method_deployment_approved=False)
    return result


def require_input_support(posterior):
    if not posterior.get('input_contract',{}).get('verified_sequence_support',False):
        raise ValueError('BLOCKED_UNVERIFIED_PROBE_INPUT: use explicitly isolated diagnostic inference; do not execute a branch')
