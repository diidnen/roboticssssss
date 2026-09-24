"""Independent joined review of completed online qualification evidence.

Produces evidence, not an automatic authorization to use final test roots.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from common import read, sha
from audit_rollout import audit as provenance
from audit_geometric_label import audit as geometry
from behavior_audit import audit as behavior
from qualification_metrics import review as probabilities
from matched_seeds import matched_native_prefixes


def review(out):
    out = Path(out)
    rows = []
    for process in sorted((out/'branches').glob('*/PROCESS_EXIT.json')):
        job = process.parent
        p, g, b = provenance(job), geometry(job), behavior(job)
        if not p['passed'] or not g['passed']:
            raise RuntimeError('Invalid provenance or geometric label: '+str(job))
        trace = read(job/'BRANCH_TRACE.json')
        d = read(job/'PLANNER_DECISION.json')
        active = [t for t in trace if not t['vla_release_intent']]
        actual = [2*min(t['normal_force_N']) for t in active]
        contact = [2*min(t['normal_force_N']) for t in active if min(t['normal_force_N']) >= .15]
        rows.append(dict(
            job=str(job), context=b['context'], method=b['method'],
            selected_force_N=d['executed_force_N'], outcome=p['outcome'],
            real_online_requests=p['real_online_requests'],
            latency_median_ms=p['latency_median_ms'],
            provenance_valid=p['passed'], geometric_label_valid=g['passed'],
            pre_release_measured_drop=g['independent_measured_drop'],
            geometric_success_after_measured_drop=g['potential_recovery_case'],
            early_regrasp_candidate=b['early_regrasp_candidate_present'],
            regrasp_candidate=b['regrasp_candidate_present'],
            action_anomaly_rate=b['VLA_ACTION_ANOMALY_RATE'],
            probe_shift=b['probe_shift'], observed_events=b['events'],
            initial_online_identity=read(job/'INITIAL_ONLINE_CHUNK_IDENTITY.json'),
            policy_seeds=b['policy_seeds'],
            measured_squeeze_nonrelease_mean_N=float(np.mean(actual)) if actual else None,
            measured_squeeze_contact_conditional_mean_N=float(np.mean(contact)) if contact else None,
            nonrelease_samples=len(actual), bilateral_contact_samples=len(contact),
            evidence_sha256={name:sha(job/name) for name in ('PROCESS_EXIT.json', 'BRANCH_RESULT.json', 'BRANCH_TRACE.json', 'PLANNER_DECISION.json')},
        ))
    paired = {}
    for cid in sorted({r['context'] for r in rows}):
        group = [r for r in rows if r['context'] == cid]
        identities = [r['initial_online_identity'] for r in group]
        seeds = [[{k:s[k] for k in ('step', 'noise_seed', 'noise_sha256')} for s in r['policy_seeds']] for r in group]
        matching = all(x == identities[0] for x in identities) and matched_native_prefixes(seeds)
        if not matching:
            raise RuntimeError('Common X/initial online chunk/noise mismatch: '+cid)
        low = [r for r in group if r['method'] == 'FIXED_3']
        high = [r for r in group if r['selected_force_N'] > 3 and r['outcome']['full_task_success_y'] == 1]
        paired[cid] = dict(
            methods=[r['method'] for r in group], initial_observation_chunk_X_identical=True,
            all_native_inference_noise_seeds_matched=True,
            seed_matching_scope='Identical noise on shared native-query prefix; valid early native termination need not consume unused later seeds.',
            low_force_drop_higher_force_success=bool(low and low[0]['pre_release_measured_drop'] and high),
            low_force_failure_higher_force_success=bool(low and low[0]['outcome']['full_task_success_y'] == 0 and high),
            qualification_role='One burned root; paired empirical force effect under closed-loop feedback, not a population threshold.',
        )
    primary = [r for r in rows if r['method'] != 'FIXED_4']
    return dict(
        role='INDEPENDENT_DEV_EVIDENCE_REVIEW_NOT_FINAL_TEST_AUTHORIZATION',
        qualification_directory=str(out), implementation_sha256=sha(__file__),
        completed_primary_branches=len(primary), completed_boundary_fixed4=len(rows)-len(primary),
        real_online_requests=sum(r['real_online_requests'] for r in rows),
        all_provenance_and_geometry_valid=True,
        all_common_random_numbers_and_initial_X_valid=True,
        early_regrasp_candidate_branch_rate=float(np.mean([r['early_regrasp_candidate'] for r in primary])) if primary else None,
        regrasp_candidate_branch_rate=float(np.mean([r['regrasp_candidate'] for r in primary])) if primary else None,
        mean_per_branch_action_anomaly_rate=float(np.mean([r['action_anomaly_rate'] for r in primary])) if primary else None,
        potential_recovered_drop_successes=[r['job'] for r in rows if r['geometric_success_after_measured_drop']],
        contexts=paired, probabilities=probabilities(out), rows=rows,
        still_required=('Inspect complete force/behavior evidence and probe sensor diagnostic; issue final admission before runtime freeze.'
            if len(primary)==36 and len(rows)==40 else
            'Complete planned qualification branches; inspect force/behavior evidence and probe sensor diagnostic before final admission.'),
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('--save', required=True)
    a = p.parse_args()
    d = review(a.out)
    Path(a.save).write_text(json.dumps(d, indent=2)+'\n')
    print(json.dumps({k:v for k,v in d.items() if k not in ('rows','contexts','probabilities')}, indent=2))
