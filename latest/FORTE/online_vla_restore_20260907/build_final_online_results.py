"""Strict final-main table builder; all192 verified online branches required."""
import argparse
import csv
import itertools
from pathlib import Path
import numpy as np
from common import read, write, sha
from run_final_online_vla import validate, METHODS
from audit_rollout import audit
from audit_geometric_label import audit as geometry
from matched_seeds import matched_native_prefixes


def build(out, destination):
    out, destination = Path(out), Path(destination)
    manifest, plans = validate(out)
    if read(out/'FINAL_MAIN_EXECUTION_COMPLETE.json')['branches'] != 192:
        raise RuntimeError('Final execution incomplete')
    if destination.exists():
        raise FileExistsError('Keep previous report artifacts; use a new version')
    manifest_sha = sha(out/'FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json')
    rows = []; hashes = {}; common = {}
    for plan in plans:
        for method in METHODS:
            job = out/'branches'/(plan['id']+'__'+method)
            admission = read(job/'FINAL_EVIDENCE_ADMISSION.json')
            if admission['final_manifest_sha256'] != manifest_sha:
                raise RuntimeError('Evidence admitted under another runtime')
            p, g = audit(job), geometry(job)
            if not p['passed'] or not g['passed']:
                raise RuntimeError('Invalid online evidence cannot enter final table: '+str(job))
            identity = read(job/'INITIAL_ONLINE_CHUNK_IDENTITY.json')
            seeds = [{k:r[k] for k in ('step','noise_seed','noise_sha256')}
                     for r in [read(x) for x in sorted((job/'RPC').glob('*.json'))]]
            signature = common.setdefault(plan['id'],dict(initial=identity,schedules=[]))
            signature['schedules'].append(seeds)
            if signature['initial'] != identity or not matched_native_prefixes(signature['schedules']):
                raise RuntimeError('Unmatched X/online initial chunk/policy randomness')
            trace = read(job/'BRANCH_TRACE.json'); decision = read(job/'PLANNER_DECISION.json')
            active = [t for t in trace if not t['vla_release_intent']]
            force = [2*min(t['normal_force_N']) for t in active]
            contact = [2*min(t['normal_force_N']) for t in active if min(t['normal_force_N']) >= .15]
            if not force:
                raise RuntimeError('No commanded-grasp observations; undefined primary force metric')
            y = p['outcome']; native = bool(y['dropped']); measured = bool(g['independent_measured_drop'])
            rows.append(dict(root=plan['root'],context=plan['id'],task=plan['task'],friction=plan['band'],method=method,
                full_task_success=int(y['full_task_success_y']),lift_success=int(y['lift_success']),
                dropped=int(native or measured),native_dropped=int(native),measured_pre_release_drop=int(measured),
                selected_force_N=decision['executed_force_N'],measured_squeeze_N=float(np.mean(force)),
                contact_conditional_squeeze_N=float(np.mean(contact)) if contact else None,
                nonrelease_samples=len(force),bilateral_contact_samples=len(contact),
                recovered_drop_success=int(g['potential_recovery_case']),real_online_requests=p['real_online_requests'],
                job=str(job),result_sha256=sha(job/'BRANCH_RESULT.json')))
            hashes[str(job/'FINAL_EVIDENCE_ADMISSION.json')] = sha(job/'FINAL_EVIDENCE_ADMISSION.json')
    if len(rows) != 192 or len(common) != 48:
        raise RuntimeError('Final denominator is incomplete')
    def aggregate(group):
        contacts = [r['contact_conditional_squeeze_N'] for r in group if r['contact_conditional_squeeze_N'] is not None]
        return dict(contexts=len(group),FULL_TASK_SR=float(np.mean([r['full_task_success'] for r in group])),
            LIFT_SR=float(np.mean([r['lift_success'] for r in group])),DROP_RATE=float(np.mean([r['dropped'] for r in group])),
            NATIVE_DROP_RATE=float(np.mean([r['native_dropped'] for r in group])),
            MEASURED_PRE_RELEASE_DROP_RATE=float(np.mean([r['measured_pre_release_drop'] for r in group])),
            MEAN_SELECTED_SETPOINT=float(np.mean([r['selected_force_N'] for r in group])),
            MEASURED_BILATERAL_SQUEEZE=float(np.mean([r['measured_squeeze_N'] for r in group])),
            BILATERAL_CONTACT_CONDITIONAL_SQUEEZE=float(np.mean(contacts)) if contacts else None,
            CONTACT_CONDITIONAL_MISSING_BRANCHES=len(group)-len(contacts),
            TOTAL_NONRELEASE_SAMPLES=sum(r['nonrelease_samples'] for r in group),
            TOTAL_BILATERAL_CONTACT_SAMPLES=sum(r['bilateral_contact_samples'] for r in group))
    main = {m:aggregate([r for r in rows if r['method']==m]) for m in METHODS}
    if any(m['contexts'] != 48 for m in main.values()):
        raise RuntimeError('Method denominator changed')
    roots = sorted({p['root'] for p in plans})
    by_root = {str(root):{m:aggregate([r for r in rows if r['root']==root and r['method']==m]) for m in METHODS} for root in roots}
    pairs = {}
    for method in METHODS[:-1]:
        differences = [by_root[str(root)]['ACTIVEFORCING']['FULL_TASK_SR']-by_root[str(root)][method]['FULL_TASK_SR'] for root in roots]
        draws = [np.mean([differences[i] for i in sample]) for sample in itertools.product(range(4),repeat=4)]
        pairs[method] = dict(paired_full_task_SR_difference=float(np.mean(differences)),
            root_differences=dict(zip(map(str,roots),differences)),
            descriptive_root_bootstrap_95_percentile=np.quantile(draws,[.025,.975]).tolist(),
            root_clusters=4,ordered_resamples=256,
            limitation='Only four independent root groups; percentile interval is descriptive, not a strong asymptotic or population-wide significance claim.')
    report = dict(role='FINAL_ONLINE_FROZEN_VLA_MAIN_EVIDENCE',runtime_manifest_sha256=manifest_sha,
        main=main,by_root=by_root,paired_AF_comparisons=pairs,
        FORCE_SAVING_VS_FIXED5=1-main['ACTIVEFORCING']['MEAN_SELECTED_SETPOINT']/5,
        force_saving_basis='Selected force setpoint; never post-drop zero-force samples',
        recovered_drop_successes=[r['job'] for r in rows if r['recovered_drop_success']],
        roots=roots,contexts=48,branches=192,real_online_requests=sum(r['real_online_requests'] for r in rows),
        all_branch_sources='ONLINE_VLA',all_initial_X_and_sampling_seeds_matched=True,
        seed_matching_scope='Shared native-query prefix; legitimately terminated failures retain their original shorter schedules.',
        source_hashes=hashes,rows=rows,scripted_results_included=False,
        remaining_paper_requirements='Small online dev ablations and final paper/table/figure validation; this table alone does not complete the master goal.')
    destination.mkdir(parents=True,exist_ok=False)
    write(destination/'FINAL_ONLINE_VLA_MAIN_RESULTS.json',report)
    with (destination/'FINAL_ONLINE_VLA_MAIN_ROWS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--destination',required=True);a=p.parse_args()
    report=build(a.out,a.destination)
    print('VERIFIED_FINAL_MAIN_BRANCHES',report['branches'])
