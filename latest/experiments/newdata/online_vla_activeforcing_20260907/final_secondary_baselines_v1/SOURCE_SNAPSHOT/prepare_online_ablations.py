"""Prepare bounded dev ablations from the final frozen online runtime.

Only the feasibility variant changes. Main runtime files remain untouched.
"""
import argparse
import difflib
import shutil
from pathlib import Path
from common import HERE,read,write,sha

def replace_once(source,old,new):
    if source.count(old)!=1:raise RuntimeError('Qualified source structure changed')
    return source.replace(old,new)

def prepare(final,out):
    m=read(final/'FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json')
    protocol_path=HERE/'ONLINE_ABLATION_PROTOCOL_FROZEN_V3.json';protocol=read(protocol_path)
    if not m['FINAL_RUNTIME_USES_ONLINE_VLA'] or not m['parameters_locked']:
        raise RuntimeError('Main online runtime must be frozen first')
    if protocol_path.as_posix() not in m['source_hashes'] or sha(protocol_path)!=m['source_hashes'][str(protocol_path)]:
        raise RuntimeError('Ablation definitions were not frozen before final main physics')
    for p,d in {**m['source_hashes'],**protocol['source_hashes']}.items():
        if sha(p)!=d:raise RuntimeError('Frozen source/model changed: '+p)
    out.mkdir(exist_ok=False);snapshot=out/'SOURCE_SNAPSHOT';snapshot.mkdir()
    for p in (final/'SOURCE_SNAPSHOT').glob('*.py'):shutil.copy2(p,snapshot/p.name)
    for name in ('online_ablation_feasibility.py','prepare_online_ablations.py','run_online_ablations.py','audit_online_ablation.py'):
        shutil.copy2(HERE/name,snapshot/name)
    original=(snapshot/'worker.py').read_text();worker=original
    worker=replace_once(worker,'from phase_free_feasibility import PhaseFreeFeasibility',
        'from phase_free_feasibility import PhaseFreeFeasibility\n        from online_ablation_feasibility import AblationFeasibility, METHODS')
    worker=replace_once(worker,'feasibility=PhaseFreeFeasibility()',
        'feasibility=AblationFeasibility(args.method) if args.method in METHODS else PhaseFreeFeasibility()')
    worker=replace_once(worker,"if args.method=='ACTIVEFORCING' else float(args.method.removeprefix('FIXED_'))",
        "if args.method in ('ACTIVEFORCING',*METHODS) else float(args.method.removeprefix('FIXED_'))")
    (snapshot/'worker.py').write_text(worker)
    launch=(snapshot/'launch.py').read_text()
    launch=replace_once(launch,"if plan['root']!=5100:raise RuntimeError('Only dev root5100 authorized by this stage')",
        "if plan['root'] not in (5100,6200):raise RuntimeError('Only the two frozen burned ablation roots are allowed')")
    (snapshot/'launch.py').write_text(launch)
    write(out/'WORKER_VARIANT_DIFF.json',dict(original_worker_sha256=sha(final/'SOURCE_SNAPSHOT/worker.py'),
        ablation_worker_sha256=sha(snapshot/'worker.py'),
        unified_diff=''.join(difflib.unified_diff(original.splitlines(True),worker.splitlines(True))),
        changed_scope='Only variant import/instantiation and selecting its force. Same probe, state reconstruction, online VLA observations/actions, arbitration, controller and labels.'))
    for name in ('runtime.py','arbitration.py','phase_free_feasibility.py','common.py'):
        if sha(snapshot/name)!=sha(final/'SOURCE_SNAPSHOT'/name):raise RuntimeError('Unintended downstream runtime change')
    # Root/friction choices are prospective. The same canonical strata are used
    # as qualification and main; no final outcomes are used to choose cases.
    template=read(Path(m['qualification_source'])/'DEV_PLAN.json')['contexts']
    contexts=[dict(p,root=r,id=f"t{p['task']}_r{r}_{p['band'].lower()}") for r in protocol['roots'] for p in template]
    write(out/'DEV_PLAN.json',dict(role='BURNED_ROOT_ONLINE_ABLATIONS',roots=protocol['roots'],contexts=contexts))
    shutil.copy2(protocol_path,out/'ONLINE_ABLATION_PROTOCOL.json')
    sources={p:d for p,d in m['source_hashes'].items() if not Path(p).is_relative_to(final/'SOURCE_SNAPSHOT')}
    sources.update(protocol['source_hashes']);sources.update({str(p):sha(p) for p in snapshot.glob('*.py')})
    sources[str(out/'DEV_PLAN.json')]=sha(out/'DEV_PLAN.json')
    sources[str(out/'ONLINE_ABLATION_PROTOCOL.json')]=sha(out/'ONLINE_ABLATION_PROTOCOL.json')
    sources[str(final/'FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json')]=sha(final/'FINAL_VLA_ACTIVEFORCING_RUNTIME_MANIFEST.json')
    candidate=dict(m,source_hashes=sources,version='ONLINE_VLA_DEV_ABLATION_RUNTIME_V1',
        final_runtime_frozen=False,final_roots_used=False,source_final_runtime=str(final),
        selected_ablation_methods=list(protocol['methods']),main_method_parameters_changed=False)
    write(out/'CANDIDATE_RUNTIME_MANIFEST.json',candidate)
    write(out/'ABLATION_PREPARATION_COMPLETE.json',dict(physics_started=False,contexts=24,
        ablation_branches=96,new_AF_comparison_branches_max=protocol['additional_online_AF_comparison_branches_max'],
        reused_current_server_AF_controls=len(protocol['prospective_AF_control_reuse']),
        run_gate='All192 final main branches complete before any ablation physics'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.final,a.out)
