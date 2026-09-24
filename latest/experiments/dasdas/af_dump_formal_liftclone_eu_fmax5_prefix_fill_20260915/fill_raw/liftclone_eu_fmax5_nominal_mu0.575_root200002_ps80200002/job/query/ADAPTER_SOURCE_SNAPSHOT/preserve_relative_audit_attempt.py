"""One-off preservation of relative-path audit receipts before absolute re-audit.

Does not edit any physics, query, action, outcome, frozen runtime, or label.
Run only after the completed group's audit/pack process has exited.
"""
import json
from pathlib import Path
from rootlocal_collection_contract import HERE, read, sha, write, now


def main():
    attempt = HERE / 'original_rootlocal_dataset_v1/groups/train_mu0.750_root200002/attempt_001'
    job = attempt / 'job'
    if read(attempt / 'PROCESS_EXIT.json')['exit_code'] != 0:
        raise ValueError('Group not successfully terminal')
    paths = [job / 'query/INDEPENDENT_NATIVE_FORCE_REBUILD.json',
             *sorted(job.glob('branch_*/INDEPENDENT_BRANCH_AUDIT.json')),
             job / 'INDEPENDENT_GROUP_AUDIT.json']
    if len(paths) != 10 or not all(p.is_file() for p in paths):
        raise ValueError('Expected exactly query + eight branch + group audits')
    for path in paths[:-1]:
        keys = list(read(path)['source_hashes'])
        if not keys or not all(not Path(k).is_absolute() for k in keys):
            raise ValueError('This repair is only for newly generated relative-path audits')
    archive = HERE.parent / 'af_dump_original_train_mu0.750_root200002_20260913.tar.gz'
    receipt = archive.with_suffix('.receipt.json')
    packed = read(receipt)
    if packed['sha256'] != sha(archive) or packed['size'] != archive.stat().st_size:
        raise ValueError('Completed pre-repair archive receipt mismatch')
    backup = job / 'AUDIT_RELATIVE_PATH_ATTEMPT_001'
    if backup.exists(): raise ValueError('Preserved attempt already exists; do not repeat')
    renamed_archive = archive.with_name(archive.name.replace('.tar.gz', '_relative_path_audit_v1.tar.gz'))
    renamed_receipt = renamed_archive.with_suffix('.receipt.json')
    if renamed_archive.exists() or renamed_receipt.exists(): raise ValueError('Preserved archive already exists')
    backup.mkdir()
    mapping = []
    for path in paths:
        destination = backup / path.relative_to(job)
        destination.parent.mkdir(parents=True, exist_ok=True)
        mapping.append({'old_path': str(path), 'preserved_path': str(destination), 'sha256': sha(path)})
    mapping.extend([{'old_path': str(archive), 'preserved_path': str(renamed_archive), 'sha256': sha(archive)},
                    {'old_path': str(receipt), 'preserved_path': str(renamed_receipt), 'sha256': sha(receipt)}])
    write(backup / 'PRESERVATION_PLAN.json', {'created_utc': now(), 'reason': 'Relative source paths depend on audit cwd',
        'raw_experiment_evidence_unchanged': True, 'files': mapping})
    for entry in mapping:
        source, target = Path(entry['old_path']), Path(entry['preserved_path'])
        source.rename(target)
        if sha(target) != entry['sha256']: raise ValueError('Preservation hash mismatch')
    write(backup / 'PRESERVATION_COMPLETE.json', {'completed': True, 'files': len(mapping), 'finished_utc': now()})
    print(json.dumps({'preserved': str(backup), 'archive': str(renamed_archive), 'ready_for_absolute_reaudit': True}))


if __name__ == '__main__':
    main()
