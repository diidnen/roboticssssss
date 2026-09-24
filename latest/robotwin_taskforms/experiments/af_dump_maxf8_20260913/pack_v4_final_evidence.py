"""Closed scientific evidence overlay; linked raw archives still require Anvil checks."""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tarfile
from pack_v4_stage import sha

HERE = Path(__file__).resolve().parent
OUT = Path('/media/volume/newdata/exouser/af_dump_maxf8_confirmation_20260913/inference_v4').resolve()
ARCHIVES = Path('/media/volume/newdata/exouser/af_dump_maxf8_archives_20260913')

def read(path):
    return json.loads(path.read_text())

def main():
    complete = read(HERE/'confirmation_driver_v1/COMPLETE.json')
    audit = read(OUT/'INDEPENDENT_FINAL_RESULT_AUDIT.json')
    final = read(OUT/'FINAL_INFERENCE_RESULTS.json')
    if complete['completed'] is not True or audit['passed'] is not True or audit['paired_rollouts_audited'] != 48:
        raise ValueError('Scientific completion unproven')
    if final['paired_rollouts'] != 48 or final['decision_contexts'] != 8:
        raise ValueError('Incomplete final scope')
    for name, key in [('FINAL_INFERENCE_RESULTS.json','final_result_sha256'),
                      ('INDEPENDENT_FINAL_RESULT_AUDIT.json','final_audit_sha256')]:
        if sha(OUT/name) != complete[key]:
            raise ValueError('Completion digest mismatch')
    sources = {}
    for path in HERE.rglob('*'):
        rel = path.relative_to(HERE)
        if '__pycache__' in rel.parts or rel.parts[:2] == ('additional_data_v1','groups'):
            continue
        if path.is_symlink():
            raise ValueError('Unexpected evidence symlink')
        if path.is_file():
            sources[path] = HERE.name+'/'+str(rel)
    for path in OUT.iterdir():
        if path.is_file():
            sources[path] = 'confirmation/inference_v4/'+path.name
    raw_hashes = audit['raw_audit_hashes']
    if sum(Path(p).name == 'INDEPENDENT_BRANCH_AUDIT.json' for p in raw_hashes) != 48:
        raise ValueError('Missing branch audit links')
    for value, expected in raw_hashes.items():
        path = Path(value).resolve()
        rel = path.relative_to(OUT)
        if sha(path) != expected:
            raise ValueError('Audit-linked file changed')
        sources[path] = 'confirmation/inference_v4/'+str(rel)
    linked = []
    for receipt_path in sorted(ARCHIVES.glob('*.receipt.json')):
        receipt = read(receipt_path)
        if receipt.get('case') is not None or receipt.get('stage') == 'models':
            linked.append(receipt)
        sources[receipt_path] = 'archive_receipts/'+receipt_path.name
    collection = [r for r in linked if not r.get('stage')]
    confirmation = [r for r in linked if r.get('stage') == 'confirmation-case']
    models = [r for r in linked if r.get('stage') == 'models']
    if len(collection) != 24 or len({r['case'] for r in collection}) != 24 or len(models) != 1:
        raise ValueError('Incomplete collection/model archive inventory')
    if len(confirmation) != 8 or {r['case'] for r in confirmation} != {c['id'] for c in read(OUT/'INFERENCE_PROTOCOL.json')['contexts']}:
        raise ValueError('Incomplete confirmation archive inventory')
    manifest = dict(case=None, stage='final-evidence-overlay', partial_run=False,
                    backup_verification_still_required=True, linked_archives=linked,
                    excluded_raw_data='Additional groups and online raw traces are in linked archives.', files={})
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dest = ARCHIVES/('af_dump_maxf8_final_evidence_'+stamp+'.tar.gz')
    with dest.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for path, name in sorted(sources.items()):
            before = path.stat()
            digest = sha(path)
            info = tarfile.TarInfo(name)
            info.size = before.st_size
            with path.open('rb') as source:
                archive.addfile(info, source)
            after = path.stat()
            if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns) or sha(path) != digest:
                raise ValueError('Source changed during pack')
            manifest['files'][name] = dict(size=info.size,sha256=digest)
        data = json.dumps(manifest,indent=2).encode()
        info = tarfile.TarInfo('ARCHIVE_MANIFEST.json')
        info.size = len(data)
        archive.addfile(info,io.BytesIO(data))
    receipt = dict(archive=str(dest),case=None,stage=manifest['stage'],size=dest.stat().st_size,
                   sha256=sha(dest),files=len(sources),linked_archives=len(linked))
    with dest.with_suffix('.receipt.json').open('x') as stream:
        json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt),flush=True)

if __name__ == '__main__':
    main()
