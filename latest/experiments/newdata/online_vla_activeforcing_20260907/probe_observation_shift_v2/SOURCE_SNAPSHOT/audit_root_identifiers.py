"""Read-only exposure inventory using identifiers, never success outcomes.

This does not select roots or authorize physics. Final chosen identifiers must
still undergo a dedicated collision scan before their prospective freeze.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime, timezone

LOCATIONS = [
    Path('/home/exouser/FORTE'),
    Path('/home/exouser/Tabero'),
    Path('/home/exouser/E3_E6_E7_LANES'),
    Path('/media/volume/newdata/exouser/online_vla_activeforcing_20260907'),
    Path('/media/volume/newdata/exouser/ACTIVEFORCING_DISK_ARCHIVE_20260902'),
]
PATTERN = r'"(?:root|root_id|root_seed|seed_idx|seed)"\s*:\s*"?[0-9]+'
PATH_PATTERN = re.compile(r'(?:_r|root|_s)([0-9]{1,7})(?=[_/\.\-]|$)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Retain exposure inventory; use a new version')
    locations = [x for x in LOCATIONS if x.exists()]
    entries = {}
    count = 0
    digest = hashlib.sha256()
    def record(value, path, field):
        # Include ordinary simulator-root and training seeds conservatively;
        # enormous per-request VLA noise seeds are a different identifier type.
        if not 0 <= value <= 2**31-1:
            return
        e = entries.setdefault(value, {'occurrences': 0, 'examples': []})
        e['occurrences'] += 1
        example = {'path': path, 'identifier': field}
        if len(e['examples']) < 4 and example not in e['examples']:
            e['examples'].append(example)
    command = ['rg', '--no-heading', '--with-filename', '--only-matching', '--null',
               '--hidden', '--no-ignore', '-g', '!**/.git/**', '-g', '*.json', '-g', '*.jsonl', PATTERN, *map(str, locations)]
    # Only matching identifier text is emitted; outcome fields are not read by
    # selection logic or retained in this inventory.
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc.stdout:
        path, sep, match = line.rstrip(b'\n').partition(b'\0')
        if not sep:
            raise RuntimeError('Unexpected rg identifier record')
        digest.update(line); count += 1
        text = match.decode()
        value = int(text.rsplit(':', 1)[1].strip().lstrip('"'))
        record(value, path.decode(), text)
    error = proc.stderr.read().decode(); rc = proc.wait()
    if rc not in (0, 1):
        raise RuntimeError('Incomplete content scan: '+error)
    path_command = ['rg', '--files', '--hidden', '--no-ignore', '-g', '!**/.git/**', *map(str, locations)]
    proc = subprocess.Popen(path_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    paths = 0
    for line in proc.stdout:
        name = line.decode().rstrip('\n'); paths += 1
        for match in PATH_PATTERN.finditer(name):
            record(int(match.group(1)), name, 'path:'+match.group(0))
    error = proc.stderr.read().decode(); rc = proc.wait()
    if rc not in (0, 1):
        raise RuntimeError('Incomplete path scan: '+error)
    old_path = Path(__file__).parent/'ROOT_EXPOSURE_AUDIT.json'
    old = json.loads(old_path.read_text())
    for value in old['excluded_root_ids']:
        record(value, str(old_path), 'prior_manifest_and_path_inventory')
    result = {
        'role': 'EXPANDED_EXPOSURE_INVENTORY_NOT_FINAL_ROOT_SELECTION',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scan_locations': list(map(str, locations)),
        'missing_optional_locations': [str(x) for x in LOCATIONS if not x.exists()],
        'content_pattern': PATTERN, 'identifier_records': count, 'paths_scanned': paths,
        'gitignored_experiment_artifacts_included': True,
        'identifier_stream_sha256': digest.hexdigest(),
        'excluded_root_or_seed_ids': sorted(entries),
        'identifier_evidence': {str(k): v for k, v in sorted(entries.items())},
        'known_split_groups': old['known_exposure_groups'],
        'outcomes_used_to_choose_roots': False, 'final_roots_selected': [],
        'limitations': [
            'Conservative union includes model-training seeds as well as simulator roots.',
            'JSON/JSONL numeric fields and file identifiers are scanned; arbitrary binary/CSV-only seed metadata may require dedicated final-candidate inspection.',
            'Native policy request noise seeds are not treated as simulator root IDs.',
            'A fresh-ID collision scan and source/split review remain mandatory before final freeze.',
        ],
    }
    with a.out.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k:result[k] for k in ('identifier_records', 'paths_scanned', 'excluded_root_or_seed_ids', 'final_roots_selected')}))


if __name__ == '__main__':
    main()
