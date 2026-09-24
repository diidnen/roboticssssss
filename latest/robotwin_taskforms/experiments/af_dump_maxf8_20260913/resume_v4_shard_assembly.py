"""Resume the verified v1 temporary archive using os.link for older Python."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from assemble_v4_shards import sha


def main(directory, expected):
    directory = directory.resolve()
    manifest_path = directory / 'MANIFEST.json'
    if sha(manifest_path) != expected:
        raise ValueError('Manifest differs from pinned source')
    manifest = json.loads(manifest_path.read_text())
    parts = manifest['parts']
    for index, row in enumerate(parts):
        path = directory / row['name']
        if row['name'] != 'part_%04d.bin' % index or path.is_symlink() or path.resolve().parent != directory:
            raise ValueError('Unsafe part')
        if path.stat().st_size != row['size'] or sha(path) != row['sha256']:
            raise ValueError('Incomplete or corrupt part')
    receipt = manifest['source_receipt']
    destination = directory.parent / Path(receipt['archive']).name
    receipt_path = destination.with_suffix('.receipt.json')
    temporary = directory / (destination.name + '.assembling')
    record_path = directory / 'ASSEMBLED.json'
    if any(p.exists() or p.is_symlink() for p in [destination, receipt_path, record_path]):
        raise ValueError('Never replace a destination, receipt or assembly record')
    if temporary.is_symlink() or temporary.resolve().parent != directory:
        raise ValueError('Unsafe temporary path')
    if sum(row['size'] for row in parts) != receipt['size']:
        raise ValueError('Invalid total part size')
    if temporary.stat().st_size != receipt['size'] or sha(temporary) != receipt['sha256']:
        raise ValueError('Retained temporary differs from complete original archive')
    # Atomic link creation fails if destination appeared; no overwrite is allowed.
    os.link(str(temporary), str(destination))
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, indent=2)
    result = {'assembled': True, 'archive': str(destination), 'sha256': receipt['sha256'],
              'bytes': receipt['size'], 'parts_verified': len(parts),
              'source_manifest_sha256': expected, 'resume_source_sha256': sha(Path(__file__)),
              'reused_verified_v1_temporary': True,
              'v1_engineering_failure': 'Path.hardlink_to unavailable on Anvil Python',
              'independent_archive_audit_still_required': True,
              'utc': datetime.now(timezone.utc).isoformat()}
    with record_path.open('x') as stream:
        json.dump(result, stream, indent=2)
    temporary.unlink()  # Remove only the duplicate hard link; destination retains bytes.
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('expected_manifest')
    args = parser.parse_args()
    main(args.directory, args.expected_manifest)
