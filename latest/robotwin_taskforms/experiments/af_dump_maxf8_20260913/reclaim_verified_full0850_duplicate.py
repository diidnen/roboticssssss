"""Reclaim one exact completed, independently verified archive duplicate only."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/media/volume/newdata/exouser/af_dump_maxf8_archives_20260913')
NAME = 'af_dump_maxf8_train_mu0.850_ps40200002_full_group_20260913T045436Z.tar.gz'
EXPECTED = 'af3bd439101151a5569e5a703c480629a0ad66d7fe90e3fd497a42485eec2072'
SIZE = 573832125

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main(apply):
    proof_path = HERE/'whole_queue_slot0_previous_audit.json'
    proof = json.loads(proof_path.read_text())
    if proof['passed'] is not True or len(proof['archives']) != 1:
        raise ValueError('Incomplete destination proof')
    row = proof['archives'][0]
    if (row['archive'], row['sha256'], row['bytes'], row['files_verified'], row['passed']) != (NAME, EXPECTED, SIZE, 715, True):
        raise ValueError('Destination identity mismatch')
    if proof['verifier_sha256'] != '5c433b905c17b1c100b5c8585601bcc0ba720ae3c3ff06e277b40470852b3de3':
        raise ValueError('Unexpected verifier')
    target = ROOT/NAME
    if target.is_symlink() or target.resolve().parent != ROOT.resolve():
        raise ValueError('Unsafe target')
    if target.stat().st_size != SIZE or sha(target) != EXPECTED:
        raise ValueError('Source identity mismatch')
    source_receipt = json.loads(target.with_suffix('.receipt.json').read_text())
    if source_receipt['sha256'] != EXPECTED or source_receipt['size'] != SIZE:
        raise ValueError('Source receipt mismatch')
    receipt = HERE/'FULL0850_DUPLICATE_RECLAIM_01.json'
    if receipt.exists():
        raise ValueError('Already recorded')
    report = dict(validated=True, applied=apply, exact_target=str(target),
                  reclaim_bytes=SIZE, source_sha256=EXPECTED,
                  destination_proof_sha256=sha(proof_path),
                  raw_data_and_anvil_copy_retained=True,
                  utc=datetime.now(timezone.utc).isoformat())
    if apply:
        target.unlink()
        with receipt.open('x') as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    main(parser.parse_args().apply)
