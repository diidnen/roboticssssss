"""Versioned, CPU-only provenance primitives for the online VLA restoration."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / 'analysis/results'
TABERO = Path('/home/exouser/Tabero')
VTLA = Path('/media/volume/newdata/exouser/tabero/Tabero-VTLA')
CHECKPOINT = Path('/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999')
V5 = BASE / 'current_matched_runtime_collection_v5_20260906'
INSTRUCTIONS = {
    0: 'pick up the alphabet soup and place it in the basket',
    1: 'pick up the cream cheese and place it in the basket',
    5: 'pick up the tomato sauce and place it in the basket',
    6: 'pick up the butter and place it in the basket',
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def array_sha(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def payload_sha(payload):
    rows = {}
    for k, v in payload.items():
        if not isinstance(k, str):
            raise ValueError('Canonical payload requires string keys')
        rows[k] = array_sha(v) if isinstance(v, np.ndarray) else v
    return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()

def clean(x):
    if hasattr(x, 'detach'): return x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [clean(v) for v in x]
    return x

def write(path, value):
    with Path(path).open('x') as f:
        json.dump(clean(value), f, indent=2, sort_keys=True, allow_nan=False)

def read(path): return json.loads(Path(path).read_text())

def checkpoint_inventory():
    # Exactly the historical tree-hash algorithm, plus individual file hashes.
    h = hashlib.sha256(); files = {}
    for p in sorted(CHECKPOINT.rglob('*')):
        if p.is_file():
            name = str(p.relative_to(CHECKPOINT)); digest = sha(p)
            files[name] = {'sha256': digest, 'bytes': p.stat().st_size}
            h.update(name.encode()); h.update(digest.encode())
    return {'checkpoint': str(CHECKPOINT), 'sha256': h.hexdigest(), 'files': files,
            'algorithm': 'sha256(concat(sorted(relative_path_utf8 + file_sha256_hex_utf8)))'}
