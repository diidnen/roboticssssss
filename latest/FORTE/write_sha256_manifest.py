#!/usr/bin/env python3
"""Write a deterministic top-level SHA256SUMS.txt for one result directory."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    args = ap.parse_args()
    root = Path(args.directory).resolve()
    files = sorted(p for p in root.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    (root / "SHA256SUMS.txt").write_text(
        "".join(f"{digest(p)}  {p.name}\n" for p in files), encoding="utf-8"
    )
    print(f"wrote {len(files)} entries to {root / 'SHA256SUMS.txt'}")


if __name__ == "__main__":
    main()
