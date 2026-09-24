#!/usr/bin/env python3
"""Write and verify the final deterministic checksum manifest."""
from __future__ import annotations

import hashlib
from pathlib import Path


OUT = Path("/home/exouser/FORTE/joint_mechanism_20260831")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest = OUT / "SHA256SUMS.txt"
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p != manifest)
    lines = [f"{digest(path)}  {path.relative_to(OUT).as_posix()}" for path in files]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    failures = [str(path) for path in files if digest(path) != lines[files.index(path)].split()[0]]
    if failures:
        raise RuntimeError(f"checksum self-verification failed: {failures}")
    print(f"wrote {manifest} with {len(lines)} entries")


if __name__ == "__main__":
    main()
