#!/usr/bin/env python3
"""Plan N=20 (or N=15) expansions for transition cells only. No Isaac."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
MUS = (0.2, 0.5, 1.0)


def load_all():
    rows = []
    seen = set()
    for name in (
        "TASK1_ORACLE_SCAN.csv",
        "TASK7_ORACLE_SCAN.csv",
        "ALL_TASK_CHEAP_SCAN.csv",
        "POSITIVE_TASK_EXPANDED.csv",
    ):
        p = OUT / name
        if p.exists() and p.stat().st_size:
            with p.open() as f:
                for row in csv.DictReader(f):
                    tid = row.get("trial_id")
                    if tid and tid in seen:
                        continue
                    if tid:
                        seen.add(tid)
                    rows.append(row)
    return rows


def cells(rows, task):
    d = defaultdict(list)
    for r in rows:
        if int(float(r["task_id"])) != int(task):
            continue
        key = (float(r["friction"]), float(r["force"]))
        d[key].append(int(float(r["full_task_success"])))
    return {k: (sum(v) / len(v), len(v)) for k, v in d.items()}


def need_expand(sr, n, min_n):
    if n >= min_n:
        return False
    if n == 0:
        return False
    # skip obvious all-fail / all-succeed once n>=3
    if n >= 3 and (sr == 0.0 or sr == 1.0):
        return False
    return True


def transition_forces(cell, mus, forces):
    """Forces near a 0.8 crossing, or mid-SR cells."""
    out = set()
    for mu in mus:
        srs = []
        for F in forces:
            sr, n = cell.get((mu, F), (None, 0))
            srs.append((F, sr, n))
        for F, sr, n in srs:
            if sr is None:
                continue
            if 0.2 < sr < 0.95:
                out.add((mu, F))
        # crossing around tau=0.8
        for (F0, sr0, n0), (F1, sr1, n1) in zip(srs, srs[1:]):
            if sr0 is None or sr1 is None:
                continue
            if (sr0 < 0.8 <= sr1) or (sr1 < 0.8 <= sr0):
                out.add((mu, F0))
                out.add((mu, F1))
    return out


def csv_for_task(tid):
    if tid == 1:
        return "TASK1_ORACLE_SCAN.csv"
    if tid == 7:
        return "TASK7_ORACLE_SCAN.csv"
    return "POSITIVE_TASK_EXPANDED.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positive-only", action="store_true")
    args = ap.parse_args()
    rows = load_all()
    summary_p = OUT / "ANALYSIS_SUMMARY.json"
    summary = json.loads(summary_p.read_text()) if summary_p.exists() else {}
    if args.positive_only:
        tasks = [int(t) for t in summary.get("positive", [])]
        min_n = 15
        seed_end = 15
        cmd_path = OUT / "logs/expand_positive_cmds.sh"
        default_forces = [4.0, 5.0, 6.0, 8.0]
    else:
        tasks = [1, 7]
        min_n = 20
        seed_end = 20
        cmd_path = OUT / "logs/expand_cmds.sh"
        default_forces = [3.0, 4.0, 5.0, 6.0, 8.0]

    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"B2={OUT}", f"PY=/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python"]
    n_jobs = 0
    for tid in tasks:
        cell = cells(rows, tid)
        forces = sorted({F for (mu, F) in cell}) or default_forces
        targets = transition_forces(cell, MUS, forces)
        # also expand F* cells with n<min_n unless all-fail/all-succeed
        for mu in MUS:
            for F in forces:
                sr, n = cell.get((mu, F), (None, 0))
                if sr is None:
                    continue
                if need_expand(sr, n, min_n) and (mu, F) in targets:
                    pass
        expand_pairs = []
        for mu, F in sorted(targets):
            sr, n = cell.get((mu, F), (None, 0))
            if sr is None:
                continue
            if not need_expand(sr, n, min_n):
                continue
            expand_pairs.append((mu, F, n, sr))
        print(f"task {tid} expand cells: {expand_pairs}")
        if not expand_pairs:
            continue
        # collector runs a rectangular grid; run remaining seeds for the union of forces/mus
        mus = sorted({p[0] for p in expand_pairs})
        fs = sorted({p[1] for p in expand_pairs})
        seed0 = 5 if not args.positive_only else 3
        nseeds = seed_end - seed0
        csv_name = csv_for_task(tid)
        mus_s = ",".join(str(x) for x in mus)
        fs_s = ",".join(str(int(x) if float(x).is_integer() else x) for x in fs)
        log = f"b2_expand_t{tid}.log"
        lines.append(
            f'echo "expand task={tid} F={fs_s} mu={mus_s} seed0={seed0} n={nseeds}"'
        )
        lines.append(
            "B2_OUT=\"$B2\" "
            f"B2_TASK_ID={tid} B2_CSV={csv_name} B2_PHASE=EXPAND "
            f"B2_N_SEEDS={nseeds} B2_SEED0={seed0} B2_FORCES={fs_s} B2_MUS={mus_s} "
            f"B2_LOG_NAME={log} OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y "
            "\"$PY\" -u \"$B2/scripts/b2_collect.py\""
        )
        n_jobs += 1
    cmd_path.write_text("\n".join(lines) + "\n")
    cmd_path.chmod(0o755)
    if n_jobs == 0:
        # empty script still valid
        cmd_path.write_text("#!/usr/bin/env bash\necho no_expand_needed\n")
        print("no_expand_needed")
    else:
        print(f"wrote {cmd_path} jobs={n_jobs}")


if __name__ == "__main__":
    main()
