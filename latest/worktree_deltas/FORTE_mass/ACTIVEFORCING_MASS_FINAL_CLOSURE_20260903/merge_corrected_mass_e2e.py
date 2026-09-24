#!/usr/bin/env python3
"""Merge the six independent corrected same-context Mass replays.

This does not alter the original sealed fresh-E2E bundle.  It refuses to
produce a merged bundle unless every requested root/band has exactly five
methods and five decision records.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXPECTED = {(8400, "LOW"), (8400, "MID"), (8400, "HIGH"),
            (8401, "LOW"), (8401, "MID"), (8401, "HIGH")}
METHODS = {"FROZEN_PI0_NATIVE_DEFAULT", "FIXED_MAX", "TRUE_NOQUERY_PRIOR",
           "ACTIVEFORCING_MASS", "GT_MASS"}


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def json_value(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, data):
    fields = []
    for row in data:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        out = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        out.writeheader(); out.writerows(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).parent)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.root
    out = args.out or root / "fresh_e2e_corrected_final"
    all_rows, all_decisions, all_queries, manifests, superseded = [], [], [], [], []
    seen = set()
    for rid, band in sorted(EXPECTED):
        candidates = sorted(root.glob(f"fresh_e2e_corrected_final_root{rid}_{band.lower()}*"))
        valid = []
        for d in candidates:
            rp, dp = d / "MASS_FRESH_E2E_ROWS.csv", d / "MASS_FRESH_E2E_DECISIONS.csv"
            if not (rp.is_file() and dp.is_file()):
                continue
            rr, dd = rows(rp), rows(dp)
            method_set = {r.get("method", "") for r in rr}
            tuple_set = {r.get("context_id", "") for r in rr}
            if len(rr) == 5 and method_set == METHODS and len(tuple_set) == 1 and len(dd) == 5:
                valid.append((d, rr, dd))
        if not valid:
            raise SystemExit(f"expected at least one complete replay for {rid}/{band}, found none: {[str(x[0]) for x in candidates]}")
        # If an interrupted/watchdog attempt later completed while its shell
        # was detached, retain it on disk but choose the newest explicit
        # version as canonical and record the other complete copy.
        valid.sort(key=lambda x: x[0].name)
        d, rr, dd = valid[-1]
        for old, _, _ in valid[:-1]:
            superseded.append({"root": rid, "band": band, "source": str(old), "reason": "duplicate_complete_replay_superseded_by_newest"})
        key = (rid, band)
        if key in seen:
            raise SystemExit(f"duplicate replay {key}")
        seen.add(key); all_rows.extend(rr); all_decisions.extend(dd)
        q_candidates = sorted(root.glob(f"fresh_e2e_corrected_query_final_root{rid}_{band.lower()}*/MASS_FRESH_E2E_QUERIES.json"))
        if q_candidates:
            if len(q_candidates) != 1:
                raise SystemExit(f"expected exactly one persisted query replay for {rid}/{band}, found {len(q_candidates)}")
            qq = json_value(q_candidates[0])
        else:
            # The query-only validation may run all six contexts in one Isaac
            # process.  Select the newest complete six-record bundle and
            # filter its record for this root/band.
            bundles = []
            for qp in sorted(root.glob("fresh_e2e_corrected_query_final_all_*/MASS_FRESH_E2E_QUERIES.json")):
                try:
                    qdata = json_value(qp)
                except Exception:
                    continue
                if len(qdata) == len(EXPECTED):
                    bundles.append((qp, qdata))
            if not bundles:
                raise SystemExit(f"no complete six-context persisted query bundle for {rid}/{band}")
            qq = [x for x in bundles[-1][1] if int(x.get("root_seed")) == rid and str(x.get("tuple_id", "")).lower().endswith("_" + band.lower())]
        if len(qq) != 1:
            raise SystemExit(f"expected one query record for {rid}/{band}, found {len(qq)}")
        all_queries.extend(qq)
        protocol = json.loads((d / "MASS_FRESH_E2E_PROTOCOL.json").read_text(encoding="utf-8")) if (d / "MASS_FRESH_E2E_PROTOCOL.json").is_file() else {}
        manifests.append({"root": rid, "band": band, "source": str(d), "rows": len(rr), "decisions": len(dd), "protocol_status": protocol.get("status")})
    if seen != EXPECTED:
        raise SystemExit(f"missing contexts: {sorted(EXPECTED - seen)}")
    out.mkdir(parents=True, exist_ok=True)
    write(out / "MASS_FRESH_E2E_ROWS.csv", all_rows)
    write(out / "MASS_FRESH_E2E_DECISIONS.csv", all_decisions)
    (out / "MASS_FRESH_E2E_QUERIES.json").write_text(json.dumps(all_queries, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (out / "CORRECTED_REPLAY_MANIFEST.json").write_text(json.dumps({"status": "PASS", "contexts": manifests, "superseded_complete_duplicates": superseded, "rows": len(all_rows), "decisions": len(all_decisions), "queries": len(all_queries), "scientific_method_change": "NONE"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(all_rows), "decisions": len(all_decisions), "out": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
