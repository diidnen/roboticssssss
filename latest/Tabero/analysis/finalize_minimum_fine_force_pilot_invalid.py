#!/usr/bin/env python3
"""Finalize the single DEV pilot after its first restore-parity failure."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path('/home/exouser/Tabero/analysis/results/minimum_fine_force_pilot_20260829_160500')
ROOT = Path('/home/exouser/Tabero')


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def wjson(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')


def wcsv(name, fields, rows=()):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    wjson('PILOT_ENGINEERING_AUDIT.json', {
        'status': 'PILOT_INVALID',
        'failure_stage': 'first_restore_parity_check_before_scientific_rollout',
        'failure': 'RESTORE_PARITY_FAIL:p5s0c_dev_t1_r06_s5106_low_mu0.262418_F5_R1',
        'parity_rows_attempted': 1,
        'scientific_rollouts_completed': 0,
        'scientific_retry': False,
        'action': 'stopped_immediately_per_protocol',
        'test_touched': False,
        'fresh_e2e': False,
        'training': False,
        'note': 'This is a protocol/engineering invalidity, not a force-failure observation and not evidence about friction-dependent F_FINE_STAR.',
    })
    wcsv('PILOT_ROLLOUT_MANIFEST.csv', ['context_id', 'friction', 'force_N', 'repeat', 'validity', 'success', 'unsafe_event'])
    wcsv('PILOT_FINE_FORCE_RESULTS.csv', ['context_id', 'friction', 'friction_band', 'force_N', 'repeats', 'success_count', 'success_rate', 'unsafe_event_rate', 'stochasticity_flag'])
    wcsv('PILOT_PHYSICAL_TRACE_SUMMARY.csv', ['context_id', 'friction', 'force_N', 'trace_status'])
    wjson('PILOT_FINE_FRONTIER_SUMMARY.json', {
        'status': 'PILOT_INVALID',
        'fine_frontier': 'NOT_IDENTIFIABLE',
        'reason': 'The first common-snapshot restore parity check failed before any scientific rollout.',
        'expected_rollouts': 18,
        'completed_valid_rollouts': 0,
        'frontier_shift': 'NOT_IDENTIFIABLE',
    })
    wjson('PILOT_MONOTONICITY_AUDIT.json', {
        'status': 'NOT_ASSESSABLE',
        'reason': 'No valid scientific rollout occurred after the first restore-parity failure.',
    })
    report = '''# STATUS

STOPPED_AT_PROTOCOL_INVALIDITY — the first restore-parity check failed before any scientific rollout.

# SINGLE SCIENTIFIC QUESTION

Whether changing only hidden friction moves a repeatable continuous minimum sufficient force in one matched task/root family.

# AUTHORITATIVE INPUTS / HASHES

The inherited controller audit and P5-S0-C provenance are in the frozen protocol and `PROVENANCE.json`. This namespace is new and does not overwrite earlier studies.

# MATCHED CONTEXT

Task 1, DEV P5-S0-C root06 standardized pre-probe state family. μ_low=0.262418 with coarse bracket 5–5.5N; μ_high=0.515227 with coarse bracket 4–4.5N. The common state and 0.25N grid were frozen before outcomes.

# CONTINUOUS FORCE GRID

μ_low: 5.00, 5.25, 5.50N. μ_high: 4.00, 4.25, 4.50N. Three repeats per force were frozen; no grid change occurred.

# VALID ROLLOUT COVERAGE

Expected 18 scientific rollouts; completed valid rollouts: 0. The first restore parity check failed at `p5s0c_dev_t1_r06_s5106_low_mu0.262418_F5_R1`.

# STATE-RESTORE PARITY

FAIL. The restored state hash did not match the frozen target hash on the first attempted branch. Per protocol, the pilot stopped immediately. No scientific failure was retried.

# μ_LOW FORCE CURVE

NOT AVAILABLE — no valid rollout.

# μ_HIGH FORCE CURVE

NOT AVAILABLE — no valid rollout.

# MONOTONICITY

NOT ASSESSABLE.

# FINE FORCE FRONTIER

F*_fine(μ_low) = NOT_IDENTIFIABLE. F*_fine(μ_high) = NOT_IDENTIFIABLE.

# FRONTIER SHIFT

ΔF* = NOT_IDENTIFIABLE.

# COARSE VS FINE FRONTIER

Coarse brackets were inherited but no fine-force outcome was collected.

# PHYSICAL TRACE INTERPRETATION

No valid physical trace exists. The failure occurred before force execution.

# PRIMARY_CLASSIFICATION

PILOT_INVALID

# WHAT THIS PILOT PROVES

Only that the selected controller is already known to accept continuous force from the inherited audit and that this new matched replay cannot proceed under the attempted restore-parity implementation.

# WHAT THIS PILOT DOES NOT PROVE

It does not prove or disprove friction-dependent fine-frontier movement. It is not multi-root, multi-task, TEST, probe benefit, continuous world-model validation, or E2E.

# METHOD CHANGE

NONE

# NEW TRAINING

NONE

# NEXT_METHOD

Repair and validate the common-snapshot restore/parity procedure in a separate DEV engineering validation, then rerun this exact frozen single-family pilot without changing its grid or repeat count.
'''
    (OUT / 'FINAL_REPORT.md').write_text(report, encoding='utf-8')
    files = sorted(x for x in OUT.iterdir() if x.is_file() and x.name != 'SHA256SUMS.txt')
    (OUT / 'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files), encoding='utf-8')


if __name__ == '__main__':
    main()
