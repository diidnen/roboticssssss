#!/usr/bin/env python3
"""Finalize an interrupted DEV-only fine-force forensic namespace.

This does not run Isaac, does not add outcomes, and does not infer a fine
frontier.  It records the bounded replay interruption explicitly.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path('/home/exouser/Tabero/analysis/results/dev_fine_force_forensic_20260829_150000')
ROOT = Path('/home/exouser/Tabero')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def write_csv(name: str, fields, rows=()) -> None:
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    telemetry = sorted((OUT / 'task0' / 'telemetry').glob('*.csv'))
    protocol = json.loads((OUT / 'DEV_FINE_FORCE_FORENSIC_PROTOCOL.json').read_text())
    write_json('REPLAY_STATUS.json', {
        'status': 'ENGINEERING_INTERRUPTION_BEFORE_COMPLETION',
        'namespace': str(OUT),
        'observed_outcomes': True,
        'completed_branch_telemetry_files': len(telemetry),
        'completed_branch_rows': 1 if telemetry else 0,
        'formal_fine_frontier_identifiable': False,
        'reason': 'Native Isaac/GelSight render path required approximately 203 seconds for one full branch; bounded DEV replay was stopped before sufficient fixed repeats completed.',
        'camera_disabled_attempt': 'INVALID_CONTROLLER_INITIALIZATION: GelSight camera requires --enable_cameras',
        'scientific_retry': False,
        'test_touched': False,
        'fresh_e2e': False,
        'training': False,
    })
    write_csv('DEV_FINE_FORCE_FRONTIER.csv', ['context_id', 'baseline', 'force_N', 'repeat', 'validity', 'success', 'unsafe_event'])
    write_csv('RESTORE_PARITY.csv', ['context_id', 'baseline', 'force_N', 'repeat', 'parity_pass'])
    write_csv('DEV_FINE_FORCE_FRONTIER_SUMMARY.csv', ['context_id', 'baseline', 'fine_force_N', 'n_valid', 'n_repeats', 'success_frequency', 'unsafe_event_frequency', 'repeat_outcome_disagreement'])
    write_csv('FRICTION_FINE_FRONTIER_PAIRS.csv', ['task', 'context_a', 'context_b', 'fine_frontier_status', 'classification'])
    write_csv('PROBE_VS_STRICT_NOPHYSICS.csv', ['context_id', 'fine_force_N', 'probe_restore_success_frequency', 'strict_preprobe_success_frequency', 'matched_force_curve_comparison_only'])
    write_csv('LEGACY_VS_STRICT_NOPHYSICS.csv', ['metric', 'legacy', 'strict', 'comparison_status'], [{
        'metric': 'replay_status',
        'legacy': 'not replayed in this namespace',
        'strict': 'protocol frozen; no completed strict branch',
        'comparison_status': 'NOT_IDENTIFIABLE',
    }])
    write_csv('OFFGRID_MODEL_DIAGNOSTIC.csv', ['status'], [{'status': 'NOT_RUN_MODEL_FROZEN_AND_NO_FORMAL_FINE_FRONTIER'}])
    write_json('FORCE_QUANTIZATION_MASKING_ANALYSIS.json', {
        'status': 'NOT_IDENTIFIABLE',
        'continuous_force_controller_supported': True,
        'formal_fine_frontier_rule': 'absent from inherited protocol',
        'quantization_masking_rate': None,
        'reason': 'Only one branch completed; fixed three-repeat per-force protocol was not completed, so no F_FINE_STAR or masking rate is declared.',
        'no_test_tuning': True,
    })
    write_json('FINALIZATION_AUDIT.json', {
        'finalized_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': sha(OUT / 'DEV_FINE_FORCE_FORENSIC_PROTOCOL.json'),
        'protocol_context_count': protocol.get('context_count'),
        'partial_telemetry': [str(x.relative_to(ROOT)) for x in telemetry],
        'scientific_classification': 'INSUFFICIENT_VALID_EVIDENCE',
        'action': 'stop_without_inference_or_method_change',
    })
    report = '''# STATUS

STOPPED_AT_EARLIEST_UNSUPPORTED_LINK — continuous-force support is valid, but the bounded DEV replay did not complete enough fixed repeats to identify a fine frontier or strict-baseline action comparison.

# SINGLE SCIENTIFIC GOAL

Test whether hidden friction moves the true minimum force inside the coarse bins and whether a strict pre-probe no-physics baseline changes the active-sensing conclusion.

# AUTHORITATIVE INPUTS / HASHES

The frozen input list and hashes are in `PROVENANCE.json` and `DEV_FINE_FORCE_FORENSIC_PROTOCOL.json`. The namespace is new and does not overwrite prior studies.

# CONTINUOUS FORCE CONTROLLER VALIDITY

PASS. The existing `_make_action` accepts float forces, maps half the requested force to each finger, and does not integerize or command-clip force. Existing half-step branches also exist. A camera-disabled startup was rejected by the validated GelSight environment, so the physical replay used the native camera-enabled path.

# DEV POPULATION

The frozen protocol selected six DEV contexts (task 0 and task 5, r06, LOW/MID/HIGH) with a fixed inclusive 0.25N grid and three repeats per force for both probe-restore and strict-preprobe conditions. This is a bounded diagnostic subset, not a TEST analysis.

# FIFTH TASK STATUS

Task 2 is the intended fifth task, but it is absent from the authoritative P5-S0-C lineage (`task2_used=false`) and has no valid DEV frontier there. It was not fabricated into this replay.

# COARSE FORCE FRONTIER

Inherited unchanged from P5-S0-C. No coarse frontier search or candidate-set change was performed.

# FINE FORCE FRONTIER

The fixed fine grid and repeats were frozen before outcomes. Only one full branch telemetry file completed before the replay was stopped for engineering-time reasons; no formal fine frontier is declared.

# FRICTION → FINE MINIMUM FORCE

NOT IDENTIFIABLE. The inherited protocol defines a deterministic single-branch frontier, not an allowed repeated-outcome aggregation rule. The required fixed three-repeat population was also incomplete.

# QUANTIZATION MASKING

NOT IDENTIFIABLE. Continuous force is natively accepted, but this run cannot estimate within-bin F* motion or a masking rate from one completed branch.

# FRONTIER STOCHASTICITY

Not assessable. No repeat set completed. No monotonicity conclusion was forced.

# STRICT PRE-PROBE BASELINE

The protocol explicitly froze: probe-informed restores the original captured pre-probe snapshot after obtaining probe evidence; strict no-physics restores that same pre-probe snapshot and executes no probe. Because the replay stopped before a complete fixed-repeat set, no strict action-selection result is claimed.

# LEGACY VS STRICT NO-PHYSICS

NOT IDENTIFIABLE. No legacy action metric was overwritten and no completed strict comparison exists.

# PROBE VS STRICT NO-PHYSICS

NOT IDENTIFIABLE as a controller comparison. Partial branch telemetry is preserved, but it is insufficient for paired frontier inference.

# OFF-GRID PHYSICS-GRU DIAGNOSTIC

Not run. No model inference or calibration was added to this forensic.

# PRIMARY_CLASSIFICATION

INSUFFICIENT_VALID_EVIDENCE

# WHAT IS NOW PROVEN

The already-validated low-level controller natively supports arbitrary continuous force commands. The frozen DEV protocol was created before the new replay outcomes. Isaac/A100 execution began and one full branch completed; no TEST, training, calibration, method change, or E2E was run.

# WHAT IS NOT YET PROVEN

This run does not prove whether quantization masks friction-dependent force requirements, whether strict no-physics changes active-sensing value, or whether a fine frontier is monotone. The partial replay cannot support those claims.

# IMPLICATION FOR FINAL METHOD

Do not upgrade to continuous force from this run. The clean next experiment requires a bounded, completed DEV replay with a preregistered repeated-outcome frontier rule and the strict pre-probe baseline.

# METHOD CHANGE

NONE.

# NEW TRAINING

NONE. Physics-GRU, friction estimator, isotonic calibration, evaluator, Pi0, and all trained artifacts remained frozen.

# NEXT_METHOD

Complete the already-frozen DEV-only fine-force replay with a formally preregistered repeated-outcome frontier rule and strict pre-probe matched baseline; do not touch TEST or implement a method upgrade.
'''
    (OUT / 'FINAL_REPORT.md').write_text(report, encoding='utf-8')
    files = sorted(x for x in OUT.iterdir() if x.is_file() and x.name != 'SHA256SUMS.txt')
    (OUT / 'SHA256SUMS.txt').write_text(''.join(f'{sha(x)}  {x.name}\n' for x in files), encoding='utf-8')


if __name__ == '__main__':
    main()
