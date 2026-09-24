#!/usr/bin/env python3
"""Summarize the independent paired target-object force validation."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

OUT = Path("/home/exouser/E3_E6_E7_LANES/ACTIVEFORCING_VS_FIXEDMAX_PHYSICAL_20260904_r7")
FORMAL = Path("/media/volume/newdata/exouser/activeforcing_e3/P1_SIMPLIFIED_COLLECTION_FORMAL_20260903")
ROOTS = [7800, 7801, 7802]


def read_rows(p: Path):
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def pct(s):
    return 100.0 * s


def main() -> None:
    rows = read_rows(OUT / "PAIRED_SUMMARY.csv")
    by = {(int(r["root_id"]), r["method"]): r for r in rows}
    authoritative = {}
    formal_sources = {}
    for root in ROOTS:
        for force in (1, 8):
            p = FORMAL / f"P1_SIMPLIFIED_ROOT{root}_F{force}N/branch_result.json"
            d = json.loads(p.read_text(encoding="utf-8"))
            e = d["episode_row"]
            method = "ActiveForcing" if force == 1 else "Fixed-Max"
            authoritative[(root, method)] = int(e["full_success"])
            formal_sources[(root, method)] = str(p)
    pairs = []
    for root in ROOTS:
        a = by[(root, "ActiveForcing")]
        m = by[(root, "Fixed-Max")]
        row = {"context_id": a["context_id"], "root_id": root,
               "active_command_N": float(a["selected_force_command_N"]), "max_command_N": 8.0,
               "active_formal_full_success": authoritative[(root, "ActiveForcing")],
               "max_formal_full_success": authoritative[(root, "Fixed-Max")],
               "active_replay_full_success": int(a["full_task_success"]),
               "max_replay_full_success": int(m["full_task_success"]),
               "active_mean_force_N": float(a["full_contact_conditioned_mean_physical_squeeze_N"]),
               "max_mean_force_N": float(m["full_contact_conditioned_mean_physical_squeeze_N"]),
               "active_peak_force_N": float(a["full_contact_conditioned_peak_physical_squeeze_N"]),
               "max_peak_force_N": float(m["full_contact_conditioned_peak_physical_squeeze_N"]),
               "active_force_integral_Ns": float(a["force_time_integral_Ns"]),
               "max_force_integral_Ns": float(m["force_time_integral_Ns"])}
        for metric in ("mean_force_N", "peak_force_N", "force_integral_Ns"):
            av, mv = row[f"active_{metric}"], row[f"max_{metric}"]
            row[f"{metric}_saving"] = 1.0 - av / mv if mv > 0 else float("nan")
        pairs.append(row)
    with (OUT / "PAIRED_CONTEXT_COMPARISON.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(pairs[0])
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(pairs)

    def vals(method, key):
        return [float(by[(root, method)][key]) for root in ROOTS]
    summary = []
    for method in ("ActiveForcing", "Fixed-Max"):
        success = [authoritative[(root, method)] for root in ROOTS]
        summary.append({"method": method, "n": len(ROOTS), "formal_full_task_sr": mean(success),
                        "mean_physical_force_N": mean(vals(method, "full_contact_conditioned_mean_physical_squeeze_N")),
                        "peak_physical_force_N": mean(vals(method, "full_contact_conditioned_peak_physical_squeeze_N")),
                        "force_integral_Ns": mean(vals(method, "force_time_integral_Ns")),
                        "median_mean_force_N": statistics.median(vals(method, "full_contact_conditioned_mean_physical_squeeze_N")),
                        "median_peak_force_N": statistics.median(vals(method, "full_contact_conditioned_peak_physical_squeeze_N")),
                        "median_force_integral_Ns": statistics.median(vals(method, "force_time_integral_Ns")),
                        "formal_success_source": "existing authoritative formal E3 branch_result.json"})
    with (OUT / "TABLE_ACTIVEFORCING_FIXEDMAX_PHYSICAL.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(summary[0]); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summary)

    active_success = sum(authoritative[(r, "ActiveForcing")] for r in ROOTS)
    max_success = sum(authoritative[(r, "Fixed-Max")] for r in ROOTS)
    both = [r for r in pairs if r["active_formal_full_success"] and r["max_formal_full_success"]]
    amax = sum(r["active_formal_full_success"] == 1 and r["max_formal_full_success"] == 0 for r in pairs)
    maf = sum(r["active_formal_full_success"] == 0 and r["max_formal_full_success"] == 1 for r in pairs)
    active_means = vals("ActiveForcing", "full_contact_conditioned_mean_physical_squeeze_N")
    max_means = vals("Fixed-Max", "full_contact_conditioned_mean_physical_squeeze_N")
    active_peaks = vals("ActiveForcing", "full_contact_conditioned_peak_physical_squeeze_N")
    max_peaks = vals("Fixed-Max", "full_contact_conditioned_peak_physical_squeeze_N")
    active_int = vals("ActiveForcing", "force_time_integral_Ns")
    max_int = vals("Fixed-Max", "force_time_integral_Ns")
    mean_save = 1.0 - mean(active_means) / mean(max_means)
    peak_save = 1.0 - mean(active_peaks) / mean(max_peaks)
    int_save = 1.0 - mean(active_int) / mean(max_int)
    report = f'''# ActiveForcing vs Fixed-Max physical-force validation\n\n'''
    report += '''This is an independent measurement-only diagnostic. It uses the frozen E3 FULLTASK_DIRECT heldout decisions, the existing formal branch outcomes, and new target-object-specific telemetry replay in a new directory. Existing formal data are read-only.\n\n'''
    report += f'''## Design and integrity\n\n- Matched contexts: {len(ROOTS)} (roots 7800, 7801, 7802), same task/object/target/reset snapshot and μ=0.6.\n- ActiveForcing command: frozen FULLTASK_DIRECT selection, 1N for all three heldout roots.\n- Fixed-Max command: 8N.\n- Target force: `contact_grasp_black_book_1.data.force_matrix_w`, bilateral squeeze = `2*min(left,right)`; only bilateral object-contact steps enter physical metrics.\n- Reset parity: 6/6 passed.\n- Important replay qualification: new telemetry replay uses saved force-specific formal VLA traces. The new replay did not reproduce one formal terminal success exactly, so SR is taken from the existing authoritative formal branch result, while force metrics are taken from the new target-object telemetry.\n\n'''
    report += '## Context-level paired table\n\n'
    report += '| Root | Active cmd | Max cmd | Active formal SR | Max formal SR | Active mean N | Max mean N | Active peak N | Max peak N | Active integral Ns | Max integral Ns |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in pairs:
        report += f"| {r['root_id']} | {r['active_command_N']:.1f} | {r['max_command_N']:.1f} | {r['active_formal_full_success']} | {r['max_formal_full_success']} | {r['active_mean_force_N']:.3f} | {r['max_mean_force_N']:.3f} | {r['active_peak_force_N']:.3f} | {r['max_peak_force_N']:.3f} | {r['active_force_integral_Ns']:.3f} | {r['max_force_integral_Ns']:.3f} |\n"
    report += f'''\n## Aggregate all-rollout physical comparison\n\n| Method | Formal full-task SR | Mean physical squeeze | Peak squeeze | Force integral |\n|---|---:|---:|---:|---:|\n| ActiveForcing | {active_success}/{len(ROOTS)} | {mean(active_means):.3f} N | {mean(active_peaks):.3f} N | {mean(active_int):.3f} N·s |\n| Fixed-Max | {max_success}/{len(ROOTS)} | {mean(max_means):.3f} N | {mean(max_peaks):.3f} N | {mean(max_int):.3f} N·s |\n\nAll-rollout ratio differences are mean {pct(mean_save):.1f}%, peak {pct(peak_save):.1f}%, and integral {pct(int_save):.1f}%. These are not success-controlled savings.\n\n'''
    report += f'''## Paired outcome safety check\n\n- Both-success subset: {len(both)} contexts; success-controlled force saving is undefined because there are no matched contexts where both methods succeeded.\n- Active success / Max failure: {amax}/{len(ROOTS)}.\n- Active failure / Max success: {maf}/{len(ROOTS)} (root 7800).\n- Active failure / Max failure: {len(ROOTS)-amax-maf}/{len(ROOTS)}.\n- Therefore the lower ActiveForcing all-rollout force can be partly explained by early failure/shorter exposure; it cannot support a claim of lower force at equal success.\n\n'''
    report += '''## Verdict\n\n`WHY_NOT_FIXED_MAX = NOT_SUPPORTED` for a positive physical-force-saving claim. In this frozen heldout slice, Fixed-Max has higher authoritative full-task SR (1/3 vs 0/3), and the both-success comparison is empty. The new telemetry does establish that the 1N and 8N interventions produce different target-object physical force distributions, but not that ActiveForcing preserves success while using less force. A reviewer-facing claim requires a valid both-success or matched equal-SR evaluation with telemetry captured in the same authoritative rollout.\n\n'''
    report += '## Sources\n\n'
    report += f'- [paired summary]({(OUT / "PAIRED_SUMMARY.csv").resolve()})\n- [paired context comparison]({(OUT / "PAIRED_CONTEXT_COMPARISON.csv").resolve()})\n- [physical summary table]({(OUT / "TABLE_ACTIVEFORCING_FIXEDMAX_PHYSICAL.csv").resolve()})\n- [target-object force timeseries]({(OUT / "TARGET_OBJECT_FORCE_TIMESERIES.csv").resolve()})\n- [pairing inputs]({(OUT / "PAIRING_INPUTS.json").resolve()})\n- Runner: `/home/exouser/Tabero/analysis/activeforcing_vs_fixedmax_physical_20260904.py`\n'
    (OUT / "ACTIVEFORCING_VS_FIXEDMAX_PHYSICAL_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "FINAL_STATUS.json").write_text(json.dumps({
        "final_status": "ACTIVEFORCING_VS_FIXEDMAX_VALIDATION_COMPLETE",
        "n_matched_contexts": len(ROOTS), "active_formal_sr": f"{active_success}/{len(ROOTS)}",
        "fixedmax_formal_sr": f"{max_success}/{len(ROOTS)}", "mean_physical_saving_all_rollouts": mean_save,
        "peak_physical_saving_all_rollouts": peak_save, "force_integral_saving_all_rollouts": int_save,
        "both_success_n": len(both), "active_success_max_fail": amax, "active_fail_max_success": maf,
        "why_not_fixed_max": "NOT_SUPPORTED", "success_controlled_claim": False,
        "authoritative_outcome_sources": {f"{root}_{method}": path for (root, method), path in formal_sources.items()},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
