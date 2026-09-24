#!/usr/bin/env python3
"""Render the final single-root ActiveForcing report from verified audit JSON."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


METHODS = ("Nominal Frozen VLA", "Fixed-Strong 8N", "ActiveForcing")


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def number(value, digits: int = 3) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def ci(values) -> str:
    return f"{pct(values[0])}-{pct(values[1])}"


def paired_label(matrix: dict) -> str:
    return (
        f"{matrix['both_success']} / {matrix['left_only']} / "
        f"{matrix['right_only']} / {matrix['both_fail']}"
    )


def claim_language(core: dict, stage_i: dict) -> list[tuple[str, str, str]]:
    analysis = core["analysis"]
    by_method = analysis["by_method"]
    af = by_method["ActiveForcing"]
    fixed = by_method["Fixed-Strong 8N"]
    nominal = by_method["Nominal Frozen VLA"]
    af_fixed = analysis["paired_AF_vs_Fixed8"]
    af_nominal = analysis["paired_AF_vs_Nominal"]
    saving = analysis["AF_setpoint_saving_when_both_AF_and_Fixed8_succeed"]

    reliability_advantage = (
        af["successes"] > fixed["successes"]
        or af["successes"] > nominal["successes"]
    )
    force_saving_successes = saving["contexts"] > 0 and saving["mean_N"] > 0
    if reliability_advantage:
        useful_status = "SUPPORTED descriptively"
    elif af["successes"] > 0 and force_saving_successes:
        useful_status = "PARTIALLY SUPPORTED; NOT A RELIABILITY ADVANTAGE"
    else:
        useful_status = "NOT DEMONSTRATED"
    useful_basis = (
        f"AF succeeded in {af['successes']}/24 contexts versus {fixed['successes']}/24 for Fixed-8 "
        f"and {nominal['successes']}/24 for Nominal. It had {af_fixed['left_only']} AF-only wins "
        f"against Fixed-8 and {af_nominal['left_only']} against Nominal; in the "
        f"{saving['contexts']} AF/Fixed-8 joint successes it saved {saving['mean_N']:.3f} N of commanded "
        "setpoint on average. Inference is restricted to root 200002."
    )

    if af["successes"] > nominal["successes"]:
        nominal_status = (
            "SUPPORTED by the paired test"
            if af_nominal["exact_mcnemar_two_sided_p"] < 0.05
            else "DIRECTIONAL, NOT STATISTICALLY RESOLVED"
        )
    else:
        nominal_status = "NOT SUPPORTED"
    nominal_basis = (
        f"Paired matrix (both/AF-only/Nominal-only/both-fail) = {paired_label(af_nominal)}; "
        f"exact McNemar p={af_nominal['exact_mcnemar_two_sided_p']:.4g}."
    )

    command_reduction = af["mean_commanded_force_N"] < 8.0
    measured_reduction = af["mean_measured_squeeze_N"] < fixed["mean_measured_squeeze_N"]
    load_status = (
        "SUPPORTED for commanded setpoint and measured mean squeeze"
        if command_reduction and measured_reduction
        else "SUPPORTED for commanded setpoint only"
        if command_reduction
        else "NOT SUPPORTED"
    )
    load_basis = (
        f"AF mean setpoint {af['mean_commanded_force_N']:.3f} N versus 8.000 N; "
        f"mean measured squeeze {af['mean_measured_squeeze_N']:.3f} N versus "
        f"{fixed['mean_measured_squeeze_N']:.3f} N."
    )

    motion = stage_i["motion_dependent_landscape"]
    motion_status = "SUPPORTED as a budget-capped Stage-I population result"
    motion_basis = (
        f"Within-friction feasible-force sets differed in {pct(motion['exact_set_mismatch_rate'])} "
        f"of 24 motion pairs (bootstrap 95% CI {pct(motion['exact_set_mismatch_bootstrap_95pct_CI'][0])}-"
        f"{pct(motion['exact_set_mismatch_bootstrap_95pct_CI'][1])}); this evidence is separate from formal efficacy."
    )
    return [
        ("AF is useful on a distinct task form", useful_status, useful_basis),
        ("Downstream motion context affects force feasibility", motion_status, motion_basis),
        ("AF beats Nominal Frozen VLA", nominal_status, nominal_basis),
        ("AF reduces load versus Fixed-Strong", load_status, load_basis),
        ("Zero-shot transfer to a new task or unseen root", "NOT CLAIMED", "Training and formal evaluation both use root 200002."),
    ]


def render(core: dict, deep: dict, stage_i: dict, incident: dict, extension: dict,
           source_hashes: dict) -> str:
    require(core.get("passed") is True, "core audit did not pass")
    require(deep.get("passed") is True, "deep archive/trace audit did not pass")
    require(core["requirements"]["formal_contexts"] == deep["archive_count"] == 24, "context count mismatch")
    require(core["requirements"]["formal_rollouts"] == deep["rollout_count"] == 72, "rollout count mismatch")
    require(stage_i["contexts"] == 16 and stage_i["robot_rollouts"] == 96, "Stage-I count mismatch")

    analysis = core["analysis"]
    by_method = analysis["by_method"]
    af_fixed = analysis["paired_AF_vs_Fixed8"]
    af_nominal = analysis["paired_AF_vs_Nominal"]
    saving = analysis["AF_setpoint_saving_when_both_AF_and_Fixed8_succeed"]
    deep_summary = deep["summary"]
    stage_methods = {row["method"]: row for row in stage_i["methods"]}
    timestamp = datetime.now(timezone.utc).isoformat()

    if by_method["ActiveForcing"]["successes"] > by_method["Nominal Frozen VLA"]["successes"]:
        if af_nominal["exact_mcnemar_two_sided_p"] < 0.05:
            nominal_sentence = "AF outperformed Nominal in the paired formal test."
        else:
            nominal_sentence = "AF was directionally better than Nominal, but the n=24 paired test did not resolve the difference at 0.05."
    else:
        nominal_sentence = "The formal data do not show AF outperforming Nominal."
    if by_method["ActiveForcing"]["successes"] > by_method["Fixed-Strong 8N"]["successes"]:
        if af_fixed["exact_mcnemar_two_sided_p"] < 0.05:
            fixed_sentence = "AF outperformed Fixed-8 in the paired formal test."
        else:
            fixed_sentence = "AF was directionally better than Fixed-8, but the n=24 paired test did not resolve the difference at 0.05."
    elif by_method["ActiveForcing"]["successes"] == by_method["Fixed-Strong 8N"]["successes"]:
        fixed_sentence = "AF and Fixed-8 have equal aggregate success counts; their paired outcomes are reported and are not called equivalent reliability."
    else:
        fixed_sentence = "AF did not outperform Fixed-8 on full-task success."

    lines = [
        "# ActiveForcing New Task-Form Final Report",
        "",
        f"Generated: {timestamp}  ",
        "Task: `dump_bin_bigbin`  ",
        "Formal population: root 200002, eight held-out pi0 paths, three friction settings  ",
        "Status: **COMPLETE — 24 paired contexts / 72 formal rollouts; all raw archives and traces independently reverified.**",
        "",
        "## Executive result",
        "",
        "| Method | Full-task success | Rate (Wilson 95% CI) | Mean commanded force | Mean measured squeeze |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = by_method[method]
        lines.append(
            f"| {method} | {row['successes']}/24 | {pct(row['success_rate'])} ({ci(row['success_rate_wilson_95'])}) "
            f"| {number(row['mean_commanded_force_N'])} N | {row['mean_measured_squeeze_N']:.3f} N |"
        )
    lines += [
        "",
        nominal_sentence + " " + fixed_sentence,
        "",
        f"AF versus Fixed-8 paired cells (both success / AF only / Fixed only / both fail) were "
        f"**{paired_label(af_fixed)}**; exact two-sided McNemar p={af_fixed['exact_mcnemar_two_sided_p']:.4g}. "
        f"AF versus Nominal cells were **{paired_label(af_nominal)}**; p={af_nominal['exact_mcnemar_two_sided_p']:.4g}.",
        "",
        "This result supports only same-root, held-out-path performance. It is not evidence of unseen-root generalization or zero-shot transfer to an unseen task form.",
        "",
        "## 1. What the original four-task protocol actually was",
        "",
        "The authoritative four-task feasibility corpus was a prospective 648-slot design (647 admitted rows after one infrastructure-unknown quarantine). Its split unit was the complete physical root family: four TRAIN roots (5100, 5101, 5102, 5106), one VAL root (6100), and one TEST root (6103). Every split contained all four tasks and all three friction bands; each task-friction context had nine force branches from 3.00 to 5.00 N.",
        "",
        "The feasibility corpus used scripted downstream motion, not a VLA. Therefore policy seed and VLA-motion holdout are N/A for its train/validation/test split. The later 96-context main evaluation used eight entirely new roots (170040-170047), online frozen pi0 replanning, and matched method siblings with a shared initial action chunk. Actions after the common prefix were generated online from each branch's own observations.",
        "",
        "| Original question | Audited answer |",
        "|---|---|",
        "| Train/validation/test split by physical root? | Yes, complete-root 4/1/1. |",
        "| Split by friction? | No; all frictions occurred in every split. |",
        "| Held-out VLA policy seeds or motions? | No / N/A for feasibility; the corpus was scripted. |",
        "| Main roots separate from feasibility roots? | Yes. |",
        "| Shared branch context? | Matched exposed decision context and common initial online chunk; no claim of byte-identical hidden PhysX solver state. |",
        "",
        "## 2. How the new-task protocol matches—and departs from—the original",
        "",
        "| Dimension | Original paper | New task formal evaluation |",
        "|---|---|---|",
        "| Feasibility data | Root-disjoint 4/1/1, scripted motion | 192 labels on root 200002; frozen without retraining |",
        "| Formal physical roots | Eight fresh roots | One reused root, 200002 |",
        "| Friction coverage | Three bands | .425, .575, .850 |",
        "| Online VLA paths | One matched online context per physical context | Eight prospectively held-out path seeds, 80200002-80200009 |",
        "| Pairing | Same context and first online chunk across methods | Same post-query handoff, reset pi0 seed, identical first chunk across all three siblings |",
        "| Methods | Fixed-3/4/5 and AF | Nominal Frozen VLA, Fixed-Strong 8 N, AF |",
        "| Claim boundary | Cross-root main evaluation | Same-root held-out-path evaluation only |",
        "",
        "The single-root deviation was explicitly requested and frozen before execution. It improves path coverage relative to the prior diagnostics but cannot substitute for a root-disjoint test.",
        "",
        "## 3. Existing versus newly collected data and exact counts",
        "",
        "| Population | Contexts / labels | Rollouts | Included in formal efficacy denominator? |",
        "|---|---:|---:|---|",
        "| Frozen root-local feasibility corpus | 192 labeled branches; 35 deduplicated belief queries | 192 existing | No; training evidence |",
        "| Nominal development smoke | 1 context | 1 new | No; admission only |",
        "| Formal main | 24 contexts | 72 new | **Yes** |",
        "| Stage-I motion-context study after budget cap | 16 contexts | 96 | No; separate Claim A evidence |",
        "| Pre-physics failed smoke attempt | 0 scientific contexts | 0 scientific rollouts | No |",
        "",
        "The formal denominator is exactly 24 contexts and 72 rollouts. The number 96 elsewhere denotes either the original paper's 96 main contexts or Stage-I's 16 contexts x six force policies; neither is mixed into this formal denominator.",
        "",
        "## 4. Formal results by friction",
        "",
        "| Friction | Nominal | Fixed-8 | AF | AF mean setpoint |",
        "|---:|---:|---:|---:|---:|",
    ]
    for friction in ("0.425", "0.575", "0.850"):
        rows = analysis["by_friction"][friction]
        lines.append(
            f"| {friction} | {rows['Nominal Frozen VLA']['successes']}/8 "
            f"| {rows['Fixed-Strong 8N']['successes']}/8 | {rows['ActiveForcing']['successes']}/8 "
            f"| {number(rows['ActiveForcing']['mean_commanded_force_N'])} N |"
        )

    lines += [
        "",
        "Per-seed results and the full 24-context outcome records are preserved in `FINAL_INDEPENDENT_AUDIT.json`; no context was added, removed, or replaced after outcomes were observed.",
        "",
        "## 5. Paired comparison",
        "",
        "| Comparison | Both succeed | Left only | Right only | Both fail | Difference | Exact McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| AF vs Fixed-8 | {af_fixed['both_success']} | {af_fixed['left_only']} | {af_fixed['right_only']} "
        f"| {af_fixed['both_fail']} | {af_fixed['success_difference_percentage_points']:+.1f} pp "
        f"| {af_fixed['exact_mcnemar_two_sided_p']:.4g} |",
        f"| AF vs Nominal | {af_nominal['both_success']} | {af_nominal['left_only']} | {af_nominal['right_only']} "
        f"| {af_nominal['both_fail']} | {af_nominal['success_difference_percentage_points']:+.1f} pp "
        f"| {af_nominal['exact_mcnemar_two_sided_p']:.4g} |",
        "",
        "The exact McNemar tests use only discordant paired contexts and are descriptive inferential checks for this fixed n=24 population. Equal marginal counts, if present, are not interpreted as equivalence.",
        "",
        "## 6. Success-force trade-off",
        "",
        f"AF selected a mean commanded force of **{by_method['ActiveForcing']['mean_commanded_force_N']:.3f} N** "
        f"(distribution: {json.dumps(analysis['AF_force_distribution_N'], sort_keys=True)}), compared with the locked 8.000 N Fixed-Strong setpoint. Nominal has no Newton-valued setpoint and is correctly reported as N/A.",
        "",
        f"Across all paired contexts, AF's mean measured squeeze differed from Fixed-8 by "
        f"{analysis['paired_mean_squeeze_difference_N']['AF_minus_Fixed8']:+.3f} N and from Nominal by "
        f"{analysis['paired_mean_squeeze_difference_N']['AF_minus_Nominal']:+.3f} N. "
        f"Among the {saving['contexts']} contexts where AF and Fixed-8 both succeeded, AF saved a mean "
        f"{number(saving['mean_N'])} N of commanded setpoint.",
        "",
        "Measured squeeze is sensor-derived and not identical to the controller's commanded setpoint; both are reported to avoid conflating command reduction with realized load reduction.",
        "",
        "## 7. Raw-trace intermediate and contact-loss metrics",
        "",
        "These metrics were recomputed after completion from every archived physics trace. They are descriptive, not additional success criteria. Sustained contact loss means at least 50 consecutive logged physics steps without qualifying target contact; a descriptive drop failure is an official task failure with that flag.",
        "",
        "| Method | Ever lifted bin 4 cm | Ever bin z >= 1 m | Ever all 5 garbage in target z-band | Sustained contact loss | Descriptive drop failures | Mean target-contact fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = deep_summary[method]
        lines.append(
            f"| {method} | {row['ever_lifted_4cm']}/24 | {row['ever_bin_at_least_1m']}/24 "
            f"| {row['ever_all_five_garbage_in_success_band']}/24 "
            f"| {row['sustained_contact_loss_50_steps']}/24 | {row['descriptive_drop_failures']}/24 "
            f"| {row['mean_target_contact_fraction']:.3f} |"
        )

    motion = stage_i["motion_dependent_landscape"]
    nonmonotonic = stage_i["nonmonotonic"]
    stage_pair = stage_i["AF_vs_Fixed8"]
    lines += [
        "",
        "## 8. Separate Stage-I motion-context evidence",
        "",
        "Stage-I is not part of the 24-context formal efficacy test. At the user's budget stop, it retained the first 16 contexts (four policy paths x four frictions) and 96 rollouts across AF plus Fixed-1/3/5/6/8. The original precommit was 32 contexts; the mid-stream budget revision is disclosed, so this is supporting rather than pristine preregistered evidence.",
        "",
        f"Among 24 within-friction motion pairs, exact fixed-force feasible sets differed at rate "
        f"**{pct(motion['exact_set_mismatch_rate'])}** (bootstrap 95% CI "
        f"{pct(motion['exact_set_mismatch_bootstrap_95pct_CI'][0])}-{pct(motion['exact_set_mismatch_bootstrap_95pct_CI'][1])}); "
        f"mean Jaccard distance was {motion['mean_jaccard_distance']:.3f}. "
        f"Nonmonotone force landscapes occurred in {nonmonotonic['count']}/16 contexts "
        f"({pct(nonmonotonic['rate'])}, bootstrap 95% CI {pct(nonmonotonic['bootstrap_95pct_CI'][0])}-"
        f"{pct(nonmonotonic['bootstrap_95pct_CI'][1])}).",
        "",
        f"Stage-I AF succeeded in {stage_methods['ActiveForcing']['successes']}/16 contexts versus "
        f"{stage_methods['Fixed-8N']['successes']}/16 for Fixed-8. Their paired cells were "
        f"both success {stage_pair['both_success']}, AF only {stage_pair['AF_only']}, "
        f"Fixed-8 only {stage_pair['Fixed8_only']}, both fail {stage_pair['both_failure']}. "
        "These numbers describe the intervention study and are not pooled with formal-main success.",
        "",
        "## 9. Claim audit",
        "",
        "| Claim | Verdict | Evidence boundary |",
        "|---|---|---|",
    ]
    for claim, verdict, basis in claim_language(core, stage_i):
        lines.append(f"| {claim} | **{verdict}** | {basis} |")

    lines += [
        "",
        "## 10. Nominal baseline validity",
        "",
        "The one-context admission smoke passed. Nominal used the frozen pi0 14D native joint-position output, including native gripper-position commands at indices 6 and 13; it installed no AF selector, no fixed-N override, and no force servo. It began at the same established-grasp/post-query handoff, executed finite native actions, produced nonempty squeeze telemetry, and kept commanded force as N/A. Smoke task success was not required and the observed task result was retained.",
        "",
        "## 11. Engineering incident and repair",
        "",
        f"The first smoke attempt stopped before any query step or policy action because `{incident['cause']}` "
        f"It generated no scientific outcome. The repair bound `AF_INFERENCE_CONTEXT` to the same frozen `SPEC.json` already supplied as `AF_FORMAL_CONTEXT`, then reran the same smoke context without changing its seed. "
        f"The extension records queue source {extension['original_queue_sha256']} -> {extension['resume_queue_sha256']} and explicitly marks model, metrics, context/seed, and scientific protocol as unchanged. The failed attempt remains archived under `engineering_failures/`.",
        "",
        "## 12. Integrity and independent audit",
        "",
        "The execution queue performed a raw per-context audit before archiving and deleting local working copies. A separate final verifier then reconstructed the frozen 24-context denominator, checked the model/runtime/plan locks, validated all 24 compact records and archive receipts, reproduced every aggregate and paired matrix, and confirmed 72 unique trace-content hashes.",
        "",
        "A second verifier ran where the raw tarballs reside on Anvil. In a single pass over every archive, it recomputed each tarball SHA256, hashed every archived member against `ARCHIVE_FILE_MANIFEST.json`, revalidated the archived per-context audits, replayed all 72 compressed physics traces, recomputed squeeze statistics and final success from raw actor states, and confirmed all results.",
        "",
        "| Artifact | SHA256 |",
        "|---|---|",
    ]
    for name, digest in source_hashes.items():
        lines.append(f"| `{name}` | `{digest}` |")
    for name, digest in core["frozen_hashes"].items():
        lines.append(f"| `{name}` | `{digest}` |")

    lines += [
        "",
        "Archive receipts and the deep audit establish 24/24 verified tarballs and 72/72 verified raw traces. The Anvil deep-audit output is itself retained with the experiment and copied to the report directory.",
        "",
        "## 13. Frozen-component confirmation",
        "",
        "No feasibility model was retrained or reselected. The frozen pi0 checkpoint, `models_v4` ensemble, 58D belief representation, phase-free 8x64 feasibility architecture, posterior integration, maxF8 utility `p*(8-F)/8 - (1-p)`, 0.05 N deployment search, success predicate, squeeze controller, arm-action path, and force support were unchanged. Formal outcomes did not control queue membership, force selection rules, retries, seed replacement, or reporting denominator.",
        "",
        "## 14. Limitations",
        "",
        "- All feasibility labels and all formal tests use physical root 200002. The formal paths are held out, but the physical root is not.",
        "- Eight path seeds x three frictions provide 24 paired contexts, not 24 independent physical roots.",
        "- The exact post-query handoff and first action chunk are matched; later online replanning intentionally diverges with each branch's observations.",
        "- Stage-I was budget-capped at 16 of the originally precommitted 32 contexts; its motion-landscape result is kept separate and explicitly qualified.",
        "- Intermediate and contact-loss summaries are post-hoc descriptive metrics. The frozen full-task predicate remains the only success outcome.",
        "- Wilson intervals and exact McNemar p-values quantify sampling uncertainty over the fixed path-friction population; they do not repair the absence of root-level replication.",
        "",
        "## Bottom line",
        "",
        f"The formal run is technically complete and auditable. ActiveForcing achieved {by_method['ActiveForcing']['successes']}/24 full-task successes at a mean selected setpoint of {by_method['ActiveForcing']['mean_commanded_force_N']:.3f} N; Fixed-8 achieved {by_method['Fixed-Strong 8N']['successes']}/24 and Nominal achieved {by_method['Nominal Frozen VLA']['successes']}/24. "
        "AF therefore demonstrates a lower-force operating point in a small set of successes, but the formal data do not support an overall reliability advantage. Interpretation must remain at the same-root held-out-path level. The separate 16-context Stage-I study supports motion-dependent force feasibility, with its budget revision clearly disclosed.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--deep", type=Path, required=True)
    parser.add_argument("--stage-i", type=Path, required=True)
    parser.add_argument("--incident", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    core = read(args.core)
    deep = read(args.deep)
    stage_i = read(args.stage_i)
    incident = read(args.incident)
    extension = read(args.extension)
    source_hashes = {
        args.core.name: sha(args.core),
        args.deep.name: sha(args.deep),
        args.stage_i.name: sha(args.stage_i),
        Path(__file__).name: sha(Path(__file__)),
    }
    args.output.write_text(render(core, deep, stage_i, incident, extension, source_hashes))
    print(args.output)


if __name__ == "__main__":
    main()
