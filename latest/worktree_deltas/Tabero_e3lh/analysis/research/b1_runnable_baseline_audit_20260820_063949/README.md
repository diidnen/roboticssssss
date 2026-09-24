# B1 — Runnable Baseline & Codebase Landscape Audit

**Status:** `B1_RUNNABLE_BASELINE_AUDIT_COMPLETE`  
**Date:** 2026-08-20  
**Mode:** READ-ONLY literature/codebase audit  
**Isolation:** D2 not touched. No Isaac launch. No GPU experiment. No Tabero source modification.

This audit answers one question:

> Which methods actually have official, inspectable code we can compare against on Tabero — and what alternative each one tests?

---

## Compact physics-decision table

| Method | Code? | What it knows | When it gets physics info | How it chooses force | Can compare now? | Role |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed Low | internal | nothing | never | one low constant | YES | B / must |
| Fixed Robust | internal | nothing | never | one high constant that covers worst μ | YES | B / must |
| Tabero-VTLA Neutral | FULL | language + tactile/force slots | during task execution (closed-loop VLA) | learned 6D force slots; force intent mainly from language | YES (local ckpt + tasks 1,7) | A / must |
| Tabero-VTLA Gentle/Firm | FULL | language adverb + tactile | language given *before* contact | learned force slots conditioned on gently/firmly | YES as reference, not fair latent-physics baseline | A / reference |
| FORTE original | FULL + hardware | air-channel pressure → force + slip | after contact | close until contact, then position increment on slip | NO on this machine | C / hardware-blocked |
| FORTE-inspired | FAITHFUL PORT of controller logic | slip/force from Tabero sensors | after contact | reactive close / raise F after slip | YES after port | C / must |
| DeliGrasp | PARTIAL (method exists; README thin; MAGPIE/UR5) | LLM semantic prior m, μ, k | before contact; optional slip loop after | analytic F ≈ mg/μ, then `deligrasp` slip correction | YES as PORT, not direct | D / must |
| Exp-Force | DATA_ONLY / no method repo | RGB + retrieved past F* | one-shot pre-contact | VLM in-context F* | NO | D / discuss-only |
| PLUME | NO_OFFICIAL_CODE | latent belief over physics params | online during trajectory | world-model planning, not grasp-F* | NO | E / discuss-only |
| RETAF / TF-Gripper | HARDWARE_ONLY; method “Coming Soon” | wrist RGB + tactile | after contact, 30–80 Hz | decoupled high-freq force adapter | NO | C-like / discuss-only |
| Oracle Physics | internal upper bound | GT μ / sim physics | before action (privileged) | F*_full(μ) from force grid | YES after oracle scan | F / must |

**Who tells it physics, when, and how it decides force** is the only axis that matters for our paper table.

---

## Direct answers

1. **Downstream benchmark:** Tabero LIBERO-object official 9-task subset (`task_id` 0,1,2,3,5,6,7,8,9). Local machine currently has HDF5 for **task 1 and task 7 only**.
2. **Fair, code-qualified comparators now:** Fixed Low, Fixed Robust, Tabero-VTLA Neutral, FORTE-inspired (port), DeliGrasp-style (port), Oracle Physics.
3. **Do not put in the quantitative matrix:** Exp-Force, PLUME, RETAF, Tactile-VLA, TaF-VLA, ForceVLA, ManipForce, FoAR, VLA-Touch (wrong task or no method code).
4. **Do not claim novelty** for tactile force control, slip-reactive grasp, semantic physical priors, or language-conditioned force slots. Remaining space is a **hypothesis**, not a claim.

---

## Artifact index

| File | Contents |
| --- | --- |
| [`RUNNABLE_BASELINE_TABLE.md`](RUNNABLE_BASELINE_TABLE.md) | Full column table |
| [`RUNNABLE_BASELINE_TABLE.csv`](RUNNABLE_BASELINE_TABLE.csv) | Same, machine-readable |
| [`CODE_AVAILABILITY_AUDIT.md`](CODE_AVAILABILITY_AUDIT.md) | Repo-tree evidence, DeliGrasp/Tabero-VTLA special audits |
| [`BASELINE_SHORTLIST.md`](BASELINE_SHORTLIST.md) | MUST / SHOULD / DISCUSS ONLY |
| [`NOVELTY_THREATS.md`](NOVELTY_THREATS.md) | Occupied vs remaining (hypothesis) |
| [`TABERO_DOWNSTREAM_TASKS.md`](TABERO_DOWNSTREAM_TASKS.md) | Official 9-task list + local data |
| [`PROPOSED_BENCHMARK_MATRIX.md`](PROPOSED_BENCHMARK_MATRIX.md) | Planned, not executed |
| [`repo_commits.json`](repo_commits.json) | Inspected commits |
| [`sources.md`](sources.md) | URLs |
| [`repositories/`](repositories/) | Shallow clones for inspect only |

---

## Isolation record

```text
D2 touched: NO
Isaac launched: NO
GPU experiment launched: NO
Tabero source modified: NO
analysis/results/d2_* written: NO
```

D2 result directory exists and was only listed, never written:

```text
analysis/results/d2_hierarchical_force_decision_20260820_054605/
```
