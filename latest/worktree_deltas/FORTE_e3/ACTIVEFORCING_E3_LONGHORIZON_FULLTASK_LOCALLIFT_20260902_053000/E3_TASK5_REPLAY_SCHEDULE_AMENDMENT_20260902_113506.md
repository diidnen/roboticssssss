# Task5 replay schedule amendment — 2026-09-02 11:35:06 UTC

## Status

`PRE_OUTCOME_OPERATIONAL_AMENDMENT_SINGLE_DEMO_FALLBACK_ACTIVE`

This amendment was written while external demo2 PID `687494` was still running and before its output or outcome was inspected.

The pre-outcome readiness document `TASK5_PI0_ONBOARDING_REAL_REPLAY_READINESS.md` (SHA-256 `92911e9774d385acca7443f1107d4cd0d4663a796737d5a4f42ec1e7d289241a`) explicitly preregistered five isolated single-demo jobs as the fallback if batch collation became operationally risky. That fallback is now active.

## Operational trigger

The authoritative remaining-four batch launcher first exited fail-closed before creating an output root because a protected E5 job appeared between coordinator GO and the launcher's final resource gate. Later, core GPU scheduler PID `679695` externally started the exact isolated demo2 launcher PID `687488` and replay PID `687494`.

This is an operational scheduling change only. It does not change task identity, TRAIN demo membership `[1,2,11,12,19]`, remaining order `[2,11,12,19]`, source trajectories, simulator, actions, success evaluator, recorder schema, tactile streams, π0 target, Utility, or Fmax. The canonical batch plan remains preserved as pre-amendment history.

## Active schedule and acceptance

Demo2 may count only if its natural output independently passes exact source ID/initial-state identity, isolated recorder code and explicit frozen ffmpeg provenance, semantic success, finite `[T,13]` actions, real force, real marker motion, four fully decoded and frame-aligned media streams, and an empty failure record.

Future IDs `11,12,19` must run one at a time in that frozen order. Each requires a fresh GPU/process/duplicate audit and explicit coordinator GO; no automatic successor is authorized. Failure of any cell stops the data gate. No TEST or ActiveForcing outcome supervision may enter the replay or acceptance decision.
