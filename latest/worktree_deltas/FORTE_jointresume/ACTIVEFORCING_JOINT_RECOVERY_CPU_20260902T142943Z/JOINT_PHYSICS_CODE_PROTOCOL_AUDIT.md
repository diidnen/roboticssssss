# Joint friction × mass code and protocol audit

Audit ID: `JOINT_CPU_RECOVERY_20260902T142943Z`  
Lane: `/home/exouser/FORTE_jointresume` (`activeforcing-recovery-joint-20260902`)  
Assessment: **NEEDS REVISION; JOINT EXECUTION BLOCKED**  
Evidence cutoff: `2026-09-02T14:29:43Z`

This is a read-only audit of the prototype Joint and Mass sources in the
`activeforcing-mass` worktree and the authoritative/recovery records. No
simulator, GPU process, model training, or joint experiment was run. There are
no joint scientific results in this package.

## Controlling protocol and present dependency

`ACTIVEFORCING_FULL_CLAIM_V2_UTILITY` requires Mass and joint
friction×mass closure, expected-utility force selection, root-disjoint
TRAIN/DEV/locked-TEST discipline, and one-axis ablations. The historical
neural model called "Joint" is explicitly rejected and is unrelated to E8b
joint physics.

The live Mass source is not a released dependency. Its accepted formal
artifact contains 10/18 complete contexts and 100/180 complete branches, all
TRAIN. Coverage is balanced within every accepted context (five forces × two
repeats), there are no duplicate `(context, force, repeat)` keys, and all ten
accepted queries are marked valid. Missing work is 20 TRAIN branches (root
8103 MID/HIGH) plus all 60 TEST branches. The root8103-MID in-flight material
is quarantined by the recovery coordinator and is not counted.

Mass completion means more than reaching 180 rows. Before Joint may use the
axis, Mass must pass exact coverage and provenance QA, freeze the valid mass
coordinate/query feature contract and belief, use the authoritative V2
expected-utility selector, complete fresh mass E2E, and publish a valid Mass
final report/table. Existence of a file is not a completion gate.

## Prototype Joint collector audit

Audited source:
`/home/exouser/FORTE_mass/collect_joint_physics_pilot.py`  
SHA-256: `6d161bade9ed7a38b3863cb71795196b26f81eb29d80b28625e9c5cc3f91dca4`

| Severity | Finding | Evidence and impact | Required correction |
|---|---|---|---|
| Critical | No safe resume contract | The script forces `P4_RESUME=0`, initializes episode rows from empty, appends timestep rows, and rewrites episode rows. Reusing an output after interruption can duplicate timesteps while discarding earlier episode rows. | Resume only from a validated immutable manifest. Commit one complete cell atomically, quarantine partial cells, and skip only keys whose episode and timestep checks both pass. |
| High | The recorded initial reset is not the query reset | The wrapper resets the environment and `run_probe_episode` immediately resets it again. No hash proves that physics cells start from the same realized state. | Have exactly one owner of reset. Hash the actual post-reset/pre-query state and verify equality across all nine cells of a root. |
| High | Physics mutation is not certified | Friction is read back, but target mass and inertia are not read back or range-checked. The state record does not include material, mass, inertia, COM, or geometry invariants. | Record requested and applied friction/mass, full mass/inertia readback, ratios, tolerances, and invariant geometry/COM checks before accepting a cell. |
| High | Split name and size cannot support a locked claim | Defaults are two TRAIN roots and one root named TEST. One independent held-out root is nine correlated cells, not `n=9` independent evidence, and the locked-test freeze fields are absent. | Use root-disjoint TRAIN/DEV/LOCKED_TEST manifests; tune on TRAIN grouped CV, gate once on DEV, and keep TEST sealed until all contracts are frozen. |
| High | Missing V2 freeze/provenance fields | The protocol omits `protocol_version`, `final_force_selector`, utility/Direct/belief hashes, candidate generator, force bounds, posterior samples, fallback, TEST-access state, and evidence role. | Add all authoritative V2 fields and fail closed on missing or legacy selector fields. |
| Medium | Cell order is drift-confounded | Every root runs LOW→MID→HIGH mass and low→high friction in a fixed order. Thermal/runtime drift is aliased with the factorial coordinates. | Freeze a deterministic balanced cell order per root before collection and preserve it on resume. |
| Medium | Query timestep schema is incomplete | Timestep rows have friction but no explicit mass, split, root family, or cell key beyond a parseable trial string. | Put the full composite key and applied-physics readbacks on every timestep row. |
| Medium | Output and source locations are hard-coded | The P4 module and Tabero root point at mutable sibling worktrees and only the P4 source hash is eventually stored. | Resolve all sources through a frozen input manifest and hash collector, P4, environment config, object profile, and dependency commits. |
| Low | Failure cleanup is weak | `os._exit` bypasses normal cleanup and the simulator app handle is not explicitly closed. | Use `finally` cleanup and leave an atomic error/status record without accepting an incomplete cell. |

The mass/inertia scaling idea is physically plausible—mass and inertia are
scaled by the same ratio—but its implementation is not yet an evidentiary
contract without readback and reset-parity checks.

## Prototype Joint analyzer audit

Audited source:
`/home/exouser/FORTE_mass/analyze_joint_physics_pilot.py`  
SHA-256: `ed60a1aa75ca04f321a922b9e84d8b30c937d67313122004944e33db571bd5a4`

| Severity | Finding | Evidence and impact | Required correction |
|---|---|---|---|
| Critical | The reported "joint" model is not a joint comparison | Its friction prediction is identical to `friction-only`; its mass prediction is identical to `mass-only`. It merely places the two independent Ridge outputs next to each other. | Define the joint target explicitly, evaluate exact 3×3 cell recovery, covariance/calibration, and nuisance robustness against axis-only models trained on the nominal nuisance slice. |
| Critical | Force-choice "truth" is invented | `quantize(1 + 2*friction + 5*mass, ...)` is an analytic proxy with no Direct probabilities, task outcomes, or authoritative Utility. Agreement with it is not joint decision accuracy. | Delete it from evidentiary analysis. Decision accuracy is unavailable until frozen Direct/Utility inputs and matched physical outcomes exist. |
| High | Invalid values silently become physical zeros | Missing, malformed, and non-finite features are converted to `0.0`, which can create a false identifying signal and conceal schema failure. | Fail on required invalid values; report missingness by feature/cell/root and allow an explicitly frozen TRAIN-only imputation rule only if preregistered. |
| High | No data-integrity gate | There are no schema, duplicate-key, 3×3 coverage, root-disjointness, physics readback, query-validity, leakage, or train-only preprocessing checks. | Run these checks before fitting and emit no metrics if any hard gate fails. |
| High | Effective held-out sample size is overstated | Nine cells from one TEST root share a root context. Accuracy is clustered at one independent root, so an 80% cell gate is not reliable evidence. | Report root-cluster counts, per-root metrics, and root bootstrap intervals; never treat cells/repeats as independent roots. |
| High | The analyzer overwrites collection provenance | When input and output directories coincide, it rewrites `JOINT_PHYSICS_PILOT_PROTOCOL.json` with a smaller analysis record. | Analysis outputs must have distinct names and an immutable pointer/hash to the collection protocol. |
| Medium | Aggregate feature contract is incomplete | It omits several recorded query dynamics while the successful Mass development gate names `all_history` as its best feature group. The mismatch is neither frozen nor justified. | Freeze a legal query-only feature allowlist after Mass release; evaluate summary and history candidates only on TRAIN grouped CV. |
| Medium | Differential query failure is hidden | Invalid queries are silently removed. If validity differs by cell, conditional accuracy can be badly biased. | Make validity/coverage by root and 3×3 cell a primary gate and separately report unconditional query reach. |
| Medium | No cross-confusion or uncertainty analysis | The code reports two MAEs and band accuracies only. | Add axis confusion matrices, exact joint-cell accuracy, cross-error correlations, nuisance-sensitivity diagnostics, root-level uncertainty, and calibration/posterior checks. |
| Medium | Fixed Ridge penalty is not selected by grouped TRAIN data | A single hard-coded penalty is used with 18 rows and 14 features. | Select from a frozen small grid using root-grouped TRAIN CV; do not use DEV or TEST for preprocessing or selection. |

## Mass code hazards that block release

The current formal Mass collector is syntactically valid and preserves
sibling force branches from a post-query state. Its present accepted CSV has
the correct 10 branches per accepted context. It nevertheless remains
incomplete and its initial-state hash is captured before the P4 helper performs
a second reset, so it does not prove actual query-start parity.

Two downstream CPU scripts cannot certify Mass closure as written:

1. `analyze_mass_formal.py` selects force with empirical success rate minus
   `0.002 * force`, not the authoritative normalized expected-utility equation.
   It also fits a three-feature Ridge rather than consuming a frozen Mass belief
   and Direct contract.
2. `assemble_mass_final.py` treats protocol-file existence as completion and
   unconditionally writes `MASS_EXTENSION_READY_FOR_PAPER`. It does not verify
   protocol status, exact 18/180 coverage, fresh-E2E status, hashes, or Joint
   status. It must not be used as a release gate in its current form.

## Allowed interpretation

- Existing Mass P4-B development evidence supports **query observability at
  fixed friction 0.5**; it does not establish a released Mass controller or
  joint identifiability.
- The current formal Mass rows are recoverable partial TRAIN data after QA;
  they are not a formal result and contain no accepted TEST rows.
- Joint collection, joint estimator performance, joint decision accuracy,
  interaction effects, SR, force, and Utility are all **NOT RUN / UNAVAILABLE**.

The companion split/resume plan and CPU analyzer skeleton implement these
fail-closed distinctions.
