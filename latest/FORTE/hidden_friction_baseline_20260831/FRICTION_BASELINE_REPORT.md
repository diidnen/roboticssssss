# Hidden-Friction Baseline Report

**现在不能诚实回答这七个排名问题：严格 matched rollout 数为 0，因此最稳定方法、Fixed-Max 是否达到类似 success、ActiveForcing 节省的 excess force、相对 FORTE 的 pre-slip 优势、NoProbe 掉点、Tabero-neutral 的自主适应能力，以及距 GT-Physics Oracle 的差距，全部为 `NOT ESTIMABLE`。** 这不是负结果，而是 Phase-0 implementation audit 的停止结论：现有 full-task frontier 使用脚本化 downstream，不是本表要求的 frozen π0 downstream；同时 `π0-Default` 与 `Tabero-Neutral` 当前映射到同一 checkpoint/neutral execution path，所谓 FORTE runner 也由其自身规格明确标为 FORTE-inspired + privileged GT slip。把旧数据填进新表会直接破坏论文的核心 matched claim。

## Technical summary

- Task 已在任何本表结果产生前冻结为 task0 / alphabet soup。选择只使用既有 B5 证据：π0 neutral 在 low/mid/high 的 pick/full-task 均为 1.0，prior frontier 为 5/4/3 N；task5 因 pick rate 0.5/0.7/0.7 被预先排除。
- 六个 untouched TEST root seeds 已冻结为 5174–5179。TRAIN 5100–5105 与 DEV 5106–5107 均禁止进入本表。
- friction 已冻结为 μ={0.2, 0.5, 1.0}；task0 force grid 已冻结为 3.0–5.0 N、0.25 N 间隔、每 cell 5 repeats、ρ=0.8。
- Frontier 计划 810 个 rollouts；8 个方法、18 cases、5 paired repeats 计划 720 个 rollouts。两者均未启动。
- Joint 不进入 main table；当前 held-out mechanism evidence 不支持其作为 selected controller。

## Why execution stopped before Phase 3

The requested causal contrast requires one exact pre-probe state and one exact visual observation per root, then friction-only mutation. Existing prospective task0 snapshots either belong to TRAIN/DEV or were reached by scripted P4-B staging rather than frozen π0. Untouched TEST roots have been preregistered but do not yet have eligible snapshots.

The only existing branch generator with dense continuous force labels calls a scripted Cartesian `downstream_branch`. The current frozen π0 runner exists in B5/P6/P7, but there is no authoritative task0 adapter that starts from the new canonical snapshot and supports every requested force method. This is not a normal scientific failure and must not be retried or silently substituted.

## Method readiness

| Method | Current implementation status | Can enter matched table now? |
|---|---|---|
| π0-Default | Raw 13-D action execution from the force-aware Tabero π0 checkpoint under neutral prompt | No; identical to Tabero-Neutral as currently specified |
| Tabero-Neutral | Authoritative B5 checkpoint/client, no force adverb | Semantics ready; matched snapshot runner missing |
| Tabero-Oracle-Force-Language | `gently/softly/firmly/tightly` are supported by converter and inference rewrite | Semantics ready; privileged mapping and matched runner pending |
| FORTE-Reactive | Frozen B4 simulator surrogate, start 3 N, 3→4→5→6→8 N, privileged GT slip | Only if labeled FORTE-inspired, not official FORTE |
| Fixed-Max | Frozen task0 robust target 5 N | Selector ready; matched runner missing |
| ActiveForcing-NoProbe | Prior [0.30,0.56,0.92], same frozen Direct backend | Selector ready; matched runner missing |
| ActiveForcing-Direct | Frozen task0 Direct ensemble; held-out GT gate failed | Runnable as a frozen evaluated method, not a prevalidated controller |
| GT-Physics Oracle | Exact μ into the same frozen Direct backend | Selector ready; privileged; matched runner missing |

## Data-quality and validation status

Overall readiness: **NOT READY FOR CLAIMS**.

- Population coverage: 0/810 frontier episodes and 0/720 method episodes.
- Matching validation: manifest rule frozen, but snapshot/RGB hashes are missing.
- Metric recomputation: not applicable; no outcome data exist.
- Failure taxonomy: schema created, no episodes to adjudicate.
- Existing scripted frontier rows are deliberately excluded rather than mixed with frozen π0 outcomes.
- Main and frontier CSVs contain explicit `NOT_ESTIMABLE_PHASE0_BLOCKER` status, not zero-valued metrics.

## Required next implementation step

Implement or identify one authoritative post-snapshot task0 runner that:

1. lets frozen π0 reach and serialize a canonical pre-probe state for each root 5174–5179;
2. restores exactly that state and identical RGB for μ=0.2/0.5/1.0;
3. changes only force-selection/control semantics while keeping the frozen π0 downstream policy and evaluator fixed;
4. records direct full-task outcomes and the full force/slip/recovery telemetry schema already frozen here;
5. resolves whether `π0-Default` and `Tabero-Neutral` are intentionally the same row or supplies an already-authoritative distinct π0 implementation without training or modification.

Only after those checks pass should the frozen 810-cell frontier run begin. The required paper claims must remain blank until then.

## Further questions

- Is an explicitly labeled `FORTE-inspired (privileged GT-slip)` surrogate acceptable, or is an official tactile FORTE simulator adapter required?
- Should the scientifically redundant π0-Default/Tabero-Neutral rows be merged, or is there an existing distinct frozen π0 checkpoint/adapter not discoverable in the current workspace?
