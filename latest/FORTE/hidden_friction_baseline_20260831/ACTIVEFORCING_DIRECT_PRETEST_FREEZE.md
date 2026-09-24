# ActiveForcing-Direct Pre-TEST Method Freeze

Status: **FROZEN BEFORE FORMAL TEST**  
Amendment status: `RESOLVED_BY_PRETEST_USER_FREEZE`

Before any formal TEST rollout, the proposed method is frozen as
`ActiveForcing-Direct`.

The proposed method is a GNP-like full-task prospective feasibility selector:

```text
probe/context
→ full-task feasibility backend(context, candidate force)
→ minimum candidate force with p_success >= 0.5
→ restore exact original root
→ fresh frozen π0 full-task execution with Newton force override
```

The online Direct feasibility threshold is `p_success >= 0.5`, matching the
executable feasibility backend. The candidate grid is:

```text
3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00 N
```

`rho_frontier = 0.8` is retained only as the empirical evaluation criterion:
the empirical frontier is the minimum force cell with at least 4 successes out
of 5 frozen-π0 full-task repeats.

Historical AFI `eta = 0.9` is retained only as the
`ActiveForcing-AFI (physics-imagination ablation)` and is not the proposed
method. AFI is not used by the Direct selector.

This is a method-definition freeze before TEST is opened. It is not a change
selected from TEST outcomes. The prior semantic blocker
`ACTIVEFORCING_DECISION_SEMANTICS_AMBIGUOUS` is therefore marked
`RESOLVED_BY_PRETEST_USER_FREEZE`; the prior artifact is retained unchanged.

## Non-negotiable separation

| Quantity | Meaning | Role |
|---|---|---|
| `p_success threshold = 0.5` | Direct online backend probability threshold | Method decision |
| `rho_frontier = 0.8` | `>=4/5` empirical full-task success criterion | Evaluation only |
| `AFI eta = 0.9` | Historical physics-imagination threshold | Ablation only |

No formal TEST result may alter the Direct threshold, backend, context,
candidate grid, fallback, normalization, or force-to-π0 interface.
