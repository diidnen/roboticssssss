# DEV Context Rebuild Equivalence

Status: **PASS**

The formal TEST context is generated online from raw P4-B telemetry, the frozen probe feature builder/estimator, and an eight-row branch_hold query skeleton derived from the post-probe EEF command.

- raw probe rows: `206`
- online μ̂: `0.76031923`; σ: `0.20929322`
- Direct input shape: `[8, 71]`
- restore max abs diff: `0`
- historical raw-probe rebuild is recorded separately in the CPU audit artifact.
- TEST roots accessed: `none`
