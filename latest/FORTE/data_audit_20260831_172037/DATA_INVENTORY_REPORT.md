# DATA INVENTORY REPORT

Audit timestamp (UTC): `20260831_172037`

Scope: read-only historical scan of `/home/exouser/FORTE` and `/home/exouser/Tabero`; no training, rollout, collector mutation, or TEST trajectory/context/RGB/outcome access.

Scanned CSV files: **14727**; candidate label-schema files: **189**; readable source files promoted: **174**; sealed-path candidates excluded before read: **112**.

## How much data

- All canonical historical label rows (excluding sealed 5174–5179): **6602**; physical roots **111**; context cells **594**.

- TRAIN-eligible canonical labels: **3239**; physical roots **72**; context cells **196**.


## Method-usable data

- Pi0-Only: roots=1, contexts=3, labels=30, success/failure=30/0.

- ActiveForcing-Direct: roots=6, contexts=72, labels=1005, success/failure=789/216.

- ActiveForcing-AFI: roots=6, contexts=72, labels=1009, success/failure=792/217.

- Joint: roots=6, contexts=72, labels=720, success/failure=575/145.

- Common: roots=6, contexts=72, labels=720, success/failure=575/145.


## Data quality

- Root diversity: median labels/root=10.0, max=507; common Joint bottleneck=24 stored root IDs (6 underlying root seeds).

- Boundary: 124 context cells audited; 83 have both outcomes; 79% are monotone after force sorting.

- Probe: indexed 277 probe files/context keys; valid exact 215x46 files=277; TRAIN direct rows with valid probe=2765.

- Collector metadata: 2 prospective collector worker(s) observed; left untouched.


## Decision

**COLLECT MORE DATA**

Increase shared TRAIN root IDs with matched probe + full-task force labels + physical targets from 24 to at least 200 (and increase the underlying independent root seeds from 6); preserve held-out DEV and keep 5174–5179 sealed.


TEST status: **TEST NOT OPENED**.
