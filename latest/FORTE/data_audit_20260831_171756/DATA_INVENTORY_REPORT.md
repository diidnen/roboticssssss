# DATA INVENTORY REPORT

Audit timestamp (UTC): `20260831_171756`

Scope: read-only historical scan of `/home/exouser/FORTE` and `/home/exouser/Tabero`; no training, rollout, collector mutation, or TEST trajectory/context/RGB/outcome access.

Scanned CSV files: **14697**; candidate label-schema files: **189**; readable source files promoted: **174**; sealed-path candidates excluded before read: **106**.

## How much data

- All canonical historical label rows (excluding sealed 5174–5179): **16890**; physical roots **111**; context cells **594**.

- TRAIN-eligible canonical labels: **7584**; physical roots **72**; context cells **196**.


## Method-usable data

- Pi0-Only: roots=1, contexts=3, labels=30, success/failure=30/0.

- ActiveForcing-Direct: roots=6, contexts=72, labels=4904, success/failure=3886/1018.

- ActiveForcing-AFI: roots=6, contexts=72, labels=4909, success/failure=3890/1019.

- Joint: roots=6, contexts=72, labels=4320, success/failure=3450/870.

- Common: roots=6, contexts=72, labels=4320, success/failure=3450/870.


## Data quality

- Root diversity: median labels/root=16.0, max=1624; common Joint bottleneck=6 physical TRAIN roots.

- Boundary: 124 context cells audited; 83 have both outcomes; 73% are monotone after force sorting.

- Probe: indexed 277 probe files/context keys; valid exact 215x46 files=277; TRAIN direct rows with valid probe=6970.

- Collector metadata: 3 prospective collector worker(s) observed; left untouched.


## Decision

**COLLECT MORE DATA**

Increase shared TRAIN physical roots with matched probe + full-task force labels + physical targets until >=200 common roots; preserve held-out DEV and keep 5174–5179 sealed.


TEST status: **TEST NOT OPENED**.
