# DATA INVENTORY REPORT

Audit timestamp (UTC): `20260831_171655`

Scope: read-only historical scan of `/home/exouser/FORTE` and `/home/exouser/Tabero`; no training, rollout, collector mutation, or TEST trajectory/context/RGB/outcome access.

Scanned CSV files: **14686**; candidate label-schema files: **189**; readable source files promoted: **174**; sealed-path candidates excluded before read: **103**.

## How much data

- All canonical historical label rows (excluding sealed 5174–5179): **16890**; physical roots **199**; context cells **783**.

- TRAIN-eligible canonical labels: **7584**; physical roots **129**; context cells **367**.


## Method-usable data

- Pi0-Only: roots=10, contexts=30, labels=30, success/failure=30/0.

- ActiveForcing-Direct: roots=30, contexts=144, labels=4904, success/failure=3886/1018.

- ActiveForcing-AFI: roots=30, contexts=144, labels=4909, success/failure=3890/1019.

- Joint: roots=30, contexts=144, labels=4320, success/failure=3450/870.

- Common: roots=30, contexts=144, labels=4320, success/failure=3450/870.


## Data quality

- Root diversity: median labels/root=32.0, max=514; common Joint bottleneck=30 physical TRAIN roots.

- Boundary: 268 context cells audited; 151 have both outcomes; 80% are monotone after force sorting.

- Probe: indexed 277 probe files/context keys; valid exact 215x46 files=277; TRAIN direct rows with valid probe=6970.

- Collector metadata: 3 prospective collector worker(s) observed; left untouched.


## Decision

**COLLECT MORE DATA**

Increase shared TRAIN physical roots with matched probe + full-task force labels + physical targets until >=200 common roots; preserve held-out DEV and keep 5174–5179 sealed.


TEST status: **TEST NOT OPENED**.
