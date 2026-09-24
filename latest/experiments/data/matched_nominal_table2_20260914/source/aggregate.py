#!/usr/bin/env python3
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path("/media/volume/data/exouser/matched_nominal_table2_20260914")
rows=[]
for p in sorted((ROOT/"contexts").glob("*/BRANCH_RESULT.json")):
    x=json.loads(p.read_text());o=x["outcome"];q=x["plan"]
    rows.append({"context":q["id"],"root":q["root"],"task":q["task"],"friction":q["band"],"actual_mu":q["mu"],
        "success":o.get("full_task_success_y"),"lift":o.get("lift_success"),"native_drop":o.get("dropped"),
        "official_squeeze_N":o.get("mean_measured_bilateral_squeeze"),"contact_squeeze_N":o.get("contact_conditional_squeeze_N"),
        "nonrelease_samples":o.get("nonrelease_samples"),"contact_samples":o.get("bilateral_contact_samples"),"result":str(p)})
if rows:
    with (ROOT/"MATCHED_NATIVE_ROWS.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean(k):
    v=[float(r[k]) for r in rows if r[k] is not None];return float(np.mean(v)) if v else None
s={"planned_contexts":96,"completed_contexts":len(rows),"success_count":sum(int(r["success"]) for r in rows),
   "success_percent":100*mean("success") if rows else None,"lift_count":sum(int(r["lift"]) for r in rows),
   "lift_percent":100*mean("lift") if rows else None,"native_drop_count":sum(int(r["native_drop"]) for r in rows),
   "commanded_force_N":None,"official_squeeze_N":mean("official_squeeze_N"),"contact_squeeze_N":mean("contact_squeeze_N"),
   "contact_squeeze_defined_contexts":sum(r["contact_squeeze_N"] is not None for r in rows)}
(ROOT/"MATCHED_NATIVE_SUMMARY.json").write_text(json.dumps(s,indent=2)+"\n");print(json.dumps(s,sort_keys=True))
