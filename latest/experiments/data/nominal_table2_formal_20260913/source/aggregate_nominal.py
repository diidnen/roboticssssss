#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); a=ap.parse_args()
root=Path(a.root); rows=[]
for p in sorted((root/"contexts").glob("*/RESULT.json")):
    r=json.loads(p.read_text()); o=r["outcome"]; plan=r["plan"]
    rows.append({"context":plan["id"].replace("__NOMINAL_VLA",""),"root":plan["root"],"task":plan["task"],
        "friction":plan["band"],"actual_mu":plan["mu"],"method":"NOMINAL_VLA",
        "full_task_success":o.get("full_task_success_y"),"lift_success":o.get("lift_success"),
        "dropped":o.get("dropped"),"official_squeeze_N":o.get("mean_measured_bilateral_squeeze"),
        "contact_conditional_squeeze_N":o.get("contact_conditional_squeeze_N"),
        "nonrelease_samples":o.get("nonrelease_samples"),"bilateral_contact_samples":o.get("bilateral_contact_samples"),
        "label_valid":o.get("label_valid"),"result":str(p)})
fields=list(rows[0]) if rows else []
with (root/"NOMINAL_CONTEXT_RESULTS.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
valid=[r for r in rows if r["label_valid"]]
def mean(k):
    x=[float(r[k]) for r in valid if r[k] is not None]; return float(np.mean(x)) if x else None
summary={"planned_contexts":96,"completed_contexts":len(rows),"valid_contexts":len(valid),
    "success_count":sum(int(r["full_task_success"]) for r in valid),"success_percent":100*mean("full_task_success") if valid else None,
    "lift_count":sum(int(r["lift_success"]) for r in valid),"lift_percent":100*mean("lift_success") if valid else None,
    "drop_count":sum(int(r["dropped"]) for r in valid),"drop_percent":100*mean("dropped") if valid else None,
    "commanded_force_mean_N":None,"official_squeeze_mean_N":mean("official_squeeze_N"),
    "contact_conditional_squeeze_mean_N":mean("contact_conditional_squeeze_N")}
(root/"NOMINAL_TABLE2_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,sort_keys=True))
