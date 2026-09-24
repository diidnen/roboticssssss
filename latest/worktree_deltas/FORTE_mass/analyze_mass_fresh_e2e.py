#!/usr/bin/env python3
"""Summarize fresh paired mass E2E rows."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.input/"MASS_FRESH_E2E_ROWS.csv").open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    policies=["Default","Fixed-Max","NoQuery-Prior","ActiveForcing-Mass","GT-Mass + Direct"];out=[]
    def mean(rs,k): return float(np.mean([float(r.get(k,0)) for r in rs])) if rs else float("nan")
    for p in policies:
        rs=[r for r in rows if r["policy"]==p];out.append({"policy":p,"n":len(rs),"query_state_reach":mean(rs,"query_state_reach"),"query_validity":mean(rs,"query_validity"),"conditional_success":mean(rs,"conditional_success"),"e2e_success":mean(rs,"e2e_success"),"mean_force_N":mean(rs,"selected_force_N"),"under_force":mean(rs,"under_force") if rs and "under_force" in rs[0] else "NA","excess_force":mean(rs,"excess_force") if rs and "excess_force" in rs[0] else "NA","query_duration_s":mean(rs,"query_duration_s"),"delayed_failure":float(np.mean([int(float(r.get("lift_success",0))==1 and float(r.get("full_task_success_y",0))==0) for r in rs])) if rs else float("nan")})
    with (a.out/"TABLE_MASS_FRESH_E2E.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    with (a.out/"TABLE_MASS_FRESH_E2E.md").open("w",encoding="utf-8") as f:
        f.write("# Mass fresh E2E\n\nPaired fresh-root reset → P4-B query → policy selection → structured completion.\n\n| Policy | n | Query reach | Query valid | Conditional SR | E2E SR | Mean force | Query duration (s) | Delayed failure |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in out:f.write(f"| {r['policy']} | {r['n']} | {r['query_state_reach']:.3f} | {r['query_validity']:.3f} | {r['conditional_success']:.3f} | {r['e2e_success']:.3f} | {r['mean_force_N']:.3f} | {r['query_duration_s']:.3f} | {r['delayed_failure']:.3f} |\n")
    (a.out/"MASS_FRESH_E2E_REPORT.md").write_text("# Mass fresh E2E report\n\nAll policies use paired fresh roots and the same fixed mass/friction contexts. ActiveForcing-Mass fits its query-only identifier on formal TRAIN contexts; GT-Mass is diagnostic upper bound using the true mass band for force selection.\n\n"+json.dumps({"status":"COMPLETED","rows":len(rows),"summary":out},indent=2)+"\n")
if __name__=="__main__":main()
