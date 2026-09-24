#!/usr/bin/env python3
"""Recompute the Table-II measured pre-release drop diagnostic for Nominal VLA."""
import csv, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT=Path("/media/volume/data/exouser/nominal_table2_formal_20260913")
SOURCE=Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_vla_v1/SOURCE_SNAPSHOT")
FORTE=Path("/home/exouser/FORTE")
sys.path[:0]=[str(SOURCE),str(FORTE),str(FORTE/"analysis/results/current_runtime_recovery_v2_20260905")]
import drop_diagnostics

manifest=json.loads((FORTE/"analysis/results/current_runtime_recovery_v2_20260905/TASK_GEOMETRY_MANIFEST.json").read_text())
geom=drop_diagnostics.old.label_module.geom
rows=[]
for result_path in sorted((ROOT/"contexts").glob("*/RESULT.json")):
    result=json.loads(result_path.read_text()); plan=result["plan"]
    trace=[json.loads(x) for x in (result_path.parent/"ACTION_TRACE.jsonl").read_text().splitlines()]
    # The first executed observation is the closest immutable natural-reset
    # baseline available; the object is still resting before pi0 establishes a grasp.
    z0=float(trace[0]["object_position_m"][2])
    proof=drop_diagnostics.causal_drop_proof(trace,z0)
    info=next(x for x in manifest["tasks"] if x["task"]==plan["task"])
    with np.load(info["label_geometry"]) as f:
        vertices=f["vertices"]; region=geom.Region(f["basket_from_site"],f["half_size"])
    checks=[]
    if proof.get("proven"):
        for frame in proof["proof_frames"]:
            t=trace[frame["step"]-1]
            g=geom.containment(region,geom.transform(t["basket_pose_w"]),geom.transform(t["object_pose_w"]),vertices)
            lo=np.asarray(g["mesh_min_site"])[:2]; hi=np.asarray(g["mesh_max_site"])[:2]
            checks.append(bool(np.any((hi < -region.half_size[:2]) | (lo > region.half_size[:2]))))
    measured=bool(proof.get("proven") and checks and all(checks))
    native=bool(result["outcome"].get("dropped"))
    rows.append({"context":plan["id"].replace("__NOMINAL_VLA",""),"root":plan["root"],"task":plan["task"],
        "friction":plan["band"],"native_drop":int(native),"measured_pre_release_drop":int(measured),
        "table2_drop":int(native or measured),"proof_event_step":proof.get("event_step"),
        "z0_source":"first immutable executed observation after natural-reset settling"})

with (ROOT/"NOMINAL_DROP_AUDIT.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={"contexts":len(rows),"native_drop_count":sum(r["native_drop"] for r in rows),
    "measured_pre_release_drop_count":sum(r["measured_pre_release_drop"] for r in rows),
    "table2_drop_count":sum(r["table2_drop"] for r in rows),
    "table2_drop_percent":100*np.mean([r["table2_drop"] for r in rows]),
    "contract":"ONLINE_OBSERVABLE_PRE_RELEASE_CONTACT_LOSS_V2 plus whole-mesh XY target exclusion",
    "z0_reconstruction_limitation":"Natural-reset z0 was not separately serialized; first executed observation is used."}
(ROOT/"NOMINAL_DROP_AUDIT.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,sort_keys=True))
