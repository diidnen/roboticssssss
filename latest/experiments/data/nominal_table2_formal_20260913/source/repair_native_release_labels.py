#!/usr/bin/env python3
"""Recompute completed Nominal outcomes with native semantic-open handling."""
import json
from pathlib import Path

ROOT=Path("/media/volume/data/exouser/nominal_table2_formal_20260913")
HORIZON=350

def native(outcome, trace):
    opened=len(trace)>=20 and all(r["vla_release_intent"] for r in trace[-20:])
    unheld=bool(outcome.get("unheld_last20")); inside=bool(outcome.get("inside_last50"))
    support=bool(outcome.get("final_support_contact")); lift=bool(outcome.get("lift_success"))
    dropped=bool(outcome.get("dropped")); timeout=any(r["terminations"].get("time_out",False) for r in trace)
    reset=any(r["physical_reset_requested"] for r in trace); complete=len(trace)==HORIZON
    flags={"NO_LIFT":not lift,"DROP":dropped,"TIMEOUT":timeout,"RESET":reset,
        "INCOMPLETE_HORIZON":not complete,"OUTSIDE_AUTHORED_REGION":not inside,
        "NOT_RELEASED":not(opened and unheld),"NO_FINAL_SUPPORT_CONTACT":not support}
    outcome.update(label_version="NOMINAL_NATIVE_OPEN_INTENT_0P039_WITH_FROZEN_PHYSICAL_TASK_CRITERIA_V1",
        full_task_success_y=int(not any(flags.values())),place_success=int(inside and opened and unheld and support),
        failure_reasons=[k for k,v in flags.items() if v],opened_last20=opened,
        native_release_semantics="raw gripper command >=0.039; no canonical 0.04 rewrite",
        af_exact_0p04_controller_invariant_not_applied=True)

count=changed=0
for p in sorted((ROOT/"contexts").glob("*/RESULT.json")):
    result=json.loads(p.read_text()); trace=[json.loads(x) for x in (p.parent/"ACTION_TRACE.jsonl").read_text().splitlines()]
    old=result["outcome"].get("full_task_success_y"); native(result["outcome"],trace)
    result["native_release_label_recomputed_from_immutable_trace"]=True
    if result["outcome"].get("full_task_success_y")!=old: changed+=1
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); count+=1
receipt={"recomputed_results":count,"changed_success_labels":changed,
    "change":"removed AF-only exact float32(0.04) output invariant; retained semantic threshold and all physical criteria"}
(ROOT/"NATIVE_RELEASE_LABEL_REPAIR_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,sort_keys=True))
