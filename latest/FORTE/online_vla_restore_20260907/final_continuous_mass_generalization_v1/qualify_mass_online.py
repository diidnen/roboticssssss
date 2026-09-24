#!/usr/bin/env python3
"""Outcome-blind structural/current-runtime qualification for MASS online VLA."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "development_online"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")


def jsonlines(path):
    with Path(path).open() as stream: return [json.loads(line) for line in stream if line.strip()]


def main():
    target = HERE / "MASS_ONLINE_DEVELOPMENT_QUALIFICATION.json"
    if target.exists(): raise FileExistsError("development qualification already decided")
    plan = read(HERE / "MASS_DEVELOPMENT_PLAN.json")
    manifest = read(HERE / "MASS_DEVELOPMENT_RUNTIME_MANIFEST.json")
    for path, digest in manifest["source_hashes"].items():
        if sha(path) != digest: raise RuntimeError("development source changed: " + path)
    for path, digest in manifest["artifact_hashes"].items():
        if sha(path) != digest: raise RuntimeError("development artifact changed: " + path)
    rows = []; violations = []; af_forces = []
    for index, context in enumerate(plan["contexts"]):
        reference = OUT / "references" / context["id"]
        if not (reference / "WORKER_COMPLETION.json").is_file() or not read(reference / "WORKER_COMPLETION.json").get("logical_success"):
            violations.append([context["id"], "REFERENCE_INVALID"]); continue
        reference_posterior = read(reference / "PREACTION_POSTERIOR.json")
        for method in plan["methods"]:
            job = OUT / "branches" / f"{context['id']}__{method}"
            if not (job / "WORKER_COMPLETION.json").is_file() or not read(job / "WORKER_COMPLETION.json").get("logical_success"):
                violations.append([context["id"], method, "BRANCH_INVALID"]); continue
            equality, posterior = read(job / "POSTPROBE_EQUALITY.json"), read(job / "PREACTION_POSTERIOR.json")
            decision, branch = read(job / "PLANNER_DECISION.json"), read(job / "BRANCH_RESULT.json")
            trace = jsonlines(job / "ACTION_TRACE.jsonl")
            checks = {
                "candidate_independent_state": equality.get("passed") is True,
                "posterior_reference_exact": posterior == reference_posterior,
                "online_vla": branch.get("online_vla_verified") is True and branch.get("downstream_action_source") == "ONLINE_VLA",
                "cadence": branch.get("rpc_count") == (branch.get("steps", 0) + 9) // 10,
                "full_task_label": branch.get("outcome", {}).get("label_valid") is True and branch.get("runtime_label_version") == "ONLINE_VLA_350STEP_WHOLE_MESH_RELEASE_SUPPORT_V1",
                "phase_free": decision.get("phase_channels") == "NONE" and decision.get("phase_representation") == "NONE",
                "online_motion_context": decision.get("feasibility_sequence_source") == "ONLINE_VLA_ACTION_CHUNK" and decision.get("scripted_prefix_used") is False,
                "one_force": all(float(row["selected_force_setpoint"]) == float(decision["executed_force_N"]) for row in trace),
                "force_active_until_release": all(row["vla_release_intent"] or float(row["active_force_setpoint"]) == float(decision["executed_force_N"]) for row in trace),
                "release_arbitration": all((not row["vla_release_intent"]) or float(row["final_gripper_command"]) == float(np.float32(0.04)) for row in trace),
                "squeeze_metric": np.isclose(branch["outcome"]["mean_measured_bilateral_squeeze"],
                    np.mean([row["measured_bilateral_squeeze"] for row in trace if not row["vla_release_intent"]]), rtol=0, atol=1e-12),
                "af_no_true_mass": method != "ACTIVEFORCING_MASS" or (decision.get("gt_mass_used") is False and
                    decision.get("simulator_mass_kg") is None and posterior.get("hidden_mass_used") is False),
                "gt_only_oracle": method != "GT_MASS" or (decision.get("gt_mass_used") is True and
                    decision.get("simulator_mass_kg") == context["mass_kg"]),
                "fixed4": method != "FIXED_4" or decision.get("executed_force_N") == 4.0,
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed: violations.append([context["id"], method, *failed])
            if method == "ACTIVEFORCING_MASS": af_forces.append(float(decision["executed_force_N"]))
            rows.append({"context_id": context["id"], "method": method, "checks": checks,
                         "selected_force_N": decision["executed_force_N"], "full_task_success_diagnostic_only": branch["outcome"]["full_task_success_y"]})
    unique = len(set(af_forces)); rule = plan["qualification_rule"]
    qualified = not violations and len(rows) == len(plan["contexts"]) * len(plan["methods"]) and unique >= rule["selected_force_unique_count_min"]
    result = {"qualified": qualified, "status": "PASS" if qualified else "FAIL", "development_only": True,
        "final_outcomes_used_for_tuning": False, "contexts": len(plan["contexts"]), "valid_method_branches": len(rows),
        "AF_selected_force_unique_count": unique, "AF_selected_forces_N": af_forces,
        "violations": violations, "checks": rows,
        "authorizes_final_freeze": qualified,
        "scope": "Software/runtime transfer qualification only; development full-task outcomes are diagnostic and do not alter models, utility, masses, or final protocol."}
    write(target, result); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
