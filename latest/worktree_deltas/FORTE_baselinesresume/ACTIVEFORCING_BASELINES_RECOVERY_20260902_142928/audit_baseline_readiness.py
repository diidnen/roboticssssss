#!/usr/bin/env python3
"""CPU-only FORTE/Tabero baseline readiness and paired-manifest audit.

This program deliberately does not import Isaac Lab, connect to serial devices,
contact the policy server, or execute a rollout.  It only validates files,
source contracts, the FORTE CPU model interface, and frozen tuple provenance.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib


OUT = Path(__file__).resolve().parent
FORTE = OUT.parent
FORTE_E1 = Path("/home/exouser/FORTE_e1diag/ACTIVEFORCING_E1_CRITICAL_CLOSURE_20260902_084000")
TABERO = Path("/home/exouser/Tabero_e1diag")
TABERO_SHARED = Path("/home/exouser/Tabero")
PROSPECTIVE = TABERO_SHARED / "analysis/results/gnp_style_visual_context_prospective_20260831_011000"
E5_RUNNER = Path(
    "/home/exouser/FORTE/analysis/results/"
    "ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_053006/run_e5_fresh_utility.py"
)
CHECKPOINT = Path(
    "/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/"
    "checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999"
)
DATASET_ROOT = Path(
    "/media/volume/newdata/exouser/tabero/data/Isaaclab_Libero/assembled_hdf5"
)

EXPECTED = {
    "forte_commit": "7f88d0184c1617ed95e67502da96e60be07b3689",
    "tabero_commit": "80ab3be09ce884f86cfc2037d3af30bc28061426",
    "forte_model_sha256": "9d9ba2449282196db1a85f31e1e41cdca7ccd56522130e2ff3deffd93e8017bc",
    "checkpoint_content_sha256_inherited": "0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17",
    "candidate_rows": 720,
    "paired_tuples": 144,
    "contexts": 72,
    "roots": 24,
    "candidates_per_tuple": 5,
    "tasks": [0, 1, 5, 6],
}

TASKS = {
    0: {
        "object": "alphabet_soup_1",
        "instruction": "pick up the alphabet soup and place it in the basket",
        "dataset": "libero_object_task0_pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5",
    },
    1: {
        "object": "cream_cheese_1",
        "instruction": "pick up the cream cheese and place it in the basket",
        "dataset": "libero_object_task1_pick_up_the_cream_cheese_and_place_it_in_the_basket_demo.hdf5",
    },
    5: {
        "object": "tomato_sauce_1",
        "instruction": "pick up the tomato sauce and place it in the basket",
        "dataset": "libero_object_task5_pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5",
    },
    6: {
        "object": "butter_1",
        "instruction": "pick up the butter and place it in the basket",
        "dataset": "libero_object_task6_pick_up_the_butter_and_place_it_in_the_basket_demo.hdf5",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def compile_only(path: Path) -> dict[str, Any]:
    try:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        return {"status": "PASS", "sha256": sha256(path), "bytes": path.stat().st_size}
    except Exception as exc:  # pragma: no cover - recorded for forensic output
        return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def build_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_path = FORTE_E1 / "E1_CANDIDATE_DECISION_CHAIN.csv"
    context_path = PROSPECTIVE / "PROSPECTIVE_CONTEXT_MANIFEST.csv"
    restore_path = PROSPECTIVE / "PROSPECTIVE_SNAPSHOT_RESTORE_AUDIT.json"
    candidates = read_csv(candidate_path)
    contexts = {row["context_id"]: row for row in read_csv(context_path)}
    restore_doc = json.loads(restore_path.read_text(encoding="utf-8"))
    restores = {row["context_id"]: row for row in restore_doc["rows"]}

    require(len(candidates) == EXPECTED["candidate_rows"], "candidate-row count drift")
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        grouped[(row["context_id"], int(row["repeat"]))].append(row)
    require(len(grouped) == EXPECTED["paired_tuples"], "paired-tuple count drift")
    require(
        {len(rows) for rows in grouped.values()} == {EXPECTED["candidates_per_tuple"]},
        "candidate coverage drift",
    )

    manifest: list[dict[str, Any]] = []
    for (context_id, repeat), candidate_rows in sorted(
        grouped.items(), key=lambda item: (int(item[1][0]["task"]), item[0][0], item[0][1])
    ):
        first = candidate_rows[0]
        task = int(first["task"])
        context = contexts.get(context_id)
        restore = restores.get(context_id)
        require(context is not None, f"context missing: {context_id}")
        require(restore is not None, f"restore audit missing: {context_id}")
        require(restore["restore_exact"] == "1", f"inexact snapshot restore: {context_id}")
        snapshot = Path(restore["snapshot_path"])
        require(snapshot.is_file(), f"snapshot unavailable: {snapshot}")
        require(
            restore["snapshot_state_hash"] == restore["restored_state_hash"]
            == restore["second_restore_hash"],
            f"snapshot hash mismatch: {context_id}",
        )
        task_meta = TASKS[task]
        dataset = DATASET_ROOT / task_meta["dataset"]
        require(dataset.is_file(), f"task dataset unavailable: {dataset}")
        force_support = sorted(float(row["force_N"]) for row in candidate_rows)
        pair_id = f"{context_id}_R{repeat}"
        manifest.append(
            {
                "pair_id": pair_id,
                "evidence_split": "TRAIN_GROUPED_ROOT_OOF_DEVELOPMENT",
                "task": task,
                "task_suite": "libero_object",
                "object_identity": task_meta["object"],
                "language_instruction": task_meta["instruction"],
                "context_id": context_id,
                "root_id": first["root_id"],
                "source_root_id": context["source_root_id"],
                "root_index": int(context["root_index"]),
                "root_seed": int(context["root_seed"]),
                "friction_band": context["friction_band"],
                "mu_GT": context["mu_GT"],
                "repeat": repeat,
                "snapshot_state_hash": restore["snapshot_state_hash"],
                "snapshot_path": str(snapshot),
                "snapshot_exact_restore_audited": 1,
                "archived_candidate_force_support_N": json.dumps(force_support),
                "forte_runtime_contract": "native_6ch_SVR_PSD_slip_Dynamixel_impedance_reaction",
                "forte_rollout_status": "NA_ADAPTER_BLOCKED",
                "tabero_runtime_contract": "native_13D_Field+FS_replan10_no_force_override",
                "tabero_rollout_status": "NA_NOT_RUN",
                "required_pairing": "same_snapshot_hash+task+root_seed+mu_GT+repeat",
                "required_telemetry": "terminal+commanded_force_LR+measured_contact_force+under_excess+latency+failure_stage",
            }
        )

    context_ids = {row["context_id"] for row in manifest}
    roots = {row["root_id"] for row in manifest}
    tasks = Counter(row["task"] for row in manifest)
    require(len(context_ids) == EXPECTED["contexts"], "context count drift")
    require(len(roots) == EXPECTED["roots"], "root count drift")
    require(tasks == Counter({task: 36 for task in EXPECTED["tasks"]}), "task balance drift")
    canonical = "\n".join(
        f"{row['pair_id']}|{row['snapshot_state_hash']}|{row['mu_GT']}" for row in manifest
    ).encode("utf-8")
    qa = {
        "status": "PASS_CPU_MANIFEST_FREEZE",
        "candidate_rows": len(candidates),
        "paired_tuples": len(manifest),
        "contexts": len(context_ids),
        "roots": len(roots),
        "tasks": dict(sorted(tasks.items())),
        "candidates_per_tuple": sorted({len(rows) for rows in grouped.values()}),
        "all_snapshot_paths_present": True,
        "all_snapshot_restore_hashes_exact": True,
        "tuple_set_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_sha256": {
            str(candidate_path): sha256(candidate_path),
            str(context_path): sha256(context_path),
            str(restore_path): sha256(restore_path),
        },
    }
    return manifest, qa


def forte_audit() -> dict[str, Any]:
    sys.path.insert(0, str(FORTE))
    from forte.runtime.force_and_slip import (  # noqa: PLC0415
        NUM_CHANNELS,
        SENSOR_HZ,
        load_model,
    )
    from forte.runtime.sys_utils import sensor2force_feature  # noqa: PLC0415

    model_path = FORTE / "models/SVR_ckpt.pkl"
    model = load_model(str(model_path))
    require(model is not None, "FORTE SVR could not be loaded")
    model_hash = sha256(model_path)
    require(model_hash == EXPECTED["forte_model_sha256"], "FORTE SVR hash drift")
    require(int(model.n_features_in_) == 24, "FORTE SVR feature dimension drift")
    # Shape-only check; no prediction and no scientific or surrogate outcome.
    import numpy as np  # noqa: PLC0415

    feature_shape = list(sensor2force_feature(np.zeros((20_000, 6))).shape)
    require(feature_shape == [24], "FORTE feature extractor shape drift")

    source_paths = [
        FORTE / "forte/runtime/force_and_slip.py",
        FORTE / "forte/runtime/sys_utils.py",
        FORTE / "forte/sensing/sensor.py",
        FORTE / "forte_gripper/FORTE_gripper.py",
        FORTE / "examples/gripper_showcase_vis_realtime.py",
    ]
    compile_checks = {str(path.relative_to(FORTE)): compile_only(path) for path in source_paths}
    require(all(v["status"] == "PASS" for v in compile_checks.values()), "FORTE compile failure")

    return {
        "status": "CPU_SOFTWARE_READY_FAITHFUL_EXECUTION_BLOCKED",
        "repo_commit": git(FORTE, "rev-parse", "HEAD"),
        "expected_commit_match": git(FORTE, "rev-parse", "HEAD") == EXPECTED["forte_commit"],
        "tracked_diff_paths": git(FORTE, "diff", "--name-only", "HEAD").splitlines(),
        "python": sys.version.split()[0],
        "packages": {
            name: metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "joblib")
        },
        "model": {
            "path": str(model_path),
            "sha256": model_hash,
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "n_features_in": int(model.n_features_in_),
        },
        "interface_contract": {
            "sensor_channels": NUM_CHANNELS,
            "sensor_hz": SENSOR_HZ,
            "svr_feature_dimensions": feature_shape[0],
            "feature_shape_check": "PASS_INTERFACE_ONLY_NO_PREDICTION",
            "slip_band_hz": [10, 50],
            "welch_nperseg": 400,
            "slip_reaction_position_increment": -0.006,
            "slip_reaction_cooldown_seconds": 0.2,
            "actuation": "Dynamixel impedance: paired position and current commands",
        },
        "source_compile_checks": compile_checks,
        "hardware": {
            "ttyACM": sorted(glob.glob("/dev/ttyACM*")),
            "ttyUSB": sorted(glob.glob("/dev/ttyUSB*")),
            "faithful_hardware_available": False,
        },
        "blockers": [
            "No six-channel FORTE analog sensor device or recorded native tactile replay is available.",
            "No audited simulator adapter maps simulator observations to calibrated six-channel SVR/PSD input semantics.",
            "No audited adapter maps the native -0.006 Dynamixel impedance-position reaction to the simulated gripper.",
        ],
        "forbidden_substitutions": [
            "oracle slip",
            "GelSight marker derivative as FORTE analog channels",
            "fixed or ladder force policy labeled FORTE",
            "measured-force servo labeled Dynamixel position reaction",
        ],
    }


def tabero_audit() -> dict[str, Any]:
    source_paths = [
        TABERO / "benchmarks/openpi/openpi_inference_client.py",
        TABERO / "benchmarks/common/closedloop_policy_inference.py",
        TABERO / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py",
    ]
    compile_checks = {str(path.relative_to(TABERO)): compile_only(path) for path in source_paths}
    require(all(v["status"] == "PASS" for v in compile_checks.values()), "Tabero compile failure")

    runner_text = source_paths[0].read_text(encoding="utf-8")
    action_text = source_paths[2].read_text(encoding="utf-8")
    markers = {
        "replan_steps_10": "replan_steps: int = 10" in runner_text,
        "native_13d_slice": "action_chunk[:, :13]" in runner_text,
        "native_13d_env_step": "env.step(action[i].reshape([1, -1]))" in runner_text,
        "force_slots_split": "actions[:, 7:10]" in action_text and "actions[:, 10:13]" in action_text,
        "hybrid_action_dim_13": "return 13" in action_text,
    }
    require(all(markers.values()), f"Tabero native action contract drift: {markers}")

    config_path = TABERO / "benchmarks/datasets/libero/config/libero_object.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected = {int(row["task_id"]): row for row in config["tasks"] if int(row["task_id"]) in TASKS}
    require(set(selected) == set(TASKS), "Tabero task config coverage drift")
    override_checks = {
        str(task): selected[task].get("control_overrides") in (None, {}) for task in TASKS
    }
    require(all(override_checks.values()), "task-level force override is enabled")

    checkpoint_files = {
        "root_metadata": CHECKPOINT / "_CHECKPOINT_METADATA",
        "params_metadata": CHECKPOINT / "params/_METADATA",
        "norm_stats": CHECKPOINT / "assets/NathanWu7/tabero/norm_stats.json",
    }
    require(all(path.is_file() for path in checkpoint_files.values()), "checkpoint metadata missing")
    datasets = {str(task): DATASET_ROOT / TASKS[task]["dataset"] for task in TASKS}
    require(all(path.is_file() for path in datasets.values()), "task dataset missing")

    inherited_manifest = (
        TABERO_SHARED
        / "analysis/results/p6g1r1_controller_grasp_vla_handoff_20260825_180558/"
        "P6G1R1_FROZEN_SYSTEM_MANIFEST.json"
    )
    frozen = json.loads(inherited_manifest.read_text(encoding="utf-8"))
    inherited_hash_match = (
        frozen.get("checkpoint_hash_sha256")
        == EXPECTED["checkpoint_content_sha256_inherited"]
    )
    require(inherited_hash_match, "inherited frozen checkpoint digest drift")

    return {
        "status": "NATIVE_RUNTIME_COMPONENTS_READY_PAIRED_ROLLOUT_NOT_RUN",
        "repo_commit": git(TABERO, "rev-parse", "HEAD"),
        "expected_commit_match": git(TABERO, "rev-parse", "HEAD") == EXPECTED["tabero_commit"],
        "repo_clean": git(TABERO, "status", "--porcelain") == "",
        "source_compile_checks": compile_checks,
        "native_action_contract_markers": markers,
        "task_control_overrides_absent": override_checks,
        "checkpoint": {
            "path": str(CHECKPOINT),
            "step": 49999,
            "policy_config": "pi0_lora_tacfield_tabero",
            "content_sha256": EXPECTED["checkpoint_content_sha256_inherited"],
            "content_hash_provenance": str(inherited_manifest),
            "content_hash_recomputed_this_audit": False,
            "small_file_sha256": {name: sha256(path) for name, path in checkpoint_files.items()},
        },
        "task_datasets": {
            task: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for task, path in datasets.items()
        },
        "existing_native_adapter": {
            "path": str(E5_RUNNER),
            "present": E5_RUNNER.is_file(),
            "sha256": sha256(E5_RUNNER) if E5_RUNNER.is_file() else None,
            "semantics": "removes only ActiveForcing force-slot overwrite from frozen VLA executor",
        },
        "blockers": [
            "No exact E1 same-tuple Tabero rollout rows exist; existing roots 7200/7201 are unpaired.",
            "No dedicated writable Tabero baseline checkout exists because the root filesystem is full; Tabero remains read-only.",
            "Simulator execution was intentionally not attempted by this CPU-only audit.",
        ],
        "required_execution": "native 13-D actions, replan_steps=10, no force-slot override, exact manifest snapshots",
    }


def write_report(provenance: dict[str, Any], manifest_qa: dict[str, Any]) -> None:
    forte = provenance["forte"]
    tabero = provenance["tabero"]
    report = f"""# Faithful FORTE/Tabero baseline recovery status

Audit time: `{provenance['created_utc']}`  
Lane: `{FORTE}`  
Execution policy: **CPU-only; no simulator, policy-server request, serial connection, or rollout.**

## Decision

- **FORTE:** `IMPLEMENTATION_BLOCKED`. The official CPU model and runtime source are intact, but the benchmark cannot be called FORTE until a six-channel analog-tactile adapter and a Dynamixel-equivalent impedance-position actuator mapping pass fidelity QA.
- **Tabero:** `RUNTIME_COMPONENTS_READY / PAIRED_DATA_NOT_RUN`. The clean source checkout, native 13-D path, frozen checkpoint metadata/norm stats, task datasets, exact snapshots, and no-override task configs are present. No paired result is claimed.
- **Paired manifest:** `{manifest_qa['paired_tuples']}` tuples (`{manifest_qa['contexts']}` contexts, `{manifest_qa['roots']}` task-specific roots, two repeats; 36 tuples per task) are frozen in `E1_EXTERNAL_BASELINE_PAIRED_TUPLE_MANIFEST.csv`.
- **Scientific result:** `NONE`. Every FORTE/Tabero outcome remains `NA`; oracle-slip, fixed/ladder-force, and unpaired native smoke rows are excluded.

## Faithfulness gates

| Gate | FORTE | Tabero |
|---|---|---|
| Source/CPU contract | PASS | PASS (compile/static contract only) |
| Frozen model/checkpoint provenance | PASS (`{forte['model']['sha256']}`) | PASS via inherited frozen content digest plus current metadata hashes |
| Exact tuple/reset manifest | Frozen; unusable until adapter exists | Frozen; all snapshot paths and exact-restore hashes present |
| Native sensor/observation semantics | BLOCKED: six analog channels absent | READY: visual/tactile/force history native path present |
| Native actuator semantics | BLOCKED: Dynamixel impedance adapter absent | READY: native 13-D force-position action path present |
| Same-tuple rollout | NOT RUN | NOT RUN |
| Paper/result row | NA | NA |

## Paired-tuple acceptance contract

A future row is admissible only when it uses the manifest's task, `root_seed`, friction, repeat, and exact snapshot hash; starts from a fresh reset/verified restore; reaches an explicit terminal transition; and records commanded left/right force separately from measured contact force, under/excess force, latency, and failure stage. Tabero must execute its learned 13-D outputs without any ActiveForcing force-slot overwrite. FORTE must preserve the native 2 kHz six-channel → 24-feature SVR → 10–50 Hz Welch/variance slip → `-0.006` impedance-position reaction with 0.2 s cooldown.

## Blockers and next legal actions

1. Build and audit the FORTE sensor/actuator simulator adapter. Until both mappings pass, omit FORTE from numerical tables.
2. Restore disk capacity, create a dedicated Tabero baseline checkout, and freeze a lane-local exact-manifest native runner.
3. Only then schedule Tabero same-tuple collection and, after FORTE fidelity passes, FORTE collection. This audit intentionally stops before either simulator action.

The machine's root filesystem was observed at 100% utilization with about 420 MiB free. That is a checkout/output prerequisite blocker, not a scientific result.
"""
    (OUT / "BASELINE_RUNTIME_READINESS.md").write_text(report, encoding="utf-8")


def main() -> None:
    created = datetime.now(timezone.utc).isoformat()
    manifest, manifest_qa = build_manifest()
    write_csv(OUT / "E1_EXTERNAL_BASELINE_PAIRED_TUPLE_MANIFEST.csv", manifest)
    write_json(OUT / "PAIRED_TUPLE_MANIFEST_QA.json", manifest_qa)

    provenance = {
        "status": "CPU_AUDIT_COMPLETE_SIMULATOR_NOT_RUN_NO_BASELINE_RESULTS",
        "created_utc": created,
        "host": {"node": platform.node(), "platform": platform.platform()},
        "execution": {
            "cpu_only": True,
            "simulator_launched": False,
            "policy_server_contacted": False,
            "serial_device_opened": False,
            "rollouts_executed": 0,
            "surrogate_results_claimed": False,
            "tabero_modified": False,
        },
        "forte": forte_audit(),
        "tabero": tabero_audit(),
        "manifest_qa": manifest_qa,
    }
    require(provenance["forte"]["expected_commit_match"], "FORTE commit mismatch")
    require(provenance["tabero"]["expected_commit_match"], "Tabero commit mismatch")
    write_json(OUT / "BASELINE_CPU_PROVENANCE.json", provenance)

    status = {
        "status": "CPU_WORK_COMPLETE_IDLE_ON_PREREQUISITES",
        "created_utc": created,
        "lane": str(FORTE),
        "simulator_run": False,
        "surrogate_results": False,
        "tabero_modified": False,
        "forte": {
            "runtime_readiness": "IMPLEMENTATION_BLOCKED",
            "paired_rows": 0,
            "paper_value": None,
            "prerequisite": "faithful six-channel tactile plus Dynamixel impedance simulator adapter",
        },
        "tabero": {
            "runtime_readiness": "COMPONENTS_READY_PAIRED_DATA_BLOCKED",
            "paired_rows": 0,
            "paper_value": None,
            "prerequisite": "dedicated checkout/disk capacity then exact-manifest native simulator collection",
        },
        "paired_tuple_manifest": {
            "path": str(OUT / "E1_EXTERNAL_BASELINE_PAIRED_TUPLE_MANIFEST.csv"),
            "rows": len(manifest),
            "sha256": sha256(OUT / "E1_EXTERNAL_BASELINE_PAIRED_TUPLE_MANIFEST.csv"),
            "qa": manifest_qa["status"],
        },
        "next_state": "IDLE_UNTIL_PREREQUISITES_AVAILABLE",
    }
    write_json(OUT / "BASELINE_RECOVERY_STATUS.json", status)
    write_report(provenance, manifest_qa)

    output_files = sorted(
        path for path in OUT.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    (OUT / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in output_files),
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
