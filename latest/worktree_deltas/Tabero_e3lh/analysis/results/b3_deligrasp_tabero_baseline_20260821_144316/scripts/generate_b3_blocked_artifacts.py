#!/usr/bin/env python3
"""Generate B3 DeliGrasp-style Tabero port artifacts.

This run is intentionally blocked before prediction/evaluation because no
official or credible DeliGrasp LLM response source is available in the current
environment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/home/exouser/Tabero")
OUT = ROOT / "analysis/results/b3_deligrasp_tabero_baseline_20260821_144316"
B2R2 = ROOT / "analysis/results/b2r2_tabero_task_breadth_20260821_073931"
DELIGRASP = (
    ROOT
    / "analysis/research/b1_runnable_baseline_audit_20260820_063949/repositories/DeliGrasp"
)
PROMPT_FILE = DELIGRASP / "magpie/prompt_planner/prompts/mp_prompt_thinker_coder_muk.py"
GRIPPER_FILE = DELIGRASP / "magpie/gripper.py"
CONV_FILE = DELIGRASP / "magpie/prompt_planner/conversation.py"
SERVER_FILE = DELIGRASP / "webserver/server.py"
TABERO_FORCE_FILE = (
    ROOT / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"
)

TASKS = [
    (0, "alphabet_soup_1", "Pick up the alphabet soup and place it in the basket."),
    (1, "cream_cheese_1", "Pick up the cream cheese and place it in the basket."),
    (2, "salad_dressing_1", "Pick up the salad dressing and place it in the basket."),
    (5, "tomato_sauce_1", "Pick up the tomato sauce and place it in the basket."),
    (6, "butter_1", "Pick up the butter and place it in the basket."),
]
NEGATIVE_TASK = (7, "milk_1", "Pick up the milk and place it in the basket.")


def run(cmd: list[str], cwd: Path = ROOT) -> str:
    return subprocess.check_output(cmd, cwd=str(cwd), text=True).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def md_table(rows: list[dict], columns: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def load_b2r2_refs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fstar = pd.read_csv(B2R2 / "FULLTASK_CANONICAL_FSTAR.csv")
    robust = pd.read_csv(B2R2 / "FIXED_ROBUST_BY_TASK.csv")
    gt = pd.read_csv(B2R2 / "GT_MINFORCE_BY_TASK.csv")
    return fstar, robust, gt


def make_reference_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fstar, robust, gt = load_b2r2_refs()
    positive_ids = [t[0] for t in TASKS]
    robust_ref = robust[robust["task_id"].isin(positive_ids)].copy()
    gt_ref = gt[gt["task_id"].isin(positive_ids)].copy()
    fstar_ref = fstar[fstar["task_id"].isin(positive_ids)].copy()
    robust_ref.to_csv(OUT / "FIXED_ROBUST_REFERENCE.csv", index=False)
    gt_ref.to_csv(OUT / "GT_MINFORCE_REFERENCE.csv", index=False)
    return fstar_ref, robust_ref, gt_ref


def official_prompt_records() -> list[dict]:
    prompt_source = PROMPT_FILE.read_text(encoding="utf-8")
    prompt_hash = hashlib.sha256(prompt_source.encode("utf-8")).hexdigest()
    records = []
    for task_id, obj, instruction in TASKS:
        records.append(
            {
                "status": "B3_BLOCKED_BY_DELIGRASP_API",
                "method_label": "DELIGRASP_STYLE_TABERO_PORT",
                "task_id": task_id,
                "object": obj,
                "user_command": instruction,
                "hidden_mu_in_prompt": False,
                "official_prompt_file": str(PROMPT_FILE.relative_to(ROOT)),
                "official_prompt_sha256": prompt_hash,
                "api_model_requested_by_official_server_default": "gpt-4-turbo",
                "temperature_in_official_conversation": 0.3,
                "note": "Prompt not sent because OpenAI SDK and API credentials were unavailable.",
            }
        )
    return records


def write_prompt_and_response_artifacts() -> None:
    with (OUT / "DELIGRASP_PROMPTS.jsonl").open("w", encoding="utf-8") as f:
        for record in official_prompt_records():
            f.write(json.dumps(record, sort_keys=False) + "\n")
    with (OUT / "DELIGRASP_RAW_RESPONSES.jsonl").open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "status": "B3_BLOCKED_BY_DELIGRASP_API",
                    "api_used": False,
                    "response": None,
                    "reason": "No OPENAI_API_KEY/CORRELL_API_KEY in environment and openai Python package is missing.",
                    "manual_or_cached_response_used": False,
                }
            )
            + "\n"
        )


def write_prediction_and_result_tables(
    fstar_ref: pd.DataFrame, robust_ref: pd.DataFrame, gt_ref: pd.DataFrame
) -> None:
    pred_fields = [
        "task_id",
        "object",
        "estimated_mass_g",
        "estimated_friction",
        "estimated_compliance",
        "estimated_spring_constant_N_per_m",
        "raw_llm_response_id",
        "analytic_force_before_clamp_N",
        "final_selected_force_N",
        "status",
    ]
    pred_rows = [
        {
            "task_id": task_id,
            "object": obj,
            "estimated_mass_g": "",
            "estimated_friction": "",
            "estimated_compliance": "",
            "estimated_spring_constant_N_per_m": "",
            "raw_llm_response_id": "",
            "analytic_force_before_clamp_N": "",
            "final_selected_force_N": "",
            "status": "B3_BLOCKED_BY_DELIGRASP_API",
        }
        for task_id, obj, _ in TASKS
    ]
    write_csv(OUT / "DELIGRASP_PREDICTIONS.csv", pred_rows, pred_fields)

    robust_by_task = {
        int(r.task_id): r.fixed_robust_force_tau_0_8 for r in robust_ref.itertuples(index=False)
    }
    fstar_by_cell = {
        (int(r.task_id), float(r.friction)): r.fstar_tau_0_8
        for r in fstar_ref.itertuples(index=False)
    }
    main_rows = []
    for task_id, obj, _instruction in TASKS:
        for mu in [0.2, 0.5, 1.0]:
            main_rows.append(
                {
                    "Task": f"{task_id} {obj}",
                    "mu": mu,
                    "Fstar_full_N": fstar_by_cell[(task_id, mu)],
                    "Robust_F_N": robust_by_task[task_id],
                    "DeliGrasp_F_N": "",
                    "Full_SR": "",
                    "Pick_SR": "",
                    "Lift_SR": "",
                    "Place_SR": "",
                    "mean_measured_force_N": "",
                    "peak_force_N": "",
                    "Under_Exact_Over": "NOT_RUN_API_BLOCKED",
                    "status": "B3_BLOCKED_BY_DELIGRASP_API",
                }
            )
    main_fields = [
        "Task",
        "mu",
        "Fstar_full_N",
        "Robust_F_N",
        "DeliGrasp_F_N",
        "Full_SR",
        "Pick_SR",
        "Lift_SR",
        "Place_SR",
        "mean_measured_force_N",
        "peak_force_N",
        "Under_Exact_Over",
        "status",
    ]
    write_csv(OUT / "DELIGRASP_MAIN_RESULTS.csv", main_rows, main_fields)
    write_text(
        OUT / "DELIGRASP_MAIN_RESULTS.md",
        "# DeliGrasp Main Results\n\n"
        "Status: `B3_BLOCKED_BY_DELIGRASP_API`.\n\n"
        "No DeliGrasp selected force was produced because no official or credible "
        "LLM response source was available. The table below records the frozen "
        "B3 evaluation cells and references only.\n\n"
        + md_table(
            main_rows,
            ["Task", "mu", "Fstar_full_N", "Robust_F_N", "DeliGrasp_F_N", "Full_SR", "Under_Exact_Over"],
        )
        + "\n",
    )

    gt_by_task = {
        int(r.task_id): r.gt_minforce_mean for r in gt_ref.itertuples(index=False)
    }
    summary_rows = [
        {
            "Task": f"{task_id} {obj}",
            "DeliGrasp_selected_F_N": "",
            "Mean_SR": "",
            "Under_rate": "",
            "Over_rate": "",
            "Fixed_Robust_F_N": robust_by_task[task_id],
            "GT_MinForce_mean_N": gt_by_task[task_id],
            "status": "B3_BLOCKED_BY_DELIGRASP_API",
        }
        for task_id, obj, _ in TASKS
    ]
    write_csv(
        OUT / "DELIGRASP_TASK_SUMMARY.csv",
        summary_rows,
        [
            "Task",
            "DeliGrasp_selected_F_N",
            "Mean_SR",
            "Under_rate",
            "Over_rate",
            "Fixed_Robust_F_N",
            "GT_MinForce_mean_N",
            "status",
        ],
    )

    write_csv(
        OUT / "OPTIONAL_DELIGRASP_REACTIVE.csv",
        [
            {
                "status": "NOT_RUN",
                "reason": "Prior response unavailable; reactive ablation deferred.",
                "method_label": "DELIGRASP_GT_SLIP_REFERENCE",
            }
        ],
        ["status", "reason", "method_label"],
    )
    write_csv(
        OUT / "OPTIONAL_NEGATIVE_CONTROL.csv",
        [
            {
                "task_id": NEGATIVE_TASK[0],
                "object": NEGATIVE_TASK[1],
                "status": "NOT_RUN",
                "reason": "Main DeliGrasp prior blocked by API.",
            }
        ],
        ["task_id", "object", "status", "reason"],
    )


def write_code_map() -> None:
    text = f"""# DeliGrasp Code Map

Method label for this project: `DELIGRASP_STYLE_TABERO_PORT`.

Source of truth:

- Repo: https://github.com/deligrasp/deligrasp
- Local path: `{DELIGRASP.relative_to(ROOT)}`
- Commit: `{run(["git", "-C", str(DELIGRASP), "rev-parse", "HEAD"])}`.
- Status: official code inspected; B3 prediction blocked before LLM response.

## LLM Prompt

Official server default config sets `grasp=dg` and `llm=gpt-4-turbo`
in `webserver/server.py:54-55`. The server requires
`CORRELL_API_KEY` at import time in `webserver/server.py:77-82`.

The inspected DeliGrasp prompt is
`magpie/prompt_planner/prompts/mp_prompt_thinker_coder_muk.py`.
It asks the LLM to produce a structured grasp description with:

- mass in grams: lines 14-15
- compliance and spring constant: lines 16-17
- gripper/object friction coefficient: line 18
- goal aperture: line 19
- slip closure and output-force increase: lines 20-21

The coder prompt exposes `set_force` and `deligrasp` to the generated
program at lines 64-78 and uses:

```python
initial_force = (mass * 9.81) / (mu * 1000)
additional_force = np.max([0.01, additional_closure * spring_constant * 0.0001])
G.set_force(initial_force, 'both')
G.deligrasp(goal_aperture, initial_force, additional_closure, additional_force, complete_grasp)
```

from lines 94-109.

## API Dependency

`magpie/prompt_planner/conversation.py:21-23` constructs `OpenAI()`.
`conversation.py:39-41` calls `client.chat.completions.create(...)`
with `temperature=0.3`. `conversation.py:74-99` appends the user
command plus "Make sure to ignore irrelevant options." and optionally an
image payload.

Current B3 environment:

- `OPENAI_API_KEY`: absent
- `CORRELL_API_KEY`: absent
- Python `openai` package: missing

Therefore no official API call was made, and no manual/cached response
was accepted for the five B3 tasks.

## Mass Estimate

Mass is not hard-coded in the repo. It is supplied by the LLM in the
thinker output using the prompt field "approximate mass ... grams".

## Friction Estimate

Friction is not read from sensors or simulation material config for the
pre-contact prior. It is supplied by the LLM as an approximate
gripper/object friction coefficient. B3 would keep this identical for
all hidden simulator `mu` values of the same object.

## Compliance / Stiffness Estimate

Compliance is a high/medium/low semantic category plus a spring constant
in N/m, supplied by the LLM. Prompt rule 6 states that spring constants
can range broadly from 20 N/m to 2000 N/m.

## Force Formula

The official prompt's default pre-contact force is:

```text
initial_force_N = mass_g * 9.81 / (mu_est * 1000)
```

This is the total command passed to `G.set_force(initial_force, 'both')`.
`Gripper.set_force` stores the total in `self.applied_force`, then
halves it before writing motor torque when `finger == 'both'`
(`magpie/gripper.py:308-320`).

## Slip-Loop Logic

`Gripper.deligrasp(x, fc, dx, df, complete=True)`:

- sets initial force `fc`
- closes to the goal aperture
- records load during closure
- calls `check_slip(load_data, fc, 'both')`
- while slippage is true, closes by `dx`, increases applied force by
  `df` when average measured force exceeds 0.10 N, and repeats

Evidence: `magpie/gripper.py:408-490`.

`check_slip` halves the stop force for two-finger grasps and returns
slippage when either finger never reaches the stop load
(`magpie/gripper.py:630-666`).

## Gripper-Specific Parameters

- AX-12 baudrate: 1,000,000 (`magpie/gripper.py:19-22`)
- finger IDs: 1 and 2 (`magpie/gripper.py:23-29`)
- default speed: 100 bits (`magpie/gripper.py:30-34`)
- default torque: 200 bits (`magpie/gripper.py:35-36`)
- default compliance margin/slope: 1 / 32 (`magpie/gripper.py:64-69`)
- measurable max/min force in prompt: 16 N / 0.15 N

## Physical Constants

- gravity: 9.81 m/s^2 in prompt force formula
- damping constant for slip force increment: 0.1 in the prompt
- low-pass no-contact threshold: average force > 0.10 N before
  increasing force in the slip loop

## Clamping / Safety Logic

`set_force` clamps the per-finger force after halving:

```text
per_finger_force = min(max(total_force / 2, 0.15), 16.1)
```

then converts N to AX-12 load with `N_to_load`
(`magpie/gripper.py:308-320`, `692-709`).

No B3 Tabero clamp was applied because no DeliGrasp selected force was
available.
"""
    write_text(OUT / "DELIGRASP_CODE_MAP.md", text)


def write_port_and_mapping() -> None:
    port = """# DeliGrasp Port Spec

Label: `DELIGRASP_STYLE_TABERO_PORT`.

This is not `ORIGINAL_DELIGRASP_REPRODUCTION`.

## Scope

Borrowed from official DeliGrasp:

- semantic physical prior over object mass, friction, and compliance
- analytic pre-contact force rule
- optional `deligrasp` slip-loop logic as a separate ablation

Tabero-specific:

- arm/downstream trajectory: Tabero scripted/oracle full-task policy
- force execution: Tabero calibrated force servo
- hidden simulator friction: not included in prompts
- benchmark tasks: frozen positive set `[0, 1, 2, 5, 6]`

## Inputs Allowed

For each object identity, the DeliGrasp prompt may include:

- object name
- neutral Tabero instruction
- RGB only if the official vision prompt is selected
- general semantic knowledge from the LLM

Forbidden:

- ground-truth friction
- simulator material config
- hidden physics label
- future slip signal for the prior-only row
- oracle F*

## API Status

Blocked in this run:

- official code requires OpenAI API access
- current environment has no API credential
- current Python environment lacks the `openai` package
- no complete official-prompt cached responses exist for all five B3 tasks

No selected force was generated.
"""
    write_text(OUT / "DELIGRASP_PORT_SPEC.md", port)

    mapping = """# DeliGrasp Force Mapping

Status: blocked before numeric mapping.

Official DeliGrasp pre-contact command:

```text
initial_force_N = mass_g * 9.81 / (mu_est * 1000)
```

Official `set_force(force, 'both')` treats the input as the total
two-finger command and halves it internally for per-finger motor torque.

Tabero force servo uses a two-finger squeeze scalar:

```text
f_sq = 2 * min(abs(fL_z), abs(fR_z))
```

Therefore the faithful planned mapping is:

```text
DeliGrasp initial_force_N -> Tabero squeeze target_N
left finger target z  = +initial_force_N / 2
right finger target z = -initial_force_N / 2
```

No rounding to `{3, 4, 5, 6, 8}` N is allowed. The raw continuous
DeliGrasp force would be preserved. If an execution clamp were required,
it would be fixed before evaluation and logged as unclamped/clamped force
for every task.

No clamp was applied in this blocked run.
"""
    write_text(OUT / "DELIGRASP_FORCE_MAPPING.md", mapping)


def write_env_and_verdict(gt_ref: pd.DataFrame) -> None:
    try:
        import numpy as np

        numpy_version = np.__version__
    except Exception:
        numpy_version = None
    try:
        import matplotlib

        matplotlib_version = matplotlib.__version__
    except Exception:
        matplotlib_version = None

    try:
        import openai  # noqa: F401

        openai_import = True
    except Exception as exc:
        openai_import = False
        openai_error = f"{type(exc).__name__}: {exc}"
    else:
        openai_error = None

    env = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tabero_root": str(ROOT),
        "tabero_commit": run(["git", "rev-parse", "HEAD"]),
        "deligrasp_repo": "https://github.com/deligrasp/deligrasp",
        "deligrasp_commit": run(["git", "-C", str(DELIGRASP), "rev-parse", "HEAD"]),
        "deligrasp_prompt_file_sha256": sha256(PROMPT_FILE),
        "deligrasp_gripper_file_sha256": sha256(GRIPPER_FILE),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": numpy_version,
        "pandas": pd.__version__,
        "matplotlib": matplotlib_version,
        "openai_import_ok": openai_import,
        "openai_import_error": openai_error,
        "OPENAI_API_KEY_present": bool(os.environ.get("OPENAI_API_KEY")),
        "CORRELL_API_KEY_present": bool(os.environ.get("CORRELL_API_KEY")),
        "benchmark_source": str((B2R2 / "FULLTASK_CANONICAL_FSTAR.csv").relative_to(ROOT)),
        "benchmark_tasks": [0, 1, 2, 5, 6],
        "method_change": "NONE",
    }
    write_json(OUT / "ENV_PROVENANCE.json", env)

    gt_by_task = {str(int(r.task_id)): r.gt_minforce_mean for r in gt_ref.itertuples(index=False)}
    verdict = {
        "status": "B3_BLOCKED_BY_DELIGRASP_API",
        "method_change": "NONE",
        "benchmark_tasks": [0, 1, 2, 5, 6],
        "deligrasp_repo": "https://github.com/deligrasp/deligrasp",
        "deligrasp_commit": env["deligrasp_commit"],
        "deligrasp_code_status": "PARTIAL",
        "api_used": False,
        "official_prompt_used": "INSPECTED_NOT_QUERIED",
        "selected_force_by_task": {},
        "mean_full_sr": None,
        "mean_selected_force_N": None,
        "under_force_rate": None,
        "exact_force_rate": None,
        "over_force_rate": None,
        "task2_low_mu_success": None,
        "fixed_robust_mean_sr": None,
        "gt_minforce_mean_force_N": sum(float(v) for v in gt_by_task.values()) / len(gt_by_task),
        "hidden_physics_failure_observed": None,
        "semantic_prior_changes_with_hidden_mu": False,
        "reactive_ablation_run": False,
        "primary_evidence": [
            "Official DeliGrasp prompt/code inspected.",
            "No OPENAI_API_KEY or CORRELL_API_KEY present.",
            "Python openai package missing.",
            "No complete official-prompt cached response exists for B3 tasks.",
        ],
        "limitations": [
            "No DeliGrasp selected forces were produced.",
            "No Tabero smoke or main evaluation was run.",
            "Downstream comparison remains pending until credible raw LLM responses are available.",
        ],
    }
    write_json(OUT / "FINAL_VERDICT.json", verdict)


def write_readme() -> None:
    readme = """# B3 - DeliGrasp-Style Semantic-Prior Baseline

Status: `B3_BLOCKED_BY_DELIGRASP_API`.

This directory is an isolated B3 artifact directory for the frozen Tabero
positive benchmark set `[0, 1, 2, 5, 6]`.

No D2, B2/B2-R2, E2E, Tabero core, oracle, benchmark, probe, rule, or
learned network files were modified.

## What Was Completed

- Official DeliGrasp code was inspected.
- `DELIGRASP_CODE_MAP.md` maps prompt, force formula, slip loop,
  hardware parameters, API dependency, constants, and clamps to source
  files and line ranges.
- `DELIGRASP_PORT_SPEC.md` defines the Tabero port as
  `DELIGRASP_STYLE_TABERO_PORT`.
- `DELIGRASP_FORCE_MAPPING.md` defines the planned force mapping from
  DeliGrasp total force to Tabero squeeze target.
- Frozen B2-R2 reference tables were copied into B3 reference CSVs.

## Why The Run Is Blocked

Official DeliGrasp depends on an OpenAI chat completion call. The current
environment has no `OPENAI_API_KEY` or `CORRELL_API_KEY`, and the Python
environment does not have the `openai` package installed. Existing local
DeliGrasp cache files do not cover all five B3 objects using the official
prompt. Therefore no credible `estimated mass / friction / stiffness`
response was available.

Following the B3 instruction, no parameters were hand-invented and no
downstream Tabero evaluation was run.

## To Resume

Provide official-prompt raw responses for the five object identities, or
enable the official API path with fixed model settings. Then generate
`DELIGRASP_PREDICTIONS.csv`, run first-smoke and main evaluation, and
replace the blocked main-result placeholders.
"""
    write_text(OUT / "README.md", readme)


def write_placeholder_plots() -> None:
    plot_names = [
        "selected_force_vs_oracle",
        "sr_by_task_friction",
        "under_over_by_task",
        "semantic_prior_hidden_mu_failure",
        "deli_vs_fixed_vs_oracle",
    ]
    for name in plot_names:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            "B3_BLOCKED_BY_DELIGRASP_API",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        ax.text(
            0.5,
            0.42,
            "No credible official-prompt DeliGrasp response available.",
            ha="center",
            va="center",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(OUT / "plots" / f"{name}.png", dpi=180)
        plt.close(fig)
    write_text(
        OUT / "plots/README.md",
        "Plots are placeholders because B3 was blocked before DeliGrasp predictions.\n",
    )


def write_logs() -> None:
    write_text(
        OUT / "logs/api_status.log",
        "\n".join(
            [
                "OPENAI_API_KEY_present=false",
                "CORRELL_API_KEY_present=false",
                "openai_python_package=missing",
                "api_used=false",
                "manual_or_cached_response_used=false",
                "status=B3_BLOCKED_BY_DELIGRASP_API",
            ]
        )
        + "\n",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    fstar_ref, robust_ref, gt_ref = make_reference_tables()
    write_prompt_and_response_artifacts()
    write_prediction_and_result_tables(fstar_ref, robust_ref, gt_ref)
    write_code_map()
    write_port_and_mapping()
    write_env_and_verdict(gt_ref)
    write_readme()
    write_placeholder_plots()
    write_logs()
    print(f"Wrote B3 blocked artifacts to {OUT}")


if __name__ == "__main__":
    main()
