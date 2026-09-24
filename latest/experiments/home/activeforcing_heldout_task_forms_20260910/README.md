# ActiveForcing held-out task-form audit

Start with [the final report](FINAL_HELDOUT_TASK_FORM_GENERALIZATION_REPORT.md). The result is **NOT_TESTED for new-form AF transfer**: no candidate passed the frozen native-reset admission screen. Supplied-grasp downstream competence remains untested.

- [Actual model inputs](CURRENT_FEASIBILITY_INPUT_AUDIT.md): explicit four-way task one-hot confirmed.
- [Existing generalization](EXISTING_GENERALIZATION_AUDIT.md): root and friction evidence, plus historical LOTO distinction.
- [Task inventory](ALL_AVAILABLE_TASKS.csv) and [form taxonomy](TASK_FORM_TAXONOMY.md).
- [Qualification](FROZEN_VLA_NEW_TASK_QUALIFICATION.md): six valid Fixed-5 episodes, 120 verified online calls; no AF calls.
- [Empty frozen final set](FINAL_HELDOUT_TASK_FORM_SET.json) and [hash](FINAL_HELDOUT_TASK_FORM_SET_SHA256.txt).
- [Candidate visual audit](TASK_FORM_OVERVIEW_CONTACT_SHEET.png): scenes and negative sequences, not successful task-form demonstrations.
- [Claim audit](HELDOUT_TASK_FORM_CLAIM_AUDIT.json), [terminal summary](TERMINAL_SUMMARY.txt), and [QA receipt](evidence/FINAL_ARTIFACT_QA.json).

Reproduce read-only QA with `python3 validate_artifacts.py`. Model tensor reconstruction requires the NumPy/PyTorch interpreter named in the model-input audit. `AUDIT_COMPANION.ipynb` provides a read-only evidence summary. Launch scripts preserve the original negative outcomes; do not rerun them as an outcome search. No models were retrained or formal fresh roots exposed. The isolated qualification policy server was stopped after all screens completed.
