# Reproducibility commands

- Assembly: `python3 /home/exouser/FORTE/assemble_full_activeforcing_claim_closure.py`
- Sealed E5 per task: `run_e5_fresh_utility.py --phase full --task {0,1,5,6} --tuple-start 9 --max-tuples 1`
- Figure export: `python3 /home/exouser/FORTE/generate_activeforcing_paper_figures.py <bundle>`

The exact checkpoint, split, runner, and freeze hashes are recorded in `REPRODUCIBILITY_FINAL.json`; raw stdout/stderr and trajectory references remain under `E5_LOCKED_TEST/` and the timestamped source roots.
