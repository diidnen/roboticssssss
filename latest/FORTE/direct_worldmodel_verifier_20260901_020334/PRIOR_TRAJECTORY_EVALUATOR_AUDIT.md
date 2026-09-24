# Prior trajectory-to-outcome evidence

The prior trajectory physical imagination run trained an outcome evaluator on real trajectories. It reached TEST AUROC 1.0 on real trajectories, but AUROC 0.5 on imagined trajectories and produced 71 imagined false-safe branches. A later forensic found real-trajectory boundary correctness 1.0 and saved model-trajectory correctness 0.0 on 11 DEV triplets. Therefore this diagnostic does not repeat the real-trajectory evaluator: it fits the evaluator directly on TRAIN imagined trajectories and tests only on held-out-context DEV imagined trajectories.

Sources:
- /home/exouser/Tabero/analysis/results/trajectory_physical_imagination_20260829_065220/FINAL_REPORT.md
- /home/exouser/Tabero/analysis/results/force_sensitivity_forensic_20260829_090000/FINAL_REPORT.md
- /home/exouser/Tabero/analysis/results/evaluator_interface_calibration_20260829_112603/FINAL_REPORT.md
