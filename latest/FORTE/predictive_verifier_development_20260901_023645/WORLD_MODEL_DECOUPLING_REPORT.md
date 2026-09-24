# World-model decoupling

WM-PhysicsOnly kept the same H8 GRU, inputs, normalization, physical trajectory target, IE target, optimizer, epochs, and seeds. Its objective was exactly `Lphysics + 1.0 LIE`; no full-task outcome label or gradient entered world-model training. The same selected verifier architecture was retrained in each predicted domain with grouped root-heldout CV.

Mean old-DEV standardized trajectory MAE: current=0.1698, PhysicsOnly=0.1677. Mean IE error: current=0.1380, PhysicsOnly=0.1275. Mean verifier CV AUROC: current=0.843, PhysicsOnly=0.770; FPR current=0.239, PhysicsOnly=0.347. Thus PhysicsOnly does not collapse in trajectory fidelity, but its verifier discrimination is weaker; the current representation should be called outcome-shaped predictive physical representation, not a purely physical world model.

This is TRAIN-CV/old-DEV mechanism evidence, not untouched TEST evidence.
