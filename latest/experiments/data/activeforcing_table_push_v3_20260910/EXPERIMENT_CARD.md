# Experiment card: Table-push ActiveForcing V3

Question: with exact simulator/controller state held fixed, does the existing OSC channel have a locally monotonic contact-force effect; if not, can a bounded local low-level wrench augmentation demonstrate one before any task claim?

Primary evidence is raw MuJoCo robot-on-plate contact force measured after the relevant control step. Secondary evidence includes state hashes, action headroom, actuator saturation, and native task result. This is a simulator diagnosis, not a hardware-force claim.

Known limitations: canonical pi0 outputs can be stochastic despite seed reset; therefore identical seed is never accepted as exact pairing. R1 tests local response only, not task feasibility.
