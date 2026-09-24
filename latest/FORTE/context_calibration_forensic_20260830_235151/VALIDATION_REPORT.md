# Validation Report

## Overall Assessment: Share with caveats

The frozen within-distribution conclusion is supported. The GT controller gate is not passed.

## Methodology Review

Protocol and checkpoint hashes were frozen before residual inspection. Calibration was TRAIN-only and applied to the dense grid before force selection. Context oracle features were deployment-time and did not include root/split IDs.

## Calculation Spot-Checks

- 54 residual rows = 2 backends × 27 real cells, with 54 unique backend/context/force keys.
- 1,008 TRAIN predictions per backend.
- Raw Feas under-force reproduces the authoritative 2/8 rule including missing decisions.
- No calibration/context variant passes all five GT gate conditions.

## Required Caveats

- Nine contexts/27 cells limit variance attribution.
- Nearest-centroid context family is a legal but narrow observable-context oracle.
- Five-repeat empirical frequencies are uncertain estimates.
