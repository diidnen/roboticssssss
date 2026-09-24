# Verifier architecture comparison

Selection used grouped 4-fold CV over 22 TRAIN roots; all branches, forces, and repeats from a root stayed together. Four pre-registered architectures and exactly three seeds were run. No DEV architecture selection and no TEST read occurred.

Safety-first selection chose **V0_LINEAR** using mean-over-seed FPR, under-force, SR, FNR, then mean force. The complete per-seed and three-seed mean rows are in `VERIFIER_ARCHITECTURE_COMPARISON.csv`; no best seed was selected.

After selection, the already-viewed DEV was used only for sanity. Number of the three historical false-negative force cells called SUCCESS by all three seeds on both repeats: `{'V0_LINEAR': 0, 'V1_SMALL_MLP': 2, 'V2_TEMPORAL_GRU': 0, 'V3_GRU_CONTEXT': 0}`. This audit did not change the selected architecture.
