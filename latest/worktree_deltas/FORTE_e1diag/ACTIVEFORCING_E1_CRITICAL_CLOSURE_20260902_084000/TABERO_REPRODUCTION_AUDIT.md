# Tabero faithful-reproduction audit

## Status

`TABERO = FAITHFUL_REPRODUCTION_BLOCKED`

## Recovered authoritative semantics

Tabero is the native Field+FS vision-tactile-language-action policy. It receives visual/tactile context and force history, emits a 13-D action containing pose, gripper, and left/right force slots, and executes those learned force outputs through its hybrid force-position controller with fresh chunk replanning (`replan_steps=10`). It is not Direct+Utility and must not receive an ActiveForcing force-slot overwrite.

Runtime objective: the learned Tabero policy and its native closed-loop force/tactile/action semantics. Common E1 evaluation can calculate realized Utility afterward, but Utility is not Tabero's runtime selector.

## Provenance

- Isolated sparse checkout: `/home/exouser/Tabero_e1diag`, commit `80ab3be`.
- Native runner: `benchmarks/openpi/openpi_inference_client.py`, SHA-256 `487bec3fa6a5003f5dbe4a327582ec1f9a4724ef2c54fb94b833dfc2f517cc94`.
- Closed-loop wrapper: `benchmarks/common/closedloop_policy_inference.py`, SHA-256 `8ee006f1586f0851f0719057153a7ed5308bb2a8f66337b6e502f4391f076bbe`.
- Frozen checkpoint: `pi0_lora_tacfield_tabero/checkpoints/.../49999`; checkpoint SHA-256 `0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17`; config `pi0_lora_tacfield_tabero`; norm stats `assets/NathanWu7/tabero`.
- The protected inference server observed during this audit is the exact frozen checkpoint/config. It was not modified or restarted.

## Faithfulness and paired-evaluation decision

The current fresh-E2E executor already has the correct Tabero/native-pi0 mode: it executes all native 13-D outputs and removes only ActiveForcing's force-slot replacement. Existing task1 native smoke rows are useful runtime smoke evidence, but their roots (7200/7201) do not match E1 roots (5100–5105), so they are excluded from the paired E1 table.

Final E1 reproduction is blocked because no fresh same-task/root/friction/initial-state/seed Tabero rollout set exists. During this audit the GPU was already carrying protected MASS, E5, and frozen-policy-server workloads above the launch threshold; starting another Isaac job would violate the throughput policy. No protected process was killed or altered.

Unblock requirement: schedule fresh native-mode rollouts on the exact E1 tuple manifest, with the same frozen checkpoint and no force override, then report commanded left/right force slots and measured contact force separately. Until then, missing values remain NA.
