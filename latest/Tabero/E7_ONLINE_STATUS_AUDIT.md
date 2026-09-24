# E7 ONLINE STATUS / EXACT-FLOAT AUDIT

审计范围：E7 continuous-force online outputs、exact-float small block、以及
此前的 5 个 off-grid controller tests。只读审计；未启动 144 matrix，未修改
ActiveForcing、π0、controller、Direct、Utility 或 evaluator 定义。

## 结论

`E7_STATUS = NEEDS_SMALL_CONTROLLER_FIX`

这里的 controller fix 是 small-block 的执行/接触/force-tracking gate
问题，不是把 interpolation accuracy 或 prediction accuracy 当作 controller
parity，也不是单纯把 evaluator 放宽。

## 1. 最近一次 online collector

最近一次 full online run：

- output：`p3_online_full_2x2_live_20260903_085120/`；
- command：`/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python -u /home/exouser/E3_E6_E7_LANES/E7_CONTINUOUS_FORCE_PLANNING/scripts/activeforcing_e7_offgrid_rollout.py --worker`；
- 当前 PID：无，collector 已退出；worker snapshot 只保留 parent PID `266851`，没有可复用的 live worker PID；
- 最近结果：`e7_full_031_UNIFORM_CONTINUOUS`；
- 最后 artifact 更新时间：约 `2026-09-03 09:22:30 UTC`；
- observed：32/144，全部来自 task0；task1/task5/task6 没有 branch result。

四个 worker 都记录为 `natural_exit=true`、`returncode=0`，没有 crash、request
timeout 或 server error 的证据。停止原因是任务/上下文执行正常返回但结果不完整，
随后 execution audit fail-closed：`result_count:32!=144`，并同时报告
controller exact-float、telemetry identity、tracking samples/nonfinite error 等问题。

## 2. exact-float small block 的 8 cases

以下是 retained 8-request block。`controller_exact_float_match` 是 request
与 controller setpoint 的原生浮点相等性；`tracking_error_N` 是 simulator
实际 measured load-bearing force 与请求值的误差，二者不能混为一谈。

| case | requested N | controller N | exact request→controller | measured N | tracking error N | 结果/失败位置 |
|---|---:|---:|---|---:|---:|---|
| 000 FIXED_GRID | 4.7500000000 | 4.7500000000 | PASS | 4.7032997391 | 0.0467002609 | 执行完成；tracking PASS |
| 001 PROPOSAL_GUIDED | 4.7764308836 | 4.7764308836 | PASS | 3.4339629715 | 1.3424679121 | 执行完成；tracking FAIL |
| 002 STRATIFIED_CONTINUOUS | 4.7537276064 | 4.7537276064 | PASS | 3.7821531621 | 0.9715744443 | 执行完成；tracking FAIL |
| 003 UNIFORM_CONTINUOUS | 4.3840363719 | 4.3840363719 | PASS | 2.8229141604 | 1.5611222115 | 执行完成；tracking FAIL |
| 004 FIXED_GRID | 6.0000000000 | 6.0000000000 | 未执行 | — | — | t1 probe 前 hard_contact_loss/context invalid |
| 005 PROPOSAL_GUIDED | 6.0000000000 | 6.0000000000 | 未执行 | — | — | t1 probe 前 hard_contact_loss/context invalid |
| 006 STRATIFIED_CONTINUOUS | 5.9870050571 | 5.9870050571 | 未执行 | — | — | t1 probe 前 hard_contact_loss/context invalid |
| 007 UNIFORM_CONTINUOUS | 5.8816949690 | 5.8816949690 | 未执行 | — | — | t1 probe 前 hard_contact_loss/context invalid |

因此“exact-float 4/8”的准确解释是：4/8 cases 有完整 branch result；这
4 个的 request→controller exact float 都 PASS，但只有 1/4 满足冻结的物理
tracking 阈值。t1 的 4 个 case 根本没有到达 controller branch，不能记为
exact-float PASS 或 FAIL。

small-block audit 为 FAIL：`observed_rollouts=4!=8`、`exact_float_match=false`
（gate completeness）、`tracking_parity=false`，且 evaluator 另有 tracking
threshold violation。4 个 t0 文件可以保留，但不能作为正式 E7 accepted branches。

## 3. 5 个 off-grid controller tests

此前的 manifest 是
`p3_offgrid_controller_test_20260903/P3_OFFGRID_CONTROLLER_TEST_MANIFEST.json`，
请求 force 为 `3.5, 4.5, 5.5, 6.5, 7.5 N`，context 为
`p5s0c_dev_t1_r06_s5106_mid_mu0.515227`。

对应 live output 没有任何 per-force branch row：collector 在共同的 P4B probe
阶段就因 `hard_contact_loss` 结束，`CONTEXT_INVALID_PROBE`，所以 5 个 case
没有 actual measured-force/controller execution 结果。它们解释了 small block
t1 半块缺失的接触/运行时问题，但不证明 t0 的 request→controller 浮点发生了
mismatch。

## 4. full online partial 的处置

full audit 明确为 FAIL，包含：

- 32/144，而不是完整 144；
- controller exact-float mismatch / controller setpoint mismatch；
- telemetry identity mismatch；
- tracking samples missing/nonfinite，以及 MAE/max error 超阈值。

这些 32 个 telemetry/branch 文件保留为 diagnostic/quarantine，正式 E7
有效计数为 **0/144**。不能只因 task0 已有 32 行就补写剩余 112 行，也不能
把 prediction/interpolation accuracy 代替 online controller parity。

## 明确回答

**A. 已有 E7 online branches 能否保留？**

能保留原始文件作诊断和复现线索，但当前不能作为正式 accepted branches，不能
进入 E7 科学表格。

**B. exact-float 还剩什么问题？**

request→controller 在已执行的 4 个 t0 case 是精确一致的；尚缺 t1 的 4 个
真实执行证据。已执行 4 个中还有 3 个 physical force tracking 超阈值；full
partial 还暴露 telemetry identity 和 missing/nonfinite tracking integrity
问题。不能据此宣称 8/8 parity。

**C. 修复后重跑范围？**

按当前 evidence，不能只补 full 的 112 个：已有 32 个未通过正式完整性验收。
应先完成新的 8-case small gate；small gate PASS 后，以 fresh immutable
lineage 运行正式 144，现有 32 只作诊断保留。此结论不授权现在启动任何作业。

**D. E5/E3 释放 GPU 后的最小动作？**

先做 protected-process、GPU、disk 的 fresh preflight；只运行冻结的 fresh
8-request exact-float small-block wrapper，确认 8/8、state parity、tracking
parity、telemetry/evaluator integrity 全 PASS 后，才讨论 144。当前不重启 144。
