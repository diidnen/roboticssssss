# Vision vs Structured Context：同 task 新 physical root 机制诊断

## 结论先行

**分类：`CURRENT_STRUCTURED_CONTEXT_INSUFFICIENT`。** 当前冻结的 11-D structured grasp context 没有解决 root-heldout 退化：它在 0/3 个 task 上同时满足 NLL 更低、frontier MAE 更低且 under-force 不恶化；只有 task5 相对 Full Visual 更好，但仍不如 Base。Structured Joint 在 0/3 个 task 上取得安全约束下的独立增益。

因此，本轮不支持“这些已记录低维 grasp geometry 就是正确表示”。同时，本轮也没有证明 generic vision 只是 sample-limited；它目前表现为 **TRAIN signal + held-root 退化，即当前六个 root-seed 家族下的 root memorization**。下一步应先冻结新的 untouched TEST，再从 S30 开始做 independent-context scaling，而不是直接采用 structured，也不是加网络容量。

## 六个问题的直答

### 1. 不同 root 到底主要哪里不同？

每个 task 有 18 个 friction-conditioned physical contexts，但只有 6 个 root-seed 家族。跨 root 最明显的已记录变化是 EEF/scene 的毫米级平移、object/EEF relative pose、gripper opening/finger asymmetry 与 friction；第一段 H8 全是 `branch_hold`，命令位移数值上近零。task0/task5 的 relative geometry 变化多为亚毫米，task1 更大，但缺少 EEF quaternion、真实 contact microstate 与 COM offset，因此 audit 不是完整接触几何描述。

特别地，task0 root06 与 root07 的 F*0.8 分别是 4.25 N 与 3.75 N，但 horizontal eccentricity 仅差 0.016 mm、vertical offset 仅差 0.003 mm、opening 仅差 0.016 mm、friction 仅差 0.0065，H8 都近零。记录到的 grasp offset / motion 解释不了 0.5 N 差值；未记录的 contact microstate 与有限 repeats 的随机性仍是合理解释。

### 2. 不同 grasp geometry 是否真的对应不同 F*0.8？

描述性相关存在，但不是稳定机制证据。TRAIN 上 horizontal eccentricity 与 F* 的 Spearman ρ：task0=0.637、task1=0.654、task5=0.715；控制 friction 后的 partial Pearson 分别为 0.734、-0.024、0.413，方向在 task1 翻转。更关键的是，把这些变量一起交给 oracle Structured 模型后仍未在任何 task 击败 Base。结论：部分几何变量与 F* 共变，但当前样本无法证明它们在同 friction 下稳定解释 root frontier。

### 3. generic vision 的 TRAIN gain 到底是不是 root memorization？

**在当前数据规模下，是。** Full Visual 的 TRAIN NLL 均低于 Base，但 root-heldout NLL 在 task0/1/5 均更差：0.215/0.247/0.231 对 0.163/0.161/0.158；TRAIN→held gap 也从 Base 的 0.018/0.041/0.058 增至 0.120/0.191/0.161。这是当前 regime 的 memorization 诊断，不是“vision 永远无效”的证明。

### 4. 低维 grasp context 能不能更稳定地预测新 root？

**不能。** Structured 的 held-root NLL 为 0.317/2.353/0.182，Base 为 0.163/0.161/0.158；frontier MAE 为 0.367/0.183/0.242 N，Base 为 0.191/0.202/0.200 N。task1 虽 frontier MAE 略低，但 decision coverage 降至 0.578、under-force 升至 0.889，且 NLL 崩溃，不能算成功。该结果甚至来自含 simulator-GT object pose 的 oracle vector，因此更不能直接形成 deployable claim。

### 5. physics Joint 在 structured context 下还有没有独立价值？

**没有稳定独立价值。** Structured Joint 相对 Structured 的 NLL 在 task0/1/5 分别恶化 +0.380/+0.727/+0.067；under-force 分别恶化 +0.056/+0.111/+0.167。task5 frontier MAE 改善 0.023 N，但安全与 NLL 同时变差，不通过预定规则。分类为 `NO_STABLE_INDEPENDENT_STRUCTURED_JOINT_VALUE`。

### 6. 下一步应该增加 independent visual roots，还是直接采用 structured？

**不要采用当前 structured 作为方法；先做冻结的 independent-context scaling。** 在训练选择冻结前创建新的 8–12 roots/task untouched TEST（优先 task0/task5），然后只扩到 S30，检查 Full Visual 是否在新 TEST 上持续改善；通过后再按 S18⊂S30⊂S50⊂S80 扩展。当前不能宣称 `VISUAL_CONTEXT_IS_SAMPLE_LIMITED`，也不能宣称 generic vision 已被证明为低效表示；学习曲线才是区分 A/B 的必要证据。

## 任务级结果

| task | model | NLL | prob MAE | Brier | frontier MAE (N) | under-force | decision coverage | TRAIN→held NLL gap | root groups > Base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Base | 0.163 | 0.121 | 0.049 | 0.191 | 0.444 | 1.000 | 0.018 | 0.000 |
| 0 | Full Visual | 0.215 | 0.119 | 0.062 | 0.226 | 0.667 | 1.000 | 0.120 | 0.167 |
| 0 | Joint-NoVisual | 0.190 | 0.067 | 0.045 | 0.236 | 0.778 | 1.000 | 0.129 | NA |
| 0 | Structured | 0.317 | 0.163 | 0.100 | 0.367 | 0.556 | 1.000 | 0.208 | 0.167 |
| 0 | Structured Joint | 0.697 | 0.135 | 0.116 | 0.384 | 0.611 | 1.000 | 0.639 | 0.000 |
| 0 | Visual Joint | 0.532 | 0.099 | 0.090 | 0.359 | 0.833 | 1.000 | 0.471 | 0.000 |
| 1 | Base | 0.161 | 0.107 | 0.048 | 0.202 | 0.744 | 0.778 | 0.041 | 0.000 |
| 1 | Full Visual | 0.247 | 0.104 | 0.065 | 0.231 | 0.811 | 0.944 | 0.191 | 0.500 |
| 1 | Joint-NoVisual | 0.267 | 0.052 | 0.044 | 0.202 | 0.811 | 1.000 | 0.207 | NA |
| 1 | Structured | 2.353 | 0.201 | 0.166 | 0.183 | 0.889 | 0.578 | 2.282 | 0.333 |
| 1 | Structured Joint | 3.079 | 0.198 | 0.191 | 0.312 | 1.000 | 0.744 | 3.042 | 0.333 |
| 1 | Visual Joint | 1.073 | 0.161 | 0.149 | 0.425 | 0.878 | 0.944 | 1.037 | 0.167 |
| 5 | Base | 0.158 | 0.103 | 0.045 | 0.200 | 0.611 | 1.000 | 0.058 | 0.000 |
| 5 | Full Visual | 0.231 | 0.116 | 0.070 | 0.279 | 0.778 | 1.000 | 0.161 | 0.333 |
| 5 | Joint-NoVisual | 0.302 | 0.060 | 0.050 | 0.210 | 0.833 | 1.000 | 0.266 | NA |
| 5 | Structured | 0.182 | 0.103 | 0.055 | 0.242 | 0.778 | 0.944 | 0.103 | 0.333 |
| 5 | Structured Joint | 0.249 | 0.057 | 0.046 | 0.219 | 0.944 | 1.000 | 0.215 | 0.333 |
| 5 | Visual Joint | 0.286 | 0.068 | 0.052 | 0.253 | 0.944 | 1.000 | 0.257 | 0.333 |

数值为 3-fold root-heldout 的 fold-macro 均值；每 fold 整体留出 2 个 root IDs（6 个 friction-conditioned contexts）。Lower is better，under-force 是安全侧错误。

## Structured Joint 机制比较

| task | task_name | StructuredJoint_minus_Structured_NLL | StructuredJoint_minus_Structured_frontier_MAE_N | StructuredJoint_minus_Structured_under_force_rate | VisualJoint_minus_FullVisual_NLL | VisualJoint_minus_FullVisual_frontier_MAE_N | VisualJoint_minus_FullVisual_under_force_rate | JointNoVisual_minus_Base_NLL | StructuredJoint_better_NLL_and_frontier_underforce_nonworse | VisualJoint_worse_NLL_than_FullVisual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | alphabet soup | 0.380 | 0.016 | 0.056 | 0.317 | 0.133 | 0.167 | 0.028 | False | True |
| 1 | cream cheese | 0.727 | 0.129 | 0.111 | 0.826 | 0.195 | 0.067 | 0.106 | False | True |
| 5 | tomato sauce | 0.067 | -0.023 | 0.167 | 0.055 | -0.026 | 0.167 | 0.143 | False | True |

## 物理相关性（仅描述）

| task | feature | n_contexts | n_root_seed_groups | spearman_rho | linear_R2 | partial_pearson_controlling_friction |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | friction | 18 | 6 | -0.593 | 0.404 | NA |
| 0 | horizontal_grasp_eccentricity_m | 18 | 6 | 0.637 | 0.666 | 0.734 |
| 0 | vertical_grasp_offset_m | 18 | 6 | -0.527 | 0.209 | -0.715 |
| 0 | gripper_opening_m | 18 | 6 | -0.655 | 0.665 | -0.769 |
| 0 | finger_joint_asymmetry_m | 18 | 6 | -0.666 | 0.623 | -0.664 |
| 1 | friction | 17 | 6 | -0.868 | 0.746 | NA |
| 1 | horizontal_grasp_eccentricity_m | 17 | 6 | 0.654 | 0.020 | -0.024 |
| 1 | vertical_grasp_offset_m | 17 | 6 | -0.593 | 0.000 | 0.101 |
| 1 | gripper_opening_m | 17 | 6 | -0.828 | 0.904 | -0.791 |
| 1 | finger_joint_asymmetry_m | 17 | 6 | -0.591 | 0.255 | -0.116 |
| 5 | friction | 18 | 6 | -0.903 | 0.781 | NA |
| 5 | horizontal_grasp_eccentricity_m | 18 | 6 | 0.715 | 0.656 | 0.413 |
| 5 | vertical_grasp_offset_m | 18 | 6 | 0.521 | 0.312 | 0.400 |
| 5 | gripper_opening_m | 18 | 6 | -0.728 | 0.568 | -0.509 |
| 5 | finger_joint_asymmetry_m | 18 | 6 | -0.715 | 0.538 | -0.478 |

这些相关性把 18 个 friction-conditioned contexts 列出，但有效独立 root-seed 家族只有 6；p 值和 partial correlation 都不应作独立样本推断，也没有被用于 feature selection。

## Deployment legality

11-D vector 中 6 个维度依赖 simulator GT object pose，因此整体只能标为 `STRUCTURED_CONTEXT_ORACLE_DIAGNOSTIC`。EEF base pose、opening、finger asymmetry 是 legal/derived-legal；object-to-EEF translation 与 object orientation 是 privileged。EEF-relative rotation、EEF quaternion、H8 orientation change 不可用。friction 已在 Base 中，合法前提仍是未来 active probe；本轮没有运行 Probe。

## 复现与限制

- Base 与 Full Visual 的所有 frozen CV comparators 在绝对容差 5e-8 内复现。
- task1/task5 Visual Joint 复现；task0 Visual Joint 的旧 CV 在相同源码/数据哈希与精确调用顺序下仍出现 material mismatch，因此报告采用当前同环境配对重训值，并在 `COMPARATOR_REPRODUCIBILITY_AUDIT.json` 中完整记录。task0 Joint 方向应视为带此 caveat。
- task1 有 reconstructed-label caveat；task5 是 direct-label 锚点。
- 这些 held roots 已看过，只是 retrospective mechanism diagnostic，不是新 TEST。
- 未运行 Probe、未采新 simulator 数据、未修改 encoder/PCA/architecture/optimizer/epoch/λ/force semantics/split/labels。

## 最终建议

`RUN_FROZEN_INDEPENDENT_CONTEXT_SCALING_STARTING_AT_S30_NOT_S80_AND_CREATE_NEW_UNTOUCHED_TEST_BEFORE_TRAINING_CHOICES`

具体执行：先冻结新 TEST；只采更多独立 π0 physical roots，不增加同 root repeats 冒充 context；先完成 S30 gate。若 S30→S50→S80 的新 TEST 表现不持续改善，则接受 generic visual embedding 是低效 physical representation，并转向重新冻结、可合法观测的更完整 contact/grasp state，而不是维护 vision 假设。
