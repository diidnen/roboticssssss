# Frozen VLA + ActiveForcing：最终实验说明

实验与消融已完成，本文档解释已验收结果。最终交付审计状态以 FINAL_REQUIRED_STATUS.json 和最终证据清单为准。

1. **VLA 是否真正在线运行？** 是。正式 192 条 rollout 共有 6,720 次真实在线 inference 请求。原始冻结 Tabero π₀ 预测 50 步、执行 10 步后重新观察并查询；没有 scripted downstream prefix，也没有 action replay。共同起点的首帧 RGB 缓存用于匹配输入，不是缓存动作；后续查询使用实时观测。
2. **AF 到底改了什么？** 从共同 established-grasp state 做 physical probe，形成 58D belief 和连续 posterior，再选择一次抓力设定，由冻结反馈控制器执行。VLA 在线生成机械臂动作，权重不重训。原始 squeeze 不覆盖所选力；原始每指开口意图 ≥0.039 m 时，执行 0.04 m canonical open 和零 squeeze。控制器原有前馈在原始 squeeze 测量 ≥1 N 时将内部参考设为输入 squeeze 的 1.9 倍；所有方法共用此规则。所选力、内部参考和实测力不能混为一谈。
3. **旧 scripted 数据哪些还能用？** V5 结果保留为 controlled-motion 开发验证；647 条标签保留为明确标注的 scripted auxiliary supervision；Local-Lift 历史训练标签亦如此。旧成功数不进入 online VLA 主表或消融表。
4. **feasibility 有哪些 scripted 依赖？** 原 71 维中，7 个 scripted phase 通道删除；6 个运动通道的来源改为动作执行前得到的当前在线 VLA chunk。最终每个序列位置 64 维，序列长度 8。没有保留额外隐藏 stage、waypoint index 或未来执行结果。部分物理观测使用模拟器对象状态，不声称已实现真实机器人感知替代。
5. **phase 重要吗，最终怎么得到？** 647 条数据的 phase 恒定，精确删除归一化后的对应输入列，使所有旧数据预测和 72 个 context 的 planner 选力保持一致。最终不使用 phase，不需要 phase proxy。Masked、zeroed、permuted、exact removal 和 retrain 的离线结果单独保存；最终未选用重训版本。
6. **feasibility 是否重训、是否补 VLA 数据？** 去除 phase 的精确派生版本通过在线 VLA 开发资格测试，故没有重新训练、校准或采集 VLA-matched 训练行（新增 0 行）。这是指定分布内的经验迁移，不能说旧近乎恒定的运动特征已学会任意 VLA 运动的 feasibility。
7. **force boundary 还存在吗？** 是。在开发 root 的 task0 LOW、task1 MID、task5 LOW、task6 LOW 重新观察到 3 N 失败而较高抓力成功，另有在线 Fixed-4 开发结果。它们是匹配 context 的经验边界，不是普适解析阈值，也不是从 scripted 结果推断。
8. **probe 是否让 VLA OOD？** 12 组 probe 前后图像、tactile 和状态检查及开发行为检查支持本次继续执行。正式主实验没有早期 regrasp heuristic 或超过 0.10 m/step 的 arm jump；Fixed-3 有 4 条掉落后的晚期 closing 候选，不能当成确认的 regrasp attempt。不能从这些检查声称完全不存在 OOD。
9. **最终 fresh-root 结果？** 4 个未用于训练、资格验证或调参的 reset roots，共 48 contexts，每种方法 48 条。Fixed-3：16/48；Fixed-4：31/48；Fixed-5：36/48；AF：32/48。AF lift 为 48/48，实测脱离 5/48，平均所选力 4.03125 N，非 release 样本的 branch-equal 实测双侧 squeeze 为 3.54244 N；双侧接触条件下为 4.16133 N。相对 Fixed-5，平均力设定减少 19.375%，但少成功 4 条任务。不能写同等成功率省力，也不能写全面优于 Fixed-4。
10. **消融说明了什么？** 两个 burned roots 的 24 contexts：AF 17/24，Posterior Mean 17/24，Prior / No-Posterior 17/24，Coarse Grid 15/24，Local-Lift 10/24。AF 与 Posterior Mean 的平均设定分别为 3.9583 和 3.8938 N，故未证明 posterior integration 的成功率优势。Prior 保留 probe，不能叫 NoProbe。Local-Lift 虽 24/24 lift，但只有 10/24 完整成功且 12/24 实测脱离，说明 lift 不能代替完整任务标签。可选 Tabero-Neutral、FORTE-style 未运行，不填入替代结果。
11. **论文能写 Frozen VLA + ActiveForcing 吗？** 可以写“从 common established-grasp state 开始，在冻结预训练 VLA 在线生成下游动作时，通过物理探测和 posterior-aware 抓力选择进行适应，且不重训 VLA”。不能扩展到 initial grasp planning、真实机器人部署、任意 checkpoint 泛化、能耗或损伤优势。论文中的性能陈述必须同时报告成功率与抓力折中。
12. **距离最终 4-root 实验还差什么？** 实验已经全部完成；目前收尾工作是最终交付审计和文件打包，不需要追加正式 physics 或重新选择 roots。

原始 checkpoint 路径、SHA256、完整指标、控制分工与所有状态字段见 FINAL_REQUIRED_STATUS.json。原始失败、服务器替换的数值差异、曾停止的首帧匹配失败均保留；未因失败重选 seed 或将 scripted 动作重命名为 VLA。
