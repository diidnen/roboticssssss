# LEGACY_E1_REPLAY

独立的历史数值重放模式。不修改CLEAN_CURRENT_PIPELINE，不训练，不启动Isaac，不重写历史结果。

准备并锁定现有历史文件（首次执行一次）：

```bash
/usr/bin/python3 /home/exouser/FORTE/LEGACY_E1_REPLAY/freeze_bundle.py
```

每次重新从保存的raw probe和冻结checkpoint推理：

```bash
/usr/bin/python3 /home/exouser/FORTE/LEGACY_E1_REPLAY/replay.py
```

每次生成一个新的runs/replay_*目录，包含REPLAY_RESULT.json、完整曲线、288个E1决策、七个golden cases以及三个旧输入回归结果。
入口首先校验MANIFEST中所有文件的SHA256；文件改变即拒绝继续，不会自动换checkpoint、重训、修改normalization或utility来凑结果。

三个测试严格分开：

1. E1主表：72个原始215行probe，OOF三成员mu取均值，原五候选和低力tie-break；目标mean4.0627713414。
2. E1 posterior ablation：同样历史输入，对三mu支持的概率求均值；目标mean4.0720132562。
3. LEGACY_407_452_419：当时保存的三个206行probe，加当时带bug的prefix，目标HIGH/MID/LOW=4.07/4.52/4.19。Bug只在这个隔离回归中保留。

历史E1实际feasibility特征使用strict-preprobe masked state。branch首步实测state是另一训练链路的输入，不能为了“恢复旧版”误塞进E1。
真实历史probe已经执行了10个向外步骤；重放读取这些保存行，不会补造当前206行probe缺失的步骤。
归档物理runner保存在source_evidence，仅作为D_CLOSED handoff等历史来源；此模式没有物理入口，也不声称重新获得历史success rate。

使用原有E1 checkpoint，故不涉及重新采样或optimizer更新。其原始训练源码只作为provenance保留，不把现有clean训练recipe替换进去。
本模式通过标准仅为历史特征/概率/选点parity，不采用新流程的calibration资格门槛。
