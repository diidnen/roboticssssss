# E3 RESET / HDF5 VALIDITY AUDIT

审计范围：E3 `libero_10/task5`，`black_book_1 -> desk_caddy_1`，当前
`P1_SIMPLIFIED_COLLECTION_DIRECT_20260903` 输出。只读审计；未终止、抢占或修改
E3/E5/π0 进程，也未启动新 GPU 作业。

## 结论

`E3_RESET_STATUS = VALID_BUT_DIFFERENT_FROM_EXPECTED_HDF5`

这里的“VALID”仅指已经生成的分支在 seeded default reset 下具有可证实的
matched-force reset 结构；它不表示满足正式 E3 所要求的 HDF5/initial-state
source。现有结果必须保留为 diagnostic/quarantine，不计入正式 72 branches。

## 1. 预期 reset source

冻结计划要求每个 root 先形成固定的 post-reset/pre-force context，再在同一
context 上比较八个 force sibling。E3 collector 通过 B5 client 的
`HDF5_TRAJ_SOURCE_DIR` 查找 `libero_10_task5_*_demo.hdf5`，若找到 episode
initial state，则调用 `env.reset_to(initial_state, ..., is_relative=True)`。

当前项目的 canonical exact-task assembled source 是：

`/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_ASSEMBLED_20260902_113000/replayed_demos/libero_10_task5_book_caddy_onboarding_5demo_7dpf_demo.hdf5`

其 lineage 记录的 SHA-256 为
`b687ccb5b2c1d2dd06d86ec4caa9a2b6c9bf51c19a52dfa1d335fa78e5316989`。

## 2. 实际配置与原因

E3 launcher 设置的 source folder 是：

`/media/volume/newdata/exouser/tabero/data/Isaaclab_Libero/assembled_hdf5`

该目录存在，但没有任何 `libero_10_task5_*_demo.hdf5`。其中的 task5 文件是
`libero_object_task5_pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5`，
不是 book-to-caddy task，不能替代 canonical source。

因此当前不是 canonical 文件本身不存在，而是：

1. launcher 指向了不包含 exact `libero_10_task5` 文件的目录；
2. launcher 只传了 `--seed root`，没有传 root→episode/initial-state manifest；
3. collector/client 的 fallback 是 fail-open：glob 失败后继续 `env.reset()`，而不是拒绝该 branch。

日志中的
`No valid HDF5 file found for libero_10_task5, will use default reset for all experiments`
与上述代码路径一致。

## 3. fallback 的实际含义

“default reset”实际是 B5 client 的裸 `env.reset()`，不是 HDF5 episode
initial state，也不是已证明的官方 HDF5 reset。它不是每个 force 都重新随机到
不同状态：通过 `--seed root` 得到的 post-reset state 是按 root deterministic
的 seeded default state。

已生成 preprobe hash 的证据如下（state digest / state file SHA 均为每个 root
唯一值）：

| root | branches observed | state digest 前缀 | state SHA 前缀 |
|---|---:|---|---|
| 7700 | 8 | `ec8223ba5538...` | `8f4aaa5b479b...` |
| 7701 | 8 | `9a14f59fa198...` | `24cb260071ad...` |
| 7702 | 8 | `a98cce6587bd...` | `3bf8abb9fced...` |
| 7703 | 8 | `e4d9ed86e44b...` | `65f52389dd9b...` |
| 7704 | 4 completed plus F5 preprobe | `bdeed35f799e...` | `d52767ff27a9...` |

所以：

- 不同 root 的 initial state 实际不同；
- 同一 root 的不同 force 实际使用了相同 state digest 和 state SHA；
- 这证明了 seeded default reset 的 matched sibling identity；
- 但没有任何 artifact 证明它来自 canonical task5 HDF5 episode/manifest。

## 4. task identity 与 root=7703 特查

所有已完成 branch 的 argv、`branch_result.json` 和 provenance 均显示：

- suite/task：`libero_10/task5`；
- object：`black_book_1`；
- target：`desk_caddy_1`；
- instruction：`pick up the book and place it in the back compartment of the caddy`。

`root=7703, force=5 N` 与 `force=6 N` 的实际 preprobe hash 完全相同：

- state digest：`e4d9ed86e44b5e778e4524fe79fe2f4beae8b4b87d31dd89db1ea5af87a2a03b`；
- state SHA：`65f52389dd9b6b55b0b5183cfa934cfbfc77f86169c7f4c5ed750f268db39efc`。

两者 task identity 也相同。F5 为 `y_lift=1, y_full=1`，F6 为
`y_lift=1, y_full=0`。因此这两个 outcome 确实来自同一个 matched seeded
default state；不能把它们解释成 force 间 reset 差异。但它们仍不是正式
HDF5-backed E3 数据。

## 5. 当前数量与处置

在本次只读快照中，已有 36 个 `branch_result.json`：7700–7703 各 8 个，
7704 的 F1–F4 各 1 个；7704 F5 只有 preprobe，collector 仍在该 branch
运行，不能把它当完成结果。用户所说的约 30 个均包含在这批 fallback 输出中。

不删除任何文件。现有 36 个 branch 标记为 diagnostic/quarantine，正式
coverage 记为 0/72，而不是把它们纳入正式 matched comparison。

## 最小修复

在不重训 π0、不修改 task、Direct、Utility、controller 或正式实验定义的前提
下，后续正式 collection 只需：

1. 绑定 canonical exact-task HDF5，并提供显式 root→episode/initial-state manifest；
2. 在 collector 启动前 fail-closed 检查 exact file/name/episode initial state；
3. 重新收集已经受 fallback reset 影响的分支，再继续剩余 branches。

当前 F5 可先自然完成；之后应暂停 E3 collection，不能继续把 fallback reset
产生的分支当作正式 72。

## Runtime HDF5 preflight addendum (2026-09-03)

The historical fallback collection above remains diagnostic/quarantined. A fresh
single-root runtime preflight was then run against the repaired exact source:

`/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_SOURCE_20260902_103300/assembled_hdf5`

For root `7700`, all eight force branches completed and recorded
`reset_source=HDF5 initial_state`, `episode_index=0`, the locked task identity,
and identical preprobe hashes:

- `state_digest=f012c07d5d73cde799d1082dbb3c3831fc4b15414c4d8af047e9355967cc0c88`
- `state_sha256=a31c5323c7ad895e47edd6c14b9f3c4855fd1c788190caaae6acc4a785b949ee`

The CPU aggregate audit passed for 8/8 force siblings. The initial aggregate
attempt incorrectly treated rendered image-file bytes as reset identity; that
lane-local check was corrected to use the preprobe state hash, while retaining
the image artifacts. The runtime preflight is therefore:

`E3_RUNTIME_PREFLIGHT = PASS`

The formal collection was started from a new output root
`P1_SIMPLIFIED_COLLECTION_FORMAL_20260903`, from `0/72`; it does not reuse the
old diagnostic branches. No controller, Direct, Utility, evaluator, or policy
definition was changed.
