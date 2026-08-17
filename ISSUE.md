# OARWM-Res 已知问题清单（Issue Tracker）

> 本文件记录会导致训练失败或训练结果弱于基线 ResWorld 的问题。
> 每项标注：位置 · 现象 · 原因 · 建议。状态：`open` / `fixed` / `verified`。

---

## P0-0 [fixed] 训练崩溃：`loss_plan_info` 的 `grid_sample` 输入维度错误

- **位置**：`projects/mmdet3d_plugin/resworld/resworld_head.py`（Stage-5 loss，`r_ego` 采样处）
- **现象**（服务器训练日志，4 卡同崩）：
  ```
  RuntimeError: grid_sampler(): expected 4D or 5D input and grid with same number
  of dimensions, but got input with sizes [2, 3, 6] and grid with sizes [2, 1, 1, 2]
  ```
- **原因**：`r_exp` 在上方已被重新赋值为**轨迹逐点采样后的 `(B, M, T)`**（B=2, M=3, T=6），
  但 `r_ego = F.grid_sample(r_exp, ego_grid, ...)` 仍把它当作 4D 风险场输入。
  `grid_sample` 要求 input 与 grid 同为 4D/5D，3D input 直接崩溃。
  （此前修复的 `.expand(B,1,1,2)` 只解决了 grid 的 batch 维，未发现 input 类型错误。）
- **修复**：`r_ego` 改从**原始 `risk_field[:, 0:1]`**（R_exp 通道，`(B,1,H,W)`）采样 ego 原点，
  与轨迹采样保持同一坐标归一化约定。
- **状态**：`fixed`（待服务器复跑验证）

---

## P0-1 [fixed] Stage-5 风险损失与 L2 回归目标对抗，且 `‖ΔB‖` 量级不可控

- **位置**：`resworld_head.py` loss（`loss_plan_risk` / `loss_plan_cvar` / `loss_plan_info`）
- **现象（已由 [DIAG] 实证，2026-08-13 iter~100-200）**：
  - `rf_mean = 6-12`、`rf_occ = 27-52`——风险场量级是 `loss_plan_reg`(2.9) 的 **3~10 倍**
  - 根因链：`L_uncertainty` 把 σ 校准到特征空间误差量级（e² 大 → σ 大）→
    `risk = σ·‖ΔB‖·boost` 直接把 σ 当乘法因子 → 风险场爆炸
  - `r_cmd = 0`（轨迹尚短未及遮挡区）——**一旦轨迹变长穿过遮挡区，
    `loss_plan_risk` 将立即以 ~10-50 量级主导训练，把轨迹强拉离 GT**
- **修复（2026-08-13 + 2026-08-14）**：
  1. σ 量级传染已断（B 方案）：`build_risk_field` 中 σ 改用
     `(σ/(1+σ)).detach()`（有界 [0,1) + 阻断梯度传染）
  2. risk 三项权重降到 0.1 量级（`loss_plan_risk/cvar = 0.1`、`loss_plan_info = 0.05`，
     约为 `loss_plan_reg` 权重 10 的 1%），风险项成为软正则而非主导目标；
     另有 `risk_plan_warmup_epochs=2` 冷启动关闭
  3. `build_risk_field` 全量 detach（见 P0-2）后 ‖ΔB‖ 不再被 risk 项拉扯，
     量级由 Stage-6 损失自然约束
  4. **risk ramp（2026-08-14）**：`risk_plan_ramp_epochs=2`——warmup 结束后 risk
     权重线性 0→1 爬升。硬切换曾使 `loss_plan_reg` 在激活 epoch 从 0.53 跳 1.24；
     ramp 后平缓过渡。**预期权衡**：激活后 `loss_plan_reg` 略高于
     `loss_plan_reg_init`（~5%，risk 梯度只作用 final 轨迹路径）是设计预期的
     "保守 vs 模仿"代价，最终以评估 L2/碰撞率为准
- **状态**：`fixed`（待服务器新日志确认 rf 量级回落 + `loss_plan_risk` 与
  `loss_plan_reg` 相对量级 ~0.01-0.1 + reg/init 差距稳定 ~5% 无跳变）

---

## P0-2 [fixed] 风险代理 `‖ΔB‖` 与 MHST 表达目标共享参数，三方梯度冲突

- **位置**：`resworld_head.py` `build_risk_field` + `OcclusionMHSTHead` + loss
- **现象（预期）**：MHST 假设坍缩（ΔB→0，risk→0，Stage-5 形同虚设）或风险主导（轨迹劣化）
- **原因**：同一 `ΔB` 被三个目标拉扯：
  - `L_occ_halluc`：ΔB **大而准**（拟合遮挡暴露内容）
  - `L_div`：ΔB **彼此分离**（多样性）
  - `L_plan_risk`：`‖ΔB‖` **小**（降风险）
  模型的"作弊"出路是把遮挡区 ΔB 学小 → risk→0 且 MHST 表达力被毁。
- **修复（2026-08-14）**：`build_risk_field` 开头对 pi / σ / ΔB / s_occ **全量 detach**——
  风险场成为分布的无梯度**测量**（gradient contract 写入 docstring）：分布只由 Stage-6
  自己的损失（似然/校准/多样性/占用）塑造，规划器经 `grid_sample` 的 grid 支路获得
  风险场空间梯度、学习**避开**风险而非**篡改**风险。同时解决 P2-3 的 σ/π 竞争。
- **状态**：`fixed`（待服务器验证：`loss_div` 维持非零、`rf_occ` 非平凡）

---

## P0-3 [fixed] 掩码轴序转置错位：OSZ axis-0=x vs BEV 特征 axis-0=y

- **位置**：`resworld_head.py::_align_osz_mask`（Stage 2/3/4/6 所有掩码消费点：注入、MHST 门控、风险场、暴露监督）
- **现象（已由 [DIAG] 实证，2026-08-13，训练日志 epoch 1-4）**：
  - `r_cmd_mean=0.00000`、`r_cmd_pos_frac=0.000`——约 800 次迭代中仅十几次非零，
    **轨迹从未采到正风险**，而同一日志 `rf_occ=0.2~2.7`（风险场明显非零）→ Stage 5 三项静默失效；
  - `loss_plan_risk / cvar / info` 数值 ≈ 0.0000-0.02，权重 1.0/1.0/0.5 形同虚设；
- **原因**：两套网格轴序**互为转置**，注入时未转置：
  - OSZ 掩码 `(3, 200, 200)`：**axis-0 = ego-x**（前进，0.15 m/cell）、axis-1 = ego-y
    （`OSZ/config.py`、`bev_height_builder.py` xi→axis-0 / yi→axis-1，在线
    `build_osz_mask_online` 同约定）；
  - ResWorld BEV 特征 `(B, C, H, W)`：**axis-0(H) = ego-y、axis-1(W) = ego-x**
    （`rcsample.py::create_grid_infos` 的 `y_coor, x_coor = meshgrid(x_coor, y_coor)`
    变量名与内容相反，`bev_coor=stack([x_coor, y_coor])` 使 x 沿 W 变化；col_attn
    `spatial_shapes=(bev_w, bev_h)` 按此采样）；
  - 同一世界点 (x=5, y=3)：OSZ 掩码 `(axis0=133, axis1=90)`，特征图 `(row=90, col=133)`
    ——掩码需 `M.T` 对齐，`_align_osz_mask` 只 `interpolate` 不转置 → 90° 错位。
- **后果链**：Stage 2 遮挡嵌入注入到非遮挡格（真实遮挡格未被处理，特征被污染，**劣于基线的最大风险源**）；
  Stage 3 多假设作用在错误格；Stage 4 风险场错位（轨迹按正确坐标采样 → r_cmd=0）；Stage 6 暴露监督错位
  （loss 仍下降，因为在错误位置拟合，曲线看似正常）。
- **修复（2026-08-13）**：`_align_osz_mask` 先 `transpose(-1,-2)` 再插值（转置后 y 向 0.3→0.6、x 向 0.15→0.3 m/cell，
  与特征图恰好一致）。**OSZ 保持 axis-0=x 约定不变**——勿"修正"OSZ 侧，否则双重转置复现 bug（docstring 已固化契约）。
  配套：`loss_depth_weight` 0.1→0.3（在线掩码质量依赖深度头）；`risk_plan_warmup_epochs=2`（前 2 epoch 关风险项，
  见 P0-1/P1-2 冷启动）。
- **状态**：`fixed`（待服务器重训验证：`r_cmd_pos_frac` 应显著上升、`loss_plan_risk/cvar/info` 有真实量级）

---

## P0-5 [fixed] 训练爆炸实录（2026-08-13）：僵尸假设 + `R_worst = max_k` 无界级联

- **位置**：`resworld_head.py`（`OcclusionMHSTHead` 的 ΔB/σ 无上界 + `build_risk_field` 的 `R_worst = max_k R^(k)` + Stage-5 风险项）
- **现象（服务器训练日志 2026-08-13 12:47 起，代码 = 4a26e25 含 P1-4/P0-2 全部修复；epoch 1 iter 3200 突变，此后指数爆炸）**：
  - `loss_uncertainty`: 0.07–0.6（iter 100–3100 正常）→ 43666（3200）→ 3.6M → 82M → 495M → epoch 2 达 1e13–1e16
  - `rf_mean/rf_occ`: 2–6 / 8–20 → 11866 / 37312 → 5e6 / 1e10；`grad_norm` 18–48 → 1348 → 1e10
  - 连带：`loss_plan_reg` 0.6 → 42 → 18099（轨迹被拉爆）；`occ_frac` 0.44 → 0.9（深度头被污染）
- **排除项（重要）**：初判"e² 支路"不成立——4a26e25 中 e² 已 detach，`loss_uncertainty` 对 ΔB 无梯度。
- **机制（由日志数据自洽性反推，2026-08-13 晚确认）**：
  1. **僵尸假设**：K 个假设中，后验权重 $w_k = \text{softmax}(\log\pi_k + \log p_k)$ 趋零的假设，
     其 `L_occ_halluc` 梯度 $\propto w_k$ → 该假设的 **ΔB 失去曝光似然监督**；
     `L_div`（单位化余弦）与 risk 项（已 detach）对 ΔB **模长均无梯度** → ΔB 模长自由漂移；
  2. **`R_worst = max_k` 无界**：僵尸假设的巨大 ‖ΔB‖ 直接进入 R_worst（Minimax 语义不乘 π）→ rf 无界。
     日志自洽性验证（iter 3200）：`e2 ≈ 4.4e4`（混合误差 ≈209，主导假设正常）而
     `rf_occ = 37312 ≈ 0.09×‖ΔB‖×2` → 僵尸假设 **‖ΔB‖ ≈ 2e5**；
  3. **级联点燃**：rf 巨大 → `loss_plan_risk/cvar/info` 梯度 ∝ rf 值 → 轨迹/col_attn/pred_bev 被推乱 →
     `loss_plan_reg` 爆炸 → L1 梯度经 fused 对 π 的梯度 ∝ ΔB（巨大）→ prior 网络被推乱 →
     更多假设变僵尸 → 指数化（grad_norm 1348→1e10 吻合）。
- **修复（2026-08-13，P0-6 硬限幅三件套）**：
  1. `OcclusionMHSTHead`：`delta.clamp(±mhst_delta_clamp=10.0)`（正常每元素 ~1-3，4 倍余量）——
     僵尸假设的 ‖ΔB‖ 有界；
  2. `sigma.clamp(max=mhst_sigma_max=10.0)` + loss 中 `e2.clamp(max=mhst_sigma_max)`——
     σ/var 有界，似然对 ΔB 的约束（梯度 ∝ 1/var）不失效，`loss_uncertainty` 数值有界；
  3. `build_risk_field` 输出 `clamp(max=risk_clamp=100.0)`（正常 rf ~1-20，5 倍余量）——
     风险项梯度有界，级联截断。
  三者均为"正常区零影响、极端区饱和"的硬限幅；Minimax 语义保留（仍是 over-K max，
  只是每个假设的 ΔB 表达被限幅——特征残差有物理量级，限幅合理）。
- **状态**：`fixed`（待服务器重训验证：`loss_uncertainty` 无指数增长、rf 有界、`grad_norm` 回落）

---

## P1-1 [fixed] `loss_plan_info` 为负损失、无下界

- **位置**：`resworld_head.py` loss（`info = (r_ego - r_end).mean(); loss = -info`）
- **现象（预期）**：`loss_plan_info` 持续下降（可转负且无下界），轨迹末端被持续推向
  "更低风险"方向，偏离 GT。
- **原因**：`r_end < r_ego` 时（起点处于高遮挡区，常见）`info > 0` → `loss < 0`，
  优化器最小化负损失 → 轨迹末端不断远离遮挡。
- **修复（2026-08-14）**：hinge 形式 `max(0, r_end - r_ego)`（只惩罚"末端风险高于起点"，
  有下界 0，不再无限推离 GT）；`r_end` 取**最后有效步**（`fut_w` 累加定位），
  不再取可能无效的 padded 末步。
- **状态**：`fixed`（待日志确认 `loss_plan_info ≥ 0` 且不持续发散）

---

## P1-2 [fixed] MHST 随机初始化在训练早期污染遮挡区特征

- **位置**：`resworld_head.py` `OcclusionMHSTHead`（`_init_layers`）
- **现象（预期）**：训练早期规划指标劣化
- **原因**：prior/backbone/experts/sigma_net **随机初始化、无预训练、无 warmup**；
  早期遮挡区 `patch = pred_bev + Σπ·ΔB` 带随机扰动（‖ΔB‖~0.5），`col_attn` 在
  遮挡区采样到噪声特征。
- **修复（2026-08-14）**：expert 输出卷积 **zero-init**（weight/bias 置零）——
  初始 ΔB^k = 0 → 遮挡区 patch == pred_bev，**第 0 步输出与基线严格一致**，
  假设在 Stage-6 损失下逐步生长（等价于"门控从 0 线性升到 1"建议的零参数实现）。
  同源修复：Stage-2 `OcclusionAwareFusion` 亦 zero-init（见 P0-4）。
- **状态**：`fixed`（待日志确认前 1-2 epoch 规划 loss 与基线量级一致）

---

## P1-3 [mitigated] 在线掩码（`use_osz_rcsample=True`）质量依赖模型自身深度

- **位置**：`resworld.py` `forward_train`（`build_osz_mask_online`，no_grad）
- **现象（已由 [DIAG] 实证，2026-08-13）**：`occ_frac = 0.42-0.48`——遮挡格占比接近
  一半，明显异常（正常城市场景应远低于此）。RCSample 深度低估 → 射线投射大面积误报遮挡
- **原因**：主配置掩码由**模型自己的 RCSample 深度**生成；深度损失权重仅 **0.1**，
  训练早期深度不准 → 掩码不可靠。离线 MiDaS 掩码（LiDAR 对齐）质量稳定得多。
- **建议**：warmup 期用离线 MiDaS 掩码（`use_osz_midas`）或提高深度损失权重；
  或在线掩码加"置信度门控"（深度不确定区域掩码置零）。
- **修复（2026-08-13）**：在线掩码生成已加 **HD-map 可行驶区域约束**——数据集从离线
  npz 加载 `drivable_mask`（与深度源无关），`build_osz_mask_online` 中
  `osz &= drivable`，路外建筑/设施不再计入遮挡（P0-1 的掩码过曝源头）。
  改动：`torch_pipeline.py`/`resworld.py`/`nuscenes_resworld_dataset.py`/`resworld_config.py`
- **过渡措施（2026-08-13）**：drivable 未开期间，`loss_depth_weight` 0.1→0.3
  提升 RCSample 深度头质量（在线掩码的直接依赖）；导出完成后回退 0.1 并置
  `use_osz_drivable=True`。另见 P0-3（掩码轴序转置修复）。
- **状态**：`mitigated`（2026-08-18 主配置切离线 MiDaS）——`use_osz_midas=True / use_osz_rcsample=False`，掩码不再依赖模型自身深度头（`loss_depth` 同步回退 0.1）；在线 rcsample 保留为部署形式与深度来源消融臂。**2026-08-17 日志仍实测 `occ_frac = 0.58-0.86`、`rf_occ ≈ 9-13`, `loss_depth ≈ 0.7`**——即切换前的掩码过曝与深度质量差被最新一轮训练证明确实存在。待 `data/osz` 用 `--use_drivable` 导出完成、离线掩码重训后核对 `occ_frac` 回落与 L2 收敛；若离线掩码下 `occ_frac` 仍偏高，再考虑对在线 rcsample 启用 `use_osz_drivable=True`（`build_osz_mask_online` 已实现 `osz &= drivable`）。

---

## P1-4 [fixed] 暴露损失梯度经 `pb` 反传，扭曲世界模型表征

- **位置**：`resworld_head.py` loss（`b_hat = pb + Σπ·ΔB`，`b_k = pb + ΔB`）
- **现象（预期）**：世界模型 latent 表征被"重建 next BEV"目标拉偏，弱化规划服务
- **原因**：`L_occ_halluc` / `L_uncertainty` 的梯度经 `pb`（tokenfuser 输出的确定性
  未来 BEV）反传到 ResWorld 整个 latent 残差路径；且监督范围是**当前帧整个遮挡区**
  （很多格子 next 帧仍未暴露，无有效真值 → 监督噪声大）。
- **修复（2026-08-14）**：
  1. `b_hat` / `b_k` 中 `pb` 分支 **stop_gradient**（`pb.detach()`）——暴露损失只训练
     MHST 头（π/ΔB/σ），确定性路径不受拉扯；
  2. `e2` 整体 **detach**——`L_uncertainty` 中 σ 被校准到**固定误差目标**，消除
     "σ↔e2 互相拉近"的自证预言循环（原实现 e2 经 ΔB 支路可被拉向 σ，校准失真）。
  3. 监督范围仍为当前帧遮挡区（next 帧暴露判定为后续可选优化，见 P2-8）。
- **状态**：`fixed`（待日志确认 `loss_occ_halluc` 冷启动不再异常主导）

---

## P0-4 [fixed] `OcclusionAwareFusion` 替换式注入同质化污染（新发现，2026-08-14）

- **位置**：`resworld_head.py::OcclusionAwareFusion.forward`（Stage-2 注入）
- **现象（代码审查）**：原实现 `B̃ = B⊙(1-M) + E_occ⊙M` 中 `E_occ` 由 1×1 Conv
  从掩码通道生成——**无空间上下文**，遮挡区所有格子的嵌入几乎相同。当掩码过曝
  （`occ_frac ≈ 0.42-0.48`，见 P1-3）时，近半 BEV 特征被替换为**同质随机向量**：
  col_attn 在遮挡区采样到无差别噪声特征，且 `osz_fusion` 随机初始化使第 0 步即
  劣于基线——**是"不如基线 ResWorld"的最大结构性风险源之一**（与 P0-3 并列）。
- **修复（2026-08-14）**：改为**残差式注入** `B̃ = B + E_occ⊙M` + `osz_embed` **zero-init**：
  - 训练第 0 步 `B̃ = B` 与基线**严格一致**；
  - 保留遮挡区原始特征的空间信息，`E_occ` 只学"遮挡偏移"；
  - 掩码质量差/过曝时退化为温和扰动而非特征抹除。
  - 设计文档公式同步（OARWM_ResWorld.md §2/Stage 2、STATUS.md §1）。
- **状态**：`fixed`（待训练日志对照 `abl_baseline` 臂 L2 曲线）

---

## P1-5 [fixed] `_rasterise_boxes_bev` 的 y 轴翻转（新发现，2026-08-14）

- **位置**：`resworld_head.py::_rasterise_boxes_bev`（L_occ_gt 的占用 GT 栅格化）
- **现象（代码审查）**：`y0 = (y_max - ys.max)/(y_max - y_min)*bev_h` 与 BEV 特征图
  行序**镜像相反**——特征图行 0 = ego-y **最小**（`rcsample.py::create_grid_infos` /
  `bevdet.py::gen_grid` 证实：行 i ↔ y = y_min + i·y_step），而该式把 y 最大映射到行 0。
- **后果链**：`L_occ_gt` 的 GT 占用在 y 方向整体翻转 → `s_occ` 学到镜像占用 →
  Stage-4 占用放大（γ·s_occ·M）放大**错误一侧**的格子 → 鬼探头感知错位，
  `L_occ_gt` 的下降曲线看似正常（在错误位置拟合）。
- **修复（2026-08-14）**：`y0/y1` 改用 `(ys - y_min)/(y_max - y_min)*bev_h`
  （与特征图行序同约定，x 方向本就正确）；docstring 固化轴契约。
- **状态**：`fixed`（待服务器验证：可视化 `occ_gt` 与图像中车辆位置一致）

---

## P2 [open] 次要问题

| # | 问题 | 位置 | 建议/状态 |
|---|---|---|---|
| P2-1 | CVaR 的 `topk` 未应用 `fut_w`，无效轨迹步的风险被计入尾部 | `resworld_head.py` loss | **fixed**（2026-08-14）：`r_cmd * fut_w` 后再 topk |
| P2-2 | `grid_sample(align_corners=False)` 半格偏移，与 deformable attention 采样约定差 ~0.5 格 | `resworld_head.py` loss | **fixed**（2026-08-14）：两处采样改 `align_corners=True`（对齐 col_attn 像素中心语义） |
| P2-3 | σ/π 多目标竞争：`L_plan_risk` 压 σ/π 小，`L_uncertainty`/`L_occ_halluc` 校准 σ/π | `resworld_head.py` loss | **fixed**（2026-08-14）：`build_risk_field` 全量 detach（P0-2），risk 项不再触碰 σ/π/ΔB/s_occ |
| P2-4 | `L_div` 余弦相似度可负，假设负相关时 `loss_div<0` 无下界 | `resworld_head.py` loss | **fixed**（2026-08-14）：`clamp(min=0)`（已分离不奖励）+ 限定遮挡区计算（可见区 ΔB 不进入融合输出，推分假设纯属浪费且无对向监督） |
| P2-5 | `L_occ_gt` 栅格化用轴对齐矩形近似旋转框 | `resworld_head.py` `_rasterise_boxes_bev` | 可接受（近似），或改旋转矩形栅格化 |
| P2-6 | `_align_osz_mask` 用 `nearest` 做 200→100 下采样，掩码边界系统性偏移 0.5-1 格 | `resworld_head.py::_align_osz_mask` | open：可改 `adaptive_max_pool2d`（保守放大语义）或先下采样后转置；影响限于边界格，低优先级 |
| P2-7 | `occ_head` 随机初始化 → `sigmoid≈0.5` → 训练早期风险场被常数 `(1+γ·0.5·M)` 放大，占用放大无意义 | `resworld_head.py` `build_risk_field` / `occ_head` | open：影响小（常数缩放，且已 detach）；可 zero-init occ_head 使初始放大为 1 |
| P2-8 | 暴露监督范围 = 当前帧整个遮挡区，很多格子 next 帧仍未暴露（无有效真值）→ 监督噪声 | `resworld_head.py` loss | open：需加载 next 帧掩码做真实暴露判定（设计接口，见 P1-4） |

---

## 验证清单（等训练日志）

- [ ] **`loss_uncertainty` 不出现指数增长**（P0-5：4a26e25 在 iter 3200 起 43666→1e16 爆炸；P0-6 限幅后应稳定 ~0.1–1，`rf_occ` ≤ 2×risk_clamp）
- [ ] `loss_plan_risk / loss_plan_cvar / loss_plan_info` vs `loss_plan_reg` 相对量级（P0-1）
- [ ] **`loss_plan_reg` vs `loss_plan_reg_init`**（P0-1 权衡指标：risk 激活后 reg 略高于 init ~5% 属预期；ramp 后激活 epoch 不应再有 0.53→1.24 式跳变）
- [ ] `grad_norm` 是否频繁触顶 35（P0-1/P0-5：爆炸时 1348→1e10，修复后应回落）
- [x] `loss_plan_info` ≥ 0 且不持续发散（P1-1：首轮训练全程 0.0577-0.2054 ✓）
- [x] `loss_div` 维持非零（P0-2：首轮训练 0.0000-0.0036，假设未完全坍缩 ✓）
- [x] `loss_occ_halluc` 冷启动不异常主导（P1-4：首轮 iter100 = 4.27 与 reg 3.18 同量级 ✓）
- [x] 前 1-2 epoch 规划 loss 正常量级（P1-2/P0-4 zero-init 生效：reg 3.18→0.53 平滑下降 ✓）
- [ ] **`traj_end` / `traj_step` 诊断**（2026-08-16 新增：首轮评估 L2 avg≈0.57、CR≈1e-4 呈"过度保守"画像，疑似轨迹缩水；对比 GT 巡航速度 ~2-5 m/step 判断是否系统性缩水）
- [ ] 对照 `abl_baseline` 臂的 L2 曲线（全局基准；P0-4 残差式注入后主臂 L2 不应显著劣于基线）
- [ ] `r_cmd_pos_frac` 显著上升、`loss_plan_risk/cvar/info` 有真实量级（P0-3 转置修复的服务器复验）
- [ ] `occ_frac` 回落到合理水平（P1-3，待 `--use_drivable` 导出完成 + `use_osz_drivable=True`；旧日志爆炸后 occ_frac 一度达 0.9）

## 首轮评估记录（2026-08-16，epoch_12_ema，12 epoch，f84169d 配置）

- 开环规划：`plan_L2_1s/2s/3s = 0.285 / 0.543 / 0.896`（Avg ≈0.57，参考 AVG 0.30 / MAX 0.59）
- 碰撞率：`plan_obj_col_1s/2s/3s ≈ 0 / 1e-4 / 1e-4`（参考 AVG 0.01/0.03/0.14）——**低于参考 3 个数量级**
- 画像：L2 接近参考 MAX + CR 趋零 = **过度保守**（轨迹系统性偏离 GT）。训练日志 reg>init ~5% 的
  risk 拉偏在此兑现；最大嫌疑放大源为 **P1-3 掩码过曝**（occ_frac 0.4-0.8，drivable 未启用）→
  风险场弥漫 → 轨迹被压向少数低风险格。
- 待办：① 跑 `abl_baseline` 同配置对照（判定"劣于基线"的唯一可靠方式）；② 收尾 P1-3
  （drivable 导出完成 → `use_osz_drivable=True` + `loss_depth_weight` 回退 0.1）后重训；
  ③ 若 baseline 对照确认 L2 劣化显著，risk 权重 0.1→0.03-0.05 或延长 ramp；④ 可视化
  val 样本确认轨迹模式。
