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

## P0-1 [open] Stage-5 风险损失与 L2 回归目标对抗，且 `‖ΔB‖` 量级不可控

- **位置**：`resworld_head.py` loss（`loss_plan_risk` / `loss_plan_cvar` / `loss_plan_info`）
- **现象（已由 [DIAG] 实证，2026-08-13 iter~100-200）**：
  - `rf_mean = 6-12`、`rf_occ = 27-52`——风险场量级是 `loss_plan_reg`(2.9) 的 **3~10 倍**
  - 根因链：`L_uncertainty` 把 σ 校准到特征空间误差量级（e² 大 → σ 大）→
    `risk = σ·‖ΔB‖·boost` 直接把 σ 当乘法因子 → 风险场爆炸
  - `r_cmd = 0`（轨迹尚短未及遮挡区）——**一旦轨迹变长穿过遮挡区，
    `loss_plan_risk` 将立即以 ~10-50 量级主导训练，把轨迹强拉离 GT**
- **建议**：
  1. **σ 量级传染已修复**（2026-08-13，B 方案）：`build_risk_field` 中 σ 改用
     `(σ/(1+σ)).detach()`（有界 [0,1) + 阻断梯度传染）——σ 只由校准损失负责，
     风险场仅消费其相对量级
  2. risk 三项权重降到 0.1 量级；或前 2-3 epoch 关闭 `use_risk_plan`（warmup）
  3. 给 `‖ΔB‖` 做移动平均归一化，抑制量级漂移
- **状态**：`partially fixed`（σ 传染已断；待新日志确认 rf 量级回落到 ~1 量级）

---

## P0-2 [open] 风险代理 `‖ΔB‖` 与 MHST 表达目标共享参数，三方梯度冲突

- **位置**：`resworld_head.py` `build_risk_field` + `OcclusionMHSTHead` + loss
- **现象（预期）**：MHST 假设坍缩（ΔB→0，risk→0，Stage-5 形同虚设）或风险主导（轨迹劣化）
- **原因**：同一 `ΔB` 被三个目标拉扯：
  - `L_occ_halluc`：ΔB **大而准**（拟合遮挡暴露内容）
  - `L_div`：ΔB **彼此分离**（多样性）
  - `L_plan_risk`：`‖ΔB‖` **小**（降风险）
  模型的"作弊"出路是把遮挡区 ΔB 学小 → risk→0 且 MHST 表达力被毁。
- **建议**：风险代理改用**与表达目标解耦**的度量（如 σ 单独驱动，或 ΔB 经 stop-gradient
  后的归一化强度）；或降低 risk 权重使其成为"软正则"而非"主导目标"。
- **状态**：`open`

---

## P1-1 [open] `loss_plan_info` 为负损失、无下界

- **位置**：`resworld_head.py` loss（`info = (r_ego - r_end).mean(); loss = -info`）
- **现象（预期）**：`loss_plan_info` 持续下降（可转负且无下界），轨迹末端被持续推向
  "更低风险"方向，偏离 GT。
- **原因**：`r_end < r_ego` 时（起点处于高遮挡区，常见）`info > 0` → `loss < 0`，
  优化器最小化负损失 → 轨迹末端不断远离遮挡。
- **建议**：改为 hinge 形式 `max(0, r_end - r_ego)`（只惩罚"末端风险高于起点"），
  或 `smooth` 变体；避免负损失无下界。
- **状态**：`open`

---

## P1-2 [open] MHST 随机初始化在训练早期污染遮挡区特征

- **位置**：`resworld_head.py` `OcclusionMHSTHead`（`_init_layers`）
- **现象（预期）**：训练早期规划指标劣化
- **原因**：prior/backbone/experts/sigma_net **随机初始化、无预训练、无 warmup**；
  早期遮挡区 `patch = pred_bev + Σπ·ΔB` 带随机扰动（‖ΔB‖~0.5），`col_attn` 在
  遮挡区采样到噪声特征。
- **建议**：expert 输出层 zero-init（或小方差初始化）；或 warmup 期遮挡区暂退化为
  确定性（门控权重从 0 线性升到 1）。
- **状态**：`open`

---

## P1-3 [open] 在线掩码（`use_osz_rcsample=True`）质量依赖模型自身深度

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
- **状态**：`on-hold`——约束实现完成但**默认关闭**（`use_osz_drivable=False`），
  避免 midas npz 导出窗口期的样本间数据冲突；待 `data/osz` 用 `--use_drivable`
  导出完成后置 True 再验证 `occ_frac` 回落

---

## P1-4 [open] 暴露损失梯度经 `pb` 反传，扭曲世界模型表征

- **位置**：`resworld_head.py` loss（`b_hat = pb + Σπ·ΔB`，`b_k = pb + ΔB`）
- **现象（预期）**：世界模型 latent 表征被"重建 next BEV"目标拉偏，弱化规划服务
- **原因**：`L_occ_halluc` / `L_uncertainty` 的梯度经 `pb`（tokenfuser 输出的确定性
  未来 BEV）反传到 ResWorld 整个 latent 残差路径；且监督范围是**当前帧整个遮挡区**
  （很多格子 next 帧仍未暴露，无有效真值 → 监督噪声大）。
- **建议**：暴露监督对 `pb` 分支 `stop_gradient`（只训练 MHST 的 ΔB/π/σ）；或用
  next 帧真实暴露判定（加载 next 帧掩码）缩小监督范围。
- **状态**：`open`

---

## P2 [open] 次要问题

| # | 问题 | 位置 | 建议 |
|---|---|---|---|
| P2-1 | CVaR 的 `topk` 未应用 `fut_w`，无效轨迹步的风险被计入尾部 | `resworld_head.py` loss | `r_cmd` 乘 `fut_w` 后再 topk |
| P2-2 | `grid_sample(align_corners=False)` 半格偏移，与 deformable attention 采样约定差 ~0.5 格 | `resworld_head.py` loss | 统一坐标约定或接受（小偏差） |
| P2-3 | σ/π 多目标竞争：`L_plan_risk` 压 σ/π 小，`L_uncertainty`/`L_occ_halluc` 校准 σ/π | `resworld_head.py` loss | 明确 σ/π 的主监督目标，risk 项对 σ/π 分支 stop_gradient |
| P2-4 | `L_div` 余弦相似度可负，假设负相关时 `loss_div<0` 无下界 | `resworld_head.py` loss | 权重已 0.1，影响有限；可 clamp 到 [0,1] |
| P2-5 | `L_occ_gt` 栅格化用轴对齐矩形近似旋转框 | `resworld_head.py` `_rasterise_boxes_bev` | 可接受（近似），或改旋转矩形栅格化 |

---

## 验证清单（等训练日志）

- [ ] `loss_plan_risk / loss_plan_cvar / loss_plan_info` vs `loss_plan_reg` 相对量级（P0-1）
- [ ] `grad_norm` 是否频繁触顶 35（P0-1）
- [ ] `loss_occ_halluc` 冷启动值是否远大于其他损失（P1-4）
- [ ] `loss_plan_info` 是否持续下降转负（P1-1）
- [ ] 对照 `abl_baseline` 臂的 L2 曲线（全局基准）
