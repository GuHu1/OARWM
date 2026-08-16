# OARWM-Res 项目状态

> 最后更新：2026-08（Stage 1-6 全部实现）
> 设计文档：`OARWM_ResWorld.md` · 安装/数据准备：`INSTALL.md`
> 服务器：8×RTX 3090，`resworld` 环境（python 3.8 + torch 1.9.1+cu111 + mmcv-full 1.4.0 + mmdet3d 0.17.1）

---

## 1. 改造总览（按 Stage）

| Stage | 改造内容 | 关键文件 | 开关 |
|---|---|---|---|
| **1 图像/BEV** | 继承 ResWorld（ResNet-50 + RCSample + BEV encoder），未改动 | — | — |
| **2 显式遮挡几何** | OSZ 高度感知射线投射生成遮挡掩码并注入 BEV：`B̃ = B + E_occ⊙M`（残差式，zero-init 起步等价基线，ISSUE P0-4）；双源（离线 npz / 在线同源）；torch GPU 后端；MiDaS v2.1 Small 深度 | `OSZ/`（config/ray_casting/depth_estimator/torch_pipeline 等）、`resworld_head.py::OcclusionAwareFusion`、`nuscenes_resworld_dataset.py` | `use_osz_midas`（离线）/ `use_osz_rcsample`（在线，互斥） |
| **3 多假设 MHST** | 遮挡区 K 假设残差转移：先验网络（多尺度膨胀邻域）+ K expert 分支 + σ 不确定性；门控合成（可见区原值、遮挡区混合）；路径 A 期望输出 + aux 旁路；**P0-6 硬限幅**：ΔB clamp ±10、σ cap 10（僵尸假设防爆，ISSUE P0-5） | `resworld_head.py::OcclusionMHSTHead` | `use_oarwm`、`mhst_k=5`、`mhst_sigma_min=0.1`、`mhst_delta_clamp`、`mhst_sigma_max` |
| **4 风险场** | 语义不可知双风险场（无梯度测量，π/σ/ΔB/s_occ 全 detach，ISSUE P0-2/P2-3）：`R^(k)=w_σ·σ‖ΔB^(k)‖·M` → `R_exp`/`R_worst` → 不确定性+占用驱动放大（π 熵 U + s_occ 占用）；**P0-6 输出 clamp**（risk_clamp=100，风险项梯度有界） | `resworld_head.py::build_risk_field` | `use_risk_field`、`risk_beta/w_sigma/gamma`、`risk_clamp` |
| **5 风险加权规划** | 风险场正则化预测轨迹（方案 A，非候选筛选）：沿程 R_worst（Minimax 语义）+ 沿程尾部 CVaR + 信息增益（hinge：末端风险高于起点，ISSUE P1-1） | `resworld_head.py::loss` | `use_risk_plan`、`loss_plan_risk/cvar/info_weight`、`cvar_beta` |
| **6 端到端损失** | 遮挡暴露混合似然 `L_occ_halluc` + 检测框占用 `L_occ_gt`（pos_weight 防平凡解）+ 多样性 `L_div`（余弦）+ σ 校准 `L_uncertainty` + 规划 `L_plan`（含风险项）+ 信息增益 `L_info`；next 帧独立监督通道 | `resworld_head.py::loss`、`resworld.py::encode_next_bev`、`nuscenes_resworld_dataset.py`（`use_next`）、`loading.py`（next 帧处理） | 各损失权重（0=关） |

**工程与基础**（贯穿各 Stage）：OSZ 网格对齐 ResWorld（200×200 各向异性，单一来源 `OSZ/config.py`）；`use_osz`/`use_oarwm` 条件实例化（开关关闭零参数，防 DDP 未使用参数）；移除 mock 合成数据（只接受真实 nuScenes）；RCSample 深度估计器与训练管线严格对齐（BICUBIC + config 断言）；`--depth-source {midas,rcsample,lidar}` 三臂；`--backend {numpy,torch,auto}` GPU 加速；`lr` 梯度裁剪等基线配置保持。

---

## 2. 训练

### 2.1 训练配置（`resworld_config.py` 顶层）

```python
# ---- Stage 2 掩码源（互斥）----
use_osz_midas = False      # 离线 npz（需先导出 data/osz）
use_osz_rcsample = True    # 在线同源（模型自身 RCSample 深度实时生成）
use_osz = use_osz_midas or use_osz_rcsample   # head 注入总闸

# ---- Stage 3 MHST ----
use_oarwm = True           # False = 不创建 MHST，严格基线
mhst_k = 5                 # K 假设数（消融 1/3/5/10）
mhst_sigma_min = 0.1       # σ 下限

# ---- Stage 4 风险场 ----
use_risk_field = True
risk_beta = 2.0            # 不确定性（π 熵）放大
risk_w_sigma = 1.0         # σ 尺度
risk_gamma = 1.0           # 占用（s_occ）放大

# ---- Stage 5 风险加权规划 ----
use_risk_plan = True
loss_plan_risk_weight = 0.1    # Minimax 语义（沿程 R_worst）——软正则（≈loss_plan_reg 权重 10 的 1%，ISSUE P0-1）
loss_plan_cvar_weight = 0.1    # CVaR（沿程尾部，topk 前乘 fut_w，ISSUE P2-1）
loss_plan_info_weight = 0.05   # 信息增益（hinge：末端风险高于起点才惩罚，ISSUE P1-1）
cvar_beta = 0.25               # 风险尾部比例
# risk_plan_warmup_epochs=2 + risk_plan_ramp_epochs=2（head 参数）：warmup 后
# risk 权重线性 0→1 爬升，替代硬切换（硬切换曾使 loss_plan_reg 0.53→1.24 跳变）

# ---- Stage 6 损失权重（0=关）----
loss_div_weight = 0.1          # 假设多样性（余弦）
loss_occ_halluc_weight = 1.0   # 遮挡暴露混合似然
loss_uncertainty_weight = 1.0  # σ 校准
loss_occ_gt_weight = 1.0       # 检测框占用 BCE
loss_occ_gt_pos_weight = 5.0   # BCE pos_weight（对抗平凡解）
```

### 2.1.2 基线配置（纯 ResWorld，全部开关关闭）

**跑基线训练只需把 `resworld_config.py` 开关区改成**（其余配置不动）：

```python
# ---- 基线：全部开关关闭 ----
use_osz_midas = False
use_osz_rcsample = False
# use_osz 由上面两行自动派生为 False，无需手改
use_oarwm = False            # 不创建 MHST（零额外参数）
use_risk_field = False       # 不算风险场
use_risk_plan = False        # 不算风险规划损失
```

然后照常训练（用独立 `--work-dir` 防快照覆盖）：

```bash
bash tools/dist_train.sh projects/configs/resworld/resworld_config.py 4 \
    --work-dir work_dirs/abl_baseline
```

要点：
- `use_oarwm=False` 时 `OcclusionMHSTHead`/`occ_head` **不创建**、`use_risk_field/plan=False` 时风险损失不算——模型严格等价纯 ResWorld（无额外参数、无额外计算）；
- `use_osz=False` 派生 `use_next=False`——数据管线**不加载 next 帧**（零开销），且 `loss_occ_halluc/uncertainty` 因无 `gt_bev_next` 自动跳过；
- Stage-6 的 `loss_occ_gt` 因无 `occ_head` 自动跳过，其余基线损失（depth/plan reg）照常；
- 恢复主配置：把开关区改回 §2.1 的值即可。

### 2.2 训练命令

```bash
conda activate resworld
# 前置：data/nuscenes/ 与 ckpts/geobev-r50-nuimage-cbgs.pth 就绪
# 掩码源：use_osz_rcsample（在线，无需导出）或 use_osz_midas（离线，先导 data/osz）
bash tools/dist_train.sh projects/configs/resworld/resworld_config.py 4
# 指定 GPU：CUDA_VISIBLE_DEVICES=4,5,6,7 bash tools/dist_train.sh ...
```

- 输出 `work_dirs/oa_resworld_config/`，EMA 权重 `epoch_12_ema.pth`（CLI `--work-dir` 可覆盖）；
- 训练要点：图像 256×704、6 相机、3 帧时序（当前+2 历史）；BEV 网格 200×200 各向异性（x 0.15 / y 0.3 m）；`samples_per_gpu=2` 可按显存调整；
- **next 帧监督通道**：数据管线加载 next 帧（`next_img_inputs`，仅作真值、不进训练输入），供 `L_occ_halluc`/`L_uncertainty` 的暴露监督；无 next 帧的样本自动跳过。

### 2.3 损失与监控

| 损失 | 含义 | 预期 |
|---|---|---|
| `loss_depth` | RCSample 深度 BCE | ~0.4-0.5 |
| `loss_plan_reg` / `loss_plan_reg_init` | 轨迹 L1（主/初始） | 训练下降 |
| `loss_div` | 假设多样性（余弦，有界 [-1,1]） | 小值（~0，可为负） |
| `loss_occ_halluc` | 遮挡区混合对数似然 | 冷启动大 → 快速下降（可转负=拟合好） |
| `loss_uncertainty` | σ 校准（`|σ−‖e‖²|`） | 冷启动大 → 收敛 ~0.1 |
| `loss_occ_gt` | 占用 BCE（pos_weight） | 平凡解上升 → 学占用 |
| `loss_plan_risk/cvar/info` | 风险加权规划三项 | 与 occ_* 同量级 |
| `grad_norm` | 裁剪前梯度范数（实际更新已 clip ≤35） | 随训练回落；指数级暴涨=发散需查 |

---

## 3. 评估

### 3.1 评估命令

```bash
bash tools/dist_test.sh projects/configs/resworld/resworld_config.py \
    work_dirs/oa_resworld_config/epoch_12_ema.pth 4 --eval bbox
```

### 3.2 指标与参考（UniAD/VAD 风格开环）

**L2（ADE，当前采用的表格口径）：**

| 配置 | L2 1s | L2 2s | L2 3s | L2 Avg |
|---|---|---|---|---|
| 官方参考 AVG | 0.14 | 0.27 | 0.49 | 0.30 |
| **基线实测（纯 ResWorld，12 epoch，开关全关）** | 0.142 | 0.271 | 0.486 | **0.300** |
| OARWM 首轮（2026-08-16，epoch_12_ema） | 0.285 | 0.543 | 0.896 | 0.574 |

**L2_stp3（终点 FDE）：**

| 配置 | stp3 1s | stp3 2s | stp3 3s | stp3 Avg |
|---|---|---|---|---|
| **基线实测（纯 ResWorld，12 epoch）** | 0.185 | 0.493 | 1.081 | **0.586** |
| OARWM 首轮（2026-08-16） | 0.375 | 0.982 | 1.808 | 1.055 |

碰撞率（CR）：OARWM 首轮 ≈ 0 / 1e-4 / 1e-4（1s/2s/3s，过度保守画像，见 ISSUE 首轮评估记录）；基线 CR 待补。

### 3.3 可视化评估结果

评估完成后（3.1 会输出 `test/resworld_config/<时间戳>/pts_bbox/results_nusc.pkl`），在**仓库根目录**执行：

```bash
# 生成 BEV 轨迹对比视频（vis.mp4，含 6 相机环视 + 预测/GT 轨迹）
python tools/analysis_tools/visualization.py \
    --result-path test/resworld_config/<时间戳>/pts_bbox/results_nusc.pkl \
    --save-path work_dirs/viz_oa
```

- 脚本硬编码 `dataroot='./data/nuscenes'`，需在仓库根目录运行；
- 输出 `work_dirs/viz_oa/vis.mp4`——直接目视预测轨迹 vs GT 轨迹的偏移/缩水模式。

### 3.4 数据说明

- `vad_nuscenes_infos_temporal_*.pkl` 用 VAD 生成文件（`tools/data_converter/vad_nuscenes_converter.py` 仅自行生成时用）；
- `nuscenes_map_anns_val.json` 首次评估自动生成；
- 评估只统计规划指标（plan_L2/碰撞率），bbox AP 由落盘 pkl 供外部 nuScenes 工具计算。

---

## 4. 消融

### 4.1 开关改法（重要）

所有开关是 `resworld_config.py` **顶层 Python 变量**，`use_switch`/assert 在 mmcv 解析时当场派生：
- ✅ 直接编辑 config 开关区再训练；
- ❌ 不要用 `--cfg-options` 覆盖（。

**每个消融臂用独立 `--work-dir`**：配置快照自动 dump 到 `work_dir/<config名>.py`（`tools/train.py:183`），同名会被覆盖；独立目录保证可追溯。

### 4.2 消融臂总表

| 消融臂 | 目的（设计文档 5.2） | config 开关 | work-dir |
|---|---|---|---|
| baseline | 纯 ResWorld 基线（5.2-1 w/o 掩码） | `use_osz_midas=False, use_osz_rcsample=False, use_oarwm=False, use_risk_field=False, use_risk_plan=False` | `abl_baseline` |
| stage2 only | 仅掩码、无 MHST/风险/规划 | 开一个掩码源，其余 False | `abl_stage2` |
| stage3 | 掩码+MHST、无风险/规划 | 掩码源 + `use_oarwm=True`，其余 False | `abl_stage3` |
| stage4 | +风险场、无风险规划 | + `use_risk_field=True` | `abl_stage4` |
| stage5（主配置） | 全链路 | + `use_risk_plan=True` | `abl_full` |
| K 消融 | 假设数（5.2-2） | `mhst_k=1/3/5/10` | `abl_k1/k3/k5/k10` |
| 风险放大消融 | 不确定性 vs 常数 α（5.2-8） | `risk_beta=0`（关 U 放大） | `abl_no_uboost` |
| 占用放大消融 | s_occ 参与风险场 on/off（5.2-8） | `risk_gamma=0` | `abl_no_occboost` |
| 规划风险消融 | Minimax/CVaR/信息增益 on/off（5.2-3/4） | `loss_plan_risk/cvar/info_weight=0` | `abl_plan_risk0` 等 |
| 深度来源 | midas / rcsample / lidar（§4.4） | `use_osz_midas` + `osz_dir` 指向对应 npz | `abl_osz_{midas,rcsample,lidar}` |

掩码源二选一：`use_osz_rcsample`（在线，无需导出）/ `use_osz_midas`（离线，先导出）。

### 4.3 完整示例（K 消融）

```bash
# ① 编辑 resworld_config.py：use_osz_rcsample=True, use_oarwm=True, mhst_k=5
bash tools/dist_train.sh projects/configs/resworld/resworld_config.py 4 \
    --work-dir work_dirs/abl_k5
# ② 编辑：mhst_k=1
bash tools/dist_train.sh projects/configs/resworld/resworld_config.py 4 \
    --work-dir work_dirs/abl_k1
# ③ 评估各臂
bash tools/dist_test.sh projects/configs/resworld/resworld_config.py \
    work_dirs/abl_k5/epoch_12_ema.pth 4 --eval bbox
```

### 4.4 深度来源消融（`--depth-source`）

| source | 深度来源 | 用途 |
|---|---|---|
| `midas`（默认） | MiDaS + LiDAR 尺度对齐 | 现状 |
| `rcsample` | ResWorld 自身 RCSample 深度头（需 `--rcsample-ckpt`） | 深度来源消融 / 部署期实时掩码 |
| `lidar` | LiDAR densified（`MockDepthEstimator`） | 上界参考 |

```bash
# arm A/B/C（离线导出，--backend torch 加速）
python OSZ/export_osz_dataset.py --dataroot data/nuscenes --version v1.0-trainval \
    --outdir data/osz{,_rcsample,_lidar} --use_drivable \
    [--depth-source rcsample --rcsample-ckpt work_dirs/oa_resworld_config/epoch_12_ema.pth]
```

注意：**训练与测试必须用同一深度来源 npz**（离线模式）；三臂比 L2/碰撞率 + 掩码质量（与 GT 遮挡 IoU/precision/recall），lidar 臂为上界参考。

---

## 5. 计划改动（V2：开环不劣于基线 + 鬼探头感知，实现指导）

> 背景（2026-08 首轮评估）：OARWM L2 Avg 0.574 vs 基线实测 0.300（+91%）、stp3 Avg
> 1.055 vs 0.586。概念设计见 OARWM_ResWorld.md §6（6.1-6.4 四项修改）；本节是逐项
> **实现指导**。实现顺序 = 编号顺序，每项独立可验证（改完一项 12 epoch + 评估对照
> §3.2 基线行）；每项改动提交前 `python -m py_compile` 检查。

### 5.1 改动 1：GT 风险上界损失（`resworld_head.py::loss`）

替换位置：`loss` 的 Stage-5 块（`if risk_active:` 内三项）。概念对应
OARWM_ResWorld.md §6.1。

实现步骤：

1. **GT 轨迹采样**：`ego_fut_gt` 是**绝对坐标** `(B, M, T, 2)`（与
   `traj_abs = ego_fut_preds.cumsum(2)` 同坐标系、同一 cmd 语义）。GT 网格：
   `nx_gt = (ego_fut_gt[..., 0] - gmin[0]) / (gsz[0] * 200)`、`ny_gt` 同法，`*2-1`
   归一化；与 pred grid 拼成一个 `grid_sample` 调用采样同一 `risk_field`
   （一次采样得 `r_worst_pred (B,M,T)` 与 `r_worst_gt (B,M,T)`）。
2. **cmd 加权**：`r_gt_cmd = (r_worst_gt * cmd_w.unsqueeze(-1)).sum(1)`，与
   `r_cmd` 同一约定。
3. **risk**：`loss_plan_risk = mean(max(0, r_cmd - r_gt_cmd - margin) * fut_w)
   * loss_plan_risk_weight * risk_scale`（fut_w 对两条轨迹相同；warmup/ramp 机制
   照乘不动）。
4. **CVaR**：`max(0, topk(r_cmd) - topk(r_gt_cmd) - margin)`，topk 前同样乘
   `fut_w`（沿用 P2-1 修复）。
5. **info**：起点（ego 原点）风险对两轨迹相同 → 相对形式退化为
   `max(0, r_end_pred - r_end_gt)`；`r_end_gt` 用与 `r_end_pred` 相同的"最后有效步"
   gather 逻辑（`fut_w` 定位）。
6. **新超参 `risk_plan_margin`**：默认 **5.0**（风险单位；正常 `rf_occ` 15-30，
   5 ≈ 1/3 容差；0 = 纯上界，先 5.0 起步）。加三处：`resworld_config.py` 顶层 +
   `pts_bbox_head` 参数传入 + `ResWorldHead.__init__` 默认值。
7. **预期**：`loss_plan_risk/cvar` 在多数样本上 ≈0（稀疏激活）；`loss_plan_reg` 与
   `loss_plan_reg_init` 差距消失；评估 L2 回到基线 ±0.05。

### 5.2 改动 2：col_attn 用干净 pred_bev（`resworld_head.py::forward`）

替换位置：`forward` 的 MHST 块（`if self.use_oarwm and osz_mask is not None:`）。
概念对应 OARWM_ResWorld.md §6.3。

实现步骤：

1. 在 `fused, mhst_aux = self.mhst(pb, mhst_mask)` 之前保存引用
   `pred_bev_clean = pred_bev`（MHST 之后 `pred_bev` 变量会被 fused 覆盖）。
2. fused 计算后**不再覆盖** `pred_bev`（即 col_attn 的 value 用干净特征）；
   `fused` 张量直接丢弃——其信息已由 `outs['mhst']`（π/σ/ΔB）与风险场承载。
3. 下游不动：`outs['pred_bev']` 为干净版本；风险场仍从 aux 构造（不依赖 fused）；
   暴露损失仍用 `mhst_aux['pred_bev_4d']`。
4. **检查**：MHST 参数梯度仍来自 Stage-6 损失（occ_halluc/uncertainty/div/occ_gt），
   无 DDP unused 参数问题；`use_oarwm=False` 路径完全不变。
5. **预期**：评估 L2 恢复到基线精修上限（0.30 水平）；风险感知仍经 risk 项生效
   （风险场构造与 fused 无关）。

### 5.3 改动 3：风险场真实性锚定（`resworld_head.py::loss` 新增损失）

替换位置：`loss` 的 Stage-6 块（`if mhst is not None:` 内、`gt_next` 分支中）。
概念对应 OARWM_ResWorld.md §6.2。零额外数据（复用已有 e2）。

实现步骤：

1. **真值变化代理**：e2 已存在（`e2 = (b_hat.detach()-gt_next).pow(2).mean(1,
   keepdim=True)`，暴露误差 `(B,1,H,W)`）；目标 =
   `e2.clamp(max=E)`，E 默认复用 `mhst_sigma_max`（10）。
2. **风险代理原始强度（未 detach 版本）**：
   `r_raw = (sigma / (1 + sigma)) * delta.norm(dim=2).mean(dim=1, keepdim=True)
   * mask`（σ、ΔB 均**不 detach**——grounding 属"分布塑造损失"，与
   build_risk_field 的 P0-2 detach 契约不冲突：该契约只约束"风险场→规划"方向）。
3. **损失**：`loss_risk_ground = MSE(r_raw, e2_clamped) * mask`（遮挡区均值）×
   `loss_risk_ground_weight`（**新超参**，默认 0.1；config 顶层 + head 参数 + 默认值）。
4. **说明**：该损失训练 π/σ/ΔB 使风险强度与真实暴露对齐；mask 之外不监督；
   `loss_risk_ground_weight=0` 即关闭。
5. **预期**：风险场收敛为"未来内容变化"的无偏估计 → GT 轨迹附近风险天然低 → 与
   5.1 上界约束联合后，risk 项只在"偏离 GT 且真实危险"处激活。
   可选扩展（暂不做）：next 帧检测框占用做更精确目标（需数据管线加载 next 帧
   检测框，改动大）。

### 5.4 改动 4：评估侧补强（新脚本，不动训练）

概念对应 OARWM_ResWorld.md §6.4。

1. **遮挡子集筛选** `tools/analysis_tools/filter_occ_subset.py`：读 `data/osz/*.npz`
   （midas 导出掩码）统计每 token 的 occ_frac，筛选 >20% 的 token 列表；对评估落盘的
   `results_nusc.pkl`（含 sample_token）过滤后重算 L2/CR（复用现有评估函数，只换
   样本子集）。注意：主配置为在线 rcsample 掩码，与 midas npz 不完全一致——筛选
   标准以 midas npz 为准（导出完成后可用）。
2. **减速行为统计** `tools/analysis_tools/traj_behavior_stats.py`：从结果 pkl 读
   预测/GT 轨迹；对每个样本统计"掩码边界 5 m 内轨迹段"的平均速度（预测 vs GT），
   输出整体减速比——"接近遮挡物主动减速"的行为证据。
3. **风险场校准（可选）**：test 时每样本保存 `risk_field` 遮挡区均值 + `e2` 均值
   （`simple_test` 小改，加一个可选 dump 开关），离线算 AUC/校准曲线。
4. **预期**：论文证据链——"L2 持平基线 + 遮挡子集 CR 更低 + 减速行为 + 风险场
   校准"四项证明"感知鬼探头"。

### 5.5 实施顺序与验证

1. 顺序：5.1 → 5.2 → 5.3 → 5.4（每步独立重训验证，与 §3.2 基线行对照）；
2. 每项改完：`py_compile` → 12 epoch → `dist_test` → 对照基线实测
   （L2 Avg 0.300 / stp3 Avg 0.586）；
3. 目标线：改动 1+2 后 L2 ≤ 0.35（基线 +0.05）、CR 不高于基线；改动 3 后风险场
   AUC 可测；改动 4 补齐论文证据。
4. 超参汇总：新增 `risk_plan_margin=5.0`、`loss_risk_ground_weight=0.1`；
   既有 risk 三项权重 0.1/0.1/0.05 与 warmup/ramp 不动。
