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
| **3 多假设 MHST** | 遮挡区 K 假设残差转移：先验网络（多尺度膨胀邻域）+ K expert 分支 + σ 不确定性；**P0-6 硬限幅**（ΔB clamp ±10、σ cap 10，僵尸假设防爆）；**V2 消费路径**：多假设不回流 BEV 特征——`col_attn` 消费干净 `pred_bev`，分布经 aux 旁路只以风险场形式进入决策 | `resworld_head.py::OcclusionMHSTHead` | `use_oarwm`、`mhst_k=5`、`mhst_sigma_min=0.1`、`mhst_delta_clamp`、`mhst_sigma_max` |
| **4 风险场** | 语义不可知双风险场（无梯度测量，π/σ/ΔB/s_occ 全 detach，ISSUE P0-2/P2-3）：`R^(k)=w_σ·σ‖ΔB^(k)‖·M` → `R_exp`/`R_worst` → 不确定性+占用驱动放大（π 熵 U + s_occ 占用）；**P0-6 输出 clamp**（risk_clamp=100）；**V2 真实性锚定**（原始强度与暴露误差 e² 对齐） | `resworld_head.py::build_risk_field` | `use_risk_field`、`risk_beta/w_sigma/gamma`、`risk_clamp` |
| **5 风险加权规划** | **V2 GT 风险上界约束**：三项风险损失均为相对形式 `max(0, R(τ_pred)−R(τ_gt)−μ)`（沿程 R_worst Minimax 语义 + 沿程尾部 CVaR + 末端风险差信息增益）——预测≈GT 时零梯度，不伤开环 L2；只在"偏离 GT 且更危险"时介入 | `resworld_head.py::loss` | `use_risk_plan`、`loss_plan_risk/cvar/info_weight`、`cvar_beta`、`risk_plan_margin` |
| **6 端到端损失** | 遮挡暴露混合似然 `L_occ_halluc` + 检测框占用 `L_occ_gt`（pos_weight 防平凡解）+ 多样性 `L_div`（余弦）+ σ 校准 `L_uncertainty` + **V2 风险场锚定 `L_risk_ground`**（零额外数据）+ 规划 `L_plan`（含相对风险项）；next 帧独立监督通道 | `resworld_head.py::loss`、`resworld.py::encode_next_bev`、`nuscenes_resworld_dataset.py`（`use_next`）、`loading.py`（next 帧处理） | 各损失权重（0=关） |

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

# ---- Stage 5 风险加权规划（V2：GT 风险上界约束）----
use_risk_plan = True
loss_plan_risk_weight = 0.1    # 沿程 R_worst，相对形式 max(0, R(pred)-R(gt)-margin)
loss_plan_cvar_weight = 0.1    # CVaR 尾部（topk 前乘 fut_w，相对 GT 尾部）
loss_plan_info_weight = 0.05   # 信息增益（末端风险差：pred vs GT）
cvar_beta = 0.25               # 风险尾部比例
risk_plan_margin = 5.0         # 容差 μ（风险单位）；0 = 纯上界
# risk_plan_warmup_epochs=2 + risk_plan_ramp_epochs=2（head 参数）：warmup 后
# risk 权重线性 0→1 爬升，替代硬切换（硬切换曾使 loss_plan_reg 0.53→1.24 跳变）

# ---- Stage 6 损失权重（0=关）----
loss_div_weight = 0.1          # 假设多样性（余弦）
loss_occ_halluc_weight = 1.0   # 遮挡暴露混合似然
loss_uncertainty_weight = 1.0  # σ 校准
loss_occ_gt_weight = 1.0       # 检测框占用 BCE
loss_occ_gt_pos_weight = 5.0   # BCE pos_weight（对抗平凡解）
loss_risk_ground_weight = 0.1  # V2 风险场锚定（原始强度 vs 暴露误差 e²，零额外数据）
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
| `loss_risk_ground` | 风险场锚定：原始强度 σ/(1+σ)·‖ΔB‖ 与暴露误差 e² 的 MSE（遮挡区） | 收敛后风险场为"未来内容变化"的无偏估计；冷启动时锚定张力大属预期，与 `risk_plan_margin` 的量纲（风险单位）联动，margin 调参前先看其收敛 |
| `loss_plan_risk/cvar/info` | 风险加权规划三项（**V2 相对形式**：risk/cvar = max(0, R(pred)−R(gt)−margin)，info = max(0, 末端风险差)） | 预测≈GT 时 ≈0（稀疏激活）；仅"比 GT 更危险"时非零 |
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

### 3.4 遮挡子集评估与行为统计（V2）

```bash
# ① 遮挡子集筛选 + 子集 L2 重算（occ_frac > 20%，midas npz 为基准）
python tools/analysis_tools/filter_occ_subset.py \
    --result-pkl test/resworld_config/<时间戳>/pts_bbox/results_nusc.pkl \
    --out-json work_dirs/occ_subset_tokens.json

# ② 近遮挡减速行为统计（掩码边界 5 m 内轨迹步的平均速度：预测 vs GT）
python tools/analysis_tools/traj_behavior_stats.py \
    --result-pkl test/resworld_config/<时间戳>/pts_bbox/results_nusc.pkl \
    [--token-json work_dirs/occ_subset_tokens.json]
```

- 两个脚本消费 `pts_bbox/results_nusc.pkl`（含 `plan_results` 预测增量轨迹 + `plan_gts` GT 增量轨迹）；L2 口径与评估一致（1s=前 2 步、2s=前 4 步、3s=前 6 步；stp3=末端误差）；
- 碰撞率子集重算需完整评估重跑（GT 物体框不在落盘 pkl 中）。

### 3.5 数据说明

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
| 风险容差消融 | GT 上界容差 μ（V2） | `risk_plan_margin=0/5/20` | `abl_margin0/5/20` |
| 锚定损失消融 | 风险场真实性锚定 on/off（V2） | `loss_risk_ground_weight=0` | `abl_no_ground` |
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
