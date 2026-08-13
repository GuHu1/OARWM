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
| **3 多假设 MHST** | 遮挡区 K 假设残差转移：先验网络（多尺度膨胀邻域）+ K expert 分支 + σ 不确定性；门控合成（可见区原值、遮挡区混合）；路径 A 期望输出 + aux 旁路 | `resworld_head.py::OcclusionMHSTHead` | `use_oarwm`、`mhst_k=5`、`mhst_sigma_min=0.1` |
| **4 风险场** | 语义不可知双风险场（无梯度测量，π/σ/ΔB/s_occ 全 detach，ISSUE P0-2/P2-3）：`R^(k)=w_σ·σ‖ΔB^(k)‖·M` → `R_exp`/`R_worst` → 不确定性+占用驱动放大（π 熵 U + s_occ 占用） | `resworld_head.py::build_risk_field` | `use_risk_field`、`risk_beta/w_sigma/gamma` |
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

# ---- Stage 6 损失权重（0=关）----
loss_div_weight = 0.1          # 假设多样性（余弦）
loss_occ_halluc_weight = 1.0   # 遮挡暴露混合似然
loss_uncertainty_weight = 1.0  # σ 校准
loss_occ_gt_weight = 1.0       # 检测框占用 BCE
loss_occ_gt_pos_weight = 5.0   # BCE pos_weight（对抗平凡解）
```

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

| 指标 | L2 1s | L2 2s | L2 3s | L2 Avg | CR 1s | CR 2s | CR 3s | CR Avg |
|---|---|---|---|---|---|---|---|---|
| L2_MAX / CR_MAX | 0.19 | 0.50 | 1.08 | 0.59 | 0.02 | 0.06 | 0.43 | 0.17 |
| L2_AVG / CR_AVG | 0.14 | 0.27 | 0.49 | 0.30 | 0.01 | 0.03 | 0.14 | 0.06 |

### 3.3 数据说明

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

## 5. 下一步改造路线（按 Stage）

1. **Stage 1-6 均已实现**（见 §1 总览）；后续为完善与扩展：
   - **评估与调优**：完整训练 12 epoch，观察各损失收敛；对照官方参考指标（§3.2）；调损失权重（如 `loss_occ_halluc/uncertainty` 冷启动梯度大时可降权）；
   - **消融矩阵**：按 §4.2 跑全消融臂，验证各 Stage 贡献；
   - **Bench2Drive 闭环**：闭环驾驶得分（Merging / Emergency Brake / Give Way / Unprotected Turn）；
   - **自建遮挡数据集**：筛选 parked car / truck / 交叉口遮挡比例 >20% 片段。
2. **可选扩展**（见 §6 未实现项）：语义解码臂、时空演化 rollout、π 信念修正——如需纳入论文再实现。

---

## 6. 设计但未实现（及原因）

- **L_recon（可见区 BEV 重建）**：ResWorld 无未来 BEV 真值监督（BEV 特征是编码器输出而非标签），无逐帧重建目标；可见区监督由 latent 残差路径 + 规划损失承担。
- **语义解码臂（附录 A 的 `RiskWeight` 类别加权）**：需语义监督（lidarseg 或检测框类别栅格化）；主方案保持语义不可知（无监督语义不可信），仅保留为可选臂设计。
- **时空演化 rollout（R_{t+τ}）**：ResWorld 单帧推理，无 T 步未来 rollout；当前为单帧风险场 R_t，时变重采样为接口。
- **Stage 3.4 遮挡-可见边界交互（π 信念修正）**：需下一帧观测 B_{t+1}^{obs} 修正假设权重，单帧推理无法提供，为设计接口。
- **显式离散假设类别（K 假设绑定空路/静止车/行人等语义）**：K 个 expert 是隐式假设（语义由数据塑造，EM 式软分配），未做类别标签绑定——混合模型常规做法，论文表述为"潜变量假设，语义由数据塑造"。

---

## 7. 已知限制

- **深度范围**：MiDaS 输出逆深度经 LiDAR 对齐后为 metric 深度，但无 LiDAR 区域仅相对深度；
- **OSZ 网格前方仅 ±15 m**（跟随 ResWorld `grid_config`）：>15 m 前方遮挡物不在世界模型 BEV 内；若需更远需同时改 `grid_config` 与 `OSZ/config.py`（单一来源同步）；
- **各向异性近似**：射线投射在 cell 空间为直线，物理空间角度略拉伸（0.15 vs 0.3 m/cell）；
- **drivable 膨胀**：按较粗轴（0.3 m）迭代，x 方向实际膨胀 0.75 m（1.5 m 设定折半），偏保守方向安全。
