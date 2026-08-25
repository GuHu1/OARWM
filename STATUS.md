# OARWM-Res 项目状态

> 设计文档：`OARWM_ResWorld.md` · 工程修订与不变量：`ISSUE.md` · 安装/数据准备：`INSTALL.md`
> 服务器：8×RTX 3090，`resworld` 环境（python 3.8 + torch 1.9.1+cu111 + mmcv-full 1.4.0 + mmdet3d 0.17.1）

---

## 1. 改造总览（按 Stage）

| Stage | 改造内容 | 关键文件 | 开关 |
|---|---|---|---|
| **1 图像/BEV** | 继承 ResWorld（ResNet-50 + RCSample + BEV encoder），未改动 | — | — |
| **2 显式遮挡几何** | OSZ 高度感知射线投射生成遮挡掩码；**三态注入**：`risk_gated`（风险门控回流 `B̃ = B + g⊙Proj([R_exp.detach(), R_worst.detach()])`，g 零初始化 + tanh 上界 + warmup 冻结 + L1 预算，Proj 常规增益）/ `raw_additive`（无监督加性残差，消融臂）/ `off`（等价基线） | `OSZ/`、`resworld_head.py::OcclusionAwareFusion`、`nuscenes_resworld_dataset.py` | `use_osz_midas`/`use_osz_rcsample`（掩码源互斥）、`osz_inject_mode`、`gate_warmup_iters`、`loss_gate_weight` |
| **3 多假设 MHST** | 遮挡区 K 假设残差转移：先验网络（多尺度膨胀邻域）+ K expert 分支（零初始化）+ σ 不确定性 + 硬限幅（ΔB ±10 / σ≤10）；**col_attn 消费干净 `pred_bev`**，混合分布不回流；occ_head 随 `use_oarwm` 创建（RiskHead 的 s_occ 输入） | `resworld_head.py::OcclusionMHSTHead` | `use_oarwm`、`mhst_k=5`、`mhst_sigma_min=0.1`、`mhst_delta_clamp`、`mhst_sigma_max` |
| **4 可学习 RiskHead** | 输入全 detach（`B_t`、`M`、`drivable`、`φ=σ/(1+σ)·‖ΔB‖`、`s_occ`、`1/(1+d)`、`cmd_embed`）；MC-dropout（末两层 dropout，T=4 前向）；输出 `μ`（sigmoid，BCE 兼容）+ `σ_epis`（dropout 方差）；双通道契约 `R_exp=clamp(μ,0,R_max)`、`R_worst=clamp(μ+β·σ_epis,0,R_max)`；高斯平滑（kernel=5, σ=1.0） | `resworld_head.py::RiskHead` | `use_risk_field`、`risk_hidden/dropout/mc_t`、`risk_beta`、`risk_max`、`risk_smooth_ks/sigma` |
| **5 绝对阈值安全下界** | `L_plan_guard = (1/T)Σ_t max(0, R_worst(τ_t) − R_safe)`；`R_safe = EMA_0.999[quantile_0.1(μ \| 碰撞格)]`（buffer，batch 无碰撞正例不更新，DDP all_reduce 同步）；采样风险场输入 detach（规划只学规避、不塑形风险场）；`gt_relative`（沿程相对 + CVaR 消融）为回退臂 | `resworld_head.py::loss` | `use_risk_plan`、`risk_plan_mode`、`loss_plan_guard_weight`、`risk_safe_quantile/ema`、`loss_plan_cvar_weight`（消融） |
| **6 端到端损失** | 八项：分布族五项（`L_div`/`L_occ_halluc`/`L_uncertainty`/`L_occ_gt`/`L_risk_ground`）+ 安全/门控三项（`L_col` 碰撞占用 BCE（GT 未来足迹 ∩ 动态占用）、`L_dyn` 动态占用 BCE（`S_dyn` = 栅格 + 沿朝向 forward margin 2 m）、`L_gate=λ·‖g‖₁`）；`L_occ_gt` 目标为动态占用 `S_gt`（遮挡区 BCE，pos_weight=5） | `resworld_head.py::loss`、`_rasterise_dynamic`、`resworld.py::encode_next_bev`、`nuscenes_resworld_dataset.py`（`use_next`）、`loading.py` | 各 `loss_*_weight`（0=关）、`dyn_forward_margin`、`dyn_vel_thresh` |

**工程与基础**（贯穿各 Stage）：OSZ 网格对齐 ResWorld（200×200 各向异性，单一来源 `OSZ/config.py`）；`use_osz`/`use_oarwm` 条件实例化（开关关闭零参数，防 DDP 未使用参数）；next 帧独立监督通道（`gt_bev_next` 不进训练输入）；`risk_safe`/`gate` 等训练状态注册为 buffer/参数，DDP 同步；`CustomSetEpochInfoHook` 每 iter 前注入 `head.iter`（gate warmup 按 iter 计）；RCSample 深度估计器与训练管线严格对齐；`--depth-source {midas,rcsample,lidar}` 三臂；`--backend {numpy,torch,auto}` GPU 加速。

---

## 2. 训练

### 2.1 训练配置（`resworld_config.py` 顶层）

```python
# ---- Stage 2 掩码源（互斥）----
use_osz_midas = True       # 离线 npz（需先导出 data/osz）
use_osz_rcsample = False   # 在线同源（部署形式）
use_osz_drivable = True    # drivable 约束（离线路径 intersect 掩码；两路径均向
                           # RiskHead 提供独立 drivable_mask 通道）
use_osz = use_osz_midas or use_osz_rcsample   # head 注入/风险总闸

# ---- Stage 2 注入（三态）----
osz_inject_mode = 'risk_gated'  # 'off'（等价基线）/ 'raw_additive'（无监督加性残差消融臂）
                                # / 'risk_gated'（风险门控回流，主配置）
gate_warmup_iters = 2000        # g 冻结期（注入值恒 0，Proj 常规增益保梯度）
loss_gate_weight = 1e-5         # ‖g‖₁ 带宽预算（g=tanh(gate_raw)，|g|≤1 自动成立）

# ---- Stage 3 MHST ----
use_oarwm = True           # False = 不创建 MHST，严格基线
mhst_k = 5                 # K 假设数（消融 1/3/5/10）
mhst_sigma_min = 0.1       # σ 下限
mhst_delta_clamp = 10.0    # ΔB 逐元素限幅（P0-6）
mhst_sigma_max = 10.0      # σ 上限（P0-6）

# ---- Stage 4 RiskHead（可学习，MC-dropout UCB）----
use_risk_field = True
risk_hidden = 64           # 隐藏层通道
risk_dropout = 0.1         # 末两层 dropout（MC-dropout）
risk_mc_t = 4              # MC 前向次数（μ=mean, σ_epis=std）
risk_mc_eval = False       # 评估期 MC（True=推理也 T 次随机前向；False=单次点估计）
risk_beta = 1.0            # UCB 系数 β（概率量纲：σ_epis 为 sigmoid std ≤0.5，β=1≈1σ
                           # 上界；β=2 已被 risk_max=1.0 截断饱和，β>2 无意义；
                           # 勿沿用旧 clamp-100 风险场的直觉；消融 0.5/1/2）
risk_max = 1.0             # R_exp/R_worst clamp 上界（概率语义）
risk_smooth_ks = 5         # 高斯平滑核
risk_smooth_sigma = 1.0    # 高斯平滑 σ
risk_cmd_proj = 8          # cmd 嵌入投影通道（共享 navi_embedding，独立投影）

# ---- Stage 5 风险规划 ----
use_risk_plan = True
risk_plan_mode = 'absolute_hinge'   # 主配置；'gt_relative'（沿程相对项）为回退臂
loss_plan_guard_weight = 0.1        # L_plan_guard 权重（λ_g）
risk_safe_quantile = 0.1            # R_safe 标定分位数（碰撞格 μ 的 0.1 分位）
risk_safe_ema = 0.999               # R_safe EMA 系数（按"正例 batch"计步：
                                    # 首次出现碰撞正例才开始更新，无正例 batch 跳过；
                                    # 半衰期 ≈693 个正例 batch，收敛速度由碰撞密度决定）
risk_plan_margin = 1.0              # 仅 gt_relative 模式使用
loss_plan_cvar_weight = 0.0         # CVaR 尾部项（消融臂，默认关）
cvar_beta = 0.25                    # CVaR 尾部比例
# risk_plan_warmup_epochs=2 + risk_plan_ramp_epochs=2（head 参数）：warmup 后
# guard 权重线性 0→1 爬升（硬切换曾使 loss_plan_reg 0.53→1.24 跳变）

# ---- Stage 6 损失权重（0=关）----
loss_div_weight = 0.1          # 假设多样性（余弦，遮挡区）
loss_occ_halluc_weight = 1.0   # 遮挡暴露混合似然
loss_uncertainty_weight = 1.0  # σ 校准（|σ−e²|）
loss_occ_gt_weight = 1.0       # 遮挡区动态占用 BCE（s_occ vs S_gt）
loss_occ_gt_pos_weight = 5.0   # BCE pos_weight
loss_risk_ground_weight = 0.1  # 风险场锚定（φ vs 暴露误差 e²）
loss_col_weight = 1.0          # 碰撞硬锚点 BCE（μ vs C_gt）
loss_col_pos_weight = 10.0     # 碰撞正例 pos_weight（正例极稀疏）
loss_dyn_weight = 1.0          # 动态占用 BCE（μ vs S_dyn，含 forward margin）
loss_dyn_pos_weight = 5.0      # BCE pos_weight
dyn_forward_margin = 2.0       # S_dyn 沿目标朝向前向 margin（m）
dyn_vel_thresh = 0.5           # 动态物速度阈值（m/s）
```

### 2.1.2 基线配置（纯 ResWorld，全部开关关闭）

**跑基线训练只需把 `resworld_config.py` 开关区改成**（其余配置不动）：

```python
# ---- 基线：全部开关关闭 ----
use_osz_midas = False
use_osz_rcsample = False
# use_osz 由上面两行自动派生为 False，无需手改
osz_inject_mode = 'off'       # 不注入（三态中最干净的基线等价）
use_oarwm = False             # 不创建 MHST/occ_head（零额外参数）
use_risk_field = False        # 不创建 RiskHead
use_risk_plan = False         # 不算风险规划损失
```

然后照常训练：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash tools/dist_train.sh projects/configs/resworld/resworld_config.py 4 > /data2/jhc/OARWM/work_dirs/train.log 2>&1 &
```

要点：
- `use_oarwm=False` 时 `OcclusionMHSTHead`/`occ_head` 不创建、`use_risk_field/plan=False` 时 `RiskHead` 不创建、风险损失不算——模型严格等价纯 ResWorld（无额外参数、无额外计算）；
- `use_osz=False` 派生 `use_next=False`——数据管线不加载 next 帧（零开销），`loss_occ_halluc/uncertainty` 因无 `gt_bev_next` 自动跳过；
- 主配置：把开关区设为 §2.1 的值即可。

### 2.2 训练命令

```bash
conda activate resworld
# 前置：data/nuscenes/ 与 ckpts/geobev-r50-nuimage-cbgs.pth 就绪
# 掩码源：use_osz_rcsample（在线，无需导出）或 use_osz_midas（离线，先导 data/osz）
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash tools/dist_train.sh projects/configs/resworld/resworld_config.py 4 > /data2/jhc/OARWM/work_dirs/train.log 2>&1 &
```

- 输出 `work_dirs/oa_resworld_config/`，EMA 权重 `epoch_12_ema.pth`（CLI `--work-dir` 可覆盖）；
- 训练要点：图像 256×704、6 相机、3 帧时序（当前+2 历史）；BEV 网格 200×200 各向异性（x 0.15 / y 0.3 m，head 内 100×100）；`samples_per_gpu=2` 可按显存调整；
- next 帧监督通道：数据管线加载 next 帧（`next_img_inputs`，仅作真值、不进训练输入），供 `L_occ_halluc`/`L_uncertainty` 的暴露监督；无 next 帧的样本自动跳过；
- RiskHead 训练期每步 MC 前向 T=4 次（轻量 CNN，100×100 输入，开销可控）。

### 2.3 损失与监控

| 损失 | 含义 | 预期 |
|---|---|---|
| `loss_depth` | RCSample 深度 BCE | ~0.4-0.5 |
| `loss_plan_reg` / `loss_plan_reg_init` | 轨迹 L1（主/初始） | 训练下降 |
| `loss_plan_guard` | 绝对阈值安全下界 `max(0, R_worst(τ_t) − R_safe)`（仅 absolute_hinge） | 激活率 ≪ 1；恒 1 = R_safe 标定失效 |
| `loss_gate` | ‖g‖₁ 带宽预算 | warmup 后应离开 0；恒 0 = 门控饥饿，查 Proj 增益 / RiskHead 监督 |
| `loss_col` | 碰撞硬锚点 BCE（μ vs C_gt，pos_weight=10） | 稀疏但正例处 μ→1 |
| `loss_dyn` | 动态占用 BCE（μ vs S_dyn 含 forward margin） | 训练下降 |
| `loss_div` | 假设多样性（余弦，有界 [-1,1]） | 小值（~0，可为负） |
| `loss_occ_halluc` | 遮挡区混合对数似然 | 冷启动大 → 快速下降（可转负=拟合好） |
| `loss_uncertainty` | σ 校准（`|σ−‖e‖²|`） | 冷启动大 → 收敛 ~0.1 |
| `loss_occ_gt` | 遮挡区动态占用 BCE（pos_weight=5） | 平凡解上升 → 学占用 |
| `loss_risk_ground` | 风险场锚定：φ=σ/(1+σ)·‖ΔB‖ 与暴露误差 e² 的 MSE（遮挡区） | 收敛后 φ 为"未来内容变化"的无偏读数 |
| `grad_norm` | 裁剪前梯度范数（实际更新已 clip ≤35） | 随训练回落；指数级暴涨=发散需查 |

**[DIAG] 行**（随 log interval 打印）：`occ_frac`、`rf_mean`、`rf_occ`（遮挡区风险均值）、`r_cmd_mean`、`r_gt_cmd_mean`、`g_l1`（‖g‖₁，warmup 后应离开 0）、`sep_mean`/`sep_var`（σ_epis 空间均值/方差，≈0 则 UCB 通道失效）、`guard_act`（guard 激活率，应 ≪ 1）、`risk_safe`（R_safe 标定值，仅在含碰撞正例的 batch 上推进；长期贴 1.0 且 `loss_col` 已收敛 = 正例密度过低）、`col_n`（本 batch 碰撞正例格数，R_safe 的标定燃料）、`traj_end`/`traj_step`（轨迹形状）。

---

## 3. 评估

### 3.1 评估命令

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash tools/dist_test.sh projects/configs/resworld/resworld_config.py work_dirs/oa_resworld_config/epoch_12_ema.pth 4 --eval bbox > work_dirs/oa_resworld_config/eval.log 2>&1 &
```

### 3.2 指标参考（UniAD/VAD 风格开环）

以下为已完成的评测记录（按评测日期区分配置）；主配置训练完成后更新：

**L2 (ADE):**

| 配置 | L2 1s | L2 2s | L2 3s | L2 Avg | CR Avg |
|---|---|---|---|---|---|
| **ResWorld** | 0.142 | 0.271 | 0.486 | **0.300** | 0.17 |
| OARWM (2026-08-23 评测, offline MiDaS) | 0.166 | 0.312 | 0.542 | 0.340 | 8.3e-4 |
| OARWM (主配置, 待训) | — | — | — | — | — |

**L2_stp3(终点 FDE):**

| 配置 | stp3 Avg | CR Avg |
|---|---|---|
| **ResWorld** | **0.586** | **0.06** |
| OARWM (2026-08-23 评测, offline MiDaS) | 0.651 | 2.6e-3 |
| OARWM (主配置, 待训) | — | — |

### 3.3 可视化评估结果

评估完成后（3.1 会输出 `test/oa_resworld_config/<时间戳>/pts_bbox/results_nusc.pkl`），在**仓库根目录**执行：

```bash
# 生成 BEV 轨迹对比视频（vis.mp4，含 6 相机环视 + 预测/GT 轨迹）
python tools/analysis_tools/visualization.py \
    --result-path test/oa_resworld_config/<时间戳>/pts_bbox/results_nusc.pkl \
    --save-path work_dirs/viz_oa
```

- 脚本硬编码 `dataroot='./data/nuscenes'`，需在仓库根目录运行；
- 输出 `work_dirs/viz_oa/vis.mp4`——直接目视预测轨迹 vs GT 轨迹的偏移/缩水模式。

### 3.4 遮挡子集评估与行为统计

```bash
# ① 遮挡子集筛选 + 子集 L2 重算（occ_frac > 20%，midas npz 为基准）
python tools/analysis_tools/filter_occ_subset.py \
    --result-pkl test/oa_resworld_config/<时间戳>/pts_bbox/results_nusc.pkl \
    --out-json work_dirs/occ_subset_tokens.json

# ② 近遮挡减速行为统计（掩码边界 5 m 内轨迹步的平均速度：预测 vs GT）
python tools/analysis_tools/traj_behavior_stats.py \
    --result-pkl test/oa_resworld_config/<时间戳>/pts_bbox/results_nusc.pkl \
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
- ❌ 不要用 `--cfg-options` 覆盖。

**每个消融臂用独立 `--work-dir`**：配置快照自动 dump 到 `work_dir/<config名>.py`（`tools/train.py:183`），同名会被覆盖；独立目录保证可追溯。

### 4.2 消融臂总表

| 消融臂 | 目的（设计文档 5.2） | config 开关 | work-dir |
|---|---|---|---|
| baseline | 纯 ResWorld 基线（5.2-1 w/o 掩码） | §2.1.2 全关 | `abl_baseline` |
| stage2 only | 仅掩码（raw_additive 臂）/ 仅掩码无注入 | 掩码源 + `osz_inject_mode` | `abl_stage2` |
| stage3 | 掩码+MHST、无 RiskHead/规划 | 掩码源 + `use_oarwm=True`，其余 False | `abl_stage3` |
| stage4 | +RiskHead（L_col/L_dyn）、无风险规划 | + `use_risk_field=True` | `abl_stage4` |
| stage5（主配置） | 全链路 | + `use_risk_plan=True` | `abl_full` |
| K 消融 | 假设数（5.2-2） | `mhst_k=1/3/5/10` | `abl_k1/k3/k5/k10` |
| 注入模式 | 门控必要性（5.2 w/o 风险注入） | `osz_inject_mode=off/raw_additive/risk_gated` | `abl_inject_{off,raw,gated}` |
| 门控消融 | 带宽自调节（5.2 w/o 门控） | `loss_gate_weight=0`（无 L1 预算，g 由规划损失自由训练） | `abl_no_gate` |
| β 扫描 | UCB 保守强度（5.2） | `risk_beta=0.5/1/2`（概率量纲：σ_epis≤0.5；2 已近 risk_max 饱和，>2 无意义） | `abl_beta{05,1,2}` |
| w/o R_worst | 仅 R_exp（5.2） | `risk_beta=0`（UCB 关闭） | `abl_no_ucb` |
| 规划模式 | absolute vs GT 相对（回退臂） | `risk_plan_mode` | `abl_plan_{abs,gtrel}` |
| 安全监督消融 | 逐项去监督（5.2） | `loss_col_weight=0` / `loss_dyn_weight=0` | `abl_no_col` / `abl_no_dyn` |
| 阈值容差 | R_safe 标定分位 | `risk_safe_quantile=0.1/0.5` | `abl_q01/q05` |
| 锚定损失消融 | 风险场真实性锚定 | `loss_risk_ground_weight=0` | `abl_no_ground` |
| MC-dropout 消融 | σ_epis 实现（5.2） | `risk_mc_t=1`（无 UCB 方差） | `abl_mc1` |
| 深度来源 | midas / rcsample / lidar（§4.4） | `use_osz_midas` + `osz_dir` 指向对应 npz | `abl_osz_{midas,rcsample,lidar}` |

掩码源二选一：`use_osz_rcsample`（在线，无需导出）/ `use_osz_midas`（离线，先导出）。

### 4.3 完整示例（K 消融）

```bash
# ① 编辑 resworld_config.py：use_osz_midas=True, use_oarwm=True, mhst_k=5
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
