# OARWM-Res: Occlusion-Aware Residual World Model with Multi-Hypothesis Latent Dynamics for End-to-End Autonomous Driving

> **基于 Basework**: [ResWorld](https://github.com/mengtan00/ResWorld) (arXiv 2026.02, 已开源)
> **定位**: 端到端规划为核心，轻量残差世界模型为推理引擎，显式遮挡几何、多假设动态与可学习风险场为安全约束

---

## 1. 引言与 Motivation

### 1.1 现有瓶颈

ResWorld 提出轻量时序残差世界模型，在 BEV 特征空间直接预测未来残差，避免了 3D Occupancy 或 Gaussian Splatting 的沉重解码开销。然而其在安全关键场景中存在**从感知到决策全链路**的根本缺陷——**遮挡信息的断裂**：

> 遮挡在感知端未被显式识别，在表示端被压缩为单点估计，在决策端无法转化为保守行为——"鬼探头"（occluded pedestrian/vehicle suddenly appearing）恰恰发生在这条断裂链上。

具体表现为三个环节的失效：

1. **感知端——遮挡区特征传播黑盒化**：残差预测对所有 BEV 位置一视同仁，模型"不知道它不知道什么"。
2. **表示端——单模态确定性转移**：每个 BEV 位置输出确定性残差，遮挡区"可能出现行人/车辆/空路"的多模态本质在表示层被坍缩。
3. **决策端——规划器缺乏风险感知能力**：规划损失仅优化轨迹与真值的 L2 偏差及碰撞避免，未显式建模遮挡带来的风险分布，导致遮挡场景中过度被动或过度激进。

**核心论点**：仅修补其中某一环无法解决鬼探头。本方案要求**不确定性显式性贯穿全链路**：每一层都将"遮挡分布"以显式形式传递至下一层，直至决策端以保守方式消费；且到达决策端的必须是有**安全语义**的风险读数，而非泛泛的预测不确定度。

### 1.2 我们的思路

将 OAIAD 的**显式遮挡几何射线追踪**深度嫁接至 ResWorld 的轻量残差世界模型框架，核心思想：

> **在轻量 BEV 残差世界模型中显式分离可见区与遮挡区的动力学，将遮挡区转化为受几何约束的多假设概率空间；多假设分布经可学习风险头翻译为具有安全语义的双通道风险场，并以受控带宽回注规划器所消费的表征，使世界模型在表征层面内化鬼探头风险、规划器在决策层面自发规避。**

信息流遵循"**遮挡 → 风险 → 表征**"的因果方向：遮挡几何证据与预测分布先被翻译为风险读数，再注入共享表征——规划器"看见"的是危险本身，而非原始的遮挡扰动。

与基线的本质区别：ResWorld 是一条确定性的特征预测链；OARWM-Res 是**四条显式层叠的遮挡感知链**——几何层回答"哪里不可信"，分布层回答"可能是什么"，风险层回答"危险在哪里"，决策层回答"怎么走才安全且高效"；每一层的输出都是下一层的显式输入，多假设信息在链路上逐级传导、不被中间环节坍缩。

---

## 2. 完整方法架构

```
输入: 多视角图像序列 I_t, I_{t-1}, I_{t-2}, 自车状态 e_t, 导航指令 n_t
│
├─► 几何层 (Stage 1+2): "哪里不可信"
│     ├─► Stage 1: 图像编码与 GeoBEV 特征提取 (继承 ResWorld)
│     │     └─► BEV 特征 B_t ∈ R^(H×W×C)
│     └─► Stage 2: 显式遮挡几何 + 风险门控注入
│           ├─► OSZ 高度感知射线投射 → 遮挡掩码 M_t^occ（几何证据，供风险头消费）
│           └─► 风险门控回流: B̃_t = B_t + g ⊙ Proj([R_exp.detach(), R_worst.detach()])
│
├─► 分布层 (Stage 3): "可能是什么"
│     ├─► 可见区: 确定性残差预测 (继承 ResWorld)
│     └─► 遮挡区: 多假设随机残差转移 MHST
│           ├─► K 假设残差 ΔB^(k) + 先验 π + 不确定度 σ
│           └─► 混合分布不回流共享 BEV（col_attn 消费干净 pred_bev），
│                 以显式风险场的形式到达决策（Stage 4）
│
├─► 风险层 (Stage 4): "危险在哪里"
│     ├─► 可学习 RiskHead，输入全 detach: B_t, M, φ = σ/(1+σ)·‖ΔB‖,
│     │                       s_occ, 1/(1+d), cmd
│     ├─► 输出: μ（风险均值）+ σ_epis（认知不确定度）
│     └─► 双通道契约:
│             R_exp   = clamp(μ, 0, R_max)
│             R_worst = clamp(μ + β·σ_epis, 0, R_max)    # UCB 上界
│
├─► 决策层 (Stage 5): "怎么走"
│     ├─► BEV Planner (继承 ResWorld, 单轨迹回归)
│     └─► 安全下界: L_plan_guard = (1/T) Σ_t max(0, R_worst(τ_t) − R_safe)
│           R_safe 由碰撞正例 μ 分位数 EMA 滑窗标定
│
└─► 端到端训练 (Stage 6):
      L_total = L_plan_reg + λ_g·L_plan_guard + λ_gate·‖g‖₁
              + ω2·L_occ_halluc + ω2'·L_occ_gt + ω2c·L_risk_ground
              + ω3·L_div + ω4·L_uncertainty + λ_col·L_col + λ_dyn·L_dyn
```

**总体设计原则**（贯穿全链路）：

1. **几何显式性**：遮挡掩码由 Stage 2 的几何射线投射显式生成（非数据驱动隐式学习），全链路复用同一份 `M_t^occ`——"哪里不可信"由几何决定，可解释、可审计。
2. **不确定性显式性**：多假设分布 (π, σ, ΔB) 以显式形式沿链路旁路传导，不在中间环节坍缩为单点估计，亦不回流共享 BEV 特征（规划头消费干净的 `pred_bev`）；风险场 (R_exp, R_worst) 经零初始门控注入共享表征，与 BEV 特征共同构成规划器的完整视角。
3. **保守性可解释性**：保守行为由风险场驱动且可归因——R_worst 消费 UCB 上界 `μ + β·σ_epis`，模型对风险判断越不确定的区域，安全余量越大；Stage 5 的安全阈值 `R_safe` 由真实碰撞样本标定，安全约束具有明确的物理尺度。

---

## 3. 分阶段详细设计

### Stage 1: 图像编码与 GeoBEV 特征提取

**处理**: ResNet-50 提取多尺度图像特征 → RCSample（ResWorld 自研 view transformer，BEVDepth4D 范式）投影到 BEV：

$$B_t = \text{GeoBEV}\big(\{I_t^v\}_{v=1}^6\big) \in \mathbb{R}^{H \times W \times C}$$

网格为 **200×200 各向异性**（$x \in [-15,15]\,\text{m} @ 0.15\,\text{m}$，$y \in [-30,30]\,\text{m} @ 0.3\,\text{m}$），BEV encoder neck 输出 C=256。时序对齐在视图变换阶段完成（RCSample 以当前帧 sensor2keyego 为参考系采样历史帧），共 3 帧（当前 + 2 历史）。

**实现载体**: 继承 ResWorld 原链路（`numC_Trans=80`），无修改——保证与基线公平对比的锚点。

---

### Stage 2: 显式遮挡几何与风险门控注入

#### 2.1 遮挡几何证据提取（OSZ 流水线）

MiDaS v2.1 Small 预测 metric 深度（稀疏 LiDAR 投影深度做尺度对齐）→ 针孔反投影为 ego 3D 点云 → BEV 高度图 → **高度感知射线投射**（360°，substep=0.25 cell，n_angles ≥ 720），产生两类阴影：

- `osz_ground`: 任意高度 > 0.05 m 的占据 cell 均阻挡光线的地面层阴影；
- `osz_eye`: 仅高度 > OBSERVER_HEIGHT_M (1.2 m) 的 cell 阻挡的**严格盲区**。

取 `osz_eye` 为严格遮挡掩码 $M_t^{occ} \in \{0,1\}^{H\times W}$，可选经 HD-map 可行驶区域过滤。掩码 4D 契约 `(B,3,H,W)` = (osz_eye, osz_ground, semi)，与 BEV 网格同格、零重采样。

**实现载体**: `OSZ/` 模块（`depth_estimator.py` / `image_to_ego.py` / `bev_height_builder.py` / `ray_casting.py` / `drivable_filter.py`），离线 `{token}.npz` 导出或在线同源生成，config `use_osz_midas` / `use_osz_rcsample` 二选一。

#### 2.2 风险门控回流

遮挡几何证据（M、drivable）与动态占用 `s_occ` 不直接写入 BEV，而是作为 Stage 4 风险头的输入；**被注入共享 BEV 的是风险头受安全监督的输出**：

$$\tilde{B}_t = B_t + g \odot \text{Proj}\big(\,[\,R_{exp}.\text{detach}(),\; R_{worst}.\text{detach}()\,]\,\big)$$

| 组件 | 定义 | 初始化 | 训练信号 |
|---|---|---|---|
| `Proj` | 1×1 conv，2 → C_bev | 常规增益（Kaiming） | 规划损失 |
| `g` | 逐通道门控标量 (C_bev,) | 0 | 规划损失 |
| `R_exp, R_worst` | 风险头输出（Stage 4） | — | 安全监督，与规划损失隔离 |

**为何如此设计**：

- **内容由安全监督决定，带宽由任务损失决定**。风险读数进注入路径前 detach，规划目标只能调节**用多少**风险信息（训练 g/Proj），无法改变风险读数的**内容**——两类训练信号各司其职，风险场的语义完整性得以保持。
- **零初始化的平滑引入**。`g` 初始为 0 使训练初值下模型与无风险注入严格一致，风险信息随训练逐步引入；Proj 保持常规增益而非零初始化，使门控自训练初期即具备有效梯度，注入规模随规划需求自然增长。
- **注入的是安全语义本身**。规划器在特征层直接"看见"危险分布（而非原始遮挡扰动），规避行为作为特征驱动的策略自然涌现。

**带宽调节**：$L_{gate} = \lambda_{gate} \|g\|_1$（λ ~ 1e-5）鼓励门控只在切实降低规划损失时开启；硬上界 `|g| ≤ 1` 保持注入幅度与原 BEV 尺度同级；初始 warmup 期（约 2000 iters）冻结 `g=0`，待风险头输出由安全监督训练成型后再放开。

**实现载体**: 注入点位于 `resworld_head.py` 的 `OcclusionAwareFusion`（config `osz_inject_mode='risk_gated'`，`'off'` 等价基线作对照）；`risk_gated` 分支取 Stage 4 输出 `outs['risk_field']` 双通道作为 Proj 输入。

---

### Stage 3: 遮挡感知多假设残差世界模型 (MHST)

ResWorld 的残差世界模型在 latent 空间做相邻帧差分：`res_latent_query = bev_embed[:-1] − bev_embed[1:]`，经 tokenfuser 融合回 BEV 得 `pred_bev`。我们将遮挡区升级为多假设转移：

**3.1 位置分类与路由**：按 $M_t^{occ}$ 将 BEV 位置分为可见区 $\mathcal{V}_t$ 与遮挡区 $\mathcal{O}_t$；掩码仅注入当前帧。

**3.2 可见区**：确定性残差预测（继承 ResWorld），直接取 `pred_bev` 原值。

**3.3 遮挡区——多假设随机残差转移 MHST**：对每个遮挡位置引入隐含内容类型 $c^{(x,y)} \in \{1..K\}$（空路/静止车辆/运动车辆/行人/骑行者），输出混合分布：

$$p\big(B_{t+1}(x,y) \,\big|\, B_t\big) = \sum_{k=1}^K \pi^{(x,y),k}_t \cdot \mathcal{N}\Big(B_t(x,y) + \Delta B^{(k)}_{t+1}(x,y),\; \Sigma_k^2\Big)$$

$$\pi^{(x,y)} = \text{Softmax}\big(g_{\text{prior}}(\{B_t(x',y')\}_{(x',y')\in\mathcal{N}(x,y)})\big)$$

- 先验网络：`[pred_bev | osz_mask]` → 1×1 conv → **多尺度膨胀邻域聚合**（3×3 conv, dilation 1/2/4，感受野 3/5/9 格）→ 1×1 conv → K 维 softmax；
- K 个假设残差：共享骨干（1×1+3×3 conv）→ K 个独立 expert 卷积分支（MoE 式），expert 输出层零初始化（训练初值下遮挡区补丁严格等价 `pred_bev`）；
- 不确定度：$\sigma = \text{softplus}(s) + \Sigma_{\min}$，$\Sigma_{\min}=0.1$（不确定性下限，防过度自信）；
- **输出有界性**：ΔB 逐元素限幅 ±10、σ 上限 10、风险场输出限幅——保证似然约束强度不随 σ 增大而衰减、下游规划梯度有界。

**消费路径**：混合分布**不回流共享 BEV**——`col_attn` 消费干净的 `pred_bev`；分布经 aux 旁路只以显式风险场的形式到达决策层。风险场是分布的**无梯度测量**：π/σ/ΔB/s_occ 进入风险场前全部 detach，规划器经 `grid_sample` 的采样坐标支路获得风险场空间梯度——规划器学习**规避**高风险区域，而分布的学习完全由自身的分布塑造损失驱动，两类目标互不干扰。

**实现载体**: `resworld_head.py` 的 `OcclusionMHSTHead`（L215），嫁接在 `pred_bev = tokenfuser(...) + bev_navi_embed` 之后、`col_attn` 之前；掩码门控保证可见区与全零掩码时为恒等映射（`use_oarwm=False` 严格等价基线）。config: `mhst_k=5`（消融 1/3/5/10）、`mhst_sigma_min=0.1`、`mhst_delta_clamp=10.0`、`mhst_sigma_max=10.0`。

**3.4 假设的语义结构化**：为使假设具备可审计的语义（空 / 静态物 / 动态物 / 行人），在每个假设上附加**语义占用锚定头**——以动态占用监督对齐各假设的语义内容，验证假设语义分离的可行性。当语义分离成立且下游规划受益时，可将假设本体迁移至**语义占用空间**：每个假设即一张 (H,W,D) 的占用配置图，K 的取值由此获得明确物理语义（如 K=2 即"空与存在智能体"两态），风险场直接自语义占用读出，风险监督与假设空间天然对齐。

---

### Stage 4: 可学习风险评估头 (RiskHead)

#### 4.1 输入（全部 detach）

| 输入 | 含义 | 来源 |
|---|---|---|
| `B_t.detach()` | 当前 BEV 特征 | Stage 1 |
| `M`, `drivable` | 遮挡掩码与可行驶约束 | Stage 2 |
| `φ_content = σ/(1+σ)·‖ΔB‖` | 分布层内容变化强度 | Stage 3 |
| `s_occ` | 遮挡区动态占用概率 | 占用头（Stage 6, L_occ_gt） |
| `1/(1+d)` | 到最近动态物的距离接近度 | 几何派生 |
| `cmd_embed` | 导航指令嵌入 | 规划头共享 |

风险头是**安全度量器**，其输入全部 detach：风险的读出不反向塑造感知与预测主干，感知层与风险层保持单向信息流。

#### 4.2 输出与消费

$$\mu = \text{RiskHead}(\cdot), \qquad \sigma_{epis} = \text{参数不确定度统计}$$

$$R_{exp} = \text{clamp}(\mu,\ 0,\ R_{max}), \qquad R_{worst} = \text{clamp}(\mu + \beta \cdot \sigma_{epis},\ 0,\ R_{max})$$

- `R_exp`：风险均值（"这里有多危险"），供常规消费；
- `R_worst`：**UCB 上界**（"我对这个判断有多没底"），供 Stage 5 安全下界消费——模型对风险判断越不确定的区域，安全余量自动越大，分布外场景下规划器自适应趋于保守。

**为何可学习**：风险的组合方式具有场景相关性，相较于固定权重的规则读数，数据驱动的风险头能以监督信号自动配比各几何证据与预测分布的贡献，获得更准确的风险定位；其输出同时接受安全监督（§4.4）与任务反馈（门控带宽，§2.2），语义与实用性同时保证。

#### 4.3 σ_epis 的实现

**当前实现——MC-dropout**：风险头末两层设置 dropout，训练与推理期间进行 T=4 次随机前向，$\mu = \text{mean}$、$\sigma_{epis} = \text{std}$。以隐式参数集成刻画模型对风险判断本身的把握，实现轻量、无需额外网络结构。

**演进判断**：若实验显示需要更强的分布外感知（σ_epis 应随输入偏离训练分布的程度而放大），将输出层替换为**谱归一化卷积 + 随机傅里叶特征（RFF）头**——谱归一化约束网络的 Lipschitz 常数，RFF 输出层使不确定度具有距离感知特性，输入离训练分布越远、σ_epis 越大。

#### 4.4 安全监督

碰撞在 nuScenes 上正例极稀疏，仅作稀疏硬锚点；**稠密监督主干由动态占用承担**：

$$L_{col} = \text{BCE}\big(\mu,\ C_{gt};\ \text{pos\_weight}\big), \qquad L_{dyn} = \text{BCE}\big(\mu,\ S_{dyn};\ \text{pos\_weight}\big)$$

- `C_gt`：未来碰撞占用（硬真值）；
- `S_dyn`：动态物占用栅格 + **沿目标朝向向外扩 forward margin**（默认 2 m）——使"车头正前方即将扫过的区域"提前获得高风险，这正是鬼探头的物理形态；

**σ_epis 的语义来源**：σ_epis 由 dropout 方差的统计直接刻画；预测残差的校准（e²）服务于分布层的 σ（L_uncertainty）与内容变化强度 φ（L_risk_ground），与认知不确定度各司其职。

#### 4.5 平滑与有界

输出后接高斯平滑（kernel=5, σ=1.0）——风险场空间平滑，`grid_sample` 采样时规划器梯度稳定；softplus 保证非负且处处可微。

**实现载体**: 当前 `build_risk_field`（`resworld_head.py` L135）输出双通道 `outs['risk_field']`。风险头以同构的 R_exp/R_worst 双通道输出对接（Stage 5 接口不变）；内部组合读数由可学习的读数层承担。

---

### Stage 5: 风险约束规划

规划器（ResWorld 单轨迹回归）输出预测轨迹 $\tau$，以**绝对阈值安全下界**约束：

$$L_{plan\_guard} = \frac{1}{T}\sum_{t=1}^T \max\big(0,\ R_{worst}(\tau_t) - R_{safe}\big), \qquad R_{safe} = \text{EMA}_{0.999}\Big[\text{quantile}_{0.1}\big(\mu \,\big|\, \text{碰撞格}\big)\Big]$$

**为何如此设计**：

- **阈值有物理尺度**：`R_safe` 由碰撞样本的风险读数分位数标定，安全约束的介入条件与真实危险水平直接挂钩；随训练以 EMA 滑窗更新，与风险头同步收敛。
- **稀疏介入、模仿主导**：正常行驶下轨迹风险远低于 R_safe，该项不激活，轨迹回归（贴合人类驾驶示范）独占训练；仅当轨迹接近碰撞级风险时安全约束介入。安全与模仿目标互不稀释。
- **消费 UCB 上界**：以 R_worst 评估轨迹安全性——风险判断不确定的区域自动要求更大的安全余量。

**双层风险感知**：风险规避行为主要由**特征层**驱动（Stage 2 门控注入使规划器在前向传播中直接感知危险分布，规避作为特征驱动的策略涌现）；本节的损失项作为**损失层安全下界**，防止极端失效。两层互补：前者细粒度、持续作用，后者稀疏、兜底。

**实现载体**: `loss()` 中 risk-plan 分支（config `use_risk_plan`、`risk_plan_mode='absolute_hinge'`、`risk_safe_quantile=0.1`、`loss_plan_risk_weight=0.1`）；风险项前 2 epoch 关闭并线性 ramp，待风险头输出稳定后再介入。

---

### Stage 6: 端到端训练目标

$$L_{total} = \underbrace{L_{plan\_reg}}_{\text{轨迹回归 (w=10)}} + \underbrace{\lambda_g L_{plan\_guard}}_{\text{安全下界}} + \underbrace{\lambda_{gate}\|g\|_1}_{\text{注入带宽}} + \underbrace{\omega_2 L_{occ\_halluc} + \omega_2' L_{occ\_gt}}_{\text{遮挡想象 (静态/动态)}} + \underbrace{\omega_{2c} L_{risk\_ground} + \omega_3 L_{div} + \omega_4 L_{uncertainty}}_{\text{分布塑造}} + \underbrace{\lambda_{col} L_{col} + \lambda_{dyn} L_{dyn}}_{\text{风险安全监督}}$$

- **$L_{occ\_halluc}$（静态内容想象）**：遮挡区条件分布 p(B_{t+1}|B_t) 以 t+Δt 曝光真值做混合模型对数似然，EM 式联合拟合 π/ΔB/σ；
- **$L_{occ\_gt}$（动态内容想象）**：`gt_bboxes_3d` 栅格化 BEV 动态占用 `S_gt`，遮挡区占用头 `s_occ` 的 BCE（pos_weight=5.0）。**实现载体**: `_rasterise_boxes_bev`（`resworld_head.py` L1033）；Stage 4 的 `S_dyn` 即该栅格化加 forward margin 的产物，同一份数据、同一处代码；
- **$L_{risk\_ground}$（内容对准）**：$(\frac{\sigma}{1+\sigma}\|\Delta B\| - \text{clip}(e^2))^2$——分布层联合产物与曝光误差对齐，使内容变化强度成为曝光变化的无偏读数，遮挡区外不监督；
- **$L_{div}$（假设多样性）**：假设残差两两余弦相似度最小化（仅遮挡区，clamp ≥ 0），维持 K 假设方向分离；
- **$L_{uncertainty}$（不确定度校准）**：$|\sigma - \|e\|^2|$——σ 与真实曝光误差对齐，与 L_occ_halluc 共享同一份曝光数据；
- **$L_{col}$ / $L_{dyn}$（风险安全监督）**：见 §4.4；
- **$L_{plan\_guard}$（安全下界）**：见 Stage 5；
- **$L_{gate}$（注入带宽）**：见 §2.2。

**权重起点**（L_col 为 1.0 标尺）：

| 项 | 权重 | 项 | 权重 |
|---|---|---|---|
| L_col | 1.0 | L_dyn | 1.0 |
| L_plan_guard | 0.1 | ‖g‖₁ | 1e-5 ~ 1e-4 |
| L_occ_halluc | 1.0 | L_occ_gt | 1.0 |
| L_risk_ground | 0.1 | L_div | 0.1 |
| L_uncertainty | 1.0 | | |

---

## 4. 与 Basework 的对比

| 模块 | ResWorld (arXiv 2026) | OARWM-Res (本设计) |
|---|---|---|
| 场景表示 | BEV Feature Map (200×200×256, 各向异性网格) | + OSZ 高度感知遮挡掩码（同格零重采样） |
| 世界模型 | 确定性 latent-token 残差预测 | 可见区确定性 + 遮挡区多假设随机转移（分布经旁路传导，不回流共享 BEV） |
| 遮挡处理 | 隐式（数据驱动） | 显式几何约束 + 多假设想象 |
| 风险度量 | 无显式风险场 | 可学习双通道风险场（μ + σ_epis UCB 上界），碰撞/动态占用监督 |
| 风险进入决策的方式 | — | 特征层：零初始门控受控回流；损失层：绝对阈值安全下界 |
| 规划目标 | L2 模仿 + 碰撞避免 | 轨迹回归主导 + 碰撞级阈值安全约束（R_safe 由碰撞样本标定，EMA 滑窗） |
| 训练监督 | BEV 重建损失 | + 遮挡曝光自监督 + 动态占用 + 假设多样性 + 不确定度校准 + 内容对准 + 风险安全监督 |
| 核心能力 | 时序残差预测 | 遮挡区内容想象 + 有安全语义的风险感知 + 不确定度驱动的保守决策 |

---

## 5. 预期实验与贡献

### 5.1 数据集与评估

- **nuScenes**: 开环规划指标（L2 @ 1s/2s/3s, Collision Rate）；
- **遮挡子集评估**: `filter_occ_subset.py` 从 OSZ npz（occ_frac > 20%）筛选 token 子集重算开环 L2——风险感知的价值必须在风险存在的地方度量；
- **近遮挡减速行为统计**: `traj_behavior_stats.py` 统计掩码边界 5 m 内轨迹步的速度（预测 vs GT）——"接近遮挡物主动减速"的行为证据；
- **Bench2Drive 闭环**作为后续验证（Merging / Emergency Brake / Give Way）。

### 5.2 消融实验设计

| arm | 配置 | 检验什么 |
|-----|------|----------|
| `w/o 遮挡掩码` | 关闭 OSZ 掩码 | 几何先验的必要性 |
| `K ∈ {1,3,5,10}` | MHST 假设数 | 多假设数量的影响 |
| `w/o R_worst` | 仅 R_exp | 最坏假设（UCB 上界）保留的必要性 |
| `β ∈ {0.5,1,2}` | UCB 系数扫描 | 保守强度与 L2 的权衡 |
| `w/o 风险注入` | `osz_inject_mode='off'` | 特征层风险感知的贡献 |
| `w/o 门控` | 注入不加 g | 带宽自调节的必要性 |
| `w/o L_col` / `w/o L_dyn` | 逐项去安全监督 | 各路监督贡献 |
| `MC-dropout vs RFF 头` | σ_epis 实现对比 | 认知不确定度实现的影响 |
| `w/ CVaR 尾部约束` | 沿程风险尾部变体 | 尾部安全约束的增量 |
| `w/o 曝光自监督` / `w/o σ 校准` | 监督逐项移除 | 分布塑造各项贡献 |
| `语义锚定头 on` | §3.4 锚定头 | 假设语义可分性 |

### 5.3 核心贡献

1. 首次将 OSZ 高度感知显式遮挡几何注入轻量残差世界模型，提出 OA-RWM；
2. 提出遮挡区多假设随机残差转移机制（MHST），在 BEV 特征空间实现遮挡区内容的概率化想象；
3. 提出可学习风险场（RiskHead + UCB）：风险由碰撞/动态占用监督获得安全语义，认知不确定度刻画风险判断的把握，双通道 (R_exp, R_worst) 契约贯穿规划；
4. 提出风险受控回流机制：零初始门控 + 单向 detach 的"内容/带宽分离"，使世界模型在表征层面内化鬼探头风险，且注入规模与规划需求自适应收敛；
5. 提出绝对阈值安全下界规划目标：R_safe 由碰撞正例标定，安全约束稀疏介入、与模仿目标解耦；
6. 提出遮挡曝光自监督与不确定度校准联合训练策略，无需额外标注即可监督遮挡区想象。