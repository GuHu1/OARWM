# OARWM-Res: Occlusion-Aware Residual World Model with Multi-Hypothesis Latent Dynamics for End-to-End Autonomous Driving

> **基于 Basework**: [ResWorld](https://github.com/mengtan00/ResWorld) (arXiv 2026.02, 已开源)  
> **定位**: 端到端规划为核心，轻量残差世界模型为推理引擎，显式遮挡几何与多假设动态为安全约束  

---

## 1. 引言与 Motivation

### 1.1 现有瓶颈

ResWorld 提出了一种轻量的时序残差世界模型（Temporal Residual World Model），通过在 BEV 特征空间直接预测未来残差，避免了 3D Occupancy 或 Gaussian Splatting 的沉重解码开销，在 8×RTX 3090 (24GB) 上即可高效训练。然而，其在安全关键场景中存在一个贯穿**从感知到决策全链路**的根本缺陷——**遮挡信息的断裂**：

> 遮挡在感知端未被显式识别，在表示端被压缩为单点估计，在决策端无法转化为保守行为——"鬼探头"（occluded pedestrian/vehicle suddenly appearing）恰恰发生在这条断裂链上：感知端不知道哪里不可信（黑盒），表示端把"可能有什么"压成"平均的什么"（单模态），决策端因此没有理由为未知留出安全余量（被动）。

这条断裂链具体表现为三个环节的失效：

1. **感知端——遮挡区特征传播黑盒化**：ResWorld 的残差预测对所有 BEV 空间位置一视同仁，缺乏显式机制区分"可见区"与"遮挡区"。遮挡区的未来特征完全由数据驱动的隐式插值推断，模型"不知道它不知道什么"，无法保证对"鬼探头"等突发风险的保守性。
2. **表示端——单模态确定性转移**：ResWorld 对每个 BEV 位置输出确定性残差（或单一高斯噪声），无法覆盖遮挡区"可能出现行人/车辆/空路"的多模态本质。当遮挡区内容在训练分布中罕见时，模型倾向于输出"平均化"的模糊特征，多模态信息在表示层即被坍缩，下游无从恢复。
3. **决策端——规划器缺乏主动感知能力**：ResWorld 的规划损失仅优化轨迹与真值的 L2 偏差及碰撞避免，未显式建模"通过自车动作减少遮挡不确定性"的主动感知能力，导致系统在遮挡场景中过度被动或过度激进。

**核心论点**：仅修补其中某一环无法解决鬼探头——感知端识别出遮挡（Stage 2）、表示端建模多模态（Stage 3），若决策端仍只消费单点估计（期望特征），多模态信息在通往决策的路上仍会丢失。因此本方案要求**不确定性显式性贯穿全链路**：每一层都将"遮挡分布"以显式形式传递至下一层，直至决策端以保守方式消费（见 §2 设计原则）。

### 1.2 我们的思路

本方案将 OAIAD 的**显式遮挡几何射线追踪**与**矢量化遮挡表示**深度嫁接至 ResWorld 的轻量残差世界模型框架中，提出 **OARWM-Res（Occlusion-Aware Residual World Model）**。核心思想是：

> **在轻量 BEV 残差世界模型中显式分离可见区与遮挡区的动力学，将遮挡区从"确定性残差的受害者"转化为"受几何约束的多假设概率空间"，并让多假设分布以显式形式贯穿感知→表示→风险→决策全链路，最终构建主动感知的安全规划器。**

与基线的本质区别：ResWorld 是"一条确定性的特征预测链"，OARWM-Res 是**四条显式层叠的遮挡感知链**——几何层回答"哪里不可信"，分布层回答"可能是什么"，风险层回答"危险在哪里"，决策层回答"怎么走才保守且高效"；每一层的输出都是下一层的显式输入，多假设信息在链路上逐级传导、不被中间环节坍缩（分布经 aux 旁路传导至风险与决策层，且不回流共享 BEV 特征，见 Stage 3）。

---

## 2. 完整方法架构

OARWM-Res 包含六大阶段，我们将遮挡信息以显式形式（掩码 / 多假设分布 / 风险场）传递给下一层：

```
输入: 多视角图像序列 I_t, I_{t-1}, I_{t-2} (k=2 历史帧), 自车状态 e_t, 导航指令 n_t
│
├─► 几何层 (Stage 1+2): "哪里不可信"
│     ├─► Stage 1: 图像编码与 GeoBEV 特征提取 (继承 ResWorld)
│     │     └─► BEV 特征 B_t ∈ R^(H×W×C)
│     └─► Stage 2: 显式遮挡几何注入 (新增 ★)
│           ├─► OSZ 高度感知射线投射 → 遮挡掩码 M_t^occ ∈ {0,1}^(H×W) (osz_eye)
│           └─► 遮挡感知 BEV 特征 B̃_t = B_t + E_occ ⊙ M_t^occ（残差式，zero-init 起步等价基线）
│           ⬇ 传递: M_t^occ（逐层复用，4D 契约 (B,3,H,W)）
│
├─► 分布层 (Stage 3): "可能是什么"
│     ├─► 可见区: 确定性残差预测 (继承 ResWorld)
│     └─► 遮挡区: 多假设随机残差转移 MHST (新增 ★)
│           ├─► K 假设残差 ΔB^(k) + 先验 π + 不确定度 σ   （旁路 aux）
│           └─► 多假设不回流 BEV 特征（col_attn 消费干净 pred_bev），仅经风险场进入决策
│           ⬇ 传递: (π, σ, ΔB^(k)) 经 outs['mhst'] 旁路传导（不在表示层坍缩）
│
├─► 风险层 (Stage 4): "危险在哪里"
│     ├─► 多假设风险分解 R^(k) = w_σ·σ·‖ΔB^(k)‖·1[M]（语义不可知代理）
│     ├─► 双输出: 期望风险 R_exp = Σ_k π_k R^(k) 与 最坏风险 R_worst = max_k R^(k)
│     └─► 不确定性+占用驱动遮挡放大 R̃ = R·(1+β·U·M)·(1+γ·s_occ·M), U = H(π)/logK
│     └─► 真实性锚定: 原始强度 σ/(1+σ)·‖ΔB‖ 与暴露误差 e² 对齐（L_risk_ground）
│           ⬇ 传递: (R_exp, R_worst) 至决策层（最坏假设保留, 对齐 Minimax）
│
├─► 决策层 (Stage 5): "怎么走"
│     ├─► BEV Planner (继承 ResWorld, 单轨迹回归)
│     ├─► 风险加权轨迹损失 L_plan_risk（沿程 R_worst, Minimax 语义）(新增)
│     ├─► 沿程风险尾部 L_plan_cvar（CVaR 语义）(新增)
│     └─► 信息增益奖励 L_plan_info = max(0, R_exp(τ_end) − R_exp(τ_end_gt))（末端风险差, 相对上界）(新增)
│     └─► 预测轨迹 τ（风险规避, 非候选筛选）
│
│   核心：risk/cvar 为 GT 风险上界相对形式 max(0, R(τ_pred) − R(τ_gt) − μ)、
│   info 为末端风险差 max(0, R(τ_end_pred) − R(τ_end_gt))（无 μ）——
│   τ_pred ≈ τ_gt 时零梯度，不伤开环 L2；只在"偏离 GT 且更危险"时介入。
│
└─► 自监督闭环 (Stage 6): 端到端训练
      ├─► 遮挡区自监督想象损失 L_occ_halluc（静态，§6.2）+ 动态占用监督 L_occ_gt（§6.2b）
      ├─► 风险场真实性锚定 L_risk_ground（§6.2c，零额外数据）
      ├─► 多假设多样性损失 L_div（对抗 L_occ_halluc 静态偏置）
      ├─► 不确定性校准损失 L_uncertainty（σ 校准, 训练契约）
      └─► 规划损失 L_plan = L_plan_reg + 相对风险项（含 L_info）
```

---

**总体设计原则**（贯穿全链路的三个约束）：

1. **几何显式性**：遮挡掩码由 Stage 2 的几何射线投射显式生成（非数据驱动隐式学习），全链路复用同一份 `M_t^occ`——"哪里不可信"由几何决定，可解释、可审计。
2. **不确定性显式性**：多假设分布 (π, σ, ΔB) 以显式形式（`outs['mhst']`）沿链路旁路传导，**不得在中间环节坍缩为单点估计**——分布**不回流共享 BEV 特征**（`col_attn` 消费干净 `pred_bev`，几何精修不受多假设扰动），而必须完整到达风险层与决策层（以显式风险场的形式）。
3. **保守性可解释性**：遮挡区的风险放大由**不确定性驱动**（π 熵 U 而非常数），最坏假设 (`R_worst`) 显式保留至决策层（Minimax）——保守行为可归因于"哪里不确定、有多不确定"。

---

## 3. 分阶段详细设计

### Stage 1: 图像编码与 GeoBEV 特征提取 (继承 ResWorld)

**输入**: 当前帧及 k=2 历史帧的多视角图像 $I_t, I_{t-1}, I_{t-2} = \{I_{t}^{v}\}_{v=1}^{6}$，图像分辨率 256×704。

**处理**:
1. **图像编码**: 使用 **ResNet-50** 提取多尺度图像特征。ResWorld 沿用 BEVFormer/BEVDet 的图像编码范式，通过 FPN 输出 1/8, 1/16, 1/32 多尺度特征图。
2. **GeoBEV 投影**: 通过 **RCSample**（ResWorld 自研 view transformer，BEVDepth4D 范式，LSS 类深度显式投影）将多视角图像特征投影到 BEV 空间：
   $$B_t = \text{GeoBEV}(\{I_{t}^{v}\}_{v=1}^{6}) \in \mathbb{R}^{H \times W \times C}$$
   其中 $H \times W$ 为 BEV 网格分辨率（**实际为 200×200 各向异性网格**：`grid_config` 中 $x \in [-15,15]\,m @ 0.15\,m$、$y \in [-30,30]\,m @ 0.3\,m$，见 `resworld_config.py`），$C$ 为特征维度（`numC_Trans=80` → BEV encoder neck 输出 **256**）。
3. **时序 BEV 对齐**: 利用自车运动参数（旋转 $R_t$、平移 $t_t$）将历史 BEV 特征 $B_{t-1}, B_{t-2}$ 对齐到当前坐标系（实际配置 `align_after_view_transfromation=False`，对齐在**视图变换阶段**完成——RCSample 以当前帧 `sensor2keyego` 为参考系生成采样网格，将历史帧特征直接采样到当前 ego 系；`shift_feature` warp 仅在 `align_after_view_transfromation=True` 时启用）：
   $$B_{t-i}^{\text{align}} = \text{Warp}(B_{t-i}, R_t, t_t), \quad i=1,2$$
   历史帧配置 `multi_adj_frame_id_cfg=(1, 1+2, 1)` → 共 3 帧（当前 + 2 历史，即 k=2）。

**输出**: 对齐后的时序 BEV 特征 $\{B_{t-2}^{\text{align}}, B_{t-1}^{\text{align}}, B_t\}$。

**作用**: 建立统一的俯视空间表征，将多视角时序图像压缩为结构化的 BEV 特征图，便于世界模型在 BEV 空间进行高效推演。

**为何保留**: ResWorld 的 GeoBEV 编码器经过验证，在 nuScenes 上能高效提取场景语义与几何信息；ResNet-50 在计算效率与表征能力之间取得平衡，且与 ResWorld 原配置一致，确保公平对比。

---

### Stage 2: 显式遮挡几何注入与 BEV 掩码生成 (新增 ★)

**输入**: 6 路环视图像 $\{I_t^v\}_{v=1}^6$ 与 LiDAR 点云 $L_t$（实际为深度尺度对齐真值与相机失效兜底，设计上必需），相机标定 $\{K^v, T_{\text{cam2ego}}^v\}_{v=1}^6$，自车状态 $e_t$。

**处理**:
1. **Metric 深度估计**: 对每路相机图像预测 metric 深度图 $D_t^v$。实现采用 **MiDaS v2.1 Small**，模型与权重均从本地加载，输出逆深度（大值=近），以稀疏 LiDAR 投影深度为真值做尺度对齐（`OSZ/modules/depth_estimator.py::align_to_lidar`）：
   - 自动检测**线性**（$d_{\text{metric}} = s \cdot d_{\text{rel}} + b$）与**逆深度**（$d_{\text{metric}} = s / d_{\text{rel}} + b$，线性拟合 scale<0 时切换）两种模型族，最小二乘拟合——MiDaS 逆深度自动走 inverse 分支；
   - 拟合退化（$|s|$ 过小）或逆模型 shift 过大时回退 robust median-ratio 估计；输出 clip 到 $[0, 70]\,m$（`MAX_METRIC_DEPTH_M`）；LiDAR 有效点 < 20（`MIN_ALIGN_POINTS`）时同样回退；
   - 模型/repo 缺失时回退 `MockDepthEstimator`（返回 LiDAR densified 深度）；**无 LiDAR 输入时仅输出相对深度，不可用于反投影**（退化路径为 LiDAR densified 兜底）。
2. **反投影为 Ego 3D 点云**: 将深度图反投影到自车坐标系（`OSZ/modules/image_to_ego.py::depth_map_to_ego_points`，标准针孔反投影，非像素坐标乘深度）：
   $$x_{\text{cam}} = \frac{u - c_x}{f_x} \cdot D_t^v(u,v), \quad y_{\text{cam}} = \frac{v - c_y}{f_y} \cdot D_t^v(u,v), \quad z_{\text{cam}} = D_t^v(u,v)$$
   经相机内参 $K^v = \{f_x, f_y, c_x, c_y\}$ 与外参 $T_{\text{cam2ego}}^v$ 的齐次变换得到统一的三维观测点云：
   $$P_t^v = T_{\text{cam2ego}}^v \cdot [x_{\text{cam}}, y_{\text{cam}}, z_{\text{cam}}, 1]^{\top}$$
   仅保留 $D \in (0, 70]\,m$ 且 $z_{\text{ego}} \in [Z_{\text{MIN}}, Z_{\text{MAX}}] = [0.8, 3.0]\,m$ 的点（地面/高楼过滤）。
3. **BEV 高度图构建**: 将点云聚合到 BEV 网格，记录每个网格的最高高度（`OSZ/modules/bev_height_builder.py::build_bev_height_max`，`np.maximum.at` 聚合）：
   $$H_t(x,y) = \max_{(x,y,z) \in \text{cell}(x,y)} z$$
   输出 BEV 高度图 $H_t \in \mathbb{R}^{H \times W}$。网格由 `OSZ/config.py` 统一定义并对齐 ResWorld `grid_config`：**200×200 各向异性网格**（$x \in [-15,15]\,m @ 0.15\,m$、$y \in [-30,30]\,m @ 0.3\,m$；`indexing='ij'`，axis-0=ego-x 前进方向，axis-1=ego-y 左侧）。
4. **高度感知射线投射**: 以自车位置为中心做 360° 射线投射（`OSZ/modules/ray_casting.py::cast_osz_height_aware`，向量化实现，`substep=0.25` cell 步进、`n_angles ≥ 720`），得到两类遮挡阴影：
   - `osz_ground`: 任意高度 **> 0.05 m** 的占据 cell 均阻挡光线（二值行为，非地面阈值）的**地面层阴影**；
   - `osz_eye`: 仅高度 **> `OBSERVER_HEIGHT_M`（默认 1.2 m）** 的 cell 阻挡光线的**严格盲区**。
   射线命中首个遮挡物后，其后方 cell 标记为阴影（遮挡物自身 cell 不标记）；自车周围 `EGO_CLEARANCE_RADIUS_M=1.0 m` 半径内高度清零，避免自遮挡。`osz_ground & ~osz_eye` 为"半透明盲区"（semi），风险低于完全盲区。
5. **可行驶区域过滤（可选）**: 将 OSZ 限制在 HD-map 可行驶区域内（`OSZ/modules/drivable_filter.py`：实现为 `osz ∩ drivable_mask`）。`drivable_mask` 由 HD-map 图层 `drivable_area + carpark_area` 光栅化（PIL 多边形绘制，多边形顶点绕 ego-yaw 旋转至 ego 系）扣除 `walkway / ped_crossing` 后，按 `DEFAULT_DRIVABLE_DILATION_M=1.5 m` 膨胀；HD-map 缺失时回退全 True。
6. **BEV 遮挡掩码**: 取 `osz_eye` 作为严格遮挡掩码：
   $$M_t^{occ}(x,y) = \text{osz\_eye}_t(x,y) \in \{0, 1\}^{H \times W}$$
7. **遮挡感知 BEV 特征（OARWM 模型端实现）**: 特征注入不在 OSZ 模块内完成——OSZ 流水线输出 `bev_height`、`osz_ground`、`osz_eye`、`semi` 与 `drivable_mask`。OARWM 以掩码通道拼接为输入构建增强 BEV 特征：
   $$\tilde{B}_t = B_t + E_{occ} \odot M_t^{occ}$$
   其中 $E_{occ} \in \mathbb{R}^{C}$ 为可学习的遮挡类型偏移（由掩码通道 `(osz_eye, osz_ground, semi)` 经 1×1 Conv 生成，区分严格盲区 / 半透明盲区 / 动态变化遮挡），通过广播与 BEV 特征逐元素相乘。
   
   **残差式注入（zero-init）**：遮挡感知特征为残差形式 $B_t + E_{occ} \odot M_t^{occ}$——遮挡区原始特征的空间信息完整保留，$E_{occ}$ 仅学习"遮挡偏移"（替换式会以位置无关的嵌入抹除遮挡区特征，破坏下游采样）；偏移头 zero-init 使训练初始严格等价基线（遮挡区零扰动），随训练渐进学习。

**输出**: OSZ 模块输出 `bev_height`、`osz_ground`、`osz_eye`、`semi`、`drivable_mask`（200×200，与 ResWorld 网格同格）；OARWM 模型端输出遮挡感知 BEV 特征 $\tilde{B}_t$（`OcclusionAwareFusion`）。

**网格对齐**: OSZ 的 BEV 网格已直接对齐 ResWorld `grid_config`——$x \in [-15,15]\,m @ 0.15\,m$、$y \in [-30,30]\,m @ 0.3\,m$（200×200 **各向异性**），定义于 `OSZ/config.py`（单一来源，需与 `resworld_config.py::grid_config` 保持同步）。OSZ 掩码与 RCSample 输出（200×200 `grid_config` 网格）同格，**导出与数据加载零重采样**；head 内 BEV 特征经 `bev_encoder` 下采样为 100×100，Stage 2/3/4 注入点统一 `F.interpolate(mode='nearest')` 对齐（见 Stage 3.1）。

- **OSZ 深度估计**：**MiDaS v2.1 Small**

- **OSZ 掩码接入（双源）**：上述几何流程（深度 → 反投影 → 射线投射）对两种掩码源相同——**离线**：MiDaS 深度导出 `{token}.npz`（`bev_height / osz_ground / osz_eye / semi / drivable_mask`，200×200），由数据管线按 token 加载；**在线**：模型自身 RCSample 深度头实时生成掩码（`build_osz_mask_online`，与主训练同源，无需导出）。两源输出通道序（`osz_eye, osz_ground, semi`）与网格约定完全一致。

**作用**: 将 OSZ 生成的显式几何先验注入 BEV 特征空间，使世界模型明确知道"哪些位置的信息不可信/需要想象"。

**为何如此设计**:
- 若仅依赖隐式世界模型学习遮挡，需要海量遮挡场景数据才能收敛；显式掩码提供了**几何归纳偏置**。
- 不同于依赖车辆 3D 检测框的解析遮挡方法，OSZ **不依赖外部检测框**，直接从原始传感器数据建 BEV 高度图并做 360° 射线投射，使整个 pipeline 保持端到端可训练。
- 通过 `osz_ground` / `osz_eye` 的分层设计，可以在"严格盲区"和"地面层保守阴影"之间做细粒度风险加权。

---

### Stage 3: 遮挡感知残差世界模型 (改进核心 ★)

ResWorld 的残差世界模型通过时序 BEV 特征预测未来残差：
$$\hat{B}_{t+1} = B_t + \Delta B_{t+1}, \quad \Delta B_{t+1} = f_{RWM}(\{B_{t-2}^{\text{align}}, B_{t-1}^{\text{align}}, B_t\})$$
我们将其扩展为**遮挡感知多假设残差世界模型（OA-RWM）**。

**与 Basework 实现的对接（实现载体）**: ResWorld 的残差并非逐 BEV 位置预测——`resworld_head.py` 先将 BEV 特征经 TokenLearner 压缩为 latent tokens，在 **latent 空间**做相邻帧差分（`res_latent_query = bev_embed[:-1] - bev_embed[1:]`），经 latent decoder 后由 tokenfuser 融合回 BEV（`pred_bev = tokenfuser(...) + bev_navi_embed`，残差连接）。因此 MHST 的**实现载体**为：在 `pred_bev` 输出后、`col_attn` 之前按掩码加门控多假设残差头（`OcclusionMHSTHead`）。**消费路径**：`col_attn` 以初始轨迹为 reference 在**干净的 `pred_bev`** 上采样——MHST 的融合输出有意不进共享 BEV 特征（几何精修不被多假设扰动），多假设分布 (π, σ, ΔB) 经 aux 旁路只以**显式风险场**的形式到达决策层（§4 风险场）。可见/遮挡路由以 Stage 2 的掩码为准；3.1-3.3 的逐位置公式为概念定义，实现按门控残差头对接。

#### 3.1 BEV 位置分类与路由

根据遮挡掩码 $M_t^{occ}(x,y)$，将 BEV 空间位置分为两类：
- **Visible Positions** ($\mathcal{V}_t$): $M_t^{occ}(x,y) = 0$，可直接观测；
- **Occluded Positions** ($\mathcal{O}_t$): $M_t^{occ}(x,y) = 1$，需要想象。

**实现**：路由以 Stage 2 的 `osz_mask` 为准——4D 契约 `(B,3,H,W)`，取 `osz_eye` 通道作门控（`mask[:, :1]`）；掩码**仅注入当前帧 t**（掩码在各帧自身 ego 系导出，与统一到当前 ego 系的三帧特征不匹配，历史帧 warp 引入孔洞/重采样误差），head 内 `F.interpolate(mode='nearest')` 对齐 BEV 分辨率。

#### 3.2 可见区：确定性残差预测 (继承 ResWorld)

对 $\mathcal{V}_t$ 中的 BEV 位置，保留 ResWorld 的确定性残差预测：
$$\Delta B_{t+1}(x,y) = f_{\text{det}}(\{B_{t-2}^{\text{align}}(x,y), B_{t-1}^{\text{align}}(x,y), B_t(x,y)\}), \quad (x,y) \in \mathcal{V}_t$$

**实现**：$f_{\text{det}}$ 由 ResWorld 原有链路整体承担——确定性残差在 latent 空间完成（TokenLearner 压缩 → 相邻帧差分 `res_latent_query = bev_embed[:-1] - bev_embed[1:]` → res_latent_decoder → tokenfuser 融合回 BEV），输出 `pred_bev` 即"确定性未来 BEV"；可见区格子直接取 `pred_bev` 原值（掩码门控保证多假设分支不触碰可见区），无需逐位置单独建模。

#### 3.3 遮挡区：多假设随机残差转移 MHST (新增 ★)

这是本方案的核心创新。对 $\mathcal{O}_t$ 中的 BEV 位置，我们不预测单一残差，而是预测**多模态未来分布**。实现载体为 `OcclusionMHSTHead`（`resworld_head.py`，嫁接在 `pred_bev` 之后、`col_attn` 之前）。

**处理**:

1. **混合隐变量建模**: 对每个遮挡位置 $(x,y) \in \mathcal{O}_t$，引入离散隐变量 $c_t^{(x,y)} \in \{1, ..., K\}$ 表示"遮挡内容类型"（空路、静止车辆、运动车辆、行人、骑行者）。**实现**：K 个假设由 `self.experts`（K 个独立 3×3 卷积分支）承载，先验 π 为 softmax 连续权重，无显式离散采样。

2. **先验网络**: 基于边界可见位置的上下文，推断遮挡区的类别先验：
   $$P(c_t^{(x,y)} = k | \mathcal{N}(x,y)) = \text{Softmax}(g_{\text{prior}}(\{B_t(x',y')\}_{(x',y') \in \mathcal{N}(x,y)}))$$
   **实现**：`g_prior` 拼接 `[pred_bev | osz_mask]` → 1×1 Conv（`prior_conv1`）→ **多尺度膨胀邻域聚合**（`prior_d1/d2/d4`，3×3 dilation 1/2/4，感受野 3/5/9 格，对应 $\mathcal{N}(x,y)$）→ concat → 1×1 Conv（`prior_out`）→ K 维 softmax = π (B,K,H,W)。

3. **多假设残差生成**: 对每个假设 $k$，生成独立的残差预测：
   $$\Delta B_{t+1}^{(k)}(x,y) = f_{\text{occ}}^{(k)}(B_t(x,y), h_t, e_{c=k}), \quad (x,y) \in \mathcal{O}_t$$
   **实现**：共享骨干（1×1 + 3×3 Conv，`self.backbone`）→ K 个**独立 expert 卷积分支**（`self.experts`，每分支即一个条件化假设头，MoE 式）→ ΔB^(k) (B,K,C,H,W)。

4. **混合分布输出**: 下一时刻遮挡位置的分布为：
   $$p(B_{t+1}(x,y) | B_t, a_t) = \sum_{k=1}^K \pi_t^{(x,y),k} \cdot \mathcal{N}(B_t(x,y) + \Delta B_{t+1}^{(k)}(x,y), \Sigma_k^2)$$
   其中 $\pi_t^{(x,y),k} = P(c_t^{(x,y)} = k | \cdot)$，且协方差 $\Sigma_k$ 在遮挡区内部**强制大于阈值** $\Sigma_{\min}$（不确定性下限）。
   **实现**：σ 为每格标量 `softplus(s) + Σ_min`（`sigma_net`，Σ_min 默认 0.1 可配置；不预测 C×C 协方差防维度爆炸）；**消费路径**：混合分布**不回流到 BEV 特征**——`col_attn` 消费干净的 `pred_bev`，分布 (π, σ, ΔB) 经 aux 旁路构成显式风险场（Stage 4）后进入决策；混合分布本身仅训练侧损失用（§6.2/6.2c）。

**作用**: 将遮挡区从"信息缺失的空白"转化为"受几何约束的多假设想象空间"。

**为何如此设计**:
- 单纯增加 dropout 或数据增强无法让模型真正理解遮挡区可能有什么；显式多假设机制迫使模型在训练时覆盖多种遮挡内容。
- 混合隐变量 $c_t^{(x,y)}$ 提供了可解释性：我们可以检查模型认为"卡车后方最可能出现什么"。
- 不确定性下限 $\Sigma_{\min}$ 是安全关键设计：即使模型"猜测"遮挡区为空，也保留最低限度的方差，防止过度自信。


---

### Stage 4: 遮挡想象风险解码与时空风险场生成 (新增 ★)

**输入**: Stage 3 的 `outs['mhst']` = $\{\pi \in [0,1]^{B\times K\times H\times W},\ \sigma \in \mathbb{R}^{B\times 1\times H\times W},\ \Delta B^{(k)} \in \mathbb{R}^{B\times K\times C\times H\times W}\}$（单帧 K 假设补丁）与 Stage 2 的遮挡掩码 $M_t^{occ} \in \{0,1\}^{H \times W}$（osz_eye，4D 契约）。

**处理**:

1. **多假设风险分解（语义不可知代理）**: 对每个假设 $k$，在遮挡区内生成风险图：
   $$R^{(k)}(x,y) = w_\sigma \cdot \sigma(x,y) \cdot \big\| \Delta B^{(k)}(x,y) \big\|_2 \cdot \mathbb{1}[\,M_t^{occ}(x,y)\,]$$
   其中 $\|\Delta B^{(k)}\|_2$ 是该假设预测的**内容变化强度**（feature-space 碰撞威胁代理），$\sigma$ 为不确定性加权，$w_\sigma$ 为尺度超参（σ 经有界变换 $\sigma/(1+\sigma)$ 进入实现，见梯度契约）。**不依赖语义解码**——理由见"为何如此设计"1；语义解码器为**可选消融臂**（见实验 §5.2 消融 9）。
   
   **梯度契约**：风险场是分布的**无梯度测量**——$\pi, \sigma, \Delta B^{(k)}, s_{occ}$ 在进入风险场前全部 detach：分布只由 Stage 6 的分布塑造损失（似然/校准/多样性/占用/锚定）训练，规划器经 `grid_sample` 的采样坐标支路获得风险场空间梯度，学习**避开**风险而非**篡改**风险（否则"把 ΔB 学小 → risk→0"的作弊解会同时摧毁 MHST 表达力与 Stage 5）。σ 被校准到特征误差量级，风险场以有界变换 $\sigma/(1+\sigma)$ 消费其相对量级。

   **表达有界性**：假设残差与不确定度为有界量（正常区零影响、极端区饱和）——ΔB 元素级限幅（±10）、σ 上界（10）、风险场输出限幅（100）。由此保证：似然对 ΔB 的约束强度不随 σ 增大而失效；风险场与规划梯度有界；Minimax 语义保留（仍是 over-K max），仅限制假设的表达幅度——特征残差有物理量级，限幅合理。

2. **双输出保留多假设**:
   $$R_{\text{exp}}(x,y) = \sum_{k=1}^K \pi_k(x,y)\, R^{(k)}(x,y), \qquad R_{\text{worst}}(x,y) = \max_{k} R^{(k)}(x,y)$$
   期望风险供 CVaR/常规规划，**最坏风险**供 Minimax——最坏假设（如"可能冲出人"的低先验假设）显式保留到决策层。

3. **不确定性 + 占用驱动遮挡放大**:
   $$\tilde{R}(x,y) = R(x,y) \cdot \big(1 + \beta \cdot U(x,y) \cdot M_t^{occ}(x,y)\big) \cdot \big(1 + \gamma \cdot s_{\text{occ}}(x,y) \cdot M_t^{occ}(x,y)\big), \quad U(x,y) = \frac{H(\pi(x,y))}{\log K} \in [0,1]$$
   其中 $H(\pi) = -\sum_k \pi_k \log \pi_k$ 为假设分布熵，$\beta$ 为不确定性放大上限，$s_{\text{occ}}$ 为遮挡区占用概率（6.2b 的占用头输出，$[0,1]$），$\gamma$ 为占用放大权重。语义：**遮挡 + 高不确定（π 均匀）→ 强放大（可能鬼探头）；遮挡 + 低不确定（信念单点，如"墙后确定是墙"）→ 不放大**；**占用高（检测框 GT 指示有动态物）→ 额外放大**——把"变化强度"过滤成"有威胁的变化"，替代初稿对所有遮挡格一视同仁的常数 α。

4. **时空演化接口（先单帧，接口为时变）**: 沿自车运动/预测轨迹重采样得 $R_{t+\tau}$；遮挡暴露后按暴露观测更新 π（信念修正）→ 风险回落 $\Delta R = R_t - R_{t+\tau}$。时变重采样与 π 更新为后续接口（与 Stage 5 信息增益奖励联动，见 5.3）。

**输出**: 概率化风险场 $\tilde{R}_{\text{exp}},\ \tilde{R}_{\text{worst}} \in \mathbb{R}^{B \times H \times W}$（存 `outs['risk_field']`，供 Stage 5 消费）。

**作用**: 将多假设想象翻译为规划器可直接消费的**双风险度量**——紧凑（sufficient statistic，规划器无需处理 K 条假设）且保守（最坏假设不丢失）。

**为何如此设计**:

1. **鬼探头源于遮挡区变化过大**：鬼探头的本质是"内容未知"而非"内容分类错误"，因此风险直接由 Stage 3 的**不确定性代理**（σ·‖ΔB‖）构造——遮挡 + 内容变化强度大 + 不确定 = 风险，无需猜语义。
2. **常数 α 放大在关键区分上失效**：初稿 $(1+\alpha M)$ 对"卡车后可能是空的"（低分歧）与"可能冲出人"（高分歧）给出相同风险——而能否区分这两种格正是鬼探头安全的核心。π 熵驱动的放大（第 3 步）逐格区分，且可解释（放大系数 = 归一化不确定性）。
3. **最坏假设必须保留到决策**：若 Stage 4 只输出期望（$P_{occ}$ 聚合），Stage 5.2 的 Minimax 便无米下锅（消融 5.2-3 的"w/o Minimax"臂）。双输出使 Minimax（消费 $R_{\text{worst}}$）与 CVaR（消费 $R_{\text{exp}}$）各自有明确输入，公式链自洽。

---

### Stage 5: 遮挡感知风险加权规划 (改进 ★)

ResWorld 的 BEV Planner 直接优化轨迹的 L2 偏差。我们将其升级为**遮挡感知风险加权规划**——不做显式候选轨迹筛选（ResWorld 是单轨迹回归），而是让 Stage 4 的**保守风险场直接正则化预测轨迹**，端到端学习规避鬼探头高风险区。

**核心设计——GT 风险上界约束**：GT 轨迹本身是人类安全驾驶示范，其沿程风险是"可接受安全水平"的上界。因此风险损失全部采用**相对形式**：只惩罚"预测轨迹比 GT 更危险"的偏离——`risk`/`cvar` 两项带容差超参 $\mu$（容忍风险场噪声），`info` 项为纯末端风险差（起点风险对两轨迹相同，无需 $\mu$）。当预测轨迹与 GT 一致时全部为零（零梯度）——开环 L2 回归独占训练、不劣于基线；仅当模型偏离 GT 且更危险（真正进入鬼探头高风险区）时才被拉回 GT 的安全水平。"保守性"由此定义为**不劣于人类示范的安全性**，而非全局最小化风险。

#### 5.1 风险加权轨迹损失（Minimax 语义，相对 GT 上界）

预测轨迹（`ego_fut_preds`，增量 cumsum 还原绝对坐标）与 GT 轨迹（`ego_fut_gt`，同为增量、同样 cumsum 还原，两者同坐标系）沿程采样**最坏风险场** $R_{\text{worst}}$（Stage 4 第 2 步，over-K 的 max，即 minimax 语义）：

$$L_{\text{plan\_risk}} = \frac{1}{T}\sum_{t=1}^T \max\Big(0,\ R_{\text{worst}}(\tau_t) - R_{\text{worst}}(\tau_t^{\text{gt}}) - \mu\Big)$$

按导航指令（`cmd`）只监督对应轨迹；$R_{\text{worst}}$ 经 `grid_sample` 双线性采样（坐标映射与 `col_attn` 的 reference_points 同约定；按有效步 `fut_w` 加权）。**最小化 = 让预测轨迹不比 GT 更接近最坏假设下的高风险格**（如"可能冲出人"的区域）。

#### 5.2 CVaR 风险约束（沿程风险尾部，相对 GT 上界）

对指令轨迹沿程的 $R_{\text{exp}}$（期望风险）值集合求**空间尾部均值**（轨迹经过的最高风险段），并与 GT 轨迹的尾部均值比较：

$$\text{CVaR}_\beta = \text{mean}\big(\text{top}_{\lceil \beta T \rceil}\{R_{\text{exp}}(\tau_t)\}_{t=1}^T\big), \qquad L_{\text{plan\_cvar}} = \max\big(0,\ \text{CVaR}_\beta(\tau) - \text{CVaR}_\beta(\tau^{\text{gt}}) - \mu\big)$$

关注轨迹的极端高风险段（如某个时刻恰好穿过鬼探头格）**且比 GT 的极端段更危险**，而非平均风险。

#### 5.3 信息增益奖励与主动感知 (新增 ★，相对 GT 上界)

以"沿轨迹风险下降"度量主动感知。自车原点（ego 起点）的风险对预测与 GT 两条轨迹**相同**（同一风险场、同一采样点），故相对形式退化为**末端风险差**：

$$L_{\text{plan\_info}} = \max\big(0,\ R_{\text{exp}}(\tau_{\text{end}}) - R_{\text{exp}}(\tau_{\text{end}}^{\text{gt}})\big)$$

只惩罚"**末端风险高于 GT 末端**"——预测轨迹收敛到与 GT 同等的低风险末端即零惩罚，$\tau_{\text{end}}$ 取**最后有效步**（`fut_w` 定位）。单帧近似，T 步 rollout 为设计接口。

#### 5.4 总规划目标

$$L_{\text{plan}} = L_{\text{plan\_reg}} + \lambda_{\text{risk}} L_{\text{plan\_risk}} + \lambda_{\text{CVaR}} L_{\text{plan\_cvar}} + \lambda_{\text{info}} L_{\text{plan\_info}}$$

其中 $L_{\text{plan\_reg}}$ 为原有 L1 轨迹回归（继承），新增三项权重进 `resworld_config.py`（`use_risk_plan` 总开关，0=关即基线；`risk_plan_margin` 为容差 $\mu$，warmup/ramp 见配置注释）。

**实现范式**：Minimax 通过 $R_{\text{worst}}$（over-K max）实现，CVaR 通过沿程风险尾部实现，信息增益通过终点风险差实现——三者都是**可微的端到端规划正则**，且都以 GT 轨迹风险为上界，保留保守性与主动感知的语义而不与模仿目标对抗（候选轨迹 + argmin 选择的不可微范式与单轨迹回归不兼容，故不采用）。

**作用**: 
- $L_{\text{plan\_risk}}$ 让轨迹不进入比 GT 更接近最坏假设的高风险区（鬼探头规避）；
- $L_{\text{plan\_cvar}}$ 约束极端高风险段不超过 GT 尾部水平（防"平均风险低但某一刻危险"）；
- $L_{\text{plan\_info}}$ 鼓励主动探测（减速增距/横向偏移降低不确定性）。

**为何如此设计**: 绝对形式（直接最小化预测轨迹风险）与 L2 模仿目标结构性对抗——正常样本也会被无差别推开，伤害开环精度。相对上界把风险项变成**安全边际守卫**：正常样本（预测 ≈ GT）零梯度、L2 独占训练；只在模仿失败且不安全时介入。风险感知的职责是"在模仿边界内守卫安全"，而非替代模仿目标。

---

### Stage 6: 端到端训练 (完整训练目标)

#### 6.2 遮挡区自监督想象损失 (新增 ★)

利用时序数据中的**自然遮挡暴露**作为监督：t 时刻区域 $A$ 被遮挡，模型生成 K 假设 $\{\hat{B}_{t+1}^{A,(k)}\}_{k=1}^K$ 与权重 $\pi$；t+Δt 时刻自车移动使 $A$ 暴露，获得真值 $B_{t+\Delta t}^{A,\text{gt}}$。用**混合模型对数似然**监督遮挡区条件分布（同时拟合先验 π、假设内容 ΔB^(k)、不确定度 σ）：

$$L_{\text{occ\_halluc}} = -\log \sum_{k=1}^K \pi_t^{A,k} \cdot \mathcal{N}\big(B_{t+\Delta t}^{A,\text{gt}};\, \hat{B}_{t+1}^{A,(k)},\ \sigma_k^2\big)$$

- **监督对象**：遮挡区的条件分布 $p(B_{t+1}|B_t, M_{\text{occ}})$——π、ΔB^(k)、σ 一起做 EM 式拟合（σ 作分布带宽被隐式监督，与 6.4 显式校准方向一致）；
- **能力边界（分工设计）**：本损失仅对"内容慢变"的暴露格有效（静态内容）；**动态内容（鬼探头）由 6.2b 的检测框 GT 监督**；6.3 的 $L_{\text{div}}$ 维持假设分离；
- **监督范围与真值**：监督范围为**当前帧遮挡区**（`osz_eye`），真值为 **next 帧编码 BEV**（`encode_next_bev`，`next_img_inputs` 独立通道，对齐当前 ego 系）——**不加载 next 帧掩码**（暴露判定直接用当前帧遮挡区）；

#### 6.2b 遮挡区动态占用监督（检测框 GT 栅格化，新增 ★）

鬼探头场景内容突变，6.2 的暴露自监督对其无效（t 时刻输入无信息）。用**已有检测框 GT** 提供遮挡区的动态内容目标——零新数据、不引入多类别语义（保持语义不可知设计）：

$$L_{\text{occ\_gt}} = \sum_{(x,y) \in \mathcal{O}_t} \text{BCE}\big(s_{\text{occ}}(\hat{B}_{t+1}(x,y)),\, S_{\text{gt}}(x,y);\ \text{pos\_weight}\big)$$

其中 $S_{\text{gt}}$ 由 `gt_bboxes_3d`（已在训练 batch）动态栅格化到 BEV 网格（二值：是否有动态障碍物），$s_{\text{occ}}$ 为遮挡区小占用头（1×1 Conv from `pred_bev`），**`pos_weight`（默认 5.0）惩罚漏报占用格**——遮挡区绝大多数格为空（类别不平衡），不加权会退化为全 0 平凡解，学到"遮挡区动态内容"。分工：**6.2 监督静态、6.2b 监督动态、6.3 维持假设分离**。$s_{\text{occ}}$ 输出被 Stage 4 风险场用作**占用驱动放大**（第 3 步的 $\gamma \cdot s_{\text{occ}}$）。

#### 6.2c 风险场真实性锚定 (新增 ★)

让风险场成为"未来内容变化"的**无偏估计**：风险代理的**原始强度**（未 detach、未限幅，与 Stage 4 的风险场同构）与真实暴露变化 $e^2$ 对齐：

$$L_{\text{risk\_ground}} = \sum_{(x,y) \in \mathcal{O}_t} \Big( \frac{\sigma}{1+\sigma}\,\big\|\Delta B\big\| - \text{clip}(e^2, 0, E) \Big)^2$$

- **监督对象**：π/σ/ΔB 的联合产物 $\frac{\sigma}{1+\sigma}\|\Delta B\|$（$\|\Delta B\|$ 取 K 维均值）——与 Stage 4 风险代理共享构成，属"分布塑造损失"一族（与 $L_{\text{occ\_halluc}}$/校准同向），与风险场对规划的 detach 契约（§4 梯度契约）不冲突；
- **零额外数据**：$e^2$ 复用 6.4 的暴露误差（已 clamp 上界），遮挡区外不监督；
- **效果**：风险场只在真实变化剧烈的区域非零——GT 轨迹上变化天然小 → 风险天然低 → Stage 5 的 GT 上界约束在正常样本上自动休眠，只在"偏离 GT 且真实危险"处激活。

#### 6.3 多假设多样性损失 (新增)

防止 $K$ 个假设坍缩为相似模式（**实现为余弦相似度**，单位化后有界 [-1,1]、梯度稳定）：
$$L_{\text{div}} = \frac{2}{K(K-1)} \sum_{k_1 < k_2} \cos\big(\Delta B^{(k_1)},\, \Delta B^{(k_2)}\big) \cdot w_{\text{div}}$$
最小化假设残差间的余弦相似度（鼓励方向分离），替代 KL 伪分布（MHST 假设是特征残差非分布）。

**仅在遮挡区计算**——可见区 ΔB 不进入决策路径（多假设仅经风险场旁路），推开可见区假设无意义且无对向监督；**clamp ≥ 0**——负相似度（已分离）不奖励，损失有下界。

#### 6.4 不确定性校准损失 (新增)

确保遮挡区预测的不确定性与实际误差匹配（**训练契约**：Stage 3 的 σ 语义由本节兑现；实现中 Σ 为标量 σ = softplus(s) + Σ_min，与 `sigma_net` 对接）：
$$L_{\text{uncertainty}} = \sum_{(x,y) \in \mathcal{O}_t} \left| \sigma_{(x,y)} - \|e_{(x,y)}\|^2 \right|$$
其中 $e_{(x,y)} = \hat{B}_{t+1}(x,y) - B_{t+\Delta t}^{\text{gt}}(x,y)$ 为**遮挡暴露误差**——与 6.2 的 $L_{\text{occ\_halluc}}$ **共享同一份暴露数据**（零额外成本）。σ 同时在 6.2 混合似然中作分布带宽被隐式监督、在本节被显式校准为预期误差，两者方向一致（误差大的位置 σ 大）。

**校准目标固定化**：$e^2$ 作为 σ 的**固定校准目标**（detach）——避免 σ 与 e² 经 ΔB 支路互相拉近导致的校准失真；$\hat{B}_{t+1}$ 中确定性分支 $pb$ 同样 stop-gradient，暴露损失只训练分布参数（π/ΔB/σ），不扰动确定性世界模型路径。

#### 6.5 规划损失 (继承 + 改进)
$$L_{\text{plan}} = L_{\text{plan\_reg}} + \lambda_{\text{risk}} L_{\text{plan\_risk}} + \lambda_{\text{CVaR}} L_{\text{plan\_cvar}} + \lambda_{\text{info}} L_{\text{plan\_info}}$$
其中 $L_{\text{plan\_reg}}$ 为 ResWorld 原有 L1 轨迹回归，新增三项为 Stage 5 的风险加权规划（**相对 GT 上界形式**，见 §5.1-5.3，`use_risk_plan` 开关，0=关即基线）。

#### 6.6 信息增益损失 (新增，即 §5.3 的 $L_{\text{plan\_info}}$)
$$L_{\text{info}} = \max\big(0,\ R_{\text{exp}}(\tau_{\text{end}}) - R_{\text{exp}}(\tau_{\text{end}}^{\text{gt}})\big)$$
鼓励轨迹末端收敛到与 GT 同等的低风险水平（相对上界：末端风险 ≤ GT 末端即零惩罚；单帧近似，T 步 rollout 为设计接口）。

#### 6.7 总体损失
$$L_{\text{total}} = \omega_2 L_{\text{occ\_halluc}} + \omega_2' L_{\text{occ\_gt}} + \omega_{2c} L_{\text{risk\_ground}} + \omega_3 L_{\text{div}} + \omega_4 L_{\text{uncertainty}} + \omega_5 L_{\text{plan}}$$
其中 $L_{\text{plan}}$ 已含 $L_{\text{plan\_reg}}$（L1）+ $L_{\text{plan\_risk}}$ + $L_{\text{plan\_cvar}}$ + $L_{\text{plan\_info}}$（即 $L_{\text{info}}$，见 §6.5/6.6）——**$L_{\text{info}}$ 是 $L_{\text{plan}}$ 的组成部分，不重复计项**；$L_{\text{recon}}$ 因 ResWorld 无未来 BEV 真值而不在总损失中。

---

## 4. 与 Basework 的对比

| 模块 | ResWorld (arXiv 2026) | OARWM-Res (本方案) |
|---|---|---|
| 场景表示 | BEV Feature Map (200×200×256, x:0.15 m/cell, y:0.3 m/cell 各向异性) | + OSZ 高度感知 BEV 遮挡掩码（200×200，与 ResWorld 网格同格，直接注入） |
| 世界模型 | 确定性 latent-token 残差预测 | 可见区确定性 + 遮挡区多假设随机转移（分布 (π,σ,ΔB) 经 aux 旁路传导，不回流共享 BEV 特征） |
| 遮挡处理 | 隐式（数据驱动） | 显式几何约束 + 多假设想象 |
| 风险度量 | 无显式风险场（隐式碰撞检查） | 语义不可知双风险场 R_exp/R_worst，不确定性（π 熵）驱动遮挡放大 |
| 规划目标 | L2 模仿 + 碰撞避免 | 风险正则化的模仿学习：GT 风险上界约束（沿程 R_worst Minimax 语义与沿程尾部 CVaR 带容差 μ、末端风险差信息增益无 μ，均为相对形式，预测≈GT 时零梯度） |
| 训练监督 | BEV 重建损失 | + 遮挡暴露自监督 + 假设多样性 + 不确定性校准（σ 训练契约）+ 风险场真实性锚定（L_risk_ground，与暴露误差对齐） |
| 推理速度 | 轻量 (~10-15 FPS) | 相当（MHST 可并行） |
| 训练算力 | 8×RTX 3090 (24GB)，torch 1.9.1+cu111 | 训练延续 8×RTX 3090（torch 1.9.1 生态，与基线公平对比），全流程单机完成 |
| 核心能力 | 时序残差预测 | 遮挡区内容想象 + 不确定性驱动保守决策 + 主动减少不确定性 |

---

## 5. 预期实验与贡献

### 5.1 数据集
- **nuScenes**: 开环规划指标（L2 @ 1s/2s/3s, Collision Rate）；
- **Bench2Drive**: 闭环驾驶得分，重点看 Merging / Emergency Brake / Give Way / Unprotected Turn；
- **自建遮挡数据集**: 从 nuScenes / Bench2Drive 中筛选 parked car、truck、交叉口等遮挡比例 > 20% 的片段，构建遮挡场景子集；
- **遮挡子集评估**：用 `tools/analysis_tools/filter_occ_subset.py` 从 OSZ npz（occ_frac > 20%）筛选 token 子集并重算开环 L2（ADE/stp3-FDE）——风险感知的价值必须在风险存在的地方度量；
- **近遮挡减速行为统计**：用 `tools/analysis_tools/traj_behavior_stats.py` 统计掩码边界 5 m 内轨迹步的平均速度（预测 vs GT）——"接近遮挡物主动减速"的行为证据；配合可视化（STATUS.md §3.3）目视轨迹模式。

### 5.2 消融实验设计
1. w/o 显式遮挡掩码（验证几何先验必要性）；
2. w/o 多假设转移（单模态 vs 多模态，K=1 vs K=3,5,10）；
3. w/o Minimax/CVaR（期望风险 vs 最坏风险）；
4. w/o 信息增益奖励（被动 vs 主动感知）；
5. w/o 遮挡暴露自监督（验证自监督有效性）；
6. w/o 不确定性校准（验证方差约束必要性）；
7. OSZ 分层设计消融（`osz_eye` only vs `osz_ground` + `osz_eye`）；
8. **风险场消融（Stage 4）**：R_exp only vs R_worst only（验证最坏假设保留的必要性）；不确定性放大 on/off（π 熵驱动 vs 常数 α 放大）；
9. **风险代理消融（Stage 4 可选臂）**：语义不可知代理 vs 语义解码（有监督后）。

### 5.3 核心贡献
1. 首次将 OSZ 高度感知显式遮挡几何注入轻量残差世界模型，提出 OA-RWM；
2. 提出遮挡区多假设随机残差转移机制（MHST），在 BEV 特征空间实现遮挡区内容的概率化想象；
3. 提出基于 Minimax-CVaR-信息增益的三层鲁棒规划目标，使系统自发学习"探头确认"行为；
4. 提出遮挡暴露自监督与不确定性校准联合训练策略，无需额外标注即可监督遮挡区想象。
