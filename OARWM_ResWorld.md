# OARWM-Res: Occlusion-Aware Residual World Model with Multi-Hypothesis Latent Dynamics for End-to-End Autonomous Driving

> **基于 Basework**: [ResWorld](https://github.com/mengtan00/ResWorld) (arXiv 2026.02, 已开源)  
> **定位**: 端到端规划为核心，轻量残差世界模型为推理引擎，显式遮挡几何与多假设动态为安全约束  
> **目标期刊/会议**: CVPR / ICCV / ECCV / ICRA / IEEE T-ITS

---

## 1. 引言与 Motivation

### 1.1 现有瓶颈

ResWorld 提出了一种轻量的时序残差世界模型（Temporal Residual World Model），通过在 BEV 特征空间直接预测未来残差，避免了 3D Occupancy 或 Gaussian Splatting 的沉重解码开销，在 8×RTX 3090 (24GB) 上即可高效训练。然而，其在安全关键场景中存在以下根本缺陷：

1. **遮挡区特征传播黑盒化**：ResWorld 的残差预测对所有 BEV 空间位置一视同仁，缺乏显式机制区分"可见区"与"遮挡区"。遮挡区的未来特征完全由数据驱动的隐式插值推断，无法保证对"鬼探头"等突发风险的保守性。
2. **单模态确定性转移**：ResWorld 对每个 BEV 位置输出确定性残差（或单一高斯噪声），无法覆盖遮挡区"可能出现行人/车辆/空路"的多模态本质。当遮挡区内容在训练分布中罕见时，模型倾向于输出"平均化"的模糊特征，导致下游规划器低估风险。
3. **规划器缺乏主动感知能力**：ResWorld 的规划损失仅优化轨迹与真值的 L2 偏差及碰撞避免，未显式建模"通过自车动作减少遮挡不确定性"的主动感知能力，导致系统在遮挡场景中过度被动或过度激进。

### 1.2 我们的思路

本方案将 OAIAD 的**显式遮挡几何射线追踪**与**矢量化遮挡表示**深度嫁接至 ResWorld 的轻量残差世界模型框架中，提出 **OARWM-Res（Occlusion-Aware Residual World Model）**。核心思想是：

> **在轻量 BEV 残差世界模型中显式分离可见区与遮挡区的动力学，将遮挡区从"确定性残差的受害者"转化为"受几何约束的多假设概率空间"，并在此基础上构建主动感知的安全规划器。**

---

## 2. 完整方法架构

OARWM-Res 包含六大阶段，其中 Stage 1、3、5 继承并扩展 ResWorld，Stage 2、4、6 为新增或核心改进。

```
输入: 多视角图像序列 I_t, I_{t-1}, I_{t-2} (k=2 历史帧), 自车状态 e_t, 导航指令 n_t
│
├─► Stage 1: 图像编码与 GeoBEV 特征提取 (继承 ResWorld)
│     ├─► 多尺度图像特征 F_t
│     └─► BEV 特征 B_t ∈ R^(H×W×C)
│
├─► Stage 2: 显式遮挡几何注入与 BEV 掩码生成 (新增 ★)
│     ├─► OSZ 高度感知射线投射
│     ├─► BEV 遮挡掩码 M_t^occ ∈ {0,1}^(H×W)
│     └─► 遮挡感知 BEV 特征 B̃_t = B_t ⊙ (1-M_t^occ) + E_occ ⊙ M_t^occ
│
├─► Stage 3: 遮挡感知残差世界模型 (改进核心 ★)
│     ├─► 可见区: 确定性残差预测 (继承 ResWorld)
│     ├─► 遮挡区: 多假设随机残差转移 MHST (新增)
│     ├─► 边界上下文聚合: 遮挡-可见交互注意力 (新增)
│     └─► 未来 BEV 特征序列 {B̂_{t+1}, ..., B̂_{t+T}}
│
├─► Stage 4: 遮挡想象解码与时空风险场生成 (新增 ★)
│     ├─► BEV Decoder → 语义占用图 O_{t+1:T}^(k)
│     ├─► 多假设概率聚合
│     └─► 时空风险场 R_t ∈ R^(H×W)
│
├─► Stage 5: 鲁棒规划与主动感知 (改进 ★)
│     ├─► BEV Planner (继承 ResWorld)
│     ├─► Minimax 安全筛选 (新增)
│     ├─► CVaR 风险约束 (新增)
│     └─► 信息增益奖励 r_info (新增)
│     └─► 最优轨迹 τ*
│
└─► Stage 6: 端到端训练 (完整训练目标)
      ├─► 可见区 BEV 重建损失 L_recon
      ├─► 遮挡区自监督想象损失 L_occ_halluc
      ├─► 多假设多样性损失 L_div
      ├─► 不确定性校准损失 L_uncertainty
      ├─► 规划损失 L_plan
      └─► 信息增益损失 L_info
```

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
1. **Metric 深度估计**: 对每路相机图像预测 metric 深度图 $D_t^v$。实现采用 **MiDaS v2.1 Small**（`midas_v21_small_256.pt`，MiDaSNet-small / EfficientNet-Lite3，2021 年模型，torch 1.9.1 原生可跑、无需 timm），模型与权重均从本地加载（`OSZ/config.py::MIDAS_REPO_PATH` / `MIDAS_MODEL_PATH`，torch.hub `source='local'`，入口 `MiDaS_small` + 官方 `small_transform`，**运行期不联网**），输出逆深度（大值=近），以稀疏 LiDAR 投影深度为真值做尺度对齐（`OSZ/modules/depth_estimator.py::align_to_lidar`）：
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
   输出 BEV 高度图 $H_t \in \mathbb{R}^{H \times W}$。网格由 `OSZ/config.py` 统一定义并对齐 ResWorld `grid_config`：**200×200 各向异性网格**（$x \in [-15,15]\,m @ 0.15\,m$、$y \in [-30,30]\,m @ 0.3\,m$；`indexing='ij'`，axis-0=ego-x 前进方向，axis-1=ego-y 左侧）。融合策略
4. **高度感知射线投射**: 以自车位置为中心做 360° 射线投射（`OSZ/modules/ray_casting.py::cast_osz_height_aware`，向量化实现，`substep=0.25` cell 步进、`n_angles ≥ 720`），得到两类遮挡阴影：
   - `osz_ground`: 任意高度 **> 0.05 m** 的占据 cell 均阻挡光线（二值行为，非地面阈值）的**地面层阴影**；
   - `osz_eye`: 仅高度 **> `OBSERVER_HEIGHT_M`（默认 1.2 m）** 的 cell 阻挡光线的**严格盲区**。
   射线命中首个遮挡物后，其后方 cell 标记为阴影（遮挡物自身 cell 不标记）；自车周围 `EGO_CLEARANCE_RADIUS_M=1.0 m` 半径内高度清零，避免自遮挡。`osz_ground & ~osz_eye` 为"半透明盲区"（semi），风险低于完全盲区。
5. **可行驶区域过滤（可选）**: 将 OSZ 限制在 HD-map 可行驶区域内（`OSZ/modules/drivable_filter.py`：实现为 `osz ∩ drivable_mask`）。`drivable_mask` 由 HD-map 图层 `drivable_area + carpark_area` 光栅化（PIL 多边形绘制，多边形顶点绕 ego-yaw 旋转至 ego 系）扣除 `walkway / ped_crossing` 后，按 `DEFAULT_DRIVABLE_DILATION_M=1.5 m` 膨胀；HD-map 缺失时回退全 True。
6. **BEV 遮挡掩码**: 取 `osz_eye` 作为严格遮挡掩码：
   $$M_t^{occ}(x,y) = \text{osz\_eye}_t(x,y) \in \{0, 1\}^{H \times W}$$
7. **遮挡感知 BEV 特征（OARWM 模型端实现）**: 特征注入不在 OSZ 模块内完成——OSZ 流水线输出 `bev_height`、`osz_ground`、`osz_eye`、`semi` 与 `drivable_mask`。OARWM 以掩码通道拼接为输入构建增强 BEV 特征：
   $$\tilde{B}_t = B_t \odot (1 - M_t^{occ}) + E_{occ} \odot M_t^{occ}$$
   其中 $E_{occ} \in \mathbb{R}^{C}$ 为可学习的遮挡类型嵌入（由掩码通道 `(osz_eye, osz_ground, semi)` 经 1×1 Conv 生成，区分严格盲区 / 半透明盲区 / 动态变化遮挡），通过广播与 BEV 特征逐元素相乘。

**输出**: OSZ 模块输出 `bev_height`、`osz_ground`、`osz_eye`、`semi`、`drivable_mask`（200×200，与 ResWorld 网格同格）；OARWM 模型端输出遮挡感知 BEV 特征

**网格对齐**: OSZ 的 BEV 网格已直接对齐 ResWorld `grid_config`——$x \in [-15,15]\,m @ 0.15\,m$、$y \in [-30,30]\,m @ 0.3\,m$（200×200 **各向异性**），定义于 `OSZ/config.py`（单一来源，需与 `resworld_config.py::grid_config` 保持同步）。OSZ 掩码/高度图与 ResWorld BEV 特征同格，**无需任何重采样**，直接注入 Stage 3 的 BEV 特征。

- **OSZ 深度估计**：**MiDaS v2.1 Small**

- **OSZ 掩码接入**：OSZ 流水线输出 `bev_height / osz_ground / osz_eye / drivable_mask`（网格与 ResWorld `grid_config` 对齐，200×200 各向异性），每帧导出 npz 供 ResWorld 数据管线加载（见 STATUS.md）。

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

**与 Basework 实现的对接（实现载体决策）**: ResWorld 的残差并非逐 BEV 位置预测——`resworld_head.py` 先将 BEV 特征经 TokenLearner 压缩为 latent tokens，在 **latent 空间**做相邻帧差分（`res_latent_query = bev_embed[:-1] - bev_embed[1:]`），经 latent decoder 后由 tokenfuser 融合回 BEV（`pred_bev = tokenfuser(...) + bev_navi_embed`，残差连接）。因此 OA-RWM 的 MHST 存在两种嫁接方式：**主方案**——在 `pred_bev` 输出后按掩码加门控多假设残差头（已实现，见 3.5）；**备选方案**——在 latent token 分支做遮挡相关 token 的 K 假设转移（概念设计，暂不实现，见 3.6）。可见/遮挡位置路由均以 Stage 2 的 200×200 掩码为准；3.1-3.3 的逐位置公式为概念定义，实现时按主方案的门控残差头对接。

#### 3.1 BEV 位置分类与路由

根据遮挡掩码 $M_t^{occ}(x,y)$，将 BEV 空间位置分为两类：
- **Visible Positions** ($\mathcal{V}_t$): $M_t^{occ}(x,y) = 0$，可直接观测；
- **Occluded Positions** ($\mathcal{O}_t$): $M_t^{occ}(x,y) = 1$，需要想象。

#### 3.2 可见区：确定性残差预测 (继承 ResWorld)

对 $\mathcal{V}_t$ 中的 BEV 位置，保留 ResWorld 的确定性残差预测：
$$\Delta B_{t+1}(x,y) = f_{\text{det}}(\{B_{t-2}^{\text{align}}(x,y), B_{t-1}^{\text{align}}(x,y), B_t(x,y)\}), \quad (x,y) \in \mathcal{V}_t$$
其中 $f_{\text{det}}$ 为轻量 CNN 或 Transformer 编码器，预测下一时刻 BEV 特征的确定性偏移。

**实现说明（对齐主方案）**: ResWorld 的确定性残差在 latent 空间完成（TokenLearner → 帧间差分 → res_latent_decoder → tokenfuser），其输出 `pred_bev` 即"确定性未来 BEV"。主方案中可见区格子**直接取 `pred_bev` 原值**（掩码门控保证多假设分支不触碰可见区），故 $f_{\text{det}}$ 在实现层面由 ResWorld 原有链路整体承担，无需逐位置单独建模——本节的逐位置公式仅作概念定义。

#### 3.3 遮挡区：多假设随机残差转移 MHST (新增 ★)

这是本方案的核心创新。对 $\mathcal{O}_t$ 中的 BEV 位置，我们不预测单一残差，而是预测**多模态未来分布**。

> 本节为**概念定义**（逐位置公式）；实现载体为主方案的门控多假设残差头（3.5），公式与实现**逐位置一一对应**。备选方案（3.6）仅在论文需要"与基线深度融合变体"时考虑。

**处理**:
1. **混合隐变量建模**: 对每个遮挡位置 $(x,y) \in \mathcal{O}_t$，引入离散隐变量 $c_t^{(x,y)} \in \{1, ..., K\}$ 表示"遮挡内容类型"（如：空路、静止车辆、运动车辆、行人、骑行者）。

2. **先验网络**: 基于边界可见位置的上下文，推断遮挡区的类别先验：
   $$P(c_t^{(x,y)} = k | \mathcal{N}(x,y)) = \text{Softmax}(g_{\text{prior}}(\{B_t(x',y')\}_{(x',y') \in \mathcal{N}(x,y)}))$$
   其中 $\mathcal{N}(x,y)$ 为 $(x,y)$ 的遮挡边界邻域（利用 2D 空间注意力提取）。先验网络为轻量 1×1 Conv + MLP。

3. **多假设残差生成**: 对每个假设 $k$，生成独立的残差预测：
   $$\Delta B_{t+1}^{(k)}(x,y) = f_{\text{occ}}^{(k)}(B_t(x,y), h_t, e_{c=k}), \quad (x,y) \in \mathcal{O}_t$$
   其中 $e_{c=k} \in \mathbb{R}^{C}$ 为假设类别嵌入，$f_{\text{occ}}^{(k)}$ 为共享参数但条件不同的残差头。实现上可采用 **Mixture-of-Experts (MoE)**：所有假设共享一个骨干网络，但通过不同的 expert 分支生成条件化残差。

4. **混合分布输出**: 下一时刻遮挡位置的分布为：
   $$p(B_{t+1}(x,y) | B_t, a_t) = \sum_{k=1}^K \pi_t^{(x,y),k} \cdot \mathcal{N}(B_t(x,y) + \Delta B_{t+1}^{(k)}(x,y), \Sigma_k^2)$$
   其中 $\pi_t^{(x,y),k} = P(c_t^{(x,y)} = k | \cdot)$，且协方差 $\Sigma_k$ 在遮挡区内部**强制大于阈值** $\Sigma_{\min}$（不确定性下限）。

5. **时序递归**: 对 $T$ 步未来进行 rollout，得到 $K$ 条平行的 BEV 特征序列：
   $$\{\{\hat{B}_{t+1:T}^{(1)}\}, \{\hat{B}_{t+1:T}^{(2)}\}, ..., \{\hat{B}_{t+1:T}^{(K)}\}\}$$

**输出**: $K$ 条未来 BEV 特征序列，每条对应遮挡区的一种可能内容假设。

**作用**: 将遮挡区从"信息缺失的空白"转化为"受几何约束的多假设想象空间"。

**为何如此设计**:
- 单纯增加 dropout 或数据增强无法让模型真正理解遮挡区可能有什么；显式多假设机制迫使模型在训练时覆盖多种遮挡内容。
- 混合隐变量 $c_t^{(x,y)}$ 提供了可解释性：我们可以检查模型认为"卡车后方最可能出现什么"。
- 不确定性下限 $\Sigma_{\min}$ 是安全关键设计：即使模型"猜测"遮挡区为空，也保留最低限度的方差，防止过度自信。

#### 3.4 遮挡-可见边界交互 (新增)

当自车执行动作 $a_t$（如减速、变道）后，遮挡掩码 $M_{t+1}^{occ}$ 会动态变化。原本被遮挡的位置可能变为可见。

**处理**:
- 对从 $\mathcal{O}_t$ 转移到 $\mathcal{V}_{t+1}$ 的位置，用实际观测 $B_{t+1}^{\text{obs}}(x,y)$ 修正假设权重：
  $$\pi_{t+1}^{(x,y),k} \propto \pi_t^{(x,y),k} \cdot \exp\left(-\|B_{t+1}^{\text{obs}}(x,y) - (B_t(x,y) + \Delta B_{t+1}^{(k)}(x,y))\|^2\right)$$
- 若所有假设与观测偏差均较大，触发**异常检测**：可能为训练分布外的危险事件（如违规闯红灯车辆）。

**作用**: 实现"预测-验证-修正"的认知闭环，模拟人类探头确认后更新信念的过程。

---

#### 3.5 主实现方案（★ 已实现 2026-08）：`pred_bev` 后门控多假设残差头 MHST-Head

**决策**：MHST 落点选在 ResWorld 残差链路的**输出端**（`pred_bev` 之后），而非改造其内部 latent 机制。理由：

1. 与 3.1–3.3 的逐位置公式**一一对应**，论文表述与实现一致；
2. 掩码直接作门控，可见区**严格保持基线行为**，`K=1`（或关掉本头）即退化为纯基线，消融最干净（对应消融 5.2-2）；
3. 不改动 ResWorld 已调通的 latent 残差路径（TokenLearner → 差分 → res_latent_decoder → tokenfuser），训练稳定性风险最低；
4. 新增头整体可开关，不影响 Stage 1/2 链路，可独立加载基线 checkpoint 继续训练。

**插入点**（`resworld_head.py` forward，行号以当前代码为准）：
`pred_bev = self.tokenfuser(res_latent_query.permute(1, 0, 2), bev_navi_embed) + bev_navi_embed`（约 `:315`）之后、`self.col_attn(...)`（约 `:317`）之前。理由：`pred_bev` 是"当前时刻对未来 BEV 的确定性预测"，是 MHST 逐位置加补丁的正确输入；`col_attn` 的 reference points 基于初始轨迹，其采样值应使用多假设融合后的 `pred_bev`，因此 MHST 必须插在 `col_attn` 之前。

**模块结构**（新增 `OcclusionMHSTHead`，命名与风格对齐现有 `OcclusionAwareFusion`，注册进 `HEADS`）：

1. **掩码输入**：复用 Stage 2 数据管线的 `osz_mask`（三通道 `(osz_eye, osz_ground, semi)`，200×200，`_load_osz_mask` 已 lru_cache）；若 BEV 特征分辨率与 200×200 不同（当前相同），`F.interpolate(mode='nearest')` 对齐。
2. **先验网络 `g_prior`**：拼接 `(osz_eye, osz_ground, semi)` 与 `pred_bev` → 1×1 Conv 投影 → **遮挡边界上下文聚合**（对掩码做膨胀邻域 max-pooling 提取边界带，或深度可分离卷积实现局部注意力，对应 3.3-2 式中的 $\mathcal{N}(x,y)$）→ MLP → `π ∈ [0,1]^{B×H×W×K}`（对 K 维 softmax）。
3. **K 假设残差生成**（对应 3.3-3 式）：共享骨干（1×1 Conv + 3×3 Conv，输入 `pred_bev`）→ K 个 expert 分支（每分支 = 类别嵌入 `e_{c=k}` 条件化 + 轻量卷积）→ `ΔB^(k) ∈ R^{B×C×H×W}`。
4. **不确定性输出**（对应 3.3-4 式与 6.4 校准接口）：每格预测标量方差 `Σ ∈ R^{B×H×W}`，`Σ = softplus(s) + Σ_min`（`Σ_min` 默认 0.1，可配置）——保证遮挡区不确定性下限，防止过度自信；不预测 C×C 协方差（维度爆炸，论文中 $\Sigma_k^2$ 按标量实现）。
5. **门控合成**（核心公式）：
   - 可见区（`osz_eye=0`）：`B̂_{t+1}(x,y) = pred_bev(x,y)`（原值，零改动）；
   - 遮挡区（`osz_eye=1`）：`B̂_{t+1}(x,y) = Σ_{k=1}^K π_k(x,y)·(pred_bev(x,y) + ΔB^(k)(x,y))`；
   - 混合分布 `p(B_{t+1}|·) = Σ_k π_k · N(B̂^(k), Σ_k²)` 仅用于训练侧损失（Stage 6），推理侧只消费融合后的单张 `pred_bev`（保持与基线相同的下游接口，规划器/评估无需改动）。

**OSZ 掩码的帧数决策（仅当前帧 t，不对三帧重复注入）**：

- 掩码语义是"**当前观测**的遮挡几何"，历史帧是已发生的事实，其遮挡状态已由 BEV 时序特征（三帧通道融合 + latent 差分）自然携带；
- **坐标系**：掩码在**各帧自身 ego 系**导出（OSZ 射线从该帧 ego 位置发出），而 head 中的三帧 BEV 特征已统一到当前 ego 系（视图变换阶段对齐，配置 `align_after_view_transfromation=False` 走 RCSample 采样网格内对齐）；历史帧掩码若要匹配须先按自车运动 warp，**二值掩码 warp 引入孔洞/重采样误差**，收益低风险高；
- Stage 3 预测的是"从当前帧出发的未来分布"，遮挡区由当前帧掩码定义即可；
- 数据管线现状（`nuscenes_resworld_dataset.py::_load_osz_mask` 只按当前样本 token 加载）无需改动，Stage 2 注入点（`bev_fusion_conv` 之后的三帧融合特征）与 Stage 3 门控共用同一份当前帧掩码。
- 备注：若未来消融需要"历史遮挡提示"，对历史帧掩码做 ego-motion warp 后以额外通道输入，作为独立实验，不在主方案内。

**训练损失接口（Stage 6 预留，本阶段只搭骨架）**：

- 已实现：`OcclusionMHSTHead` 输出 `pred_bev`（融合后）、`π`、`Σ`、`ΔB^(k)`（存 `outs['mhst']`）；`head.loss` 加极小权重占位正则 `loss_mhst_sigma = Σ.mean() × 1e-4`——作用仅是让 `sigma_net` 参与梯度（否则 `find_unused_parameters=False` 下 DDP 报未使用参数），Stage 6 由 `L_uncertainty` 替换；
- `L_occ_halluc`（遮挡暴露自监督）：需加载相邻帧掩码 `osz_mask_{t+Δt}`（按相邻 token 复用 `_load_osz_mask`）与相邻帧真值 BEV；待 Stage 6 接入；
- `L_div`（多样性，防坍缩）：`-Σ_{k1≠k2} KL(p^(k1)‖p^(k2))` 或等价假设残差两两距离项，待 Stage 6 接入；
- `L_uncertainty`（校准）：`|Σ - ‖e‖²|`，`e` 为遮挡区预测误差，待 Stage 6 接入；
- 以上损失均以掩码选择遮挡区位置计算，权重系数进 `resworld_config.py`。

**配置开关**（对齐现有 `use_osz_midas` / `use_osz_rcsample` 风格，一个 Stage 一组开关）：

```python
use_oarwm = True          # Stage 3 总开关；False = 不创建 MHST-Head，pred_bev 直通（严格基线，零开销）
mhst_k = 3                # 假设数 K（消融 5.2-2：1/3/5/10；K=1 为单假设消融，仍有残差补丁）
mhst_sigma_min = 0.1      # 遮挡区不确定性下限 Σ_min
mhst_sigma_reg_weight = 1e-4  # sigma 占位正则权重（Stage 6 换 L_uncertainty）
# use_oarwm=True 依赖掩码源：assert use_oarwm -> use_osz_midas or use_osz_rcsample
```

**消融等价性**：`use_oarwm=False` 时 MHST-Head 不创建，`pred_bev` 直通，与基线**逐位一致**、零额外参数——这是相对方案 A 的显著优势（A 需额外确认路由机制关闭后是否严格复原）。注：`K=1` 是**单假设消融**（遮挡区仍有确定性残差补丁 `pred_bev + ΔB`），并非纯基线。

**已知风险与应对**：

- 遮挡区监督稀疏（无逐位置真值未来 BEV 遮挡内容）→ 依赖 6.2 遮挡暴露自监督 + 先验网络正则，冷启动期可先用重建损失近似；
- K 个假设分支增加显存/耗时 → 共享骨干 + 轻量 expert 分支，`K=3` 参数量增量预计 <5%；评估期只做前向，训练期才反向；
- 遮挡/可见交界处特征突变 → 门控前对 mask 做小半径平滑（如 3×3 均值）缓解。

#### 3.6 备选实现方案（仅概念设计，暂不实现）：latent token 分支嫁接

**概念**：不新增输出头，而是改造 ResWorld 自身的 latent 残差机制，让残差 token 的生成与解码感知遮挡：

1. **掩码引导的 token 选择**：将 `osz_mask` 下采样到 token 粒度，调制 `res_tokenlearner` 的选择权重 `res_selected`（如 `selected = softmax(selected + mask_prior)`），使遮挡位置的 BEV 信息集中到部分 token；
2. **K 假设 latent 转移**：对"遮挡相关 token"复制 K 份，分别过 `res_latent_decoder`，输出 K 组残差 latent；
3. **K 路融合**：K 组残差 latent 分别经 `tokenfuser` 得到 K 张 `pred_bev^(k)`，再按掩码加权合成最终 `pred_bev`（可见区取原值/平均）。

**优点**：完全复用 ResWorld 现有 decoder 与 tokenfuser，参数量增量最小；与基线深度融合，叙事上更"端到端"。

**缺点**：token 是全局的（16 个 token 混合可见/遮挡信息），"逐位置 K 假设"的论文语义被摊平，需额外路由机制保证遮挡信息确实进入多假设分支；K 路 tokenfuser 与权重合成的行为不直观，消融（K=1）不保证严格复原基线；调试与可解释性差于方案 B。

**决策记录**：主方案选 B（逐位置对应、门控清晰、消融干净、风险最低）；A 仅作为论文中"与基线深度融合变体"的可选对比，暂不实现。

---

### Stage 4: 遮挡想象解码与时空风险场生成 (新增 ★)

**输入**: $K$ 条未来 BEV 特征序列 $\{\hat{B}_{t+1:T}^{(k)}\}_{k=1}^K$。

**处理**:
1. **BEV Decoder**: 使用 ResWorld 的 BEV Decoder（轻量反卷积/上采样网络），将每条 BEV 特征序列解码为语义占用图：
   $$O_{t+1:T}^{(k)} = \text{BEVDecoder}(\hat{B}_{t+1:T}^{(k)}) \in [0,1]^{H \times W \times C_{sem}}$$
   ResWorld 的 decoder 输出语义分割图（如车辆、行人、道路、建筑等类别），而非沉重的 3D occupancy voxels。

2. **遮挡区概率聚合**: 对遮挡区内的每个 BEV 位置 $(x,y)$，聚合 $K$ 个假设的语义分布：
   $$P_{occ}(x,y,t) = \sum_{k=1}^K \pi^{(k)} \cdot \text{Softmax}(O_t^{(k)}(x,y))$$

3. **时空风险场生成**: 计算碰撞风险热力图：
   $$R_t(x,y) = \sum_{c \in \{\text{vehicle, pedestrian, cyclist}\}} P_{occ}(x,y,t,c) \cdot \text{RiskWeight}(c)$$
   对遮挡区内部的风险值进行保守放大：
   $$\tilde{R}_t(x,y) = R_t(x,y) \cdot (1 + \alpha \cdot M_t^{occ}(x,y))$$
   其中 $\alpha > 0$ 为遮挡风险放大系数（如 $\alpha = 2.0$）。

**输出**: 概率化语义图 $P_{occ}$，时空风险场 $\tilde{R}_t \in \mathbb{R}^{H \times W}$。

**作用**: 将多假设想象翻译为规划器可直接使用的风险度量。

**为何如此设计**: 规划器无法直接消费 $K$ 条完整 BEV 序列（计算爆炸）；风险场是紧凑的 sufficient statistic。

---

### Stage 5: 鲁棒规划与主动感知 (改进 ★)

ResWorld 的 BEV Planner 直接优化轨迹的 L2 偏差与碰撞避免。我们将其升级为**遮挡感知鲁棒规划器**。

#### 5.1 候选轨迹生成 (继承)

生成 $M$ 条候选轨迹 $\{\tau_1, ..., \tau_M\}$，每条为 $T$ 步的 2D/3D 轨迹点序列。

#### 5.2 Minimax 安全筛选 (新增)

对每条候选轨迹，计算在 $K$ 条假设未来中的**最坏情况代价**：
$$J_{\text{minimax}}(\tau) = \max_{k \in \{1..K\}} \sum_{t=1}^T \left[ \lambda_1 \cdot \text{Collision}_k(\tau_t) + \lambda_2 \cdot \tilde{R}_t(\tau_t) + \lambda_3 \cdot \text{Comfort}(\tau_t) \right]$$
选择最小化最坏代价的轨迹：
$$\tau^*_{\text{safe}} = \arg\min_{\tau} J_{\text{minimax}}(\tau)$$

#### 5.3 CVaR 风险约束 (新增)

在 Minimax 基础上，进一步引入 **CVaR（条件风险价值）** 约束，关注尾部风险：
$$\text{CVaR}_\beta(\tau) = \mathbb{E}\left[ J(\tau) \mid J(\tau) \geq \text{VaR}_\beta \right]$$
其中 $\beta$ 为置信水平（如 95%）。最终规划目标为：
$$\tau^* = \arg\min_{\tau} J_{\text{minimax}}(\tau) \quad \text{s.t.} \quad \text{CVaR}_\beta(\tau) \leq \delta$$

#### 5.4 信息增益奖励与主动感知 (新增 ★)

在轨迹优化中增加**信息增益奖励**，鼓励自车通过动作减少遮挡不确定性：
$$r_{\text{info}}(\tau) = \sum_{t=1}^T \sum_{(x,y) \in M_t^{occ}} \mathbb{1}\left[(x,y) \notin M_{t+1}^{occ}(\tau_t)\right] \cdot \Delta H(x,y)$$
其中 $\Delta H(x,y)$ 衡量该位置从遮挡变为可见后，其语义分布熵的下降量：
$$\Delta H(x,y) = H(P_{occ}(x,y,t)) - H(P_{occ}(x,y,t+1))$$

总规划目标：
$$\tau^* = \arg\min_{\tau} \left[ J_{\text{minimax}}(\tau) + \lambda_{\text{CVaR}} \cdot \text{CVaR}_\beta(\tau) - \gamma \cdot r_{\text{info}}(\tau) + \lambda_4 \cdot J_{\text{efficiency}}(\tau) \right]$$

**作用**: 
- Minimax 保证在最坏假设下仍安全；
- CVaR 约束防止极端尾部风险；
- 信息增益奖励使系统自发学会"探头"行为（如减速增距、轻微横向偏移）。

**为何如此设计**: 纯保守规划会导致过度被动（如 VAD 在交互场景中完全停车）；信息增益项使系统在安全和效率之间取得平衡，且无需人工设计规则。

---

### Stage 6: 端到端训练 (完整训练目标)

#### 6.1 可见区 BEV 重建损失 (继承 ResWorld)
$$L_{\text{recon}} = \sum_{\tau=t+1}^{t+T} \sum_{(x,y) \in \mathcal{V}_\tau} \| \hat{B}_\tau(x,y) - B_\tau^{\text{gt}}(x,y) \|_2^2$$

#### 6.2 遮挡区自监督想象损失 (新增 ★)

利用时序数据中的**自然遮挡暴露**作为免费监督：
- 在时刻 $t$，区域 $A$ 被遮挡，模型生成想象 $\{\hat{B}_{t+1}^{A,(k)}\}_{k=1}^K$；
- 在时刻 $t+\Delta t$，自车移动使 $A$ 变为可见，获得真值 $B_{t+\Delta t}^{A,\text{gt}}$；
- 假设遮挡区内内容在 $\Delta t$ 内变化不大，用 $B_{t+\Delta t}^{A,\text{gt}}$ 监督 $t$ 时刻的想象：
  $$L_{\text{occ\_halluc}} = -\sum_{k=1}^K \pi_t^{A,k} \log P(B_{t+\Delta t}^{A,\text{gt}} | \hat{B}_{t+1}^{A,(k)})$$
  实现上采用高斯对数似然：
  $$L_{\text{occ\_halluc}} = \sum_{k=1}^K \pi_t^{A,k} \cdot \left\| \hat{B}_{t+1}^{A,(k)} - B_{t+\Delta t}^{A,\text{gt}} \right\|_2^2$$

#### 6.3 多假设多样性损失 (新增)

防止 $K$ 个假设坍缩为相似模式：
$$L_{\text{div}} = -\sum_{k_1 \neq k_2} \text{KL}(p^{(k_1)} \| p^{(k_2)})$$
鼓励不同假设之间的分布差异最大化。

#### 6.4 不确定性校准损失 (新增)

确保遮挡区预测的不确定性与实际误差匹配：
$$L_{\text{uncertainty}} = \sum_{(x,y) \in \mathcal{O}_t} \left| \text{tr}(\Sigma_{(x,y)}) - \|e_{(x,y)}\|^2 \right|$$
其中 $e_{(x,y)} = \hat{B}_{t+1}(x,y) - B_{t+1}^{\text{gt}}(x,y)$ 为实际误差。

#### 6.5 规划损失 (继承 + 改进)
$$L_{\text{plan}} = \|\tau^* - \tau^{\text{gt}}\|_2 + \lambda_{\text{coll}} \cdot \text{CollisionRate}(\tau^*) + \lambda_{\text{info}} \cdot r_{\text{info}}$$

#### 6.6 信息增益损失 (新增)
$$L_{\text{info}} = -r_{\text{info}}(\tau^*)$$
将信息增益奖励转化为可优化的损失项。

#### 6.7 总体损失
$$L_{\text{total}} = \omega_1 L_{\text{recon}} + \omega_2 L_{\text{occ\_halluc}} + \omega_3 L_{\text{div}} + \omega_4 L_{\text{uncertainty}} + \omega_5 L_{\text{plan}} + \omega_6 L_{\text{info}}$$

---

## 4. 与 Basework 的对比

| 模块 | ResWorld (arXiv 2026) | OARWM-Res (本方案) |
|---|---|---|
| 场景表示 | BEV Feature Map (200×200×256, x:0.15 m/cell, y:0.3 m/cell 各向异性) | + OSZ 高度感知 BEV 遮挡掩码（200×200，与 ResWorld 网格同格，直接注入） |
| 世界模型 | 确定性 latent-token 残差预测 | 可见区确定性 + 遮挡区多假设随机转移 |
| 遮挡处理 | 隐式（数据驱动） | 显式几何约束 + 多假设想象 |
| 规划目标 | L2 模仿 + 碰撞避免 | + Minimax + CVaR + 信息增益主动感知 |
| 训练监督 | BEV 重建损失 | + 遮挡区自监督暴露损失 + 假设多样性 + 不确定性校准 |
| 推理速度 | 轻量 (~10-15 FPS) | 相当（MHST 可并行） |
| 训练算力 | 8×RTX 3090 (24GB)，torch 1.9.1+cu111 | 训练延续 8×RTX 3090（torch 1.9.1 生态，与基线公平对比），全流程单机完成 |
| 核心能力 | 时序残差预测 | 遮挡区内容想象 + 主动减少不确定性 |

---

## 5. 预期实验与贡献

### 5.1 数据集
- **nuScenes**: 开环规划指标（L2 @ 1s/2s/3s, Collision Rate）；
- **Bench2Drive**: 闭环驾驶得分，重点看 Merging / Emergency Brake / Give Way / Unprotected Turn；
- **自建遮挡数据集**: 从 nuScenes / Bench2Drive 中筛选 parked car、truck、交叉口等遮挡比例 > 20% 的片段，构建遮挡场景子集。

### 5.2 消融实验设计
1. w/o 显式遮挡掩码（验证几何先验必要性）；
2. w/o 多假设转移（单模态 vs 多模态，K=1 vs K=3,5,10）；
3. w/o Minimax/CVaR（期望风险 vs 最坏风险）；
4. w/o 信息增益奖励（被动 vs 主动感知）；
5. w/o 遮挡暴露自监督（验证自监督有效性）；
6. w/o 不确定性校准（验证方差约束必要性）；
7. OSZ 分层设计消融（`osz_eye` only vs `osz_ground` + `osz_eye`）。

### 5.3 核心贡献
1. 首次将 OSZ 高度感知显式遮挡几何注入轻量残差世界模型，提出 OA-RWM；
2. 提出遮挡区多假设随机残差转移机制（MHST），在 BEV 特征空间实现遮挡区内容的概率化想象；
3. 提出基于 Minimax-CVaR-信息增益的三层鲁棒规划目标，使系统自发学习"探头确认"行为；
4. 提出遮挡暴露自监督与不确定性校准联合训练策略，无需额外标注即可监督遮挡区想象。