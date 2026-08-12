# OARWM-Res 项目状态（Stage 2 接入完成）

> 最后更新：2026-07（本次改造）
> 设计文档：`OARWM_ResWorld.md` · 安装/数据准备：`INSTALL.md` · 训练：见本文档 §3

---

## 1. 本次改造总览（已完成）

| 改造 | 文件 | 说明 |
|---|---|---|
| **移除 `common/` 包** | `common/` 已删除 | 网格单一来源改为 `OSZ/config.py`，与 ResWorld `grid_config` 对齐 |
| **OSZ 网格对齐 ResWorld** | `OSZ/config.py` | `x∈[-15,15] @ 0.15 m`、`y∈[-30,30] @ 0.3 m` → **200×200 各向异性**，掩码与 BEV 特征同格、**无需重采样**；`common/bev_config.py` 的 ±50 m/0.2 m 网格废弃 |
| **几何函数迁移** | `OSZ/utils/geometry.py` | 吸收 former `common/coords.py` 中 OSZ 实际使用的函数（`get_map_name` / `ego_pose_from_sample_data` / `get_vehicle_states_ego` / `bev_box_corners_ego` / `VEHICLE_CATEGORIES`）+ 网格函数（`bev_grid_shape` / `bev_coords_to_pixel` / `pixel_to_bev_coords` / `bev_extent`）；未使用的死代码未迁移（YAGNI） |
| **各向异性适配** | `ray_casting.py` / `bev_height_builder.py` / `drivable_filter.py` / `bev_viz.py` / `run_osz_pipeline.py` / `visualize_nuscenes_sample.py` | `RayCaster3D.bev_res_x/y`、ego 清空半径按物理米、像素↔米映射按轴、drivable 膨胀按粗轴；`bev_range/bev_res` 参数收敛到单一网格来源 |
| **修复双包名 bug** | `bev_height_builder.py` / `ray_casting.py` | `from modules.xxx` 隐式导入统一为 `from OSZ.modules.xxx`（旧代码中 `OSZ.modules` 与 `modules` 是不同模块对象，导致 `isinstance(MockDepthEstimator)` 失效、mock 深度退化为 20 m 常量墙；修复后 occupied 从 215 → 51 真实 LiDAR 高度）。**现在从仓库根直接运行即可，不再需要 `cd OSZ`** |
| **MiDaS v2.1 Small 深度估计** | `depth_estimator.py` / `config.py` | 默认深度模型改为 MiDaS v2.1 Small（MiDaSNet-small / EfficientNet-Lite3，torch 1.9.1 原生可跑、无需 timm）：本地权重 `OSZ/weights/midas_v21_small_256.pt` + 本地 repo `OSZ/third_party/MiDaS`（`torch.hub source='local'`，入口 `MiDaS_small` + 官方 `small_transform`，已核对上游 hubconf），**运行期零网络**；逆深度自动走 `align_to_lidar` inverse 分支；缺失回退 `MockDepthEstimator` |
| **OSZ 批量导出** | `OSZ/export_osz_dataset.py`（新增） | 逐帧导出 `{token}.npz`（`bev_height / osz_ground / osz_eye / semi / drivable_mask`，200×200）；支持 `--num_workers` 进程池、`--shard/--num_shards` 外壳并行、`--overwrite` 断点续跑 |
| **ResWorld Stage 2 接入** | `nuscenes_resworld_dataset.py` / `resworld_config.py` / `resworld_head.py` / `resworld.py` | 数据管线按 token 加载 osz npz（`lru_cache`，缺失全零=全可见）；`CustomCollect3D` keys 加 `osz_mask`；head 新增 `OcclusionAwareFusion`：1×1 Conv 从 `(osz_eye, osz_ground, semi)` 生成可学习嵌入 `E_occ`，`B̃ = B⊙(1-M) + E_occ⊙M`，注入点在 `bev_fusion_conv` 之后、TokenLearner 之前；训练/测试全链路传递 `osz_mask`（测试路径 `kwargs.pop` 防重复关键字 TypeError + `[:1]` 保留 batch 维） |
| **Stage 2 开关化（use_osz_midas / use_osz_rcsample）** | `nuscenes_resworld_dataset.py` / `resworld_config.py` | 两个**互斥源开关**（assert 防同开）+ 派生注入总闸：`use_osz_midas`（离线 npz：dataset 加载 + `CustomCollect3D` 收集 `osz_mask`；LiDAR 上界臂复用此开关，npz 目录决定来源）、`use_osz_rcsample`（在线同源，见下）；`use_osz = use_osz_midas or use_osz_rcsample` 仅作 head 融合分支总闸。双 False = 严格基线 / 消融 5.2-1（dataset 不加载 npz、keys 条件化移除 `osz_mask`、融合分支整体跳过，零开销） |
| **审查修复（3 轮 review）** | `resworld.py` / `nuscenes_resworld_dataset.py` / `depth_estimator.py` / `ray_casting.py` | 修复：①`simple_test`→`simple_test_pts` 断链（评估时注入静默失效）；②`osz_mask=kwargs.get()+**kwargs` 重复关键字 TypeError → `kwargs.pop`+显式传参；③`osz_mask[0]` 3D 与 head 4D 不匹配 → `[:1]` 保留 batch 维；④lru_cache 共享数组 → 三路径 `writeable=False` + `(3,200,200)` 形状断言；⑤`align_to_lidar` linear 模式对称 `shift>10` 回退（防幻影墙）；⑥MiDaS 加载 `model.` 前缀剥离；⑦`z_res` 收敛到 `config.py::Z_RES_M` 单一来源；⑧重复/死导入清理 |
| **测试** | `tests/test_osz_grid.py` / `test_osz_export.py` / `test_resworld_osz.py` / `test_docs_consistency.py`（新增） | **23 项 pytest 全部通过**（本机）：网格 200×200、npz 导出/加载、`OcclusionAwareFusion` 数学性质、接线回归守卫、`_load_osz_mask` 行为、`align_to_lidar` 回退、文档-代码一致性 |
| **use_osz 条件实例化（修复 DDP 未使用参数）** | `resworld_head.py` / `resworld_config.py` | `use_osz=False` 时不再创建 `osz_fusion` 模块（`pts_bbox_head` 配置透传 `use_osz=use_osz`，`_init_layers` 条件实例化 + forward 双条件守卫）：消除 4 卡 DDP（`find_unused_parameters=False`）下 "Parameter indices 27 28 did not receive grad" 报错，真正兑现"严格基线零开销"。**注意**：开关切换后 state_dict 结构不同，ckpt 不可跨开关 resume |
| **移除 mock 合成数据路径** | `nuscenes_loader.py` / `run_osz_pipeline.py` / `export_osz_dataset.py` / `OSZ/README.md` / `OSZ/utils/__init__.py` | OSZ 只接受真实 nuScenes 输入：loader 数据/依赖缺失改为 `raise FileNotFoundError`（不再静默合成），`--mock` 参数、`_mock_iter` 合成场景（~100 行）全部删除，docstring/README 同步；`MockDepthEstimator` **保留**（`--depth-source lidar` 上界臂依赖，返回真实 LiDAR 稠密深度） |
| **RCSample 深度估计器与训练管线严格对齐** | `rcsample_depth_estimator.py` | 图像 resize 改用 PIL 默认 BICUBIC（与 `loading.py::img_transform_core` 一致，原 BILINEAR 有亚像素偏差）；`_load()` 新增 config 断言（`input_size`/`resize_test=0`/`crop_h=(0,0)`），配置漂移时 fail loudly；复制的 `_make_mlp_input`/`_training_aug` 逻辑已逐项与 `rcsample.py::get_mlp_input`、`loading.py` 核对 |
| **OSZ 几何 GPU 化（--backend torch）** | `torch_pipeline.py`（新增）/ `depth_estimator.py` / `rcsample_depth_estimator.py` / `export_osz_dataset.py` | torch 版反投影+BEV 高度图+高度感知射线，与 numpy 版同签名（`compute_osz_height_aware_from_cameras_torch`，输出 numpy）；`--backend {numpy,torch,auto}`（auto=有 GPU 用 torch）；estimator 新增 `infer_tensor`（RCSample 全 GPU、MiDaS 前向 GPU+对齐 CPU、LiDAR 直接取张量），深度与几何免逐帧 CPU 往返；**兼容服务器 torch 1.9.1**：无 `scatter_reduce_`，BEV 高度 max 聚合用「float64 编码 idx·BIG + cummax + 组末 scatter_」段式 max；`use_uncertainty=True` 的逆不确定性融合已实现（相机不确定性∝距离、LiDAR∝1/密度，`scatter_add_` 聚合，对拍一致）；对拍测试 `tests/test_torch_osz.py`（numpy vs torch 逐格 IoU/数值一致，本机 4 项通过） |
| **训练在线生成 OSZ（use_osz_rcsample）** | `resworld.py` / `resworld_config.py` / `torch_pipeline.py::build_osz_mask_online` | 互斥双源开关之一 `use_osz_rcsample`：`True` 时 ResWorld 用自身 RCSample 深度（view transformer 已产出）在训练/推理循环内实时生成掩码（同源，掩码随模型感知演化）；`False` 走离线 npz（`use_osz_midas`）。关键几何修正：等效内参 `K_eff = A@intrins`（`A=[[R,t],[0,0,1]]`，`R`=post_rots 2×2、`t`=post_trans 前两维——裁剪平移必须在 `K_eff` 里）；`depth_scale` 约定（GT 深度加载时除以 `depth_scale`，`post_rot[2,2]=1/depth_scale`）需在反投影前乘回 `ds=1/post_rots[2,2]`，否则几何缩小 ±20%。训练/测试掩码对称（均不传 lidar_depth，纯模型深度）；掩码为 detach 条件输入（几何不可微）。4 项对拍全过（含 `depth_scale≠1` 严格用例） |
| **Stage 3 MHST-Head（已实现）** | `resworld_head.py` / `resworld_config.py` | 新增 `OcclusionMHSTHead`（嫁接在 `pred_bev = tokenfuser(...)+bev_navi_embed` 之后、`col_attn` 之前）：先验网络 `[pred_bev|mask]→1×1→3×3(邻域)→K logits→softmax=π`、共享骨干 + K 个 expert 分支（ΔB^k）、`σ=softplus(s)+Σ_min`；门控合成 `fused = pred_bev·(1-M) + Σ_k π_k·(pred_bev+ΔB^k)·M`（M=osz_eye）。**布局契约（确定性）**：`pred_bev` 恒为 3D `(B, HW=10000, C)`（tokenfuser 输出、`col_attn` 经 `permute(1,0,2)` 消费，HW=100×100 由 `bev_query` 与 `pos_embd` cat 约束）；MHST 是逐位置卷积头，统一转 4D `(B,C,100,100)` 操作、处理完还原 3D（无损布局转换）；`osz_mask` 恒为 4D `(B,3,H,W)`（在线 mask 直接返回、离线 npz 经 torch `default_collate` stack 加 batch 维，已本机验证），两处 4D 断言 fail loudly。**use_oarwm=False = 严格基线**：MHST 不创建、`pred_bev` 直通 `col_attn`（零改动），可随时关开关做消融 5.2-2。`mhst_k`（K=1 单假设消融）、`mhst_sigma_min`、`mhst_sigma_reg_weight=1e-4`（sigma 占位正则，Stage 6 换 `L_uncertainty`）；`outs['mhst']` 输出 π/Σ/ΔB 供 Stage 6 损失（L_occ_halluc/L_div/L_uncertainty）。本机数学性质自检全过（恒等/路由/pi 归一/σ 下限/K=1/全参数梯度/3D↔4D 无损/4D 契约断言） |

---

## 2. 你需要做的事（服务器待办）

### 2.1 一次性准备（8×3090 训练机，resworld 环境）

```shell
# 1) MiDaS 权重（约 86 MB，GitHub release，一般无需翻墙）
mkdir -p OSZ/weights
curl -L -o OSZ/weights/midas_v21_small_256.pt \
    https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt
# 2) MiDaS 推理代码（官方 repo，一次性 clone）
git clone https://github.com/isl-org/MiDaS.git OSZ/third_party/MiDaS
# 注：无需 timm——MiDaS_small 是 EfficientNet-Lite3 编码器，不是 DPT
```

### 2.2 服务器验证

```shell
# 2) MiDaS 真实深度估计冒烟（权重就位后，任意一张图）
#    期望看到 [align_to_lidar] ... inverse 拟合日志（无 LiDAR 时仅相对深度）
python OSZ/run_osz_pipeline.py --dataroot data/nuscenes \
    --version v1.0-mini --max_samples 1 --outdir ./osz_output

# 3) 批量导出 OSZ 掩码（nuScenes 官方划分：train 28130 + val 6019 = 34149 帧；
mkdir -p work_dirs/logs
for i in $(seq 0 7); do
  nohup python OSZ/export_osz_dataset.py --dataroot data/nuscenes \
      --version v1.0-trainval --outdir data/osz --use_drivable \
      --shard $i --num_shards 8 \
      > work_dirs/logs/osz_shard_$i.log 2>&1 &
done
# 4) 训练/评估
bash tools/dist_train.sh projects/configs/resworld/resworld_config.py 4
nohup bash -c "CUDA_VISIBLE_DEVICES=4,5,6,7 bash tools/dist_train.sh projects/configs/resworld/resworld_config.py 4" > /data2/jhc/OARWM/work_dirs/train.log 2>&1 &
```

---

## 3. 训练与评估

### 3.1 训练

```shell
conda activate resworld
# 前置：data/nuscenes/ 与 ckpts/geobev-r50-nuimage-cbgs.pth 就绪
bash tools/dist_train.sh projects/configs/resworld/resworld_config.py 4
```

- 输出 `work_dirs/oa_resworld_config/`（`resworld_config.py` 内 `work_dir='work_dirs/oa_resworld_config'`，CLI `--work-dir` 可覆盖），EMA 权重 `epoch_12_ema.pth`。
- 配置要点（`resworld_config.py`）：图像 256×704（源 900×1600）、6 相机、`num_frames=3`（`multi_adj_frame_id_cfg=(1, 1+2, 1)`）；BEV 网格 200×200 各向异性（x∈[-15,15]@0.15 m、y∈[-30,30]@0.3 m）；loss = depth + 检测 + 地图 + 规划（`loss_plan_reg=L1, w=10.0`）。
- OSZ：`osz_dir='data/osz/'` 按 token 加载掩码（`OcclusionAwareFusion` 注入 `B̃=B⊙(1-M)+E_occ⊙M`）；未导出时全零回退 = 基线等价。

### 3.2 评估（UniAD/VAD 风格开环指标）

```shell
bash 
nohup bash -c "CUDA_VISIBLE_DEVICES=4,5,6,7 tools/dist_test.sh projects/configs/resworld/resworld_config.py work_dirs/oa_resworld_config/epoch_12_ema.pth 4 --eval bbox" > /data2/jhc/OARWM/work_dirs/eval.log 2>&1 &
```

官方参考指标（README）：

| 指标 | L2 1s | L2 2s | L2 3s | L2 Avg | CR 1s | CR 2s | CR 3s | CR Avg |
|---|---|---|---|---|---|---|---|---|
| L2_MAX / CR_MAX | 0.19 | 0.50 | 1.08 | 0.59 | 0.02 | 0.06 | 0.43 | 0.17 |
| L2_AVG / CR_AVG | 0.14 | 0.27 | 0.49 | 0.30 | 0.01 | 0.03 | 0.14 | 0.06 |

### 3.3 数据说明

- `vad_nuscenes_infos_temporal_*.pkl` 直接用 VAD 生成文件；`tools/data_converter/vad_nuscenes_converter.py` 仅在需要自行生成时用。
- `nuscenes_map_anns_val.json` 首次评估时由代码自动生成（见 `nuscenes_vad_dataset.py::_format_gt`）。
- `samples_per_gpu=2`（8×RTX 3090），可按显存调整。

### 3.4 OSZ 深度来源消融（--depth-source）

`OSZ/export_osz_dataset.py` 新增三种深度来源，导出**同构 npz**，训练端零改动（照常按 token 加载）：

| source | 深度来源 | 用途 |
|---|---|---|
| `midas`（默认） | MiDaS v2.1 Small + LiDAR 尺度对齐 | 现状 |
| `rcsample` | ResWorld 自身 RCSample 深度头（`OSZ/modules/rcsample_depth_estimator.py`，需 mmdet3d 环境 + `--rcsample-ckpt`） | 深度来源消融（离线导出，arm B）；也是 `use_osz_rcsample=True` 在线同源掩码的深度来源 |
| `lidar` | LiDAR densified（`MockDepthEstimator`） | 上界参考 |

```shell
# arm A: MiDaS（现状，掩码已导出则跳过）
python OSZ/export_osz_dataset.py --dataroot data/nuscenes \
    --version v1.0-trainval --outdir data/osz --use_drivable

# arm B: RCSample（需 ResWorld 训练完成后 + mmdet3d 环境）
python OSZ/export_osz_dataset.py --dataroot data/nuscenes \
    --version v1.0-trainval --outdir data/osz_rcsample --use_drivable \
    --depth-source rcsample --rcsample-ckpt work_dirs/oa_resworld_config/epoch_12_ema.pth

# arm C: LiDAR 上界（LiDAR densified ≈ 真值，衡量深度误差造成的掩码损失）
python OSZ/export_osz_dataset.py --dataroot data/nuscenes \
    --version v1.0-trainval --outdir data/osz_lidar --use_drivable \
    --depth-source lidar
```

加速（GPU 机器上）：加 `--backend torch`（或 `--backend auto`，有 GPU 自动选 torch）——深度+几何整链在 GPU 上，几何部分相对 numpy 提速约一个量级；`--use_uncertainty` 的 torch 实现已支持（逆不确定性融合）。

注意：
- **训练与测试必须用同一深度来源的 npz**（离线模式），否则分布漂移毁掉对比结论；在线模式无此约束（训练/测试同源）；
- 对比实验：三臂（midas / rcsample / lidar）同一 ResWorld 配置训练，比规划 L2、碰撞率 + 掩码质量（与 GT 遮挡的 IoU / precision / recall）。`lidar` 臂为**上界参考**：深度≈真值时 OSZ 几何的最好结果，用于分离"深度误差代价"与"OSZ 方法本身"——A/B 越接近 C 说明 OSZ 对深度来源越鲁棒。

### 3.5 OARWM 消融实验运行指南（Stage 2/3 已实现部分）

**开关怎么改（重要）**：所有开关（`use_osz_midas` / `use_osz_rcsample` / `use_oarwm` / `mhst_k` / `mhst_sigma_min` / `mhst_sigma_reg_weight`）都是 `resworld_config.py` **文件顶层的 Python 变量**，且 `use_osz = use_osz_midas or use_osz_rcsample`、互斥 assert、`use_oarwm → use_osz` 的 assert 都是在 mmcv 解析该文件时**当场执行派生**的。因此：

- ✅ 正确做法：**直接编辑 `resworld_config.py` 顶部的开关区**，保存后跑训练；
- ❌ 不要用 `--cfg-options use_osz_rcsample=True` 之类命令行覆盖——它不会重新派生 `use_osz`、也不会重跑 assert，会导致掩码源与 head 注入总闸（`use_osz`）不一致，掩码注入静默失效。

**每个消融臂用独立 `--work-dir`**：`train.py` 会把本次解析后的完整配置（含所有开关）dump 成 `work_dir/<config名>.py` 快照（`tools/train.py:183`），**同名文件会被下一次训练覆盖**；独立 work_dir 才能保证每个消融臂的配置快照留存。work_dir 内的 `{timestamp}.log`（mmcv 打印完整配置）也一并保留，双保险可追溯。

**当前可执行的消融臂**（对应设计文档 5.2 中 Stage 2/3 已实现部分；Stage 5 Minimax/CVaR/信息增益、Stage 6 损失消融待实现后补充）：

| 消融臂 | 目的（对应 5.2） | config 开关 | 运行命令 |
|---|---|---|---|
| baseline | 纯 ResWorld 基线（5.2-1：w/o 显式掩码） | `use_osz_midas=False, use_osz_rcsample=False, use_oarwm=False` | `bash tools/dist_train.sh projects/configs/resworld/resworld_config.py 4 --work-dir work_dirs/abl_baseline` |
| stage2 only | 仅掩码注入、无 MHST（对照"几何先验 vs 多假设"） | 开一个掩码源，`use_oarwm=False` | 同上（`--work-dir work_dirs/abl_stage2`） |
| oarwm 主配置 | Stage 2 + 3 完整（5.2-2 的多假设主体） | 开一个掩码源，`use_oarwm=True, mhst_k=3` | `--work-dir work_dirs/abl_oarwm_k3` |
| K 消融 | 多假设数（5.2-2：K=1 vs 3/5/10） | `use_oarwm=True`，`mhst_k` 分别设 1/3/5/10 | `--work-dir work_dirs/abl_k1` / `abl_k3` / `abl_k5` / `abl_k10` |

掩码源二选一：
- `use_osz_rcsample=True`：在线同源掩码（用模型自身 RCSample 深度实时生成），训练/测试对称、无需先导出 npz；
- `use_osz_midas=True`：离线 npz（需先按 §3.4 导出；`osz_dir` 指向哪个 npz 目录就对应哪个深度来源 arm）。

**完整示例（K=1 vs K=3 消融）**：

```bash
# ① 编辑 resworld_config.py 顶部：use_osz_rcsample=True, use_oarwm=True, mhst_k=3
bash tools/dist_train.sh projects/configs/resworld/resworld_config.py 4 \
    --work-dir work_dirs/abl_k3
# ② 编辑 resworld_config.py：mhst_k=1（其余不动）
bash tools/dist_train.sh projects/configs/resworld/resworld_config.py 4 \
    --work-dir work_dirs/abl_k1
# ③ 评估（各臂用自己的 EMA 权重；nuscenes_map_anns_val.json 首次自动生成）
bash tools/dist_test.sh projects/configs/resworld/resworld_config.py \
    work_dirs/abl_k3/epoch_12_ema.pth 4 --eval bbox
```

注意：
- `mhst_k=1` 是**单假设消融**（遮挡区仍有确定性残差补丁 `pred_bev + ΔB`），不是纯基线；纯基线必须 `use_oarwm=False`；
- **ckpt 不可跨开关加载**：`use_osz`/`use_oarwm` 组合不同时模型结构不同（`osz_fusion`/`mhst` 条件实例化），resume 前必须确认开关一致，跨开关等于换配置重新起训；
- 训练输出统一在 `work_dirs/oa_resworld_config/`（config 内 `work_dir`），消融臂用 `--work-dir` 覆盖，互不干扰。

### 3.6 Stage 3（MHST）训练要点与配置

**已实现输出**：`OcclusionMHSTHead` 输出融合后 `pred_bev`、`π`、`σ`、`ΔB^(k)`（存 `outs['mhst']`）；`head.loss` 加极小权重占位正则 `loss_mhst_sigma = σ.mean() × 1e-4`——仅让 `sigma_net` 参与梯度（`find_unused_parameters=False` 下 DDP 要求），Stage 6 由 `L_uncertainty` 替换。Stage 6 完整损失见设计文档 §6.2 / 6.2b / 6.4。

**配置开关**（`resworld_config.py` 顶层）：

```python
use_oarwm = True          # Stage 3 总开关；False = 不创建 MHST-Head，pred_bev 直通（严格基线，零开销）
mhst_k = 3                # 假设数 K（消融：1/3/5/10；K=1 为单假设消融，仍有残差补丁）
mhst_sigma_min = 0.1      # 不确定性下限 Σ_min
mhst_sigma_reg_weight = 1e-4  # sigma 占位正则权重（Stage 6 换 L_uncertainty）
# assert use_oarwm -> use_osz_midas or use_osz_rcsample
```

**消融等价性**：`use_oarwm=False` → MHST-Head 不创建、`pred_bev` 直通，与基线逐位一致、零额外参数；`K=1` 是单假设消融（仍有确定性残差补丁），并非纯基线。

**已知风险**：①遮挡区监督稀疏 → 依赖设计文档 §6.2/6.2b（暴露自监督 + 检测框 GT）；②K 假设分支显存/耗时 → 共享骨干 + 轻量 expert，K=3 参数增量 <5%；③遮挡/可见交界特征突变 → 门控前对 mask 小半径平滑（3×3 均值）缓解。

---

## 4. 下一步改造路线（按设计文档 Stage 3 → 6）

1. **Stage 3 遮挡区多假设随机残差转移（MHST）—— 已实现（`use_oarwm` 开关）**：
   - 掩码已在 head 内可用（`osz_mask`，200×200，与 BEV 特征同格）→ 可见/遮挡位置路由的输入就绪；
   - 注入点确定：`bev_fusion_conv` 之后（`resworld_head.py` forward，Stage 2）；`pred_bev` 之后（Stage 3，`OcclusionMHSTHead`）；
   - **ResWorld 残差在 latent token 空间**（`res_latent_query = bev_embed[:-1] - bev_embed[1:]`，`resworld_head.py`）——MHST 实现在 `pred_bev` 输出端（设计文档 Stage 3）；
   - 待办（Stage 6）：`L_occ_halluc`（遮挡暴露自监督，需时序相邻帧掩码）→ `L_div`（多样性，防假设坍缩）→ `L_uncertainty`（校准，替换现有 sigma 占位正则）；权重系数进 `resworld_config.py`；
   - **学术自洽补强（设计文档 2026-08 修订）**：新增"多假设价值链条"论证（路径 A：期望输出 + aux 旁路传导至 Stage 4/5）；三条 limitation 已给出解决：①边界感受野 → **已实现多尺度膨胀邻域**（`g_prior` 三个 3×3 dilation=1/2/4 卷积，感受野 3/5/9 格）②自监督静态偏置 → 分工设计（6.2 静态 + 6.2b 检测框 GT 动态）③σ 校准 → 6.4 显式 `L_uncertainty`；
   - **设计接口（未实现，勿当作已实现能力）**：§3.4 信念修正（π 更新需下一帧观测，ResWorld 当前单帧推理）与 3.3 时序递归（T 步 rollout，当前为单帧 K 假设补丁）——后者属 Stage 4 时空演化接口；论文写作须与实现状态区分；
   - 提示性/讨论性内容（答辩词、limitation 讨论、L_occ_halluc 设计记录、σ 校准权衡）已移至仓库根 **`TEMP.md`**，设计文档只保留符合代码最终设计的内容。
2. **Stage 4 遮挡想象风险解码与时空风险场 —— 已实现（`use_risk_field`，默认开）**：
   - 语义不可知风险分解 `R^(k) = w_σ·σ·‖ΔB^(k)‖·1[M]` → 双输出 `R_exp`（CVaR）/ `R_worst`（Minimax）→ 不确定性驱动放大 `(1+β·U·M)`、`U=H(π)/logK` → **占用驱动放大 `(1+γ·s_occ·M)`**（`s_occ` 为遮挡区占用概率，把"变化强度"过滤成"有威胁的变化"）；输出 `outs['risk_field']`；
   - 无学习参数（纯函数 `build_risk_field`）；时空演化接口 `R_{t+τ}` 与语义解码器可选臂仍为设计态；
   - 配置：`use_risk_field=True`（默认）、`risk_beta=2.0`、`risk_w_sigma=1.0`、`risk_gamma=1.0`。
3. **Stage 5 鲁棒规划**：Minimax 安全筛选 + CVaR 约束 + 信息增益奖励（`planner/` 与 `plan_loss.py` 改造）。
4. **Stage 6 训练目标（四项损失已实现，`L_info` 待 Stage 5）**：`L_occ_halluc`（**混合模型对数似然**，同时拟合 π/ΔB/σ）+ `L_occ_gt`（**检测框 GT 栅格化** BEV 占用，BCE 带 `pos_weight=5.0` 对抗全 0 平凡解，动态内容监督）+ `L_div`（**余弦相似度**，有界 [-1,1] 梯度稳定，假设多样性）+ `L_uncertainty`（**显式校准**，σ = 暴露误差平方，替代原占位正则 `loss_mhst_sigma`）；均需 next 帧暴露真值（数据管线已加独立 `next_img_inputs` 通道，仅作监督不进训练输入）；权重 `loss_div/occ_halluc/uncertainty/occ_gt_weight`（0=关）+ `loss_occ_gt_pos_weight` 进 `resworld_config.py`；**当前 `mhst_k=5`**（K 假设数）。
5. **消融**：w/o 掩码注入（`use_osz_midas=False` 且 `use_osz_rcsample=False` 即等价基线）、K=1 vs K=3/5/10（`mhst_k`）、Minimax/CVaR/信息增益开关。

---

## 5. 已知限制

- **深度范围**：MiDaS 输出逆深度经 LiDAR 对齐后为 metric 深度，但无 LiDAR 区域仅相对深度
- **OSZ 网格前方仅 ±15 m**（跟随 ResWorld `grid_config`）：>15 m 前方遮挡物不在世界模型 BEV 内；若需更远，需同时改 ResWorld `grid_config` 与 `OSZ/config.py`（单一来源同步）。
- **各向异性近似**：射线投射在 cell 空间为直线，物理空间角度略拉伸（0.15 vs 0.3 m/cell），OSZ 几何为近似。
- **drivable 膨胀**：按较粗轴（0.3 m）迭代膨胀，x 方向实际膨胀 0.75 m（1.5 m 的设定值折半），偏保守方向安全。