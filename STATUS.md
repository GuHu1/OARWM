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
| **Stage 2 开关化（use_osz）** | `nuscenes_resworld_dataset.py` / `resworld_config.py` | 顶层 `use_osz` 总开关（一个 Stage 一个开关）：`True` 注入掩码（默认）；`False` = 严格基线 / 消融 5.2-1——dataset 不加载 npz、`CustomCollect3D` keys 条件化移除 `osz_mask`（mmcv collate 对 None 不兼容）、模型融合分支整体跳过（`osz_mask=None` 全链路安全，零开销）。模型层零改动 |
| **审查修复（3 轮 review）** | `resworld.py` / `nuscenes_resworld_dataset.py` / `depth_estimator.py` / `ray_casting.py` | 修复：①`simple_test`→`simple_test_pts` 断链（评估时注入静默失效）；②`osz_mask=kwargs.get()+**kwargs` 重复关键字 TypeError → `kwargs.pop`+显式传参；③`osz_mask[0]` 3D 与 head 4D 不匹配 → `[:1]` 保留 batch 维；④lru_cache 共享数组 → 三路径 `writeable=False` + `(3,200,200)` 形状断言；⑤`align_to_lidar` linear 模式对称 `shift>10` 回退（防幻影墙）；⑥MiDaS 加载 `model.` 前缀剥离；⑦`z_res` 收敛到 `config.py::Z_RES_M` 单一来源；⑧重复/死导入清理 |
| **测试** | `tests/test_osz_grid.py` / `test_osz_export.py` / `test_resworld_osz.py` / `test_docs_consistency.py`（新增） | **23 项 pytest 全部通过**（本机）：网格 200×200、npz 导出/加载、`OcclusionAwareFusion` 数学性质、接线回归守卫、`_load_osz_mask` 行为、`align_to_lidar` 回退、文档-代码一致性 |
| **use_osz 条件实例化（修复 DDP 未使用参数）** | `resworld_head.py` / `resworld_config.py` | `use_osz=False` 时不再创建 `osz_fusion` 模块（`pts_bbox_head` 配置透传 `use_osz=use_osz`，`_init_layers` 条件实例化 + forward 双条件守卫）：消除 4 卡 DDP（`find_unused_parameters=False`）下 "Parameter indices 27 28 did not receive grad" 报错，真正兑现"严格基线零开销"。**注意**：开关切换后 state_dict 结构不同，ckpt 不可跨开关 resume |
| **移除 mock 合成数据路径** | `nuscenes_loader.py` / `run_osz_pipeline.py` / `export_osz_dataset.py` / `OSZ/README.md` / `OSZ/utils/__init__.py` | OSZ 只接受真实 nuScenes 输入：loader 数据/依赖缺失改为 `raise FileNotFoundError`（不再静默合成），`--mock` 参数、`_mock_iter` 合成场景（~100 行）全部删除，docstring/README 同步；`MockDepthEstimator` **保留**（`--depth-source lidar` 上界臂依赖，返回真实 LiDAR 稠密深度） |
| **RCSample 深度估计器与训练管线严格对齐** | `rcsample_depth_estimator.py` | 图像 resize 改用 PIL 默认 BICUBIC（与 `loading.py::img_transform_core` 一致，原 BILINEAR 有亚像素偏差）；`_load()` 新增 config 断言（`input_size`/`resize_test=0`/`crop_h=(0,0)`），配置漂移时 fail loudly；复制的 `_make_mlp_input`/`_training_aug` 逻辑已逐项与 `rcsample.py::get_mlp_input`、`loading.py` 核对 |
| **OSZ 几何 GPU 化（--backend torch）** | `torch_pipeline.py`（新增）/ `depth_estimator.py` / `rcsample_depth_estimator.py` / `export_osz_dataset.py` | torch 版反投影+BEV 高度图+高度感知射线，与 numpy 版同签名（`compute_osz_height_aware_from_cameras_torch`，输出 numpy）；`--backend {numpy,torch,auto}`（auto=有 GPU 用 torch）；estimator 新增 `infer_tensor`（RCSample 全 GPU、MiDaS 前向 GPU+对齐 CPU、LiDAR 直接取张量），深度与几何免逐帧 CPU 往返；**兼容服务器 torch 1.9.1**：无 `scatter_reduce_`，BEV 高度 max 聚合用「float64 编码 idx·BIG + cummax + 组末 scatter_」段式 max；`use_uncertainty=True` 的逆不确定性融合已实现（相机不确定性∝距离、LiDAR∝1/密度，`scatter_add_` 聚合，对拍一致）；对拍测试 `tests/test_torch_osz.py`（numpy vs torch 逐格 IoU/数值一致，本机 4 项通过） |
| **训练在线生成 OSZ（use_rcsample）** | `resworld.py` / `resworld_config.py` / `torch_pipeline.py::build_osz_mask_online` | 新开关 `use_rcsample`：`True` 时 ResWorld 用自身 RCSample 深度（view transformer 已产出）在训练/推理循环内实时生成掩码（同源，掩码随模型感知演化），`False` 走离线 npz。关键几何修正：等效内参 `K_eff = A@intrins`（`A=[[R,t],[0,0,1]]`，`R`=post_rots 2×2、`t`=post_trans 前两维——裁剪平移必须在 `K_eff` 里）；`depth_scale` 约定（GT 深度加载时除以 `depth_scale`，`post_rot[2,2]=1/depth_scale`）需在反投影前乘回 `ds=1/post_rots[2,2]`，否则几何缩小 ±20%。训练/测试掩码对称（均不传 lidar_depth，纯模型深度）；掩码为 detach 条件输入（几何不可微）。4 项对拍全过（含 `depth_scale≠1` 严格用例） |

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

- 输出 `work_dirs/resworld_config/`，EMA 权重 `epoch_12_ema.pth`。
- 配置要点（`resworld_config.py`）：图像 256×704（源 900×1600）、6 相机、`num_frames=3`（`multi_adj_frame_id_cfg=(1, 1+2, 1)`）；BEV 网格 200×200 各向异性（x∈[-15,15]@0.15 m、y∈[-30,30]@0.3 m）；loss = depth + 检测 + 地图 + 规划（`loss_plan_reg=L1, w=10.0`）。
- OSZ：`osz_dir='data/osz/'` 按 token 加载掩码（`OcclusionAwareFusion` 注入 `B̃=B⊙(1-M)+E_occ⊙M`）；未导出时全零回退 = 基线等价。

### 3.2 评估（UniAD/VAD 风格开环指标）

```shell
bash tools/dist_test.sh projects/configs/resworld/resworld_config.py \
    work_dirs/resworld_config/epoch_12_ema.pth 4 --eval bbox
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
| `rcsample` | ResWorld 自身 RCSample 深度头（`OSZ/modules/rcsample_depth_estimator.py`，需 mmdet3d 环境 + `--rcsample-ckpt`） | 深度来源消融（离线导出，arm B）；也是 `use_rcsample=True` 在线同源掩码的深度来源 |
| `lidar` | LiDAR densified（`MockDepthEstimator`） | 上界参考 |

```shell
# arm A: MiDaS（现状，掩码已导出则跳过）
python OSZ/export_osz_dataset.py --dataroot data/nuscenes \
    --version v1.0-trainval --outdir data/osz --use_drivable

# arm B: RCSample（需 ResWorld 训练完成后 + mmdet3d 环境）
python OSZ/export_osz_dataset.py --dataroot data/nuscenes \
    --version v1.0-trainval --outdir data/osz_rcsample --use_drivable \
    --depth-source rcsample --rcsample-ckpt work_dirs/resworld_config/epoch_12_ema.pth

# arm C: LiDAR 上界（LiDAR densified ≈ 真值，衡量深度误差造成的掩码损失）
python OSZ/export_osz_dataset.py --dataroot data/nuscenes \
    --version v1.0-trainval --outdir data/osz_lidar --use_drivable \
    --depth-source lidar
```

加速（GPU 机器上）：加 `--backend torch`（或 `--backend auto`，有 GPU 自动选 torch）——深度+几何整链在 GPU 上，几何部分相对 numpy 提速约一个量级；`--use_uncertainty` 的 torch 实现已支持（逆不确定性融合）。

注意：
- **训练与测试必须用同一深度来源的 npz**（离线模式），否则分布漂移毁掉对比结论；在线模式无此约束（训练/测试同源）；
- 对比实验：三臂（midas / rcsample / lidar）同一 ResWorld 配置训练，比规划 L2、碰撞率 + 掩码质量（与 GT 遮挡的 IoU / precision / recall）。`lidar` 臂为**上界参考**：深度≈真值时 OSZ 几何的最好结果，用于分离"深度误差代价"与"OSZ 方法本身"——A/B 越接近 C 说明 OSZ 对深度来源越鲁棒。

---

## 4. 下一步改造路线（按设计文档 Stage 3 → 6）

1. **Stage 3 遮挡区多假设随机残差转移（MHST）** —— 本次接入的核心接口已就位：
   - 掩码已在 head 内可用（`osz_mask`，200×200，与 BEV 特征同格）→ 可见/遮挡位置路由的输入就绪；
   - 注入点确定：`bev_fusion_conv` 之后（`resworld_head.py` forward）；
   - **ResWorld 残差在 latent token 空间**（`res_latent_query = bev_embed[:-1] - bev_embed[1:]`，`resworld_head.py:227`）——MHST 应嫁接在 latent token 分支（遮挡相关 tokens 走 K 假设转移），或在 `pred_bev` 输出后按掩码加门控残差头（见设计文档 Stage 3"与 Basework 实现的对接"）。
2. **Stage 4 时空风险场**：多假设 BEV 序列解码 + 概率聚合 + 风险放大（`α·M`）。
3. **Stage 5 鲁棒规划**：Minimax 安全筛选 + CVaR 约束 + 信息增益奖励（`planner/` 与 `plan_loss.py` 改造）。
4. **Stage 6 训练目标**：`L_occ_halluc`（遮挡暴露自监督，需时序相邻帧掩码）→ `L_div`（多样性）→ `L_uncertainty`（校准）→ `L_info`；权重系数进 `resworld_config.py`。
5. **消融**：w/o 掩码注入（config 里 `osz_dir=''` 即等价基线）、K=1 vs K=3/5、Minimax/CVaR/信息增益开关。

---

## 5. 已知限制

- **深度范围**：MiDaS 输出逆深度经 LiDAR 对齐后为 metric 深度，但无 LiDAR 区域仅相对深度
- **OSZ 网格前方仅 ±15 m**（跟随 ResWorld `grid_config`）：>15 m 前方遮挡物不在世界模型 BEV 内；若需更远，需同时改 ResWorld `grid_config` 与 `OSZ/config.py`（单一来源同步）。
- **各向异性近似**：射线投射在 cell 空间为直线，物理空间角度略拉伸（0.15 vs 0.3 m/cell），OSZ 几何为近似。
- **drivable 膨胀**：按较粗轴（0.3 m）迭代膨胀，x 方向实际膨胀 0.75 m（1.5 m 的设定值折半），偏保守方向安全。