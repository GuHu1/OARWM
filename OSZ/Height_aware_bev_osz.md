> **注意（2026-07）**：本文档中 `common/bev_config.py`、±50 m @ 0.2 m / 500×500 网格的描述已过时。`common/` 已移除，BEV 网格现定义于 `OSZ/config.py` 并对齐 ResWorld `grid_config`（200×200 各向异性：x 0.15 m/cell、y 0.3 m/cell）。最新状态见仓库根 `STATUS.md`。

# Height-Aware BEV OSZ 设计文档

## 1. 问题定义

传统二值 BEV 把任何占据 cell 后方的区域都判为盲区，忽略了遮挡物高度和观察者眼高。现实中轿车（~1.5 m）不会完全挡住视线，而大车（>2.5 m）会。本流水线将 BEV 占据升级为高度图，并基于虚拟眼高进行分层射线投射。

## 2. 坐标约定

采用 nuScenes ego 坐标系：

- `x`：车辆前进方向
- `y`：车辆左侧
- `z`：向上

BEV 数组 `bev[i, j]` 使用 `indexing='ij'`：

- `i` 对应 ego-x（forward）
- `j` 对应 ego-y（left）

可视化使用 `origin='lower'` + `extent=[y_max, y_min, x_min, x_max]`，图像上方为前方、左侧为左方。

BEV 网格在 `common/bev_config.py` 中统一定义，默认 `BEV_RESOLUTION_M = 0.2`、`BEV_RANGE_XYXY = (-50, 50, -50, 50)`。`OSZ/config.py` 从 `common.bev_config` 导入，不再自行定义。

## 3. 流水线总览

```text
nuScenes sample
    │
    ▼
NuScenesOSZLoader → 6 相机图像 + K/T + 稀疏 LiDAR 深度
    │
    ▼
densify_depth_map → dense LiDAR 深度
    │
    ▼
DepthEstimator.infer → metric 深度（每相机）
    │
    ▼
depth_map_to_ego_points → ego 3D 点云
    │
    ▼
build_bev_height_fused → bev_height_max
    │
    ▼
cast_osz_height_aware → osz_ground, osz_eye
    │
    ▼
build_drivable_mask → 可行驶区域 mask
    │
    ▼
filter_osz_by_drivable → PA-relevant OSZ
    │
    ▼
visualize/bev_viz.py → 可视化
```

## 4. 模块设计

- **`common/bev_config.py`**：BEV 网格单一来源。默认分辨率 0.2 m，范围 ±50 m，导出 OSZ/visibility/pa_osz_mining 三种约定。
- **`OSZ/config.py`**：从 `common.bev_config` 导入 `BEV_RANGE_M`、`BEV_RESOLUTION_M`；定义高度门控、眼高、相机、深度截断、HD 地图层等。
- **`utils/nuscenes_loader.py`**：加载图像、内参 `K`、外参 `T_cam2ego`，聚合 LiDAR sweep， densify 稀疏深度，并提供 mock 数据。
- **`utils/geometry.py`**：不再自行实现几何函数，而是从 `common.coords` re-export `get_map_name`、`get_gt_boxes_ego`、`box_corners_ego`、`ego_pose_from_sample_data`。
- **`modules/depth_estimator.py`**：Depth Anything V2 Small + LiDAR 尺度对齐；支持线性/逆深度模型，退化时回退到 median-ratio；`MockDepthEstimator` 在模型不可用时返回 densified LiDAR 深度。
- **`modules/image_to_ego.py`**：将单相机 metric 深度图反投影为 ego 坐标系 3D 点，并按 `z` 与最大深度门控。
- **`modules/bev_height_builder.py`**：`build_bev_height_max` 按 cell 取最大高度；`build_bev_height_fused` 相机为主、LiDAR fallback；可选 uncertainty 加权融合。
- **`modules/ray_casting.py`**：`cast_osz_height_aware` 从高度图出发，沿 360° 方向以 `substep=0.25` cell 步进；ground 层阻挡高度 >0.05 m 的 cell，eye 层仅阻挡高度 >`observer_height` 的 cell；自车中心按 `EGO_CLEARANCE_RADIUS_M` 清空。
- **`modules/drivable_filter.py`**：从 nuScenes HD 地图读取可行驶层，旋转到 ego 坐标后光栅化并膨胀，最后过滤 OSZ。
- **`visualize/bev_viz.py`**：绘制 GT 叠加与 OSZ 解释图。

## 5. 关键设计决策

1. **BEV 网格单一来源**：统一放在 `common/bev_config.py`，`OSZ/` 只导入，避免多模块硬编码不一致。
2. **相机为主、LiDAR 为辅**：推理优先使用相机预测深度，LiDAR 只在相机为空或失效区域 fallback。
3. **高度感知射线投射**：用 `bev_height_max` 替代二值占据，区分 `osz_ground` 与 `osz_eye`，避免“被轿车包围就全盲”。
4. **深度对齐鲁棒性**：自动检测 inverse-like 相对深度并切换模型；退化 fit 自动回退 median-ratio；截断异常深度。
5. **多帧 LiDAR 默认关闭**：`n_sweeps=0`，避免多帧聚合增厚障碍物墙导致 OSZ 虚高。
6. **磁盘缓存按配置哈希隔离**：OSZ 缓存位于 `pa_osz_mining/output/osz_cache/{config_hash}/`，由 `(BEV_RANGE_XYXY, BEV_RESOLUTION_M, Z_MIN, Z_MAX, Z_RES)` 的 MD5 决定，修改任一参数自动切换目录。

## 6. 局限与下一步

1. **同向相邻车道过保守**：右侧公交车在几何上正确遮挡，但语义上对自车影响有限。下一步在 `drivable_filter.py` 中引入车道方向或 ego 轨迹走廊过滤。
2. **参数敏感**：`observer_height`、`z_min`、`drivable_dilation` 对 OSZ 比例影响大。下一步在服务器上做组合调参实验，用 CSV 指标定量对比。
3. **深度模型部署**：默认使用 Hugging Face 模型 id，仍需确认服务器缓存与 CUDA 环境。下一步支持命令行传入本地模型路径。
4. **Uncertainty 融合未量化**：`--use_uncertainty` 已实现，但缺少与硬 fallback 的批量对比。
