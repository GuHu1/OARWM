> **注意（2026-07）**：本文档为 OSZ 子模块历史状态，其中 `common/`、±50 m @ 0.2 m 网格、`pa_osz_mining` 缓存等描述已过时（`common/` 已移除，网格对齐 ResWorld 200×200 各向异性）。当前改造状态见仓库根 `STATUS.md`。

# OSZ 项目状态

> 最后更新：2026-07-26

## 1. 当前目标

在 nuScenes 上实现一个相机为主、LiDAR 为辅的高度感知遮挡阴影区（OSZ）流水线。核心输出是自车周围 BEV 平面上两类阴影：`osz_ground`（地面层盲区，任何高于地面的遮挡物均产生阴影）与 `osz_eye`（眼高视角盲区，仅高于观察者眼高的障碍物才阻挡视线）。差异区域 `osz_ground & ~osz_eye` 表示“半透明盲区”，风险低于完全盲区。

## 2. 文件结构

```text
OSZ/
├── config.py                       # 从 common.bev_config 导入 BEV 网格；定义高度/深度/相机/地图层
├── run_osz_pipeline.py             # 端到端主入口
├── modules/
│   ├── depth_estimator.py          # Depth Anything V2 + LiDAR 尺度对齐
│   ├── image_to_ego.py             # 深度图反投影到 ego
│   ├── bev_height_builder.py       # 相机/LiDAR 融合 BEV 高度图
│   ├── ray_casting.py              # 高度感知 2D 射线投射
│   └── drivable_filter.py          # HD 地图可行驶区域过滤
├── utils/
│   ├── geometry.py                 # 从 common/coords.py re-export 坐标工具
│   └── nuscenes_loader.py          # nuScenes 加载、LiDAR 聚合、mock 数据
└── visualize/
    ├── bev_viz.py                  # BEV OSZ 可视化
    └── visualize_nuscenes_sample.py # 环视图 + ego-centric HD 地图
```

## 3. 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BEV_RANGE_M` | `(-50, 50, -50, 50)` | OSZ 使用 `(x_min, x_max, y_min, y_max)`，单位 m |
| `BEV_RESOLUTION_M` | `0.2` | BEV 分辨率，单一定义在 `common/bev_config.py` |
| `Z_MIN_M` | `0.8` | 地面过滤高度 |
| `Z_MAX_M` | `3.0` | 最高障碍物高度 |
| `OBSERVER_HEIGHT_M` | `1.2` | 眼高，用于 eye OSZ |
| `EGO_CLEARANCE_RADIUS_M` | `1.0` | 自车中心清空半径 |
| `MAX_METRIC_DEPTH_M` | `70.0` | 深度截断上限 |
| `MIN_ALIGN_POINTS` | `20` | LiDAR 对齐最少点数 |
| `DEFAULT_DRIVABLE_DILATION_M` | `1.5` | 可行驶 mask 膨胀半径 |
| `NUSCENES_CAMERAS` | 6 个环视相机 | 标准 nuScenes 相机名列表 |

## 4. 当前状态

- 主入口 `run_osz_pipeline.py` 可运行：支持单 sample、批量、mock 模式，已通过语法与基础流程校验。
- BEV 网格已统一迁移到 `common/bev_config.py`，`OSZ/config.py` 仅导入；`OSZ/utils/geometry.py` 已从 `common/coords` re-export，避免重复实现。
- OSZ 磁盘缓存位于 `pa_osz_mining/output/osz_cache/{config_hash}/`，由 `(BEV_RANGE_XYXY, BEV_RESOLUTION_M, Z_MIN, Z_MAX, Z_RES)` 的 MD5 决定，修改任一参数自动切换缓存目录。
- 本地无 nuScenes 环境，真实数据验证需在服务器执行；mock 模式可本地快速跑通流程。
- 当前 `--use_drivable` 仅按 HD 地图层过滤，可行驶 mask 仍包含路口、对向车道等区域，导致部分同向相邻车道大车产生几何正确但语义过保守的 OSZ。

## 5. 待解决问题与下一步

1. **同向相邻车道过保守**：在 `drivable_filter.py` 中引入车道方向或 ego 轨迹走廊过滤，降低 PA-relevant OSZ 虚高。
2. **参数调优**：组合对比 `observer_height`、`z_min`、`drivable_dilation`，用 CSV 指标定量评估 OSZ 比例。
3. **Uncertainty 融合量化**：批量对比 `--use_uncertainty` 与硬 fallback 的差异。
4. **深度模型部署**：确认服务器 CUDA/模型缓存，支持本地模型路径传入。
