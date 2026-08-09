# OSZ（Occlusion Shadow Zone）模块

从多相机图像 + LiDAR 估计 BEV 高度图，再做高度感知射线投射，输出遮挡阴影掩码
（`osz_eye` 严格遮挡 / `osz_ground` 地面层遮挡 / `semi` 半遮挡），供 ResWorld
的遮挡感知注入（`OcclusionAwareFusion`）使用。

## 参数

### 网格与高度（`OSZ/config.py`）

| 常量 | 默认 | 说明 |
|---|---|---|
| `BEV_X_MIN/MAX`, `BEV_X_RES` | -15 / 15 / 0.15 | ego-x（前进）范围与分辨率 |
| `BEV_Y_MIN/MAX`, `BEV_Y_RES` | -30 / 30 / 0.3 | ego-y（左侧）范围与分辨率 |
| `BEV_NX/NY` | 200 / 200 | 网格尺寸（与 ResWorld 对齐） |
| `Z_MIN_M` / `Z_MAX_M` / `Z_RES_M` | 0.8 / 3.0 / 0.3 | 障碍物高度门控 / 上限 / 体素 z 分辨率 |
| `OBSERVER_HEIGHT_M` | 1.2 | 观察者眼高（eye 层射线用） |
| `EGO_CLEARANCE_RADIUS_M` | 1.0 | 自车周围清空半径 |
| `MAX_METRIC_DEPTH_M` | 70.0 | 深度截断（防远处幻影） |
| `NUSCENES_CAMERAS` | 6 相机 | nuScenes 相机列表 |
| `MIDAS_MODEL_PATH` / `MIDAS_REPO_PATH` | `OSZ/weights/...` / `OSZ/third_party/MiDaS` | MiDaS 本地权重与本地 repo |
| `DRIVABLE_MAP_LAYERS` / `EXCLUDED_MAP_LAYERS` | drivable_area, carpark_area / walkway, ped_crossing | 可行驶层 |
| `DEFAULT_DRIVABLE_DILATION_M` | 1.5 | drivable 掩码膨胀半径 |
| `MIN_ALIGN_POINTS` | 20 | LiDAR 深度对齐最少点数 |

### CLI 参数

`OSZ/run_osz_pipeline.py`（单帧/少量帧可视化）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataroot` | `/data/sets/nuscenes` | nuScenes 数据根（应与 `resworld_config.py` 一致，如 `data/nuscenes`） |
| `--version` | `v1.0-mini` | 数据集版本 |
| `--sample_token` | 无 | 只处理单个样本 |
| `--max_samples` | 1 | 处理帧数 |
| `--mock` | 关 | 合成数据（无需 nuScenes） |
| `--outdir` | `./osz_output` | PNG/CSV 输出目录 |
| `--observer_height` | 1.2 | 观察者眼高 |
| `--n_sweeps` | 0 | 聚合历史 LiDAR sweep 数（默认仅关键帧） |
| `--simulate_dropout` | 0.0 | 模拟相机深度条带缺失比例（测 LiDAR 回退） |
| `--use_uncertainty` | 关 | 逆不确定度加权融合 |
| `--use_drivable` | 关 | 与可行驶区域掩码取交集 |

`OSZ/export_osz_dataset.py`（批量导出 npz）：`--dataroot --version --outdir
--max_samples --use_drivable --use_uncertainty --num_workers --shard
--num_shards --overwrite --mock`。每个样本输出 `{outdir}/{token}.npz`，含
`bev_height / osz_ground / osz_eye / semi / drivable_mask`（均 200×200）。

## 环境安装

在 `resworld` 环境（python 3.8 + torch 1.9.1+cu111）中：

```bash
# Python 依赖
pip install numpy scipy matplotlib pillow pyquaternion nuscenes-devkit==1.1.9
pip install "timm==0.6.13"
pip install geffnet

mkdir -p OSZ/weights
curl -L -o OSZ/weights/midas_v21_small_256.pt \
    https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt
git clone https://github.com/isl-org/MiDaS.git OSZ/third_party/MiDaS
```

首次运行会经 `torch.hub` 下载并解压 gen-efficientnet-pytorch 仓库到
`~/.cache/torch/hub/`。

## 运行

```bash
# 1) 单帧可视化
python OSZ/run_osz_pipeline.py --dataroot data/nuscenes \
    --version v1.0-trainval --sample_token <TOKEN> \
    --outdir data/osz_viz --use_drivable

# 2) 批量导出掩码
mkdir -p work_dirs/logs
for i in $(seq 0 7); do
  nohup python OSZ/export_osz_dataset.py --dataroot data/nuscenes \
      --version v1.0-trainval --outdir data/osz --use_drivable \
      --shard $i --num_shards 8 \
      > work_dirs/logs/osz_shard_$i.log 2>&1 &
done
ls data/osz | wc -l   # 全量 = 34149（train+val）
```