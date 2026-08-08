# OARWM-Res 复现指南（Linux 服务器）

> OARWM-Res（设计文档见 `OARWM_ResWorld.md`）。

---

## 1. 项目结构

```text
OARWM/
├── OARWM_ResWorld.md            # OARWM-Res 改进设计文档（Stage 1-6）
├── OSZ/                         # 高度感知遮挡阴影区流水线（设计文档 Stage 2）
│   ├── config.py                #   BEV 网格 + 高度/眼高/深度/相机/HD-map 参数
│   ├── run_osz_pipeline.py      #   端到端主入口（单 sample/批量/mock）
│   ├── modules/                 #   深度估计 → 反投影 → BEV 高度 → 射线投射 → drivable 过滤
│   ├── utils/                   #   nuScenes 加载器（LiDAR 聚合/深度 densify/mock）
│   └── visualize/               #   6 面板 BEV 可视化、GT 叠加、HD-map 图
├── projects/                    # ResWorld 模型（mmdet3d v0.17.1 插件）
│   ├── configs/resworld/resworld_config.py
│   └── mmdet3d_plugin/resworld/ #   bevdet.py / resworld.py / resworld_head.py /
│                                #   planner/ / utils/plan_loss.py ...
└── tools/                       # train.py / test.py / dist_train.sh / dist_test.sh /
                                 # generate_point_label.py / data_converter/
```

> 注：`pa_visibility/`、`pa_osz_mining/` 模块及 `osz_cache` 磁盘缓存不在
> 当前仓库中（裁剪自更大的工程）；`common/` 包已移除，BEV 网格现定义于
> `OSZ/config.py`（对齐 ResWorld `grid_config`），几何函数在
> `OSZ/utils/geometry.py`。

---

## 2. 环境：单一训练环境 + 一次性 OSZ 预处理

OSZ 的深度估计使用 **MiDaS v2.1 Small**（`midas_v21_small_256.pt`，2021
年模型；MiDaS权重离线下载到本地路径，代码**本地加载**。

### 2.1 训练环境 resworld（唯一长期环境，官方配置）

```shell
conda create -n resworld python=3.8 -y
conda activate resworld

# CUDA 编译工具链
conda install -c "nvidia/label/cuda-11.3.1" --override-channels cuda-toolkit -y
conda install -c conda-forge "gcc_linux-64=10.*" "gxx_linux-64=10.*" -y
conda install -c conda-forge libxcrypt -y
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install ninja                  # mmcv/mmdet3d 编译加速
pip install "numpy==1.19.5"
pip install "matplotlib==3.5.3" "scikit-image==0.19.3"
pip install mmcv-full==1.4.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d && git checkout -f v0.17.1 && python setup.py develop --no-deps && cd ..
pip install nuscenes-devkit==1.1.9
pip install pyquaternion shapely tqdm tensorboard
# （scikit-image 已在上文 numpy 段统一安装，版本 0.19.3）
```

### 2.2 编译环境变量自动注入

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/3090_env.sh" <<'EOF'
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.6"
export PYTHONPATH="/data2/jhc/OARWM:$PYTHONPATH"   #改成你的仓库路径
EOF

mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/3090_env.sh" <<'EOF'
unset CUDA_HOME
unset TORCH_CUDA_ARCH_LIST
export PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -v "^$CONDA_PREFIX/bin$" | paste -sd:)
export LD_LIBRARY_PATH=$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "^$CONDA_PREFIX/lib64$" | paste -sd:)
export PYTHONPATH=$(printf '%s' "$PYTHONPATH" | tr ':' '\n' | grep -v "^/data2/jhc/OARWM$" | paste -sd:)#改成你的仓库路径
EOF
```

> 编译提示：mmcv-full 1.4.0 / mmdet3d v0.17.1 的 CUDA 算子用 conda 的
> nvcc 11.3 编译（与 cu111 wheel 同 soname 兼容，驱动 >= 465 即可）。
> 注：conda 无 11.1/11.2 工具链（`cuda-nvcc` 最早 11.3.58；11.1/11.2 label
> 为空），且 `cuda-11.3.0` label 不完整（缺 `cuda-nvml-dev`/`cuda-samples`，
> 装不上），须用补丁版 `cuda-11.3.1` label（2026-07 实测 repodata 依赖闭环）。
> 编译前建议设置 `export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"`（RTX 3090 = 8.6）。

### 2.3 MiDaS 模型准备
**下载权重**（GitHub release，一般无需翻墙；约 86 MB）：

```shell
# 方式 1：浏览器直接下载
#   https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt
# 方式 2（服务器上）：
mkdir -p OSZ/weights
curl -L -o OSZ/weights/midas_v21_small_256.pt \
    https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt
```

**放置路径**：`OSZ/weights/midas_v21_small_256.pt`（目录不存在会自动创建）。

**MiDaS 推理代码**（同样本地化，一次性 clone 官方 repo）：

```shell
git clone https://github.com/isl-org/MiDaS.git OSZ/third_party/MiDaS
```

**代码读取规则**：`OSZ/config.py::MIDAS_MODEL_PATH` 指定权重本地路径、
`MIDAS_REPO_PATH` 指定 repo 本地路径；`DepthEstimator` 用
`torch.hub.load(..., source='local')` 加载官方 `hubconf.py` 的 `MiDaS_small`
入口（MiDaSNet-small，官方 `small_transform` 预处理，保证输入分布与训练
一致；注意 `DPT_Small` 在 hubconf 中不存在），**只从本地读取、绝不联网下载**；
repo/权重缺失时明确报错并自动回退 `MockDepthEstimator`（LiDAR densified
深度），不影响其余流程。

---

## 3. 数据准备（nuScenes）

### 3.1 目录结构（放在仓库根下，与 `resworld_config.py` 的 `data_root` 一致）

```text
OARWM/data/nuscenes/
├── lidarseg/                    # nuScenes-lidarseg 标注（生成深度标签必需）
├── maps/                        # HD maps（OSZ --use_drivable 必需）
├── samples/                     # 相机关键帧
├── samples_point_label/         # generate_point_label.py 输出
├── sweeps/                      # LiDAR sweeps
├── v1.0-trainval/               # 标注 JSON
├── vad_nuscenes_infos_temporal_train.pkl   # VAD 生成
├── vad_nuscenes_infos_temporal_val.pkl
└── nuscenes_map_anns_val.json
```

### 3.2 下载清单

| 资源 | 来源 | 用途 |
|---|---|---|
| nuScenes 完整 trainval（samples/sweeps/maps + 标注） | https://www.nuscenes.org/download | 训练/评估/OSZ |
| nuScenes-lidarseg 标注 | 同上（lidarseg 包） | `generate_point_label.py` 生成 GeoBEV 深度标签 |
| `vad_nuscenes_infos_temporal_train.pkl` | https://drive.google.com/file/d/1OVd6Rw2wYjT_ylihCixzF6_olrAQsctx/view?usp=sharing | ResWorld 数据管线 |
| `vad_nuscenes_infos_temporal_val.pkl` | https://drive.google.com/file/d/16DZeA-iepMCaeyi57XSXL3vYyhrOQI9S/view?usp=sharing | 评估 |
| `geobev-r50-nuimage-cbgs.pth` | https://drive.google.com/file/d/1B8Bz4_CpHGjgBrD84JbBJrAtx4TOnUG0/view?usp=sharing → 放 `ckpts/` | 预训练 backbone（`load_from`） |

### 3.3 生成深度标签（环境 resworld）

```shell
conda activate resworld
python tools/generate_point_label.py   # 按脚本顶部 dataroot/save_dir 配置执行
```

### 3.4 磁盘与时间预算

- nuScenes 全量约 300+ GB（sweeps 占大头）；`samples_point_label` 额外数十 GB。
- 基线 12 epoch 训练：8×RTX 3090 约数天；OARWM 训练同样在
  8×RTX 3090（torch 1.9.1 生态），`samples_per_gpu=2` 可按显存调整。

---

## 4. ResWorld 基线复现

### 4.1 训练（4 GPU 示例，脚本见 `tools/dist_train.sh`）

```shell
conda activate resworld
# 确保 data/nuscenes/ 与 ckpts/geobev-r50-nuimage-cbgs.pth 就绪
bash tools/dist_train.sh projects/configs/resworld/resworld_config.py 4
```

- 输出到 `work_dirs/resworld_config/`，EMA 权重 `epoch_12_ema.pth`。
- 配置要点（`resworld_config.py`）：
  - 图像输入 `256×704`（源图 900×1600）；6 相机；`num_frames=3`（当前 + 2 历史帧，`multi_adj_frame_id_cfg=(1, 1+2, 1)`）。
  - BEV 网格 `grid_config`：`x∈[-15,15] @ 0.15 m`（200 cells）、`y∈[-30,30] @ 0.3 m`（200 cells）——**200×200 各向异性**，与世界模型 BEV 特征（`numC_Trans=80 → 256` 通道）对应。
  - 损失：depth loss + 检测 + 地图 + 规划（`loss_plan_reg=L1, w=10.0`，`plan_loss.py`）。

### 4.2 评估（UniAD/VAD 风格开环指标）

```shell
bash tools/dist_test.sh projects/configs/resworld/resworld_config.py \
    work_dirs/resworld_config/epoch_12_ema.pth 4 --eval bbox
```

官方参考指标（README）：

| 指标 | L2 1s | L2 2s | L2 3s | L2 Avg | CR 1s | CR 2s | CR 3s | CR Avg |
|---|---|---|---|---|---|---|---|---|
| L2_MAX / CR_MAX | 0.19 | 0.50 | 1.08 | 0.59 | 0.02 | 0.06 | 0.43 | 0.17 |
| L2_AVG / CR_AVG | 0.14 | 0.27 | 0.49 | 0.30 | 0.01 | 0.03 | 0.14 | 0.06 |

> 数据集/训练细节与原版 VAD 一致：`vad_nuscenes_infos_temporal_*.pkl` 直接
> 使用 VAD 生成文件，无需自行转换（`tools/data_converter/vad_nuscenes_converter.py`
> 仅在你需要自己生成 pkl 时使用）。

---

## 5. OSZ 模块复现

### 5.1 本地快速验证（无需 nuScenes/torch，任何机器可跑）

```shell
conda activate resworld  # mock 不需要 torch/深度模型，任意环境即可
python OSZ/run_osz_pipeline.py --mock --outdir ./osz_output
```

产出：`height_aware_osz_mock_*.png`（6 面板：BEV 高度 / osz_ground / osz_eye /
semi / 组合 / 统计）+ `summary.csv`。已验证可跑通（200×200 网格，
mock 场景 ground 与 eye 阴影一致）。

### 5.2 真实数据运行（服务器，resworld 环境）

```shell
python OSZ/run_osz_pipeline.py \
    --dataroot /path/to/nuscenes \
    --version v1.0-mini \          # 或 v1.0-trainval
    --outdir ./osz_output \
    --max_samples 10 \
    --use_drivable \               # 需要 maps/；HD-map 缺失时自动回退全 True
    --use_uncertainty \            # inverse-uncertainty 相机-LiDAR 融合
    --n_sweeps 0                   # 默认 0：仅关键帧 LiDAR，避免 OSZ 虚高
# 单 sample：--sample_token <TOKEN>
```

产出：每帧 `height_aware_osz_<token>_sweeps0[_uncertainty][_drivable].png`、
`gt_osz_*.png`、`osz_explained_*.png` + `summary{suffix}.csv`
（occupied / osz_ground / osz_eye / semi 比例）。

### 5.3 关键参数（`OSZ/config.py`，BEV 网格定义于 `OSZ/config.py` 并对齐 ResWorld `grid_config`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `BEV_RANGE_M`（`BEV_X_RES`/`BEV_Y_RES`） | (-15,15,-30,30)，0.15 / 0.3 m/cell → 200×200 | 网格单一来源：`OSZ/config.py`（对齐 ResWorld `grid_config`） |
| `Z_MIN_M` / `Z_MAX_M` | 0.8 / 3.0 m | 反投影高度门控（地面过滤、高楼过滤） |
| `OBSERVER_HEIGHT_M` | 1.2 m | 眼高：`osz_eye` 仅被高度 > 1.2 m 的 cell 阻挡 |
| `EGO_CLEARANCE_RADIUS_M` | 1.0 m | 自车周围清空半径（防自遮挡） |
| `MAX_METRIC_DEPTH_M` | 70.0 m | 深度截断（防远处幻影墙） |
| `MIN_ALIGN_POINTS` | 20 | LiDAR 尺度对齐最少点数 |
| `DEFAULT_DRIVABLE_DILATION_M` | 1.5 m | drivable mask 膨胀半径 |
| 射线投射 | substep=0.25 cell, n_angles≥720 | `cast_osz_height_aware` 向量化实现 |

---

## 6. OARWM-Res 集成路径（Stage 2 → 6）

OSZ 已实现设计文档 **Stage 2** 的 1-6 步（输出 `bev_height`、`osz_ground`、
`osz_eye`、`semi`、`drivable_mask`）；Stage 1/3/5 继承 ResWorld（`projects/`），
Stage 3 的 MHST、Stage 4 风险场、Stage 5 的 Minimax-CVaR-信息增益规划、
Stage 6 新损失**均为待实现部分**。集成时注意以下接口事实（已按代码修正设计文档）：

1. **网格已对齐（无需重采样）**：OSZ 的 BEV 网格已直接对齐 ResWorld
   `grid_config`（`x∈[-15,15] @ 0.15 m`、`y∈[-30,30] @ 0.3 m` → 200×200
   各向异性，定义于 `OSZ/config.py`，需与 `resworld_config.py` 保持同步）。
   掩码/高度图与 ResWorld BEV 特征同格，直接注入即可。
2. **离线预计算（推荐）**：OSZ 全流程在 resworld 环境（MiDaS 内联）批量执行，
   每帧导出 `{sample_token}.npz`（`bev_height / osz_ground / osz_eye / semi /
   drivable_mask`，200×200 与 ResWorld 同格）：

   ```shell
   conda activate resworld
   python OSZ/export_osz_dataset.py \
       --dataroot /data/nuscenes --version v1.0-trainval \
       --outdir data/osz --use_drivable --num_workers 8
   # 或外壳分片并行：--shard $i --num_shards 8（tmux 起 8 个进程）
   # 已导出的 token 默认跳过（断点续跑）；重新生成加 --overwrite
   ```

   训练时按 token 加载（`nuscenes_resworld_dataset.py`，`lru_cache`，缺失
   回退全零=全可见），训练服务器保持单一 resworld 环境。
3. **特征注入归属**：设计文档 Stage 2 步骤 7 的遮挡类型嵌入 `E_occ` 在
   OARWM 模型端实现（OSZ 只出掩码），掩码通道拼接
   `(osz_eye, osz_ground, semi)` 经 1×1 Conv 生成。
4. **残差结构对接**：ResWorld 的残差在 **latent token 空间**（TokenLearner
   压缩 → 相邻帧 latent 差分 `res_latent_query = bev_embed[:-1]-bev_embed[1:]`
   → latent decoder → tokenfuser 还原 `pred_bev`，残差连接），并非逐 BEV
   位置预测。MHST 建议嫁接在 latent token 分支，或在 `pred_bev` 输出加
   掩码门控残差头。
5. **训练硬件**：基线官方 8×RTX 3090（torch 1.9.1+cu111）；OARWM 训练
   同样在这台 8×3090 上（与基线同一环境，保证公平对比），全流程单机完成。

---

## 7. 服务器部署检查清单

```shell
nvidia-smi                      # GPU 可用 & 驱动（conda CUDA 不含驱动，驱动必须系统装）
nvcc --version                  # conda cuda-toolkit 11.3 的 nvcc（编译 mmcv-full/mmdet3d 用）
df -h /data                    # nuScenes 300+ GB + 训练产出
conda env list                  # resworld（训练，长期）；osz_prep（可选：DA V2 离线预处理）
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
