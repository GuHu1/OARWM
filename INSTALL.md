# OARWM-Res 安装与环境准备（INSTALL）

## 1. 项目结构

```text
OARWM/
├── OARWM_ResWorld.md            # 改进设计文档（Stage 1-6）
├── OSZ/                         # 遮挡阴影区流水线（config.py / run_osz_pipeline.py / export_osz_dataset.py / modules/ / utils/ / visualize/）
├── projects/                    # ResWorld 模型（configs/resworld/resworld_config.py + mmdet3d_plugin/resworld/）
└── tools/                       # train.py / test.py / dist_train.sh / dist_test.sh / generate_point_label.py
```

## 2. 环境（resworld，python 3.8 + torch 1.9.1+cu111）

### 2.1 创建环境与依赖

```shell
conda create -n resworld python=3.8 -y
conda activate resworld
conda install -c "nvidia/label/cuda-11.3.1" --override-channels cuda-toolkit -y
conda install -c conda-forge "gcc_linux-64=10.*" "gxx_linux-64=10.*" -y
conda install -c conda-forge libxcrypt -y

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install ninja plyfile black flake8 plotly pytest pyquaternion shapely tqdm tensorboard geffnet
pip install "numpy==1.19.5" "matplotlib==3.5.3" "scikit-image==0.19.3" "networkx==2.2" "pandas==1.4.4" "yapf==0.31.0"
pip install "numba==0.56.4"
pip install "timm==0.6.13"
pip install mmcv-full==1.4.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
pip install mmdet==2.14.0 mmsegmentation==0.14.1
pip install lyft_dataset_sdk --no-deps
pip install nuscenes-devkit==1.1.9

git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d && git checkout -f v0.17.1 && python setup.py develop --no-deps && cd ..

sed -i 's/from numba.errors import NumbaPerformanceWarning/from numba.core.errors import NumbaPerformanceWarning/' \
    mmdetection3d/mmdet3d/datasets/pipelines/data_augment_utils.py
sed -i 's/self.class_names = self.class_range.keys()/self.class_names = list(self.class_range.keys())/' \
    $CONDA_PREFIX/lib/python3.8/site-packages/nuscenes/eval/detection/data_classes.py
sed -i 's/^from setuptools import distutils$/import distutils.version/' \
    $CONDA_PREFIX/lib/python3.8/site-packages/torch/utils/tensorboard/__init__.py
```

### 2.2 编译环境变量自动注入

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/3090_env.sh" <<'EOF'
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.6"
export PYTHONPATH="<仓库根>:$PYTHONPATH"
EOF

mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/3090_env.sh" <<'EOF'
unset CUDA_HOME
unset TORCH_CUDA_ARCH_LIST
export PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -v "^$CONDA_PREFIX/bin$" | paste -sd:)
export LD_LIBRARY_PATH=$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "^$CONDA_PREFIX/lib64$" | paste -sd:)
export PYTHONPATH=$(printf '%s' "$PYTHONPATH" | tr ':' '\n' | grep -v "^<仓库根>$" | paste -sd:)
EOF
```

### 2.3 MiDaS 模型准备（本地权重 + 本地 repo，运行期零网络）

```shell
mkdir -p OSZ/weights
curl -L -o OSZ/weights/midas_v21_small_256.pt \
    https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt
git clone https://github.com/isl-org/MiDaS.git OSZ/third_party/MiDaS
```

## 3. 数据准备（nuScenes）

### 3.1 目录结构（放仓库根下，与 `resworld_config.py` 的 `data_root` 一致）

```text
OARWM/data/nuscenes/
├── lidarseg/                    # nuScenes-lidarseg 标注（生成深度标签必需）
├── maps/                        # HD maps（OSZ --use_drivable 必需）
├── samples/                     # 相机关键帧
├── samples_point_label/         # generate_point_label.py 输出
├── sweeps/                      # LiDAR sweeps
├── v1.0-trainval/               # 标注 JSON
├── vad_nuscenes_infos_temporal_train.pkl
├── vad_nuscenes_infos_temporal_val.pkl
└── nuscenes_map_anns_val.json   # 首次评估自动生成，勿手动下载
```

### 3.2 下载清单

| 资源 | 来源 | 用途 |
|---|---|---|
| nuScenes 完整 trainval（samples/sweeps/maps + 标注） | https://www.nuscenes.org/download | 训练/评估/OSZ |
| nuScenes-lidarseg 标注 | 同上（lidarseg 包） | `generate_point_label.py` 生成深度标签 |
| `vad_nuscenes_infos_temporal_train.pkl` | https://drive.google.com/file/d/1OVd6Rw2wYjT_ylihCixzF6_olrAQsctx/view?usp=sharing | 数据管线 |
| `vad_nuscenes_infos_temporal_val.pkl` | https://drive.google.com/file/d/16DZeA-iepMCaeyi57XSXL3vYyhrOQI9S/view?usp=sharing | 评估 |
| `geobev-r50-nuimage-cbgs.pth` | https://drive.google.com/file/d/1B8Bz4_CpHGjgBrD84JbBJrAtx4TOnUG0/view?usp=sharing → 放 `ckpts/` | 预训练 backbone（`load_from`） |

### 3.3 生成深度标签

```shell
conda activate resworld
python tools/generate_point_label.py   # 按脚本顶部 dataroot/save_dir 配置执行
```

### 3.4 磁盘预算

nuScenes 全量约 300+ GB（sweeps 占大头）；`samples_point_label` 额外数十 GB。

## 4. 准备 OSZ 掩码

```shell
conda activate resworld
# 单进程预热（首次联网下载 gen-efficientnet 仓库到 ~/.cache/torch/hub，此后离线）
python OSZ/export_osz_dataset.py --dataroot data/nuscenes \
    --version v1.0-trainval --outdir data/osz --use_drivable \
    --max_samples 3 --overwrite --num_workers 1

# 分片并行全量（勿用 --num_workers>1：多进程同时 torch.hub 会损坏缓存）
for i in $(seq 0 7); do
  python OSZ/export_osz_dataset.py --dataroot data/nuscenes \
      --version v1.0-trainval --outdir data/osz --use_drivable \
      --shard $i --num_shards 8 &
done
wait

ls data/osz | wc -l   # 全量 = 34149（train 28130 + val 6019）
```

输出 `data/osz/{token}.npz`（`bev_height / osz_ground / osz_eye / semi / drivable_mask`，200×200）。训练时 `resworld_config.py` 的 `osz_dir='data/osz/'` 自动按 token 加载。