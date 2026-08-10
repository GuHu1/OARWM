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

### 2.3 验证状态说明

- ✅ 本机已验证：OSZ 网格 200×200、mock 全流程、npz 导出/加载、`OcclusionAwareFusion` 数学性质（全零掩码=恒等）、全部改动文件语法、文档-代码一致性（23 项 pytest）。
- ⏳ 服务器待验证：MiDaS 真实推理（权重+repo 就位后）、ResWorld 完整前向/训练（本机无 mmdet3d；`osz_mask` 链路已由 3 轮 review 静态核验 + 接线回归守卫覆盖）。

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