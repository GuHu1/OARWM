_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

# --- Stage-2 OSZ switches (mutually exclusive mask sources) ---
# False/False = strict baseline / "w/o explicit occlusion mask" ablation
#               (5.2-1): no mask at all — osz_mask is neither loaded nor
#               collected; the head's occlusion branches are skipped
#               entirely (zero overhead, baseline-equal).
# use_osz_midas    = offline masks: precomputed {token}.npz (MiDaS depth)
#                    loaded by the dataset pipeline. The LiDAR upper-bound
#                    arm reuses this switch — the depth source is decided
#                    by which npz dir `osz_dir` points at.
# use_osz_rcsample = online same-source masks: the ResWorld model's own
#                    RCSample depth (already computed by the view
#                    transformer) produces the mask inside the train/test
#                    loop (GPU OSZ geometry adds tens of ms/step).
# MiDaS + LiDAR-aligned masks are the stable training source; online is
# the deployment form. Requires data/osz fully exported with
# --use_drivable BEFORE training (missing npz falls back to all-zeros =
# unconstrained samples).
use_osz_midas = True
use_osz_rcsample = False
assert not (use_osz_midas and use_osz_rcsample), \
    'use_osz_midas and use_osz_rcsample are mutually exclusive'
# Drivable-area constraint: (a) the offline midas path intersects the npz
# mask with its ``drivable_mask`` channel; (b) BOTH paths also hand the
# raw ``drivable_mask`` channel to the head as a RiskHead input.
# Off-road shadows must not gate on-road planning.
use_osz_drivable = True
# Injection gate for the head: True iff any mask source is active.
use_osz = use_osz_midas or use_osz_rcsample

# --- Stage-2 injection (three-state) ---
# 'off'          : no injection — planner BEV stays baseline-clean
#                  (strict baseline / "w/o risk injection" ablation 5.2).
# 'raw_additive' : unsupervised residual offset B~ = B + E_occ*M (ablation
#                  arm — a free-form perturbation of the planner features,
#                  kept only to quantify the effect of injecting occlusion
#                  evidence without safety supervision).
# 'risk_gated'   : risk-gated back-flow (main configuration)
#                  B~ = B + g ⊙ Proj([R_exp.detach(), R_worst.detach()])
#                  — the CONTENT is the safety-supervised RiskHead reading
#                  (detached, ISSUE I2) and the BANDWIDTH is the
#                  zero-init per-channel gate g = tanh(gate_raw) trained
#                  by the planning loss under an L1 budget (ISSUE I3).
#                  Proj keeps the default Kaiming gain (zero-multiply
#                  deadlock guard, ISSUE 2.1).
osz_inject_mode = 'risk_gated'
assert osz_inject_mode in ('off', 'raw_additive', 'risk_gated'), \
    f"unknown osz_inject_mode: {osz_inject_mode!r}"
# Gate warmup in ITERS (design doc Stage 2, ~2000): during warmup the
# gate is detached so the injection value stays EXACTLY 0 until the risk
# head has been shaped by its own supervision (L_col/L_dyn).
gate_warmup_iters = 2000
# L1 bandwidth budget on g (design doc Stage 2, λ ~ 1e-5~1e-4). Always
# computed (warmup included) so gate_raw stays in the autograd graph.
loss_gate_weight = 1e-5

# --- Stage-3 OARWM switch (multi-hypothesis stochastic transition, MHST) ---
# True = graft the MHST head on pred_bev (OARWM_ResWorld.md
#        Stage 3); False = strict baseline (head not even created).
# MHST needs a mask to know where the occluded cells are, so it implies
# a mask source: use_oarwm=True requires use_osz_midas or use_osz_rcsample.
use_oarwm = True
mhst_k = 5            # hypotheses K (ablation 5.2-2: 1 / 3 / 5 / 10)
mhst_sigma_min = 0.1  # occluded-cell uncertainty lower bound Σ_min
# Hard bounds (zombie-hypothesis guard): a hypothesis whose posterior
# collapses to ~0 loses its exposure-supervision gradient and its dB
# would drift freely; the caps keep var/dB bounded.
mhst_delta_clamp = 10.0   # per-element cap on dB^(k)
mhst_sigma_max = 10.0     # sigma upper bound (also caps the e2 target)
assert (not use_oarwm) or use_osz, \
    'use_oarwm (Stage 3 MHST) requires a mask source: ' \
    'set use_osz_midas or use_osz_rcsample'

# --- Stage-4 learnable RiskHead (MC-dropout UCB) ---
use_risk_field = True      # False = no RiskHead (pure MHST forward)
risk_hidden = 64           # conv hidden channels
risk_dropout = 0.1         # MC-dropout rate on the last two layers
risk_mc_t = 4              # MC forwards (mu = mean, sigma_epis = std)
risk_mc_eval = False       # True = MC also at eval (T stochastic forwards);
                           # False = single point estimate at eval
# UCB coefficient β — PROBABILITY-scale semantics: sigma_epis is the std
# of the sigmoid outputs, bounded by 0.5 (std of a Bernoulli), typically
# 0.05-0.2 in practice. beta=1 therefore lifts R_worst by at most
# ~0.2-0.5 over mu; beta=2 already saturates against risk_max=1.0 whenever
# mu > 1 - 2*sigma_epis, and beta > 2 is meaningless.
# Sweep {0.5, 1, 2} = {0.5σ, 1σ, 2σ} UCB.
risk_beta = 1.0            # 1σ UCB upper bound (default)
risk_max = 1.0             # R_exp / R_worst clamp upper bound (probability
                           # semantics: mu = sigmoid(logit), BCE-compatible)
risk_smooth_ks = 5         # Gaussian smoothing kernel size
risk_smooth_sigma = 1.0    # Gaussian smoothing sigma
risk_cmd_proj = 8          # cmd-embedding projection channels (shared
                           # navi_embedding, independent small projection)
assert (not use_risk_field) or use_oarwm, \
    'use_risk_field (Stage 4 RiskHead) requires use_oarwm (Stage 3)'

# --- Stage-5 risk-weighted planning ---
use_risk_plan = True
# 'absolute_hinge' (main, design doc Stage 5): L_plan_guard = mean_t
#   max(0, R_worst(tau_t) - R_safe); R_safe = EMA_0.999[quantile_0.1(
#   mu | collision cells)] — sparse fallback floor with a physical
#   scale; risk avoidance itself lives in the feature layer (Stage-2
#   injection). The field is sampled DETACHED (ISSUE I2): the planner
#   only learns to AVOID risk via the sampling-coordinate branch.
# 'gt_relative' (ablation arm):
#   along-path risk upper-bounded by the GT trajectory (+ CVaR term).
risk_plan_mode = 'absolute_hinge'
assert risk_plan_mode in ('absolute_hinge', 'gt_relative'), \
    f"unknown risk_plan_mode: {risk_plan_mode!r}"
loss_plan_guard_weight = 0.1     # λ_g on L_plan_guard (soft floor)
loss_plan_risk_weight = 0.1      # gt_relative arm only
loss_plan_cvar_weight = 0.0      # CVaR tail term (ablation, default OFF)
cvar_beta = 0.25                 # CVaR tail fraction of the trajectory steps
# R_safe calibration (absolute_hinge): quantile of the SAMPLED R_worst
# over collision-positive cells, EMA'ed — the same bilinear grid_sample
# read the guard hinge consumes, so threshold and hinge share one scale.
# Steps without positives on ANY rank advance nothing (there is no
# quantile to calibrate against). Half-life of EMA 0.99 is ~69 positive
# steps — the sampled scale (~0.001-0.13) is far below the 1.0 init, so
# the faster EMA matters (with 0.999 the guard would stay dormant for
# most of the run). The buffer is aggregated by ONE always-executed
# all_reduce per step (per-rank positive quantiles packed with a valid
# flag, so every rank reaches the collective unconditionally —
# conditionally executed collectives desynchronise the ranks and
# deadlock NCCL) and updated in place (register_buffer binding).
# Init 1.0 = guard dormant until calibrated (R_worst <= risk_max = 1.0).
risk_safe_quantile = 0.1
risk_safe_ema = 0.99
# GT-risk upper-bound margin (gt_relative mode only): max(0, R(pred) -
# R(gt) - margin); 0 = pure upper bound.
risk_plan_margin = 1.0
assert (not use_risk_plan) or use_risk_field, \
    'use_risk_plan (Stage 5) requires use_risk_field (Stage 4)'

# --- Stage-6 loss weights (0 disables a term) ---
loss_div_weight = 0.1            # hypothesis diversity (ΔB pairwise)
loss_occ_halluc_weight = 1.0     # exposure mixture likelihood
loss_uncertainty_weight = 1.0    # σ calibration
loss_occ_gt_weight = 1.0         # occluded-cell dynamic-occupancy BCE
                                 # (s_occ vs S_gt, dynamic boxes only)
loss_occ_gt_pos_weight = 5.0     # BCE pos_weight (penalise missed occupancies)
# risk-field grounding (design doc §6.2c): anchors the raw risk
# intensity (sigma-weighted ||dB||) to the exposure error e2 — zero extra
# data; makes phi an unbiased "future content change" estimate.
loss_risk_ground_weight = 0.1    # 0 disables the term
# Stage-4 safety supervision: trains the RiskHead ONLY (ISSUE I4).
loss_col_weight = 1.0            # collision hard-anchor BCE (mu vs C_gt)
loss_col_pos_weight = 10.0       # pos_weight (collision cells are rare)
loss_dyn_weight = 1.0            # dynamic-occupancy BCE (mu vs S_dyn)
loss_dyn_pos_weight = 5.0        # pos_weight
# S_dyn forward margin along the object heading (m) and the
# dynamic-object velocity threshold (m/s) for the BEV raster.
dyn_forward_margin = 2.0
dyn_vel_thresh = 0.5

#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
voxel_size = [0.15, 0.15, 4]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
num_classes = len(class_names)
# map has classes: divider, ped_crossing, boundary
map_classes = ['divider', 'ped_crossing', 'boundary']
map_num_vec = 100
map_fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
map_fixed_ptsnum_per_pred_line = 20
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = len(map_classes)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

grid_config = {
    'x': [-15, 15, 0.15],
    'y': [-30, 30, 0.3],
    'z': [-5, 3, 8],
    'depth': [1.0, 35, 0.5],
}

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.2, 0.2),
    'rot': (-0, 0),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
bev_h_ = 100
bev_w_ = 100
queue_length = 0 # each sequence contains `queue_length` frames.
total_epochs = 12

# OARWM experiment output dir (train.py: cfg.work_dir wins over the
# auto-derived work_dirs/<config-name>; CLI --work-dir overrides it).
work_dir = 'work_dirs/oa_resworld_config'

multi_adj_frame_id_cfg = (1, 1+2, 1)
numC_Trans=80

model = dict(
    type='ResWorld',
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    use_osz_rcsample=use_osz_rcsample,
    use_oarwm=use_oarwm,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='RCSample',
        scale_num=1,
        grid_config=grid_config,
        input_size=data_config['input_size'],
        ins_channels=[512],
        out_channels=numC_Trans,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
        # The offline MiDaS npz is the mask source, so the online mask
        # does not depend on this depth head; 0.1 keeps the depth loss a
        # small share of the total.
        loss_depth_weight=[0.1],
        downsamples=[16]),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        extra_upsample=1,
        out_channels=256),
    pre_process=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_layer=[2,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    pts_bbox_head=dict(
        type='ResWorldHead',
        embed_dims=_dim_,
        num_frames=len(range(*multi_adj_frame_id_cfg))+1,
        grid_config=grid_config,
        num_reg_fcs=2,
        ego_lcf_feat_idx=None,
        valid_fut_ts=6,
        use_osz=use_osz,  # mask gate (offline or online source)
        osz_inject_mode=osz_inject_mode,  # three-state injection
        gate_warmup_iters=gate_warmup_iters,  # gate frozen period (iters)
        loss_gate_weight=loss_gate_weight,  # L1 bandwidth budget on g
        use_oarwm=use_oarwm,  # Stage-3 MHST head
        mhst_k=mhst_k,
        mhst_sigma_min=mhst_sigma_min,
        mhst_delta_clamp=mhst_delta_clamp,  # hard bound on dB^(k)
        mhst_sigma_max=mhst_sigma_max,      # hard bound on sigma
        use_risk_field=use_risk_field,  # Stage-4 learnable RiskHead
        risk_hidden=risk_hidden,
        risk_dropout=risk_dropout,
        risk_mc_t=risk_mc_t,
        risk_mc_eval=risk_mc_eval,
        risk_beta=risk_beta,        # UCB coefficient
        risk_max=risk_max,          # R_exp/R_worst clamp bound
        risk_smooth_ks=risk_smooth_ks,
        risk_smooth_sigma=risk_smooth_sigma,
        risk_cmd_proj=risk_cmd_proj,
        loss_div_weight=loss_div_weight,
        loss_occ_halluc_weight=loss_occ_halluc_weight,
        loss_uncertainty_weight=loss_uncertainty_weight,
        loss_occ_gt_weight=loss_occ_gt_weight,
        loss_occ_gt_pos_weight=loss_occ_gt_pos_weight,
        loss_risk_ground_weight=loss_risk_ground_weight,  # grounding
        loss_col_weight=loss_col_weight,   # collision hard-anchor BCE
        loss_col_pos_weight=loss_col_pos_weight,
        loss_dyn_weight=loss_dyn_weight,   # dynamic-occupancy BCE
        loss_dyn_pos_weight=loss_dyn_pos_weight,
        dyn_forward_margin=dyn_forward_margin,  # S_dyn heading margin (m)
        dyn_vel_thresh=dyn_vel_thresh,          # dynamic velocity threshold
        use_risk_plan=use_risk_plan,
        risk_plan_mode=risk_plan_mode,  # absolute_hinge (main) / gt_relative
        loss_plan_guard_weight=loss_plan_guard_weight,  # L_plan_guard weight
        loss_plan_risk_weight=loss_plan_risk_weight,    # gt_relative arm
        loss_plan_cvar_weight=loss_plan_cvar_weight,    # CVaR ablation term
        cvar_beta=cvar_beta,
        risk_safe_quantile=risk_safe_quantile,  # R_safe calibration quantile
        risk_safe_ema=risk_safe_ema,            # R_safe EMA coefficient
        risk_plan_warmup_epochs=2,  # risk terms off for epochs 0-1 (the
                                    # risk head needs its safety supervision
                                    # before steering the planner)
        # Linear ramp of the risk weights after warmup (0 -> 1 over N
        # epochs): a hard 0->1 switch would spike loss_plan_reg at the
        # activation epoch; the ramp lets reg settle smoothly.
        risk_plan_ramp_epochs=2,
        # GT-risk upper-bound margin (gt_relative mode only).
        risk_plan_margin=risk_plan_margin,
        latent_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=3,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8),
                ],
                feedforward_channels=_ffn_dim_,
                # ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        res_latent_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=3,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8),
                ],
                feedforward_channels=_ffn_dim_,
                # ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        way_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8),
                ],
                feedforward_channels=_ffn_dim_,
                # ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        use_pe=True,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=300,
        num_classes=num_classes,
        in_channels=_dim_,
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_plan_reg=dict(type='L1Loss', loss_weight=10.0)),
    
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range),
        map_assigner=dict(
            type='MapHungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost', weight=1.0),
            pc_range=point_cloud_range))))

dataset_type = 'ResWorldCustomNuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        load_point_label=True,
        sequential=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_ego=True),
    dict(type='CustomCollect3D',\
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img_inputs', 'ego_his_trajs', 'gt_depth', 'can_bus',
               'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat', 'gt_attr_labels']
         + (['osz_mask'] if use_osz_midas else [])
         + (['drivable_mask'] if use_osz_drivable else [])
         + (['next_img_inputs'] if use_osz else []))
]

test_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=False,
        data_config=data_config,
        load_point_label=False,
        sequential=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # dict(type='RandomScaleImageMultiViewImage', scales=[0.4]),
            # dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_label=False, with_ego=True),
            dict(type='CustomCollect3D',\
                 keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'fut_valid_flag', 'can_bus',
                       'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd',
                       'ego_lcf_feat', 'gt_attr_labels']
                 + (['osz_mask'] if use_osz_midas else [])
                 + (['drivable_mask'] if use_osz_drivable else []))])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
        ann_file=data_root + 'vad_nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        queue_length=queue_length,
        # Precomputed OSZ masks (see OSZ/export_osz_dataset.py), loaded only
        # when use_osz_midas=True (offline source). Empty/missing masks fall
        # back to all-zeros = identity.
        osz_dir='data/osz/',
        use_osz=use_osz_midas,
        use_osz_rcsample=use_osz_rcsample,
        use_osz_drivable=use_osz_drivable,
        use_next=use_osz,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        custom_eval_version='vad_nusc_detection_cvpr_2019'),
    val=dict(type=dataset_type,
             data_root=data_root,
             multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
             pc_range=point_cloud_range,
             ann_file=data_root + 'vad_nuscenes_infos_temporal_val.pkl',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1,
             osz_dir='data/osz/',
             use_osz=use_osz_midas,
             use_osz_rcsample=use_osz_rcsample,
             use_osz_drivable=use_osz_drivable,
             use_next=use_osz,
             map_classes=map_classes,
             map_ann_file=data_root + 'nuscenes_map_anns_val.json',
             map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
             map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
             use_pkl_result=True,
             custom_eval_version='vad_nusc_detection_cvpr_2019'),
    test=dict(type=dataset_type,
              data_root=data_root,
              multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
              pc_range=point_cloud_range,
              ann_file=data_root + 'vad_nuscenes_infos_temporal_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality, samples_per_gpu=1,
              osz_dir='data/osz/',
              use_osz=use_osz_midas,
              # Same OSZ switches as the val split — without these the
              # dataset never injects 'drivable_mask' into input_dict while
              # test_pipeline's CustomCollect3D still collects it
              # (top-level use_osz_drivable=True), crashing with
              # KeyError: 'drivable_mask' on the first batch.
              use_osz_rcsample=use_osz_rcsample,
              use_osz_drivable=use_osz_drivable,
              use_next=use_osz,
              map_classes=map_classes,
              map_ann_file=data_root + 'nuscenes_map_anns_val.json',
              map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
              map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
              use_pkl_result=True,
              custom_eval_version='vad_nusc_detection_cvpr_2019'),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[24,])

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# fp16 = dict(loss_scale='dynamic')
# find_unused_parameters = True
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)

custom_hooks = [
    dict(type='CustomSetEpochInfoHook'),
    dict(
        type='DiagLoggerHook',
        interval=log_config['interval'],
        priority='LOW',
    ),
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='SyncbnControlHook',
        syncbn_start_epoch=0,
    ),
]
load_from = 'ckpts/geobev-r50-nuimage-cbgs.pth'

