_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

# --- Stage-2 OSZ switches (mutually exclusive mask sources) ---
# False/False = strict baseline / "w/o explicit occlusion mask" ablation
#               (5.2-1): no mask at all — osz_mask is neither loaded nor
#               collected; the head's occlusion fusion branch is skipped
#               entirely (zero overhead, baseline-equal).
# use_osz_midas    = offline masks: precomputed {token}.npz (MiDaS depth)
#                    loaded by the dataset pipeline. The LiDAR upper-bound
#                    arm reuses this switch — the depth source is decided
#                    by which npz dir `osz_dir` points at.
# use_osz_rcsample = online same-source masks: the ResWorld model's own
#                    RCSample depth (already computed by the view
#                    transformer) produces the mask inside the train/test
#                    loop (GPU OSZ geometry adds tens of ms/step).
# One switch per stage; later stages (3+) get their own flags.
# 2026-08-18: switched to the offline MiDaS masks (route A). The online
# rcsample masks tracked the model's own depth head, whose loss sat at
# ~0.7 and pushed occ_frac to 0.58-0.86 (ISSUE.md P1-3) — the mask
# over-exposure was the top suspect for the L2 gap vs baseline. MiDaS +
# LiDAR-aligned masks are stable; online stays the deployment form.
# Requires data/osz fully exported with --use_drivable BEFORE training
# (missing npz falls back to all-zeros = unconstrained samples).
use_osz_midas = True
use_osz_rcsample = False
assert not (use_osz_midas and use_osz_rcsample), \
    'use_osz_midas and use_osz_rcsample are mutually exclusive'
# Drivable-area constraint, now applied on BOTH mask paths (P1-3):
#   (a) offline midas path — the dataset intersects the npz mask with its
#       ``drivable_mask`` channel (identity if the npz was exported
#       without --use_drivable, so this switch is always safe);
#   (b) online rcsample path — the dataset loads the drivable mask and the
#       online geometry intersects with it.
# Off-road shadows must not gate on-road planning.
use_osz_drivable = True
# Injection gate for the head: True iff any mask source is active.
use_osz = use_osz_midas or use_osz_rcsample

# --- Stage-3 OARWM switch (multi-hypothesis stochastic transition, MHST) ---
# True = graft the MHST head on pred_bev (OARWM_ResWorld.md
#        Stage 3); False = strict baseline (head not even created).
# MHST needs a mask to know where the occluded cells are, so it implies
# a mask source: use_oarwm=True requires use_osz_midas or use_osz_rcsample.
use_oarwm = True
mhst_k = 5            # hypotheses K (ablation 5.2-2: 1 / 3 / 5 / 10)
mhst_sigma_min = 0.1  # occluded-cell uncertainty lower bound Σ_min
# P0-6 hard bounds (2026-08): zombie-hypothesis explosion guard. A
# hypothesis whose posterior collapses to ~0 loses its exposure-supervision
# gradient and its dB drifts upward freely; R_worst = max_k then inherits
# the unbounded magnitude (2026-08-13 training explosion). Normal ||dB|| per
# element ~1-3 (||dB||_2 ~16-40) and sigma ~0.1-1, so these caps leave
# ~4-10x headroom. risk_clamp bounds the planner's risk gradients, which
# scale with the field value (set None to disable).
mhst_delta_clamp = 10.0   # per-element cap on dB^(k)
mhst_sigma_max = 10.0     # sigma upper bound (also caps the e2 target)
assert (not use_oarwm) or use_osz, \
    'use_oarwm (Stage 3 MHST) requires a mask source: ' \
    'set use_osz_midas or use_osz_rcsample'

# --- Stage-4 risk field switch (semantic-agnostic proxy, no params) ---
use_risk_field = True     # False = skip the risk field (pure MHST forward)
risk_beta = 2.0           # uncertainty-driven boost upper bound (β)
risk_w_sigma = 1.0        # σ scaling in the risk proxy (w_σ)
risk_gamma = 1.0          # occupancy-driven boost weight (γ, on s_occ·M)
risk_clamp = 100.0        # risk-field upper bound (normal ~1-20; 5x headroom)
assert (not use_risk_field) or use_oarwm, \
    'use_risk_field (Stage 4) requires use_oarwm (Stage 3)'

# --- Stage-5 risk-weighted planning (approach A: no candidate selection) ---
use_risk_plan = True             # False = skip risk-weighted planning losses
# P0-1 (2026-08): the risk terms are soft regularisers at ~0.1 of
# loss_plan_reg (L1 weight 10.0) so the risk field steers the trajectory
# without overpowering the GT regression (ISSUE.md P0-1 suggestion 2).
# Combined with risk_plan_warmup_epochs=2 and the build_risk_field detach
# (P0-2), the risk terms can no longer dominate training.
loss_plan_risk_weight = 0.1      # minimax-style mean step risk (soft regulariser)
loss_plan_cvar_weight = 0.1      # CVaR of the step-risk tail (soft regulariser)
loss_plan_info_weight = 0.05     # info gain（末端风险差：pred vs GT，无 margin）
cvar_beta = 0.25                 # CVaR tail fraction of the trajectory steps
# REVIEWED 2026-08-18: 5.0 was calibrated off rf_occ (~10-30), which was the
# WRONG baseline — the trajectory only ever SAMPLES r_cmd ≈ 0.2-0.8 of that
# field (the high-risk cells are off the path). Against ~0.5 the margin 5.0
# made max(0, Dr - 5.0) identically 0 and loss_plan_risk/cvar stayed 0.0000
# for the whole run — Stage-5 minimax/CVaR were never exercised. 1.0 ≈ 1-2
# typical step-risks: it still tolerates field noise around the GT path but
# activates when the predicted path is clearly riskier. Re-check against the
# new [DIAG] r_gt_cmd_mean after the MiDaS mask switch changes the field.
risk_plan_margin = 1.0           # 容差 μ（风险单位）；0 = 纯上界
assert (not use_risk_plan) or use_risk_field, \
    'use_risk_plan (Stage 5) requires use_risk_field (Stage 4)'

# --- Stage-6 loss weights (0 disables a term) ---
loss_div_weight = 0.1            # hypothesis diversity (ΔB pairwise)
loss_occ_halluc_weight = 1.0     # exposure mixture likelihood
loss_uncertainty_weight = 1.0    # σ calibration
loss_occ_gt_weight = 1.0         # detection-box BEV occupancy BCE
loss_occ_gt_pos_weight = 5.0     # BCE pos_weight (penalise missed occupancies)
# risk-field grounding (design doc §6.2c): anchors the raw risk
# intensity (sigma-weighted ||dB||) to the exposure error e2 — zero extra
# data; makes the risk field an unbiased "future content change" estimate.
loss_risk_ground_weight = 0.1    # 0 disables the term

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
        # P1-3 closed (2026-08-18): mask source moved to the offline MiDaS
        # npz (use_osz_midas=True), so the online mask no longer depends on
        # this depth head. Back to the baseline weight 0.1 — at 0.3 the
        # depth loss was ~0.7, the single largest term of total loss ~2.2.
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
        use_osz=use_osz,  # injection gate (offline or online source)
        use_oarwm=use_oarwm,  # Stage-3 MHST head (plan B)
        mhst_k=mhst_k,
        mhst_sigma_min=mhst_sigma_min,
        mhst_delta_clamp=mhst_delta_clamp,  # P0-6 bound
        mhst_sigma_max=mhst_sigma_max,      # P0-6 bound
        use_risk_field=use_risk_field,  # Stage-4 risk field
        risk_beta=risk_beta,
        risk_w_sigma=risk_w_sigma,
        risk_gamma=risk_gamma,
        risk_clamp=risk_clamp,           # P0-6 bound
        loss_div_weight=loss_div_weight,
        loss_occ_halluc_weight=loss_occ_halluc_weight,
        loss_uncertainty_weight=loss_uncertainty_weight,
        loss_occ_gt_weight=loss_occ_gt_weight,
        loss_occ_gt_pos_weight=loss_occ_gt_pos_weight,
        loss_risk_ground_weight=loss_risk_ground_weight,  # grounding
        use_risk_plan=use_risk_plan,
        loss_plan_risk_weight=loss_plan_risk_weight,
        loss_plan_cvar_weight=loss_plan_cvar_weight,
        loss_plan_info_weight=loss_plan_info_weight,
        cvar_beta=cvar_beta,
        risk_plan_warmup_epochs=2,  # risk terms off for epochs 0-1 (P0-1/P1-2)
        # Linear ramp of risk weights after warmup (0 -> 1 over N epochs):
        # the hard switch made loss_plan_reg spike 0.53 -> 1.24 at the
        # activation epoch (2026-08-13/14 log); the ramp lets reg settle
        # smoothly (expected final: reg slightly above init, ~5%).
        risk_plan_ramp_epochs=2,
        # GT-risk upper-bound margin (design doc §5.1): risk terms are
        # relative — max(0, R(pred) - R(gt) - margin); 0 = pure upper bound.
        # 1.0 ≈ 1-2 typical r_cmd values (path-sampled risk ~0.2-0.8, see
        # config top-level risk_plan_margin); was 5.0 which never fired.
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
         + (['drivable_mask'] if use_osz_rcsample and use_osz_drivable else [])
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
                 + (['drivable_mask'] if use_osz_rcsample and use_osz_drivable else []))])
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

