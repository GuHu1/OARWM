import torch
from mmdet.models import DETECTORS

from projects.mmdet3d_plugin.resworld.planner.metric_stp3 import PlanningMetric
from .bevdet import BEVDepth4D

@DETECTORS.register_module()
class ResWorld(BEVDepth4D):
    def __init__(self, 
                fut_ts=6,
                fut_mode=6, 
                use_osz_rcsample=False,
                use_oarwm=False,
                **kwargs):
        super(ResWorld, self).__init__(**kwargs)
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        # Online (same-source) OSZ: use the model's own RCSample depth to
        # compute the occlusion mask in the training/inference loop instead
        # of loading precomputed {token}.npz masks. Mutually exclusive with
        # the offline source (use_osz_midas in the config); the head's
        # injection gate use_osz is True iff either source is active:
        #   use_osz_midas=False & use_osz_rcsample=False -> baseline
        #   use_osz_midas=True                           -> offline npz masks
        #   use_osz_rcsample=True (this)                 -> online RCSample masks
        self.use_osz_rcsample = use_osz_rcsample
        # Stage-3 switch (mirrors the head's use_oarwm): controls whether the
        # next-frame exposure ground truth is encoded.
        self.use_oarwm = use_oarwm
        self._osz_bin_center = None
        self.planning_metric = None

    def encode_next_bev(self, next_img_inputs):
        """Encode the next-frame BEV as exposure ground truth.

        ``next_img_inputs`` is the independent single-frame channel built by
        the pipeline (``sensor2keyego`` already maps next sensor -> CURRENT
        ego frame), so the encoded B_{t+1} lands in the current ego frame.
        Pure supervision channel: runs under no_grad and NEVER enters the
        main training inputs / multi-frame fusion / col_attn.
        """
        imgs_next, sensor2keyegos_next, ego2globals_next, intrins_next, \
            post_rots_next, post_trans_next, bda_next = next_img_inputs
        mlp_input = self.img_view_transformer.get_mlp_input(
            sensor2keyegos_next, ego2globals_next, intrins_next,
            post_rots_next, post_trans_next, bda_next)
        inputs_next = (imgs_next, sensor2keyegos_next, ego2globals_next,
                       intrins_next, post_rots_next, post_trans_next,
                       bda_next, mlp_input)
        with torch.no_grad():
            bev_next, _ = self.prepare_bev_feat(*inputs_next)
            # prepare_bev_feat returns the RCSample grid (200x200); the main
            # branch then passes through bev_encoder (downsample to 100x100)
            # — the exposure ground truth must match the head's BEV feature
            # resolution, so apply the same encoder here.
            bev_next = self.bev_encoder(bev_next)
        return bev_next  # (B, C, H, W) in the current ego frame

    def build_osz_mask_online(self, depth, img_inputs, lidar_depth=None):
        """Compute the OSZ mask from the model's own RCSample depth.

        ``depth`` is the key-frame depth distribution from the view
        transformer ((B*N, D, H, W) softmax). The mask is grid-aligned with
        the BEV feature map (200x200, see OSZ/config.py) and channel-ordered
        like the offline npz: (osz_eye, osz_ground, semi). It is detached
        downstream (discrete geometry) and used as a conditioning input.
        """
        from OSZ.modules.torch_pipeline import build_osz_mask_online as _build
        if self._osz_bin_center is None:
            depth_cfg = self.img_view_transformer.grid_config['depth']
            d = torch.arange(*depth_cfg, dtype=torch.float32,
                             device=depth.device)
            self._osz_bin_center = d + 0.5 * float(depth_cfg[2])
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
            bda, _ = self.prepare_inputs(img_inputs)
        return _build(
            depth, self._osz_bin_center,
            intrins[0], post_rots[0], post_trans[0], sensor2keyegos[0],
            device=depth.device,
            lidar_depth=lidar_depth,
        )

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            return self.extract_img_feat_sequential(img, kwargs['feat_prev'])
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, _ = self.prepare_inputs(img)
        bev_feat_list = []
        depth_list = []
        key_frame = True  # back propagation for key frame only
        for img, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(
                imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin, post_rot,
                               post_tran, bda, mlp_input)
                if key_frame:
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                else:
                    with torch.no_grad():
                        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False
        if pred_prev:
            assert self.align_after_view_transfromation
            assert sensor2keyegos[0].shape[0] == 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)
            ego2globals_curr = \
                ego2globals[0].repeat(self.num_frame - 1, 1, 1, 1)
            sensor2keyegos_curr = \
                sensor2keyegos[0].repeat(self.num_frame - 1, 1, 1, 1)
            ego2globals_prev = torch.cat(ego2globals[1:], dim=0)
            sensor2keyegos_prev = torch.cat(sensor2keyegos[1:], dim=0)
            bda_curr = bda.repeat(self.num_frame - 1, 1, 1)
            return feat_prev, [imgs[0],
                               sensor2keyegos_curr, ego2globals_curr,
                               intrins[0],
                               sensor2keyegos_prev, ego2globals_prev,
                               post_rots[0], post_trans[0],
                               bda_curr]
        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[adj_id]],
                                       bda)

        bev_feat = torch.cat(bev_feat_list, dim=0)
        x = self.bev_encoder(bev_feat)

        return [x], depth_list[0]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      img_inputs=None,
                      gt_depth=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      map_gt_bboxes_ignore=None,
                      ego_his_trajs=None,
                      ego_fut_trajs=None,
                      ego_fut_masks=None,
                      ego_fut_cmd=None,
                      ego_lcf_feat=None,
                      gt_attr_labels=None,
                      **kwargs):

        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)
        if self.use_osz_rcsample:
            # Online (same-source) OSZ: mask from the model's own depth.
            # Overrides any dataset-loaded osz_mask (offline npz path); the
            # discrete geometry detaches it, so it conditions like GT masks.
            # No lidar_depth on purpose: training and test masks stay
            # symmetric (both pure model depth).
            # RCSample returns the depth as a per-scale list ([tensor] for
            # scale_num=1); the online mask needs the single main-scale
            # softmax distribution.
            kwargs['osz_mask'] = self.build_osz_mask_online(depth[0], img_inputs)
        bev_inputs = [img_feats[0], kwargs['can_bus']]
        # Next-frame exposure ground truth (independent supervision channel
        # for L_occ_halluc / L_uncertainty): encoded from the pipeline's
        # next_img_inputs; never part of the main training inputs.
        gt_bev_next = None
        if self.use_oarwm and 'next_img_inputs' in kwargs \
                and kwargs['next_img_inputs'] is not None:
            gt_bev_next = self.encode_next_bev(kwargs['next_img_inputs'])
        losses_pts = self.forward_pts_train(bev_inputs, gt_bboxes_3d, gt_labels_3d,
                                            map_gt_bboxes_3d, map_gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, map_gt_bboxes_ignore,
                                            ego_his_trajs=ego_his_trajs, ego_fut_trajs=ego_fut_trajs,
                                            ego_fut_masks=ego_fut_masks, ego_fut_cmd=ego_fut_cmd,
                                            ego_lcf_feat=ego_lcf_feat, gt_attr_labels=gt_attr_labels,
                                            osz_mask=kwargs.get('osz_mask'),
                                            gt_bev_next=gt_bev_next)
        losses.update(losses_pts)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          map_gt_bboxes_3d,
                          map_gt_labels_3d,                          
                          img_metas,
                          gt_bboxes_ignore=None,
                          map_gt_bboxes_ignore=None,
                          ego_his_trajs=None,
                          ego_fut_trajs=None,
                          ego_fut_masks=None,
                          ego_fut_cmd=None,
                          ego_lcf_feat=None,
                          gt_attr_labels=None,
                          osz_mask=None,
                          gt_bev_next=None):

        outs = self.pts_bbox_head(pts_feats, img_metas,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat, cmd=ego_fut_cmd,
                                  osz_mask=osz_mask,
                                  gt_bev_next=gt_bev_next)
        loss_inputs = [
            gt_bboxes_3d, gt_labels_3d, map_gt_bboxes_3d, map_gt_labels_3d,
            outs, ego_fut_trajs, ego_fut_masks, ego_fut_cmd, gt_attr_labels,
        ]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        img_inputs=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        # img_metas[0][0]['can_bus'][-1] = 0
        # img_metas[0][0]['can_bus'][:3] = 0
        kwargs['can_bus'] = kwargs['can_bus'][0]
        # Pop osz_mask before **kwargs expansion: the key is carried by the
        # test data when use_osz=True (Collect3D), so passing it both
        # explicitly and via **kwargs would raise TypeError. Keep the batch
        # dim (1, 3, H, W) — the head's osz_fusion indexes mask[:, :1] and
        # broadcasts against (bs, C, H, W).
        osz_mask = kwargs.pop('osz_mask', None)
        if osz_mask is not None:
            osz_mask = osz_mask[:1]

        bbox_results = self.simple_test(
            img_metas=img_metas[0],
            # img=img[0],
            img_inputs=img_inputs[0],
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            ego_his_trajs=ego_his_trajs[0],
            ego_fut_trajs=ego_fut_trajs[0],
            ego_fut_cmd=ego_fut_cmd[0],
            ego_lcf_feat=ego_lcf_feat[0],
            gt_attr_labels=gt_attr_labels,
            osz_mask=osz_mask,
            **kwargs
        )

        return bbox_results

    def simple_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        map_gt_bboxes_3d,
        map_gt_labels_3d,
        img=None,
        img_inputs=None,
        points=None,
        fut_valid_flag=None,
        rescale=False,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        osz_mask=None,
        **kwargs
    ):
        """Test function without augmentaiton."""
        img_feats, _, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        if self.use_osz_rcsample and depth is not None:
            # Online (same-source) mask at test time too (deployment form);
            # overrides the offline osz_mask passed by forward_test.
            # depth is a per-scale list from RCSample; take the main scale.
            osz_mask = self.build_osz_mask_online(depth[0], img_inputs)
        bbox_list = [dict() for i in range(len(img_metas))]
        bev_inputs = [img_feats[0], kwargs['can_bus']]
        bbox_pts, metric_dict = self.simple_test_pts(
            bev_inputs,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            map_gt_bboxes_3d,
            map_gt_labels_3d,
            fut_valid_flag=fut_valid_flag,
            rescale=rescale,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
            osz_mask=osz_mask,
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict

        return bbox_list

    def simple_test_pts(
        self,
        x,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        map_gt_bboxes_3d,
        map_gt_labels_3d,
        prev_bev=None,
        fut_valid_flag=None,
        rescale=False,
        start=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        osz_mask=None,
    ):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev, cmd=ego_fut_cmd,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat,
                                  osz_mask=osz_mask)

        bbox_results = []
        for i in range(len(outs['ego_fut_preds'])):
            bbox_result=dict()
            bbox_result['ego_fut_preds'] = outs['ego_fut_preds'][i].cpu()
            bbox_result['ego_fut_cmd'] = ego_fut_cmd.cpu()
            bbox_results.append(bbox_result)

        assert len(bbox_results) == 1, 'only support batch_size=1 now'
        with torch.no_grad():
            gt_bbox = gt_bboxes_3d[0][0]
            gt_map_bbox = map_gt_bboxes_3d[0]
            gt_map_label = map_gt_labels_3d[0].to('cpu')
            gt_attr_label = gt_attr_labels[0][0].to('cpu')
            fut_valid_flag = bool(fut_valid_flag[0][0])
      
            metric_dict={}
            # ego planning metric
            assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'
            ego_fut_preds = bbox_result['ego_fut_preds']
            ego_fut_trajs = ego_fut_trajs[0, 0]

            ego_fut_cmd = ego_fut_cmd[0, 0, 0]
            ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd)[0, 0]

            ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx]
            ego_fut_pred = ego_fut_pred.cumsum(dim=-2)
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs = ego_fut_pred[None],
                gt_ego_fut_trajs = ego_fut_trajs[None],
                gt_agent_boxes = gt_bbox,
                gt_agent_feats = gt_attr_label.unsqueeze(0),
                gt_map_boxes = gt_map_bbox,
                gt_map_labels = gt_map_label,
                fut_valid_flag = fut_valid_flag
            )
            metric_dict.update(metric_dict_planner_stp3)

        return bbox_results, metric_dict

    ### same planning metric as stp3
    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        gt_map_boxes,
        gt_map_labels,
        fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3."""
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
            # 'plan_obj_col_plus_1s':0,
            # 'plan_obj_col_plus_2s':0,
            # 'plan_obj_col_plus_3s':0,
            # 'plan_obj_box_col_plus_1s':0,
            # 'plan_obj_box_col_plus_2s':0,
            # 'plan_obj_box_col_plus_3s':0,
        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian, segmentation_plus = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats, gt_map_boxes, gt_map_labels)
        occupancy = torch.logical_or(segmentation, pedestrian)

        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i+1)*2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                traj_L2_stp3 = self.planning_metric.compute_L2_stp3(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                # obj_coll_plus, obj_box_coll_plus = self.planning_metric.evaluate_coll(
                #     pred_ego_fut_trajs[:, :cur_time].detach(),
                #     gt_ego_fut_trajs[:, :cur_time],
                #     segmentation_plus)
                metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
                metric_dict['plan_obj_col_{}s'.format(i + 1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i + 1)] = obj_box_coll.mean().item()
                metric_dict['plan_L2_stp3_{}s'.format(i+1)] = traj_L2_stp3
                metric_dict['plan_obj_col_stp3_{}s'.format(i + 1)] = obj_coll[-1].item()
                metric_dict['plan_obj_box_col_stp3_{}s'.format(i + 1)] = obj_box_coll[-1].item()
            else:
                metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_L2_stp3_{}s'.format(i + 1)] = 0.0
            
        return metric_dict

    def set_epoch(self, epoch): 
        self.pts_bbox_head.epoch = epoch
