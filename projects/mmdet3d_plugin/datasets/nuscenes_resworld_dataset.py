import os
import random
import functools
from typing import Optional

import numpy as np
from mmdet.datasets import DATASETS
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .nuscenes_vad_dataset import VADCustomNuScenesDataset

@functools.lru_cache(maxsize=2048)
def _load_drivable_mask(osz_dir: Optional[str], token: str) -> Optional[np.ndarray]:
    """Load the HD-map drivable mask from the offline npz.

    Returns a (200, 200) bool mask (grid-aligned, see OSZ/config.py), or
    None when no npz / no ``drivable_mask`` channel exists — the online
    (rcsample) mask then skips the drivable constraint. The drivable mask
    depends only on the HD map and ego pose, so it is valid for ANY depth
    source (midas / rcsample / lidar).
    """
    if not osz_dir:
        return None
    path = os.path.join(osz_dir, f"{token}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        if "drivable_mask" not in data:
            return None
        m = data["drivable_mask"].astype(np.float32)
    if m.shape != (200, 200):
        raise ValueError(
            f"drivable_mask for {token} has shape {m.shape}, expected "
            "(200, 200) — resync OSZ/config.py with grid_config."
        )
    m.flags.writeable = False
    return m


@functools.lru_cache(maxsize=2048)
def _load_osz_mask(osz_dir: Optional[str], token: str) -> np.ndarray:
    """Load the precomputed OSZ mask for a sample token.

    Returns a ``(3, 200, 200)`` float32 array of channels
    ``(osz_eye, osz_ground, semi)`` — grid-aligned with ResWorld's BEV
    feature map (see ``OSZ/config.py``). Falls back to all-zeros
    (= fully visible, identity for the injection) when ``osz_dir`` is unset
    or the npz is missing, so training stays strictly equivalent to the
    baseline without OSZ.
    """
    if osz_dir:
        path = os.path.join(osz_dir, f"{token}.npz")
        if os.path.exists(path):
            with np.load(path) as data:
                mask = np.stack([
                    data["osz_eye"].astype(np.float32),
                    data["osz_ground"].astype(np.float32),
                    data["semi"].astype(np.float32),
                ])
            # Fail loudly on grid drift instead of silently misaligning.
            if mask.shape != (3, 200, 200):
                raise ValueError(
                    f"osz mask for {token} has shape {mask.shape}, expected "
                    "(3, 200, 200) — resync OSZ/config.py with grid_config."
                )
            # Shared cache entries must be read-only: no in-place writes.
            mask.flags.writeable = False
            return mask
    mask = np.zeros((3, 200, 200), dtype=np.float32)
    mask.flags.writeable = False
    return mask


@DATASETS.register_module()
class ResWorldCustomNuScenesDataset(VADCustomNuScenesDataset):
    def __init__(
        self,
        multi_adj_frame_id_cfg,
        *args,
        **kwargs
    ):
        # Stage-2 OSZ, offline source: `use_osz` here is the config's
        # `use_osz_midas` (precomputed {token}.npz masks). When off, no npz
        # is loaded — online masks (`use_osz_rcsample`) are computed by the
        # model itself, so the dataset stays out of it (strict baseline =
        # both sources off).
        self.use_osz = kwargs.pop('use_osz', False)
        self.osz_dir = kwargs.pop('osz_dir', None)
        # Stage-2 OSZ, online source: the model computes masks from its own
        # RCSample depth; the dataset only supplies the HD-map drivable mask
        # (from the offline npz) so the online mask stays inside the
        # drivable area (ISSUE.md P1-3).
        self.use_osz_rcsample = kwargs.pop('use_osz_rcsample', False)
        # Next-frame supervision channel: independent of the mask source
        # (both midas and rcsample need the exposure ground truth).
        self.use_next = kwargs.pop('use_next', False)
        super().__init__(*args, **kwargs)
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        # token -> index lookup for the next-frame supervision channel.
        self._token2idx = {
            inf['token']: i for i, inf in enumerate(self.data_infos)
        }
        

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
            fut_valid_flag=info['fut_valid_flag'],
            map_location=info['map_location'],
            ego_his_trajs=info['gt_ego_his_trajs'],
            ego_fut_trajs=info['gt_ego_fut_trajs'],
            ego_fut_masks=info['gt_ego_fut_masks'],
            ego_fut_cmd=info['gt_ego_fut_cmd'],
            ego_lcf_feat=info['gt_ego_lcf_feat'],
        )
        if self.use_osz:
            # Stage-2 OSZ masks (see OSZ/export_osz_dataset.py). Missing npz
            # falls back to all-zeros (= identity for the fusion).
            input_dict['osz_mask'] = _load_osz_mask(
                self.osz_dir, info['token'])
        if self.use_osz_rcsample:
            # Drivable-area constraint for the online mask (optional; None
            # when the offline npz is missing).
            input_dict['drivable_mask'] = _load_drivable_mask(
                self.osz_dir, info['token'])
        if 'occ_path' in info:
            input_dict['occ_gt_path'] = info['occ_path']
        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        input_dict["lidar2ego"] = lidar2ego

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            input_dict["camera2ego"] = []
            input_dict["camera_intrinsics"] = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)
            
                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(
                    cam_info["sensor2ego_rotation"]
                ).rotation_matrix
                camera2ego[:3, 3] = cam_info["sensor2ego_translation"]
                input_dict["camera2ego"].append(camera2ego)
                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = cam_info["cam_intrinsic"]
                input_dict["camera_intrinsics"].append(camera_intrinsics)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        # NOTE: now we load gt in test_mode for evaluating
        # if not self.test_mode:
        #     annos = self.get_ann_info(index)
        #     input_dict['ann_info'] = annos

        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = Quaternion(input_dict['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = input_dict['lidar2ego_translation']
        ego2global = np.eye(4)
        ego2global[:3,:3] = Quaternion(input_dict['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = input_dict['ego2global_translation']
        lidar2global = ego2global @ lidar2ego
        input_dict['lidar2global'] = lidar2global

        input_dict.update(dict(curr=info))
        info_adj_list = self.get_adj_info(info, index)
        input_dict.update(dict(adjacent=info_adj_list))
        if self.use_next and info.get('next'):
            # Next-frame supervision channel (exposure ground truth): the
            # t+1 info is loaded as an INDEPENDENT key — it never enters the
            # main img_inputs / multi-frame fusion; it is only encoded to
            # B_{t+1} as supervision for L_occ_halluc / L_uncertainty.
            next_idx = self._token2idx.get(info['next'])
            if next_idx is not None:
                input_dict['next'] = self.data_infos[next_idx]
        if 'scene_token' in info:
            input_dict['scene_token'] = info['scene_token']
        if 'lidar_token' in info:
            input_dict['lidar_token'] = info['lidar_token']
        if 'ann_infos' in info:
            input_dict['ann_infos'] = info['ann_infos']

        return input_dict

    def get_adj_info(self, info, index):
        info_adj_list = []
        adj_id_list = list(range(*self.multi_adj_frame_id_cfg))
        # if self.stereo:
        #     assert self.multi_adj_frame_id_cfg[0] == 1
        #     assert self.multi_adj_frame_id_cfg[2] == 1
        #     adj_id_list.append(self.multi_adj_frame_id_cfg[1])
        for select_id in adj_id_list:
            select_id = max(index - select_id, 0)
            if not self.data_infos[select_id]['scene_token'] == info[
                    'scene_token']:
                info_adj_list.append(info)
            else:
                info_adj_list.append(self.data_infos[select_id])

        for input_dict in info_adj_list:
            rotation = Quaternion(input_dict['ego2global_rotation'])
            translation = input_dict['ego2global_translation']
            can_bus = input_dict['can_bus']
            can_bus[:3] = translation
            can_bus[3:7] = rotation
            patch_angle = quaternion_yaw(rotation) / np.pi * 180
            if patch_angle < 0:
                patch_angle += 360
            can_bus[-2] = patch_angle / 180 * np.pi
            can_bus[-1] = patch_angle
            input_dict['can_bus'] = can_bus
        return info_adj_list

    def prepare_train_data(self, index):

        # temporal aug
        prev_indexs_list = list(range(index-self.queue_length, index))
        random.shuffle(prev_indexs_list)
        prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
        ##

        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        if self.use_next and 'next' not in input_dict:
            # No next frame (sequence end): the exposure ground truth would
            # be missing, leaving sigma_net without supervision (DDP unused-
            # parameter failure). Skip this sample.
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        example = self.vectormap_pipeline(example,input_dict)
        if self.filter_empty_gt and \
                ((example is None or ~(example['gt_labels_3d']._data != -1).any()) or \
                    (example is None or ~(example['map_gt_labels_3d']._data != -1).any())):
            return None
        return example