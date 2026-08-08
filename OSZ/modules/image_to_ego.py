"""Back-project camera depth maps to ego-frame 3D points.

Given a metric depth map from a single camera, the camera intrinsic matrix
``K``, and the camera-to-ego extrinsic ``T_cam2ego``, this module produces a
set of 3D points in the ego vehicle coordinate system (x=forward, y=left,
z=up).

The resulting point cloud is the building block for BEV height maps: each
camera's depth map is back-projected independently, then all points are
merged and binned into BEV cells by ``bev_height_builder.py``.
"""
from __future__ import annotations

import numpy as np

from OSZ import config as cfg


def depth_map_to_ego_points(
    depth_map: np.ndarray,
    K: np.ndarray,
    T_cam2ego: np.ndarray,
    z_min: float = cfg.Z_MIN_M,
    z_max: float = cfg.Z_MAX_M,
    max_depth: float = cfg.MAX_METRIC_DEPTH_M,
) -> np.ndarray:
    """Back-project a metric depth map to 3D points in the ego vehicle frame.

    Parameters
    ----------
    depth_map : np.ndarray
        (H, W) float32 metric depth in metres (z in the camera frame).
    K : np.ndarray
        (3, 3) camera intrinsic matrix.
    T_cam2ego : np.ndarray
        (4, 4) camera -> ego extrinsic transform.
    z_min : float, optional
        Discard ego-frame points below this height (ground filtering).
    z_max : float, optional
        Discard ego-frame points above this height (building/sky filtering).
    max_depth : float, optional
        Discard points farther than this distance (metres).

    Returns
    -------
    np.ndarray
        (N, 3) float32 points in the ego frame (x=fwd, y=left, z=up).
    """
    H, W = depth_map.shape

    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    u = u.ravel()
    v = v.ravel()
    d = depth_map.ravel()

    valid = (d > 0.0) & (d < max_depth)
    u = u[valid]
    v = v[valid]
    d = d[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    pts_cam_h = np.concatenate(
        [pts_cam, np.ones((len(pts_cam), 1), dtype=np.float32)], axis=1
    )
    pts_ego = (T_cam2ego @ pts_cam_h.T).T[:, :3]

    pts_ego = pts_ego[(pts_ego[:, 2] >= z_min) & (pts_ego[:, 2] <= z_max)]
    return pts_ego.astype(np.float32)
