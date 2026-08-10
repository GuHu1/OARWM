"""nuScenes data loading for the OSZ pipeline.

This module provides utilities for loading nuScenes sensor data and
preparing the per-camera dense depth maps consumed by the OSZ pipeline.

Classes
-------
NuScenesOSZLoader
    Iterate over nuScenes samples and return per-camera depth maps,
    intrinsics, extrinsics, and camera images.

Functions
---------
aggregate_lidar_sweeps
    Aggregate the current and past LiDAR sweeps into the current ego frame.
project_lidar_to_camera
    Project ego-frame LiDAR points onto a camera image plane.
densify_depth_map
    Fill a sparse depth map with nearest-neighbour interpolation while
    preserving depth discontinuities.
filter_ground_points
    Remove LiDAR points at or below ground level.

Notes
-----
Coordinate conventions:

- All sensor poses are stored as sensor -> ego -> global.
- Processing is performed entirely in the ego frame at each timestamp.
- ``depth_map`` contains LiDAR points projected onto each camera image,
  where depth equals the camera-frame :math:`z` coordinate.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False
    print("[WARN] nuscenes-devkit not found; install it or check the env.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from scipy.spatial import cKDTree

from OSZ import config as cfg


def densify_depth_map(depth_map: np.ndarray,
                      max_radius: int = 16,
                      depth_discontinuity_thresh: float = 4.0) -> np.ndarray:
    """Fill a sparse depth map using nearest-neighbour interpolation.

    Naive nearest-neighbour interpolation smears object boundaries: a
    background hole next to a vehicle can be filled with the vehicle's
    depth because its nearest valid pixel lies on the vehicle surface.
    That widens the silhouette, inflates occluder voxels after
    back-projection, and makes OSZ spread beyond the real vehicle width.

    For each invalid pixel this function examines the four nearest valid
    neighbours. If their depths differ by more than
    ``depth_discontinuity_thresh``, the pixel lies on a depth edge and is
    left unknown (0). Otherwise the closest neighbour's depth is used.

    Parameters
    ----------
    depth_map : np.ndarray
        Sparse depth map with shape ``(H, W)``. Zero denotes no
        measurement.
    max_radius : int, optional
        Maximum search radius (in pixels) for filling invalid pixels.
        Pixels farther than ``max_radius`` from any valid measurement
        remain zero. Default is 16.
    depth_discontinuity_thresh : float, optional
        Depth range threshold (in metres). If the four nearest valid
        neighbours span a larger range, the pixel is considered to be on
        a depth discontinuity and is not filled. Default is 4.0.

    Returns
    -------
    np.ndarray
        Dense depth map with shape ``(H, W)`` and dtype ``float32``.
        Original valid pixels are preserved; unfilled pixels remain 0.

    Notes
    -----
    The default parameters were tuned per ``OSZ_ERROR_AUDIT.md`` P0:

    - ``max_radius`` increased from 8 to 16 to fill larger object
      interiors.
    - ``depth_discontinuity_thresh`` increased from 1.5 to 4.0 to reduce
      edge gaps that caused fragmented BEV occupancy.
    """
    H, W = depth_map.shape
    valid = depth_map > 0
    if valid.sum() == 0:
        return depth_map.copy()

    coords = np.array(np.nonzero(valid)).T          # (N, 2)  (y, x)
    values = depth_map[valid]

    grid_y, grid_x = np.mgrid[0:H, 0:W]
    grid_coords = np.stack([grid_y.ravel(), grid_x.ravel()], axis=1)

    tree = cKDTree(coords)

    # query K=4 nearest neighbors to detect depth-discontinuity boundaries
    k_neighbors = min(4, len(coords))
    dist_k, idx_k = tree.query(grid_coords, k=k_neighbors)
    if k_neighbors == 1:
        dist_k = dist_k[:, None]
        idx_k  = idx_k[:, None]

    values_k = values[idx_k]                          # (H*W, k) candidate depths
    depth_spread = values_k.max(axis=1) - values_k.min(axis=1)  # (H*W,)

    # nearest-neighbor (k=1) depth and distance are the default interpolation
    nearest_dist  = dist_k[:, 0]
    nearest_depth = values_k[:, 0]

    # pixel is at a depth discontinuity if its K nearest valid neighbors span
    # a large depth range, e.g. it sits between a car edge and the background.
    # In that case interpolation is unreliable, so keep it unknown (0).
    is_discontinuous = depth_spread > depth_discontinuity_thresh

    dense = nearest_depth.copy()
    dense[is_discontinuous] = 0.0
    dense = dense.reshape(H, W)
    dist_map = nearest_dist.reshape(H, W)

    # do not fill pixels far from any valid measurement
    dense_mask = dist_map <= max_radius
    dense = dense * dense_mask.astype(np.float32)

    # keep original valid pixels unchanged
    dense[valid] = depth_map[valid]
    return dense


def _get_transform(nusc, record: Dict) -> np.ndarray:
    """Return calibrated sensor -> ego transform (4x4 float32)."""
    cs = nusc.get('calibrated_sensor', record['calibrated_sensor_token'])
    T = transform_matrix(
        cs['translation'],
        Quaternion(cs['rotation']),
        inverse=False,
    )
    return T.astype(np.float32)


def _get_intrinsic(nusc, cam_token: str) -> np.ndarray:
    """Return the camera intrinsic matrix (3x3)."""
    sample_data = nusc.get('sample_data', cam_token)
    cs = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    K = np.array(cs['camera_intrinsic'], dtype=np.float32)
    return K


def filter_ground_points(pts_ego: np.ndarray,
                         z_thresh: float = cfg.Z_MIN_M) -> np.ndarray:
    """Remove LiDAR points at or below ground level.

    Ground points projected to camera depth maps create false obstacle
    voxels because the ground at 15 m has the same depth as a 0.4 m-tall
    obstacle at 15 m. Filtering them before projection eliminates the
    largest source of phantom OSZ on the road surface.

    Parameters
    ----------
    pts_ego : np.ndarray
        LiDAR points in the ego frame with shape ``(N, 3)``, where
        ``x`` is forward, ``y`` is left, and ``z`` is up.
    z_thresh : float, optional
        Points with ``z_ego < z_thresh`` are classified as ground and
        removed. Defaults to ``cfg.Z_MIN_M``.

    Returns
    -------
    np.ndarray
        Filtered points with shape ``(M, 3)`` containing only points
        with ``z_ego >= z_thresh``.
    """
    return pts_ego[pts_ego[:, 2] >= z_thresh]


def _ego_pose_tf(nusc, sample_token: str):
    """Build ego -> global transform from the LiDAR sample_data ego pose."""
    sample = nusc.get('sample', sample_token)
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ep = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    T = transform_matrix(ep['translation'], Quaternion(ep['rotation']), inverse=False)
    return T.astype(np.float64)


def aggregate_lidar_sweeps(nusc, sample_token: str,
                           n_sweeps: int = 0,
                           ground_z_thresh: float = cfg.Z_MIN_M) -> np.ndarray:
    """Aggregate current and historical LiDAR sweeps into one ego-frame cloud.

    Loads the keyframe LiDAR sweep plus up to ``n_sweeps`` previous
    sweeps (via ``sample_data['prev']`` chain, past only). Each
    historical sweep's points are ego-motion-compensated
    (past_ego -> global -> current_ego) and ground-filtered before
    concatenation.

    This bypasses the depth-map pipeline entirely: the caller can
    voxelize the returned points directly in 3D, avoiding densify or
    voxel-cast errors.

    Parameters
    ----------
    nusc : nuscenes.nuscenes.NuScenes
        Initialized nuScenes dataset instance.
    sample_token : str
        Keyframe sample token for the current frame.
    n_sweeps : int, optional
        Number of past sweeps to include. ``0`` means keyframe only.
        Default is 0.
    ground_z_thresh : float, optional
        ``z_ego`` cutoff for ground removal, in metres. Defaults to
        ``cfg.Z_MIN_M``.

    Returns
    -------
    np.ndarray
        All aggregated points in the current ego frame with shape
        ``(M, 3)`` and dtype ``float32``. Ground points have been
        removed.
    """
    sample = nusc.get('sample', sample_token)
    lidar_tok = sample['data']['LIDAR_TOP']
    lidar_sd = nusc.get('sample_data', lidar_tok)

    # Current ego -> global
    T_cur_ego2global = _ego_pose_tf(nusc, sample_token)

    all_pts = []

    # Current keyframe sweep
    pc = LidarPointCloud.from_file(str(Path(nusc.dataroot) / lidar_sd['filename']))
    T_l2e = _get_transform(nusc, lidar_sd)
    pts_h = np.concatenate([pc.points[:3].T, np.ones((pc.points.shape[1], 1))], axis=1)
    pts_ego = (T_l2e @ pts_h.T).T[:, :3]
    pts_ego = filter_ground_points(pts_ego, ground_z_thresh)
    all_pts.append(pts_ego.astype(np.float32))

    # Historical sweeps (prev chain, past-only)
    tok = lidar_sd['prev']
    for _ in range(n_sweeps):
        if not tok:
            break
        sd = nusc.get('sample_data', tok)

        # Load past sweep points -> past ego frame
        pc_p = LidarPointCloud.from_file(str(Path(nusc.dataroot) / sd['filename']))
        T_l2e_p = _get_transform(nusc, sd)
        pts_p_h = np.concatenate([pc_p.points[:3].T,
                                  np.ones((pc_p.points.shape[1], 1))], axis=1)
        pts_p_ego = (T_l2e_p @ pts_p_h.T).T[:, :3]

        # Ground filter BEFORE motion compensation (z in past ego frame)
        pts_p_ego = filter_ground_points(pts_p_ego, ground_z_thresh)
        if len(pts_p_ego) == 0:
            tok = sd['prev']
            continue

        # Ego-motion compensation: past_ego -> global -> current_ego
        ep = nusc.get('ego_pose', sd['ego_pose_token'])
        T_past_ego2global = transform_matrix(
            ep['translation'], Quaternion(ep['rotation']), inverse=False
        ).astype(np.float64)
        pts_global_h = (T_past_ego2global @
                        np.concatenate([pts_p_ego,
                                        np.ones((len(pts_p_ego), 1))], axis=1).T).T
        pts_cur_ego_h = (np.linalg.inv(T_cur_ego2global) @ pts_global_h.T).T
        pts_cur_ego = pts_cur_ego_h[:, :3].astype(np.float32)

        all_pts.append(pts_cur_ego)

        tok = sd['prev']

    print(f"  [aggregate] {len(all_pts)} sweeps, "
          f"{sum(len(p) for p in all_pts)} points total")
    return np.concatenate(all_pts, axis=0)


def project_lidar_to_camera(
    points_ego: np.ndarray,
    K: np.ndarray,
    T_cam2ego: np.ndarray,
    img_h: int,
    img_w: int,
    min_dist: float = 1.0,
) -> np.ndarray:
    """Project ego-frame LiDAR points onto a camera image.

    Parameters
    ----------
    points_ego : np.ndarray
        LiDAR points in the ego frame with shape ``(N, 3)``.
    K : np.ndarray
        Camera intrinsic matrix with shape ``(3, 3)``.
    T_cam2ego : np.ndarray
        Camera -> ego rigid transform with shape ``(4, 4)``.
    img_h : int
        Output image height in pixels.
    img_w : int
        Output image width in pixels.
    min_dist : float, optional
        Minimum camera-frame depth to retain. Points with ``z_cam <=
        min_dist`` are discarded. Default is 1.0.

    Returns
    -------
    np.ndarray
        Dense depth map with shape ``(img_h, img_w)`` and dtype
        ``float32``. Pixels without a measurement are 0. When multiple
        points project to the same pixel, the closest point is kept.
    """
    T_ego2cam = np.linalg.inv(T_cam2ego)

    # Transform to camera frame
    pts_h = np.concatenate([points_ego, np.ones((len(points_ego), 1))], axis=1)
    pts_cam = (T_ego2cam @ pts_h.T).T[:, :3]  # (N, 3)

    # Keep points in front of camera
    mask = pts_cam[:, 2] > min_dist
    pts_cam = pts_cam[mask]

    # Project
    uvw = (K @ pts_cam.T).T  # (M, 3)
    z = uvw[:, 2]
    u = (uvw[:, 0] / z).astype(np.int32)
    v = (uvw[:, 1] / z).astype(np.int32)

    # Filter to image bounds
    in_img = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v, z = u[in_img], v[in_img], z[in_img]

    # Build depth map (keep closest point per pixel)
    depth_map = np.zeros((img_h, img_w), dtype=np.float32)
    # Sort by descending depth so closer points overwrite farther ones
    order = np.argsort(-z)
    depth_map[v[order], u[order]] = z[order]
    return depth_map


class NuScenesOSZLoader:
    """Iterate over nuScenes samples and return OSZ inputs.

    For each sample the loader produces a dictionary containing the
    sample token and per-camera data: dense depth maps, sparse depth
    maps, RGB images, intrinsics, and camera-to-ego transforms.

    The loader requires a real nuScenes dataset; if the devkit or data
    root is unavailable it raises ``FileNotFoundError`` (no synthetic
    data fallback).

    Parameters
    ----------
    dataroot : str, optional
        Path to the nuScenes dataset root. Default is
        ``'/data/sets/nuscenes'``.
    version : str, optional
        nuScenes split version. Default is ``'v1.0-mini'``.
    cameras : list[str] | None, optional
        Camera names to load. Defaults to ``cfg.NUSCENES_CAMERAS``.
    max_samples : int | None, optional
        Maximum number of samples to iterate. ``None`` iterates the
        full dataset.
    img_h : int, optional
        Output image height. Default is 900.
    img_w : int, optional
        Output image width. Default is 1600.
    n_sweeps : int, optional
        Number of past LiDAR sweeps to aggregate before projection.
        Default is 0.

    Yields
    ------
    dict
        Frame dictionary with keys:

        - ``'sample_token'`` (str): nuScenes sample token.
        - ``'cameras'`` (dict): mapping from camera name to a dict
          containing ``'depth_map'``, ``'depth_map_sparse'``,
          ``'image'``, ``'K'``, ``'T_cam2ego'``, ``'img_h'``,
          ``'img_w'``.

    Examples
    --------
    >>> loader = NuScenesOSZLoader(dataroot='/data/nuscenes',
    ...                            version='v1.0-mini')
    >>> for frame in loader:
    ...     print(frame['sample_token'])
    """

    def __init__(
        self,
        dataroot: str = '/data/sets/nuscenes',
        version: str = 'v1.0-mini',
        cameras: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        img_h: int = 900,
        img_w: int = 1600,
        n_sweeps: int = 0,
    ):
        """Initialize the loader from a real nuScenes dataset.

        Raises
        ------
        FileNotFoundError
            If the nuScenes devkit is unavailable or ``dataroot`` does
            not exist.
        """
        self.cameras = cameras or cfg.NUSCENES_CAMERAS
        self.max_samples = max_samples
        self.img_h = img_h
        self.img_w = img_w
        self.n_sweeps = n_sweeps

        if not (NUSCENES_AVAILABLE and Path(dataroot).exists()):
            raise FileNotFoundError(
                f"nuScenes data not found at {dataroot} (or nuscenes-devkit "
                f"missing). OSZ requires real data input."
            )
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.samples = self.nusc.sample
        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        """Return the number of frames (real samples)."""
        return len(self.samples)

    def __iter__(self):
        """Yield frames from the real dataset."""
        yield from self._nuscenes_iter()

    def build_frame_for_token(self, sample_token: str) -> dict:
        """Build a single frame dict for ``sample_token`` without iterating.

        This is the token-based entry point used by
        ``run_osz_pipeline.py`` when ``--sample_token`` is supplied. It
        looks up the sample record directly so it works whether or not
        ``self.samples`` was populated during initialization.

        Parameters
        ----------
        sample_token : str
            nuScenes sample token.

        Returns
        -------
        dict
            Frame dictionary with ``'sample_token'`` and ``'cameras'``
            keys. See the class docstring for the camera dict contents.
        """
        sample = self.nusc.get('sample', sample_token)
        frame = {'sample_token': sample['token'], 'cameras': {}}

        # Load LiDAR (ego frame)
        # Aggregate current + past sweeps for denser point coverage before
        # projecting to cameras. This improves depth completion quality and
        # fallback reliability in occluded/distant regions.
        pts_ego = aggregate_lidar_sweeps(
            self.nusc, sample_token, n_sweeps=self.n_sweeps
        )

        # Per-camera projection
        for cam_name in self.cameras:
            if cam_name not in sample['data']:
                continue
            cam_token = sample['data'][cam_name]
            cam_sd    = self.nusc.get('sample_data', cam_token)
            T_cam2ego = _get_transform(self.nusc, cam_sd)
            K         = _get_intrinsic(self.nusc, cam_token)

            depth_sparse = project_lidar_to_camera(
                pts_ego, K, T_cam2ego,
                self.img_h, self.img_w,
            )
            depth_dense = densify_depth_map(depth_sparse)
            # Clip to the same max depth used for model predictions so that
            # downstream modules never see 100 m phantom walls.
            depth_dense = np.clip(depth_dense, 0.0, cfg.MAX_METRIC_DEPTH_M)

            # Load camera image
            image = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
            if PIL_AVAILABLE:
                try:
                    img_path = Path(self.nusc.dataroot) / cam_sd['filename']
                    image = np.array(Image.open(img_path).convert('RGB'))
                except Exception:
                    pass

            frame['cameras'][cam_name] = {
                'depth_map': depth_dense,        # (H, W) metres, densified
                'depth_map_sparse': depth_sparse,# (H, W) original sparse
                'image':     image,              # (H, W, 3) RGB
                'K':         K,                  # (3, 3)
                'T_cam2ego': T_cam2ego,          # (4, 4)
                'img_h':     self.img_h,
                'img_w':     self.img_w,
            }

        return frame

    def _nuscenes_iter(self):
        """Yield frames by iterating the loaded nuScenes samples."""
        for sample in self.samples:
            yield self.build_frame_for_token(sample['token'])
