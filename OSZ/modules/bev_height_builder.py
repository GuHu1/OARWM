"""Build a height-aware BEV grid from ego-frame 3D points.

This module aggregates per-camera and LiDAR depth measurements into
Bird's-Eye View (BEV) height maps.  It supports camera-primary fusion
with LiDAR fallback and optional uncertainty-weighted fusion.
"""

import numpy as np
from typing import Tuple

from OSZ import config as cfg


def build_bev_height_max(
    pts_ego: np.ndarray,
    caster,
    z_min: float = cfg.Z_MIN_M,
    z_max: float = cfg.Z_MAX_M,
) -> np.ndarray:
    """Compute the maximum height per BEV cell from ego-frame points.

    For each BEV cell, store the maximum z-coordinate (height) of any point
    falling into that cell. Empty cells remain zero.

    Parameters
    ----------
    pts_ego : np.ndarray
        (N, 3) float32 points in the ego frame, already ground-filtered.
    caster : RayCaster3D
        Instance providing ``bev_range``, ``nx``, ``ny``, ``bev_res_x`` and
        ``bev_res_y``.
    z_min : float, optional
        Minimum height to consider (extra safety filter).
    z_max : float, optional
        Maximum height to consider (building/sky filtering).

    Returns
    -------
    np.ndarray
        (nx, ny) float32 array with the maximum height per cell. Zero means
        the cell is empty.
    """
    x_min, x_max, y_min, y_max = caster.bev_range
    nx, ny = caster.nx, caster.ny
    res_x, res_y = caster.bev_res_x, caster.bev_res_y

    xi = np.floor((pts_ego[:, 0] - x_min) / res_x).astype(np.int32)
    yi = np.floor((y_max - pts_ego[:, 1]) / res_y).astype(np.int32)
    zi = pts_ego[:, 2]

    valid = ((xi >= 0) & (xi < nx) &
             (yi >= 0) & (yi < ny) &
             (zi >= z_min) & (zi <= z_max))
    xi, yi, zi = xi[valid], yi[valid], zi[valid]

    bev_height = np.zeros((nx, ny), dtype=np.float32)
    np.maximum.at(bev_height, (xi, yi), zi)
    return bev_height


def _prepare_camera_depths(
    cameras: dict,
    estimator=None,
    depth_key: str = "depth_map",
) -> dict:
    """Build camera-primary depth maps from an estimator or an existing key.

    Parameters
    ----------
    cameras : dict
        Mapping from camera name to camera data dictionary.
    estimator : object, optional
        Depth estimator used to predict depth from each camera image. If
        ``None``, the existing ``depth_key`` depth map is used directly.
    depth_key : str, optional
        Key to read from ``cameras`` when ``estimator`` is ``None``.

    Returns
    -------
    dict
        ``{cam_name: {'depth_map': (H, W), 'K': (3, 3), 'T_cam2ego': (4, 4)}}``.
    """
    if estimator is not None:
        from OSZ.modules.depth_estimator import MockDepthEstimator
        from OSZ.modules.rcsample_depth_estimator import RCSampleDepthEstimator
        use_mock = isinstance(estimator, MockDepthEstimator)
        is_rcsample = isinstance(estimator, RCSampleDepthEstimator)
        cam_depths = {}
        for cam_name, cam_data in cameras.items():
            # RCSample's camera-aware depth head conditions on intrinsics and
            # camera->ego extrinsics; pass them through when available.
            est_kwargs = {}
            if is_rcsample:
                est_kwargs = dict(
                    K=cam_data['K'],
                    T_cam2ego=cam_data['T_cam2ego'],
                )
            if use_mock:
                pred = estimator.infer(
                    cam_data['image'],
                    lidar_dense_depth=cam_data.get('depth_map'),
                    **est_kwargs,
                )
            else:
                pred = estimator.infer(
                    cam_data['image'],
                    lidar_sparse_depth=cam_data.get('depth_map_sparse'),
                    **est_kwargs,
                )
            cam_depths[cam_name] = {
                'depth_map': pred,
                'K': cam_data['K'],
                'T_cam2ego': cam_data['T_cam2ego'],
            }
    else:
        cam_depths = {
            n: {'depth_map': cameras[n][depth_key],
                'K': cameras[n]['K'],
                'T_cam2ego': cameras[n]['T_cam2ego']}
            for n in cameras
        }
    return cam_depths


def _prepare_lidar_fallback_depths(cameras: dict) -> dict:
    """Build LiDAR fallback depth maps, preferring sparse over densified."""
    return {
        cam_name: {
            'depth_map': cam_data.get('depth_map_sparse', cam_data.get('depth_map')),
            'K': cam_data['K'],
            'T_cam2ego': cam_data['T_cam2ego'],
        }
        for cam_name, cam_data in cameras.items()
    }


def build_bev_height_fused(
    cameras: dict,
    caster,
    estimator=None,
    depth_key: str = "depth_map",
    z_min: float = cfg.Z_MIN_M,
    z_max: float = cfg.Z_MAX_M,
) -> np.ndarray:
    """Fuse camera-primary and LiDAR-fallback depth into a BEV height map.

    Strategy
    --------
    - If a learned depth estimator is provided, predict depth from each
      camera image (camera-primary mode).
    - Otherwise use the existing camera ``depth_map`` (typically
      LiDAR-densified).
    - Build a LiDAR-only height map from ``depth_map_sparse``/``depth_map``
      as fallback.
    - For each BEV cell, use the camera prediction if available; otherwise
      fill with the LiDAR measurement.

    Parameters
    ----------
    cameras : dict
        ``{cam_name: {depth_map, depth_map_sparse, K, T_cam2ego, image}}``.
    caster : RayCaster3D
        Instance defining the BEV grid.
    estimator : object, optional
        Optional depth estimator for camera-primary prediction.
    depth_key : str, optional
        Camera depth key to use when ``estimator`` is ``None``.
    z_min : float, optional
        Ground filter height in metres.
    z_max : float, optional
        Maximum obstacle height in metres; discard points above this.

    Returns
    -------
    np.ndarray
        (nx, ny) float32 array with the fused maximum height per cell.
    """
    # Camera-primary height map
    cam_depths = _prepare_camera_depths(cameras, estimator=estimator, depth_key=depth_key)
    bev_height_cam = build_bev_height_from_cameras(
        cam_depths, caster, depth_key='depth_map', z_min=z_min, z_max=z_max
    )

    # LiDAR fallback height map: use raw sparse LiDAR projection so it can
    # fill cells where the learned/densified camera depth is empty.
    lidar_depths = _prepare_lidar_fallback_depths(cameras)
    bev_height_lidar = build_bev_height_from_cameras(
        lidar_depths, caster, depth_key='depth_map', z_min=z_min, z_max=z_max
    )

    # Fallback: camera where available, LiDAR where camera is empty
    bev_height_fused = bev_height_cam.copy()
    lidar_fallback = (bev_height_fused <= 0.05) & (bev_height_lidar > 0.05)
    bev_height_fused[lidar_fallback] = bev_height_lidar[lidar_fallback]

    n_fallback = int(lidar_fallback.sum())
    n_total = bev_height_fused.size
    print(
        f"[bev_height_fused] camera cells={(bev_height_cam > 0.05).sum()}, "
        f"lidar cells={(bev_height_lidar > 0.05).sum()}, "
        f"fallback cells={n_fallback} ({n_fallback / max(n_total, 1) * 100:.2f}%)"
    )

    return bev_height_fused


def build_bev_height_from_cameras(
    cameras: dict,
    caster,
    depth_key: str = "depth_map",
    z_min: float = cfg.Z_MIN_M,
    z_max: float = cfg.Z_MAX_M,
) -> np.ndarray:
    """Build a BEV height map by back-projecting multiple camera depth maps.

    Parameters
    ----------
    cameras : dict
        ``{cam_name: {'depth_map': (H, W), 'K': (3, 3), 'T_cam2ego': (4, 4)}}``.
    caster : RayCaster3D
        Instance defining the BEV grid.
    depth_key : str, optional
        Depth map key to use. ``'depth_map'`` for densified/predicted depth,
        ``'depth_map_sparse'`` for raw sparse LiDAR depth.
    z_min : float, optional
        Ground filter height in metres.
    z_max : float, optional
        Maximum obstacle height in metres; discard points above this.

    Returns
    -------
    np.ndarray
        (nx, ny) float32 array with the maximum height per BEV cell.
    """
    from OSZ.modules.image_to_ego import depth_map_to_ego_points

    bev_height = np.zeros((caster.nx, caster.ny), dtype=np.float32)

    for cam_name, cam_data in cameras.items():
        depth_map = cam_data[depth_key]
        K = cam_data['K']
        T_cam2ego = cam_data['T_cam2ego']

        pts_ego = depth_map_to_ego_points(depth_map, K, T_cam2ego, z_min=z_min, z_max=z_max)
        if len(pts_ego) == 0:
            continue

        cam_height = build_bev_height_max(pts_ego, caster, z_min=z_min, z_max=z_max)
        np.maximum(bev_height, cam_height, out=bev_height)

    return bev_height


def build_bev_height_and_uncertainty_from_cameras(
    cameras: dict,
    caster,
    depth_key: str = "depth_map",
    z_min: float = cfg.Z_MIN_M,
    z_max: float = cfg.Z_MAX_M,
    uncertainty_mode: str = "depth",
    max_depth: float = cfg.MAX_METRIC_DEPTH_M,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-cell height, average uncertainty, and point density maps.

    Parameters
    ----------
    cameras : dict
        ``{cam_name: {'depth_map': (H, W), 'K': (3, 3), 'T_cam2ego': (4, 4)}}``.
    caster : RayCaster3D
        Instance defining the BEV grid.
    depth_key : str, optional
        Depth map key to use.
    z_min : float, optional
        Ground filter height in metres.
    z_max : float, optional
        Maximum obstacle height in metres; discard points above this.
    uncertainty_mode : {'depth', 'constant'}, optional
        ``'depth'`` makes uncertainty proportional to ego xy distance
        (farther = noisier). ``'constant'`` assigns uniform uncertainty per
        point (useful for LiDAR density-only weighting).
    max_depth : float, optional
        Distance normalisation factor for ``uncertainty_mode='depth'``.

    Returns
    -------
    bev_height : np.ndarray
        (nx, ny) float32 array with the maximum height per cell.
    bev_uncertainty : np.ndarray
        (nx, ny) float32 array with the average point uncertainty (``inf``
        for empty cells).
    bev_density : np.ndarray
        (nx, ny) float32 array with the number of points in each cell.
    """
    from OSZ.modules.image_to_ego import depth_map_to_ego_points

    nx, ny = caster.nx, caster.ny
    bev_height = np.zeros((nx, ny), dtype=np.float32)
    bev_unc_sum = np.zeros((nx, ny), dtype=np.float32)
    bev_density = np.zeros((nx, ny), dtype=np.float32)

    x_min, x_max, y_min, y_max = caster.bev_range

    for cam_name, cam_data in cameras.items():
        depth_map = cam_data[depth_key]
        K = cam_data['K']
        T_cam2ego = cam_data['T_cam2ego']

        pts_ego = depth_map_to_ego_points(
            depth_map, K, T_cam2ego, z_min=z_min, z_max=z_max, max_depth=max_depth
        )
        if len(pts_ego) == 0:
            continue

        if uncertainty_mode == "depth":
            dist = np.linalg.norm(pts_ego[:, :2], axis=1)
            point_unc = dist / max_depth
        else:
            point_unc = np.ones(len(pts_ego), dtype=np.float32)

        xi = np.floor((pts_ego[:, 0] - x_min) / caster.bev_res_x).astype(np.int32)
        yi = np.floor((y_max - pts_ego[:, 1]) / caster.bev_res_y).astype(np.int32)
        zi = pts_ego[:, 2]

        valid = ((xi >= 0) & (xi < nx) &
                 (yi >= 0) & (yi < ny) &
                 (zi >= z_min) & (zi <= z_max))
        xi, yi, zi, point_unc = xi[valid], yi[valid], zi[valid], point_unc[valid]

        np.maximum.at(bev_height, (xi, yi), zi)
        np.add.at(bev_unc_sum, (xi, yi), point_unc)
        np.add.at(bev_density, (xi, yi), 1.0)

    bev_uncertainty = np.full((nx, ny), np.inf, dtype=np.float32)
    valid_cells = bev_density > 0
    bev_uncertainty[valid_cells] = bev_unc_sum[valid_cells] / bev_density[valid_cells]

    return bev_height, bev_uncertainty, bev_density


def build_bev_height_fused_uncertainty(
    cameras: dict,
    caster,
    estimator=None,
    depth_key: str = "depth_map",
    z_min: float = cfg.Z_MIN_M,
    z_max: float = cfg.Z_MAX_M,
) -> np.ndarray:
    """Uncertainty-aware camera-LiDAR fusion for ``bev_height_max``.

    Strategy
    --------
    - Camera depth predicts the primary height; its uncertainty grows with
      distance.
    - LiDAR sparse depth provides fallback and supplementary height; its
      uncertainty is inversely proportional to local point density.
    - Where both sensors have a measurement, fuse by inverse-uncertainty
      weights.
    - Where the camera measurement is empty, fall back to LiDAR (same hard
      fallback as before).

    Parameters
    ----------
    cameras : dict
        ``{cam_name: {depth_map, depth_map_sparse, K, T_cam2ego, image}}``.
    caster : RayCaster3D
        Instance defining the BEV grid.
    estimator : object, optional
        Optional depth estimator for camera-primary prediction.
    depth_key : str, optional
        Camera depth key to use when ``estimator`` is ``None``.
    z_min : float, optional
        Ground filter height in metres.
    z_max : float, optional
        Maximum obstacle height in metres; discard points above this.

    Returns
    -------
    np.ndarray
        (nx, ny) float32 array with the fused height per cell.
    """
    # Camera-primary depth maps
    cam_depths = _prepare_camera_depths(cameras, estimator=estimator, depth_key=depth_key)
    h_cam, unc_cam, density_cam = build_bev_height_and_uncertainty_from_cameras(
        cam_depths, caster, depth_key='depth_map', z_min=z_min, z_max=z_max, uncertainty_mode='depth'
    )

    # LiDAR sparse fallback
    lidar_depths = _prepare_lidar_fallback_depths(cameras)
    h_lidar, _, density_lidar = build_bev_height_and_uncertainty_from_cameras(
        lidar_depths, caster, depth_key='depth_map', z_min=z_min, z_max=z_max, uncertainty_mode='constant'
    )

    # LiDAR uncertainty: higher local density -> more trustworthy.
    unc_lidar = 1.0 / (density_lidar + 1.0)

    eps = 1e-6
    w_cam = 1.0 / (unc_cam + eps)
    w_lidar = 1.0 / (unc_lidar + eps)

    cam_valid = h_cam > 0.05
    lidar_valid = h_lidar > 0.05
    both_valid = cam_valid & lidar_valid

    bev_height_fused = np.zeros_like(h_cam)

    # Only camera
    bev_height_fused[cam_valid & ~lidar_valid] = h_cam[cam_valid & ~lidar_valid]

    # Only LiDAR (hard fallback)
    bev_height_fused[~cam_valid & lidar_valid] = h_lidar[~cam_valid & lidar_valid]

    # Both: inverse-uncertainty weighted fusion
    denom = w_cam[both_valid] + w_lidar[both_valid]
    bev_height_fused[both_valid] = (
        w_cam[both_valid] * h_cam[both_valid] +
        w_lidar[both_valid] * h_lidar[both_valid]
    ) / denom

    n_fallback = int((~cam_valid & lidar_valid).sum())
    n_weighted = int(both_valid.sum())
    n_total = bev_height_fused.size
    print(
        f"[bev_height_fused_uncertainty] camera cells={cam_valid.sum()}, "
        f"lidar cells={lidar_valid.sum()}, both={n_weighted}, "
        f"fallback cells={n_fallback} ({n_fallback / max(n_total, 1) * 100:.2f}%)"
    )

    return bev_height_fused
