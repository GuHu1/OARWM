"""Voxel casting and height-aware ray casting for OSZ computation.

Pipeline
--------
1. For each camera's depth map, project voxels in the ``(z_min, z_max)``
   height gate onto the image plane and compare voxel depth to the measured
   depth. Voxels within ``surface_tolerance`` of the measured depth are real
   occluder surfaces, not shadows.
2. Max-pool along Z and union across cameras -> ``bev_occ`` (nx, ny) bool.
3. 2D ray casting over ``bev_occ`` -> ``osz_mask`` (nx, ny) bool: every cell
   behind the first occluder along its ray from ego.

Notes
-----
Density-binning the LiDAR point cloud directly into a BEV grid produces
range-dependent gaps in distant occluders. A gap causes the 2D ray caster to
"see through" the wall, so everything beyond becomes shadow. The voxel cast
fills those gaps by querying every voxel against the measured depth at its
projected pixel, regardless of range.

Coordinate convention
---------------------
- axis-0 = ego-x (forward), axis-1 = ego-y (left).
- ``bev_occ[i, j]`` where ``i`` is the ego-x index and ``j`` is the ego-y
  index.
"""
from __future__ import annotations

import numpy as np

from OSZ import config as cfg
from OSZ.utils.geometry import bev_grid_shape


def build_bev_occ_from_voxel_cast(
    cameras: dict[str, dict],
    caster: "RayCaster3D",
) -> np.ndarray:
    """Build a solid BEV obstacle map by per-camera 3D voxel casting.

    Each camera returns occluder-surface voxels. Max-pooling along the height
    axis and unioning across cameras gives ``bev_occ`` -- the true obstacles.
    Occlusion shadow is computed separately by ``cast_osz_2d`` over this
    solid map.

    Parameters
    ----------
    cameras : dict[str, dict]
        Mapping from camera name to camera data dictionary.
    caster : RayCaster3D
        Instance used to cast voxels.

    Returns
    -------
    np.ndarray
        (nx, ny) bool union of all cameras' occluder surfaces.
    """
    nx, ny = caster.nx, caster.ny
    bev_occ = np.zeros((nx, ny), dtype=bool)

    for cam_name, cam_data in cameras.items():
        V_occ = caster.cast(
            depth_map=cam_data["depth_map"],
            intrinsic=cam_data["K"],
            cam2ego=cam_data["T_cam2ego"],
        )
        M_occ = voxel_to_bev_maxpool(V_occ)
        bev_occ |= M_occ

    return bev_occ


def cast_osz_2d(
    bev_occ: np.ndarray,
    caster: "RayCaster3D",
    substep: float = 0.25,
) -> np.ndarray:
    """Ego-centric 360 degree 2D ray casting over a solid ``bev_occ`` grid.

    Vectorized implementation: all rays advance simultaneously per step using
    NumPy. The outer loop runs ``max_steps`` times instead of
    ``n_angles * max_steps``.

    Parameters
    ----------
    bev_occ : np.ndarray
        (nx, ny) bool solid obstacle map (no point-density gaps).
    caster : RayCaster3D
        Provides ``nx``, ``ny``, ``bev_range``, and ``bev_res``.
    substep : float, optional
        Ray step size in BEV cells (not metres). ``0.25`` means the ray
        advances a quarter-cell per iteration -- small enough to not skip
        a single occupied cell.

    Returns
    -------
    np.ndarray
        (nx, ny) bool mask of cells behind the first occluder per ray.
    """
    nx, ny = caster.nx, caster.ny
    x_min, x_max, y_min, y_max = caster.bev_range

    ego_xi = int(np.floor((0.0 - x_min) / caster.bev_res_x))
    ego_yi = int(np.floor((y_max - 0.0) / caster.bev_res_y))

    osz_mask = np.zeros((nx, ny), dtype=bool)
    if not (0 <= ego_xi < nx and 0 <= ego_yi < ny):
        return osz_mask

    max_range_cells = max(nx, ny)
    n_angles = int(2 * np.pi * max_range_cells / substep)
    n_angles = max(n_angles, 720)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    dx = np.cos(angles) * substep
    dy = np.sin(angles) * substep

    x = np.full(n_angles, float(ego_xi))
    y = np.full(n_angles, float(ego_yi))

    hit = np.zeros(n_angles, dtype=bool)
    active = np.ones(n_angles, dtype=bool)
    max_steps = int(max_range_cells / substep)

    for _ in range(max_steps):
        x[active] += dx[active]
        y[active] += dy[active]

        xi = np.rint(x).astype(np.int32)
        yi = np.rint(y).astype(np.int32)

        in_b = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        active &= in_b
        if not active.any():
            break

        idx = np.where(active)[0]
        xi_a = xi[idx]
        yi_a = yi[idx]
        occ = bev_occ[xi_a, yi_a]

        # Rays that hit an occluder in a PREVIOUS step mark the current cell
        # as OSZ. The occluder cell itself is NOT marked.
        prev_hit = hit[idx]
        osz_mask[xi_a[prev_hit], yi_a[prev_hit]] = True

        hit[idx] |= occ

    return osz_mask


def compute_osz_from_ego_raycasting(
    cameras: dict[str, dict],
    caster: "RayCaster3D",
) -> tuple[np.ndarray, np.ndarray]:
    """Full OSZ pipeline: 3D voxel cast -> solid BEV occupancy -> 2D ray cast.

    Parameters
    ----------
    cameras : dict[str, dict]
        Mapping from camera name to camera data dictionary.
    caster : RayCaster3D
        Instance defining the BEV grid and voxel casting.

    Returns
    -------
    osz_mask : np.ndarray
        (nx, ny) bool ego-centric occlusion shadow zone.
    bev_occ : np.ndarray
        (nx, ny) bool solid BEV obstacle map.
    """
    bev_occ = build_bev_occ_from_voxel_cast(cameras, caster)
    osz_mask = cast_osz_2d(bev_occ, caster)
    return osz_mask, bev_occ


class RayCaster3D:
    """Vectorized 3D ray casting for a single camera.

    All coordinates are in the ego-vehicle world frame (nuScenes convention).

    Parameters
    ----------
    bev_range : tuple, optional
        ``(x_min, x_max, y_min, y_max)`` in metres, ego-centred.
    bev_res_x : float, optional
        BEV grid resolution along ego-x (metres per cell). Defaults to
        ``cfg.BEV_X_RES`` (aligned with ResWorld ``grid_config``).
    bev_res_y : float, optional
        BEV grid resolution along ego-y (metres per cell). Defaults to
        ``cfg.BEV_Y_RES``.
    z_min : float, optional
        Lower bound of the vehicle-body voxel height gate (metres).
    z_max : float, optional
        Upper bound of the vehicle-body voxel height gate (metres).
    z_res : float, optional
        Voxel height resolution (metres). Defaults to ``cfg.Z_RES_M``.
    depth_scale : float, optional
        Multiplier to convert raw depth map values to metres.
    """

    def __init__(
        self,
        bev_range: tuple[float, float, float, float] = cfg.BEV_RANGE_M,
        bev_res_x: float | None = None,
        bev_res_y: float | None = None,
        z_min: float = cfg.Z_MIN_M,
        z_max: float = cfg.Z_MAX_M,
        z_res: float = cfg.Z_RES_M,
        depth_scale: float = 1.0,
    ):
        # Anisotropic grid (aligned with ResWorld grid_config):
        # 0.15 m/cell along ego-x, 0.3 m/cell along ego-y.
        self.bev_range = bev_range
        self.bev_res_x = cfg.BEV_X_RES if bev_res_x is None else bev_res_x
        self.bev_res_y = cfg.BEV_Y_RES if bev_res_y is None else bev_res_y
        self.z_min = z_min
        self.z_max = z_max
        self.z_res = z_res
        self.depth_scale = depth_scale

        self.nx, self.ny = bev_grid_shape(bev_range)
        self.nz = int(round((z_max - z_min) / z_res))

        xs = np.linspace(bev_range[0] + self.bev_res_x / 2,
                         bev_range[1] - self.bev_res_x / 2, self.nx)
        ys = np.linspace(bev_range[3] - self.bev_res_y / 2,
                         bev_range[2] + self.bev_res_y / 2, self.ny)
        zs = np.linspace(z_min + z_res / 2, z_max - z_res / 2, self.nz)

        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        self.voxel_centers = np.stack(
            [xx.ravel(), yy.ravel(), zz.ravel()], axis=-1
        ).astype(np.float32)

    def cast(
        self,
        depth_map: np.ndarray,
        intrinsic: np.ndarray,
        cam2ego: np.ndarray,
        max_depth: float = cfg.MAX_METRIC_DEPTH_M,
        surface_tolerance: float | None = None,
    ) -> np.ndarray:
        """Return a (nx, ny, nz) bool array marking occluder surfaces.

        A voxel is occupied only if the measured depth at its projected pixel
        is approximately equal to the voxel's distance from the camera (within
        ``surface_tolerance``). This identifies real physical surfaces, not
        shadows.

        Parameters
        ----------
        depth_map : np.ndarray
            (H, W) metric depth map.
        intrinsic : np.ndarray
            (3, 3) camera intrinsic matrix.
        cam2ego : np.ndarray
            (4, 4) camera -> ego extrinsic transform.
        max_depth : float, optional
            Maximum depth in metres to consider valid.
        surface_tolerance : float, optional
            Voxels within this distance (metres) of the measured depth are
            considered on the surface. Defaults to three times the larger of
            ``bev_res`` and ``z_res``.

        Returns
        -------
        np.ndarray
            (nx, ny, nz) bool array of occluder surface voxels.
        """
        if surface_tolerance is None:
            surface_tolerance = max(
                self.bev_res_x, self.bev_res_y, self.z_res
            ) * 3.0

        H, W = depth_map.shape
        K = intrinsic
        T_c2e = cam2ego
        T_e2c = np.linalg.inv(T_c2e)

        n = self.voxel_centers.shape[0]
        pts_ego_h = np.concatenate(
            [self.voxel_centers, np.ones((n, 1), dtype=np.float32)], axis=1
        )
        pts_cam_h = (T_e2c @ pts_ego_h.T).T
        pts_cam = pts_cam_h[:, :3]

        valid_front = pts_cam[:, 2] > 0.1
        pts_cam_v = pts_cam[valid_front]

        uvw = (K @ pts_cam_v.T).T
        z_cam = uvw[:, 2]
        u = np.rint(uvw[:, 0] / z_cam).astype(np.int32)
        v = np.rint(uvw[:, 1] / z_cam).astype(np.int32)

        in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx_valid = np.where(valid_front)[0][in_image]
        u_valid = u[in_image]
        v_valid = v[in_image]
        z_valid = z_cam[in_image]

        d_obs = depth_map[v_valid, u_valid].astype(np.float32)
        valid_depth = (d_obs > 0) & (d_obs < max_depth)
        on_surface = np.abs(z_valid - d_obs) <= surface_tolerance
        is_occluder = valid_depth & on_surface

        V_occ = np.zeros(self.nx * self.ny * self.nz, dtype=bool)
        V_occ[idx_valid[is_occluder]] = True
        return V_occ.reshape(self.nx, self.ny, self.nz)


def voxel_to_bev_maxpool(V_occ: np.ndarray) -> np.ndarray:
    """Max-pool a voxel occupancy array along the Z axis.

    Parameters
    ----------
    V_occ : np.ndarray
        (nx, ny, nz) bool voxel occupancy array.

    Returns
    -------
    np.ndarray
        (nx, ny) bool BEV occupancy map.
    """
    return V_occ.any(axis=2)


def build_bev_occ_from_pointcloud(
    pts_ego: np.ndarray,
    caster: "RayCaster3D",
    min_points: int = 1,
    do_closing: bool = True,
) -> np.ndarray:
    """Build a solid BEV obstacle map by direct 3D voxelization of a point cloud.

    Each LiDAR point is assigned to a 3D voxel. A voxel with at least
    ``min_points`` points is marked occupied. Z-axis max-pool gives BEV
    occupancy.

    Parameters
    ----------
    pts_ego : np.ndarray
        (N, 3) float32 points in the current ego frame, ground-filtered.
    caster : RayCaster3D
        Provides ``nx``, ``ny``, ``nz``, ``bev_range``, ``z_min``, etc.
    min_points : int, optional
        Minimum points per voxel to count as occupied.
    do_closing : bool, optional
        If ``True``, apply binary closing to fill small holes.

    Returns
    -------
    np.ndarray
        (nx, ny) bool solid BEV obstacle map.
    """
    nx, ny, nz = caster.nx, caster.ny, caster.nz
    x_min, x_max, y_min, y_max = caster.bev_range

    xi = np.floor((pts_ego[:, 0] - x_min) / caster.bev_res_x).astype(np.int32)
    yi = np.floor((y_max - pts_ego[:, 1]) / caster.bev_res_y).astype(np.int32)
    zi = np.floor((pts_ego[:, 2] - caster.z_min) / caster.z_res).astype(np.int32)

    in_grid = (
        (xi >= 0) & (xi < nx) &
        (yi >= 0) & (yi < ny) &
        (zi >= 0) & (zi < nz)
    )
    xi, yi, zi = xi[in_grid], yi[in_grid], zi[in_grid]

    flat_idx = xi * (ny * nz) + yi * nz + zi
    counts = np.bincount(flat_idx, minlength=nx * ny * nz)
    V_occ = counts.reshape(nx, ny, nz) >= min_points

    bev_occ = V_occ.any(axis=2)

    if do_closing:
        try:
            from scipy.ndimage import binary_closing
            bev_occ = binary_closing(bev_occ, iterations=2)
        except ImportError:
            pass

    return bev_occ


def compute_osz_from_pointcloud(
    pts_ego: np.ndarray,
    caster: "RayCaster3D",
) -> tuple[np.ndarray, np.ndarray]:
    """Full OSZ from a point cloud: voxelize -> BEV occ -> 2D ray cast.

    Parameters
    ----------
    pts_ego : np.ndarray
        (N, 3) float32 points in the current ego frame.
    caster : RayCaster3D
        Instance defining the BEV grid.

    Returns
    -------
    osz_mask : np.ndarray
        (nx, ny) bool ego-centric occlusion shadow zone.
    bev_occ : np.ndarray
        (nx, ny) bool solid BEV obstacle map.
    """
    bev_occ = build_bev_occ_from_pointcloud(pts_ego, caster)
    osz_mask = cast_osz_2d(bev_occ, caster)
    return osz_mask, bev_occ


def cast_osz_height_aware(
    bev_height: np.ndarray,
    caster: "RayCaster3D",
    observer_height: float = cfg.OBSERVER_HEIGHT_M,
    substep: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Ego-centric 360 degree ray casting over a BEV height map.

    Two shadow masks are produced:

    - ``osz_ground``: any occupied cell blocks the ray (binary behaviour).
    - ``osz_eye``: only cells whose height exceeds ``observer_height`` block.

    The difference ``osz_ground & ~osz_eye`` is the semi-transparent zone:
    ground is occluded but upper volume is visible.

    Parameters
    ----------
    bev_height : np.ndarray
        (nx, ny) float32 maximum height per BEV cell (0 = empty).
    caster : RayCaster3D
        Provides ``nx``, ``ny``, ``bev_range``, and ``bev_res``.
    observer_height : float, optional
        Height of the observer's eye in metres.
    substep : float, optional
        Ray step size in BEV cells.

    Returns
    -------
    osz_ground : np.ndarray
        (nx, ny) bool ground-layer fully occluded cells.
    osz_eye : np.ndarray
        (nx, ny) bool eye-height occluded cells.
    """
    nx, ny = caster.nx, caster.ny
    x_min, x_max, y_min, y_max = caster.bev_range

    ego_xi = int(np.floor((0.0 - x_min) / caster.bev_res_x))
    ego_yi = int(np.floor((y_max - 0.0) / caster.bev_res_y))

    osz_ground = np.zeros((nx, ny), dtype=bool)
    osz_eye = np.zeros((nx, ny), dtype=bool)

    if not (0 <= ego_xi < nx and 0 <= ego_yi < ny):
        return osz_ground, osz_eye

    # Clear a small radius around ego to prevent self-occlusion.
    # The grid is anisotropic, so the distance must be computed in metres.
    bev_height = bev_height.copy()
    radius_m = cfg.EGO_CLEARANCE_RADIUS_M
    xg, yg = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    dist_m = np.sqrt(
        ((xg - ego_xi) * caster.bev_res_x) ** 2
        + ((yg - ego_yi) * caster.bev_res_y) ** 2
    )
    bev_height[dist_m < radius_m] = 0.0

    max_range_cells = max(nx, ny)
    n_angles = int(2 * np.pi * max_range_cells / substep)
    n_angles = max(n_angles, 720)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    dx = np.cos(angles) * substep
    dy = np.sin(angles) * substep
    x = np.full(n_angles, float(ego_xi))
    y = np.full(n_angles, float(ego_yi))

    hit_ground = np.zeros(n_angles, dtype=bool)
    hit_eye = np.zeros(n_angles, dtype=bool)
    active = np.ones(n_angles, dtype=bool)
    max_steps = int(max_range_cells / substep)

    for _ in range(max_steps):
        x[active] += dx[active]
        y[active] += dy[active]
        xi = np.rint(x).astype(np.int32)
        yi = np.rint(y).astype(np.int32)

        in_b = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        active &= in_b
        if not active.any():
            break

        idx = np.where(active)[0]
        xi_a, yi_a = xi[idx], yi[idx]
        h = bev_height[xi_a, yi_a]

        prev_g = hit_ground[idx]
        osz_ground[xi_a[prev_g], yi_a[prev_g]] = True
        hit_ground[idx] |= (h > 0.05)

        prev_e = hit_eye[idx]
        osz_eye[xi_a[prev_e], yi_a[prev_e]] = True
        hit_eye[idx] |= (h > observer_height)

    return osz_ground, osz_eye


def compute_osz_height_aware_from_cameras(
    cameras: dict[str, dict],
    caster: "RayCaster3D",
    estimator=None,
    observer_height: float = cfg.OBSERVER_HEIGHT_M,
    depth_key: str = "depth_map",
    use_uncertainty: bool = False,
    z_min: float = cfg.Z_MIN_M,
    z_max: float = cfg.Z_MAX_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """End-to-end height-aware OSZ from camera depth maps with LiDAR fallback.

    Pipeline
    --------
    1. Build fused ``bev_height_max`` (camera primary, LiDAR fallback).
    2. Height-aware ray cast -> ``osz_ground``, ``osz_eye``.

    Parameters
    ----------
    cameras : dict[str, dict]
        ``{cam_name: {depth_map, K, T_cam2ego, image, ...}}``.
    caster : RayCaster3D
        Instance defining the BEV grid.
    estimator : object, optional
        Optional depth estimator; predicts depth from each camera image when
        provided. Otherwise ``depth_key`` is used.
    observer_height : float, optional
        Eye height in metres.
    depth_key : str, optional
        Which depth map to use when ``estimator`` is ``None``.
    use_uncertainty : bool, optional
        If ``True``, use inverse-uncertainty weighted fusion between camera
        and LiDAR; otherwise use hard fallback.
    z_min : float, optional
        Ground filter height in metres.
    z_max : float, optional
        Maximum obstacle height in metres; discard points above this.

    Returns
    -------
    bev_height : np.ndarray
        (nx, ny) float32 maximum height per cell.
    osz_ground : np.ndarray
        (nx, ny) bool ground-layer shadow.
    osz_eye : np.ndarray
        (nx, ny) bool eye-height shadow.
    """
    from OSZ.modules.bev_height_builder import (
        build_bev_height_fused,
        build_bev_height_fused_uncertainty,
    )

    if use_uncertainty:
        bev_height = build_bev_height_fused_uncertainty(
            cameras, caster, estimator=estimator, depth_key=depth_key,
            z_min=z_min, z_max=z_max,
        )
    else:
        bev_height = build_bev_height_fused(
            cameras, caster, estimator=estimator, depth_key=depth_key,
            z_min=z_min, z_max=z_max,
        )

    bev_height = np.clip(bev_height, 0.0, z_max)
    osz_ground, osz_eye = cast_osz_height_aware(bev_height, caster, observer_height)
    return bev_height, osz_ground, osz_eye
