"""GPU (torch) implementation of the OSZ geometry pipeline.

Mirrors the numpy pipeline (``image_to_ego`` / ``bev_height_builder`` /
``ray_casting``) with the same public entry point so
``export_osz_dataset.py`` can switch backends without touching the
depth-source logic::

    compute_osz_height_aware_from_cameras_torch(
        cameras, caster, estimator=None, observer_height=...,
        depth_key=..., use_uncertainty=False, z_min=..., z_max=...,
        device=None,
    ) -> (bev_height, osz_ground, osz_eye)   # numpy arrays

Correctness is guarded by ``tests/test_torch_osz.py``, which runs both
backends on identical synthetic input and compares the outputs.

Notes
-----
- ``use_uncertainty=True`` is supported on the torch path: inverse-
  uncertainty fusion is implemented (camera uncertainty ~ distance,
  LiDAR ~ 1/density; ``scatter_add_`` aggregation), matching the numpy
  backend cell-by-cell (see ``depth_maps_to_bev_height_uncertainty_torch``).
- Depth sources go through an optional ``infer_tensor`` on the estimator
  (RCSample / MiDaS) so depth prediction stays on the device; the generic
  fallback converts the numpy ``infer`` result to a tensor.
- The ray caster mirrors the numpy float64 stepping exactly (angles,
  substep, max_steps, ego clearance radius) so masks agree cell-by-cell.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch optional for numpy backend
    torch = None
    F = None

from OSZ import config as cfg


def _require_torch():
    if torch is None:
        raise ImportError(
            "torch is required for the OSZ GPU backend. Install it or use "
            "--backend numpy."
        )


def resolve_device(device: Optional[str] = None) -> str:
    """Pick a usable device: explicit arg > cuda > cpu."""
    _require_torch()
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class _Grid:
    """Precomputed BEV grid tensors, mirroring RayCaster3D geometry."""

    def __init__(self, caster, device: str):
        x_min, x_max, y_min, y_max = caster.bev_range
        nx, ny = caster.nx, caster.ny
        self.nx, self.ny = nx, ny
        self.res_x, self.res_y = caster.bev_res_x, caster.bev_res_y
        self.x_min, self.y_max = x_min, y_max
        self.ego_xi = int(np.floor((0.0 - x_min) / caster.bev_res_x))
        self.ego_yi = int(np.floor((y_max - 0.0) / caster.bev_res_y))
        # (nx*ny) flat index base for scatter_reduce.
        self.stride = ny
        self.device = device


def depth_maps_to_bev_height_torch(
    depth: "torch.Tensor",       # (N, H, W) float32
    K: "torch.Tensor",           # (N, 3, 3) float32
    T_cam2ego: "torch.Tensor",   # (N, 4, 4) float32
    grid: _Grid,
    z_min: float = cfg.Z_MIN_M,
    z_max: float = cfg.Z_MAX_M,
    max_depth: float = cfg.MAX_METRIC_DEPTH_M,
) -> "torch.Tensor":
    """Back-project N depth maps and max-pool heights into a (nx, ny) map."""
    _require_torch()
    N, H, W = depth.shape
    dev = depth.device

    u = torch.arange(W, dtype=torch.float32, device=dev).view(1, 1, W)
    v = torch.arange(H, dtype=torch.float32, device=dev).view(1, H, 1)

    fx = K[:, 0, 0].view(N, 1, 1)
    fy = K[:, 1, 1].view(N, 1, 1)
    cx = K[:, 0, 2].view(N, 1, 1)
    cy = K[:, 1, 2].view(N, 1, 1)

    d = depth  # (N, H, W)
    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d

    # (N, H, W, 3) -> homogeneous (N, H, W, 4, 1)
    pts_cam = torch.stack([x_cam, y_cam, z_cam,
                           torch.ones_like(z_cam)], dim=-1).unsqueeze(-1)
    # T: (N, 1, 1, 4, 4) @ (N, H, W, 4, 1) -> (N, H, W, 4, 1)
    pts_ego = (T_cam2ego.reshape(N, 1, 1, 4, 4) @ pts_cam)[..., 0]  # (N, H, W, 4)

    valid = (d > 0.0) & (d < max_depth)
    z_ok = (pts_ego[..., 2] >= z_min) & (pts_ego[..., 2] <= z_max)
    valid = valid & z_ok

    xi = torch.floor((pts_ego[..., 0] - grid.x_min) / grid.res_x).long()
    yi = torch.floor((grid.y_max - pts_ego[..., 1]) / grid.res_y).long()
    in_grid = (
        (xi >= 0) & (xi < grid.nx) & (yi >= 0) & (yi < grid.ny)
    )
    valid = valid & in_grid

    idx = (xi[valid] * grid.stride + yi[valid])
    zvals = pts_ego[..., 2][valid]

    bev = torch.zeros(grid.nx * grid.ny, dtype=torch.float32, device=dev)
    if idx.numel() > 0:
        # torch 1.9 has no scatter_reduce(amax) (it is torch>=1.12).
        # Segment-max via a float64 key: key = z + idx*BIG keeps the largest
        # z of each cell group under cummax (BIG >> max height), and the
        # group max is read at each group's last sorted position.
        order = torch.argsort(idx)
        idx_s = idx[order]
        z_s = zvals[order]
        big = 1.0e4
        key = z_s.to(torch.float64) + idx_s.to(torch.float64) * big
        cm = torch.cummax(key, dim=0).values
        is_last = torch.ones_like(idx_s, dtype=torch.bool)
        is_last[:-1] = idx_s[:-1] != idx_s[1:]
        group_max = (
            cm[is_last] - idx_s[is_last].to(torch.float64) * big
        ).to(torch.float32)
        bev.scatter_(0, idx_s[is_last], group_max)
    return bev.view(grid.nx, grid.ny)


def depth_maps_to_bev_height_uncertainty_torch(
    depth: "torch.Tensor",       # (N, H, W) float32
    K: "torch.Tensor",           # (N, 3, 3) float32
    T_cam2ego: "torch.Tensor",   # (N, 4, 4) float32
    grid: _Grid,
    z_min: float = cfg.Z_MIN_M,
    z_max: float = cfg.Z_MAX_M,
    max_depth: float = cfg.MAX_METRIC_DEPTH_M,
    uncertainty_mode: str = "depth",
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Back-project N depth maps -> (bev_height, bev_unc, bev_density).

    Mirrors ``bev_height_builder.build_bev_height_and_uncertainty_from_cameras``:
    per-cell max height, mean point uncertainty, and point count. Empty
    cells keep ``bev_unc = inf``. ``uncertainty_mode``: ``'depth'`` grows
    uncertainty with ego-xy distance; ``'constant'`` uses uniform weight
    (LiDAR density-only).
    """
    _require_torch()
    N, H, W = depth.shape
    dev = depth.device

    u = torch.arange(W, dtype=torch.float32, device=dev).view(1, 1, W)
    v = torch.arange(H, dtype=torch.float32, device=dev).view(1, H, 1)
    fx = K[:, 0, 0].view(N, 1, 1)
    fy = K[:, 1, 1].view(N, 1, 1)
    cx = K[:, 0, 2].view(N, 1, 1)
    cy = K[:, 1, 2].view(N, 1, 1)

    d = depth
    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d
    pts_cam = torch.stack([x_cam, y_cam, z_cam,
                           torch.ones_like(z_cam)], dim=-1).unsqueeze(-1)
    pts_ego = (T_cam2ego.reshape(N, 1, 1, 4, 4) @ pts_cam)[..., 0]  # (N,H,W,4)

    valid = (d > 0.0) & (d < max_depth)
    z_ok = (pts_ego[..., 2] >= z_min) & (pts_ego[..., 2] <= z_max)
    valid = valid & z_ok

    xi = torch.floor((pts_ego[..., 0] - grid.x_min) / grid.res_x).long()
    yi = torch.floor((grid.y_max - pts_ego[..., 1]) / grid.res_y).long()
    in_grid = ((xi >= 0) & (xi < grid.nx) & (yi >= 0) & (yi < grid.ny))
    valid = valid & in_grid

    idx = (xi[valid] * grid.stride + yi[valid])
    zvals = pts_ego[..., 2][valid]

    if uncertainty_mode == "depth":
        dist = torch.norm(pts_ego[..., :2], dim=-1)  # (N, H, W)
        unc_vals = (dist / max_depth)[valid]
    elif uncertainty_mode == "constant":
        unc_vals = torch.ones_like(zvals)
    else:
        raise ValueError(f"unknown uncertainty_mode: {uncertainty_mode!r}")

    bev = torch.zeros(grid.nx * grid.ny, dtype=torch.float32, device=dev)
    unc_sum = torch.zeros(grid.nx * grid.ny, dtype=torch.float32, device=dev)
    density = torch.zeros(grid.nx * grid.ny, dtype=torch.float32, device=dev)
    if idx.numel() > 0:
        order = torch.argsort(idx)
        idx_s = idx[order]
        z_s = zvals[order]
        big = 1.0e4
        key = z_s.to(torch.float64) + idx_s.to(torch.float64) * big
        cm = torch.cummax(key, dim=0).values
        is_last = torch.ones_like(idx_s, dtype=torch.bool)
        is_last[:-1] = idx_s[:-1] != idx_s[1:]
        group_max = (
            cm[is_last] - idx_s[is_last].to(torch.float64) * big
        ).to(torch.float32)
        bev.scatter_(0, idx_s[is_last], group_max)
        unc_sum.scatter_add_(0, idx, unc_vals)
        density.scatter_add_(0, idx, torch.ones_like(unc_vals))

    bev_unc = torch.full_like(bev, float("inf"))
    nonempty = density > 0
    bev_unc[nonempty] = unc_sum[nonempty] / density[nonempty]
    return bev.view(grid.nx, grid.ny), bev_unc.view(grid.nx, grid.ny), \
        density.view(grid.nx, grid.ny)


def cast_osz_height_aware_torch(
    bev_height: "torch.Tensor",  # (nx, ny) float32
    grid: _Grid,
    observer_height: float = cfg.OBSERVER_HEIGHT_M,
    substep: float = 0.25,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Height-aware 360-degree ray casting (mirrors the numpy version)."""
    _require_torch()
    nx, ny = grid.nx, grid.ny
    dev = bev_height.device

    osz_ground = torch.zeros((nx, ny), dtype=torch.bool, device=dev)
    osz_eye = torch.zeros((nx, ny), dtype=torch.bool, device=dev)
    if not (0 <= grid.ego_xi < nx and 0 <= grid.ego_yi < ny):
        return osz_ground, osz_eye

    # Clear a small radius around ego to prevent self-occlusion (in metres).
    height = bev_height.clone()
    radius_m = cfg.EGO_CLEARANCE_RADIUS_M
    xg = torch.arange(nx, dtype=torch.float32, device=dev).view(nx, 1)
    yg = torch.arange(ny, dtype=torch.float32, device=dev).view(1, ny)
    dist_m = torch.sqrt(
        ((xg - grid.ego_xi) * grid.res_x) ** 2
        + ((yg - grid.ego_yi) * grid.res_y) ** 2
    )
    height[dist_m < radius_m] = 0.0

    max_range_cells = max(nx, ny)
    n_angles = int(2 * np.pi * max_range_cells / substep)
    n_angles = max(n_angles, 720)
    # float64 stepping exactly like the numpy version (np.linspace endpoint=False
    # == arange(n)/n*2pi).
    angles = torch.arange(n_angles, dtype=torch.float64, device=dev) \
        * (2.0 * np.pi / n_angles)
    dx = torch.cos(angles) * substep
    dy = torch.sin(angles) * substep

    x = torch.full((n_angles,), float(grid.ego_xi), dtype=torch.float64,
                   device=dev)
    y = torch.full((n_angles,), float(grid.ego_yi), dtype=torch.float64,
                   device=dev)
    hit_ground = torch.zeros(n_angles, dtype=torch.bool, device=dev)
    hit_eye = torch.zeros(n_angles, dtype=torch.bool, device=dev)
    active = torch.ones(n_angles, dtype=torch.bool, device=dev)
    max_steps = int(max_range_cells / substep)

    for _ in range(max_steps):
        x[active] += dx[active]
        y[active] += dy[active]
        xi = torch.round(x).long()
        yi = torch.round(y).long()

        in_b = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        active &= in_b
        if not active.any():
            break

        idx = active.nonzero(as_tuple=False).flatten()
        xi_a, yi_a = xi[idx], yi[idx]
        h = height[xi_a, yi_a]

        prev_g = hit_ground[idx]
        osz_ground[xi_a[prev_g], yi_a[prev_g]] = True
        hit_ground[idx] |= (h > 0.05)

        prev_e = hit_eye[idx]
        osz_eye[xi_a[prev_e], yi_a[prev_e]] = True
        hit_eye[idx] |= (h > observer_height)

    return osz_ground, osz_eye


def _camera_tensors(cameras: dict, device: str):
    """Stack per-camera depth maps, intrinsics and extrinsics into tensors."""
    _require_torch()
    names = list(cameras.keys())
    depth = np.stack([cameras[n]["depth_map"] for n in names])
    K = np.stack([cameras[n]["K"] for n in names])
    T = np.stack([cameras[n]["T_cam2ego"] for n in names])
    return (
        torch.from_numpy(depth).to(device),
        torch.from_numpy(K).to(device),
        torch.from_numpy(T).to(device),
    )


def _estimate_depths_torch(cameras: dict, estimator, depth_key: str,
                           device: str):
    """Produce camera-primary depth tensors from an estimator (or precomputed).

    Returns
    -------
    (depth_t, K_t, T_t) where depth_t is (N, H, W) on ``device``.
    """
    _require_torch()
    names = list(cameras.keys())
    K_t = torch.from_numpy(
        np.stack([cameras[n]["K"] for n in names])
    ).to(device)
    T_t = torch.from_numpy(
        np.stack([cameras[n]["T_cam2ego"] for n in names])
    ).to(device)

    if estimator is None:
        depth = np.stack([cameras[n][depth_key] for n in names])
        return torch.from_numpy(depth).to(device), K_t, T_t

    from OSZ.modules.depth_estimator import MockDepthEstimator

    if isinstance(estimator, MockDepthEstimator):
        # LiDAR-densified depth is already metric; no network involved.
        depth = np.stack(
            [cameras[n][depth_key] for n in names]
        )
        return torch.from_numpy(depth).to(device), K_t, T_t

    if hasattr(estimator, "infer_tensor"):
        depth = torch.stack([
            estimator.infer_tensor(
                cameras[n]["image"],
                lidar_sparse_depth=cameras[n].get("depth_map_sparse"),
                K=cameras[n]["K"],
                T_cam2ego=cameras[n]["T_cam2ego"],
                device=device,
            )
            for n in names
        ])
        return depth, K_t, T_t

    # Generic fallback: numpy interface -> tensor (one copy).
    depth = np.stack([
        estimator.infer(
            cameras[n]["image"],
            lidar_sparse_depth=cameras[n].get("depth_map_sparse"),
        )
        for n in names
    ])
    return torch.from_numpy(depth).to(device), K_t, T_t


def build_osz_mask_online(
    depth_prob: "torch.Tensor",   # (B*N, D, Hf, Wf) softmax depth distribution
    bin_center: "torch.Tensor",   # (D,) expected-depth bin centres (metres)
    intrins: "torch.Tensor",      # (B, N, 3, 3) original intrinsics
    post_rots: "torch.Tensor",    # (B, N, 3, 3) augmented-image rotation/resize
    post_trans: "torch.Tensor",   # (B, N, 3) augmented-image translation (crop)
    sensor2ego: "torch.Tensor",   # (B, N, 4, 4) camera -> current ego
    device: str = "cuda",
    lidar_depth: Optional["torch.Tensor"] = None,  # (B, N, 256, 704) optional
    in_size: tuple = (256, 704),
    z_min: float = cfg.Z_MIN_M,
    z_max: float = cfg.Z_MAX_M,
    observer_height: float = cfg.OBSERVER_HEIGHT_M,
    max_depth: float = cfg.MAX_METRIC_DEPTH_M,
) -> "torch.Tensor":
    """Compute an OSZ mask online from the model's own depth distribution.

    Used by ``ResWorld(use_osz_rcsample=True)``: the RCSample depth (already
    produced by the view transformer) is turned into a metric depth map,
    back-projected with the *effective* intrinsics of the augmented image,
    and run through the GPU OSZ geometry.

    Effective intrinsics: the training pipeline maps original pixels to
    augmented pixels via ``[u',v',1] = A @ [u,v,1]`` with
    ``A = [[R, t],[0,0,1]]`` where ``R = post_rots[:2,:2]`` (resize+rot+flip)
    and ``t = post_trans[:2]`` (crop offset). Hence ``K_eff = A @ intrins``.

    Depth scale: GT depth is divided by ``depth_scale`` at load time
    (loading.py:215) and ``post_rot[2,2] = 1/depth_scale`` (loading.py:223),
    so the model predicts ``d' = Z/depth_scale``. Back-projecting with the
    augmented ``K_eff`` alone leaves a residual ``1/depth_scale`` in the
    camera coordinates (the ``resize`` factor in ``R`` cancels top and
    bottom, but ``depth_scale`` does not). We therefore multiply the
    expected depth back by ``ds = 1/post_rots[2,2] = depth_scale`` before
    back-projection.

    Returns ``(B, 3, 200, 200)`` float32 channels ``(osz_eye, osz_ground,
    semi)``, grid-aligned with the ResWorld BEV feature map — identical to
    the channels of the offline ``{token}.npz`` masks.
    """
    _require_torch()
    from OSZ.modules.ray_casting import RayCaster3D

    in_h, in_w = in_size
    B, N = intrins.shape[0], intrins.shape[1]
    BN = B * N

    caster = RayCaster3D(z_min=z_min, z_max=z_max)
    grid = _Grid(caster, device)

    # Full augmented-image transform A (post_rot has 1/depth_scale at [2,2]
    # for the camera-aware MLP; the geometry transform's third row is 1).
    A = post_rots.clone()
    A[:, :, :2, 2] = post_trans[:, :, :2]
    A[:, :, 2, 2] = 1.0
    K_eff = torch.matmul(A, intrins).reshape(BN, 3, 3)
    # sensor2ego comes from prepare_inputs (matmul -> float -> split ->
    # squeeze), whose strides are not contiguous; reshape (not view) copies
    # when needed.
    T_ego = sensor2ego.reshape(BN, 4, 4)

    masks = []
    with torch.no_grad():
        # Metric depth expectation: (B*N, Hf, Wf) -> (B*N, 1, in_h, in_w).
        metric = (depth_prob * bin_center.view(1, -1, 1, 1)).sum(
            dim=1, keepdim=True
        )
        metric = F.interpolate(
            metric, size=(in_h, in_w), mode="bilinear", align_corners=False,
        )
        # Undo the GT/depth_scale convention: post_rot[2,2] == 1/depth_scale.
        ds = (1.0 / post_rots[:, :, 2, 2]).reshape(BN, 1, 1, 1).to(metric.dtype)
        metric = metric * ds
        for b in range(B):
            sl = slice(b * N, (b + 1) * N)
            bev_cam = depth_maps_to_bev_height_torch(
                metric[sl, 0], K_eff[sl], T_ego[sl], grid,
                z_min=z_min, z_max=z_max, max_depth=max_depth,
            )
            if lidar_depth is not None:
                # gt_depth uses the same /depth_scale convention; undo it
                # consistently with the camera branch.
                bev_lidar = depth_maps_to_bev_height_torch(
                    lidar_depth[b].to(device).float() * ds[b],
                    K_eff[sl], T_ego[sl],
                    grid, z_min=z_min, z_max=z_max, max_depth=max_depth,
                )
                fused = bev_cam.clone()
                fallback = (fused <= 0.05) & (bev_lidar > 0.05)
                fused[fallback] = bev_lidar[fallback]
            else:
                fused = bev_cam
            fused = torch.clamp(fused, 0.0, float(z_max))
            osz_g, osz_e = cast_osz_height_aware_torch(
                fused, grid, observer_height=observer_height
            )
            semi = osz_g & ~osz_e
            masks.append(
                torch.stack([osz_e, osz_g, semi], dim=0).float()
            )
    return torch.stack(masks, dim=0)  # (B, 3, 200, 200)


def compute_osz_height_aware_from_cameras_torch(
    cameras: dict[str, dict],
    caster,
    estimator=None,
    observer_height: float = cfg.OBSERVER_HEIGHT_M,
    depth_key: str = "depth_map",
    use_uncertainty: bool = False,
    z_min: float = cfg.Z_MIN_M,
    z_max: float = cfg.Z_MAX_M,
    device: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """End-to-end height-aware OSZ on GPU (same signature as the numpy path).

    Returns numpy arrays ``(bev_height, osz_ground, osz_eye)`` so the
    caller (export script / future online inference) is backend-agnostic.
    """
    _require_torch()
    dev = resolve_device(device)
    grid = _Grid(caster, dev)

    with torch.no_grad():
        # Camera-primary depth.
        cam_d, K_t, T_t = _estimate_depths_torch(
            cameras, estimator, depth_key, dev)

        # LiDAR fallback depth (sparse preferred, dense as fallback).
        names = list(cameras.keys())
        lidar_d = np.stack([
            cameras[n].get("depth_map_sparse", cameras[n].get("depth_map"))
            for n in names
        ])
        lidar_t = torch.from_numpy(lidar_d).to(dev)

        if use_uncertainty:
            # Inverse-uncertainty weighted fusion (mirrors
            # build_bev_height_fused_uncertainty): camera uncertainty grows
            # with distance, LiDAR uncertainty with sparse density.
            h_cam, unc_cam, _ = depth_maps_to_bev_height_uncertainty_torch(
                cam_d, K_t, T_t, grid, z_min=z_min, z_max=z_max,
                uncertainty_mode="depth",
            )
            h_lidar, _, dens_lidar = depth_maps_to_bev_height_uncertainty_torch(
                lidar_t, K_t, T_t, grid, z_min=z_min, z_max=z_max,
                uncertainty_mode="constant",
            )
            unc_lidar = 1.0 / (dens_lidar + 1.0)
            eps = 1e-6
            w_cam = 1.0 / (unc_cam + eps)
            w_lidar = 1.0 / (unc_lidar + eps)

            cam_valid = h_cam > 0.05
            lidar_valid = h_lidar > 0.05
            both = cam_valid & lidar_valid
            fused = torch.zeros_like(h_cam)
            fused[cam_valid & ~lidar_valid] = h_cam[cam_valid & ~lidar_valid]
            fused[~cam_valid & lidar_valid] = h_lidar[~cam_valid & lidar_valid]
            fused[both] = (
                w_cam[both] * h_cam[both] + w_lidar[both] * h_lidar[both]
            ) / (w_cam[both] + w_lidar[both])
        else:
            # Hard fallback: camera where available, LiDAR where empty.
            bev_cam = depth_maps_to_bev_height_torch(
                cam_d, K_t, T_t, grid, z_min=z_min, z_max=z_max
            )
            bev_lidar = depth_maps_to_bev_height_torch(
                lidar_t, K_t, T_t, grid, z_min=z_min, z_max=z_max
            )
            fused = bev_cam.clone()
            fallback = (fused <= 0.05) & (bev_lidar > 0.05)
            fused[fallback] = bev_lidar[fallback]

        fused = torch.clamp(fused, 0.0, float(z_max))
        osz_ground, osz_eye = cast_osz_height_aware_torch(
            fused, grid, observer_height=observer_height
        )

    return (
        fused.cpu().numpy(),
        osz_ground.cpu().numpy(),
        osz_eye.cpu().numpy(),
    )
