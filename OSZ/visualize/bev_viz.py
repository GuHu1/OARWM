"""Bird's-eye view (BEV) visualizations for the OSZ pipeline.

This module produces BEV figures that overlay OSZ shadow regions,
occluders, ground-truth bounding boxes, depth maps, and ego vehicle
pose.

Notes
-----
Coordinate system (applies to all ``imshow`` calls in this file):

- Ego frame: ``x`` forward, ``y`` left, ``z`` up (nuScenes convention).
- BEV array: shape ``(nx, ny)`` with ``indexing='ij'``. ``array[i, j]``
  maps ``i`` to ego-x (axis 0) and ``j`` to ego-y (axis 1).
- ``imshow`` call convention::

      imshow(data,                # NO transpose
             origin='lower',
             extent=[y_max, y_min, x_min, x_max])
      set_xlim(y_max, y_min)      # left = ego-left = +y
      set_ylim(x_min, x_max)      # forward = UP (+x at top)

No transpose is used because, with ``origin='lower'``,
``data[row=i, col=j]`` is placed at axes ``(x=col_coord, y=row_coord)``.
Thus ``i`` (ego-x = forward) maps to the vertical axis and ``j``
(ego-y = left) maps to the horizontal axis.

The y-extent is inverted (``y_max`` on the left, ``y_min`` on the
right) because ego-y is positive leftward; the image left side should
therefore correspond to ego-left (positive y).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from OSZ import config as cfg
from OSZ.utils.geometry import (
    ego_pose_from_sample_data,
    get_map_name,
    get_vehicle_states_ego,
    bev_box_corners_ego,
    bev_extent as _bev_extent,
    VEHICLE_CATEGORIES,
)


CAMERA_COLORS = {
    "CAM_FRONT": "#4ECDC4",
    "CAM_FRONT_LEFT": "#45B7D1",
    "CAM_FRONT_RIGHT": "#96CEB4",
    "CAM_BACK": "#FFEAA7",
    "CAM_BACK_LEFT": "#DDA0DD",
    "CAM_BACK_RIGHT": "#F0E68C",
}

_OSZ_PALETTE = {
    "road": (0.29, 0.29, 0.29),
    "grass": (0.30, 0.69, 0.31),
    "obstacle": (1.00, 0.60, 0.00),
    "osz": (0.00, 0.00, 0.00),
    "ego": (0.10, 0.46, 0.82),
    "lane": (1.00, 1.00, 1.00),
    "bg": (1.00, 1.00, 1.00),
    "text": (0.13, 0.13, 0.13),
    "text_mid": (0.33, 0.33, 0.33),
}

_VEHICLE_CATS = VEHICLE_CATEGORIES


def _draw_ego(ax: plt.Axes, size: float = 2.0, color: str = "#1976d2") -> None:
    """Draw the ego vehicle as a small rectangle with a forward arrow."""
    rect = plt.Rectangle(
        (-size / 2, -size), size, size * 2,
        linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.85,
    )
    ax.add_patch(rect)
    ax.annotate(
        "", xy=(0, size * 1.5), xytext=(0, size),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
    )


# bev_box_corners_ego() and get_vehicle_states_ego() live in
# OSZ/utils/geometry.py (former common/coords.py, now removed).


def plot_camera_osz_comparison(
    images: dict[str, np.ndarray],
    depth_maps: dict[str, np.ndarray],
    per_cam_masks: dict[str, np.ndarray],
    osz_mask: np.ndarray,
    refined_mask: np.ndarray | None,
    depth_bev: np.ndarray,
    bev_occ: np.ndarray | None = None,
    osz_pa: np.ndarray | None = None,
    bev_range: tuple[float, float, float, float] = cfg.BEV_RANGE_M,
    sample_token: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot camera images, dense depths, and fused BEV OSZ results.

    The figure contains one row per camera plus a final row showing the
    fused OSZ mask, PA-relevant OSZ, and BEV depth map.

    Parameters
    ----------
    images : dict[str, np.ndarray]
        Mapping from camera name to RGB image array ``(H, W, 3)``.
    depth_maps : dict[str, np.ndarray]
        Mapping from camera name to dense depth map ``(H, W)``.
    per_cam_masks : dict[str, np.ndarray]
        Mapping from camera name to per-camera BEV OSZ mask
        ``(nx, ny)``.
    osz_mask : np.ndarray
        Fused raw OSZ BEV mask ``(nx, ny)``.
    refined_mask : np.ndarray | None
        Optional refined OSZ BEV mask ``(nx, ny)``. Displayed in the
        PA-relevant panel when ``osz_pa`` is ``None``.
    depth_bev : np.ndarray
        BEV depth map ``(nx, ny)``.
    bev_occ : np.ndarray | None, optional
        Optional BEV occupancy / occluder mask ``(nx, ny)``. Used as a
        background layer. Default is ``None``.
    osz_pa : np.ndarray | None, optional
        Optional PA-relevant OSZ mask ``(nx, ny)``. Takes precedence
        over ``refined_mask`` in the second-to-last panel. Default is
        ``None``.
    bev_range : tuple[float, float, float, float], optional
        BEV extent as ``(x_min, x_max, y_min, y_max)`` in metres.
        Defaults to ``cfg.BEV_RANGE_M``.
    sample_token : str, optional
        Sample token shown in the figure title. Default is empty.
    save_path : str | None, optional
        If provided, the figure is saved to this path. Default is
        ``None``.

    Returns
    -------
    plt.Figure
        The generated Matplotlib figure.
    """
    cam_names = list(images.keys())
    n_cam = len(cam_names)
    n_rows = n_cam + 1
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.2))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(
        f"OSZ Comparison — Frame: {sample_token}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    extent, xlim, ylim = _bev_extent(bev_range)
    shadow_cmap = ListedColormap(["none", "#d32f2f"])

    for i, cam_name in enumerate(cam_names):
        ax_img, ax_depth, ax_bev = axes[i][0], axes[i][1], axes[i][2]

        ax_img.imshow(images[cam_name])
        ax_img.set_title(f"{cam_name} — Image", fontsize=10)
        ax_img.axis("off")

        dmap = depth_maps[cam_name]
        im_d = ax_depth.imshow(dmap, cmap="plasma", vmin=0, vmax=cfg.MAX_METRIC_DEPTH_M)
        ax_depth.set_title(f"{cam_name} — Dense Depth", fontsize=10)
        ax_depth.axis("off")
        plt.colorbar(im_d, ax=ax_depth, fraction=0.046, pad=0.04)

        mask = per_cam_masks[cam_name]
        color = CAMERA_COLORS.get(cam_name, "#888888")
        cmap_cam = ListedColormap(["none", color])
        if bev_occ is not None:
            ax_bev.imshow(
                bev_occ, origin="lower", extent=extent,
                cmap="Greys", vmin=0, vmax=1, interpolation="nearest", alpha=0.3,
            )
        ax_bev.imshow(
            mask, origin="lower", extent=extent,
            cmap=cmap_cam, vmin=0, vmax=1, interpolation="nearest", alpha=0.85,
        )
        ax_bev.set_xlim(*xlim)
        ax_bev.set_ylim(*ylim)
        ax_bev.set_title(f"{cam_name} — Shadow BEV", fontsize=10)
        ax_bev.set_xlabel("y (m)", fontsize=7)
        ax_bev.set_ylabel("x (m)", fontsize=7)
        ax_bev.tick_params(labelsize=6)
        _draw_ego(ax_bev)

    # Last row: fused results
    ax_osz = axes[-1][0]
    if bev_occ is not None:
        ax_osz.imshow(
            bev_occ, origin="lower", extent=extent,
            cmap="Greys", vmin=0, vmax=1, interpolation="nearest", alpha=0.4,
        )
    ax_osz.imshow(
        osz_mask, origin="lower", extent=extent,
        cmap=shadow_cmap, vmin=0, vmax=1, interpolation="nearest",
    )
    ax_osz.set_xlim(*xlim)
    ax_osz.set_ylim(*ylim)
    ax_osz.set_title("OSZ Raw (ego ray casting)", fontsize=10)
    ax_osz.set_xlabel("y (m)", fontsize=7)
    ax_osz.set_ylabel("x (m)", fontsize=7)
    _draw_ego(ax_osz)

    ax_ref = axes[-1][1]
    if osz_pa is not None:
        ax_ref.imshow(
            osz_pa, origin="lower", extent=extent,
            cmap=shadow_cmap, vmin=0, vmax=1, interpolation="nearest",
        )
    elif refined_mask is not None:
        ax_ref.imshow(
            refined_mask, origin="lower", extent=extent,
            cmap="Reds", vmin=0, vmax=1, interpolation="bilinear",
        )
    ax_ref.set_xlim(*xlim)
    ax_ref.set_ylim(*ylim)
    ax_ref.set_title("PA-relevant OSZ", fontsize=10)
    ax_ref.set_xlabel("y (m)", fontsize=7)
    ax_ref.set_ylabel("x (m)", fontsize=7)
    _draw_ego(ax_ref)

    ax_dbev = axes[-1][2]
    im_db = ax_dbev.imshow(
        depth_bev, origin="lower", extent=extent,
        cmap="plasma", vmin=0, vmax=cfg.MAX_METRIC_DEPTH_M, interpolation="bilinear",
    )
    if bev_occ is not None:
        ax_dbev.imshow(
            bev_occ, origin="lower", extent=extent,
            cmap="Greys", vmin=0, vmax=1, interpolation="nearest", alpha=0.25,
        )
    ax_dbev.set_xlim(*xlim)
    ax_dbev.set_ylim(*ylim)
    ax_dbev.set_title("BEV Depth", fontsize=10)
    ax_dbev.set_xlabel("y (m)", fontsize=7)
    ax_dbev.set_ylabel("x (m)", fontsize=7)
    _draw_ego(ax_dbev)
    plt.colorbar(im_db, ax=ax_dbev, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    return fig


def plot_gt_osz(
    osz_pa: np.ndarray,
    bev_occ: np.ndarray,
    drivable_mask: np.ndarray,
    nusc,
    sample_token: str,
    bev_range: tuple[float, float, float, float] = cfg.BEV_RANGE_M,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot GT boxes and PA-relevant OSZ on a single BEV panel.

    Parameters
    ----------
    osz_pa : np.ndarray
        PA-relevant OSZ BEV mask ``(nx, ny)``.
    bev_occ : np.ndarray
        BEV occupancy / occluder mask ``(nx, ny)``.
    drivable_mask : np.ndarray
        Drivable-area BEV mask ``(nx, ny)``.
    nusc : nuscenes.nuscenes.NuScenes
        Initialized nuScenes dataset instance.
    sample_token : str
        nuScenes sample token used to retrieve ground-truth boxes.
    bev_range : tuple[float, float, float, float], optional
        BEV extent as ``(x_min, x_max, y_min, y_max)`` in metres.
        Defaults to ``cfg.BEV_RANGE_M``.
    save_path : str | None, optional
        If provided, the figure is saved to this path. Default is
        ``None``.

    Returns
    -------
    plt.Figure
        The generated Matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    x_min, x_max, y_min, y_max = bev_range
    extent, xlim, ylim = _bev_extent(bev_range)
    nx, ny = osz_pa.shape
    # Anisotropic grid: x 0.15 m/cell, y 0.3 m/cell (aligned with ResWorld).
    rx, ry = cfg.BEV_X_RES, cfg.BEV_Y_RES

    # Background layers
    overlay = np.zeros((nx, ny, 3), dtype=np.float32)
    if drivable_mask is not None and drivable_mask.any():
        non_drivable = ~(drivable_mask | bev_occ)
        overlay[non_drivable] = _OSZ_PALETTE["grass"]
        overlay[drivable_mask] = _OSZ_PALETTE["road"]
    else:
        overlay[:] = _OSZ_PALETTE["grass"]
        overlay[~bev_occ] = _OSZ_PALETTE["road"]

    overlay[bev_occ] = _OSZ_PALETTE["obstacle"]
    if drivable_mask is not None and drivable_mask.any():
        overlay[osz_pa & drivable_mask] = _OSZ_PALETTE["osz"]
    else:
        overlay[osz_pa] = _OSZ_PALETTE["osz"]

    ax.imshow(overlay, origin="lower", extent=extent)

    # GT boxes
    boxes = get_vehicle_states_ego(nusc, sample_token, bev_range)

    for box in boxes:
        w, l = box["width"], box["length"]
        cos_h, sin_h = np.cos(box["yaw"]), np.sin(box["yaw"])
        half_local = np.array(
            [[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]],
            dtype=np.float32,
        )
        R = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float32)
        corners = (R @ half_local.T).T + np.array(
            [box["cx"], box["cy"]], dtype=np.float32
        )

        i_lo = max(0, int(np.floor((corners[:, 0].min() - x_min) / rx)))
        i_hi = min(nx - 1, int(np.ceil((corners[:, 0].max() - x_min) / rx)))
        j_lo = max(0, int(np.floor((y_max - corners[:, 1].max()) / ry)))
        j_hi = min(ny - 1, int(np.ceil((y_max - corners[:, 1].min()) / ry)))

        Rt = R.T
        total = 0
        n_in_occ = 0
        for i in range(i_lo, i_hi + 1):
            x_c = x_min + (i + 0.5) * rx
            for j in range(j_lo, j_hi + 1):
                y_c = y_max - (j + 0.5) * ry
                dx = x_c - box["cx"]
                dy = y_c - box["cy"]
                lx = Rt[0, 0] * dx + Rt[0, 1] * dy
                ly = Rt[1, 0] * dx + Rt[1, 1] * dy
                if abs(lx) > l / 2 or abs(ly) > w / 2:
                    continue
                total += 1
                if bev_occ[i, j]:
                    n_in_occ += 1

        box["in_osz"] = (
            (n_in_occ == 0)
            and (total > 0)
            and bool(osz_pa[
                int(np.rint((box["cx"] - x_min) / rx)),
                int(np.rint((y_max - box["cy"]) / ry)),
            ])
        )
        box["occ_overlap"] = n_in_occ / max(total, 1)
        box["footprint_cells"] = total

    n_phantom = sum(
        1 for b in boxes if b["in_osz"] and b["category"] in _VEHICLE_CATS
    )

    for box in boxes:
        corners = bev_box_corners_ego(
            box["cx"], box["cy"], box["yaw"], box["width"], box["length"]
        )
        poly_ax_x = list(corners[:, 1]) + [corners[0, 1]]  # ego-y -> horizontal
        poly_ax_y = list(corners[:, 0]) + [corners[0, 0]]  # ego-x -> vertical

        cat = box["category"]
        if cat in _VEHICLE_CATS:
            if box["in_osz"]:
                base_color = "#ff0000"
                edge_color = "#ffffff"
                facecolor = "none"
                lw = 2.2
                alpha = 1.0
            else:
                base_color = "#ff9800"
                edge_color = "#e65100"
                facecolor = "#ff9800"
                lw = 1.8
                alpha = 0.45
        else:
            base_color = "#AAAAAA"
            edge_color = "#666666"
            facecolor = "none"
            lw = 1.0
            alpha = 0.9

        if facecolor != "none":
            ax.fill(
                poly_ax_x, poly_ax_y, facecolor=facecolor, alpha=alpha,
                edgecolor="none", zorder=4,
            )
        ax.plot(
            poly_ax_x, poly_ax_y, color=edge_color, linewidth=lw + 0.6, zorder=5
        )
        ax.plot(
            poly_ax_x, poly_ax_y, color=base_color, linewidth=lw,
            alpha=alpha if facecolor == "none" else 0.95, zorder=5,
        )

        front_len = box["length"] * 0.4
        ax.annotate(
            "",
            xy=(box["cy"] + sin_h * front_len, box["cx"] + cos_h * front_len),
            xytext=(box["cy"], box["cx"]),
            arrowprops=dict(arrowstyle="->", color=base_color, lw=lw * 0.8),
        )

    _draw_ego(ax, size=2.5)

    for d in range(-40, 50, 10):
        ax.axhline(d, color="#dddddd", lw=0.4, alpha=0.8)
        ax.axvline(d, color="#dddddd", lw=0.4, alpha=0.8)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.axvline(0, color="#999999", lw=0.8)

    legend_items = [
        mpatches.Patch(facecolor=_OSZ_PALETTE["road"], label="Road"),
        mpatches.Patch(facecolor=_OSZ_PALETTE["grass"], label="Non-drivable ground"),
        mpatches.Patch(facecolor=_OSZ_PALETTE["obstacle"], label="Occluder (LiDAR/depth)"),
        mpatches.Patch(
            facecolor=_OSZ_PALETTE["osz"],
            label=f"PA-relevant OSZ ({osz_pa.sum()} cells)",
        ),
        mpatches.Patch(
            facecolor="#ff9800", edgecolor="#e65100", alpha=0.55,
            label="Vehicle (visible) — can cause OSZ",
        ),
        plt.Line2D(
            [0], [0], color="#ff0000", lw=2,
            label=f"Vehicle in OSZ ({n_phantom}) — phantom candidate",
        ),
        mpatches.Patch(facecolor="#AAAAAA", label="Other object"),
    ]
    ax.legend(handles=legend_items, fontsize=7, loc="upper right", framealpha=0.9)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(
        "y (m)  ← ego-left | ego-right →", fontsize=9, color=_OSZ_PALETTE["text"]
    )
    ax.set_ylabel("x (m)  ↑ forward", fontsize=9, color=_OSZ_PALETTE["text"])
    ax.tick_params(labelsize=8, colors=_OSZ_PALETTE["text"])
    ax.set_title(
        f"BEV GT + PA-relevant OSZ  |  {sample_token[:16]}...\n"
        f"{len(boxes)} annotations  |  {n_phantom} phantom candidates",
        fontsize=11, fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    return fig


def plot_osz_explained(
    osz_pa: np.ndarray,
    bev_occ: np.ndarray,
    drivable_mask: np.ndarray | None = None,
    bev_range: tuple[float, float, float, float] = cfg.BEV_RANGE_M,
    sample_token: str = "",
    title_extra: str = "",
    save_path: str | None = None,
    draw_lanes: bool = False,
    nusc=None,
) -> plt.Figure:
    """Plot a single BEV panel with occluders, OSZ shadow, road and ego.

    Parameters
    ----------
    osz_pa : np.ndarray
        OSZ shadow BEV mask ``(nx, ny)``.
    bev_occ : np.ndarray
        BEV occupancy / occluder mask ``(nx, ny)``.
    drivable_mask : np.ndarray | None, optional
        Optional drivable-area BEV mask ``(nx, ny)``. Default is
        ``None``.
    bev_range : tuple[float, float, float, float], optional
        BEV extent as ``(x_min, x_max, y_min, y_max)`` in metres.
        Defaults to ``cfg.BEV_RANGE_M``.
    sample_token : str, optional
        Sample token shown in the title. Default is empty.
    title_extra : str, optional
        Extra text appended to the title. Default is empty.
    save_path : str | None, optional
        If provided, the figure is saved to this path. Default is
        ``None``.
    draw_lanes : bool, optional
        Whether to overlay HD-map lane lines. Requires ``nusc`` and
        ``sample_token`` to be provided. Default is ``False``.
    nusc : nuscenes.nuscenes.NuScenes | None, optional
        nuScenes instance used for lane rendering when ``draw_lanes`` is
        ``True``. Default is ``None``.

    Returns
    -------
    plt.Figure
        The generated Matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    x_min, x_max, y_min, y_max = bev_range
    extent, xlim, ylim = _bev_extent(bev_range)
    nx, ny = osz_pa.shape

    overlay = np.zeros((nx, ny, 3), dtype=np.float32)
    if drivable_mask is not None and drivable_mask.any():
        non_drivable = ~(drivable_mask | bev_occ)
        overlay[non_drivable] = _OSZ_PALETTE["grass"]
        overlay[drivable_mask] = _OSZ_PALETTE["road"]
    else:
        overlay[:] = _OSZ_PALETTE["grass"]
        overlay[~bev_occ] = _OSZ_PALETTE["road"]

    overlay[bev_occ] = _OSZ_PALETTE["obstacle"]
    if drivable_mask is not None and drivable_mask.any():
        overlay[osz_pa & drivable_mask] = _OSZ_PALETTE["osz"]
    else:
        overlay[osz_pa] = _OSZ_PALETTE["osz"]

    ax.imshow(overlay, origin="lower", extent=extent)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    _draw_ego(ax, size=2.5)

    # Optional HD-map lane lines.
    if draw_lanes and nusc is not None and sample_token:
        try:
            from nuscenes.map_expansion.map_api import NuScenesMap
            import pyquaternion
            location = get_map_name(nusc, sample_token)
            lidar_sd_token = nusc.get("sample", sample_token)["data"]["LIDAR_TOP"]
            ep = ego_pose_from_sample_data(nusc, lidar_sd_token)
            ego_t = np.array(ep["translation"], dtype=np.float64)
            ego_q = pyquaternion.Quaternion(ep["rotation"])
            if location:
                nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=location)
                recs = nusc_map.get_records_in_radius(
                    float(ego_t[0]), float(ego_t[1]), 55.0,
                    ["lane", "road_segment"],
                )
                for layer in ("lane", "road_segment"):
                    for tok in recs.get(layer, []):
                        rec = nusc_map.get(layer, tok)
                        poly_rec = nusc_map.get("polygon", rec["polygon_token"])
                        nodes = [nusc_map.get("node", nt)
                                 for nt in poly_rec["exterior_node_tokens"]]
                        pts = []
                        for nd in nodes:
                            gpos = np.array([nd["x"], nd["y"], 0.0], dtype=np.float32)
                            delta = gpos - ego_t
                            epos = ego_q.inverse.rotate(delta)
                            pts.append((epos[1], epos[0]))
                        if len(pts) >= 2:
                            xs, ys = zip(*pts)
                            ax.plot(xs, ys, "-", color=_OSZ_PALETTE["lane"],
                                    linewidth=0.4, alpha=0.5, zorder=1)
        except Exception:
            pass

    for d in range(-40, 50, 10):
        ax.axhline(d, color="#666666", lw=0.3, alpha=0.35)
        ax.axvline(d, color="#666666", lw=0.3, alpha=0.35)
    ax.axhline(0, color="#888888", lw=0.6)
    ax.axvline(0, color="#888888", lw=0.6)

    n_osz = int(osz_pa.sum())
    n_occ = int(bev_occ.sum())
    pct_osz = n_osz / max(osz_pa.size, 1) * 100
    pct_occ = n_occ / max(bev_occ.size, 1) * 100
    title = (
        f"OSZ explained | {pct_osz:.1f}% OSZ ({n_osz} cells) "
        f"| {pct_occ:.1f}% occluders ({n_occ} cells)"
    )
    if sample_token:
        title = f"{title}\n{sample_token[:24]}..."
    if title_extra:
        title = f"{title}  {title_extra}"
    ax.set_title(title, fontsize=10, fontweight="bold", color=_OSZ_PALETTE["text"])

    ax.set_xlabel(
        "y (m)  ← ego-left | ego-right →", fontsize=8, color=_OSZ_PALETTE["text_mid"]
    )
    ax.set_ylabel(
        "x (m)  ↑ forward", fontsize=8, color=_OSZ_PALETTE["text_mid"]
    )
    ax.tick_params(labelsize=7)

    handles = [
        mpatches.Patch(color=_OSZ_PALETTE["road"], label="Road (drivable)"),
        mpatches.Patch(color=_OSZ_PALETTE["grass"], label="Non-drivable ground"),
        mpatches.Patch(
            color=_OSZ_PALETTE["obstacle"],
            label=f"Occluders ({n_occ} cells) — cause the shadow",
        ),
        mpatches.Patch(
            color=_OSZ_PALETTE["osz"],
            label=f"OSZ ({n_osz} cells) — the shadow",
        ),
        plt.Line2D(
            [0], [0], marker="^", color=_OSZ_PALETTE["ego"],
            markersize=8, linestyle="none", label="Ego",
        ),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.9)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    return fig
