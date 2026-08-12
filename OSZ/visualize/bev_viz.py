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

from OSZ import config as cfg
from OSZ.utils.geometry import (
    get_vehicle_states_ego,
    bev_box_corners_ego,
    bev_extent as _bev_extent,
    VEHICLE_CATEGORIES,
)



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


