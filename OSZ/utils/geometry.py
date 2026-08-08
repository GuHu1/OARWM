"""Geometry and BEV-grid helpers for the OSZ pipeline.

Notes
-----
The former ``common/`` package was removed so the OSZ BEV grid can stay
aligned with ResWorld's ``grid_config`` (see ``OSZ/config.py``). Everything
that used to live in ``common/coords.py`` and is actually used by OSZ lives
here now.

Coordinate convention (nuScenes ego frame): x = forward, y = left, z = up.
BEV arrays use shape ``(nx, ny)`` with ``indexing='ij'``: axis-0 = ego-x,
axis-1 = ego-y. The grid is ANISOTROPIC (0.15 m/cell in x, 0.3 m/cell in y),
so every pixel<->metric mapping goes through the per-axis resolutions.
"""
from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import pyquaternion

from OSZ import config as cfg


# ───────────────────────────────────────────────────────────────────────────
# BEV grid helpers (aligned with ResWorld grid_config — see OSZ/config.py)
# ───────────────────────────────────────────────────────────────────────────

def bev_grid_shape(
    bev_range: Tuple[float, float, float, float],
) -> Tuple[int, int]:
    """Return ``(nx, ny)`` for the project-wide BEV grid.

    The grid is fixed by ``OSZ/config.py`` (aligned with ResWorld's
    ``grid_config``); ``bev_range`` is validated against it so a future
    ResWorld grid change fails loudly instead of silently misaligning.
    """
    x_min, x_max, y_min, y_max = bev_range
    nx = int(round((x_max - x_min) / cfg.BEV_X_RES))
    ny = int(round((y_max - y_min) / cfg.BEV_Y_RES))
    if (nx, ny) != (cfg.BEV_NX, cfg.BEV_NY):
        raise ValueError(
            f"BEV range {bev_range} does not match the project grid "
            f"({cfg.BEV_NX}x{cfg.BEV_NY}); update OSZ/config.py to stay in "
            "sync with ResWorld's grid_config."
        )
    return nx, ny


def bev_coords_to_pixel(x_ego: float, y_ego: float) -> Tuple[int, int]:
    """Convert metric ego coordinates to BEV pixel indices.

    Ego vehicle is at ``(BEV_NX // 2, BEV_NY // 2)`` in pixel space.
    ``col`` indexes ego-x (axis 0), ``row`` indexes ego-y (axis 1, reversed so
    that ``imshow(origin='lower')`` puts ego-left (+y) on the left).

    Parameters
    ----------
    x_ego : float
        Forward distance from ego (metres).
    y_ego : float
        Leftward distance from ego (metres).

    Returns
    -------
    Tuple[int, int]
        ``(col, row)`` pixel indices.
    """
    col = int((x_ego - cfg.BEV_X_MIN) / cfg.BEV_X_RES)
    row = int((cfg.BEV_Y_MAX - y_ego) / cfg.BEV_Y_RES)
    return col, row


def pixel_to_bev_coords(col: int, row: int) -> Tuple[float, float]:
    """Inverse of :func:`bev_coords_to_pixel`: pixel -> metric ego centre."""
    x = cfg.BEV_X_MIN + (col + 0.5) * cfg.BEV_X_RES
    y = cfg.BEV_Y_MAX - (row + 0.5) * cfg.BEV_Y_RES
    return x, y


def bev_extent(
    bev_range: Tuple[float, float, float, float]
) -> Tuple[list, Tuple[float, float], Tuple[float, float]]:
    """Return matplotlib ``extent``, ``xlim`` and ``ylim`` for a BEV range.

    Parameters
    ----------
    bev_range : Tuple[float, float, float, float]
        ``(x_min, x_max, y_min, y_max)`` in metres.

    Returns
    -------
    extent : list
        ``[y_max, y_min, x_min, x_max]`` — horizontal axis is ego-y (inverted
        so ego-left appears on the left), vertical axis is ego-x (forward is
        up).
    xlim : Tuple[float, float]
        ``(y_max, y_min)`` for ``ax.set_xlim()``.
    ylim : Tuple[float, float]
        ``(x_min, x_max)`` for ``ax.set_ylim()``.
    """
    x_min, x_max, y_min, y_max = bev_range
    return [y_max, y_min, x_min, x_max], (y_max, y_min), (x_min, x_max)


# ───────────────────────────────────────────────────────────────────────────
# nuScenes ego-pose helpers
# ───────────────────────────────────────────────────────────────────────────

def get_map_name(nusc, sample_token: str) -> str:
    """Return the map location (e.g. ``'boston-seaport'``) for a sample."""
    sample = nusc.get("sample", sample_token)
    scene = nusc.get("scene", sample["scene_token"])
    log = nusc.get("log", scene["log_token"])
    return log["location"]


def ego_pose_from_sample_data(nusc, sample_data_token: str) -> dict:
    """Return the ego pose dict for a sample_data token."""
    sd = nusc.get("sample_data", sample_data_token)
    return nusc.get("ego_pose", sd["ego_pose_token"])


# ───────────────────────────────────────────────────────────────────────────
# BEV box geometry
# ───────────────────────────────────────────────────────────────────────────

def bev_box_corners_ego(
    x_ego: float, y_ego: float, heading: float, w: float, l: float
) -> np.ndarray:
    """Return the four BEV corners of an ego-frame box.

    ``heading=0`` means the box faces ``+x`` (forward). Positive heading is a
    left turn (counter-clockwise, toward ``+y``). ``w`` is width (lateral) and
    ``l`` is length (forward).

    Returns
    -------
    np.ndarray
        ``(4, 2)`` corner array of ``(x_ego, y_ego)``.
    """
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    local = np.array(
        [[l / 2, w / 2],
         [l / 2, -w / 2],
         [-l / 2, -w / 2],
         [-l / 2, w / 2]],
        dtype=np.float32,
    )
    corners = local @ R.T
    corners[:, 0] += x_ego
    corners[:, 1] += y_ego
    return corners


def get_vehicle_states_ego(
    nusc, sample_token: str, bev_range: Tuple[float, float, float, float]
) -> List[dict]:
    """Return ego-frame vehicle states inside a BEV range.

    Each dict contains ``cx, cy, length, width, yaw, category, token,
    in_osz``.
    """
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ep = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    ego_t = np.array(ep["translation"], dtype=np.float64)
    ego_q = pyquaternion.Quaternion(ep["rotation"])

    x_min, x_max, y_min, y_max = bev_range
    boxes = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        delta = np.array(ann["translation"]) - ego_t
        pt_ego = ego_q.inverse.rotate(delta)

        cx, cy = float(pt_ego[0]), float(pt_ego[1])
        if not (x_min <= cx <= x_max and y_min <= cy <= y_max):
            continue

        box_q = pyquaternion.Quaternion(ann["rotation"])
        box_q_ego = ego_q.inverse * box_q
        yaw_ego = box_q_ego.yaw_pitch_roll[0]

        boxes.append({
            "cx": cx,
            "cy": cy,
            "length": ann["size"][1],
            "width": ann["size"][0],
            "yaw": yaw_ego,
            "category": ann["category_name"],
            "token": ann_token,
            "in_osz": False,
        })
    return boxes


# ───────────────────────────────────────────────────────────────────────────
# Vehicle category set (shared across modules)
# ───────────────────────────────────────────────────────────────────────────

VEHICLE_CATEGORIES = {
    "vehicle.car",
    "vehicle.truck",
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.motorcycle",
    "vehicle.bicycle",
    "vehicle.trailer",
    "vehicle.construction",
    "vehicle.emergency.ambulance",
    "vehicle.emergency.police",
}
