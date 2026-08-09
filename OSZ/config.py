"""Global configuration constants for the OSZ pipeline.

Notes
-----
All numeric defaults are chosen for nuScenes mini / trainval:

- The BEV grid is defined HERE, aligned to ResWorld's ``grid_config``
  (``projects/configs/resworld/resworld_config.py``) — the model's single
  source of truth for the BEV grid. Keep the two in sync; then OSZ masks
  inject into the BEV feature map with NO resampling.
- Height thresholds correspond to a passenger vehicle with roof-mounted sensors.
- Depth truncation prevents distant phantom obstacles.
- ``MIDAS_MODEL_PATH`` points to a LOCAL MiDaS checkpoint; the depth
  estimator loads it from disk only and never downloads from the network.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# BEV grid — aligned with ResWorld `grid_config`
# ---------------------------------------------------------------------------
# ResWorld (projects/configs/resworld/resworld_config.py) uses:
#   grid_config = {'x': [-15, 15, 0.15], 'y': [-30, 30, 0.3], ...}
# which yields a 200x200 ANISOTROPIC grid: 0.15 m/cell along ego-x (forward),
# 0.3 m/cell along ego-y (left). OSZ must match exactly so masks can be fed
# into the ResWorld BEV feature map without resampling.
BEV_X_MIN: float = -15.0
BEV_X_MAX: float = 15.0
BEV_X_RES: float = 0.15
BEV_Y_MIN: float = -30.0
BEV_Y_MAX: float = 30.0
BEV_Y_RES: float = 0.3

#: Cells along ego-x (forward) / ego-y (left). Both 200 for ResWorld's grid.
BEV_NX: int = int(round((BEV_X_MAX - BEV_X_MIN) / BEV_X_RES))
BEV_NY: int = int(round((BEV_Y_MAX - BEV_Y_MIN) / BEV_Y_RES))

#: OSZ convention: (x_min, x_max, y_min, y_max) in metres.
BEV_RANGE_M: tuple[float, float, float, float] = (
    BEV_X_MIN, BEV_X_MAX, BEV_Y_MIN, BEV_Y_MAX,
)

# ---------------------------------------------------------------------------
# Height thresholds for obstacle / free-space classification
# ---------------------------------------------------------------------------
Z_MIN_M: float = 0.8
"""Minimum height for a point to be considered an obstacle."""

Z_MAX_M: float = 3.0
"""Maximum plausible obstacle height; taller points are clipped."""

Z_RES_M: float = 0.3
"""Voxel resolution along z for the 3D ray caster (single source of truth)."""

OBSERVER_HEIGHT_M: float = 1.2
"""Height of the virtual observer used in eye-level ray casting."""

EGO_CLEARANCE_RADIUS_M: float = 1.0
"""Radius around the ego vehicle that is cleared of spurious obstacles."""

# ---------------------------------------------------------------------------
# Cameras and depth
# ---------------------------------------------------------------------------
NUSCENES_CAMERAS: list[str] = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
"""Canonical nuScenes camera names."""

MAX_METRIC_DEPTH_M: float = 70.0
"""Metric depth values above this are truncated to avoid distant phantoms."""

MIDAS_MODEL_PATH: str = "OSZ/weights/midas_v21_small_256.pt"
"""Local MiDaS v2.1 Small checkpoint (MiDaSNet-small, 256x256 input).
Loaded from disk only, never downloaded."""

MIDAS_REPO_PATH: str = "OSZ/third_party/MiDaS"
"""Local clone of https://github.com/isl-org/MiDaS (cloned once, has
``hubconf.py``). Loaded via ``torch.hub.load(..., source='local')`` so no
network access is needed at runtime."""

MIDAS_MODEL_URL: str = (
    "https://github.com/isl-org/MiDaS/releases/download/v2_1/"
    "midas_v21_small_256.pt"
)
"""Upstream URL for manual download (documented in INSTALL.md); not fetched at runtime."""

MIN_ALIGN_POINTS: int = 20
"""Minimum LiDAR points required for depth-scale alignment."""

# ---------------------------------------------------------------------------
# Drivable-area filtering
# ---------------------------------------------------------------------------
DRIVABLE_MAP_LAYERS: list[str] = ["drivable_area", "carpark_area"]
"""HD-map layers used to define drivable regions."""

EXCLUDED_MAP_LAYERS: list[str] = ["walkway", "ped_crossing"]
"""HD-map layers explicitly excluded from the drivable mask."""

DEFAULT_DRIVABLE_DILATION_M: float = 1.5
"""Dilation radius for the drivable mask in metres."""
