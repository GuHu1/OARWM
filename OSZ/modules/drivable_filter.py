"""Intersect geometric OSZ with the vehicle-plausible area from the HD map.

Coordinate frames
-----------------
nuScenes ``get_map_geom(patch_angle=0)`` returns polygons in LOCAL coords
centred at ego, but still aligned to GLOBAL north-up axes.

Our BEV grid uses EGO-CENTRIC coords:

- axis-0 = ego forward (ego +x)
- axis-1 = ego left   (ego +y)

The rotation from global to ego is a 2D rotation by ``-ego_yaw`` around the
origin.  We apply this rotation directly to the polygon vertices (exact, no
image interpolation) and then rasterize in ego frame.  This avoids the
axis-direction confusion and interpolation artifacts of rotating a raster.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from OSZ import config as cfg
from OSZ.utils.geometry import get_map_name
from OSZ.utils.geometry import bev_grid_shape

try:
    from nuscenes.map_expansion.map_api import NuScenesMap
    import pyquaternion
    from shapely import affinity
    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False


_map_cache: dict[tuple[str, str], "NuScenesMap"] = {}


def get_nusc_map(dataroot: str, map_name: str) -> "NuScenesMap":
    """Return a cached ``NuScenesMap`` instance.

    Parameters
    ----------
    dataroot : str
        Path to the nuScenes dataset root.
    map_name : str
        Name of the map to load (e.g. ``'singapore-onenorth'``).

    Returns
    -------
    NuScenesMap
        Cached map instance.
    """
    key = (dataroot, map_name)
    if key not in _map_cache:
        try:
            _map_cache[key] = NuScenesMap(dataroot=dataroot, map_name=map_name)
        except Exception as e:
            raise RuntimeError(
                f"NuScenesMap({dataroot}, {map_name}) failed: {e}  "
                f"— Is map data present at {dataroot}/maps/?"
            ) from e
    return _map_cache[key]


def _get_ego_pose(nusc, sample_token: str) -> tuple[np.ndarray, float]:
    """Return the ego pose for a sample.

    Parameters
    ----------
    nusc : NuScenes
        nuScenes dataset handle.
    sample_token : str
        Token of the sample to query.

    Returns
    -------
    tuple
        ``(ego_translation_global, ego_yaw_rad)``.
    """
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ep = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    t = np.array(ep["translation"], dtype=np.float64)
    q = pyquaternion.Quaternion(ep["rotation"])
    return t, q.yaw_pitch_roll[0]


def _rotate_to_ego(geom, ego_yaw_rad: float):
    """Rotate a Shapely geometry from global-aligned to ego-centric coords.

    The input is already centred at ego (patch-centred) but still aligned to
    global north-up.  We rotate by ``-ego_yaw`` so that ego-forward becomes +x.

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
        Geometry to rotate.
    ego_yaw_rad : float
        Ego yaw angle in radians.

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        Rotated geometry in ego-centric coordinates.
    """
    angle_deg = -np.degrees(ego_yaw_rad)
    return affinity.rotate(geom, angle_deg, origin=(0, 0), use_radians=False)


def _rasterize_polygons(
    geometries: list,
    bev_range: tuple[float, float, float, float],
    canvas_size: tuple[int, int],
    fill_value: int = 1,
) -> np.ndarray:
    """Rasterize Shapely geometries into a (nx, ny) BEV mask using PIL.

    Coordinate mapping (ego-centric):

    - metric x ∈ [x_min, x_max] -> pixel col ∈ [0, nx)
    - metric y ∈ [y_min, y_max] -> pixel row ∈ [0, ny)

    Parameters
    ----------
    geometries : list
        List of Shapely geometries to rasterize.
    bev_range : tuple
        ``(x_min, x_max, y_min, y_max)`` in metres.
    canvas_size : tuple
        ``(nx, ny)`` output grid size.
    fill_value : int, optional
        Value used to fill the interior of polygons.

    Returns
    -------
    np.ndarray
        (nx, ny) array with ``indexing='ij'`` matching ``RayCaster3D``.
    """
    x_min, x_max, y_min, y_max = bev_range
    nx, ny = canvas_size
    scale_x = nx / (x_max - x_min)
    scale_y = ny / (y_max - y_min)

    img = Image.new("L", (nx, ny), 0)
    draw = ImageDraw.Draw(img)

    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        elif geom.geom_type == "GeometryCollection":
            polys = []
            for g in geom.geoms:
                if g.geom_type == "Polygon":
                    polys.append(g)
                elif g.geom_type == "MultiPolygon":
                    polys.extend(list(g.geoms))
        else:
            continue

        for poly in polys:
            if poly.is_empty:
                continue

            def _to_pix(coords):
                """Map (x, y) ego coords to (col, row) in PIL."""
                return [
                    ((x - x_min) * scale_x, (y_max - y) * scale_y)
                    for x, y in coords
                ]

            ext = _to_pix(poly.exterior.coords)
            if len(ext) >= 3:
                draw.polygon(ext, fill=fill_value)
            for interior in poly.interiors:
                hole = _to_pix(interior.coords)
                if len(hole) >= 3:
                    draw.polygon(hole, fill=0)

    # PIL image is (W=nx, H=ny); np.array gives (H=ny, W=nx).
    # Transpose to (nx, ny) with indexing='ij'.
    return np.array(img, dtype=np.uint8).T


def build_drivable_mask(
    nusc,
    sample_token: str,
    bev_range: tuple[float, float, float, float] = cfg.BEV_RANGE_M,
    dilation_m: float = cfg.DEFAULT_DRIVABLE_DILATION_M,
) -> np.ndarray:
    """Build a (nx, ny) bool mask in the ego-centric BEV frame.

    ``True`` means the cell is vehicle-plausible (road/lane, not walkway).
    Falls back to all-``True`` when the HD map is unavailable.

    Parameters
    ----------
    nusc : NuScenes
        nuScenes dataset handle.
    sample_token : str
        Token of the sample to query.
    bev_range : tuple, optional
        ``(x_min, x_max, y_min, y_max)`` in metres.
    dilation_m : float, optional
        Morphological dilation radius in metres. Dilation is applied in cell
        steps using the coarser axis resolution (conservative: dilates at
        least the requested metres along ego-y, slightly less along ego-x).

    Returns
    -------
    np.ndarray
        (nx, ny) bool drivable-area mask.
    """
    nx, ny = bev_grid_shape(bev_range)

    if not MAP_AVAILABLE:
        return np.ones((nx, ny), dtype=bool)

    map_name = get_map_name(nusc, sample_token)
    try:
        nusc_map = get_nusc_map(nusc.dataroot, map_name)
    except Exception as e:
        raise RuntimeError(
            f"NuScenesMap failed for dataroot={nusc.dataroot}, map={map_name}: {e}"
        ) from e

    ego_t, ego_yaw = _get_ego_pose(nusc, sample_token)

    # patch_box is in GLOBAL metres; nuScenes API uses (x, y, H, W).
    half_x = (bev_range[1] - bev_range[0]) / 2.0
    half_y = (bev_range[3] - bev_range[2]) / 2.0
    patch_box = (float(ego_t[0]), float(ego_t[1]), half_x * 2, half_y * 2)

    include_geoms = nusc_map.get_map_geom(
        patch_box, patch_angle=0.0, layer_names=cfg.DRIVABLE_MAP_LAYERS
    )
    exclude_geoms = nusc_map.get_map_geom(
        patch_box, patch_angle=0.0, layer_names=cfg.EXCLUDED_MAP_LAYERS
    )

    all_include, all_exclude = [], []
    for _, geom_list in include_geoms:
        all_include.extend(geom_list)
    for _, geom_list in exclude_geoms:
        all_exclude.extend(geom_list)

    include_ego = [_rotate_to_ego(g, ego_yaw) for g in all_include]
    exclude_ego = [_rotate_to_ego(g, ego_yaw) for g in all_exclude]

    canvas_size = (nx, ny)
    include_mask = _rasterize_polygons(include_ego, bev_range, canvas_size)
    exclude_mask = _rasterize_polygons(exclude_ego, bev_range, canvas_size)

    ego_mask = (include_mask > 0) & ~(exclude_mask > 0)

    if dilation_m > 0:
        # Anisotropic grid: use the coarser axis so the mask dilates by at
        # least the requested metres along ego-y (0.3 m/cell).
        dilation_px = max(1, int(round(dilation_m / max(cfg.BEV_X_RES, cfg.BEV_Y_RES))))
        ego_mask = binary_dilation(ego_mask, iterations=dilation_px)

    return ego_mask.astype(bool)


def filter_osz_by_drivable(
    osz_mask: np.ndarray,
    drivable_mask: np.ndarray,
) -> np.ndarray:
    """Filter an OSZ mask by the drivable area.

    Parameters
    ----------
    osz_mask : np.ndarray
        Geometric occlusion-shadow-zone mask.
    drivable_mask : np.ndarray
        Drivable-area mask of the same shape.

    Returns
    -------
    np.ndarray
        ``geometric_osz ∩ drivable_area``.
    """
    if osz_mask.shape != drivable_mask.shape:
        raise ValueError(
            f"Shape mismatch: osz_mask {osz_mask.shape} vs "
            f"drivable_mask {drivable_mask.shape}. "
            "Both must use the same bev_range (and grid)."
        )
    return osz_mask.astype(bool) & drivable_mask
