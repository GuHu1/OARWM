"""Visualize one nuScenes sample with surround-view cameras and ego-centric HD map.

This script is a standalone entry point. It loads a single nuScenes
sample, renders a 2x3 surround-view camera mosaic, and plots an
ego-centric bird's-eye view of the HD map with drivable area and lane
lines.

Functions
---------
get_camera_images
    Return ``{cam_name: image_path}`` for the six canonical cameras.
plot_surround_view
    Save a 2x3 surround-view mosaic with camera labels.
plot_ego_map
    Plot an ego-centric HD map with drivable area and lane lines.
main
    CLI entry point that orchestrates image and map rendering.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from OSZ import config as cfg
from OSZ.modules.drivable_filter import build_drivable_mask
from OSZ.utils.geometry import ego_pose_from_sample_data, get_map_name


def get_camera_images(nusc: NuScenes, sample_token: str) -> dict[str, Path]:
    """Return ``{cam_name: image_path}`` for the six canonical cameras.

    Parameters
    ----------
    nusc : nuscenes.nuscenes.NuScenes
        Initialized nuScenes dataset instance.
    sample_token : str
        nuScenes sample token.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping from camera name to the corresponding image file path.
    """
    sample = nusc.get("sample", sample_token)
    images = {}
    for cam in cfg.NUSCENES_CAMERAS:
        sd_token = sample["data"][cam]
        sd = nusc.get("sample_data", sd_token)
        images[cam] = Path(nusc.dataroot) / sd["filename"]
    return images


def plot_surround_view(images: dict[str, Path], sample_token: str, save_path: str) -> None:
    """Save a 2x3 surround-view mosaic with camera labels.

    Parameters
    ----------
    images : dict[str, pathlib.Path]
        Mapping from camera name to image file path.
    sample_token : str
        Sample token shown in the figure title.
    save_path : str
        Path where the mosaic PNG will be saved.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Surround View | {sample_token}", fontsize=14, fontweight="bold")

    order = [
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
    ]
    for ax, cam in zip(axes.flat, order):
        img = np.array(Image.open(images[cam]))
        ax.imshow(img)
        ax.set_title(cam.replace("CAM_", ""), fontsize=11)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[saved] {save_path}")
    plt.close(fig)


def plot_ego_map(
    nusc: NuScenes,
    sample_token: str,
    bev_range: tuple[float, float, float, float],
    save_path: str,
) -> None:
    """Plot an ego-centric HD map with drivable area and lane lines.

    Parameters
    ----------
    nusc : nuscenes.nuscenes.NuScenes
        Initialized nuScenes dataset instance.
    sample_token : str
        nuScenes sample token.
    bev_range : tuple[float, float, float, float]
        BEV extent as ``(x_min, x_max, y_min, y_max)`` in metres.
    save_path : str
        Path where the map PNG will be saved.
    """
    import pyquaternion

    sample = nusc.get("sample", sample_token)
    location = get_map_name(nusc, sample_token)

    lidar_sd_token = sample["data"]["LIDAR_TOP"]
    ep = ego_pose_from_sample_data(nusc, lidar_sd_token)
    ego_t = np.array(ep["translation"], dtype=np.float64)
    ego_q = pyquaternion.Quaternion(ep["rotation"])

    nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=location)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    x_min, x_max, y_min, y_max = bev_range
    extent = [y_max, y_min, x_min, x_max]

    drivable_mask = build_drivable_mask(nusc, sample_token, bev_range)
    overlay = np.zeros((*drivable_mask.shape, 3), dtype=np.float32)
    overlay[:] = (0.30, 0.69, 0.31)  # grass
    overlay[drivable_mask] = (0.29, 0.29, 0.29)  # road
    ax.imshow(overlay, origin="lower", extent=extent)

    recs = nusc_map.get_records_in_radius(
        float(ego_t[0]), float(ego_t[1]),
        max(bev_range[1], bev_range[3]) + 10.0,
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
                ax.plot(xs, ys, "-", color="white", linewidth=0.5, alpha=0.4)

    ax.plot(
        0, 0, marker="^", color="#1976d2", markersize=14,
        markeredgecolor="white", markeredgewidth=1.5,
    )

    ax.set_xlim(y_max, y_min)
    ax.set_ylim(x_min, x_max)
    ax.set_xlabel("y (m)  ← ego-left | ego-right →", fontsize=9)
    ax.set_ylabel("x (m)  ↑ forward", fontsize=9)
    ax.set_title(
        f"HD Map (ego-centric) | {sample_token[:16]}... | {location}",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"[saved] {save_path}")
    plt.close(fig)


def main() -> None:
    """Parse CLI arguments and render the surround-view and HD-map figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="/data/sets/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--sample_token", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./osz_viz_fixed")
    parser.add_argument("--bev_range", type=float, nargs=4,
                        default=list(cfg.BEV_RANGE_M))
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    images = get_camera_images(nusc, args.sample_token)

    surround_path = Path(args.outdir) / f"surround_{args.sample_token}.png"
    plot_surround_view(images, args.sample_token, str(surround_path))

    map_path = Path(args.outdir) / f"map_{args.sample_token}.png"
    plot_ego_map(
        nusc, args.sample_token,
        tuple(args.bev_range), str(map_path),
    )


if __name__ == "__main__":
    main()
