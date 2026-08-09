"""Height-Aware Occlusion Shadow Zone (OSZ) pipeline on nuScenes.

Pipeline stages
---------------
1. Load a nuScenes sample (6-camera images, LiDAR sweeps, calibration).
2. Predict per-camera metric depth (Depth Anything V2 + LiDAR alignment).
3. Back-project depth maps to the ego frame and build a BEV height map.
4. Run height-aware 360° ray casting to produce ground OSZ and eye OSZ.
5. Optionally intersect OSZ with the nuScenes HD-map drivable area.
6. Visualize and export per-frame results plus a summary CSV.

Examples
--------
Real nuScenes data with uncertainty fusion and drivable filtering::

    python run_osz_pipeline.py \\
        --dataroot /data/jhc \\
        --version v1.0-mini \\
        --outdir ./osz_output \\
        --use_uncertainty \\
        --use_drivable \\
        --max_samples 10

Single sample by token::

    python run_osz_pipeline.py \\
        --dataroot /data/jhc --sample_token <TOKEN>

Synthetic mock (no nuScenes needed)::

    python run_osz_pipeline.py --mock --outdir ./osz_output
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from OSZ import config as cfg
from OSZ.utils.geometry import bev_grid_shape
from OSZ.utils.nuscenes_loader import NuScenesOSZLoader
from OSZ.modules.depth_estimator import DepthEstimator, MockDepthEstimator
from OSZ.modules.ray_casting import RayCaster3D, compute_osz_height_aware_from_cameras
from OSZ.modules.drivable_filter import build_drivable_mask, filter_osz_by_drivable
from OSZ.visualize.bev_viz import (
    _bev_extent,
    plot_gt_osz,
)


def get_estimator(mock_only: bool = False) -> DepthEstimator | MockDepthEstimator:
    """Create a depth estimator, falling back to mock if loading fails.

    Parameters
    ----------
    mock_only : bool, optional
        If True, return a mock estimator without attempting to load the real
        depth model. The default is False.

    Returns
    -------
    DepthEstimator | MockDepthEstimator
        A ready-to-use depth estimator instance.

    Notes
    -----
    Prints a warning to stdout when falling back to the mock estimator.
    """
    if mock_only:
        print("[INFO] Using MockDepthEstimator (--mock).")
        return MockDepthEstimator()
    try:
        est = DepthEstimator()
        est._load()
        return est
    except Exception as e:
        print(f"[WARN] Could not load real depth model: {e}")
        print("[WARN] Falling back to MockDepthEstimator.")
        return MockDepthEstimator()


def apply_camera_dropout(
    cameras: dict[str, dict], dropout_ratio: float = 0.3, seed: int = 42
) -> None:
    """Zero-out a deterministic vertical stripe in each camera depth map.

    Parameters
    ----------
    cameras : dict[str, dict]
        Mapping from camera name to camera data dictionaries. Each inner dict
        must contain a ``depth_map`` entry with shape (H, W).
    dropout_ratio : float, optional
        Width of the dropped stripe as a ratio of the image width. The default
        is 0.3.
    seed : int, optional
        Random seed for deterministic stripe placement. The default is 42.

    Returns
    -------
    None
        The input ``cameras`` dictionary is modified in place.

    Notes
    -----
    Cameras without a ``depth_map`` are skipped. The stripe width is at least
    one pixel.
    """
    rng = np.random.default_rng(seed)
    for cam_data in cameras.values():
        depth = cam_data.get("depth_map")
        if depth is None:
            continue
        H, W = depth.shape
        stripe_width = max(1, int(W * dropout_ratio))
        start = rng.integers(0, max(1, W - stripe_width))
        depth[:, start:start + stripe_width] = 0.0


def _setup_caster() -> RayCaster3D:
    """Create a 3D ray caster for the height-aware OSZ pipeline.

    The BEV grid and height bounds all come from ``OSZ.config`` (the grid is
    aligned with ResWorld's ``grid_config``).
    """
    return RayCaster3D(
        z_min=cfg.Z_MIN_M,
        z_max=cfg.Z_MAX_M,
    )


def _prepare_cameras(
    frame: dict,
    simulate_dropout: float,
) -> dict[str, dict]:
    """Shallow-copy camera data and optionally apply deterministic dropout.

    Parameters
    ----------
    frame : dict
        Frame dictionary containing a ``cameras`` mapping.
    simulate_dropout : float
        Dropout ratio passed to :func:`apply_camera_dropout`. If zero or
        negative, no dropout is applied.

    Returns
    -------
    dict[str, dict]
        Shallow copy of the camera dictionary, with optional dropout applied
        in place.
    """
    cameras = {k: dict(v) for k, v in frame["cameras"].items()}
    if simulate_dropout > 0:
        apply_camera_dropout(cameras, dropout_ratio=simulate_dropout)
    return cameras


def _compute_height_aware_osz(
    cameras: dict[str, dict],
    caster: RayCaster3D,
    observer_height: float,
    use_uncertainty: bool,
    estimator: DepthEstimator | MockDepthEstimator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the BEV height map and the ground/eye OSZ masks.

    Parameters
    ----------
    cameras : dict[str, dict]
        Camera data dictionary with per-camera ``depth_map`` entries.
    caster : RayCaster3D
        Configured 3D ray caster.
    observer_height : float
        Observer eye height in metres.
    use_uncertainty : bool
        Whether to use inverse-uncertainty weighted camera-LiDAR fusion.
    estimator : DepthEstimator | MockDepthEstimator | None, optional
        Depth estimator used when camera depth maps need to be generated. If
        None, the caller is assumed to have already populated ``depth_map``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(bev_height, osz_ground, osz_eye)`` arrays.
    """
    bev_height, osz_ground, osz_eye = compute_osz_height_aware_from_cameras(
        cameras,
        caster,
        estimator=estimator,
        observer_height=observer_height,
        use_uncertainty=use_uncertainty,
        z_min=caster.z_min,
        z_max=caster.z_max,
    )
    return bev_height, osz_ground, osz_eye


def _apply_drivable_filter(
    osz_ground: np.ndarray,
    osz_eye: np.ndarray,
    drivable_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect OSZ masks with the drivable area and log coverage.

    Parameters
    ----------
    osz_ground : np.ndarray
        Binary ground OSZ mask.
    osz_eye : np.ndarray
        Binary eye-level OSZ mask.
    drivable_mask : np.ndarray | None
        Boolean drivable-area mask, or None to skip filtering.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Filtered ``(osz_ground, osz_eye)`` masks. If ``drivable_mask`` is
        None, the inputs are returned unchanged.
    """
    if drivable_mask is not None:
        osz_ground = filter_osz_by_drivable(osz_ground, drivable_mask)
        osz_eye = filter_osz_by_drivable(osz_eye, drivable_mask)
        coverage = drivable_mask.sum() / drivable_mask.size * 100
        print(f"[drivable] mask covers {drivable_mask.sum()}/{drivable_mask.size} ({coverage:.1f}%)")
    return osz_ground, osz_eye


def _log_height_diagnostics(
    bev_height: np.ndarray,
    observer_height: float,
    ego_xi: int,
    ego_yi: int,
) -> None:
    """Print a BEV height summary and the ego cell value.

    Parameters
    ----------
    bev_height : np.ndarray
        BEV height map.
    observer_height : float
        Observer eye height in metres (used for threshold reporting).
    ego_xi : int
        Ego vehicle grid index along the x axis.
    ego_yi : int
        Ego vehicle grid index along the y axis.

    Returns
    -------
    None
        Output is written to stdout.
    """
    print(f"[height] min={bev_height.min():.2f} max={bev_height.max():.2f} "
          f"mean={bev_height.mean():.2f} occupied={(bev_height > 0.05).sum()}")
    print(f"[height] > 0.3m: {(bev_height > 0.3).sum()}, "
          f"> {observer_height}m: {(bev_height > observer_height).sum()}")
    print(f"[height] ego cell ({ego_xi}, {ego_yi}) = {bev_height[ego_xi, ego_yi]:.2f}")


def _log_depth_diagnostics(cameras: dict[str, dict]) -> None:
    """Print per-camera depth map summary statistics.

    Parameters
    ----------
    cameras : dict[str, dict]
        Camera data dictionary with per-camera ``depth_map`` entries.

    Returns
    -------
    None
        Output is written to stdout.
    """
    for cam_name, cam_data in cameras.items():
        d = cam_data["depth_map"]
        print(f"[depth] {cam_name} min={d.min():.2f} max={d.max():.2f} "
              f"mean={d[d > 0].mean():.2f} valid={(d > 0).sum()}/{d.size}")


def _plot_osz_panels(
    fig: plt.Figure,
    axes: np.ndarray,
    bev_height: np.ndarray,
    osz_ground: np.ndarray,
    osz_eye: np.ndarray,
    semi: np.ndarray,
    caster: RayCaster3D,
    observer_height: float,
    frame: dict,
    drivable_mask: np.ndarray | None,
) -> None:
    """Draw the 6-panel height-aware OSZ visualization.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure to draw on.
    axes : np.ndarray
        2x3 array of matplotlib axes.
    bev_height : np.ndarray
        BEV height map.
    osz_ground : np.ndarray
        Binary ground OSZ mask.
    osz_eye : np.ndarray
        Binary eye-level OSZ mask.
    semi : np.ndarray
        Binary semi-transparent zone mask (ground minus eye).
    caster : RayCaster3D
        Ray caster that produced the OSZ masks (used for extent/resolution).
    observer_height : float
        Observer eye height in metres.
    frame : dict
        Frame dictionary containing the ``sample_token``.
    drivable_mask : np.ndarray | None
        Drivable-area mask, or None if filtering was not applied.

    Returns
    -------
    None
        The figure is modified in place.
    """
    title_filter = " | drivable-filtered" if drivable_mask is not None else ""
    fig.suptitle(
        f"Height-Aware OSZ | token={frame['sample_token']} | "
        f"observer={observer_height}m{title_filter}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    extent, xlim, ylim = _bev_extent(caster.bev_range)

    # Row 0
    ax_h = axes[0, 0]
    im_h = ax_h.imshow(
        bev_height, origin="lower", extent=extent,
        cmap="viridis", vmin=0, vmax=cfg.Z_MAX_M,
    )
    ax_h.set_xlim(*xlim)
    ax_h.set_ylim(*ylim)
    ax_h.set_title("BEV height max", fontsize=11)
    ax_h.set_xlabel("y (m)", fontsize=8)
    ax_h.set_ylabel("x (m)", fontsize=8)
    plt.colorbar(im_h, ax=ax_h, fraction=0.046, pad=0.04, label="height (m)")

    ax_g = axes[0, 1]
    ax_g.imshow(
        bev_height, origin="lower", extent=extent,
        cmap="Greys", vmin=0, vmax=cfg.Z_MAX_M, alpha=0.3,
    )
    ax_g.imshow(
        osz_ground, origin="lower", extent=extent,
        cmap=ListedColormap(["none", "#d32f2f"]), vmin=0, vmax=1, alpha=0.85,
    )
    ax_g.set_xlim(*xlim)
    ax_g.set_ylim(*ylim)
    ax_g.set_title(f"OSZ ground (binary) | {osz_ground.sum()} cells", fontsize=11)
    ax_g.set_xlabel("y (m)", fontsize=8)
    ax_g.set_ylabel("x (m)", fontsize=8)

    ax_e = axes[0, 2]
    ax_e.imshow(
        bev_height, origin="lower", extent=extent,
        cmap="Greys", vmin=0, vmax=cfg.Z_MAX_M, alpha=0.3,
    )
    ax_e.imshow(
        osz_eye, origin="lower", extent=extent,
        cmap=ListedColormap(["none", "#7b1fa2"]), vmin=0, vmax=1, alpha=0.85,
    )
    ax_e.set_xlim(*xlim)
    ax_e.set_ylim(*ylim)
    ax_e.set_title(f"OSZ eye (h > {observer_height}m) | {osz_eye.sum()} cells", fontsize=11)
    ax_e.set_xlabel("y (m)", fontsize=8)
    ax_e.set_ylabel("x (m)", fontsize=8)

    # Row 1
    ax_s = axes[1, 0]
    ax_s.imshow(
        bev_height, origin="lower", extent=extent,
        cmap="Greys", vmin=0, vmax=cfg.Z_MAX_M, alpha=0.3,
    )
    ax_s.imshow(
        semi, origin="lower", extent=extent,
        cmap=ListedColormap(["none", "#ff9800"]), vmin=0, vmax=1, alpha=0.85,
    )
    ax_s.set_xlim(*xlim)
    ax_s.set_ylim(*ylim)
    ax_s.set_title(f"Semi-transparent zone | {semi.sum()} cells", fontsize=11)
    ax_s.set_xlabel("y (m)", fontsize=8)
    ax_s.set_ylabel("x (m)", fontsize=8)

    ax_c = axes[1, 1]
    overlay = np.zeros((*bev_height.shape, 3))
    h_norm = np.clip(bev_height / cfg.Z_MAX_M, 0.0, 1.0)
    overlay[:, :, 0] = h_norm
    overlay[:, :, 1] = h_norm
    overlay[:, :, 2] = h_norm
    ax_c.imshow(overlay, origin="lower", extent=extent)
    ax_c.imshow(
        osz_ground, origin="lower", extent=extent,
        cmap=ListedColormap(["none", "#d32f2f"]), vmin=0, vmax=1, alpha=0.5,
    )
    ax_c.imshow(
        semi, origin="lower", extent=extent,
        cmap=ListedColormap(["none", "#ff9800"]), vmin=0, vmax=1, alpha=0.6,
    )
    ax_c.imshow(
        osz_eye, origin="lower", extent=extent,
        cmap=ListedColormap(["none", "#7b1fa2"]), vmin=0, vmax=1, alpha=0.7,
    )
    ax_c.set_xlim(*xlim)
    ax_c.set_ylim(*ylim)
    ax_c.set_title("Combined: red=ground, orange=semi, purple=eye", fontsize=11)
    ax_c.set_xlabel("y (m)", fontsize=8)
    ax_c.set_ylabel("x (m)", fontsize=8)

    ax_stats = axes[1, 2]
    ax_stats.axis("off")
    total = bev_height.size
    stats_text = (
        f"BEV grid: {caster.nx} x {caster.ny} = {total} cells\n"
        f"BEV resolution: {caster.bev_res_x} x {caster.bev_res_y} m/cell\n"
        f"Observer height: {observer_height} m\n\n"
        f"Occupied cells: {(bev_height > 0).sum()} "
        f"({(bev_height > 0).sum() / total * 100:.1f}%)\n"
        f"OSZ ground cells: {osz_ground.sum()} "
        f"({osz_ground.sum() / total * 100:.1f}%)\n"
        f"OSZ eye cells: {osz_eye.sum()} "
        f"({osz_eye.sum() / total * 100:.1f}%)\n"
        f"Semi-transparent cells: {semi.sum()} "
        f"({semi.sum() / total * 100:.1f}%)\n\n"
        f"Eye reduction vs ground:\n"
        f"  {osz_eye.sum() / max(osz_ground.sum(), 1) * 100:.1f}% of ground shadow\n"
        f"  diff = {osz_ground.sum() - osz_eye.sum()} cells"
    )
    ax_stats.text(
        0.1, 0.5, stats_text, transform=ax_stats.transAxes,
        fontsize=11, verticalalignment="center", family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )


def _build_stats_dict(
    frame: dict,
    bev_height: np.ndarray,
    osz_ground: np.ndarray,
    osz_eye: np.ndarray,
    semi: np.ndarray,
    caster: RayCaster3D,
) -> dict:
    """Assemble the per-frame statistics dictionary.

    Parameters
    ----------
    frame : dict
        Frame dictionary containing the ``sample_token``.
    bev_height : np.ndarray
        BEV height map.
    osz_ground : np.ndarray
        Binary ground OSZ mask.
    osz_eye : np.ndarray
        Binary eye-level OSZ mask.
    semi : np.ndarray
        Binary semi-transparent zone mask.
    caster : RayCaster3D
        Ray caster that produced the OSZ masks (used for grid dimensions).

    Returns
    -------
    dict
        Dictionary with sample_token, occupied, osz_ground, osz_eye, semi,
        osz_ground_ratio, osz_eye_ratio, bev_height, and bev_occ entries.
    """
    total = bev_height.size
    bev_occ = bev_height > 0.05
    return {
        "sample_token": frame["sample_token"],
        "occupied": int(bev_occ.sum()),
        "osz_ground": int(osz_ground.sum()),
        "osz_eye": int(osz_eye.sum()),
        "semi": int(semi.sum()),
        "osz_ground_ratio": float(osz_ground.sum()) / max(total, 1),
        "osz_eye_ratio": float(osz_eye.sum()) / max(total, 1),
        "osz_ground_mask": osz_ground,
        "osz_eye_mask": osz_eye,
        "bev_height": bev_height,
        "bev_occ": bev_occ,
    }


def visualize_height_aware_osz(
    frame: dict,
    save_path: str,
    observer_height: float = cfg.OBSERVER_HEIGHT_M,
    estimator: DepthEstimator | MockDepthEstimator | None = None,
    simulate_dropout: float = 0.0,
    use_uncertainty: bool = False,
    drivable_mask: np.ndarray | None = None,
) -> dict:
    """Run height-aware OSZ on a single frame and save a 6-panel visualization.

    The BEV grid is fixed by ``OSZ.config`` (aligned with ResWorld's
    ``grid_config``).

    Parameters
    ----------
    frame : dict
        Frame dictionary from :class:`NuScenesOSZLoader` containing cameras,
        calibration, and ``sample_token``.
    save_path : str
        File path where the PNG visualization will be written.
    observer_height : float, optional
        Observer eye height in metres. The default is taken from
        ``cfg.OBSERVER_HEIGHT_M``.
    estimator : DepthEstimator | MockDepthEstimator | None, optional
        Depth estimator. If None, a default estimator is created on demand.
    simulate_dropout : float, optional
        Camera dropout ratio for robustness testing. The default is 0.0.
    use_uncertainty : bool, optional
        Use inverse-uncertainty weighted fusion. The default is False.
    drivable_mask : np.ndarray | None, optional
        Boolean drivable-area mask. If provided, OSZ is intersected with it.

    Returns
    -------
    dict
        Per-frame statistics with keys: ``sample_token``, ``occupied``,
        ``osz_ground``, ``osz_eye``, ``semi``, ``osz_ground_ratio``,
        ``osz_eye_ratio``, ``bev_height``, ``bev_occ``.
    """
    caster = _setup_caster()
    cameras = _prepare_cameras(frame, simulate_dropout)

    if estimator is None:
        estimator = get_estimator()

    bev_height, osz_ground, osz_eye = _compute_height_aware_osz(
        cameras,
        caster,
        observer_height,
        use_uncertainty,
        estimator=estimator,
    )

    osz_ground, osz_eye = _apply_drivable_filter(
        osz_ground, osz_eye, drivable_mask
    )

    ego_xi = int(np.floor((0.0 - caster.bev_range[0]) / caster.bev_res_x))
    ego_yi = int(np.floor((caster.bev_range[3] - 0.0) / caster.bev_res_y))
    _log_height_diagnostics(bev_height, observer_height, ego_xi, ego_yi)
    _log_depth_diagnostics(cameras)

    semi = osz_ground & ~osz_eye

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    _plot_osz_panels(
        fig,
        axes,
        bev_height,
        osz_ground,
        osz_eye,
        semi,
        caster,
        observer_height,
        frame,
        drivable_mask,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[saved] {save_path}")
    plt.close(fig)

    stats = _build_stats_dict(frame, bev_height, osz_ground, osz_eye, semi, caster)

    if stats["osz_ground_ratio"] > 0.8:
        print(f"[WARN] RAW OSZ anomaly: ground OSZ covers "
              f"{stats['osz_ground_ratio'] * 100:.1f}% of BEV")

    return stats


def build_args() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the pipeline.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser ready for :func:`parse_args`.
    """
    parser = argparse.ArgumentParser(
        description="Height-Aware OSZ Pipeline on nuScenes"
    )
    parser.add_argument("--dataroot", type=str, default="/data/sets/nuscenes",
                        help="nuScenes data root directory.")
    parser.add_argument("--version", type=str, default="v1.0-mini",
                        help="nuScenes dataset version.")
    parser.add_argument("--sample_token", type=str, default=None,
                        help="Process a single sample by token.")
    parser.add_argument("--mock", action="store_true",
                        help="Use synthetic mock data (no nuScenes needed).")
    parser.add_argument("--outdir", type=str, default="./osz_output",
                        help="Output directory for PNG and CSV files.")
    parser.add_argument("--observer_height", type=float, default=cfg.OBSERVER_HEIGHT_M,
                        help="Observer eye height in metres.")
    parser.add_argument("--max_samples", type=int, default=1,
                        help="Number of samples to process.")
    parser.add_argument("--n_sweeps", type=int, default=0,
                        help="Past LiDAR sweeps to aggregate (0 = keyframe only).")
    parser.add_argument("--simulate_dropout", type=float, default=0.0,
                        help="Zero-out a vertical stripe of this width ratio "
                             "in camera depth maps to test LiDAR fallback.")
    parser.add_argument("--use_uncertainty", action="store_true",
                        help="Use inverse-uncertainty weighted camera-LiDAR fusion.")
    parser.add_argument("--use_drivable", action="store_true",
                        help="Intersect OSZ with nuScenes drivable area mask.")
    return parser


def main() -> None:
    """Run the height-aware OSZ pipeline from the command line.

    Notes
    -----
    Parses command-line arguments, loads nuScenes data (or mock data),
    optionally builds a drivable-area mask, runs :func:`visualize_height_aware_osz`
    for each selected sample, writes per-frame PNGs, and exports a summary CSV.
    """
    parser = build_args()
    args = parser.parse_args()

    loader = NuScenesOSZLoader(
        dataroot=args.dataroot,
        version=args.version,
        max_samples=args.max_samples,
        n_sweeps=args.n_sweeps,
    )
    if args.mock:
        loader._use_mock = True

    suffix = f"_sweeps{args.n_sweeps}"
    if args.simulate_dropout > 0:
        suffix += "_dropout"
    if args.use_uncertainty:
        suffix += "_uncertainty"
    if args.use_drivable:
        suffix += "_drivable"

    estimator = get_estimator(mock_only=args.mock)
    all_stats = []

    bev_range = cfg.BEV_RANGE_M

    if args.sample_token:
        frames = [loader.build_frame_for_token(args.sample_token)]
    else:
        frames = list(loader)

    for frame in frames:
        token = frame["sample_token"]
        print(f"\nLoaded sample: {token}")

        drivable_mask = None
        if args.use_drivable:
            if args.mock:
                nx, ny = bev_grid_shape(bev_range)
                drivable_mask = np.ones((nx, ny), dtype=bool)
                print("[INFO] mock mode: drivable mask is all-True.")
            else:
                drivable_mask = build_drivable_mask(
                    loader.nusc, token, bev_range
                )

        save_path = Path(args.outdir) / f"osz_{token}{suffix}.png"
        stats = visualize_height_aware_osz(
            frame,
            str(save_path),
            observer_height=args.observer_height,
            estimator=estimator,
            simulate_dropout=args.simulate_dropout,
            use_uncertainty=args.use_uncertainty,
            drivable_mask=drivable_mask,
        )

        # 6-camera image panel (raw RGB for cross-checking against BEV OSZ).
        try:
            cam_save = Path(args.outdir) / f"cameras_{token}{suffix}.png"
            fig_cam, axes_cam = plt.subplots(2, 3, figsize=(15, 8))
            axes_cam = np.array(axes_cam).ravel()
            for ax, (cam_name, cam_data) in zip(axes_cam, frame["cameras"].items()):
                ax.imshow(cam_data["image"])
                ax.set_title(cam_name, fontsize=10)
                ax.axis("off")
            for ax in axes_cam[len(frame["cameras"]):]:
                ax.axis("off")
            fig_cam.suptitle(
                f"Cameras — {token}", fontsize=14, fontweight="bold", y=0.98
            )
            fig_cam.tight_layout(rect=[0, 0, 1, 0.96])
            fig_cam.savefig(cam_save, dpi=120, bbox_inches="tight")
            plt.close(fig_cam)
            print(f"  [saved] {cam_save}")
        except Exception as e:
            print(f"[WARN] camera panel viz failed: {e}")

        # GT overlay: generate when drivable mask is available and not mock.
        if drivable_mask is not None and not args.mock and hasattr(loader, "nusc"):
            try:
                gt_save_path = Path(args.outdir) / f"gt_osz_{token}{suffix}.png"
                fig_gt = plot_gt_osz(
                    osz_pa=stats["osz_ground_mask"],
                    bev_occ=stats["bev_occ"],
                    drivable_mask=drivable_mask,
                    nusc=loader.nusc,
                    sample_token=token,
                    bev_range=bev_range,
                    save_path=str(gt_save_path),
                )
                plt.close(fig_gt)
            except Exception as e:
                print(f"[WARN] GT visualization failed: {e}")

        all_stats.append({
            k: v for k, v in stats.items()
            if k not in ("bev_height", "bev_occ", "osz_ground_mask", "osz_eye_mask")
        })

    # Summary
    if all_stats:
        print("\n" + "=" * 90)
        filter_note = " | drivable-filtered" if args.use_drivable else ""
        print(f"Summary across samples (n_sweeps={args.n_sweeps}){filter_note}")
        print("=" * 90)
        print(f"{'token':<36} {'occ':>8} {'ground':>9} {'eye':>9} "
              f"{'semi':>9} {'g%':>7} {'e%':>7}")
        for s in all_stats:
            print(f"{s['sample_token']:<36} {s['occupied']:>8} "
                  f"{s['osz_ground']:>9} {s['osz_eye']:>9} {s['semi']:>9} "
                  f"{s['osz_ground_ratio'] * 100:>7.1f} "
                  f"{s['osz_eye_ratio'] * 100:>7.1f}")
        print("-" * 90)
        n = len(all_stats)
        print(f"{'mean':<36} "
              f"{sum(s['occupied'] for s in all_stats) / n:>8.1f} "
              f"{sum(s['osz_ground'] for s in all_stats) / n:>9.1f} "
              f"{sum(s['osz_eye'] for s in all_stats) / n:>9.1f} "
              f"{sum(s['semi'] for s in all_stats) / n:>9.1f} "
              f"{sum(s['osz_ground_ratio'] for s in all_stats) / n * 100:>7.1f} "
              f"{sum(s['osz_eye_ratio'] for s in all_stats) / n * 100:>7.1f}")

        csv_path = Path(args.outdir) / f"summary{suffix}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_token", "occupied", "osz_ground",
                    "osz_eye", "semi", "osz_ground_ratio", "osz_eye_ratio",
                ],
            )
            writer.writeheader()
            writer.writerows(all_stats)
        print(f"[saved] {csv_path}")

        n_anomaly = sum(1 for s in all_stats if s["osz_ground_ratio"] > 0.8)
        if n_anomaly:
            print(f"[WARN] {n_anomaly}/{n} samples have ground OSZ > 80% "
                  f"(potential RAW OSZ bug)")
        else:
            print(f"[OK] No RAW OSZ anomaly detected "
                  f"(ground OSZ <= 80% in all {n} samples)")


if __name__ == "__main__":
    main()
