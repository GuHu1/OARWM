"""Batch-export per-frame OSZ masks as ``{sample_token}.npz`` for ResWorld.

For each nuScenes sample (or a shard of them) this script runs the
height-aware OSZ pipeline and writes:

    {outdir}/{sample_token}.npz
        bev_height      (200, 200) float32  max height per BEV cell
        osz_ground      (200, 200) bool     ground-layer occlusion shadow
        osz_eye         (200, 200) bool     strict (eye-level) shadow
        semi            (200, 200) bool     ground & ~eye ("translucent")
        drivable_mask   (200, 200) bool     drivable-area filter (all-True
                                            when the HD map is unavailable)

The masks are grid-aligned with ResWorld (see ``OSZ/config.py``), so the
ResWorld dataset loader can inject them directly without resampling.

Design (Karpathy rule): one obvious way to run. Single process by default;
``--num_workers`` fans out to a process pool (each worker builds its own
loader). ``--shard/--num_shards`` lets you parallelise further with plain
shell loops. Already-exported tokens are skipped unless ``--overwrite``.

Examples
--------
    python OSZ/export_osz_dataset.py --dataroot /data/nuscenes \\
        --version v1.0-trainval --outdir data/osz --use_drivable
    # 8 shell shards (no extra deps):
    for i in $(seq 0 7); do
        python OSZ/export_osz_dataset.py --dataroot /data/nuscenes \\
            --outdir data/osz --use_drivable --shard $i --num_shards 8 &
    done
    # depth-source ablation:
    #   midas   (default): MiDaS v2.1 Small + LiDAR scale alignment
    #   rcsample          : ResWorld's own RCSample depth head
    #                       (mmdet3d env + --rcsample-ckpt, e.g. the trained
    #                       EMA weights work_dirs/oa_resworld/
    #                       epoch_12_ema.pth — export AFTER training)
    #   lidar             : LiDAR densified depth (upper bound)
    python OSZ/export_osz_dataset.py --dataroot data/nuscenes \\
        --version v1.0-trainval --outdir data/osz_rcsample \\
        --depth-source rcsample \\
        --rcsample-ckpt work_dirs/oa_resworld/epoch_12_ema.pth
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from OSZ import config as cfg
from OSZ.modules.depth_estimator import DepthEstimator, MockDepthEstimator
from OSZ.modules.drivable_filter import build_drivable_mask
from OSZ.modules.ray_casting import RayCaster3D, compute_osz_height_aware_from_cameras
from OSZ.modules.rcsample_depth_estimator import (
    DEFAULT_CONFIG_PATH,
    RCSampleDepthEstimator,
)
from OSZ.run_osz_pipeline import _prepare_cameras
from OSZ.utils.nuscenes_loader import NuScenesOSZLoader

from OSZ.modules.torch_pipeline import (  # noqa: E402
    compute_osz_height_aware_from_cameras_torch,
)

NPZ_KEYS = ("bev_height", "osz_ground", "osz_eye", "semi", "drivable_mask")

#: Per-process estimator cache: building a ResWorld / MiDaS model per frame
#: is wasteful. Keyed by (source, config, ckpt) so each worker process builds
#: each depth source at most once.
_ESTIMATOR_CACHE = {}


def _torch_backend_available() -> bool:
    """True when the torch backend can actually run (torch importable)."""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _get_estimator(
    depth_source: str,
    rcsample_config: str,
    rcsample_ckpt: str,
):
    """Return (and cache) the depth estimator for a depth source.

    Sources
    -------
    ``midas``    : MiDaS v2.1 Small + LiDAR scale alignment (default).
    ``lidar``    : LiDAR densified depth (``MockDepthEstimator``, upper bound).
    ``rcsample`` : the ResWorld model's own RCSample depth head (requires the
                   mmdet3d environment and ``--rcsample-ckpt``).
    """
    key = (depth_source, rcsample_config, rcsample_ckpt)
    if key in _ESTIMATOR_CACHE:
        return _ESTIMATOR_CACHE[key]

    if depth_source == "lidar":
        print("[INFO] depth source: LiDAR densified (upper bound).")
        est = MockDepthEstimator()
    elif depth_source == "midas":
        try:
            est = DepthEstimator()
            est._load()
        except Exception as e:
            # A silent fallback would mix two depth sources in the same
            # ablation output dir — fail loudly instead.
            raise RuntimeError(
                f"MiDaS depth model unavailable (--depth-source midas): "
                f"{e}. Fix the model/repo, or choose another source "
                f"(--depth-source lidar / rcsample)."
            ) from e
    elif depth_source == "rcsample":
        est = RCSampleDepthEstimator(rcsample_config, rcsample_ckpt)
        try:
            est._load()
        except Exception as e:
            raise RuntimeError(
                f"RCSample depth source unavailable: {e}. Make sure "
                f"--rcsample-ckpt points to a trained ResWorld checkpoint."
            ) from e
    else:
        raise ValueError(f"unknown --depth-source: {depth_source!r}")

    _ESTIMATOR_CACHE[key] = est
    return est


def _load_frame(loader, token: str) -> dict:
    return loader.build_frame_for_token(token)


def _export_one(args: dict) -> tuple:
    """Export OSZ masks for one sample token. Returns (token, status)."""
    token = args["token"]
    outdir = args["outdir"]
    overwrite = args["overwrite"]
    out_path = Path(outdir) / f"{token}.npz"
    if not overwrite and out_path.exists():
        return token, "skip"

    loader = NuScenesOSZLoader(
        dataroot=args["dataroot"], version=args["version"],
        max_samples=1,
    )
    frame = _load_frame(loader, token)

    caster = RayCaster3D(z_min=cfg.Z_MIN_M, z_max=cfg.Z_MAX_M)
    cameras = _prepare_cameras(frame, simulate_dropout=0.0)
    estimator = _get_estimator(
        args["depth_source"], args["rcsample_config"],
        args["rcsample_ckpt"],
    )

    # Backend selection: torch (GPU geometry) or numpy (reference).
    backend = args["backend"]
    if backend == "torch":
        bev_height, osz_ground, osz_eye = \
            compute_osz_height_aware_from_cameras_torch(
                cameras, caster,
                observer_height=cfg.OBSERVER_HEIGHT_M,
                estimator=estimator,
                use_uncertainty=args["use_uncertainty"],
                z_min=caster.z_min,
                z_max=caster.z_max,
            )
    else:
        bev_height, osz_ground, osz_eye = \
            compute_osz_height_aware_from_cameras(
                cameras, caster,
                observer_height=cfg.OBSERVER_HEIGHT_M,
                estimator=estimator,
                use_uncertainty=args["use_uncertainty"],
                z_min=caster.z_min,
                z_max=caster.z_max,
            )

    if args["use_drivable"]:
        drivable = build_drivable_mask(loader.nusc, token, cfg.BEV_RANGE_M)
    else:
        drivable = np.ones((cfg.BEV_NX, cfg.BEV_NY), dtype=bool)

    Path(outdir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        bev_height=bev_height,
        osz_ground=osz_ground,
        osz_eye=osz_eye,
        semi=osz_ground & ~osz_eye,
        drivable_mask=drivable,
    )
    return token, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-export OSZ masks as {token}.npz for ResWorld"
    )
    parser.add_argument("--dataroot", type=str, default="/data/sets/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--outdir", type=str, default="data/osz")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="0 = all samples.")
    parser.add_argument("--use_drivable", action="store_true")
    parser.add_argument("--use_uncertainty", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--depth-source", type=str, default="midas",
        choices=["midas", "rcsample", "lidar"],
        help="Which depth feeds the OSZ pipeline. midas=MiDaS+LiDAR scale "
             "alignment (default); rcsample=the ResWorld model's own RCSample "
             "depth head (needs the mmdet3d env + --rcsample-ckpt); "
             "lidar=LiDAR densified depth (upper bound).")
    parser.add_argument(
        "--rcsample-config", type=str, default=DEFAULT_CONFIG_PATH,
        help="ResWorld config used to build the RCSample depth head.")
    parser.add_argument(
        "--rcsample-ckpt", type=str, default="",
        help="Trained ResWorld checkpoint (required for --depth-source "
             "rcsample), e.g. work_dirs/oa_resworld/epoch_12_ema.pth "
             "(EMA weights, export AFTER training).")
    parser.add_argument(
        "--backend", type=str, default="auto",
        choices=["numpy", "torch", "auto"],
        help="Geometry backend. auto=torch when a GPU is available, else "
             "numpy. torch keeps depth+geometry on the device (no per-frame "
             "CPU round-trips); numpy is the reference implementation. "
             "--use_uncertainty is supported on both backends.")
    args = parser.parse_args()

    if args.backend == "auto":
        args.backend = "torch" if _torch_backend_available() else "numpy"
        print(f"[INFO] --backend auto -> {args.backend}")

    loader = NuScenesOSZLoader(
        dataroot=args.dataroot, version=args.version,
        max_samples=args.max_samples or None,
    )
    tokens = [s["token"] for s in loader.samples]

    tokens = tokens[args.shard::args.num_shards]
    print(f"[export] {len(tokens)} tokens (shard {args.shard}/{args.num_shards}) "
          f"-> {args.outdir}")

    params = [
        dict(token=t, dataroot=args.dataroot, version=args.version,
             outdir=args.outdir, use_drivable=args.use_drivable,
             use_uncertainty=args.use_uncertainty,
             overwrite=args.overwrite,
             depth_source=args.depth_source,
             rcsample_config=args.rcsample_config,
             rcsample_ckpt=args.rcsample_ckpt,
             backend=args.backend)
        for t in tokens
    ]
    if args.depth_source == "rcsample" and args.num_workers > 1:
        print("[INFO] --depth-source rcsample: each worker builds its own "
              "ResWorld copy; prefer --num_workers 1 to bound GPU memory.")
    if args.backend == "torch" and args.num_workers > 1:
        print("[INFO] --backend torch: each worker uses the same GPU; "
              "prefer --num_workers 1 unless the GPU has spare capacity.")

    stats = {"ok": 0, "skip": 0, "err": 0}
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            results = pool.map(_export_one, params)
            for token, status in results:
                stats[status if status in stats else "err"] += 1
                print(f"  {status:4s} {token}")
    else:
        for p in params:
            token, status = _export_one(p)
            stats[status if status in stats else "err"] += 1
            print(f"  {status:4s} {token}")

    print(f"[export] done: {stats}")


if __name__ == "__main__":
    main()
