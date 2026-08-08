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
    # mock smoke test (no nuScenes needed):
    python OSZ/export_osz_dataset.py --mock --outdir .tmp_osz_export
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
from OSZ.modules.drivable_filter import build_drivable_mask
from OSZ.modules.ray_casting import RayCaster3D, compute_osz_height_aware_from_cameras
from OSZ.run_osz_pipeline import _prepare_cameras, get_estimator
from OSZ.utils.nuscenes_loader import NuScenesOSZLoader

NPZ_KEYS = ("bev_height", "osz_ground", "osz_eye", "semi", "drivable_mask")


def _load_frame(loader, token: str, mock: bool) -> dict:
    if mock:
        # Mock loader has no nuScenes samples; iterate and match by token.
        for frame in loader:
            if frame["sample_token"] == token:
                return frame
        raise KeyError(f"mock frame not found: {token}")
    return loader.build_frame_for_token(token)


def _export_one(args: tuple) -> tuple:
    """Export OSZ masks for one sample token. Returns (token, status)."""
    token, dataroot, version, outdir, use_drivable, use_uncertainty, overwrite, mock = args
    out_path = Path(outdir) / f"{token}.npz"
    if not overwrite and out_path.exists():
        return token, "skip"

    # Mock frames are cheap and generated lazily; ask for a large count so
    # any mock token can be matched (n_mock = max_samples or 3).
    loader = NuScenesOSZLoader(
        dataroot=dataroot, version=version,
        max_samples=1000 if mock else 1,
    )
    if mock:
        loader._use_mock = True
    frame = _load_frame(loader, token, mock)

    caster = RayCaster3D(z_min=cfg.Z_MIN_M, z_max=cfg.Z_MAX_M)
    cameras = _prepare_cameras(frame, simulate_dropout=0.0)
    estimator = get_estimator()
    bev_height, osz_ground, osz_eye = compute_osz_height_aware_from_cameras(
        cameras, caster,
        observer_height=cfg.OBSERVER_HEIGHT_M,
        estimator=estimator,
        use_uncertainty=use_uncertainty,
        z_min=caster.z_min,
        z_max=caster.z_max,
    )

    if use_drivable and not mock:
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
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock:
        loader = NuScenesOSZLoader(max_samples=args.max_samples or 1)
        loader._use_mock = True
        tokens = [f["sample_token"] for f in loader]
    else:
        loader = NuScenesOSZLoader(
            dataroot=args.dataroot, version=args.version,
            max_samples=args.max_samples or None,
        )
        tokens = [s["token"] for s in loader.samples]

    tokens = tokens[args.shard::args.num_shards]
    print(f"[export] {len(tokens)} tokens (shard {args.shard}/{args.num_shards}) "
          f"-> {args.outdir}")

    params = [
        (t, args.dataroot, args.version, args.outdir,
         args.use_drivable, args.use_uncertainty, args.overwrite, args.mock)
        for t in tokens
    ]

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
