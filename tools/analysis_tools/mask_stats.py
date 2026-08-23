"""Offline OSZ mask statistics: quantify occlusion over-exposure.

Prints, per sample and in aggregate, the raw occlusion fraction and the
fraction after intersecting with the npz's ``drivable_mask`` channel —
the probe for ISSUE.md P1-3 (occ_frac ~0.6-0.8 is the top suspect for the
persistent L2 gap vs baseline). The training-time occ_frac must reflect
THIS drivable-intersected number; if it stays high here, the over-exposure
is geometric (ray casting / depth thresholds), not off-road clutter.

Usage (server, resworld env):

    python tools/analysis_tools/mask_stats.py --osz-dir data/osz \
        [--max-samples 200] [--occ-threshold 0.2]

The script needs only numpy (no torch/mmdet): it reads the raw npz, not the
training pipeline.
"""
import argparse
import glob
import os

import numpy as np

THRESHOLD = 0.05  # same as the ray caster's "occupied" threshold


def mask_frac(m: np.ndarray) -> float:
    """Fraction of cells >0 for a bool/float (H, W) map."""
    if m.dtype == bool:
        return float(m.mean())
    return float((m > THRESHOLD).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osz-dir", default="data/osz",
                    help="npz directory from OSZ/export_osz_dataset.py")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--occ-threshold", type=float, default=None,
                    help="report the fraction of samples whose DRIVABLE "
                         "occ_frac exceeds this (default: 0.2)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.osz_dir, "*.npz")))
    if args.max_samples > 0:
        files = files[:args.max_samples]
    if not files:
        raise SystemExit(f"no npz under {args.osz_dir}")

    raw_eye, raw_gnd, drv_eye, drv_gnd = [], [], [], []
    has_drivable = 0
    for f in files:
        with np.load(f) as d:
            eye = d["osz_eye"]
            gnd = d["osz_ground"]
            drv = None if "drivable_mask" not in d else d["drivable_mask"]
        raw_eye.append(mask_frac(eye))
        raw_gnd.append(mask_frac(gnd))
        if drv is not None:
            has_drivable += 1
            drv_eye.append(mask_frac(eye & drv))
            drv_gnd.append(mask_frac(gnd & drv))

    def agg(name, vals):
        v = np.asarray(vals)
        return (f"{name:>14}: n={len(v):4d} min={v.min():.4f} "
                f"max={v.max():.4f} avg={v.mean():.4f}")

    print(f"npz   : {len(files)}  (drivable channel in {has_drivable})")
    print(agg("osz_eye raw   ", raw_eye))
    print(agg("osz_ground raw", raw_gnd))
    if drv_eye:
        print(agg("osz_eye ∩driv ", drv_eye))
    if drv_gnd:
        print(agg("osz_ground∩drv", drv_gnd))

    if args.occ_threshold is not None and drv_eye:
        over = sum(1 for x in drv_eye if x > args.occ_threshold)
        print(f"\nfraction of samples with drivable occ_frac > "
              f"{args.occ_threshold}: {over}/{len(drv_eye)} "
              f"({over / len(drv_eye) * 100:.1f}%)")

    if drv_eye and has_drivable == len(files):
        frac = (sum(drv_eye) / len(drv_eye)
                / max(1e-9, sum(raw_eye) / len(raw_eye)))
        print(f"\ndrivable keeps {frac * 100:.1f}% of the raw eye"
              f" occlusion cells")


if __name__ == "__main__":
    main()
