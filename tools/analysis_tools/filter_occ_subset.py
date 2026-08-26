"""Occlusion-subset evaluation: token filtering + L2 re-scoring (V2, design doc §5.1).

Evaluates the planner metrics on the occlusion-heavy subset (OSZ occ_frac
above a threshold) — the "does the model perceive ghost-probe risk" evidence.
Two functions:

1. ``--out-json`` : writes the subset token list (used by the full-re-eval
   path and by traj_behavior_stats.py).
2. re-scores plan L2 (ADE) and L2_stp3 (FDE) on the subset from the dumped
   ``pts_bbox/results_nusc.pkl`` (same math as
   metric_stp3.py::compute_L2 / compute_L2_stp3: 1s = first 2 steps,
   2s = first 4, 3s = all 6; stp3 = end-point error).

Notes:
* The subset mask uses the OFFLINE midas npz (data/osz) — the main config
  trains with online rcsample masks, so the two are correlated but not
  identical; the midas mask is the stable reference.
* Collision rate is NOT re-scored here (GT boxes are not in the result pkl);
  run the full evaluation with the filtered dataset for CR, or extend the
  script with the val infos' gt boxes.
* The result pkl layout (nuscenes_vad_dataset.format_results):
    plan_results[token] = [ego_fut_preds (M,T,2) increments, ego_fut_cmd]
    plan_gts[token]     = gt_ego_fut_trajs (T,2) increments

Usage (from the repo root):
    python tools/analysis_tools/filter_occ_subset.py \
        --result-pkl test/oa_resworld_config/<ts>/pts_bbox/results_nusc.pkl \
        --out-json work_dirs/occ_subset_tokens.json
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
import mmcv

L2_STEPS = {1: 2, 2: 4, 3: 6}   # 0.5 s per step; 1s/2s/3s horizons


def load_plan_pkl(pkl_path):
    """Return the TOKEN-KEYED plan pkl ``{'plan_results', 'plan_gts'}``.

    Two pkl layouts exist under each eval timestamp dir (both written by
    nuscenes_vad_dataset.py):

      * ``<ts>/pts_bbox/results_nusc.pkl`` — token-keyed dict written by
        ``_format_bbox`` (what this script needs);
      * ``<ts>/results_nusc_all.pkl``      — the raw per-sample LIST dumped
        by ``format_results`` (NO token keys — unusable for subsetting).

    If handed the list dump (or any dict without ``plan_results``), fall
    back to the sibling ``pts_bbox/results_nusc.pkl``.
    """
    data = mmcv.load(pkl_path)
    if isinstance(data, dict) and 'plan_results' in data:
        return data
    cand = osp.join(osp.dirname(pkl_path), 'pts_bbox', 'results_nusc.pkl')
    if osp.exists(cand):
        alt = mmcv.load(cand)
        if isinstance(alt, dict) and 'plan_results' in alt:
            print(f'note: {pkl_path} carries no token keys; using {cand}')
            return alt
    raise SystemExit(
        f"'{pkl_path}' is not a token-keyed plan pkl and '{cand}' does not "
        f"exist either. List the eval dir to locate "
        f"pts_bbox/results_nusc.pkl:\n"
        f"  find {osp.dirname(pkl_path) or 'test/'} -name '*.pkl'")


def occ_frac_of(npz_path):
    with np.load(npz_path, allow_pickle=True) as d:
        eye = d.get('osz_eye')
        if eye is None:
            eye = d.get('osz_ground')
    return float(np.asarray(eye).mean())


def collect_subset(osz_dir, thresh, tokens_in_pkl):
    subset = []
    skipped = 0
    for name in sorted(os.listdir(osz_dir)):
        if not name.endswith('.npz'):
            continue
        token = name[:-4]
        if token not in tokens_in_pkl:
            skipped += 1
            continue
        if occ_frac_of(os.path.join(osz_dir, name)) >= thresh:
            subset.append(token)
    print(f'subset: {len(subset)} tokens (occ_frac >= {thresh}); '
          f'{skipped} npz without result entry skipped')
    return subset


def rescore_l2(subset, pkl_path):
    data = load_plan_pkl(pkl_path)
    plan_results = data['plan_results']
    plan_gts = data['plan_gts']

    sums = {h: 0.0 for h in L2_STEPS}
    sums_stp3 = {h: 0.0 for h in L2_STEPS}
    n = 0
    for token in subset:
        pred_inc, cmd = plan_results[token]          # (M,T,2) tensor, cmd tensor
        gt_inc = plan_gts[token]                     # (T,2) increments
        cmd_flat = np.asarray(cmd.cpu() if hasattr(cmd, 'cpu') else cmd)
        idx = int(np.flatnonzero(cmd_flat.reshape(-1))[0])
        pred_inc = np.asarray(
            pred_inc.cpu() if hasattr(pred_inc, 'cpu') else pred_inc)
        pred = pred_inc[idx].cumsum(axis=0)          # (T,2) absolute
        gt = np.asarray(gt_inc.cpu() if hasattr(gt_inc, 'cpu') else gt_inc)
        gt = gt.cumsum(axis=0)                       # (T,2) absolute
        for h, steps in L2_STEPS.items():
            err = pred[:steps] - gt[:steps]
            ade = float(np.mean(np.sqrt((err ** 2).sum(axis=1))))
            fde = float(np.sqrt((err[-1] ** 2).sum()))
            sums[h] += ade
            sums_stp3[h] += fde
        n += 1

    if n == 0:
        print('empty subset — nothing to score')
        return
    print(f'--- occlusion subset (n={n}) ---')
    for h in L2_STEPS:
        print(f'plan_L2_{h}s: {sums[h] / n:.6f}')
    for h in L2_STEPS:
        print(f'plan_L2_stp3_{h}s: {sums_stp3[h] / n:.6f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-pkl', required=True,
                        help='pts_bbox/results_nusc.pkl from dist_test')
    parser.add_argument('--osz-dir', default='data/osz',
                        help='offline midas OSZ npz dir')
    parser.add_argument('--occ-thresh', type=float, default=0.2,
                        help='occluded-cell fraction threshold')
    parser.add_argument('--out-json', default='work_dirs/occ_subset_tokens.json',
                        help='where to save the subset token list')
    args = parser.parse_args()

    data = mmcv.load(args.result_pkl)
    # tokens must come from the TOKEN-KEYED pkl; the list dump has none.
    if not (isinstance(data, dict) and 'plan_results' in data):
        data = load_plan_pkl(args.result_pkl)
    tokens_in_pkl = set(data['plan_results'].keys())
    subset = collect_subset(args.osz_dir, args.occ_thresh, tokens_in_pkl)

    mmcv.mkdir_or_exist(os.path.dirname(args.out_json) or '.')
    with open(args.out_json, 'w') as f:
        json.dump({'occ_thresh': args.occ_thresh, 'tokens': subset}, f)
    print('token list written to', args.out_json)

    rescore_l2(subset, args.result_pkl)


if __name__ == '__main__':
    main()
