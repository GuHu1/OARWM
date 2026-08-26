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
METRIC_KEYS = ['plan_L2_1s', 'plan_L2_2s', 'plan_L2_3s',
               'plan_L2_stp3_1s', 'plan_L2_stp3_2s', 'plan_L2_stp3_3s']
# Optional ann-file cross-check for the token order (set by --ann-file).
RESCORE_ANN_TOKENS = None


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


def occ_frac_of(npz_path, use_drivable=True):
    """Occluded-cell fraction, by default intersected with the npz's
    ``drivable_mask`` channel — the SAME convention the training pipeline
    applies (use_osz_drivable=True) and that fixed the P1-3 over-exposure.
    The RAW osz_eye fraction saturates near 1.0 for almost every sample
    (6004/6019 val samples >= 0.2), so it cannot discriminate a subset."""
    with np.load(npz_path, allow_pickle=True) as d:
        eye = d.get('osz_eye')
        if eye is None:
            eye = d.get('osz_ground')
        eye = np.asarray(eye).astype(np.float32)
        if use_drivable and 'drivable_mask' in d:
            eye = eye * d['drivable_mask'].astype(np.float32)
    return float(eye.mean())


def collect_subset(osz_dir, thresh, tokens_in_pkl, use_drivable=True):
    subset = []
    skipped = 0
    for name in sorted(os.listdir(osz_dir)):
        if not name.endswith('.npz'):
            continue
        token = name[:-4]
        if token not in tokens_in_pkl:
            skipped += 1
            continue
        if occ_frac_of(os.path.join(osz_dir, name),
                       use_drivable=use_drivable) >= thresh:
            subset.append(token)
    print(f'subset: {len(subset)} tokens (occ_frac >= {thresh}, '
          f'drivable-intersected={use_drivable}); '
          f'{skipped} npz without result entry skipped')
    return subset


def rescore_l2(subset, result_pkl):
    """Re-score plan L2 / L2_stp3 on the subset.

    Numbers come from the per-sample ``metric_results`` carried by the eval
    LIST dump — the SAME per-sample values ``dataset.evaluate()`` averages.
    Re-deriving them from ``plan_results``/``plan_gts`` would re-count every
    ``fut_valid_flag=False`` sample that evaluate() skips, which is exactly
    what inflated the first run of this script (subset L2_1s 0.287 > the
    true full-set 0.159).

    TOKEN ALIGNMENT: ``_format_bbox`` fills its token-keyed dicts by
    iterating the same ``results`` list that ``format_results`` dumped, and
    python dicts preserve insertion order — so
    ``list(plan_results.keys())`` IS the per-position token sequence of the
    list dump. An optional ``--ann-file`` cross-check validates that order.
    """
    raw = mmcv.load(result_pkl)
    if not isinstance(raw, list):
        raise SystemExit(
            'per-sample metric_results not found: pass the eval timestamp '
            "dir's results_nusc_all.pkl (the per-sample LIST dump written "
            'by format_results) as --result-pkl')
    tok = load_plan_pkl(result_pkl)
    tokens_seq = list(tok['plan_results'].keys())
    if len(tokens_seq) != len(raw):
        raise SystemExit(
            f'token sequence length {len(tokens_seq)} != list dump length '
            f'{len(raw)} — the two pkls are from different runs')
    if RESCORE_ANN_TOKENS is not None:
        if RESCORE_ANN_TOKENS == tokens_seq:
            print('ann-file token order check: OK')
        else:
            print('WARNING: ann-file token order differs from the pkl '
                  'insertion order — subset alignment may be wrong')

    sel = set(subset)
    sums = {k: 0.0 for k in METRIC_KEYS}          # subset
    full = {k: 0.0 for k in METRIC_KEYS}          # full-set reference
    n_all_valid, n_sel_valid, n_sel_seen = 0, 0, 0
    for i, item in enumerate(raw):
        token = tokens_seq[i]
        is_sel = token in sel
        n_sel_seen += is_sel
        mr = item.get('metric_results', {})
        if not mr.get('fut_valid_flag', False):
            continue                              # same filter as evaluate()
        n_all_valid += 1
        n_sel_valid += is_sel
        for k in METRIC_KEYS:
            full[k] += float(mr[k])
            if is_sel:
                sums[k] += float(mr[k])

    print(f'--- occlusion subset (valid n={n_sel_valid} of {len(sel)} '
          f'selected) | full set valid n={n_all_valid} ---')
    for k in METRIC_KEYS:
        print(f'{k}: subset {sums[k] / max(n_sel_valid, 1):.6f} | '
              f'full {full[k] / max(n_all_valid, 1):.6f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-pkl', required=True,
                        help='eval timestamp dir: results_nusc_all.pkl (the '
                             'per-sample LIST dump; its sibling '
                             'pts_bbox/results_nusc.pkl supplies tokens)')
    parser.add_argument('--osz-dir', default='data/osz',
                        help='offline midas OSZ npz dir')
    parser.add_argument('--occ-thresh', type=float, default=0.2,
                        help='occluded-cell fraction threshold '
                             '(drivable-intersected, same as training)')
    parser.add_argument('--no-drivable', action='store_true',
                        help='use the RAW osz_eye fraction (saturates near '
                             '1.0 — kept only for diagnostics)')
    parser.add_argument('--out-json', default='work_dirs/occ_subset_tokens.json',
                        help='where to save the subset token list')
    parser.add_argument('--ann-file', default=None,
                        help='OPTIONAL cross-check: pkl of the val infos '
                             '(vad_nuscenes_infos_temporal_val.pkl); its '
                             'token order must match the result pkls')
    args = parser.parse_args()
    if args.ann_file:
        global RESCORE_ANN_TOKENS
        RESCORE_ANN_TOKENS = [info['token'] for info in mmcv.load(args.ann_file)['infos']]

    data = mmcv.load(args.result_pkl)
    # tokens must come from the TOKEN-KEYED pkl; the list dump has none.
    if not (isinstance(data, dict) and 'plan_results' in data):
        data = load_plan_pkl(args.result_pkl)
    tokens_in_pkl = set(data['plan_results'].keys())
    subset = collect_subset(args.osz_dir, args.occ_thresh, tokens_in_pkl,
                            use_drivable=not args.no_drivable)

    mmcv.mkdir_or_exist(os.path.dirname(args.out_json) or '.')
    with open(args.out_json, 'w') as f:
        json.dump({'occ_thresh': args.occ_thresh,
                   'drivable_intersected': not args.no_drivable,
                   'tokens': subset}, f)
    print('token list written to', args.out_json)

    rescore_l2(subset, args.result_pkl)


if __name__ == '__main__':
    main()
