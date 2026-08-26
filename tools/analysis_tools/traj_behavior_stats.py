"""Near-occlusion braking behavior stats (V2, design doc §5.1).

Evidence that the model "perceives" occlusion: compare the average speed of
trajectory steps taken within 5 m of an occlusion boundary, predicted vs GT.
A risk-aware model is expected to slow down MORE than the GT near occluders
(gt itself is a safe demo; the interesting signal is pred speed <= gt speed
near occluders, and pred speed == gt speed far from them).

Speed per step = step length / 0.5 s (steps are 0.5 s apart).

Mask layout: OSZ npz osz_eye is 200x200, axis-0 = ego-x (0.15 m/cell,
x in [-15,15]), axis-1 = ego-y (0.3 m/cell, y in [-30,30]) — see
OSZ/config.py. The distance transform runs on cell units and is scaled by
the coarser axis (0.3 m/cell) as the conservative approximation (same
convention as the drivable-dilation note in STATUS.md).

Usage (from the repo root):
    python tools/analysis_tools/traj_behavior_stats.py \
        --result-pkl test/oa_resworld_config/<ts>/pts_bbox/results_nusc.pkl \
        [--token-json work_dirs/occ_subset_tokens.json]  # optional subset
"""
import argparse
import json
import os.path as osp

import numpy as np
import mmcv
from scipy.ndimage import distance_transform_edt

GRID = dict(x_min=-15.0, x_cell=0.15, y_min=-30.0, y_cell=0.3)
NEAR_M = 5.0            # "near an occlusion boundary" radius (metres)
CELL_M = 0.3            # coarse-axis cell size used for the distance scale
STEP_S = 0.5            # seconds per trajectory step


def load_plan_pkl(pkl_path):
    """Return the TOKEN-KEYED plan pkl ``{'plan_results', 'plan_gts'}``.

    Same dual-layout contract as filter_occ_subset.py: if handed the
    per-sample LIST dump (``<ts>/results_nusc_all.pkl``, written by
    ``format_results``), fall back to the sibling token-keyed
    ``pts_bbox/results_nusc.pkl`` (written by ``_format_bbox``).
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


def mask_dist(npz_path):
    with np.load(npz_path, allow_pickle=True) as d:
        eye = d.get('osz_eye')
        if eye is None:
            eye = d.get('osz_ground')
    eye = np.asarray(eye) > 0
    return distance_transform_edt(~eye).astype(np.float32)  # cell units


def step_speed(inc):
    """inc: (T,2) per-step increments -> speed per step in m/s."""
    return np.sqrt((inc ** 2).sum(axis=1)) / STEP_S


def point_cells(xy, dist):
    """xy: (T,2) absolute ego coords -> nearest dist value per point."""
    xs = np.clip(np.round((xy[:, 0] - GRID['x_min']) / GRID['x_cell']),
                 0, dist.shape[0] - 1).astype(int)
    ys = np.clip(np.round((xy[:, 1] - GRID['y_min']) / GRID['y_cell']),
                 0, dist.shape[1] - 1).astype(int)
    return dist[xs, ys] * CELL_M      # metres


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-pkl', required=True)
    parser.add_argument('--osz-dir', default='data/osz')
    parser.add_argument('--token-json', default=None,
                        help='optional subset list from filter_occ_subset.py')
    args = parser.parse_args()

    data = load_plan_pkl(args.result_pkl)
    plan_results = data['plan_results']
    plan_gts = data['plan_gts']

    tokens = list(plan_results.keys())
    if args.token_json:
        with open(args.token_json) as f:
            sel = set(json.load(f)['tokens'])
        tokens = [t for t in tokens if t in sel]

    pred_speeds, gt_speeds = [], []
    near_pred, near_gt, far_pred, far_gt = [], [], [], []
    missing = 0
    for token in tokens:
        npz_path = args.osz_dir + '/' + token + '.npz'
        if not mmcv.is_filepath(npz_path) and not __import__('os').path.exists(npz_path):
            missing += 1
            continue
        dist = mask_dist(npz_path)
        pred_inc, cmd = plan_results[token]
        gt_inc = plan_gts[token]
        cmd_flat = np.asarray(cmd.cpu() if hasattr(cmd, 'cpu') else cmd)
        idx = int(np.flatnonzero(cmd_flat.reshape(-1))[0])
        pred_inc = np.asarray(
            pred_inc[idx].cpu() if hasattr(pred_inc, 'cpu')
            else np.asarray(pred_inc)[idx])
        gt_inc = np.asarray(gt_inc.cpu() if hasattr(gt_inc, 'cpu') else gt_inc)

        pred_abs = pred_inc.cumsum(axis=0)
        gt_abs = gt_inc.cumsum(axis=0)
        # GT trajectories are zero-padded at the tail for short scenes —
        # mask those steps out of BOTH sides so the speed statistics stay
        # symmetric (a zero step is not a "slow GT", it is no GT at all).
        gt_valid = (gt_inc ** 2).sum(axis=1) > 1e-6
        if not gt_valid.any():
            continue          # scene-end token with an all-zero GT: skip
        pred_inc = pred_inc[gt_valid]
        pred_abs = pred_abs[gt_valid]
        gt_inc = gt_inc[gt_valid]
        gt_abs = gt_abs[gt_valid]
        d_pred = point_cells(pred_abs, dist)
        d_gt = point_cells(gt_abs, dist)
        sp_pred = step_speed(pred_inc)
        sp_gt = step_speed(gt_inc)

        m_pn, m_gn = d_pred < NEAR_M, d_gt < NEAR_M
        near_pred.append(sp_pred[m_pn])
        near_gt.append(sp_gt[m_gn])
        far_pred.append(sp_pred[~m_pn])
        far_gt.append(sp_gt[~m_gn])
        pred_speeds.append(sp_pred)
        gt_speeds.append(sp_gt)

    if missing:
        print(f'{missing} tokens skipped (no npz)')
    n = len(pred_speeds)
    print(f'--- near-occlusion behavior (n={n} samples) ---')
    if n == 0:
        return
    np_ = np.concatenate(near_pred)
    ng_ = np.concatenate(near_gt)
    fp_ = np.concatenate(far_pred)
    fg_ = np.concatenate(far_gt)
    print(f'near (<{NEAR_M}m): pred speed {np_.mean():.2f} m/s '
          f'({len(np_)} steps), gt {ng_.mean():.2f} m/s ({len(ng_)} steps), '
          f'ratio {np_.mean() / max(ng_.mean(), 1e-6):.3f}')
    print(f'far  (>={NEAR_M}m): pred speed {fp_.mean():.2f} m/s, '
          f'gt {fg_.mean():.2f} m/s, '
          f'ratio {fp_.mean() / max(fg_.mean(), 1e-6):.3f}')


if __name__ == '__main__':
    main()
