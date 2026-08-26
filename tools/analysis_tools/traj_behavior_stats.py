"""Near-occlusion braking behavior stats (V2, design doc §5.1).

Evidence that the model "perceives" occlusion: compare the average speed of
trajectory steps taken within 5 m of an occlusion boundary, predicted vs GT.
A risk-aware model is expected to slow down MORE than the GT near occluders
(gt itself is a safe demo; the interesting signal is pred speed <= gt speed
near occluders, and pred speed == gt speed far from them).

Speed per step = step length / 0.5 s (steps are 0.5 s apart).

Also reports a zero-displacement ("turtle") analysis: samples whose
commanded trajectory ends within ``ZERO_END_M`` of the start are counted
separately with their near-occluder step fraction and mean speed — to tell
whether over-conservative trajectories cluster at the occluder (risk
channel over-braking) or are scene-wide (independent failure mode).

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
ZERO_END_M = 0.5        # "zero-displacement" threshold: end displacement (m)
                        # below this marks a turtle trajectory (the
                        # over-conservatism signature seen in training DIAG:
                        # traj_end=0.00-0.04m samples)


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

    # Skip samples whose future GT is invalid (fut_valid_flag=False): their
    # zero-padded trajectories would masquerade as "braking" in the speed
    # stats — the same filter dataset.evaluate() applies. The flag lives in
    # the per-sample LIST dump; align it by insertion order (dicts keep
    # order and _format_bbox filled the token-keyed dicts from that list).
    invalid = set()
    raw_pkl = args.result_pkl
    raw = mmcv.load(raw_pkl)
    if not isinstance(raw, list):
        cand = osp.join(osp.dirname(raw_pkl), 'results_nusc_all.pkl')
        if osp.exists(cand):
            raw = mmcv.load(cand)
    if isinstance(raw, list):
        seq = list(plan_results.keys())
        if len(seq) == len(raw):
            for tok_i, item in zip(seq, raw):
                mr = item.get('metric_results', {})
                if not mr.get('fut_valid_flag', False):
                    invalid.add(tok_i)
            if invalid:
                print(f'{len(invalid)} tokens skipped (fut_valid_flag=False)')
        else:
            print(f'WARNING: list dump length {len(raw)} != token count '
                  f'{len(seq)} — fut-valid filter skipped')

    if args.token_json:
        with open(args.token_json) as f:
            sel = set(json.load(f)['tokens'])
        tokens = [t for t in tokens if t in sel]
    tokens = [t for t in tokens if t not in invalid]

    pred_speeds, gt_speeds = [], []
    near_pred, near_gt, far_pred, far_gt = [], [], [], []
    # Per-sample zero-displacement (turtle) analysis: end displacement,
    # near-occluder step fraction and mean speed — to test whether the
    # turtle trajectories cluster near occlusion (over-conservatism at the
    # occluder) or are scene-wide (independent failure mode).
    end_disp, near_frac, mean_speed = [], [], []
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
        end_disp.append(float(np.linalg.norm(pred_abs[-1])))
        near_frac.append(float(m_pn.mean()))
        mean_speed.append(float(sp_pred.mean()))

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

    # ---- zero-displacement (turtle) analysis ----
    # Are the turtle trajectories clustered at the occluder (the risk
    # channel over-braking where it should) or spread over the scene?
    end_disp = np.asarray(end_disp)
    near_frac = np.asarray(near_frac)
    mean_speed = np.asarray(mean_speed)
    turtle = end_disp < ZERO_END_M
    normal = ~turtle
    print(f'--- zero-displacement analysis (end < {ZERO_END_M}m) ---')
    print(f'turtle samples: {int(turtle.sum())}/{n} '
          f'({100.0 * turtle.mean():.1f}%)')
    if turtle.any():
        print(f'turtle: near-occluder step fraction {near_frac[turtle].mean():.3f} '
              f'vs {near_frac[normal].mean():.3f} for normal samples; '
              f'mean speed {mean_speed[turtle].mean():.2f} m/s '
              f'(normal {mean_speed[normal].mean():.2f} m/s)')
        # GT reference at the same (turtle) tokens: is the GT itself
        # stopped there? gt mean speed per sample at turtle tokens:
        gt_ms = np.asarray([float(np.asarray(s).mean()) for s in gt_speeds])
        print(f'at turtle tokens: GT mean speed {gt_ms[turtle].mean():.2f} m/s '
              f'(GT at normal tokens {gt_ms[normal].mean():.2f} m/s)')
    else:
        print('no turtle samples in this subset')


if __name__ == '__main__':
    main()
