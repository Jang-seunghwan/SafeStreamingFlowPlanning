"""
Build the preprocessed training data from the raw laps of scripts/collect_trajectories.py.

Pipeline per track: read every CSV of logs/<Track>/ ([x, y, vx_w, vy_w] at 100 Hz),
subsample 10x (10 Hz), drop episodes whose length is outside the 1st-99th
percentile, and resample every episode to the track horizon. Two files are
written, differing only in the velocity channels:

  processed_data/m_per_s/<track>.npz     velocity interpolated from the CSV (m/s)
  processed_data/m_per_step/<track>.npz  forward difference of the resampled positions
                                         (m per 0.1 s step)

The paper models use m_per_step for Diffuser and for the CFM model of FM /
FlowMatcher (and SafeFM / SafeFlowMatcher on Budapest), and m_per_s for the SFP
model and for the CFM model of SafeFM / SafeFlowMatcher on Catalunya.

Usage:
    python scripts/regenerate_processed_data.py
"""
from __future__ import annotations
import os, sys, glob, argparse, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config.f1tenth import PROJECT_ROOT, TRACKS, TRACK_DIR_MAP, get_horizon, get_processed_path

SUBSAMPLE = 10             # 100 Hz -> 10 Hz
OUTLIER_PERCENTILE = 99    # keep episodes with length in [1st, 99th] percentile


def resample_m_per_s(traj, horizon):
    """Resample a trajectory to `horizon` waypoints; positions and velocities are
    time-interpolated, so the velocity stays in m/s."""
    N = len(traj)
    if N == horizon:
        return traj.copy()
    t_orig = np.linspace(0, 1, N)
    t_new = np.linspace(0, 1, horizon)
    out = np.zeros((horizon, 4), dtype=np.float32)
    for c in range(4):
        out[:, c] = np.interp(t_new, t_orig, traj[:, c])
    return out


def resample_m_per_step(traj, horizon):
    """Resample positions as in `resample_m_per_s`; the velocity columns are the
    forward differences of the resampled positions without division by dt
    (v_i = p_{i+1} - p_i, last row repeated). Trajectories that already have
    `horizon` waypoints are returned unchanged (m/s velocity from the CSV)."""
    N = len(traj)
    if N == horizon:
        return traj.copy()
    t_orig = np.linspace(0, 1, N)
    t_new = np.linspace(0, 1, horizon)
    out = np.zeros((horizon, 4), dtype=np.float32)
    out[:, 0] = np.interp(t_new, t_orig, traj[:, 0])
    out[:, 1] = np.interp(t_new, t_orig, traj[:, 1])
    out[:-1, 2:4] = out[1:, :2] - out[:-1, :2]
    out[-1, 2:4] = out[-2, 2:4]
    return out


def load_track_csvs(logs_dir: str, track: str):
    """Read all CSVs for a track, subsample, drop length outliers."""
    dir_name = TRACK_DIR_MAP[track]
    csv_files = sorted(glob.glob(os.path.join(logs_dir, dir_name, '*.csv')))
    if not csv_files:
        raise FileNotFoundError(f'No CSVs for {track} at {logs_dir}/{dir_name}')

    raw_trajs = []
    sub_lengths = []
    for f in csv_files:
        try:
            data = np.genfromtxt(f, delimiter=',', skip_header=1, dtype=np.float32)
            if data.ndim == 1 or len(data) < SUBSAMPLE * 2:
                continue
            sub = data[:, :4][::SUBSAMPLE].copy()     # [x, y, vx_w, vy_w], 100Hz → 10Hz
            raw_trajs.append(sub)
            sub_lengths.append(len(sub))
        except Exception as e:
            print(f'  WARN skip {os.path.basename(f)}: {e}')
            continue

    if not raw_trajs:
        raise ValueError(f'No valid CSVs in {logs_dir}/{dir_name}')

    sub_lengths = np.array(sub_lengths)
    p_low = np.percentile(sub_lengths, 100 - OUTLIER_PERCENTILE)
    p_high = np.percentile(sub_lengths, OUTLIER_PERCENTILE)
    filtered = [t for t, s in zip(raw_trajs, sub_lengths) if p_low <= s <= p_high]
    print(f'  read={len(raw_trajs)}  after outlier filter={len(filtered)}  '
          f'(len range: {p_low:.0f} - {p_high:.0f})')
    return filtered


def regenerate_track(track: str, logs_dir: str):
    horizon = get_horizon(track)
    print(f'\n=== {track}  (horizon={horizon}) ===')
    t0 = time.time()
    trajs = load_track_csvs(logs_dir, track)
    for units, resample in (('m_per_s', resample_m_per_s), ('m_per_step', resample_m_per_step)):
        arr = np.stack([resample(t, horizon) for t in trajs], axis=0).astype(np.float32)
        out_path = get_processed_path(track, units)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez(out_path, observations=arr, horizon=horizon, n_episodes=len(arr))
        print(f'  Wrote {out_path}  shape={arr.shape}')
    print(f'  ({time.time()-t0:.1f}s)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracks', nargs='+', default=TRACKS, choices=TRACKS)
    parser.add_argument('--logs_dir', type=str, default=os.path.join(PROJECT_ROOT, 'logs'),
                        help='Directory with one CSV sub-directory per track (logs/<Track>/).')
    args = parser.parse_args()
    for track in args.tracks:
        regenerate_track(track, args.logs_dir)
    print('\nAll done.')


if __name__ == '__main__':
    main()
