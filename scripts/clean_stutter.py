#!/usr/bin/env python3
"""Clean position stutter from Gazebo rollout CSVs.

The Gazebo logger publishes velocity at the loop rate (~20 Hz) but position
at ~1 Hz (every ~18 rows), so consecutive timesteps repeat (x, y) while
(v_x, v_y) continue to update.  This breaks the dynamic-consistency
assumption `delta_pos ≈ vel * dt`, which the SFP residual base relies on
and which the diffuser/CFM trajectory distribution should also satisfy at
deployment time.

Cleaning strategy (velocity-driven, anchored):
  - Detect "anchor" rows: row 0 and any row where (x, y) differs from i-1.
    Between anchors the original logger emitted stale positions.
  - For each gap (anchor_a → anchor_b), integrate forward from anchor_a
    using the recorded velocity (trapezoidal rule), then add a linear
    correction term so the integrated value lands exactly on anchor_b.
    This preserves the anchor positions AND makes the in-between trajectory
    satisfy delta_pos ≈ vel * dt to within the linear residual.
  - Velocity / heading columns are untouched.

The output CSV preserves all rows and columns — only x, y change.

Usage:
    python clean_stutter.py --input-dir LOGS_RAW --output-dir LOGS_CLEAN [--n-workers 8]
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

import numpy as np
import pandas as pd

POSITION_COLS = ('x', 'y')
TIME_COL = 't'


def find_anchors(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return boolean mask: True if row i is an anchor (position differs from i-1).

    Row 0 is always an anchor.
    """
    if len(x) == 0:
        return np.zeros(0, dtype=bool)
    anchors = np.empty(len(x), dtype=bool)
    anchors[0] = True
    anchors[1:] = (x[1:] != x[:-1]) | (y[1:] != y[:-1])
    return anchors


def interpolate_by_velocity(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    anchors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Velocity-driven, anchor-corrected position interpolation.

    For each anchor-to-anchor gap [a, b]:
      1. Integrate forward from x[a] using trapezoidal rule on (vx, vy)
         to get a free-running predicted position at index b.
      2. Compute the residual = anchor_b - predicted_b.
      3. Distribute that residual linearly across the gap so the final
         value at index b lands exactly on the recorded anchor.

    Result: delta_pos[i] = vel[i] * dt[i] + small_drift_correction,
    so the residual-base formula in SFP holds to first order, and the
    cleaned trajectory still hits each recorded anchor exactly.
    """
    new_x = x.copy()
    new_y = y.copy()
    anchor_idx = np.where(anchors)[0]
    n_filled = 0

    for k in range(len(anchor_idx) - 1):
        a = int(anchor_idx[k])
        b = int(anchor_idx[k + 1])
        if b - a < 2:
            continue
        gap = b - a
        dt = np.diff(t[a:b + 1])      # length = gap
        # guard against non-monotone or degenerate t
        if not np.all(dt > 0):
            # fall back to linear interp on time
            dt_total = t[b] - t[a]
            if dt_total <= 0:
                fractions = np.arange(1, gap) / gap
            else:
                fractions = (t[a + 1:b] - t[a]) / dt_total
            new_x[a + 1:b] = x[a] + fractions * (x[b] - x[a])
            new_y[a + 1:b] = y[a] + fractions * (y[b] - y[a])
            n_filled += gap - 1
            continue

        # Trapezoidal integration of velocity over each subinterval
        avg_vx = (vx[a:b] + vx[a + 1:b + 1]) / 2.0     # length = gap
        avg_vy = (vy[a:b] + vy[a + 1:b + 1]) / 2.0
        cum_dx = np.cumsum(avg_vx * dt)                # length = gap
        cum_dy = np.cumsum(avg_vy * dt)
        pred_x = x[a] + cum_dx
        pred_y = y[a] + cum_dy

        # Residual at b: where free integration landed vs recorded anchor
        # Distribute linearly across the gap so anchor_b is hit exactly.
        residual_x = x[b] - pred_x[-1]
        residual_y = y[b] - pred_y[-1]
        t_frac = (t[a + 1:b + 1] - t[a]) / (t[b] - t[a])  # 0 → 1

        # Final corrected positions
        new_x[a + 1:b + 1] = pred_x + residual_x * t_frac
        new_y[a + 1:b + 1] = pred_y + residual_y * t_frac
        # Anchors are exact (last position lands on x[b], y[b]) by construction
        n_filled += gap - 1

    return new_x, new_y, n_filled


def clean_csv(path_in: str, path_out: str) -> Tuple[int, int, int]:
    """Clean one CSV file. Returns (n_rows, n_anchors, n_filled)."""
    df = pd.read_csv(path_in)
    n_rows = len(df)

    required = {'x', 'y', 'v_x', 'v_y'}
    if n_rows < 2 or not required.issubset(df.columns):
        df.to_csv(path_out, index=False)
        return n_rows, n_rows, 0

    t = df[TIME_COL].values.astype(np.float64) if TIME_COL in df.columns else np.arange(n_rows, dtype=np.float64)
    x = df['x'].values.astype(np.float64)
    y = df['y'].values.astype(np.float64)
    vx = df['v_x'].values.astype(np.float64)
    vy = df['v_y'].values.astype(np.float64)

    anchors = find_anchors(x, y)
    if anchors.sum() == n_rows:
        # Already clean — copy through
        df.to_csv(path_out, index=False)
        return n_rows, n_rows, 0

    new_x, new_y, n_filled = interpolate_by_velocity(t, x, y, vx, vy, anchors)
    df['x'] = new_x
    df['y'] = new_y

    os.makedirs(os.path.dirname(path_out) or '.', exist_ok=True)
    df.to_csv(path_out, index=False, float_format='%.10g')
    return n_rows, int(anchors.sum()), n_filled


def _worker(args: Tuple[str, str]) -> Tuple[str, int, int, int]:
    path_in, path_out = args
    n_rows, n_anchors, n_filled = clean_csv(path_in, path_out)
    return path_in, n_rows, n_anchors, n_filled


def main() -> None:
    parser = argparse.ArgumentParser(description='Clean position stutter from Gazebo rollout CSVs.')
    parser.add_argument('--input-dir', required=True, help='Source directory of rollout_*.csv files.')
    parser.add_argument('--output-dir', required=True, help='Destination directory for cleaned CSVs.')
    parser.add_argument('--n-workers', type=int, default=os.cpu_count() or 4)
    parser.add_argument('--progress-every', type=int, default=500)
    args = parser.parse_args()

    in_dir = os.path.expanduser(args.input_dir)
    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(in_dir) if f.startswith('rollout_') and f.endswith('.csv'))
    if not files:
        raise FileNotFoundError(f'No rollout_*.csv in {in_dir}')

    print(f'Processing {len(files)} CSVs from {in_dir} -> {out_dir}  ({args.n_workers} workers)')
    tasks = [(os.path.join(in_dir, f), os.path.join(out_dir, f)) for f in files]

    total_rows = 0
    total_anchors = 0
    total_filled = 0
    started = time.time()

    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        for i, fut in enumerate(as_completed(pool.submit(_worker, t) for t in tasks), start=1):
            path_in, n_rows, n_anchors, n_filled = fut.result()
            total_rows += n_rows
            total_anchors += n_anchors
            total_filled += n_filled
            if i == 1 or i % args.progress_every == 0 or i == len(files):
                elapsed = time.time() - started
                rate = i / elapsed if elapsed > 0 else 0
                print(f'  [{i:5d}/{len(files)}] {os.path.basename(path_in):<24s} '
                      f'rows={n_rows} anchors={n_anchors} filled={n_filled} '
                      f'| {rate:.1f} files/s')

    elapsed = time.time() - started
    stutter_frac = total_filled / max(total_rows, 1)
    print()
    print(f'Done in {elapsed:.1f}s')
    print(f'Total rows:          {total_rows:,}')
    print(f'Anchor rows:         {total_anchors:,}  ({total_anchors / total_rows * 100:.1f}%)')
    print(f'Stutter rows filled: {total_filled:,}  ({stutter_frac * 100:.1f}%)')


if __name__ == '__main__':
    main()
