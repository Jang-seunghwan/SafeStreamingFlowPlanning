#!/usr/bin/env python3
"""Trajectory metrics for the warehouse benchmark.

- acceleration_smoothness: Sm., port of diffuser/utils/trajectory_metrics.py
  (main branch) applied to the /cmd_vel stream.
- min_barrier: execution-level safety, the minimum barrier value over every
  recorded /odom sample of a trial.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Sequence

# Safety circles (cx, cy, r) of the warehouse benchmark; kept in sync with
# ssf_gazebo.cbf.WAREHOUSE_OBSTACLES (order n = 2).
SAFETY_CIRCLES = ((8.5, 10.0, 1.5), (16.0, 11.0, 1.5))


def acceleration_smoothness(vx: Sequence[float],
                            vy: Sequence[float],
                            times: Sequence[float]) -> float:
    """Mean magnitude of cmd_vel acceleration (m/s^2).

    Mirrors `acceleration_smoothness` from diffuser/utils/trajectory_metrics.py
    (main branch) but applied to a cmd_vel (Twist.linear.x, .y) stream instead
    of a normalized trajectory.

    a_i = (v_{i+1} - v_i) / (t_{i+1} - t_i)
    Sm  = mean(|a_i|)

    Returns 0.0 if fewer than 3 samples (need at least one valid difference).
    """
    n = min(len(vx), len(vy), len(times))
    if n < 3:
        return 0.0
    vx_arr = np.asarray(vx[:n], dtype=np.float64)
    vy_arr = np.asarray(vy[:n], dtype=np.float64)
    t_arr  = np.asarray(times[:n], dtype=np.float64)

    dt = np.diff(t_arr)
    # mask out non-positive dt (clock glitches / duplicate stamps)
    valid = dt > 1e-6
    if not valid.any():
        return 0.0

    ax = np.diff(vx_arr) / np.where(valid, dt, 1.0)
    ay = np.diff(vy_arr) / np.where(valid, dt, 1.0)
    mag = np.hypot(ax, ay)
    mag = np.where(valid & np.isfinite(mag), mag, 0.0)
    if not valid.any():
        return 0.0
    return float(mag[valid].sum() / valid.sum())


def min_barrier(xs: Sequence[float], ys: Sequence[float],
                circles=SAFETY_CIRCLES) -> Optional[float]:
    """min_k min_i h_i(x_k) with h_i(x) = ((x-cx)/r)^2 + ((y-cy)/r)^2 - 1.

    This is the barrier of the paper with n = 2 and kappa = 0 (the nominal
    safety set; the filter margin kappa = 0.05 is a tightening of the filter,
    not part of the safe set). A trial violates safety iff the value is < 0.
    Returns None if there are no samples.
    """
    n = min(len(xs), len(ys))
    if n == 0:
        return None
    x = np.asarray(xs[:n], dtype=np.float64)
    y = np.asarray(ys[:n], dtype=np.float64)
    h = [((x - cx) / r) ** 2 + ((y - cy) / r) ** 2 - 1.0 for cx, cy, r in circles]
    return float(np.min(np.stack(h, axis=1)))
