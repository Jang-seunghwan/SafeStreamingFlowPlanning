#!/usr/bin/env python3
"""Figure 3: top-down overlay of executed trajectories on the warehouse map.

Background is the Nav2 PGM (physical shelves/walls/crates, world-aligned), with
the two CBF safety circles overlaid, then the executed SSF (closed-loop) vs
SafeDiffuser trajectories (odom_path_xy of the trial JSONs) for one pair.

Usage: python -m bench.plot_traj_overlay --pair 58 [--results-dir results/bench_results]
Output: <out-prefix>.{pdf,png} (default results/warehouse_traj_overlay)
"""
from __future__ import annotations
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_PATH = os.path.join(REPO, 'src', 'ssf_gazebo', 'maps', 'ssf_map_warehouse.yaml')
RES = os.path.join(REPO, 'results', 'bench_results')

OBSTACLES = [{'center': (8.5, 10.0), 'radius': 1.5},
             {'center': (16.0, 11.0), 'radius': 1.5}]
SSF_COLOR = '#1b9e3e'        # green
SD_COLOR = '#ff7f0e'         # orange


def load_pgm(yaml_path):
    meta = yaml.safe_load(open(yaml_path))
    res = float(meta['resolution'])
    ox, oy = float(meta['origin'][0]), float(meta['origin'][1])
    pgm = os.path.join(os.path.dirname(yaml_path), meta['image'])
    with open(pgm, 'rb') as f:
        assert f.readline().strip() == b'P5'
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        w, h = map(int, line.split())
        _ = int(f.readline().strip())
        data = np.frombuffer(f.read(w * h), dtype=np.uint8)
    img = np.flipud(data.reshape(h, w))
    extent = (ox, ox + w * res, oy, oy + h * res)
    return img, extent


def smooth(path, n=400):
    """Light arc-length resampling + spline for a clean curve (faithful to data)."""
    p = np.asarray(path, dtype=float)
    if len(p) < 4:
        return p
    # remove consecutive duplicates
    keep = np.concatenate([[True], np.any(np.diff(p, axis=0) != 0, axis=1)])
    p = p[keep]
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if seg[-1] < 1e-6:
        return p
    t = np.linspace(0, seg[-1], n)
    try:
        from scipy.interpolate import make_interp_spline
        k = min(3, len(p) - 1)
        sx = make_interp_spline(seg, p[:, 0], k=k)(t)
        sy = make_interp_spline(seg, p[:, 1], k=k)(t)
        return np.column_stack([sx, sy])
    except Exception:
        xs = np.interp(t, seg, p[:, 0])
        ys = np.interp(t, seg, p[:, 1])
        return np.column_stack([xs, ys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', type=int, default=58)
    ap.add_argument('--ssf', default='safe_sfp_online')
    ap.add_argument('--sd', default='safe_diffuser_pd')
    ap.add_argument('--results-dir', default=RES)
    ap.add_argument('--out-prefix', default=os.path.join(REPO, 'results', 'warehouse_traj_overlay'))
    args = ap.parse_args()

    img, extent = load_pgm(YAML_PATH)
    a = json.load(open(os.path.join(args.results_dir, f'{args.ssf}_{args.pair:03d}.json')))
    b = json.load(open(os.path.join(args.results_dir, f'{args.sd}_{args.pair:03d}.json')))
    ssf = smooth(a['odom_path_xy'])
    sd = smooth(b['odom_path_xy'])
    start = a['start']
    goal = a['goal']

    plt.rcParams.update({'font.size': 28, 'axes.labelsize': 32,
                         'xtick.labelsize': 24, 'ytick.labelsize': 24})
    fig, ax = plt.subplots(figsize=(12, 9.6), dpi=150)

    # faded occupancy background so trajectories pop
    ax.imshow(img, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=255,
              alpha=0.85, interpolation='nearest', zorder=0)

    # CBF safety circles (nominal radius): filled + dashed outline
    for o in OBSTACLES:
        cx, cy = o['center']; r = o['radius']
        ax.add_patch(Circle((cx, cy), r, facecolor='crimson', alpha=0.20, zorder=2))
        ax.add_patch(Circle((cx, cy), r, fill=False, ec='crimson', lw=2.2,
                            ls='--', zorder=3))

    # trajectories (drawn thick with a white casing for contrast on the map)
    for traj, col in [(ssf, SSF_COLOR), (sd, SD_COLOR)]:
        ax.plot(traj[:, 0], traj[:, 1], color='white', lw=6.5,
                solid_capstyle='round', solid_joinstyle='round', zorder=4)
    ax.plot(ssf[:, 0], ssf[:, 1], color=SSF_COLOR, lw=3.6,
            solid_capstyle='round', solid_joinstyle='round', zorder=6)
    ax.plot(sd[:, 0], sd[:, 1], color=SD_COLOR, lw=3.6,
            solid_capstyle='round', solid_joinstyle='round', zorder=5)

    # start / goal
    ax.plot(start[0], start[1], 'o', mfc='white', mec='black', mew=2.2, ms=15, zorder=8)
    ax.plot(goal[0], goal[1], '*', mfc='gold', mec='black', mew=1.8, ms=26, zorder=8)
    ax.annotate('start', (start[0], start[1]), xytext=(10, 10),
                textcoords='offset points', fontsize=22, fontweight='bold', zorder=9)
    ax.annotate('goal', (goal[0], goal[1]), xytext=(10, -26),
                textcoords='offset points', fontsize=22, fontweight='bold', zorder=9)

    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal'); ax.grid(True, alpha=0.25, ls=':')

    handles = [
        Line2D([0], [0], color=SSF_COLOR, lw=3.6, label='SSF (closed-loop)'),
        Line2D([0], [0], color=SD_COLOR, lw=3.6, label='SafeDiffuser'),
        Line2D([0], [0], color='crimson', lw=2.2, ls='--', label='CBF safety circle (r=1.5 m)'),
        Line2D([0], [0], marker='o', color='w', mfc='white', mec='black', mew=2, ms=12, label='start'),
        Line2D([0], [0], marker='*', color='w', mfc='gold', mec='black', ms=18, label='goal'),
    ]
    ax.legend(handles=handles, loc='upper left', framealpha=0.9, fontsize=21)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    pdf = args.out_prefix + '.pdf'
    png = args.out_prefix + '.png'
    plt.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.savefig(png, dpi=150, bbox_inches='tight')
    print(f'saved {pdf}\nsaved {png}  pair={args.pair}')


if __name__ == '__main__':
    main()
