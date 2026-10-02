#!/usr/bin/env python3
"""Generate the 100 (start, goal) benchmark pairs (bench/test_pairs.json).

Sampling (seed 42):
  - start, goal ~ U([1, 24] x [1, 19]) (warehouse interior, margin from walls),
  - accepted if ||goal - start|| in [5, 15] m and both points pass the map
    check below; candidates are drawn until 400 are accepted. The first 100
    accepted candidates form the base list.
Replacement rule (keeps the robot from starting or ending inside a CBF safety
circle, where the safe set is empty at t = 0 or the goal is unreachable):
  1. every pair whose START lies inside a safety circle is replaced by the next
     unused accepted candidate whose start is outside;
  2. then every pair whose GOAL lies inside a safety circle is replaced by the
     next unused accepted candidate whose start and goal are both outside.
"Inside" uses the filter margin kappa = 0.05: ((p - c) / r)^2 summed < 1.05.
With the default arguments this reproduces bench/test_pairs.json exactly
(12 replaced pairs: 10 23 29 57 69 16 19 34 81 86 91 92).

The map check reads the Nav2 PGM as stored (row 0 = top row of the image) and
indexes it with row = int((y - origin_y) / res), treating every non-occupied
pixel (free or unknown) as free in a (2 * ceil(r / res) + 1)^2 neighbourhood.
This is exactly the check used to create the published pair list; it is kept
unchanged so that the list is reproduced bit for bit.

Usage: python -m bench.generate_test_pairs [--out bench/test_pairs.json]
"""
import argparse
import json
import os

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_YAML = os.path.join(REPO, 'src', 'ssf_gazebo', 'maps', 'ssf_map_warehouse.yaml')
# CBF safety circles ((cx, cy), r) — ssf_gazebo.cbf.WAREHOUSE_OBSTACLES.
CIRCLES = [((8.5, 10.0), 1.5), ((16.0, 11.0), 1.5)]
KAPPA = 0.05


def read_pgm(path):
    """Binary (P5) PGM -> uint8 array, row 0 = top row of the image."""
    with open(path, 'rb') as f:
        assert f.readline().strip() == b'P5'
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        w, h = map(int, line.split())
        int(f.readline().strip())
        data = np.frombuffer(f.read(w * h), dtype=np.uint8)
    return data.reshape(h, w)


def inside(xy):
    return any(((xy[0] - c[0]) / r) ** 2 + ((xy[1] - c[1]) / r) ** 2 < 1.0 + KAPPA
               for c, r in CIRCLES)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-pairs', type=int, default=100)
    p.add_argument('--num-candidates', type=int, default=400,
                   help='Accepted candidates drawn (base list + spares for replacement).')
    p.add_argument('--min-dist', type=float, default=5.0)
    p.add_argument('--max-dist', type=float, default=15.0)
    p.add_argument('--xmin', type=float, default=1.0)
    p.add_argument('--xmax', type=float, default=24.0)
    p.add_argument('--ymin', type=float, default=1.0)
    p.add_argument('--ymax', type=float, default=19.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--robot-radius-m', type=float, default=0.25)
    p.add_argument('--out', default=os.path.join(REPO, 'bench', 'test_pairs.json'))
    args = p.parse_args()

    meta = yaml.safe_load(open(MAP_YAML))
    img = read_pgm(os.path.join(os.path.dirname(MAP_YAML), meta['image']))
    res = float(meta['resolution'])
    ox, oy = meta['origin'][:2]
    free = img > 0.65
    rc = int(np.ceil(args.robot_radius_m / res))

    def is_free(xy):
        c = int((xy[0] - ox) / res)
        r = int((xy[1] - oy) / res)
        H, W = free.shape
        if not (0 <= r < H and 0 <= c < W):
            return False
        return bool(free[max(0, r - rc):min(H, r + rc + 1), max(0, c - rc):min(W, c + rc + 1)].all())

    rng = np.random.default_rng(args.seed)
    acc = []
    tries = 0
    lo, hi = [args.xmin, args.ymin], [args.xmax, args.ymax]
    while len(acc) < args.num_candidates and tries < 1000 * args.num_candidates:
        tries += 1
        s = rng.uniform(lo, hi)
        g = rng.uniform(lo, hi)
        d = float(np.linalg.norm(g - s))
        if not (args.min_dist <= d <= args.max_dist) or not is_free(s) or not is_free(g):
            continue
        acc.append((s, g, d))
    if len(acc) < args.num_pairs:
        raise RuntimeError(f'only {len(acc)} candidates accepted')

    cur = [acc[i] for i in range(args.num_pairs)]
    src = {}
    used = set()
    replaced = []

    def take(ok):
        for j in range(args.num_pairs, len(acc)):
            if j not in used and ok(*acc[j][:2]):
                used.add(j)
                return acc[j], j
        raise RuntimeError('ran out of spare candidates')

    for i in range(args.num_pairs):
        if inside(cur[i][0]):
            cur[i], src[i] = take(lambda s, g: not inside(s))
            replaced.append(i)
    for i in range(args.num_pairs):
        if inside(cur[i][1]):
            cur[i], src[i] = take(lambda s, g: not inside(s) and not inside(g))
            replaced.append(i)

    pairs = []
    for i, (s, g, d) in enumerate(cur):
        pairs.append({'idx': i, 'start': [float(s[0]), float(s[1]), 0.0, 0.0],
                      'goal': [float(g[0]), float(g[1])], 'dist_m': d,
                      'source_candidate': int(src.get(i, i))})
    out = {
        'seed': args.seed, 'num_pairs': args.num_pairs,
        'distance_range_m': [args.min_dist, args.max_dist],
        'bounds_xy': [args.xmin, args.ymin, args.xmax, args.ymax],
        'robot_radius_m': args.robot_radius_m,
        'replaced_idx': replaced,
        'replacement_rule': f'start_goal-inside at kappa {KAPPA} -> next accepted candidate '
                            f'of the seed-{args.seed} stream',
        'pairs': pairs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'{len(pairs)} pairs ({len(replaced)} replaced: {replaced}) -> {args.out}')


if __name__ == '__main__':
    main()
