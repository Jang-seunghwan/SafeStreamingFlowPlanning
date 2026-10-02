"""
F1TENTH trajectory collector (parallel), used to build the training data.

Generates diverse laps by tracking blended reference paths
(raceline <-> centerline) with a Pure Pursuit controller in f110_gym. Diversity comes from:
  - Alpha blending: reference = alpha*raceline + (1-alpha)*centerline, alpha ~ U(0, 1)
  - Lookahead distances (cycled over 0.5, 0.8, 1.2, 1.8 m)
Every episode starts at the start/finish line (origin) with the centerline heading
and records one lap.

Speed is fixed (default 0.8x the raceline speed) for a consistent horizon per track.
Each worker process creates its own F110Env instance for full CPU parallelism.

Output: CSV files <Track>_a<alpha>_l<lookahead>_<idx>.csv with columns
[x, y, vx_w, vy_w] recorded at 100Hz (sim rate), written to --log_dir.
scripts/regenerate_processed_data.py reads logs/<Track>/*.csv, so pass --log_dir logs/<Track>.

Requires a clone of https://github.com/f1tenth/f1tenth_racetracks (raceline,
centerline and map files), see config/f1tenth.py RACETRACKS_DIR.

Usage (one track per call):
    python scripts/collect_trajectories.py --tracks Budapest --n_episodes 10000 --log_dir logs/Budapest
"""
import os
import sys
import argparse
import numpy as np
import csv
from scipy.spatial import cKDTree
from multiprocessing import Pool, Value, Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.f1tenth import PROJECT_ROOT, RACETRACKS_DIR
from f110_gym.envs import F110Env
from f110_gym.envs.base_classes import Integrator


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def load_raceline(track_dir, track_name):
    """Load pre-computed raceline from f1tenth_racetracks."""
    path = os.path.join(track_dir, track_name, f'{track_name}_raceline.csv')
    data = np.loadtxt(path, delimiter=';', comments='#')
    return {
        's': data[:, 0],
        'x': data[:, 1],
        'y': data[:, 2],
        'psi': data[:, 3],
        'kappa': data[:, 4],
        'vx': data[:, 5],
        'ax': data[:, 6],
    }


def load_centerline(track_dir, track_name):
    """Load centerline from f1tenth_racetracks."""
    path = os.path.join(track_dir, track_name, f'{track_name}_centerline.csv')
    data = np.loadtxt(path, delimiter=',', comments='#')
    return {
        'x': data[:, 0],
        'y': data[:, 1],
        'w_right': data[:, 2],
        'w_left': data[:, 3],
    }


def get_map_path(track_dir, track_name):
    return os.path.join(track_dir, track_name, f'{track_name}_map')


# ---------------------------------------------------------------------------
#  Reference path blending
# ---------------------------------------------------------------------------

def match_centerline_to_raceline(raceline, centerline):
    """Pre-compute centerline points matched to each raceline waypoint.

    Uses cKDTree for fast nearest-neighbor lookup. Called once per track
    in each worker, then reused across all jobs.
    """
    cl_xy = np.column_stack([centerline['x'], centerline['y']])
    rl_xy = np.column_stack([raceline['x'], raceline['y']])
    tree = cKDTree(cl_xy)
    _, cl_indices = tree.query(rl_xy)
    return cl_xy[cl_indices]  # shape (n_raceline, 2)


def blend_reference_path(raceline, matched_cl_xy, alpha):
    """Blend raceline and matched centerline to create diverse reference path.

    alpha=1.0 -> pure raceline, alpha=0.0 -> pure centerline.
    Uses pre-matched centerline points for efficiency.
    """
    blended_x = alpha * raceline['x'] + (1 - alpha) * matched_cl_xy[:, 0]
    blended_y = alpha * raceline['y'] + (1 - alpha) * matched_cl_xy[:, 1]

    # Recompute heading from blended path (finite differences on closed loop)
    dx = np.roll(blended_x, -1) - blended_x
    dy = np.roll(blended_y, -1) - blended_y
    blended_psi = np.arctan2(dy, dx)

    return {
        'x': blended_x,
        'y': blended_y,
        'psi': blended_psi,
        'vx': raceline['vx'],
        's': raceline['s'],
        'kappa': raceline['kappa'],
        'ax': raceline['ax'],
    }


# ---------------------------------------------------------------------------
#  Pure Pursuit Controller
# ---------------------------------------------------------------------------

class PurePursuitTracker:
    def __init__(self, ref_path, lookahead_dist=1.0, speed_scale=1.0,
                 wheelbase=0.3302):
        self.waypoints_xy = np.column_stack([ref_path['x'], ref_path['y']])
        self.waypoints_v = ref_path['vx'] * speed_scale
        self.waypoints_psi = ref_path['psi']
        self.lookahead_base = lookahead_dist
        self.wheelbase = wheelbase
        self.n_waypoints = len(ref_path['x'])
        self.nearest_idx = 0
        self.nearest_dist = 0.0
        self.max_track_deviation = 2.0

    def _find_nearest(self, x, y):
        search_range = 150
        start = self.nearest_idx - search_range // 4
        indices = np.arange(start, start + search_range) % self.n_waypoints
        dists = np.hypot(
            self.waypoints_xy[indices, 0] - x,
            self.waypoints_xy[indices, 1] - y
        )
        best = np.argmin(dists)
        self.nearest_idx = indices[best]
        self.nearest_dist = dists[best]
        return self.nearest_idx

    def _find_lookahead_point(self, x, y, nearest_idx, current_speed):
        max_v = np.max(self.waypoints_v)
        speed_ratio = current_speed / max(max_v, 1.0)
        effective_la = self.lookahead_base * (0.5 + 0.5 * speed_ratio)
        effective_la = max(effective_la, 0.3)

        cumul_dist = 0.0
        idx = nearest_idx
        for _ in range(self.n_waypoints):
            next_idx = (idx + 1) % self.n_waypoints
            seg_len = np.hypot(
                self.waypoints_xy[next_idx, 0] - self.waypoints_xy[idx, 0],
                self.waypoints_xy[next_idx, 1] - self.waypoints_xy[idx, 1]
            )
            cumul_dist += seg_len
            idx = next_idx
            if cumul_dist >= effective_la:
                return idx
        return (nearest_idx + 1) % self.n_waypoints

    def control(self, obs):
        x = obs['poses_x'][0]
        y = obs['poses_y'][0]
        theta = obs['poses_theta'][0]
        current_speed = abs(obs['linear_vels_x'][0])

        nearest_idx = self._find_nearest(x, y)
        if self.nearest_dist > self.max_track_deviation:
            return None

        lookahead_idx = self._find_lookahead_point(
            x, y, nearest_idx, current_speed)

        lx = self.waypoints_xy[lookahead_idx, 0]
        ly = self.waypoints_xy[lookahead_idx, 1]

        dx = lx - x
        dy = ly - y
        local_x = dx * np.cos(theta) + dy * np.sin(theta)
        local_y = -dx * np.sin(theta) + dy * np.cos(theta)

        ld_sq = local_x * local_x + local_y * local_y
        if ld_sq < 1e-6:
            steer = 0.0
        else:
            steer = np.arctan2(2.0 * self.wheelbase * local_y, ld_sq)

        steer = np.clip(steer, -0.4189, 0.4189)

        speed = float(self.waypoints_v[nearest_idx])
        if self.nearest_dist > 0.5:
            speed *= max(0.3, 1.0 - self.nearest_dist / self.max_track_deviation)
        speed = max(speed, 0.5)

        return steer, speed

    def lap_completed(self, start_idx, step):
        if step < 500:
            return False
        progress = (self.nearest_idx - start_idx) % self.n_waypoints
        return progress < 50 and step > 1000


# ---------------------------------------------------------------------------
#  Episode collection
# ---------------------------------------------------------------------------

def collect_episode(env, tracker, start_x, start_y, start_psi,
                    start_idx, max_sim_steps=30000):
    """Collect one lap. Returns list of [x, y, vx_w, vy_w] dicts or None."""
    obs, _, done, _ = env.reset(np.array([[start_x, start_y, start_psi]]))
    if obs['collisions'][0] > 0:
        return None

    tracker.nearest_idx = start_idx
    trajectory = []

    for step in range(max_sim_steps):
        x = float(obs['poses_x'][0])
        y = float(obs['poses_y'][0])
        theta = float(obs['poses_theta'][0])
        vx_body = float(obs['linear_vels_x'][0])
        vy_body = float(obs['linear_vels_y'][0])

        if np.isnan(x) or np.isnan(vx_body) or abs(theta) > 1e6:
            return None

        vx_w = vx_body * np.cos(theta) - vy_body * np.sin(theta)
        vy_w = vx_body * np.sin(theta) + vy_body * np.cos(theta)
        trajectory.append({
            'x': x, 'y': y, 'vx_w': vx_w, 'vy_w': vy_w,
        })

        if tracker.lap_completed(start_idx, step):
            return trajectory
        if obs['collisions'][0] > 0:
            return None

        result = tracker.control(obs)
        if result is None:
            return None
        steer, speed = result
        obs, _, done, _ = env.step(np.array([[steer, speed]]))

    return None


# ---------------------------------------------------------------------------
#  Worker
# ---------------------------------------------------------------------------

_counter = None
_counter_lock = None


def _init_counter(counter, lock):
    global _counter, _counter_lock
    _counter = counter
    _counter_lock = lock


def worker_fn(task):
    """Worker: create env, blend paths, run episodes, save CSVs."""
    track_name = task['track_name']
    env = F110Env(
        map=task['map_path'],
        map_ext='.png',
        num_agents=1,
        timestep=task['sim_dt'],
        integrator=Integrator.RK4,
    )
    raceline = load_raceline(task['track_dir'], track_name)
    centerline = load_centerline(task['track_dir'], track_name)

    # Pre-match centerline to raceline (once per worker, reused across jobs)
    matched_cl_xy = match_centerline_to_raceline(raceline, centerline)

    success = 0
    fail = 0

    for speed_scale, lookahead, seed, alpha in task['jobs']:
        np.random.seed(seed)

        # Create blended reference path
        ref_path = blend_reference_path(raceline, matched_cl_xy, alpha)

        # Fixed start at S/F line (centerline origin)
        sf_x, sf_y = 0.0, 0.0
        # Find the closest raceline waypoint to (0,0) for heading reference
        cl = centerline
        sf_heading = np.arctan2(cl['y'][1] - cl['y'][0], cl['x'][1] - cl['x'][0])
        # Find closest ref path index to (0,0) for tracker initialization
        ref_xy = np.column_stack([ref_path['x'], ref_path['y']])
        sf_dists = np.hypot(ref_xy[:, 0] - sf_x, ref_xy[:, 1] - sf_y)
        sf_ref_idx = int(np.argmin(sf_dists))

        tracker = PurePursuitTracker(
            ref_path,
            lookahead_dist=lookahead,
            speed_scale=speed_scale,
        )

        traj = collect_episode(
            env, tracker, sf_x, sf_y, sf_heading, sf_ref_idx,
            max_sim_steps=task['max_sim_steps'],
        )

        if traj is not None and len(traj) >= 100:
            with _counter_lock:
                idx = _counter.value
                _counter.value += 1

            fname = os.path.join(
                task['log_dir'],
                f'{track_name}_a{alpha:.2f}_l{lookahead:.1f}'
                f'_{idx:05d}.csv'
            )
            with open(fname, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f, fieldnames=['x', 'y', 'vx_w', 'vy_w'])
                writer.writeheader()
                writer.writerows(traj)
            success += 1
        else:
            fail += 1

    env.close()
    return track_name, success, fail


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

TRACKS = ['Budapest', 'Catalunya']

SPEED_SCALE = 0.8
LOOKAHEAD_DISTS = [0.5, 0.8, 1.2, 1.8]


def main():
    parser = argparse.ArgumentParser(
        description='Collect F1TENTH raceline trajectories (parallel)')
    parser.add_argument('--track_dir', type=str, default=RACETRACKS_DIR,
                        help='Clone of f1tenth_racetracks')
    parser.add_argument('--tracks', type=str, nargs='+', default=TRACKS)
    parser.add_argument('--n_episodes', type=int, default=1000,
                        help='Total episodes to collect')
    parser.add_argument('--sim_dt', type=float, default=0.01)
    parser.add_argument('--max_sim_steps', type=int, default=30000)
    parser.add_argument('--log_dir', type=str, default=os.path.join(PROJECT_ROOT, 'logs'),
                        help='Output directory for the CSV files')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--speed_scale', type=float, default=SPEED_SCALE,
                        help='Fixed speed scale (1.0 = raceline speed)')
    parser.add_argument('--n_workers', type=int, default=32,
                        help='Number of parallel worker processes')
    parser.add_argument('--jobs_per_worker', type=int, default=20,
                        help='Episodes each worker attempts')
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    eps_per_track = args.n_episodes // len(args.tracks)
    attempts_per_track = int(eps_per_track * 1.5)

    print(f'=== F1TENTH Trajectory Collection (Alpha Blending) ===')
    print(f'Tracks: {args.tracks}')
    print(f'Speed scale: {args.speed_scale}x (fixed)')
    print(f'Lookaheads: {LOOKAHEAD_DISTS}')
    print(f'Alpha blending: Uniform(0, 1) raceline<->centerline')
    print(f'Target: {eps_per_track} episodes/track, {args.n_episodes} total')
    print(f'Workers: {args.n_workers}, jobs/worker: {args.jobs_per_worker}')
    print(f'Sim+Record: {1/args.sim_dt:.0f}Hz')
    print()

    counter = Value('i', 0)
    lock = Lock()

    total_success = 0
    total_fail = 0

    for track_name in args.tracks:
        raceline = load_raceline(args.track_dir, track_name)
        n_wps = len(raceline['x'])
        map_path = get_map_path(args.track_dir, track_name)

        print(f'--- {track_name}: {n_wps} wps, '
              f'{raceline["s"][-1]:.0f}m, '
              f'v_max={raceline["vx"].max():.1f}m/s ---')

        # Generate jobs: cycle lookaheads, random alpha and per-job seed. A start
        # index and a lateral offset are drawn as well (and discarded) so that the
        # random stream matches the one used to collect the paper data.
        all_jobs = []
        for i in range(attempts_per_track):
            la = LOOKAHEAD_DISTS[i % len(LOOKAHEAD_DISTS)]
            rng.randint(0, n_wps)
            seed = int(rng.randint(0, 2**31))
            alpha = float(rng.uniform(0.0, 1.0))
            rng.uniform(-0.5, 0.5)
            all_jobs.append((args.speed_scale, la, seed, alpha))

        # Split into worker tasks
        n_tasks = max(1, min(
            args.n_workers, len(all_jobs) // args.jobs_per_worker))
        chunks = np.array_split(range(len(all_jobs)), n_tasks)

        tasks = []
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            tasks.append({
                'track_name': track_name,
                'track_dir': args.track_dir,
                'map_path': map_path,
                'sim_dt': args.sim_dt,
                'max_sim_steps': args.max_sim_steps,
                'log_dir': args.log_dir,
                'jobs': [all_jobs[j] for j in chunk],
            })

        with Pool(processes=n_tasks,
                  initializer=_init_counter,
                  initargs=(counter, lock)) as pool:
            results = pool.map(worker_fn, tasks)

        track_ok = sum(r[1] for r in results)
        track_fail = sum(r[2] for r in results)
        total_success += track_ok
        total_fail += track_fail

        rate = track_fail / max(track_ok + track_fail, 1) * 100
        print(f'  {track_name}: {track_ok} episodes, '
              f'{track_fail} fails ({rate:.0f}% fail)')

    print(f'\n=== Done ===')
    print(f'{total_success} episodes saved to {args.log_dir}')
    total = total_success + total_fail
    if total > 0:
        print(f'Success rate: {total_success/total*100:.1f}%')


if __name__ == '__main__':
    main()
