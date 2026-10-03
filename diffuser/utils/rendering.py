import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt

from diffuser.datasets.d4rl import load_environment

#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#

def zipsafe(*args):
    length = len(args[0])
    assert all([len(a) == length for a in args])
    return zip(*args)

def zipkw(*args, **kwargs):
    nargs = len(args)
    keys = kwargs.keys()
    vals = [kwargs[k] for k in keys]
    zipped = zipsafe(*args, *vals)
    for items in zipped:
        zipped_args = items[:nargs]
        zipped_kwargs = {k: v for k, v in zipsafe(keys, items[nargs:])}
        yield zipped_args, zipped_kwargs

def plot2img(fig, remove_margins=True):
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    DPI = 100
    fig.set_dpi(DPI)

    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.frombuffer(img_as_string, dtype='uint8').reshape((height, width, 4))

#-----------------------------------------------------------------------------#
#----------------------------------- maze2d ----------------------------------#
#-----------------------------------------------------------------------------#

MAZE_BOUNDS = {
    'maze2d-umaze-v1': (0, 5, 0, 5),
    'maze2d-medium-v1': (0, 8, 0, 8),
    'maze2d-large-v1': (0, 9, 0, 12),
}

class Maze2dRenderer:
    """Draws Maze2D trajectories (walls, obstacles, goal, optional safety-filter correction arrows)."""

    def __init__(self, env):
        self.env_name = env
        self.env = load_environment(env)
        self.goal = None
        self._background = self.env.maze_arr == 10
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)
        self._rendered_obstacles = []

    def set_obstacles(self, obstacles):
        """Transform config obstacle coordinates to rendered coordinates.

        Rendered coords = (physical + 0.7) / scale
        Physical center = config_center + center_offset
        center_offset: -0.5 for large, -0.7 for umaze/medium
        """
        _, iscale, _, jscale = MAZE_BOUNDS[self.env_name]

        # Same offset logic as CBF: large=-0.5, others=-0.7
        center_offset = -0.5 if 'large' in self.env_name else -0.7
        render_offset = 0.7  # observations += 0.7 in renders()

        self._rendered_obstacles = []
        for obs in obstacles:
            cx, cy = obs['center']
            order = obs.get('order', 2)
            r = obs.get('radius', 0.5)
            rx = obs.get('radius_x', r)
            ry = obs.get('radius_y', r)
            # config center → physical center → rendered center
            phys_cx = cx + center_offset
            phys_cy = cy + center_offset
            self._rendered_obstacles.append({
                'cx': (phys_cx + render_offset) / jscale,
                'cy': (phys_cy + render_offset) / iscale,
                'rx': rx / jscale,
                'ry': ry / iscale,
                'order': order,
            })

    def _draw_obstacles(self):
        """Draw all stored obstacles as superellipses on the current matplotlib axes."""
        theta = np.linspace(0, 2 * np.pi, 200)
        for obs in self._rendered_obstacles:
            exp = 2.0 / obs['order']
            x = obs['rx'] * np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** exp + obs['cx']
            y = obs['ry'] * np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** exp + obs['cy']
            plt.plot(x, y, c='red', zorder=10)
            plt.fill(x, y, 'red', alpha=0.3)

    def _draw_correction_arrows(self, corrections, arrow_stride=10):
        """Draw the safety-filter correction (after - before) as black arrows, every `arrow_stride` steps,
        only near obstacles.

        Args:
            corrections: list of dicts with 'current', 'before', 'after' keys (rendered coordinates)
        """
        NEAR_FACTOR = 1.6
        obstacles = self._rendered_obstacles

        def _near_obstacle(cx, cy):
            if not obstacles:
                return True
            for o in obstacles:
                rad = max(o.get('rx', 0.05), o.get('ry', 0.05)) * NEAR_FACTOR
                if (cx - o['cx']) ** 2 + (cy - o['cy']) ** 2 <= rad ** 2:
                    return True
            return False

        for i, corr in enumerate(corrections):
            if corr is None or i % arrow_stride != 0:
                continue

            current = corr['current']  # [py, px, vy, vx]
            before = corr['before']    # before CBF correction
            after = corr['after']      # after CBF correction

            # Current position (note: x=index 1, y=index 0 for plotting)
            curr_x = current[1]
            curr_y = current[0]
            if not _near_obstacle(curr_x, curr_y):
                continue

            correction_dx = (after[1] - before[1])
            correction_dy = (after[0] - before[0])
            arrow_scale = 36.0  # for visibility
            if abs(correction_dx) > 1e-6 or abs(correction_dy) > 1e-6:
                plt.arrow(curr_x, curr_y, correction_dx * arrow_scale, correction_dy * arrow_scale,
                         head_width=0.015, head_length=0.008, fc='black', ec='black',
                         alpha=0.9, zorder=26, linewidth=0.75)

    def renders(self, observations, corrections=None, show_correction_arrows=True):
        """observations: [horizon x obs_dim] (py, px, ...). The goal drawn is self.goal (set by the harness)."""
        _, iscale, _, jscale = MAZE_BOUNDS[self.env_name]

        observations = observations + .7    # rendering offset
        observations[:, 0] /= iscale
        observations[:, 1] /= jscale

        goal_plot = None
        if self.goal is not None:
            goal_plot = np.array(self.goal, dtype=np.float32).reshape(-1)[:2] + 0.7
            goal_plot[0] /= iscale
            goal_plot[1] /= jscale

        scaled_corrections = None
        if corrections is not None and len(corrections) > 0:
            scaled_corrections = []
            for corr in corrections:
                if corr is None:
                    scaled_corrections.append(None)
                    continue
                scaled_corr = {}
                for key in ['current', 'before', 'after']:
                    val = corr[key].copy()
                    val[:2] += 0.7
                    val[0] /= iscale  # y
                    val[1] /= jscale  # x
                    scaled_corr[key] = val
                scaled_corrections.append(scaled_corr)

        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        plt.imshow(self._background * .5,
            extent=self._extent, cmap=plt.cm.binary, vmin=0, vmax=1)

        path_length = len(observations)
        colors = plt.cm.jet(np.linspace(0,1,path_length))
        plt.plot(observations[:,1], observations[:,0], c='black', zorder=10)
        plt.scatter(observations[:,1], observations[:,0], c=colors, zorder=20)

        if show_correction_arrows and scaled_corrections is not None:
            self._draw_correction_arrows(scaled_corrections)

        if goal_plot is not None:
            plt.scatter(goal_plot[1], goal_plot[0], marker='*', s=180, c='gold',
                        edgecolors='black', linewidths=1.0, zorder=30)

        self._draw_obstacles()

        plt.axis('off')
        img = plot2img(fig, remove_margins=self._remove_margins)
        return img

    def composite(self, savepath, paths, ncol=5, show_correction_arrows=True, **kwargs):
        '''
            savepath : str
            paths : [ n_paths x horizon x obs_dim ]
        '''
        assert len(paths) % ncol == 0, 'Number of paths must be divisible by number of columns'

        images = []
        for path, kw in zipkw(paths, **kwargs):
            img = self.renders(*path, show_correction_arrows=show_correction_arrows, **kw)
            images.append(img)
        images = np.stack(images, axis=0)

        nrow = len(images) // ncol
        images = einops.rearrange(images,
            '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
        imageio.imsave(savepath, images)
        print(f'Saved {len(paths)} samples to: {savepath}')
