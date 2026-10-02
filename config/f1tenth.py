"""
F1TENTH experiment configuration: tracks, obstacles, safety parameters, the
method table used by scripts/eval_all.py, and data / checkpoint locations.

Every track is trained and evaluated separately with its own normalizer.
"""
import os

# Repository root (parent of config/).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Clone of https://github.com/f1tenth/f1tenth_racetracks (centerline / raceline CSVs).
RACETRACKS_DIR = os.environ.get('F1TENTH_RACETRACKS',
                                os.path.join(PROJECT_ROOT, 'f1tenth_racetracks'))

TRACKS = ['budapest', 'catalunya']

# CLI track name -> directory / map name.
TRACK_DIR_MAP = {'budapest': 'Budapest', 'catalunya': 'Catalunya'}

# Track index used to seed the evaluation start/goal noise: rng = default_rng([seed, TRACK_ID]).
TRACK_ID = {'budapest': 0, 'catalunya': 1}

# Per-track settings.
#   horizon:   median subsampled (10 Hz) lap length, rounded to a multiple of 4.
#   obstacles: super-ellipse obstacles  |dx/rx|^n + |dy/ry|^n >= 1  (world frame, metres).
TRACK_CONFIGS = {
    'budapest': {
        'horizon': 696,
        'obstacles': [
            {'center': (16.1, -12.1), 'radius': 0.40, 'order': 2},
        ],
    },
    'catalunya': {
        'horizon': 724,
        'obstacles': [
            {'center': (-65.49, -67.39), 'radius': 0.3, 'order': 2},
            {'center': (0.54, 19.95), 'radius': 0.25, 'order': 2},
            {'center': (-50.45, -30.50), 'radius': 0.25, 'order': 2},
        ],
    },
}

# Preprocessed training data (scripts/regenerate_processed_data.py). The velocity
# channels are stored in two unit conventions; each model family was trained on one:
#   m_per_s:    velocity interpolated from the recorded laps (m/s)
#   m_per_step: forward difference of the resampled positions (m per 0.1 s step)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'processed_data')
VELOCITY_UNITS = ('m_per_step', 'm_per_s')


def get_processed_path(track, velocity_units):
    """Path of the preprocessed .npz of a track for one velocity convention."""
    assert velocity_units in VELOCITY_UNITS, velocity_units
    return os.path.join(PROCESSED_DIR, velocity_units, f'{track}.npz')


def get_horizon(track):
    """Return the track-specific horizon."""
    return TRACK_CONFIGS[track]['horizon']


# Safety (CBF) parameters shared across methods.
SAFETY_DEFAULTS = {
    # Barrier robustness margin kappa: b = |dx/rx|^n + |dy/ry|^n - (1 + kappa).
    'robust_term': 0.01,
    'center_offset': 0.0,
    'pos_x_idx': 0,  # x at state dim 0
    'pos_y_idx': 1,  # y at state dim 1
    # Relaxed 1st-order CBF-QP (SafeFM / SafeFlowMatcher).
    'eps': 1.0,
    'rho': 0.5,
    'relax_threshold': 0.9,
    # 2nd-order ECBF-QP of SSF (safe_sfp_off).
    'alpha_cbf': 10.0,      # pole placement (s + alpha)^2 -> alpha_0 = alpha^2, alpha_1 = 2 alpha
    'ecbf_threshold': 5.0,  # an obstacle's constraint is active when its barrier value < threshold
    'n_sub_cbf': 10,        # ECBF sub-steps per trajectory step (dt_sub = 0.01 s)
    'u_max_cbf': 200.0,     # box bound on the QP acceleration
}

# method -> (model_type, safety_enabled, integrator, safety_method)
#   integrator:    None (Diffuser / SFP) | 'pure' (FM, Euler) | 'pc' (FlowMatcher, predictor-corrector)
#   safety_method: 'none' | 'invariance' (SafeDiffuser) | 'gd' (classifier guidance) | 'cbf'
METHOD_CONFIG = {
    'diffuser':         ('diffuser', False, None,   'none'),
    'diffuser_cg':      ('diffuser', True,  None,   'gd'),
    'safediffuser':     ('diffuser', True,  None,   'invariance'),
    'fm':               ('cfm',      False, 'pure', 'none'),
    'safefm':           ('cfm',      True,  'pure', 'cbf'),
    'flowmatcher':      ('cfm',      False, 'pc',   'none'),
    'safeflowmatcher':  ('cfm',      True,  'pc',   'cbf'),
    'sfp_off':          ('sfp',      False, None,   'none'),   # StreamingFlow (open loop)
    'safe_sfp_off':     ('sfp',      True,  None,   'cbf'),    # SSF (open loop)
}

# Checkpoint layout under a checkpoint root (written by the training scripts' defaults):
#   diffuser/<track>/checkpoint_final.pt      Diffuser, Diffuser+CG, SafeDiffuser   (m_per_step data)
#   cfm/<track>/checkpoint_final.pt           FM, FlowMatcher, SafeFM/SafeFlowMatcher on Budapest (m_per_step)
#   cfm_m_per_s/<track>/checkpoint_final.pt   SafeFM, SafeFlowMatcher on Catalunya  (m_per_s data)
#   sfp/<track>/sfp_velocity_policy.pt        StreamingFlow, SSF                    (m_per_s data)
CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, 'checkpoints')


def checkpoint_path(method, track, root=CHECKPOINT_ROOT):
    """Checkpoint used for a table row (method, track)."""
    model_type = METHOD_CONFIG[method][0]
    if model_type == 'sfp':
        return os.path.join(root, 'sfp', track, 'sfp_velocity_policy.pt')
    family = model_type
    if method in ('safefm', 'safeflowmatcher') and track == 'catalunya':
        family = 'cfm_m_per_s'
    return os.path.join(root, family, track, 'checkpoint_final.pt')
