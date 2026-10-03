from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names by labelling folders with these args

train_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]

sfp_args_to_watch = [
    ('prefix', ''),
    ('k', 'k'),
    ('sigma_train', 's'),
]

plan_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('closed_loop', 'closed'),
]

#-- safety defaults (shared by the sfp / cfm / base plan configs) --#
_safety_defaults = {
    'safety_enabled': False,

    ## Diffuser family: 'gd' = Diffuser+CG, 'invariance' = SafeDiffuser (ReS, closed form)
    'safety_method': 'invariance',
    ## FM family: 'pure' = FM / SafeFM, 'pc' = FlowMatcher / SafeFlowMatcher (prediction-correction)
    'integrator': 'pc',
    'one_shot_enabled': False,     # FlowMatcher: one-step prediction stage before the correction stage

    ## obstacles (set per map below)
    'obstacles': [],

    ## class-K CBF of SafeDiffuser / SafeFM / SafeFlowMatcher (diffuser/models/cbf_diffuser.py, diffusion.py)
    'eps': 1.0,                    # finite-time class-K gain
    'rho': 0.5,                    # finite-time class-K exponent
    'relax_threshold': 0.95,       # relaxation active for t <= relax_threshold (SafeFM / SafeFlowMatcher)
    ## robustness margin of the barrier: SSF / A* / RRT* use 0.01, the safe baselines are run with 0.1
    'robust_term': 0.01,

    ## DT-HOCBF of SSF (diffuser/models/cbf.py), class-K coefficients, 0 < kv <= kp <= 1
    'kp_hocbf': 0.15,
    'kv_hocbf': 0.08,

    ## visualization of the per-episode trajectories
    'show_correction_arrows': True,
}

base = {

    'train': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 256,
        'n_diffusion_steps': 256,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),

        ## dataset
        'loader': 'datasets.GoalDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(train_args_to_watch),

        ## training
        'n_steps_per_epoch': 5000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 64,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 1,
        'ema_decay': 0.995,
        'save_freq': 50000,
        'n_saves': 50,
        'device': 'cuda',
    },

    'plan': {
        'batch_size': 1,
        'device': 'cuda',
        'seed': 42,

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,

        ## serialization
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': 'test',

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',

        **_safety_defaults,
    },

}

sfp = {

    'train': {
        'seed': 42,
        ## dataset
        'loader': 'datasets.SequenceDataset',
        'horizon': 256,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'use_padding': False,
        'max_path_length': 40000,
        'termination_penalty': None,

        ## serialization
        'logbase': 'logs',
        'prefix': 'sfp/',
        'exp_name': watch(sfp_args_to_watch),

        ## streaming flow policy parameters
        'k': 0.1,
        'sigma_train': 0.005,
        'device': 'cuda',

        ## training
        'batch_size': 256,
        'learning_rate': 1e-4,
        'weight_decay': 1e-6,
        'ema_decay': 0.995,
        'n_epochs': 50,
        'num_workers': 4,
        'model_filename': 'sfp_velocity_policy.pt',
    },
    'plan': {
        'batch_size': 1,
        'device': 'cuda',
        'seed': 42,

        ## model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'k': 0.1,
        'sigma_train': 0.005,
        'closed_loop': False,

        ## serialization
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': 'test',

        ## loading
        'diffusion_loadpath': 'f:sfp/k{k}_s{sigma_train}',
        'checkpoint': 'sfp_velocity_policy.pt',

        **_safety_defaults,
    },
}

cfm = {

    'train': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.CFM',
        'horizon': 256,
        'n_diffusion_steps': 256,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),

        ## dataset
        'loader': 'datasets.GoalDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'cfm/',
        'exp_name': watch(train_args_to_watch),

        ## training
        'n_steps_per_epoch': 5000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 64,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 1,
        'ema_decay': 0.995,
        'save_freq': 50000,
        'n_saves': 50,
        'device': 'cuda',
    },

    'plan': {
        'batch_size': 1,
        'device': 'cuda',
        'seed': 42,

        ## cfm model
        'horizon': 256,
        'n_diffusion_steps': 256,

        ## serialization
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': 'test',

        ## loading
        'diffusion_loadpath': 'f:cfm/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',

        **_safety_defaults,
    },

}

#------------------------ overrides ------------------------#

'''
    Obstacle format:
        {'order': n, 'center': (cx, cy), 'radius': r}
        - order: 2 = ellipse, 4 = superellipse (squircle), higher = more rectangular
        - center: (cx, cy); in the observation (qpos) frame the centre is
          (s0, s1) = (cy + offset, cx + offset), offset = -0.5 on large and -0.7 on umaze / medium
          (diffuser/models/cbf.py, diffuser/utils/safety_metrics.py)
        - radius: size of obstacle in maze units
        - For non-uniform radii, use 'radius_x' and 'radius_y' instead of 'radius'
'''

maze2d_umaze_v1 = {
    'train': {
        'horizon': 128,
        'n_diffusion_steps': 64,
    },
    'plan': {
        'horizon': 128,
        'n_diffusion_steps': 64,

        ## one circular obstacle around the centre wall protrusion (5x5 maze, corridors ~1 unit wide)
        'obstacles': [
            {'order': 2, 'center': (2.5, 2.5), 'radius': 0.9},
        ],
    },
}

maze2d_medium_v1 = {
    'train': {
        'horizon': 256,
        'n_diffusion_steps': 256,
    },
    'plan': {
        'horizon': 256,
        'n_diffusion_steps': 256,

        ## obstacles for medium (8x8 maze)
        'obstacles': [
            {'order': 2, 'center': (3.5, 4.5), 'radius': 0.8},   # overlaps the wall block at x≈2
            {'order': 2, 'center': (5.5, 3.5), 'radius': 0.8},   # overlaps the walls in the upper right
        ],
    },
}

maze2d_large_v1 = {
    'train': {
        'horizon': 384,
        'n_diffusion_steps': 256,
    },
    'plan': {
        'horizon': 384,
        'n_diffusion_steps': 256,

        ## obstacles for large (9x12 maze)
        'obstacles': [
            {'order': 2, 'center': (5.6, 4.8), 'radius': 1},
            {'order': 2, 'center': (5.1, 1.8), 'radius': 1},
        ],
    },
}
