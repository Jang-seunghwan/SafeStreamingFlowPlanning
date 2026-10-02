from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names by labelling folders with these args
args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
]

sfp_args_to_watch = [
    ('prefix', ''),
    ('k', 'k'),
    ('sigma_train', 's'),
]

logbase = 'logs'

## Diffuser family (Diffuser, Diffuser+CG, SafeDiffuser): GaussianDiffusion trajectory model.
## Training defaults are the settings of the Diffuser model used in the paper.
base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 600,
        'n_diffusion_steps': 20,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'dim_mults': (1, 2, 4, 8),
        'time_scale': 1.0,

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'GaussianNormalizer',
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 2500,
        'loss_type': 'l2',
        'n_train_steps': 2.5e5,
        'batch_size': 128,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 1,
        'ema_decay': 0.995,
        'save_freq': 20000,
        'n_saves': 5,
        'device': 'cuda',
        'seed': 42,
    },
}


## FM family (FM, SafeFM, FlowMatcher, SafeFlowMatcher): CFM trajectory model + value function.
## The value function is also used for the reward guidance of the Diffuser family.
cfm = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.CFM',
        'horizon': 600,
        'n_diffusion_steps': 20,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'dim_mults': (1, 2, 4, 8),
        'time_scale': 20.0,

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': logbase,
        'prefix': 'cfm/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 1,
        'ema_decay': 0.995,
        'save_freq': 20000,
        'n_saves': 5,
        'device': 'cuda',
        'seed': 42,
    },

    'values': {
        'model': 'models.ValueFunction',
        'diffusion': 'models.ValueCFM',
        'horizon': 600,
        'n_diffusion_steps': 20,
        'dim_mults': (1, 2, 4, 8),

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': -100,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'LimitsNormalizer',
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l1',
        'n_train_steps': 2e5,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 1,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'n_saves': 5,
        'device': 'cuda',
        'seed': 42,
    },

    ## scripts/plan_loco.py
    'plan': {
        'max_episode_length': 1000,
        'device': 'cuda',
        'seed': 42,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/cfm',
        'exp_name': watch(args_to_watch),

        ## model / value function (loaded from <loadbase>/<dataset>/<loadpath>)
        'horizon': 600,
        'n_diffusion_steps': 20,
        'discount': 0.99,
        'diffusion_loadpath': 'f:cfm/defaults_H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}',
        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        ## value guidance of the flow field
        'guidance_scale': 1.0,

        ## safety (SafeFM / SafeFlowMatcher): CBF on the torso height during sampling
        'safety_enabled': False,
        'height': 1.5,             # torso-height ceiling (physical space)
        'height_idx': 3,           # torso height in the [action, observation] layout
        'cbf_epsilon': 1.0,        # CBF class-K parameter
        'cbf_rho': 0.99,           # CBF finite-time exponent
    },
}


## scripts/eval_sfp.py --method safediffuser
safediffuser = {
    'plan': {
        'max_episode_length': 1000,
        'device': 'cuda',
        'seed': 42,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/diffuser',
        'exp_name': watch(args_to_watch),

        ## model / value function (loaded from <loadbase>/<dataset>/<loadpath>)
        'horizon': 600,
        'n_diffusion_steps': 20,
        'discount': 0.99,
        'diffusion_loadpath': 'f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}',

        ## value-guided sampling (SafeDiffuser defaults)
        'n_guide_steps': 2,
        't_stopgrad': 4,
        'scale_grad_by_std': True,
    },
}


## StreamingFlow / SSF: hierarchical streaming flow policy,
## trained with scripts/train_sfp.py and evaluated with scripts/eval_sfp.py --method sfp
sfp = {
    'train': {
        ## dataset
        'loader': 'datasets.SequenceDataset',
        'horizon': 1000,
        'normalizer': 'LimitsNormalizer',
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': logbase,
        'prefix': 'sfp/',
        'exp_name': watch(sfp_args_to_watch),

        ## training-tube parameters: sigma_t = sigma_train * exp(-k t)
        'k': 2.0,
        'sigma_train': 1e-4,
        'device': 'cuda',
        'seed': 42,

        ## velocity training
        'batch_size': 512,
        'learning_rate': 1e-4,
        'weight_decay': 1e-6,
        'ema_decay': 0.995,
        'n_epochs': 50,
        'num_workers': 8,
        'model_filename': 'sfp_velocity_policy.pt',
    },
    'plan': {
        'max_episode_length': 1000,
        'device': 'cuda',
        'seed': 42,
        'horizon': 1000,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/sfp',
        'exp_name': watch(args_to_watch),
    },
}
