"""
plan_loco.py

Evaluate the FM family on Hopper (receding-horizon planning, one plan per env step):
    FM               --integrator pure --safety_enabled False
    SafeFM           --integrator pure --safety_enabled True
    FlowMatcher      --integrator pc   --safety_enabled False
    SafeFlowMatcher  --integrator pc   --safety_enabled True

FM integrates the learned flow with K Euler steps. FlowMatcher first makes a one-shot
prediction of the trajectory (one flow step over the whole interval) and then corrects it
with K steps. The Safe variants apply a CBF on the torso height to every integration step.

The flow-matching model and the value function are read from
<loadbase>/<dataset>/<diffusion_loadpath>/ and <loadbase>/<dataset>/<value_loadpath>/.
Every episode runs the full `max_episode_length` steps: gym's `done` (healthy check) does
not end the episode, matching scripts/eval_sfp.py. With safety enabled, the episode ends at
the first torso-height ceiling violation (rootz > 1.5). Everything that is sampled (python,
numpy, torch, env resets) is seeded from --seed (default 42).

Example:
    python -u scripts/plan_loco.py --integrator pc --safety_enabled True --n_eval_episodes 100 --loadbase logs
"""
import time
import numpy as np
import diffuser.utils as utils
from diffuser.sampling.policies import GuidedPolicy
from diffuser.models.cbf_loco import CBFLoco
from diffuser.utils.setup import set_seed
from diffuser.utils.serialization import load_experiment

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    # Other options (seed, loadbase, diffusion_loadpath, value_loadpath, max_episode_length,
    # safety_enabled, guidance_scale, ...) come from config/locomotion.py:cfm['plan'] and
    # can be overridden on the command line, e.g. --safety_enabled True.
    dataset: str = 'hopper-medium-expert-v2'
    config: str = 'config.locomotion'
    method: str = 'cfm'

    # FM / FlowMatcher switch
    integrator: str = 'pc'          # 'pure' = FM (Euler), 'pc' = FlowMatcher (one-shot prediction + correction)
    n_eval_episodes: int = 5

args = Parser().parse_args('plan')
assert 'hopper' in args.dataset, 'this release covers Hopper only'
HEIGHT_LIMIT = 1.5  # torso-height ceiling (physical space)

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

loadbase = args.loadbase or args.logbase

## flow-matching model
dataset, diffusion = load_experiment(
    loadbase, args.dataset, args.diffusion_loadpath, args.diffusion_epoch)

## value function for reward guidance
_, value_model = load_experiment(
    loadbase, args.dataset, args.value_loadpath, args.value_epoch)

# Reward guidance of the flow field with the learned value function
diffusion.enable_guidance(value_model=value_model, scale=args.guidance_scale)

# Select integrator: 'pure' = FM (Euler), 'pc' = FlowMatcher
diffusion.integrator = args.integrator
# FlowMatcher / SafeFlowMatcher use the one-shot prediction stage before the correction steps
diffusion.one_shot_enabled = (args.integrator == 'pc')
print(f"[ plan_loco ] integrator={diffusion.integrator} "
      f"({'FM' if args.integrator == 'pure' else 'FlowMatcher'})")

# Safety (SafeFM / SafeFlowMatcher): CBF on the torso height during sampling
if args.safety_enabled:
    _obs_norm = dataset.normalizer.normalizers['observations']
    diffusion.cbf = CBFLoco(
        height=args.height,
        norm_mins=_obs_norm.mins,
        norm_maxs=_obs_norm.maxs,
        height_idx=args.height_idx,
        epsilon=args.cbf_epsilon,
        rho=args.cbf_rho,
    )
    diffusion.safety_enabled = True
    print(f"[ plan_loco ] safety enabled: height={args.height}, height_idx={args.height_idx}")

policy = GuidedPolicy(diffusion_model=diffusion, normalizer=dataset.normalizer)

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

env = dataset.env

scores = []          # normalized score x100, per episode
violated = []        # 1 if the executed torso height exceeded the ceiling
t_opt_all = []       # per env step (s): CBF time of one plan
step_time_all = []   # per env step (s)
n_episodes = args.n_eval_episodes

set_seed(args.seed)
env.seed(args.seed)

for kk in range(n_episodes):
    observation = env.reset()
    total_reward = 0
    first_done_t = None
    ep_violated = False

    for t in range(args.max_episode_length):
        ## format current observation for conditioning
        conditions = {0: observation}

        start = time.time()
        action, timing_avg = policy(conditions, batch_size=1, verbose=(t == 0))
        step_time_all.append(time.time() - start)
        # timing_avg = [iter_time, nn_time, safety_time] averaged over integrator steps;
        # multiply by n_timesteps for the per-env-step total
        if args.safety_enabled:
            t_opt_all.append(timing_avg[2] * diffusion.n_timesteps)

        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward

        # gym's `done` is ignored (the episode runs the full max_episode_length);
        # only the first done step is reported.
        if terminal and first_done_t is None:
            first_done_t = t
            print(f"  [done-ignored] ep {kk} first done at t={t}, "
                  f"rootz={next_observation[0]:.3f}", flush=True)

        # Safety violation (executed torso height above the ceiling) ends the episode
        if args.safety_enabled and next_observation[0] > HEIGHT_LIMIT:
            ep_violated = True
            break

        observation = next_observation

    score = 100 * env.get_normalized_score(total_reward)
    scores.append(score)
    line = (f"  ep {kk + 1}/{n_episodes} | steps={t + 1} | R={total_reward:.2f} | score={score:.2f} | "
            f"{np.mean(step_time_all[-(t + 1):])*1000:.1f}ms/step")
    if args.safety_enabled:
        violated.append(int(ep_violated))
        line += f" | viol={int(ep_violated)}"
    print(line, flush=True)

#-----------------------------------------------------------------------------#
#------------------------- summary: Table 4 columns --------------------------#
#-----------------------------------------------------------------------------#

scores = np.array(scores)
name = ('Safe' if args.safety_enabled else '') + ('FM' if args.integrator == 'pure' else 'FlowMatcher')
print(f"\n{'='*60}")
print(f"  {name} | {args.dataset}")
print(f"{'='*60}")
print(f"  Episodes:      {n_episodes}")
print(f"  Score:         {scores.mean():.2f} +/- {scores.std():.2f}")
if args.safety_enabled:
    print(f"  Viol (%):      {100 * np.mean(violated):.1f}  ({int(np.sum(violated))}/{len(violated)} episodes)")
    print(f"  t_Opt (ms):    {np.mean(t_opt_all)*1000:.3f}")
print(f"  t_total (ms):  {np.mean(step_time_all)*1000:.2f}")
print(f"{'='*60}")
