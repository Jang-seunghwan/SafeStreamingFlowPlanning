"""
eval_sfp.py

Evaluate the streaming flow policies and the Diffuser family on Hopper
(hopper-medium-expert-v2) with a torso-height ceiling (rootz <= 1.5).

    StreamingFlow  --method sfp
    SSF            --method sfp --safety_enable
    Diffuser       --method safediffuser
    Diffuser+CG    --method safediffuser --safety_enable --safety_method cg
    SafeDiffuser   --method safediffuser --safety_enable --safety_method invariance

Checkpoints are read from <loadbase>/<dataset>/: `sfp/<sfp_ckpt>/` for the streaming
policies; `<diffusion_loadpath>/` and `<value_loadpath>/` for the Diffuser family.
Every episode runs the full `max_episode_length` steps: gym's `done` (healthy check)
does not end the episode. With safety enabled, the episode ends at the first ceiling
violation. Everything that is sampled (python, numpy, torch, env resets) is seeded
from --seed (default 42).

Example:
    python -u scripts/eval_sfp.py --method sfp --safety_enable --n_episodes 100 --loadbase logs
"""
import os
import time
import numpy as np
import torch
import diffuser.utils as utils
from diffuser.utils.setup import set_seed
from diffuser.utils.serialization import load_experiment


class Parser(utils.Parser):
    # Options defined in config/locomotion.py (seed, max_episode_length, loadbase,
    # diffusion_loadpath, value_loadpath, ...) can also be overridden on the command line.
    dataset: str = 'hopper-medium-expert-v2'
    config: str = 'config.locomotion'
    method: str = 'sfp'              # 'sfp' (StreamingFlow / SSF) or 'safediffuser' (Diffuser family)

    n_episodes: int = 10
    guide_scale: float = 0.001       # Diffuser family: value-guidance scale (0 disables guidance)
    safety_enable: bool = False      # SSF: HOCBF-QP action filter; Diffuser family: see safety_method
    safety_method: str = 'invariance'  # Diffuser family: 'invariance' (SafeDiffuser QP) or 'cg' (truncate)
    cbf_beta: float = 0.08           # HOCBF velocity look-ahead (SSF)
    cbf_k1: float = 0.15             # HOCBF class-K gain (SSF)
    sfp_ckpt: str = 'k2.0_s0.0001'   # SFP checkpoint dir under <loadbase>/<dataset>/sfp/


args = Parser().parse_args('plan')
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
loadbase = args.loadbase or args.logbase

# Torso-height ceiling (physical space)
assert 'hopper' in args.dataset, 'this release covers Hopper only'
HEIGHT_LIMIT = 1.5
pos_dim, vel_dim = 5, 6          # observation = [qpos[1:] (5), qvel (6)]
obs_dim = pos_dim + vel_dim

# ══════════════════════════════════════════════════════════════
#  Method-specific model loading
# ══════════════════════════════════════════════════════════════

cbf_filter = None
if args.method == 'sfp':
    from diffuser.models.cond_unet1D_loco import ConditionalUnet1DLoco
    from diffuser.datasets.sequence import SequenceDataset

    dataset = SequenceDataset(
        env=args.dataset,
        horizon=args.horizon,
        normalizer='LimitsNormalizer',
        use_padding=True,
        max_path_length=1000,
    )
    normalizer = dataset.normalizer
    act_dim = dataset.action_dim
    env = dataset.env

    # Load SFP velocity model
    ckpt_path = os.path.join(loadbase, args.dataset, 'sfp', args.sfp_ckpt, 'sfp_velocity_policy.pt')
    print(f"[ eval ] Loading SFP: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    horizon = checkpoint['config']['horizon']

    velocity_net = ConditionalUnet1DLoco(
        pos_dim=pos_dim, vel_dim=vel_dim, act_dim=act_dim,
        global_cond_dim=obs_dim * 2,
        fc_timesteps=1, horizon=horizon,
    ).to(device)
    velocity_net.load_state_dict(checkpoint['velocity_state_dict'])
    velocity_net.eval()

    # HOCBF-QP safety filter (SSF)
    if args.safety_enable:
        from diffuser.models.cbf_hopper_sfp import CBFHopperHOCBF
        cbf_filter = CBFHopperHOCBF(
            env=env, z_max=HEIGHT_LIMIT, beta=args.cbf_beta,
            delta=0.01, k1=args.cbf_k1,
        )
        print(f"[ eval ] HOCBF-QP: z_max={HEIGHT_LIMIT}, beta={args.cbf_beta}, k1={args.cbf_k1}, dt={cbf_filter.dt}")

elif args.method == 'safediffuser':
    from diffuser.sampling.policies import GuidedPolicy
    from diffuser.sampling.functions import n_step_guided_p_sample
    from diffuser.sampling.guides import ValueGuide

    print(f"[ eval ] Loading Diffuser: {loadbase}/{args.dataset}/{args.diffusion_loadpath}/")
    sd_dataset, diffusion = load_experiment(loadbase, args.dataset, args.diffusion_loadpath)
    normalizer = sd_dataset.normalizer
    act_dim = sd_dataset.action_dim
    env = sd_dataset.env

    # Value-guided sampling (SafeDiffuser defaults: scale=0.001, n_guide_steps=2, t_stopgrad=4)
    sample_kwargs = {}
    if args.guide_scale > 0:
        print(f"[ eval ] Loading value model: {loadbase}/{args.dataset}/{args.value_loadpath}/")
        _, value_function = load_experiment(loadbase, args.dataset, args.value_loadpath)
        # These kwargs flow: forward -> conditional_sample -> p_sample_loop -> n_step_guided_p_sample
        sample_kwargs = dict(
            guide=ValueGuide(model=value_function).to(device),
            sample_fn=n_step_guided_p_sample,
            n_guide_steps=args.n_guide_steps,
            scale=args.guide_scale,
            t_stopgrad=args.t_stopgrad,
            scale_grad_by_std=args.scale_grad_by_std,
        )
        print(f"[ eval ] Value guidance: scale={args.guide_scale}, n_guide_steps={args.n_guide_steps}")
    else:
        print("[ eval ] No guidance (--guide_scale 0)")

    # The safety filters act in GaussianNormalizer space: pass observation mean/std
    obs_normalizer = normalizer.normalizers['observations']
    diffusion.mean = obs_normalizer.means
    diffusion.std = obs_normalizer.stds
    if args.safety_enable:
        diffusion.safety_enabled = True
        diffusion.safety_method = args.safety_method  # 'invariance' or 'cg'
        print(f"[ eval ] Diffuser safety: {args.safety_method}, z_max={HEIGHT_LIMIT}")

    policy = GuidedPolicy(diffusion_model=diffusion, normalizer=normalizer, **sample_kwargs)

else:
    raise ValueError(f"Unknown method: {args.method}. Choose 'sfp' or 'safediffuser'.")

# ══════════════════════════════════════════════════════════════
#  Evaluation loop
# ══════════════════════════════════════════════════════════════

scores = []           # normalized score x100, per episode
violated = []         # 1 if the executed torso height exceeded the ceiling
h_mins = []           # SSF: per-episode min of the HOCBF barrier h
step_times_all = []   # per env step (s)
t_opt_times_all = []  # per env step (s): safety filter time

set_seed(args.seed)
env.seed(args.seed)

for ep in range(args.n_episodes):
    observation = env.reset()
    total_reward = 0.0
    ep_step_times = []
    ep_h_min = float('inf')
    ep_violated = False

    # SFP: per-episode conditioning
    if args.method == 'sfp':
        dt = 1.0 / (horizon - 1)
        obs_start_norm = normalizer.normalize(observation, 'observations')
        obs_start_t = torch.from_numpy(obs_start_norm).float().to(device)
        end_obs = torch.zeros_like(obs_start_t)
        cond = torch.stack([obs_start_t, end_obs], dim=0)        # (2, obs_dim)
        cond_flat = cond.unsqueeze(0).flatten(start_dim=1)         # (1, 2*obs_dim)
        prev_action_norm = np.zeros(act_dim, dtype=np.float32)

    for t_idx in range(args.max_episode_length):
        t0 = time.time()

        if args.method == 'sfp':
            # ── SFP: 1 velocity_net forward per step ──
            flow_t = min(t_idx * dt, 1.0)

            obs_norm = normalizer.normalize(observation, 'observations')
            state = np.concatenate([obs_norm, prev_action_norm]).astype(np.float32)
            state_t = torch.from_numpy(state).float().to(device).unsqueeze(0)

            with torch.no_grad():
                x_in = state_t.unsqueeze(1)  # (1, 1, state_dim)
                t_tensor = torch.tensor([flow_t], device=device, dtype=torch.float32)
                velocity = velocity_net(
                    sample=x_in, timestep=t_tensor, global_cond=cond_flat,
                ).squeeze(1)  # (1, state_dim)

            # Euler step
            next_state = state_t + velocity * dt

            next_state_np = next_state[0].cpu().numpy()

            # Extract action
            action_norm = next_state_np[obs_dim:]
            prev_action_norm = action_norm.copy()
            action = normalizer.unnormalize(action_norm, 'actions')
            action = np.clip(action, env.action_space.low, env.action_space.high)

            # CBF safety filter (after action extraction, before env.step)
            if cbf_filter is not None:
                _t_opt0 = time.perf_counter()
                action = cbf_filter.apply(observation, action)
                t_opt_times_all.append(time.perf_counter() - _t_opt0)

        else:
            # ── Diffuser: full receding-horizon planning ──
            conditions = {0: observation}
            # Reset the per-plan safety-time accumulator (accumulated during sampling)
            diffusion._safety_time_acc = 0.0
            action, _ = policy(conditions, batch_size=1, verbose=(t_idx == 0 and ep == 0))
            if args.safety_enable:
                t_opt_times_all.append(diffusion._safety_time_acc)

        ep_step_times.append(time.time() - t0)

        # Step environment (gym's `done` is ignored: the episode runs max_episode_length steps)
        observation, reward, terminal, _ = env.step(action)
        total_reward += reward

        if args.safety_enable:
            rootz = observation[0]
            if cbf_filter is not None:
                ep_h_min = min(ep_h_min, cbf_filter.barrier(rootz, observation[6]))
            # Safety violation (executed torso height above the ceiling) ends the episode
            if rootz > HEIGHT_LIMIT:
                ep_violated = True
                break

    score = 100 * env.get_normalized_score(total_reward)
    scores.append(score)
    step_times_all.extend(ep_step_times)

    line = (f"  ep {ep+1}/{args.n_episodes} | steps={t_idx+1} | R={total_reward:.2f} | "
            f"score={score:.2f} | {np.mean(ep_step_times)*1000:.1f}ms/step")
    if args.safety_enable:
        violated.append(int(ep_violated))
        line += f" | viol={int(ep_violated)}"
        if cbf_filter is not None:
            h_mins.append(ep_h_min)
            line += f" | h_min={ep_h_min:.4f}"
    print(line, flush=True)

# ══════════════════════════════════════════════════════════════
#  Summary: the Table 4 columns
# ══════════════════════════════════════════════════════════════

scores = np.array(scores)
print(f"\n{'='*60}")
print(f"  {args.method} | {args.dataset} | safety={'on' if args.safety_enable else 'off'}"
      + (f" ({args.safety_method})" if args.safety_enable and args.method == 'safediffuser' else ''))
print(f"{'='*60}")
print(f"  Episodes:      {args.n_episodes}")
print(f"  Score:         {scores.mean():.2f} +/- {scores.std():.2f}")
if args.safety_enable:
    print(f"  Viol (%):      {100 * np.mean(violated):.1f}  ({int(np.sum(violated))}/{len(violated)} episodes)")
if h_mins:
    print(f"  h_min:         {np.mean(h_mins):.4f} +/- {np.std(h_mins):.4f}")
if t_opt_times_all:
    print(f"  t_Opt (ms):    {np.mean(t_opt_times_all)*1000:.3f}")
print(f"  t_total (ms):  {np.mean(step_times_all)*1000:.2f}")
print(f"{'='*60}")
