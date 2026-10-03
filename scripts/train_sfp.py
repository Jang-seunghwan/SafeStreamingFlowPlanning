import os
import copy
import json

import numpy as np
import torch
import diffuser.utils as utils
from torch.utils.data import DataLoader
from diffuser.utils.training import EMA
from tqdm.auto import tqdm

torch.backends.cudnn.benchmark = True

from diffuser.models.cond_unet1D import ConditionalUnet1D
from diffuser.models.sfpd import StreamingFlowPolicyDeterministic


def make_collate_fn(
    sigma: float,
    k: float,
    vel_weight=None,
):
    sigma = float(sigma)
    k = float(k)
    w = np.array(vel_weight, dtype=np.float32) if vel_weight is not None and len(vel_weight) > 0 else None

    def collate(samples):
        trajectories = np.stack(
            [np.asarray(sample.trajectories, dtype=np.float32) for sample in samples],
            axis=0,
        )
        if trajectories.shape[1] < 2:
            raise ValueError('Trajectories must contain at least two states.')

        batch_size, horizon, state_dim = trajectories.shape
        pos_dim = state_dim // 2
        cond = np.stack([trajectories[:, 0], trajectories[:, -1]], axis=1)

        seg_idx = np.random.randint(0, horizon-1, size=batch_size, dtype=np.int64)
        t = seg_idx / (horizon-1)
        t = t.astype(np.float32)

        batch_idx = np.arange(batch_size)
        x = trajectories[batch_idx, seg_idx]
        x_next = trajectories[batch_idx, seg_idx + 1]

        dt = 1.0 / (horizon - 1)
        x_dot = (x_next - x) / dt

        # Dynamics-consistent Gaussian tube perturbation
        # 1) Sample position noise: eps_pos ~ N(0, σ_t² I)
        # 2) Derive velocity noise from dynamics coupling: eps_vel = w · eps_pos
        # Tube contraction via σ_t = σ₀·e^{-kt}
        exp_term = np.exp(-k * t).astype(np.float32)
        eps_pos = sigma * exp_term[:, None] * np.random.randn(batch_size, pos_dim).astype(np.float32)
        if w is not None:
            # The velocity of the perturbed input state is not perturbed (the factor 0 makes this term zero);
            # the target velocity carries the position noise mapped through the dynamics (semi-implicit Euler).
            eps_vel_curr = ((1-k*dt)*eps_pos * w[None, :])*(0)*dt
            eps_vel_next = eps_pos * w[None, :]
        else:
            # No dynamics coupling: zero velocity perturbation
            eps_vel_curr = np.zeros((batch_size, pos_dim), dtype=np.float32)
            eps_vel_next = np.zeros((batch_size, pos_dim), dtype=np.float32)
        sampled_error = np.concatenate([eps_pos, eps_vel_curr], axis=-1)
        x_train = x + sampled_error
        sampled_error = np.concatenate([eps_pos, eps_vel_next], axis=-1)
        v_train = x_dot - k * sampled_error

        x_label = x_train + v_train * dt
        return {
            'cond': torch.from_numpy(cond),
            'x': torch.from_numpy(x_train[:, None, :]),
            'x_label': torch.from_numpy(x_label[:, None, :]),
            'v': torch.from_numpy(v_train[:, None, :]),
            't': torch.from_numpy(t),
        }

    return collate


def train_sfp(args):
    # Reproducibility
    seed = getattr(args, 'seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # D4RL dataset (maze2d)
    dataset_kwargs = dict(
        env=args.dataset,
        horizon=args.horizon,
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
    )

    dataset_config = utils.Config(args.loader, **dataset_kwargs)

    dataset = dataset_config()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # Infer state dim directly from the sample trajectories returned by SequenceDataset.
    sample = dataset[0]
    state_dim = sample.trajectories.shape[-1] # e.g., 4 for maze2d (x, y, vx, vy)
    num_cond = 2   # conditioning on start and goal states
    condition_dim = state_dim * num_cond
    args.safety_enabled = False  # Disable safety during training

    # Velocity scaling of the maze dynamics, from the normalizer: w = pos_range / (dt_sim * vel_range)
    pos_dim = state_dim // 2
    obs_norm = dataset.normalizer.normalizers['observations']
    obs_range = (obs_norm.maxs - obs_norm.mins)[:state_dim]
    dt_sim = dataset.env.dt
    vel_weight = (obs_range[:pos_dim] / (dt_sim * obs_range[pos_dim:])).tolist()
    print(f"[ train_sfp ] vel_weight = {vel_weight} (dt_sim={dt_sim})")

    velocity_net = ConditionalUnet1D(
        input_dim=state_dim,
        global_cond_dim=condition_dim,
        fc_timesteps=1,
        horizon=args.horizon,
        vel_weight=vel_weight,
    ).to(device)

    policy = StreamingFlowPolicyDeterministic(
        velocity_net=velocity_net,
        state_dim=state_dim,
        device=device,
        normalizer=dataset.normalizer,
        args=args,
    ).to(device)

    collate_fn = make_collate_fn(
        sigma=args.sigma_train,
        k=args.k,
        vel_weight=vel_weight,
    )

    dataloader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )
    if args.num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = 2
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    ema = EMA(args.ema_decay)
    ema_velocity = copy.deepcopy(policy.velocity_net).to(device)

    # vel_net gets higher lr (needs to learn larger weight magnitudes)
    vel_net_params = list(policy.velocity_net.vel_net.parameters())
    vel_net_ids = {id(p) for p in vel_net_params}
    other_params = [p for p in policy.velocity_net.parameters() if id(p) not in vel_net_ids]
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': args.learning_rate * 5},      # pos_net: 5e-4
        {'params': vel_net_params, 'lr': args.learning_rate * 3},    # vel_net: 3e-4
    ], weight_decay=args.weight_decay)

    def save_checkpoint(path, epoch_label=None):
        checkpoint = {
            'epoch': epoch_label,
            'policy_state_dict': policy.state_dict(),
            'velocity_state_dict': policy.velocity_net.state_dict(),
            'config': {
                'dataset': args.dataset,
                'horizon': args.horizon,
                'sigma_train': args.sigma_train,
                'k': args.k,
                'state_dim': state_dim,
                'condition_dim': condition_dim,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'weight_decay': args.weight_decay,
                'ema_decay': args.ema_decay,
                'n_epochs': args.n_epochs,
                'num_workers': args.num_workers,
                'device': args.device,
                'vel_weight': vel_weight,
            },
            'normalizer': dataset.normalizer,
        }
        torch.save(checkpoint, path)
        print(f"[ train_sfp ] Saved checkpoint to {path}")

    os.makedirs(args.savepath, exist_ok=True)

    # Training history for loss curve plotting
    training_history = {
        'config': {
            'dataset': args.dataset,
            'horizon': args.horizon,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'n_epochs': args.n_epochs,
            'sigma_train': args.sigma_train,
            'k': args.k,
        },
        'epoch_losses': [],
        'epoch_p_losses': [],
        'epoch_vel_losses': [],
        'epoch_vel_base_losses': [],
    }
    history_path = os.path.join(args.savepath, 'training_history.json')

    def save_history():
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)

    policy.train()
    global_step = 0
    with tqdm(range(args.n_epochs), desc='Epoch') as epoch_bar:
        for epoch_idx in epoch_bar:
            epoch_losses = []
            epoch_p_losses = []
            epoch_vel_losses = []
            epoch_vel_base_losses = []
            with tqdm(dataloader, desc='Batch', leave=False) as batch_bar:
                for batch in batch_bar:
                    loss, p_loss, vel_loss, vel_loss_base = policy.Loss(batch)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update_model_average(ema_velocity, policy.velocity_net)

                    global_step += 1

                    # Sync to CPU only at log intervals to avoid GPU stalls
                    if global_step % 10 == 0:
                        loss_val = loss.item()
                        p_loss_val = p_loss.item()
                        vel_loss_val = vel_loss.item()
                        vel_loss_base_val = vel_loss_base.item()
                        epoch_losses.append(loss_val)
                        epoch_p_losses.append(p_loss_val)
                        epoch_vel_losses.append(vel_loss_val)
                        epoch_vel_base_losses.append(vel_loss_base_val)

                        batch_bar.set_postfix({
                            "loss": f"{loss_val:.4f}",
                            "p_loss": f"{p_loss_val:.4f}",
                            "vel_loss": f"{vel_loss_val:.4f}",
                            "vel_base": f"{vel_loss_base_val:.4f}",
                        })


            if epoch_losses:
                mean_loss = float(np.mean(epoch_losses))
                mean_p_loss = float(np.mean(epoch_p_losses))
                mean_vel_loss = float(np.mean(epoch_vel_losses))
                mean_vel_base_loss = float(np.mean(epoch_vel_base_losses))

                # Record epoch-level losses
                training_history['epoch_losses'].append(mean_loss)
                training_history['epoch_p_losses'].append(mean_p_loss)
                training_history['epoch_vel_losses'].append(mean_vel_loss)
                training_history['epoch_vel_base_losses'].append(mean_vel_base_loss)

                epoch_bar.set_postfix({
                    "loss": mean_loss,
                    "p_loss": mean_p_loss,
                    "vel_loss": mean_vel_loss,
                    "vel_base": mean_vel_base_loss,
                })

            # No intermediate checkpoints — save only at the end

    policy.velocity_net.load_state_dict(ema_velocity.state_dict())

    # Save final training history
    save_history()
    print(f"[ train_sfp ] Saved final training history to {history_path}")

    model_path = os.path.join(args.savepath, args.model_filename)
    save_checkpoint(model_path, epoch_label='final')
    print(f"[ train_sfp ] Saved SFP velocity model to {model_path}")


if __name__ == '__main__':
    class Parser(utils.Parser):
        dataset: str = 'maze2d-umaze-v1'
        config: str = 'config.maze2d'
        method: str = 'sfp'

    args = Parser().parse_args('train')
    args.safety_enabled = False
    train_sfp(args)
