# Safe Streaming Flow Planning by Aligning Sampling Dynamics with Execution Dynamics

Seunghwan Jang, Jeongyong Yang, Siddharth Ancha, SooJean Han

**CoRL 2026** · [Project page](https://jang-seunghwan.github.io/SafeStreamingFlowPlanning/) · [Paper](https://arxiv.org/abs/XXXX.XXXXX) · [OpenReview](https://openreview.net/forum?id=gvM7KWEhVI)

With control barrier functions (CBFs), safe generative planners enforce safety constraints at inference time, including constraints unseen during training. But they enforce them on the plan, and the robot's execution can still violate them. SafeStreamingFlow samples in execution time, so the safety filter acts on the step the robot executes.

## Branches

| Branch | Environment | Paper |
|---|---|---|
| [`main`](https://github.com/Jang-seunghwan/SafeStreamingFlowPlanning/tree/main) | Maze2D (D4RL) | Table 2 |
| [`F1TENTH-Data`](https://github.com/Jang-seunghwan/SafeStreamingFlowPlanning/tree/F1TENTH-Data) | F1TENTH autonomous racing | Table 3 |
| [`SafeStreamingLoco`](https://github.com/Jang-seunghwan/SafeStreamingFlowPlanning/tree/SafeStreamingLoco) | MuJoCo Hopper | Table 4 |
| [`gazebo`](https://github.com/Jang-seunghwan/SafeStreamingFlowPlanning/tree/gazebo) | Warehouse navigation in Gazebo (ROS 2) | Table 5, Figure 3 |

This is the `main` branch: the Maze2D experiments (Table 2). The project page source is in `docs/`.

## Installation

Tested with Python 3.9, PyTorch 2.7.1 (CUDA 12.6), gym 0.23.1, mujoco-py 2.1.2.14 with MuJoCo 2.1.0, and D4RL
(commit `f2a05c0`), on Linux with an NVIDIA GPU.

1. Install the MuJoCo 2.1.0 binaries in `~/.mujoco/mujoco210` (required by mujoco-py; see the
   [mujoco-py instructions](https://github.com/openai/mujoco-py#install-mujoco)).
2. Create the environment and install this repository:

```bash
conda env create -f environment.yml
conda activate ssf
pip install -e .
```

All commands below are run from the repository root.

## Data

Training uses the D4RL Maze2D datasets `maze2d-umaze-v1`, `maze2d-medium-v1` and `maze2d-large-v1`.
D4RL downloads them automatically on first use (to `~/.d4rl/datasets`).

## Training

Pretrained checkpoints are not released. Three models are trained per map (`<map>` is `maze2d-umaze-v1`,
`maze2d-medium-v1` or `maze2d-large-v1`); the defaults in `config/maze2d.py` are the settings used for the paper.

| Model | Table 2 rows that use it | Command | Output (under `logs/<map>/`) |
|---|---|---|---|
| Streaming flow policy | StreamingFlow, SSF (open- and closed-loop) | `python scripts/train_sfp.py --config config.maze2d --dataset <map> --method sfp` | `sfp/k0.1_s0.005/sfp_velocity_policy.pt` |
| Diffuser | Diffuser, Diffuser + CG, SafeDiffuser | `python scripts/train.py --config config.maze2d --dataset <map> --method base` | `diffusion/H{H}_T{T}/` |
| Flow matching | FM, SafeFM, FlowMatcher, SafeFlowMatcher | `python scripts/train.py --config config.maze2d --dataset <map> --method cfm` | `cfm/H{H}_T{T}/` |

`H{H}_T{T}` is `H128_T64` (umaze), `H256_T256` (medium) and `H384_T256` (large). A* and RRT* need no training.
Add `--logbase <dir>` to write the checkpoints somewhere other than `logs/`.

## Evaluation

Each command evaluates one row on one map: 100 start/goal pairs, with everything that is sampled (Python `random`,
NumPy, PyTorch and the environment) seeded from `--seed` (default 42), so repeated runs use the same episodes.
The checkpoint directory is given with `--logbase` (the directory the training wrote to; default `logs`).
`scripts/run_baselines.sh` runs all rows on all three maps.

Common prefix of the learned rows:
`--config config.maze2d --dataset <map> --logbase <checkpoint dir>`; `SAFE_HP="--robust_term 0.1 --relax_threshold 0.95"`.

| Table 2 row | Command |
|---|---|
| RRT* | `python scripts/plan_maze2d_classic.py --dataset <map> --planner_type rrt_star` |
| A* | `python scripts/plan_maze2d_classic.py --dataset <map> --planner_type astar` |
| Diffuser | `python scripts/plan_maze2d.py <prefix> --method base --safety_enabled false` |
| Diffuser + CG | `python scripts/plan_maze2d.py <prefix> --method base --safety_enabled true --safety_method gd` |
| SafeDiffuser | `python scripts/plan_maze2d.py <prefix> --method base --safety_enabled true --safety_method invariance $SAFE_HP` |
| FM | `python scripts/plan_maze2d.py <prefix> --method cfm --safety_enabled false --integrator pure` |
| SafeFM | `python scripts/plan_maze2d.py <prefix> --method cfm --safety_enabled true --integrator pure $SAFE_HP` |
| FlowMatcher | `python scripts/plan_maze2d.py <prefix> --method cfm --safety_enabled false --integrator pc --one_shot_enabled true` |
| SafeFlowMatcher | `python scripts/plan_maze2d.py <prefix> --method cfm --safety_enabled true --integrator pc --one_shot_enabled true $SAFE_HP` |
| StreamingFlow (open-loop) | `python scripts/plan_maze2d_sfp.py <prefix> --method sfp --safety_enabled false --closed_loop false` |
| StreamingFlow (closed-loop) | `python scripts/plan_maze2d_sfp.py <prefix> --method sfp --safety_enabled false --closed_loop true` |
| SSF (open-loop) | `python scripts/plan_maze2d_sfp.py <prefix> --method sfp --safety_enabled true --closed_loop false` |
| SSF (closed-loop) | `python scripts/plan_maze2d_sfp.py <prefix> --method sfp --safety_enabled true --closed_loop true` |

A* and RRT* run on the CPU (`CUDA_VISIBLE_DEVICES=""` is fine). Use `--suffix <name>` to name the output directory
and `--n_episodes` to change the number of episodes.

### Outputs

Each run writes to `<logbase>/<map>/plans/<exp_name>/<suffix>/`:

- `summary.json`: the Table 2 columns of the run;
- `episodes.csv`: one row per episode (`goal`, `min_h0`, `viol0`, `safe_succ0`, `s_smooth`, `trap`);
- per-episode trajectory images: the plan (`results/`) for the Diffuser / FM rows; the executed trajectory
  (`control_results/`) and, open-loop, the plan (`plan_results/`) for the streaming rows.

| Table 2 column | `summary.json` field | Definition (`diffuser/utils/metrics.py`) |
|---|---|---|
| Succ. | `safe_succ0_pct` | goal reached (reward > 0.95 at the last step) and no violation |
| Viol. | `viol0_pct` | some executed state is inside an obstacle: min over states and obstacles of h0 < 0, with h0 = ((s0 - o0)/r)^2 + ((s1 - o1)/r)^2 - 1 |
| Sm. | `sm_mean`, `sm_std` | acceleration smoothness, mean of ‖p<sub>k+1</sub> - 2p<sub>k</sub> + p<sub>k-1</sub>‖ / dt² (executed trajectory; for the Diffuser and FM rows, the planned trajectory in normalized coordinates) |
| Trap | `trap_pct` | the plan's denoising path contains a jump (Diffuser and FM rows; 0 by construction for the streaming rows, not reported for A* / RRT*) |

`goal_pct` (goal reached) and `timing_ms` are also reported. Table 2 lists Trap only for the rows with a safety
layer.

## Citation

```bibtex
@inproceedings{jang2026safe,
  title={Safe Streaming Flow Planning by Aligning Sampling Dynamics with Execution Dynamics},
  author={Jang, Seunghwan and Yang, Jeongyong and Ancha, Siddharth and Han, SooJean},
  booktitle={10th Annual Conference on Robot Learning},
  year={2026},
  url={https://openreview.net/forum?id=gvM7KWEhVI}
}
```

## License and acknowledgements

MIT License (see `LICENSE`). This code builds on [Diffuser](https://github.com/jannerm/diffuser) and
[SafeDiffuser](https://github.com/Weixy21/SafeDiffuser) (both MIT), and the streaming flow policy follows
[Streaming Flow Policy](https://github.com/siddancha/streaming-flow-policy). The environments and datasets are from
[D4RL](https://github.com/Farama-Foundation/D4RL); flow matching uses
[TorchCFM](https://github.com/atong01/conditional-flow-matching) and [torchdyn](https://github.com/DiffEqML/torchdyn).
