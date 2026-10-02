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

This is the `SafeStreamingLoco` branch: the MuJoCo Hopper experiments of Table 4 (D4RL `hopper-medium-expert-v2`
with a torso-height ceiling of 1.5).

## Installation

Tested with Python 3.9, PyTorch 2.8 (CUDA 12.8) on Linux with an NVIDIA RTX 4090.

1. MuJoCo 2.1.0 for `mujoco-py`: extract the [MuJoCo 2.1.0 binaries](https://github.com/google-deepmind/mujoco/releases/tag/2.1.0)
   to `~/.mujoco/mujoco210` and add `~/.mujoco/mujoco210/bin` to `LD_LIBRARY_PATH`.
2. Create the environment and install this package:
   ```bash
   conda env create -f environment.yml   # gym 0.24.1, mujoco-py 2.1.2.14, D4RL @ f2a05c0, torchcfm, qpth, ...
   conda activate ssf_loco
   pip install -e .
   ```

## Data

The offline dataset is D4RL `hopper-medium-expert-v2`. D4RL downloads it automatically on first use
(to `~/.d4rl/datasets`). No other data is needed.

## Training

Pretrained checkpoints are not released; train the models with the commands below. The defaults in
`config/locomotion.py` are the settings used for the paper. Checkpoints are written to
`logs/hopper-medium-expert-v2/<name>/`.

| Model | Command | Output | Used by (Table 4 rows) |
|---|---|---|---|
| Streaming flow velocity network | `python scripts/train_sfp.py` | `sfp/k2.0_s0.0001/` | StreamingFlow, SSF |
| CFM trajectory model | `python scripts/train.py --method cfm` | `cfm/defaults_H600_T20/` | FM, SafeFM, FlowMatcher, SafeFlowMatcher |
| Diffusion trajectory model | `python scripts/train.py --method base` | `diffusion/defaults_H600_T20/` | Diffuser, Diffuser+CG, SafeDiffuser |
| Value function (reward guidance) | `python scripts/train_values.py` | `values/defaults_H600_T20_d0.99/` | all Diffuser and FM rows |

The diffusion model of the paper was trained with the original SafeDiffuser codebase using these same settings
(`base['diffusion']` in the config: batch size 128, 2.5e5 steps).

## Evaluation

All commands evaluate 100 episodes of 1000 steps. `--loadbase` is the directory that contains the
`hopper-medium-expert-v2/` checkpoint folders from training (the checkpoint sub-folders can be changed with
`--sfp_ckpt`, `--diffusion_loadpath` and `--value_loadpath`). Episodes are seeded with `--seed` (default 42).

| Paper row | Command |
|---|---|
| Diffuser | `python scripts/eval_sfp.py --method safediffuser --n_episodes 100 --loadbase logs` |
| Diffuser+CG | `python scripts/eval_sfp.py --method safediffuser --safety_enable --safety_method cg --n_episodes 100 --loadbase logs` |
| SafeDiffuser | `python scripts/eval_sfp.py --method safediffuser --safety_enable --safety_method invariance --n_episodes 100 --loadbase logs` |
| FM | `python scripts/plan_loco.py --integrator pure --safety_enabled False --n_eval_episodes 100 --loadbase logs` |
| SafeFM | `python scripts/plan_loco.py --integrator pure --safety_enabled True --n_eval_episodes 100 --loadbase logs` |
| FlowMatcher | `python scripts/plan_loco.py --integrator pc --safety_enabled False --n_eval_episodes 100 --loadbase logs` |
| SafeFlowMatcher | `python scripts/plan_loco.py --integrator pc --safety_enabled True --n_eval_episodes 100 --loadbase logs` |
| StreamingFlow | `python scripts/eval_sfp.py --method sfp --n_episodes 100 --loadbase logs` |
| SSF | `python scripts/eval_sfp.py --method sfp --safety_enable --n_episodes 100 --loadbase logs` |

Method notes: the Diffuser and FM rows plan a full trajectory (horizon 600, K = 20 denoising / integration steps)
at every env step and execute its first action. FM integrates the flow with K Euler steps; FlowMatcher makes a
one-shot prediction of the trajectory and then corrects it with K steps. The safe planners filter every denoising /
integration step (Diffuser+CG: truncation; SafeDiffuser, SafeFM, SafeFlowMatcher: CBF). StreamingFlow and SSF make
one network forward pass per env step; SSF filters the executed action with a discrete-time HOCBF-QP.
gym's `done` does not end an episode; with a safety filter, an episode ends at the first ceiling violation.

Output: each script prints one line per episode and, at the end, a summary block on stdout. Summary field → Table 4
column:

| Summary field | Table 4 column |
|---|---|
| `Score` | Score (D4RL normalized score ×100, mean ± std over episodes) |
| `h_min` | h̄_min (SSF only: per-episode minimum of the barrier h = 1.5 − z − 0.08 ż − 0.01, mean ± std) |
| `Viol (%)` | Viol (%) (episodes in which the executed torso height exceeded 1.5; safe rows only) |
| `t_Opt (ms)` | t_Opt (safety-filter time per env step; safe rows only) |
| `t_total (ms)` | t_total (planning / policy time per env step) |

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

MIT License (see [LICENSE](LICENSE)). This branch builds on
[Diffuser](https://github.com/jannerm/diffuser) and [SafeDiffuser](https://github.com/Weixy21/SafeDiffuser)
(trajectory models, guided sampling and the Diffuser-family safety filters),
[streaming-flow-policy](https://github.com/siddancha/streaming-flow-policy),
[D4RL](https://github.com/Farama-Foundation/D4RL), [torchcfm](https://github.com/atong01/conditional-flow-matching)
and [qpth](https://github.com/locuslab/qpth).
