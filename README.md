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

This is the `F1TENTH-Data` branch: F1TENTH racing on the Budapest and Catalunya tracks (Table 3).

## Installation

The paper runs used two conda environments at the same time. The code ran in
`ssf` (PyTorch and the planners), and the site-packages directory of `f1tenth`
was placed **first** on `PYTHONPATH`. As a result, numpy 2.0.2, scipy, gymnasium 1.1.1,
numba, PyYAML and pillow came from `f1tenth`. The environment files list the versions
that were used. They were not re-created from scratch for this release.

```bash
conda env create -f envs/ssf.yml
conda env create -f envs/f1tenth.yml
conda activate ssf
export PYTHONPATH=$(conda run -n f1tenth python -c "import site; print(site.getsitepackages()[0])"):$(pwd)

# Track data (centerline / raceline / maps), not included in this repository
git clone https://github.com/f1tenth/f1tenth_racetracks.git   # commit b95c4ef was used
# or: export F1TENTH_RACETRACKS=/path/to/f1tenth_racetracks
```

Versions used:

| Component | Version |
|---|---|
| Python | 3.9.23 |
| PyTorch | 2.7.1, CUDA 12.6 (RTX 4090) |
| numpy | 2.0.2 (needed to load checkpoints, which are pickled with numpy 2) |
| gymnasium | 1.1.1 |
| f1tenth_gym | v1.0.0 (commit 5a301bd) |
| qpth | 0.0.18 |
| torchcfm | 1.0.7 |
| torchdyn | 1.0.6 |

`f1tenth_gym` downloads the Budapest and Catalunya maps on first use. Run all commands from the repository root.

## Data

1. **Raw laps.** Run `scripts/collect_trajectories.py` once per track:
   ```bash
   python scripts/collect_trajectories.py --tracks Budapest --n_episodes 10000 --log_dir logs/Budapest
   ```
   It tracks a random blend of the minimum-curvature raceline and the centerline with a Pure Pursuit controller, starting at the start/finish line. It records `[x, y, vx, vy]` at 100 Hz.
   - The paper used 10,000 laps per track.
   - This step needs `f110_gym` from f1tenth_gym v0.2.x, with its `gym` imports switched to `gymnasium`. This package is not pip-installable; the `f1tenth` environment above held such a local copy.
   - The exact number of attempted episodes was not recorded. A new collection therefore gives a comparable dataset, not an identical one.
2. **Preprocessing.**
   ```bash
   python scripts/regenerate_processed_data.py
   ```
   It subsamples to 10 Hz, removes length outliers, and resamples each lap to the track horizon (H = 696 Budapest, 724 Catalunya). It writes two versions that differ only in the units of the velocity channels:
   - `processed_data/m_per_s/<track>.npz`: velocity in m/s.
   - `processed_data/m_per_step/<track>.npz`: velocity in m per 0.1 s step.

   The paper models were trained on different versions:
   - m per step: Diffuser, and the CFM model of FM / FlowMatcher (and of SafeFM / SafeFlowMatcher on Budapest).
   - m/s: SFP, and the CFM model of SafeFM / SafeFlowMatcher on Catalunya.

   From the raw laps used for the paper, this step reproduces the paper training data exactly.

## Training

The script defaults are the settings of the paper models. `bash scripts/train.sh` runs preprocessing and all of the commands below.

| Model | Command (per track `T` in `budapest`, `catalunya`) | Table 3 rows | Checkpoint |
|---|---|---|---|
| Diffuser (K = 512) | `python scripts/train_diffuser.py --track T` | Diffuser, Diffuser + CG, SafeDiffuser | `checkpoints/diffuser/T/` |
| CFM (time scale 256) | `python scripts/train_cfm.py --track T` | FM, FlowMatcher; SafeFM, SafeFlowMatcher on Budapest | `checkpoints/cfm/T/` |
| CFM, m/s data | `python scripts/train_cfm.py --track catalunya --velocity_units m_per_s` | SafeFM, SafeFlowMatcher on Catalunya | `checkpoints/cfm_m_per_s/catalunya/` |
| Streaming flow policy | `python scripts/train_sfp.py --track T` | StreamingFlow, SSF | `checkpoints/sfp/T/` |

The safety filters are applied only at evaluation. Each row and its safe variant use the same weights.

## Evaluation

Pretrained checkpoints are not released. Train them as above. Evaluation takes the checkpoint root (the layout above) as an argument.

Each call evaluates one row on one track. The defaults:
- 100 episodes, `--seed 42`;
- start and goal at the start/finish line with N(0, 0.02²) m noise (`--sigma 0.02`);
- K = 512 flow-integration steps for the FM family (`--k_cfm 512`).

`bash scripts/eval.sh [checkpoint_root]` runs all 18 cells.

| Paper row (Table 3) | Command |
|---|---|
| Diffuser | `python scripts/eval_all.py --method diffuser --track T --checkpoint_root checkpoints` |
| Diffuser + CG | `python scripts/eval_all.py --method diffuser_cg --track T --checkpoint_root checkpoints` |
| SafeDiffuser | `python scripts/eval_all.py --method safediffuser --track T --checkpoint_root checkpoints` |
| FM | `python scripts/eval_all.py --method fm --track T --checkpoint_root checkpoints` |
| SafeFM | `python scripts/eval_all.py --method safefm --track T --checkpoint_root checkpoints` |
| FlowMatcher | `python scripts/eval_all.py --method flowmatcher --track T --checkpoint_root checkpoints` |
| SafeFlowMatcher | `python scripts/eval_all.py --method safeflowmatcher --track T --checkpoint_root checkpoints` |
| StreamingFlow | `python scripts/eval_all.py --method sfp_off --track T --checkpoint_root checkpoints` |
| SSF (ours) | `python scripts/eval_all.py --method safe_sfp_off --track T --checkpoint_root checkpoints` |

Results are written to `results/<method>/<track>.json`: a `summary` and one `per_episode` record per episode (the start/goal pair and the per-episode metrics). Summary fields:

| Summary field | Meaning |
|---|---|
| `goal_rate` | Goal: the final executed position is within 1 m of the goal, with no collision |
| `viol_k0_rate`, `viol_k001_rate` | Viol: the executed positions enter an obstacle, with barrier margin κ = 0 or κ = 0.01 |
| `succ_k0_rate`, `succ_k001_rate` | Succ: Goal and no Viol, at κ = 0 or κ = 0.01 |
| `plan_safe` | h̄_min > 0 mark: every plan waypoint lies outside every obstacle (κ = 0) in all episodes |
| `sm_mean` ± `sm_std` | Sm: (1/H) Σ‖a_{t+1} − a_t‖ with a_t = (p_{t+2} − 2p_{t+1} + p_t)/Δt² on the plan positions |
| `t_opt_s_mean` ± `t_opt_s_std` | t_Opt: time spent in the safety layer (QP solves, guidance) per plan |
| `t_total_s_mean` ± `t_total_s_std` | t_total: wall-clock planning time per plan |

The safe rows track the plan with an obstacle-aware PD reference; the other rows use plain linear interpolation of the plan.

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

MIT, see [LICENSE](LICENSE). This code builds on several projects:
- **Diffuser** (https://github.com/jannerm/diffuser) and **SafeDiffuser** (https://github.com/Weixy21/SafeDiffuser): the Diffuser and SafeDiffuser code; their MIT notices are kept in LICENSE.
- **streaming-flow-policy** (https://github.com/siddancha/streaming-flow-policy): the streaming flow formulation.
- **SafeFlowMatcher** (Yang et al.): the FM and FlowMatcher baselines.
- **conditional-flow-matching / torchcfm** (https://github.com/atong01/conditional-flow-matching), **torchdyn**, **qpth**: libraries.
- **f1tenth_gym** (https://github.com/f1tenth/f1tenth_gym) and **f1tenth_racetracks** (https://github.com/f1tenth/f1tenth_racetracks): the simulator and track data.
