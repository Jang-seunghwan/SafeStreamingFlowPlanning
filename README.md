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

This is the `gazebo` branch: warehouse navigation with a mecanum robot in Gazebo Fortress under ROS 2 Humble and Nav2 (Table 5, Figure 3).

## Installation

Tested on Ubuntu 22.04 with ROS 2 Humble, Gazebo Fortress (gz-sim 6.16.0), Python 3.10 (conda), PyTorch 2.7.1 with CUDA 11.8, numpy 1.26.4, scipy 1.15.2, qpth 0.0.18, ompl 2.0.1.

```bash
# 1. ROS 2 Humble (https://docs.ros.org/en/humble/Installation.html), then Gazebo Fortress / Nav2 / colcon:
bash install_deps.sh

# 2. Python 3.10 (required to import ROS 2 Humble's rclpy)
conda create -n ssf_gazebo python=3.10 -y && conda activate ssf_gazebo
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 3. Build the ROS 2 package (from the repository root)
source /opt/ros/humble/setup.bash
colcon build --base-paths src
source install/setup.bash
export PYTHONPATH=$PWD:$PWD/src/ssf_gazebo:$PYTHONPATH
```

All commands below are run from the repository root in this shell. The world
includes the Sun and Ground Plane models from Gazebo Fuel, which are downloaded on
the first launch. `bench/run_table5.sh` runs Gazebo headless and sets
`IGN_IP=127.0.0.1`.

Layout: `src/ssf_gazebo/` (ROS 2 package: world, map, robot, launch files, planners,
CBF, training scripts, data generator), `bench/` (benchmark runner, aggregation,
timing, Figure 3), `scripts/clean_stutter.py` (data cleaning).

## Data

The training data are 10,000 Nav2 rollouts in the warehouse. Each rollout teleports
the robot to a random start in the aisles, samples a goal at least 15 m away and
records `(t, x, y, v_x, v_y, heading, heading rate)` at 20 Hz (simulation time) until
Nav2 succeeds or the robot is within 0.1 m of the goal. Rollouts that time out
(150 s) or are aborted are discarded. Starts and goals are drawn with seed 42.

```bash
ros2 launch ssf_gazebo data_generation.launch.py headless:=true logs_dir:=logs_raw
python scripts/clean_stutter.py --input-dir logs_raw --output-dir logs
```

`clean_stutter.py` fills the rows where the AMCL position was not yet updated by
integrating the recorded velocity between position updates. The training scripts read `logs/`.

The world file `src/ssf_gazebo/worlds/warehouse.sdf` uses `real_time_factor` 1, which
the benchmark requires (see Evaluation). The generator runs on simulation time, so the
data can be collected faster with an uncapped simulator: set
`<real_time_factor>0</real_time_factor>` in that file, run `colcon build --base-paths src`
again (or build once with `--symlink-install`), and set it back to 1 before you run the benchmark.

The 100 benchmark (start, goal) pairs are in `bench/test_pairs.json`. To regenerate
the file (byte for byte), run `python -m bench.generate_test_pairs`. The script uses the
following rule. Draw uniformly in [1, 24] x [1, 19] m with seed 42. Keep pairs that are
5 to 15 m apart and pass the map check. Then replace every pair whose start, and
after that every pair whose goal, lies inside a CBF safety circle (filter margin
kappa = 0.05) with the next accepted candidate.

## Training

Pretrained checkpoints are not released. Each command trains one model with the
paper's settings (horizon H = 512, 256 denoising steps for Diffuser and FM). It writes
`models_h512/<name>.pt` (final EMA), `models_h512/<name>_best.pt` (best validation loss, used for
evaluation) and periodic snapshots. The first run builds the shared data cache
`cache/sfp_raw_all_min50.npz` from `logs/`.

| Model | Command | Table 5 rows using it |
|---|---|---|
| Diffuser | `python -m ssf_gazebo.train_diffuser_planner --planner-type diffusion` | Diffuser, Diffuser + CG, SafeDiffuser |
| FM | `python -m ssf_gazebo.train_diffuser_planner --planner-type flow_matching` | FM, SafeFM, FlowMatcher, SafeFlowMatcher |
| StreamingFlow | `python -m ssf_gazebo.train_sfp_planner` | StreamingFlow (open/closed-loop), SSF (open/closed-loop) |

The safe variants use the same weights and add the CBF at inference. FlowMatcher uses
the FM model with its prediction-correction sampler. RRT* and A* need no training.

## Evaluation

Pretrained checkpoints are not released. Evaluation takes the directory that holds
`diffuser_planner_best.pt`, `cfm_planner_best.pt` and `sfp_planner_best.pt`
(`models_h512` after training).

Each worker of `bench/run_table5.sh` starts its own headless Gazebo + map_server + AMCL
+ Nav2 stack. Workers use separate ROS domains and Gazebo partitions. The script runs every
(pair, planner) trial with a 90 s episode limit and writes one JSON per trial to
`results/bench_results/<planner>_<pair>.json`:

```bash
# all 13 rows on the 100 pairs, 8 parallel workers on 4 GPUs
GPUS="0 1 2 3" PYTHON=$(which python) bash bench/run_table5.sh models_h512 8 0 100
# planning-time columns (simulator-free, one idle GPU)
python -m bench.measure_planning_time --models-dir models_h512 --num-pairs 15
# the table
python -m bench.aggregate --results-dir results/bench_results \
    --timing-json results/planning_time.json --csv results/table5.csv
```

| Paper row (Table 5) | Planner name | Single-row command |
|---|---|---|
| RRT* | `rrt_star` | `PLANNERS=rrt_star bash bench/run_table5.sh models_h512` |
| A* | `a_star` | `PLANNERS=a_star bash bench/run_table5.sh models_h512` |
| Diffuser | `diffuser_pd` | `PLANNERS=diffuser_pd bash bench/run_table5.sh models_h512` |
| Diffuser + CG | `diffuser_cg_pd` | `PLANNERS=diffuser_cg_pd bash bench/run_table5.sh models_h512` |
| SafeDiffuser | `safe_diffuser_pd` | `PLANNERS=safe_diffuser_pd bash bench/run_table5.sh models_h512` |
| FM | `cfm_pd` | `PLANNERS=cfm_pd bash bench/run_table5.sh models_h512` |
| SafeFM | `safe_fm_pd` | `PLANNERS=safe_fm_pd bash bench/run_table5.sh models_h512` |
| FlowMatcher | `flow_matcher_pd` | `PLANNERS=flow_matcher_pd bash bench/run_table5.sh models_h512` |
| SafeFlowMatcher | `safe_flow_matcher_pd` | `PLANNERS=safe_flow_matcher_pd bash bench/run_table5.sh models_h512` |
| StreamingFlow (open-loop) | `sfp_offline_pd` | `PLANNERS=sfp_offline_pd bash bench/run_table5.sh models_h512` |
| SSF (open-loop) | `safe_sfp_offline_pd` | `PLANNERS=safe_sfp_offline_pd bash bench/run_table5.sh models_h512` |
| StreamingFlow (closed-loop) | `sfp_online` | `PLANNERS=sfp_online bash bench/run_table5.sh models_h512` |
| SSF (closed-loop) | `safe_sfp_online` | `PLANNERS=safe_sfp_online bash bench/run_table5.sh models_h512` |

`bench/run_table5.sh CHECKPOINT_DIR [NUM_WORKERS] [START_PAIR] [END_PAIR]` also reads
`OUT`, `PAIRS`, `GPUS`, `PYTHON`, `SEED`, `DOMAIN_BASE` and `TRIAL_TIMEOUT` from the
environment. With a stack already running, one trial is
`python -m bench.runner_one_pair --pair-idx 58 --planner safe_sfp_online --models-dir models_h512`.

Output fields and Table 5 columns (`bench/aggregate.py`):

| Table 5 column | Source |
|---|---|
| Succ. | `success` (goal within 0.25 m) of the trial JSONs. For the rows with active safety (Diffuser + CG, SafeDiffuser, SafeFM, SafeFlowMatcher, SSF open/closed-loop) a trial counts only if it also has no violation |
| h̄_min > 0 | `min_h`: the minimum over every recorded `/odom` sample of the barrier h (kappa = 0); `violation` = `min_h < 0`. The aggregate prints the violation rate (`Viol`) and the mean ± std of `min_h` |
| Sm. | `sm_acc`: mean magnitude of the `/cmd_vel` acceleration (m/s²), mean ± std over trials |
| t_Opt (s) | `t_qp_ms` in `planning_time.json`: safety-correction (QP / guidance / HOCBF-QCQP) time of one plan |
| t_total (s) | `t_total_ms` in `planning_time.json`: full planning time of one plan (online planners: per-step cost x H) |

Settings used by the code:
- The world runs at real-time factor 1. The learned planners' 20 Hz control loops are timed on the wall clock, so one tick is 0.05 s of simulated time, as in the training data.
- All learned planners share a PD controller on `/cmd_vel` (gains 2.0 / 0.2, speed clamp 0.5 m/s). After the plan or horizon ends, the PD pulls the robot toward the goal until the 90 s limit, or until it makes no progress for 10 s.
- A* and RRT* plan on the map augmented with the safety circles and are tracked by Nav2 FollowPath (Regulated Pure Pursuit).
- The safety filter uses two circles, (8.5, 10.0) and (16.0, 11.0) with r = 1.5 m, and filter margin kappa = 0.05. SSF uses HOCBF gains (kp, kv) = (0.5, 0.3). In closed loop, the filter also constrains the speed to the 0.5 m/s clamp.

Reproducibility: every trial seeds python `random`, numpy, torch and OMPL with
`--seed` (default 42) plus the pair index, so planner sampling is repeatable. A
Gazebo trial is not bit-exact across runs, because the control loop runs on the wall
clock against the simulator and AMCL. Planning times depend on the GPU and CPU load.
With OMPL 2.0.1, RRT* uses its full 5 s time budget whenever a solution exists, so its
planning time does not match the RRT* entry of Table 5.

Figure 3 (executed SSF closed-loop and SafeDiffuser trajectories for pair 58):

```bash
python -m bench.plot_traj_overlay --pair 58 --results-dir results/bench_results
# -> results/warehouse_traj_overlay.{pdf,png}
```

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
- **Diffuser** (https://github.com/jannerm/diffuser): the TemporalUnet backbone and diffusion sampler; its MIT notice is kept in LICENSE.
- **streaming-flow-policy** (https://github.com/siddancha/streaming-flow-policy): the streaming flow formulation.
- **SafeDiffuser** (Xiao et al.) and **SafeFlowMatcher** (Yang et al.): the safe baselines, re-implemented for this environment.
- **ROS 2**, **Gazebo** / **ros_gz**, **Nav2** (AMCL, map server, Regulated Pure Pursuit), **OMPL** (RRT*) and **qpth**: simulator, navigation stack and libraries.
