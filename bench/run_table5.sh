#!/usr/bin/env bash
# Warehouse navigation benchmark (paper Table 5): all 13 rows on the 100 pairs
# of bench/test_pairs.json.
#
# Each worker brings up its own Gazebo (worlds/warehouse.sdf, headless) +
# map_server + AMCL + Nav2 stack in an isolated ROS_DOMAIN_ID / IGN_PARTITION
# and runs bench.run_bench on a contiguous slice of the pairs. Every worker
# runs in its own process group and is cleaned up by process group only.
#
# Usage (from anywhere; paths are resolved relative to the repository):
#   bash bench/run_table5.sh CHECKPOINT_DIR [NUM_WORKERS=8] [START_PAIR=0] [END_PAIR=100]
# CHECKPOINT_DIR holds diffuser_planner_best.pt, cfm_planner_best.pt and
# sfp_planner_best.pt (written by the training commands to models_h512/).
# Environment overrides:
#   PYTHON        python interpreter with the requirements (default: python3)
#   OUT           per-trial JSON directory   (default: results/bench_results)
#   PAIRS         pair list                  (default: bench/test_pairs.json)
#   PLANNERS      space-separated planner names (default: the 13 Table 5 rows)
#   GPUS          space-separated GPU ids; worker i uses GPUS[i % n] (default: "0")
#   DOMAIN_BASE   worker i uses ROS_DOMAIN_ID = DOMAIN_BASE + i (default: 80)
#   TRIAL_TIMEOUT hard wall-clock limit per trial subprocess, s (default: 150)
#   SEED          base seed; each trial is seeded with SEED + pair index (default: 42)
# Prerequisite: `colcon build` of src/ssf_gazebo in the repository root
# (install/setup.bash).
# (no `set -u`: ROS setup.bash uses unbound variables)

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PYTHON:-python3}
OUT=${OUT:-$REPO/results/bench_results}
PAIRS=${PAIRS:-$REPO/bench/test_pairs.json}
PLANNERS=${PLANNERS:-"rrt_star a_star diffuser_pd diffuser_cg_pd safe_diffuser_pd cfm_pd safe_fm_pd flow_matcher_pd safe_flow_matcher_pd sfp_offline_pd safe_sfp_offline_pd sfp_online safe_sfp_online"}
GPUS=${GPUS:-0}
DOMAIN_BASE=${DOMAIN_BASE:-80}
TRIAL_TIMEOUT=${TRIAL_TIMEOUT:-150}
SEED=${SEED:-42}
if [ -z "$1" ]; then echo "usage: bash bench/run_table5.sh CHECKPOINT_DIR [NUM_WORKERS] [START_PAIR] [END_PAIR]"; exit 1; fi
MODELS="$(cd "$1" && pwd)" || exit 1
NUM_WORKERS=${2:-8}
P0=${3:-0}
P1=${4:-100}
MAP_YAML="$REPO/src/ssf_gazebo/maps/ssf_map_warehouse.yaml"
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"
MASTER="$LOGS/master.log"

worker_main() {
    local i=$1 s=$2 e=$3
    local logdir="$LOGS/w$i"; mkdir -p "$logdir"
    local gpus=($GPUS)
    export ROS_DOMAIN_ID=$(( DOMAIN_BASE + i ))
    export IGN_PARTITION="ssf_bench_$i" GZ_PARTITION="ssf_bench_$i"
    export CUDA_VISIBLE_DEVICES=${gpus[$(( i % ${#gpus[@]} ))]}
    export IGN_IP=127.0.0.1
    export QT_QPA_PLATFORM=offscreen
    unset DISPLAY
    # Headless NVIDIA EGL for the ogre2 lidar sensor, when available.
    if [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
        export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
        export __GLX_VENDOR_LIBRARY_NAME=nvidia
    fi
    source /opt/ros/humble/setup.bash 2>/dev/null
    source "$REPO/install/setup.bash" 2>/dev/null
    export PYTHONPATH="$REPO:$REPO/src/ssf_gazebo:${PYTHONPATH:-}"
    cd "$REPO"
    local mark="[w$i dom$ROS_DOMAIN_ID pairs $s-$e]"
    echo "$mark START $(date -u +%FT%TZ)" >> "$MASTER"

    ros2 launch ssf_gazebo gz_sim.launch.py world:=warehouse robot:=mecanum name:=robot \
        x:=1.0 y:=1.0 z:=0.2 headless:=true > "$logdir/gz.log" 2>&1 &
    local deadline=$((SECONDS + 90))
    until ros2 topic list 2>/dev/null | grep -q '^/odom$'; do
        sleep 2; (( SECONDS > deadline )) && { echo "$mark TIMEOUT /odom" >> "$MASTER"; return 1; }
    done
    ros2 launch ssf_gazebo map_server.launch.py map:=$MAP_YAML use_sim_time:=true \
        > "$logdir/map.log" 2>&1 &
    deadline=$((SECONDS + 45))
    until ros2 topic list 2>/dev/null | grep -q '^/map$'; do
        sleep 1; (( SECONDS > deadline )) && { echo "$mark TIMEOUT /map" >> "$MASTER"; return 1; }
    done
    ros2 launch ssf_gazebo localization.launch.py use_sim_time:=true > "$logdir/amcl.log" 2>&1 &
    sleep 6
    ros2 launch ssf_gazebo navigation.launch.py use_sim_time:=true > "$logdir/nav.log" 2>&1 &
    deadline=$((SECONDS + 90))
    until ros2 action list 2>/dev/null | grep -q 'follow_path'; do
        sleep 2; (( SECONDS > deadline )) && { echo "$mark TIMEOUT follow_path" >> "$MASTER"; return 1; }
    done
    echo "$mark stack up $(date -u +%FT%TZ)" >> "$MASTER"

    $PYTHON -u -m bench.run_bench --start-from $s --num-pairs $e --planners $PLANNERS \
        --pairs-json "$PAIRS" --models-dir "$MODELS" --seed $SEED \
        --trial-timeout-sec $TRIAL_TIMEOUT --out-dir "$OUT" \
        > "$logdir/bench.log" 2>&1
    echo "$mark bench rc=$? $(date -u +%FT%TZ)" >> "$MASTER"
}

declare -a PGIDS=()
cleanup() {
    for g in "${PGIDS[@]}"; do kill -INT -- -$g 2>/dev/null; done
    sleep 3
    for g in "${PGIDS[@]}"; do kill -KILL -- -$g 2>/dev/null; done
}
trap cleanup EXIT INT TERM

N=$(( P1 - P0 )); PER=$(( (N + NUM_WORKERS - 1) / NUM_WORKERS ))
echo "===== start $(date -u +%FT%TZ) workers=$NUM_WORKERS pairs=[$P0,$P1) =====" >> "$MASTER"
export -f worker_main
export REPO PYTHON OUT PAIRS PLANNERS GPUS DOMAIN_BASE TRIAL_TIMEOUT SEED MODELS MAP_YAML LOGS MASTER
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    s=$(( P0 + i * PER )); e=$(( P0 + (i + 1) * PER )); (( e > P1 )) && e=$P1
    (( s >= P1 )) && continue
    setsid bash -c "worker_main $i $s $e" &
    PGIDS+=($!)
    echo "worker $i pgid $! pairs [$s,$e)" >> "$MASTER"
    sleep 8
done
wait
echo "===== done $(date -u +%FT%TZ) =====" >> "$MASTER"
