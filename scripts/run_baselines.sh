#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Reproduce the Maze2D table (Umaze / Medium / Large): 13 methods x 3 maps, 100 episodes each, seed 42.
#
# Method -> harness / safety layer:
#   StreamingFlow / SSF (open, closed)  -> scripts/plan_maze2d_sfp.py, diffuser/models/cbf.py (DT-HOCBF)
#   FM / SafeFM / FlowMatcher
#     / SafeFlowMatcher                 -> scripts/plan_maze2d.py, diffuser/models/cbf_diffuser.py (class-K CBF)
#   Diffuser / Diffuser+CG (=GD)
#     / SafeDiffuser (ReS)              -> scripts/plan_maze2d.py, diffuser/models/diffusion.py
#   A* / RRT*                           -> scripts/plan_maze2d_classic.py (CPU)
#
# Checkpoints are read from $LOGBASE/<dataset>/{diffusion,cfm,sfp}/..., where the training scripts write them:
#   Diffuser family  diffusion/H{horizon}_T{n_diffusion_steps}
#   FM family        cfm/H{horizon}_T{n_diffusion_steps}
#   Streaming family sfp/k0.1_s0.005 (all three maps)
#
# Usage: [PY=python] [GPUS="0 1 2 3"] [JOBS_PER_GPU=3] [N_ITERS=100] [LOGBASE=logs] [RESULTS_ROOT=runs/<date>] scripts/run_baselines.sh
# Per-run results: $LOGBASE/<dataset>/plans/release_*/<TAG>/{episodes.csv,summary.json} (A*/RRT* included)
# ----------------------------------------------------------------------------

set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

PY=${PY:-python}
GPUS=${GPUS:-"0 1 2 3"}                    # GPU pool
JOBS_PER_GPU=${JOBS_PER_GPU:-3}            # concurrent jobs per GPU
N_ITERS=${N_ITERS:-100}                    # evaluation episodes per run
LOGBASE="${LOGBASE:-logs}"
SEED="${SEED:-42}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/runs/$(date +%Y%m%d_%H%M%S)}"   # launcher logs
mkdir -p "$RESULTS_ROOT"
echo "[INFO] Output logs → $RESULTS_ROOT"

# ---------------------------------------------------------------- job spec
# Each entry: "TAG | SCRIPT | METHOD | DATASET | EXTRA_FLAGS..."
JOBS=()

add() {
    JOBS+=("$1|$2|$3|$4|$5")
}

# Streaming flow checkpoint (k, sigma_train): sfp/k0.1_s0.005 on every map
SFP_KS="--k 0.1 --sigma_train 0.005"

# Safety-correction hyperparameters of SafeDiffuser / SafeFM / SafeFlowMatcher
SAFE_HP="--robust_term 0.1 --relax_threshold 0.95"

for DS in maze2d-umaze-v1 maze2d-medium-v1 maze2d-large-v1; do
    # ---------- Diffuser family ----------
    add "diffuser__${DS}"           plan_maze2d.py     base $DS  "--safety_enabled false"
    add "diffuser_cg__${DS}"        plan_maze2d.py     base $DS  "--safety_enabled true --safety_method gd"
    add "safediffuser__${DS}"       plan_maze2d.py     base $DS  "--safety_enabled true --safety_method invariance ${SAFE_HP}"
    # ---------- FM family ----------
    add "fm__${DS}"                 plan_maze2d.py     cfm  $DS  "--safety_enabled false --integrator pure"
    add "safefm__${DS}"             plan_maze2d.py     cfm  $DS  "--safety_enabled true --integrator pure ${SAFE_HP}"
    # ---------- FlowMatcher family (integrator='pc' + one-step prediction stage) ----------
    add "flowmatcher__${DS}"        plan_maze2d.py     cfm  $DS  "--safety_enabled false --integrator pc --one_shot_enabled true"
    add "safeflowmatcher__${DS}"    plan_maze2d.py     cfm  $DS  "--safety_enabled true --integrator pc --one_shot_enabled true ${SAFE_HP}"
    # ---------- Streaming family (StreamingFlow = no filter, SSF = DT-HOCBF filter) ----------
    add "streamingflow_open__${DS}"   plan_maze2d_sfp.py  sfp  $DS  "${SFP_KS} --safety_enabled false --closed_loop false"
    add "streamingflow_closed__${DS}" plan_maze2d_sfp.py  sfp  $DS  "${SFP_KS} --safety_enabled false --closed_loop true"
    add "ssf_open__${DS}"             plan_maze2d_sfp.py  sfp  $DS  "${SFP_KS} --safety_enabled true --closed_loop false --kp_hocbf 0.15 --kv_hocbf 0.08"
    add "ssf_closed__${DS}"           plan_maze2d_sfp.py  sfp  $DS  "${SFP_KS} --safety_enabled true --closed_loop true --kp_hocbf 0.15 --kv_hocbf 0.08"
done

# ---------------------------------------------------------------- launcher
run_one() {
    local gpu="$1" tag="$2" script="$3" method="$4" ds="$5" extras="$6"
    local log="${RESULTS_ROOT}/${tag}.log"
    echo "[GPU ${gpu}] ▶ ${tag}"
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/$script \
        --config config.maze2d \
        --dataset $ds \
        --logbase $LOGBASE \
        --method $method \
        --seed $SEED \
        --suffix $tag \
        --n_episodes $N_ITERS \
        $extras >"$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[GPU ${gpu}] ✔ ${tag}"
    else
        echo "[GPU ${gpu}] ✘ ${tag}  (rc=$rc, see $log)"
    fi
    return $rc
}

# Slot pool: JOBS_PER_GPU slots per GPU. Each slot tracks one PID.
SLOT_GPUS=()
for g in $GPUS; do
    for ((i=0; i<JOBS_PER_GPU; i++)); do
        SLOT_GPUS+=("$g")
    done
done
N_SLOTS=${#SLOT_GPUS[@]}
declare -a SLOT_PID
for ((s=0; s<N_SLOTS; s++)); do SLOT_PID[$s]=0; done

acquire_slot() {
    while true; do
        for ((s=0; s<N_SLOTS; s++)); do
            pid=${SLOT_PID[$s]}
            if [ "$pid" = "0" ] || ! kill -0 "$pid" 2>/dev/null; then
                echo "$s"; return
            fi
        done
        sleep 1
    done
}

echo "[INFO] Total jobs: ${#JOBS[@]}    GPU pool: $GPUS    Slots/GPU: $JOBS_PER_GPU    Total slots: $N_SLOTS"
START_TS=$(date +%s)

# A* / RRT* run on the CPU, in the background while the GPU jobs run
CLASSIC_PIDS=()
for DS in maze2d-umaze-v1 maze2d-medium-v1 maze2d-large-v1; do
    for PL in astar rrt_star; do
        tag="${PL}__${DS}"
        echo "[CPU] ▶ ${tag}"
        CUDA_VISIBLE_DEVICES="" $PY scripts/plan_maze2d_classic.py \
            --dataset $DS --planner_type $PL --logbase $LOGBASE --seed $SEED --suffix $tag \
            --n_episodes $N_ITERS >"${RESULTS_ROOT}/${tag}.log" 2>&1 &
        CLASSIC_PIDS+=($!)
    done
done

for spec in "${JOBS[@]}"; do
    IFS='|' read -r tag script method ds extras <<< "$spec"
    s=$(acquire_slot)
    gpu=${SLOT_GPUS[$s]}
    run_one "$gpu" "$tag" "$script" "$method" "$ds" "$extras" &
    SLOT_PID[$s]=$!
done

wait
END_TS=$(date +%s)
elapsed=$(( END_TS - START_TS ))
echo
echo "============================================================"
echo "All ${#JOBS[@]} GPU jobs + ${#CLASSIC_PIDS[@]} CPU jobs done in $((elapsed/60))m $((elapsed%60))s"
echo "Output logs:    $RESULTS_ROOT"
echo "Per-method results under:"
echo "  ${LOGBASE}/<dataset>/plans/release_*/<TAG>/summary.json, episodes.csv"
echo "============================================================"
