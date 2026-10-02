#!/usr/bin/env bash
# Evaluate the nine rows of the F1TENTH table on Budapest and Catalunya with the
# eval_all.py defaults: 100 episodes, seed 42, start/goal noise sigma = 0.02 m,
# K = 512 for the FM family.
# Usage: bash scripts/eval.sh [checkpoint_root] [extra eval_all.py arguments]
# Choose the GPU with CUDA_VISIBLE_DEVICES and the interpreter with PYTHON.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
ROOT=${1:-checkpoints}
shift || true

for T in budapest catalunya; do
  for M in diffuser diffuser_cg safediffuser fm safefm flowmatcher safeflowmatcher sfp_off safe_sfp_off; do
    $PY scripts/eval_all.py --method "$M" --track "$T" --checkpoint_root "$ROOT" "$@"
  done
done
