#!/usr/bin/env bash
# Train every model used in the F1TENTH table (Budapest, Catalunya) with the
# settings of the paper models (the scripts' defaults).
# Prerequisite: raw laps in logs/<Track>/*.csv (scripts/collect_trajectories.py).
# Choose the GPU with CUDA_VISIBLE_DEVICES and the interpreter with PYTHON.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

# Preprocess: processed_data/m_per_s/ and processed_data/m_per_step/
$PY scripts/regenerate_processed_data.py

for T in budapest catalunya; do
  $PY scripts/train_diffuser.py --track $T   # Diffuser, Diffuser+CG, SafeDiffuser
  $PY scripts/train_cfm.py --track $T        # FM, FlowMatcher; SafeFM, SafeFlowMatcher on Budapest
  $PY scripts/train_sfp.py --track $T        # StreamingFlow, SSF
done
# SafeFM, SafeFlowMatcher on Catalunya: CFM trained on the m/s data
$PY scripts/train_cfm.py --track catalunya --velocity_units m_per_s
