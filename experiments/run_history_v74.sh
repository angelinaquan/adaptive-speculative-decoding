#!/usr/bin/env bash
set -euo pipefail
R=/data/scene-rep/u/aquan/speculative-tonight-20261109
export SERVING_SCRATCH=/tmp/aquan-serving-v14-3043052.VFLAcm
export WANDB_BASE_URL=https://api.wandb.ai
"$SERVING_SCRATCH/env/bin/python" "$R/replay_history_v74.py"
"$SERVING_SCRATCH/env/bin/python" "$R/publish_history_v74.py"
