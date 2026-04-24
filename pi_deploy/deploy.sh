#!/bin/bash
# deploy.sh — Sync pi_deploy to Raspberry Pi and optionally run inference
#
# Usage:
#   bash deploy.sh              # sync files only
#   bash deploy.sh --run        # sync + run inference on all slides in input/
#   bash deploy.sh --run --mxa  # sync + run with Memryx MXA accelerator

set -e

PI_HOST="admin@20.0.0.154"
PI_PASS="admin"
PI_PATH="~/Desktop/clam_pipeline/pi_deploy"
SSH="sshpass -p '$PI_PASS' ssh -o StrictHostKeyChecking=no $PI_HOST"
SCP="sshpass -p '$PI_PASS' scp -o StrictHostKeyChecking=no"

RUN=false
USE_MXA=false

for arg in "$@"; do
  case $arg in
    --run) RUN=true ;;
    --mxa) USE_MXA=true ;;
  esac
done

echo "=== Syncing pi_deploy → Pi ($PI_HOST) ==="

# Sync pipeline script and checkpoints
$SCP pipeline_mxa.py         $PI_HOST:$PI_PATH/pipeline_mxa.py
$SCP checkpoints/s_0_checkpoint.pt $PI_HOST:$PI_PATH/checkpoints/s_0_checkpoint.pt

echo "Sync complete."

if [ "$RUN" = true ]; then
  MXA_FLAG=""
  [ "$USE_MXA" = true ] && MXA_FLAG="--use_mxa"

  echo ""
  echo "=== Running inference on Pi ==="
  eval $SSH "cd $PI_PATH && source venv/bin/activate && \
    python pipeline_mxa.py \
      --input input/ \
      --output output/ \
      --batch \
      $MXA_FLAG"
fi
