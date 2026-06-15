#!/bin/bash

# HTCondor training script for a2c_run1 experiments (main repo)
# Usage: ./train_a2c_run1.sh ENV_NAME MODEL_TYPE HIDDEN_DIM TOTAL_FRAMES

set -e

ENV_NAME=$1
MODEL_TYPE=$2
HIDDEN_DIM=$3
TOTAL_FRAMES=$4

LEARNING_RATE=0.001
MAX_EPISODE_STEPS=1000
ENTROPY_COEF=0.01
VALUE_COEF=0.5
FRAMES_PER_BATCH=100
NUM_LAYERS=1

ENV_SHORT=$(echo $ENV_NAME | sed 's/-v0//' | sed 's/IntervalDiscrimination/intdisc/' | sed 's/GoNogo/gonogo/' | tr '[:upper:]' '[:lower:]')
MODEL_SHORT=$(echo $MODEL_TYPE | sed 's/-/_/g')
EXP_NAME="a2c_run1-${ENV_SHORT}-${MODEL_SHORT}"

echo "========================================"
echo "A2C Run1 Training Job (main repo)"
echo "========================================"
echo "Experiment: $EXP_NAME"
echo "Environment: $ENV_NAME"
echo "Model Type: $MODEL_TYPE"
echo "Hidden Dim: $HIDDEN_DIM"
echo "Total Frames: $TOTAL_FRAMES"
echo "Learning Rate: $LEARNING_RATE"
echo "----------------------------------------"
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
echo "========================================"
echo ""

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Starting training..."
python main_a2c.py train-neurogym \
    --experiment-name ${EXP_NAME} \
    --env-name ${ENV_NAME} \
    --max-episode-steps ${MAX_EPISODE_STEPS} \
    --total-frames ${TOTAL_FRAMES} \
    --model-type ${MODEL_TYPE} \
    --hidden-dim ${HIDDEN_DIM} \
    --num-layers ${NUM_LAYERS} \
    --learning-rate ${LEARNING_RATE} \
    --entropy-coef ${ENTROPY_COEF} \
    --value-coef ${VALUE_COEF} \
    --frames-per-batch ${FRAMES_PER_BATCH} \
    --device gpu \
    --num-eval-episodes 10 \
    --wandb \
    --wandb-project mpn-rl \
    --tag a2c-run1

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Training completed"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
