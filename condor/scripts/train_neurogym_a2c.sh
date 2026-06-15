#!/bin/bash

# HTCondor training script for NeuroGym environments using A2C
# Usage: ./train_neurogym_a2c.sh ENV_NAME MODEL_TYPE HIDDEN_DIM TOTAL_FRAMES [NUM_LAYERS] [TAG]

set -e

ENV_NAME=$1
MODEL_TYPE=$2
HIDDEN_DIM=$3
TOTAL_FRAMES=$4
NUM_LAYERS=${5:-1}
TAG=${6:-""}

LEARNING_RATE=0.001
MAX_EPISODE_STEPS=1000
ENTROPY_COEF=0.01
VALUE_COEF=0.5
FRAMES_PER_BATCH=100

ENV_SHORT=$(echo $ENV_NAME | sed 's/-v0//' | tr '[:upper:]' '[:lower:]')
EXP_NAME="${ENV_SHORT}-${MODEL_TYPE}"

echo "========================================"
echo "NeuroGym A2C Training Job"
echo "========================================"
echo "Experiment: $EXP_NAME"
echo "Environment: $ENV_NAME"
echo "Model Type: $MODEL_TYPE"
echo "Hidden Dim: $HIDDEN_DIM"
echo "Num Layers: $NUM_LAYERS"
echo "Total Frames: $TOTAL_FRAMES"
echo "Max Episode Steps: $MAX_EPISODE_STEPS"
echo "Learning Rate: $LEARNING_RATE"
echo "Entropy Coef: $ENTROPY_COEF"
echo "Value Coef: $VALUE_COEF"
echo "Frames per Batch: $FRAMES_PER_BATCH"
echo "Tag: $TAG"
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
    ${TAG:+--tag "$TAG"}

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Training completed"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
