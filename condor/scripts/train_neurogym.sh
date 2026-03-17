#!/bin/bash

# HTCondor training script for NeuroGym environments
# Usage: ./train_neurogym.sh ENV_NAME MODEL_TYPE HIDDEN_DIM TOTAL_FRAMES

set -e  # Exit on error

# Parse arguments
ENV_NAME=$1      # e.g., GoNogo-v0, PerceptualDecisionMaking-v0
MODEL_TYPE=$2    # mpn, mpn-frozen, or mpn-poly
HIDDEN_DIM=$3    # e.g., 128
TOTAL_FRAMES=$4  # e.g., 50000

# Fixed parameters
NUM_LAYERS=1
LEARNING_RATE=0.001
MAX_EPISODE_STEPS=1000

# Generate experiment name: env-model (clean env name)
ENV_SHORT=$(echo $ENV_NAME | sed 's/-v0//' | tr '[:upper:]' '[:lower:]')
EXP_NAME="${ENV_SHORT}-${MODEL_TYPE}"

# Print job information
echo "========================================"
echo "NeuroGym Training Job"
echo "========================================"
echo "Experiment: $EXP_NAME"
echo "Environment: $ENV_NAME"
echo "Model Type: $MODEL_TYPE"
echo "Hidden Dim: $HIDDEN_DIM"
echo "Num Layers: $NUM_LAYERS"
echo "Total Frames: $TOTAL_FRAMES"
echo "Max Episode Steps: $MAX_EPISODE_STEPS"
echo "Learning Rate: $LEARNING_RATE"
echo "----------------------------------------"
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
echo "========================================"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Run training
echo "Starting training..."
python main.py train-neurogym \
    --experiment-name ${EXP_NAME} \
    --env-name ${ENV_NAME} \
    --max-episode-steps ${MAX_EPISODE_STEPS} \
    --total-frames ${TOTAL_FRAMES} \
    --model-type ${MODEL_TYPE} \
    --hidden-dim ${HIDDEN_DIM} \
    --num-layers ${NUM_LAYERS} \
    --learning-rate ${LEARNING_RATE} \
    --device gpu \
    --buffer-size 20000 \
    --num-eval-episodes 10 \
    --wandb \
    --wandb-project mpn-rl

# Check exit status
EXIT_CODE=$?

echo ""
echo "========================================"
echo "Training completed"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
