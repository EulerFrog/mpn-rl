#!/bin/bash

# HTCondor training script for GoNogo neurogym task
# Usage: ./train_gonogo.sh MODEL_TYPE ETA LAMBDA HIDDEN_DIM NUM_LAYERS TOTAL_FRAMES MAX_EPISODE_STEPS LEARNING_RATE SUFFIX

set -e  # Exit on error

# Parse arguments
MODEL_TYPE=$1   # mpn, mpn-frozen, rnn, or lstm
ETA=$2
LAMBDA=$3
HIDDEN_DIM=$4
NUM_LAYERS=$5
TOTAL_FRAMES=$6
MAX_EPISODE_STEPS=$7
LEARNING_RATE=$8
SUFFIX=$9    # suffix (e.g., "spicy_layer1")

# Generate experiment name: gonogo-model-verb-noun
EXP_NAME="gonogo-${MODEL_TYPE}-${SUFFIX}"

# Print job information
echo "========================================"
echo "GoNogo Training Job"
echo "========================================"
echo "Experiment: $EXP_NAME"
echo "Model Type: $MODEL_TYPE"
echo "Eta: $ETA"
echo "Lambda: $LAMBDA"
echo "Hidden Dim: $HIDDEN_DIM"
echo "Num Layers: $NUM_LAYERS"
echo "Total Frames: $TOTAL_FRAMES"
echo "Max Episode Steps: $MAX_EPISODE_STEPS"
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
    --env-name GoNogo-v0 \
    --max-episode-steps ${MAX_EPISODE_STEPS} \
    --total-frames ${TOTAL_FRAMES} \
    --model-type ${MODEL_TYPE} \
    --hidden-dim ${HIDDEN_DIM} \
    --num-layers ${NUM_LAYERS} \
    --eta ${ETA} \
    --lambda-decay ${LAMBDA} \
    --learning-rate ${LEARNING_RATE} \
    --device gpu \
    --buffer-size 20000 \
    --num-eval-episodes 10

# Check exit status
EXIT_CODE=$?

echo ""
echo "========================================"
echo "Training completed"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"


exit $EXIT_CODE
