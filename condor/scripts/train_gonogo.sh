#!/bin/bash

# HTCondor training script for GoNogo neurogym task
# Usage: ./train_gonogo.sh MODEL_TYPE ETA LAMBDA HIDDEN_DIM NUM_EPISODES MAX_EPISODE_STEPS SUFFIX

set -e  # Exit on error

# Parse arguments
MODEL_TYPE=$1   # mpn, mpn-frozen, or rnn
ETA=$2
LAMBDA=$3
HIDDEN_DIM=$4
NUM_EPISODES=$5
MAX_EPISODE_STEPS=$6
LEARNING_RATE=$7
SEQUENCE_LENGTH=$8
SUFFIX=$9       # verb-noun suffix (e.g., "running-tiger")

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
echo "Episodes: $NUM_EPISODES"
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
    --num-episodes ${NUM_EPISODES} \
    --model-type ${MODEL_TYPE} \
    --hidden-dim ${HIDDEN_DIM} \
    --eta ${ETA} \
    --lambda-decay ${LAMBDA} \
    --sequence-length ${SEQUENCE_LENGTH} \
    --learning-rate ${LEARNING_RATE} \
    --buffer-size 20

# Check exit status
EXIT_CODE=$?

echo ""
echo "========================================"
echo "Training completed"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"


exit $EXIT_CODE
