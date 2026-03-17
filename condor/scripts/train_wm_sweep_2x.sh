#!/bin/bash

# HTCondor training script for Working Memory hyperparameter sweep (2x frames)
# Usage: ./train_wm_sweep_2x.sh ENV_NAME MODEL_TYPE LEARNING_RATE ETA LAMBDA_DECAY

set -e  # Exit on error

# Parse arguments
ENV_NAME=$1      # e.g., DelayMatchSample-v0
MODEL_TYPE=$2    # mpn, mpn-frozen, rnn, or lstm
LEARNING_RATE=$3 # e.g., 0.001
ETA=$4           # e.g., 0.1 (MPN only)
LAMBDA=$5        # e.g., 0.95 (MPN only)

# ---- Sweep identity ----
TAG="wm-sweep-v2"

# ---- Fixed parameters for this sweep ----
HIDDEN_DIM=128
NUM_LAYERS=1
TOTAL_FRAMES=400000
MAX_EPISODE_STEPS=500
BUFFER_SIZE=50000
BATCH_SIZE=16
UTD=32
EPSILON_START=0.3
EPSILON_END=0.01
TARGET_UPDATE_TAU=0.995
NUM_EVAL_EPISODES=10
PRINT_FREQ=2000
CHECKPOINT_FREQ=20000

# Generate experiment name (experiment-id appended automatically by main.py)
ENV_SHORT=$(echo $ENV_NAME | sed 's/-v0//' | tr '[:upper:]' '[:lower:]')
LR_TAG=$(echo $LEARNING_RATE | sed 's/^0\.//' | sed 's/\./_/')

if [ "$MODEL_TYPE" = "mpn" ]; then
    ETA_TAG=$(echo $ETA | sed 's/^0\.//' | sed 's/\./_/')
    LAM_TAG=$(echo $LAMBDA | sed 's/^0\.//' | sed 's/\./_/')
    EXP_NAME="wm-${ENV_SHORT}-${MODEL_TYPE}-lr${LR_TAG}-eta${ETA_TAG}-lam${LAM_TAG}"
else
    EXP_NAME="wm-${ENV_SHORT}-${MODEL_TYPE}-lr${LR_TAG}"
fi

# Print job information
echo "========================================"
echo "Working Memory Sweep Job (2x frames)"
echo "========================================"
echo "Experiment: $EXP_NAME"
echo "Tag: $TAG"
echo "Environment: $ENV_NAME"
echo "Model Type: $MODEL_TYPE"
echo "Hidden Dim: $HIDDEN_DIM"
echo "Num Layers: $NUM_LAYERS"
echo "Total Frames: $TOTAL_FRAMES"
echo "Learning Rate: $LEARNING_RATE"
echo "Eta: $ETA (MPN only)"
echo "Lambda: $LAMBDA (MPN only)"
echo "Buffer Size: $BUFFER_SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "UTD: $UTD"
echo "Epsilon: $EPSILON_START -> $EPSILON_END"
echo "Target Update Tau: $TARGET_UPDATE_TAU"
echo "----------------------------------------"
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
echo "========================================"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Run training (experiment-id auto-generated and appended to name by main.py)
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
    --buffer-size ${BUFFER_SIZE} \
    --batch-size ${BATCH_SIZE} \
    --utd ${UTD} \
    --epsilon-start ${EPSILON_START} \
    --epsilon-end ${EPSILON_END} \
    --target-update-tau ${TARGET_UPDATE_TAU} \
    --num-eval-episodes ${NUM_EVAL_EPISODES} \
    --print-freq ${PRINT_FREQ} \
    --checkpoint-freq ${CHECKPOINT_FREQ} \
    --tag ${TAG} \
    --device gpu \
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
