#!/bin/bash

# HTCondor training script for NeuroGym environments
# Usage: ./train_neurogym.sh ENV_NAME MODEL_TYPE HIDDEN_DIM TOTAL_FRAMES [NUM_LAYERS] [BUFFER_SIZE] [TAG] [MASK_FIXATION]

set -e  # Exit on error

# Parse arguments
ENV_NAME=$1      # e.g., GoNogo-v0, PerceptualDecisionMaking-v0
MODEL_TYPE=$2    # mpn, mpn-frozen, or mpn-poly
HIDDEN_DIM=$3    # e.g., 128
TOTAL_FRAMES=$4  # e.g., 50000
NUM_LAYERS=${5:-1}         # e.g., 1 (default: 1)
BUFFER_SIZE=${6:-2000}     # replay buffer capacity in frames (default: 2000)
TAG=${7:-""}               # optional wandb tag
MASK_FIXATION=${8:-"0"}    # 1 = zero fixation-period rewards, 0 = keep them
LEARNING_RATE=${9:-0.001}  # learning rate (default: 0.001)

MAX_EPISODE_STEPS=1000

# Short env name for experiment naming
case "$ENV_NAME" in
    ContextDecisionMaking-v0)        ENV_SHORT="cdm"  ;;
    DelayComparison-v0)              ENV_SHORT="dc"   ;;
    DelayMatchSampleDistractor1D-v0) ENV_SHORT="dmsd" ;;
    DelayPairedAssociation-v0)       ENV_SHORT="dpa"  ;;
    *) ENV_SHORT=$(echo $ENV_NAME | sed 's/-v0//' | tr '[:upper:]' '[:lower:]') ;;
esac

# Short LR string for experiment naming
case "$LEARNING_RATE" in
    0.00001) LR_SHORT="1e-05" ;;
    0.0001)  LR_SHORT="1e-4"  ;;
    0.001)   LR_SHORT="1e-3"  ;;
    *)       LR_SHORT="$LEARNING_RATE" ;;
esac

EXP_NAME="${TAG:+${TAG}-}${ENV_SHORT}-${MODEL_TYPE}-l${NUM_LAYERS}-h${HIDDEN_DIM}-${LR_SHORT}"

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
echo "Buffer Size: $BUFFER_SIZE"
echo "Tag: $TAG"
echo "Mask Fixation Reward: $MASK_FIXATION"
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
python main_a2c.py train-neurogym \
    --experiment-name ${EXP_NAME} \
    --env-name ${ENV_NAME} \
    --max-episode-steps ${MAX_EPISODE_STEPS} \
    --total-frames ${TOTAL_FRAMES} \
    --model-type ${MODEL_TYPE} \
    --hidden-dim ${HIDDEN_DIM} \
    --num-layers ${NUM_LAYERS} \
    --learning-rate "${LEARNING_RATE}" \
    --device gpu \
    --buffer-size ${BUFFER_SIZE} \
    --num-eval-episodes 10 \
    --wandb \
    --wandb-project mpn-rl \
    ${TAG:+--tag "$TAG"} \
    $([[ "$MASK_FIXATION" == "1" ]] && echo "--mask-fixation-reward")

# Check exit status
EXIT_CODE=$?

echo ""
echo "========================================"
echo "Training completed"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
