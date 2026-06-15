#!/bin/bash

# HTCondor training script for a2c_run2 (main repo, episode-based BPTT)
# Usage: ./train_a2c_run2.sh ENV_NAME MODEL_TYPE HIDDEN_DIM NUM_EPISODES

set -e

ENV_NAME=$1
MODEL_TYPE=$2
HIDDEN_DIM=$3
MAX_FRAMES=$4

ENV_SHORT=$(echo $ENV_NAME | sed 's/-v0//' | sed 's/IntervalDiscrimination/intdisc/' | sed 's/GoNogo/gonogo/' | tr '[:upper:]' '[:lower:]')
MODEL_SHORT=$(echo $MODEL_TYPE | sed 's/-/_/g')
EXP_NAME="a2c_run2-${ENV_SHORT}-${MODEL_SHORT}"

echo "========================================"
echo "A2C Run2 Training Job (main repo, BPTT)"
echo "========================================"
echo "Experiment: $EXP_NAME"
echo "Environment: $ENV_NAME"
echo "Model Type: $MODEL_TYPE"
echo "Hidden Dim: $HIDDEN_DIM"
echo "Num Episodes: $NUM_EPISODES"
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
    --total-frames ${MAX_FRAMES} \
    --model-type ${MODEL_TYPE} \
    --hidden-dim ${HIDDEN_DIM} \
    --learning-rate 0.0001 \
    --gamma 0.98 \
    --value-coef 1.0 \
    --entropy-coef 0.01 \
    --grad-clip 10.0 \
    --tbptt-len 50 \
    --print-freq 500 \
    --num-eval-episodes 10 \
    --device gpu \
    --wandb \
    --wandb-project mpn-rl \
    --tag a2c-run2

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Training completed"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
