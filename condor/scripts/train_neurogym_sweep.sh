#!/bin/bash

# HTCondor script for large A2C hyperparameter sweep on NeuroGym environments
# Usage: ./train_neurogym_sweep.sh ENV_NAME MODEL_TYPE NUM_LAYERS HIDDEN_DIM LEARNING_RATE [LSTM_FORGET_BIAS] [TAG]

set -e

ENV_NAME=$1           # e.g., GoNogo-v0, PerceptualDecisionMaking-v0
MODEL_TYPE=$2         # mpn, mpn-frozen, rnn, lstm
NUM_LAYERS=$3         # 1, 2, 3
HIDDEN_DIM=$4         # 64, 128, 256
LEARNING_RATE=$5      # 0.0001, 0.00001
LSTM_FORGET_BIAS=${6:-0.0}  # forget gate bias init (default: 0.0)
TAG=${7:-ng-sweep-v1}

NUM_EPISODES=200000
MAX_EPISODE_STEPS=500
GAMMA=0.98
VALUE_COEF=1.0
GRAD_CLIP=10.0
PRINT_FREQ=50

case "$ENV_NAME" in
    ContextDecisionMaking-v0)              ENV_SHORT="cdm"   ;;
    DelayComparison-v0)                    ENV_SHORT="dc"    ;;
    DelayMatchSample-v0)                   ENV_SHORT="dms"   ;;
    DelayMatchSampleDistractor1D-v0)       ENV_SHORT="dmsd"  ;;
    DelayPairedAssociation-v0)             ENV_SHORT="dpa"   ;;
    GoNogo-v0)                             ENV_SHORT="gonogo";;
    IntervalDiscrimination-v0)             ENV_SHORT="id"    ;;
    MultiSensoryIntegration-v0)            ENV_SHORT="msi"   ;;
    PerceptualDecisionMaking-v0)           ENV_SHORT="pdm"   ;;
    PerceptualDecisionMakingDelayResponse-v0) ENV_SHORT="pdmdr";;
    ProbabilisticReasoning-v0)             ENV_SHORT="pr"    ;;
    *)                                     ENV_SHORT=$(echo "$ENV_NAME" | sed 's/-v0//' | tr '[:upper:]' '[:lower:]') ;;
esac

case "$LEARNING_RATE" in
    0.0001)  LR_SHORT="1e-4" ;;
    0.00001) LR_SHORT="1e-5" ;;
    *)       LR_SHORT="$LEARNING_RATE" ;;
esac

FB_SUFFIX=$([ "$(echo "$LSTM_FORGET_BIAS != 0.0" | bc -l)" = "1" ] && echo "-fb${LSTM_FORGET_BIAS}" || echo "")
EXP_NAME="${TAG}-${ENV_SHORT}-${MODEL_TYPE}${FB_SUFFIX}-l${NUM_LAYERS}-h${HIDDEN_DIM}-${LR_SHORT}"

echo "========================================"
echo "A2C NeuroGym Hyperparameter Sweep"
echo "========================================"
echo "Experiment:    $EXP_NAME"
echo "Environment:   $ENV_NAME"
echo "Model:         $MODEL_TYPE"
echo "Num layers:    $NUM_LAYERS"
echo "Hidden dim:    $HIDDEN_DIM"
echo "Learning rate: $LEARNING_RATE"
echo "Num episodes:  $NUM_EPISODES"
echo "Gamma:         $GAMMA"
echo "LSTM forget bias: $LSTM_FORGET_BIAS"
echo "Tag:           $TAG"
echo "----------------------------------------"
echo "Start time:    $(date)"
echo "Hostname:      $(hostname)"
echo "Working dir:   $(pwd)"
echo "========================================"
echo ""

source .venv/bin/activate

python main_a2c.py train-neurogym \
    --experiment-name "${EXP_NAME}" \
    --env-name "${ENV_NAME}" \
    --model-type "${MODEL_TYPE}" \
    --num-layers "${NUM_LAYERS}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --num-episodes "${NUM_EPISODES}" \
    --learning-rate "${LEARNING_RATE}" \
    --max-episode-steps "${MAX_EPISODE_STEPS}" \
    --gamma "${GAMMA}" \
    --value-coef "${VALUE_COEF}" \
    --grad-clip "${GRAD_CLIP}" \
    --print-freq "${PRINT_FREQ}" \
    --device gpu \
    --wandb \
    --wandb-project mpn-rl \
    --tag "${TAG}" \
    ${LSTM_FORGET_BIAS:+$([ "$(echo "$LSTM_FORGET_BIAS != 0.0" | bc -l)" = "1" ] && echo "--lstm-forget-bias ${LSTM_FORGET_BIAS}")}

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Training completed"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
