#!/bin/bash
#SBATCH --job-name=genba_ckpt_eval
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=ckpt_eval_%j.out

set -euo pipefail

# ─────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────
DEFAULT_BASE="../../checkpoints/training"
DEFAULT_DATASET="./dataset/simple.json"
DEFAULT_OUTDIR="./out_ckpts_it"

DEFAULT_MODELS_BASE="../other_models/"

# checkpoint mode defaults
DEFAULT_CKPT_REGEX='^checkpoint([0-9]+)$'
DEFAULT_CKPT_RANGE="0:41500:500"

# dataset filters
DEFAULT_FILTER_TYPE=""
DEFAULT_FILTER_LANG=""
DEFAULT_FILTER_CATEGORY=""
DEFAULT_FILTER_BIAS=""

# generation knobs
DEFAULT_MAX_NEW=128
DEFAULT_TEMP=0.7
DEFAULT_TOP_P=0.9

# container/workdir
CONTAINER_NAME="engeba_eval"
CONTAINER_IMAGE="../enroot/engeba_eval.sqsh"
HOST_MOUNT_SRC="/dss/dssfs03/pn69za/pn69za-dss-0005/engeba"
CONTAINER_MOUNT_DST="/workspace/engeba"
WORKDIR_IN_CONTAINER="/workspace/engeba/aisystem/checkpoint_work"

# ─────────────────────────────────────────────────────────────
# Usage
# ─────────────────────────────────────────────────────────────
usage() {
  echo "Usage:"
  echo
  echo "  Checkpoint mode:"
  echo "    sbatch $0"
  echo "    sbatch $0 checkpoint"
  echo
  echo "  Other-model mode:"
  echo "    sbatch --export=MODE=models,MODELS=EuroLLM-9B-Instruct,MODELS_BASE=../../aisystem/other_models/EU,DATASET=./dataset/living_traditions_customs.json $0"
  echo "    sbatch --export=DATASET=./dataset $0"
  echo "    sbatch --export=DATASET=./dataset/politics.json,./dataset/language.json $0"
  echo
  echo "Environment variables:"
  echo "  MODE=checkpoint|models"
  echo "  BASE=...                # checkpoint base dir"
  echo "  DATASET=...             # one JSON file, comma-separated JSON files, or a directory of JSON files"
  echo "  OUTDIR=..."
  echo "  CKPT_REGEX=..."
  echo "  CKPT_RANGE=..."
  echo
  echo "  MODELS=model1,model2"
  echo "  MODELS_BASE=...         # base dir containing model subdirs"
  echo
  echo "  FILTER_TYPE=mcq,classify,generate"
  echo "  FILTER_LANG=en,de,bar"
  echo "  FILTER_CATEGORY=..."
  echo "  FILTER_BIAS=cultural,political,gender"
  echo
  echo "  MAX_NEW=128"
  echo "  TEMP=0.7"
  echo "  TOP_P=0.9"
  exit 1
}

# ─────────────────────────────────────────────────────────────
# Read inputs
# ─────────────────────────────────────────────────────────────
MODE=models
#MODELS=genba-10b-it,EuroLLM-9B-Instruct,granite-3.0-8b-instruct,Llama-3.1-8B-Instruct,LLaMmlein_7B_chat,Mistral-7B-Instruct-v0.3,occiglot-7b-eu5-instruct,OLMo-2-1124-7B-Instruct,Pharia-1-LLM-7B-control-hf,Qwen2.5-7B-Instruct,salamandra-7b-instruct,Teuken-7B-instruct-research-v0.4,gemma-2-9b-it,leo-hessianai-7b-chat,aya-expanse-8b,Apertus-8B-Instruct-2509
#MODELS=EuroLLM-9B,LLaMmlein_7B,Llama-3.1-8B,aya-expanse-8b,salamandra-7b,Teuken-7B-base-v0.6,Mistral-7B-v0.3,granite-3.0-8b-base,Qwen2.5-7B,OLMo-2-1124-7B,occiglot-7b-eu5,bloom-7b1,granite-3.0-8b-base
MODELS=genba-10b-it,Apertus-8B-Instruct-2509
MODELS_BASE=../other_models
#MODELS_BASE=../other_models
#DATASET="./dataset/politics.json"
DATASET=./dataset
MODE="${MODE:-${1:-checkpoint}}"

BASE="${BASE:-$DEFAULT_BASE}"
DATASET="${DATASET:-$DEFAULT_DATASET}"
OUTDIR="${OUTDIR:-$DEFAULT_OUTDIR}"

CKPT_REGEX="${CKPT_REGEX:-$DEFAULT_CKPT_REGEX}"
CKPT_RANGE="${CKPT_RANGE:-$DEFAULT_CKPT_RANGE}"

MODELS="${MODELS:-}"
MODELS_BASE="${MODELS_BASE:-$DEFAULT_MODELS_BASE}"

FILTER_TYPE="${FILTER_TYPE:-$DEFAULT_FILTER_TYPE}"
FILTER_LANG="${FILTER_LANG:-$DEFAULT_FILTER_LANG}"
FILTER_CATEGORY="${FILTER_CATEGORY:-$DEFAULT_FILTER_CATEGORY}"
FILTER_BIAS="${FILTER_BIAS:-$DEFAULT_FILTER_BIAS}"

MAX_NEW="${MAX_NEW:-$DEFAULT_MAX_NEW}"
TEMP="${TEMP:-$DEFAULT_TEMP}"
TOP_P="${TOP_P:-$DEFAULT_TOP_P}"

case "$MODE" in
  checkpoint|models)
    ;;
  *)
    echo "[ERROR] Invalid MODE: $MODE"
    usage
    ;;
esac

if [[ "$MODE" == "models" && -z "$MODELS" ]]; then
  echo "[ERROR] MODE=models requires MODELS=model1,model2,..."
  usage
fi

echo "==> Mode            : $MODE"
echo "==> Dataset         : $DATASET"
echo "==> Output dir      : $OUTDIR"
echo "==> Filter type     : $FILTER_TYPE"
echo "==> Filter lang     : $FILTER_LANG"
echo "==> Filter category : $FILTER_CATEGORY"
echo "==> Filter bias     : $FILTER_BIAS"
echo "==> Max new         : $MAX_NEW"
echo "==> Temperature     : $TEMP"
echo "==> Top-p           : $TOP_P"

if [[ "$MODE" == "checkpoint" ]]; then
  echo "==> Checkpoint base : $BASE"
  echo "==> CKPT regex      : $CKPT_REGEX"
  echo "==> CKPT range      : $CKPT_RANGE"
else
  echo "==> Models          : $MODELS"
  echo "==> Models base     : $MODELS_BASE"
fi

# ─────────────────────────────────────────────────────────────
# Container start
# ─────────────────────────────────────────────────────────────
if [ ! -d "/raid/enroot/data/user-4236196/${CONTAINER_NAME}" ]; then
  echo "Creating ${CONTAINER_NAME} container..."
  enroot create --name="${CONTAINER_NAME}" "${CONTAINER_IMAGE}"
fi

# ─────────────────────────────────────────────────────────────
# Run inside container
# ─────────────────────────────────────────────────────────────
enroot start \
  --mount "${HOST_MOUNT_SRC}:${CONTAINER_MOUNT_DST}" \
  "${CONTAINER_NAME}" \
  bash <<EOF
set -euo pipefail

cd "${WORKDIR_IN_CONTAINER}"
echo "Inside container: \$(pwd)"
nvidia-smi

pip install --upgrade pip
pip install torch transformers accelerate

mkdir -p "${OUTDIR}"

COMMON_ARGS=(
  --dataset "${DATASET}"
  --outdir "${OUTDIR}"
  --max-new "${MAX_NEW}"
  --temperature "${TEMP}"
  --top-p "${TOP_P}"
)

if [[ -n "${FILTER_TYPE}" ]]; then
  COMMON_ARGS+=( --type "${FILTER_TYPE}" )
fi
if [[ -n "${FILTER_LANG}" ]]; then
  COMMON_ARGS+=( --lang "${FILTER_LANG}" )
fi
if [[ -n "${FILTER_CATEGORY}" ]]; then
  COMMON_ARGS+=( --category "${FILTER_CATEGORY}" )
fi
if [[ -n "${FILTER_BIAS}" ]]; then
  COMMON_ARGS+=( --bias "${FILTER_BIAS}" )
fi

if [[ "${MODE}" == "checkpoint" ]]; then
  echo
  echo "============================================================"
  echo "Running checkpoint evaluation"
  echo "Base       : ${BASE}"
  echo "Dataset    : ${DATASET}"
  echo "Outdir     : ${OUTDIR}"
  echo "CKPT regex : ${CKPT_REGEX}"
  echo "CKPT range : ${CKPT_RANGE}"
  echo "============================================================"

  python run_checkpoints_eval_v4.py \
    --base "${BASE}" \
    --ckpt-regex "${CKPT_REGEX}" \
    \$( [[ -n "${CKPT_RANGE}" ]] && printf '%q ' --range "${CKPT_RANGE}" ) \
    "\${COMMON_ARGS[@]}"

else
  echo
  echo "============================================================"
  echo "Running standalone model evaluation"
  echo "Models     : ${MODELS}"
  echo "Models base: ${MODELS_BASE}"
  echo "Dataset    : ${DATASET}"
  echo "Outdir     : ${OUTDIR}"
  echo "============================================================"

  python run_checkpoints_eval_v4.py \
    --models "${MODELS}" \
    --models-base "${MODELS_BASE}" \
    "\${COMMON_ARGS[@]}"
fi

echo "DONE"
EOF
