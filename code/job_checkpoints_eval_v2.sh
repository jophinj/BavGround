#!/bin/bash
#SBATCH --job-name=genba_ckpt_eval
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=ckpt_eval_%j.out

set -euo pipefail

# ─────────────────────────────────────────────────────────────
# User variables (edit these)
# ─────────────────────────────────────────────────────────────
BASE="/dss/dssfs03/pn69za/pn69za-dss-0005/engeba/checkpoints/training"
BASE="../../checkpoints/training"
DATASET="/dss/dssfs03/pn69za/pn69za-dss-0005/engeba/aisystem/checkpoint_work/dataset/simple.json"
DATASET="./dataset/simple.json"
OUTDIR="./out_ckpts"

# checkpoint selection
CKPT_REGEX='^checkpoint([0-9]+)$'
CKPT_RANGE="0:2000:500"   # start:end:step  (set empty "" to evaluate ALL matched)

# dataset filters (set "" to not filter)
FILTER_TYPE=""            # e.g. "mcq" or "mcq,classify"
FILTER_LANG=""            # e.g. "en" or "en,de,bar"
FILTER_CATEGORY=""        # e.g. "food,festival"
FILTER_BIAS=""            # e.g. "cultural,political,gender"

# generation knobs
MAX_NEW=128
TEMP=0.7
TOP_P=0.9

# ─────────────────────────────────────────────────────────────
# Container start (same pattern you already use)
# ─────────────────────────────────────────────────────────────
if [ ! -d "/raid/enroot/data/user-4236196/engeba_eval" ]; then
  echo "Creating engeba_eval container..."
  enroot create --name=engeba_eval ../enroot/engeba_eval.sqsh
fi

enroot start \
  --mount /dss/dssfs03/pn69za/pn69za-dss-0005/engeba:/workspace/engeba \
  engeba_eval \
  bash <<EOF
set -euo pipefail

cd /workspace/engeba/aisystem/checkpoint_work
echo "Inside container: \$(pwd)"
nvidia-smi

pip install --upgrade pip
pip install torch transformers accelerate

mkdir -p "$OUTDIR"

python run_checkpoints_eval_v2.py \
  --base "$BASE" \
  --dataset "$DATASET" \
  --outdir "$OUTDIR" \
  --ckpt-regex "$CKPT_REGEX" \
  $( [[ -n "$CKPT_RANGE" ]] && echo --range "$CKPT_RANGE" ) \
  $( [[ -n "$FILTER_TYPE" ]] && echo --type "$FILTER_TYPE" ) \
  $( [[ -n "$FILTER_LANG" ]] && echo --lang "$FILTER_LANG" ) \
  $( [[ -n "$FILTER_CATEGORY" ]] && echo --category "$FILTER_CATEGORY" ) \
  $( [[ -n "$FILTER_BIAS" ]] && echo --bias "$FILTER_BIAS" ) \
  --max-new "$MAX_NEW" \
  --temperature "$TEMP" \
  --top-p "$TOP_P"

echo "DONE"
EOF

