#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

LR="${LR:-2e-5}"
GPU_ID="${GPU_ID:-1}"
MODEL_PATH="${MODEL_PATH:-./checkpoints/roberta-large}"
MAX_STEPS="${MAX_STEPS:-10000}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS=(22 42 66 99 111 1234)

# The complete RoBERTa checkpoint is available locally.  Force Transformers
# and huggingface_hub to avoid slow network retries on offline servers.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

for MODEL_FILE in config.json tokenizer.json vocab.json merges.txt; do
    if [[ ! -s "$MODEL_PATH/$MODEL_FILE" ]]; then
        echo "Missing local RoBERTa file: $MODEL_PATH/$MODEL_FILE" >&2
        exit 1
    fi
done
if [[ ! -s "$MODEL_PATH/model.safetensors" && ! -s "$MODEL_PATH/pytorch_model.bin" ]]; then
    echo "Missing local RoBERTa weights under: $MODEL_PATH" >&2
    exit 1
fi

REQUIRED_DATA=(
    ./data/ace_eeqa/train_convert.json
    ./data/ace_eeqa/dev_convert.json
    ./data/ace_eeqa/test_convert.json
)
for DATA_FILE in "${REQUIRED_DATA[@]}"; do
    if [[ ! -s "$DATA_FILE" ]]; then
        echo "Missing licensed ACE05 file: $PROJECT_ROOT/${DATA_FILE#./}" >&2
        echo "Place the PAIE-style converted ACE05 splits in data/ace_eeqa/." >&2
        exit 1
    fi
done

for SEED in "${SEEDS[@]}"; do
    WORK_PATH="exps/ace05/$SEED/$LR"
    mkdir -p "$WORK_PATH"

    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u engine.py \
        --model_type=LPG-EAE \
        --dataset_type=ace05 \
        --model_name_or_path="$MODEL_PATH" \
        --role_path=./data/dset_meta/description_ace.csv \
        --prompt_path=./data/prompts/prompts_ace_full.csv \
        --feature_cache_dir=./exps/ace05/feature_cache \
        --seed="$SEED" \
        --output_dir="$WORK_PATH" \
        --learning_rate="$LR" \
        --max_steps="$MAX_STEPS" \
        --batch_size=4 \
        --infer_batch_size=32 \
        --max_enc_seq_length=512 \
        --max_prompt_seq_length=512 \
        --max_span_length=10 \
        --warmup_steps=0.1 \
        --light_prompt_layers=1 \
        --light_prompt_heads=8 \
        --light_prompt_ffn_ratio=2.0 \
        --light_prompt_dropout=0.1 \
        --use_dual_stream_gat \
        --gat_query_fuse_weight=0.2 \
        --lamb=0.1 \
        --bipartite
done
