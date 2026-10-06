#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Create/reuse a deterministic PMID-disjoint 90/10 train/dev split.
"$PYTHON_BIN" -u scripts/split_mlee_train_dev.py \
    --source=./data/MLEE/data_final/train.json \
    --train-output=./data/MLEE/data_final/train_split.json \
    --dev-output=./data/MLEE/data_final/dev.json \
    --manifest=./data/MLEE/data_final/split_manifest.json \
    --dev-ratio=0.1 \
    --seed=42 \
    --trials=50000

# Learning rate 固定
MODEL_PATH="${MODEL_PATH:-./checkpoints/roberta-large}"
GPU_ID="${GPU_ID:-0}"

LR="${LR:-2e-5}"
MAX_STEPS="${MAX_STEPS:-10000}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# 定义种子数组
SEEDS=(22 42 66 99 111 1234)

# 遍历种子数组
for SEED in "${SEEDS[@]}"
do
    # 设置工作路径
    work_path=exps/mlee/$SEED/$LR
    mkdir -p "$work_path"

    # 运行 Python 脚本
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u engine.py \
        --model_type=LPG-EAE \
        --dataset_type=MLEE \
        --context_representation=decoder \
        --model_name_or_path="$MODEL_PATH" \
        --role_path=./data/MLEE/MLEE_role_name_mapping.json \
        --prompt_path=./data/prompts/prompts_MLEE_full.csv \
        --seed=$SEED \
        --output_dir=$work_path \
        --feature_cache_dir=./exps/mlee/feature_cache \
        --learning_rate=$LR \
        --batch_size=4 \
        --max_steps="$MAX_STEPS" \
        --max_enc_seq_length 512 \
        --max_prompt_seq_length 512 \
        --max_dec_seq_length 512 \
        --window_size 250 \
        --bipartite \
        --lamb 0.1    # --max_span_length 10 --warmup_steps 500 \
done
