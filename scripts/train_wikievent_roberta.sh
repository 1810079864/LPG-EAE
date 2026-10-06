#!/usr/bin/env bash
set -euo pipefail

# Always run relative paths from this project, regardless of the caller's cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

MODEL_PATH="${MODEL_PATH:-./checkpoints/roberta-large}"
# Learning rate固定
# LR=3e-5
LR=2e-5
# 消融开关：
# 图只用于共享双流 GAT 和触发词 Query 增强，不参与上下文压缩。

# 定义种子数组
SEEDS=(22 42 66 99 111 1234)
# SEEDS=(22)


# 遍历种子数组
for SEED in "${SEEDS[@]}"
do
    # 设置工作路径
    work_path=exps/wikievent/$SEED/$LR
    mkdir -p "$work_path"

    python -u engine.py \
        --model_type=LPG-EAE \
        --dataset_type=wikievent \
        --model_name_or_path="$MODEL_PATH" \
        --role_path=./data/dset_meta/description_wikievent.csv \
        --prompt_path=./data/prompts/prompts_wikievent_full.csv \
        --seed=$SEED \
        --output_dir=$work_path \
        --feature_cache_dir=./exps/wikievent/feature_cache \
        --learning_rate=$LR \
        --max_steps=10000 \
        --max_enc_seq_length 512 \
        --max_prompt_seq_length 512 \
        --light_prompt_layers 1 \
        --light_prompt_heads 8 \
        --light_prompt_ffn_ratio 2.0 \
        --light_prompt_dropout 0.1 \
        --use_dual_stream_gat \
        --gat_query_fuse_weight 0.2 \
        --bipartite
        
done
