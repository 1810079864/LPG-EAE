#!/usr/bin/env bash
set -euo pipefail

# One paired full-model reference plus four RAMS module ablations:
#   1) without_light_prompt
#   2) without_trigger_graph
#   3) without_dependency_coreference_graph
#   4) without_entire_graph

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

GPU_ID="${GPU_ID:-1}"
LR="${LR:-2e-5}"
MAX_STEPS="${MAX_STEPS:-10000}"
MODEL_PATH="${MODEL_PATH:-./checkpoints/roberta-large}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_ROOT="${RESULT_ROOT:-exps_module_ablation/rams}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-./exps/rams_large/feature_cache}"
SEEDS=(22 42 66 99)

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable is unavailable: $PYTHON_BIN" >&2
    exit 1
fi

ABLATIONS=(
    full_model
    without_light_prompt
    without_trigger_graph
    without_dependency_coreference_graph
    without_entire_graph
)

mkdir -p "$RESULT_ROOT"

for ABLATION in "${ABLATIONS[@]}"; do
    EXTRA_ARGS=()
    case "$ABLATION" in
        full_model)
            ;;
        without_light_prompt)
            EXTRA_ARGS+=(--no_light_prompt)
            ;;
        without_trigger_graph)
            EXTRA_ARGS+=(--no_trigger_graph)
            ;;
        without_dependency_coreference_graph)
            EXTRA_ARGS+=(--no_dependency_coref_graph)
            ;;
        without_entire_graph)
            EXTRA_ARGS+=(--no_dual_stream_gat)
            ;;
        *)
            echo "Unknown ablation: $ABLATION" >&2
            exit 2
            ;;
    esac

    for SEED in "${SEEDS[@]}"; do
        WORK_PATH="$RESULT_ROOT/$ABLATION/$SEED/$LR"
        mkdir -p "$WORK_PATH"
        echo "[ModuleAblation] dataset=rams group=$ABLATION seed=$SEED output=$WORK_PATH"

        CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u engine.py \
            --model_type=LPG-EAE \
            --dataset_type=rams \
            --model_name_or_path="$MODEL_PATH" \
            --role_path=./data/dset_meta/description_rams.csv \
            --prompt_path=./data/prompts/prompts_rams_full.csv \
            --feature_cache_dir="$FEATURE_CACHE_DIR" \
            --seed="$SEED" \
            --output_dir="$WORK_PATH" \
            --learning_rate="$LR" \
            --max_steps="$MAX_STEPS" \
            --batch_size=4 \
            --infer_batch_size=32 \
            --max_enc_seq_length=512 \
            --max_prompt_seq_length=512 \
            --max_span_length=10 \
            --warmup_steps=500 \
            --light_prompt_layers=1 \
            --light_prompt_heads=8 \
            --light_prompt_ffn_ratio=2.0 \
            --light_prompt_dropout=0.1 \
            --gat_query_fuse_weight=0.2 \
            --bipartite \
            "${EXTRA_ARGS[@]}"
    done
done
