# LPG-EAE

Official anonymous implementation of **LPG-EAE**, a light-prompt and graph-enhanced framework for document-level multi-event argument extraction.

LPG-EAE jointly encodes the document and multi-event prompt with RoBERTa, refines prompt representations through lightweight prompt-to-document cross-attention, and injects trigger-centred evidence from a dual-stream graph encoder into each role query. Argument boundaries are predicted over the complete, uncompressed document representation.

## Architecture

1. **Shared document-prompt encoding** preserves every document token for span extraction.
2. **Light Prompt Cross-Attention** reconstructs role queries from document evidence without a second full RoBERTa pass.
3. **Dual-Stream GAT** separately models dependency/coreference edges and trigger-association edges.
4. **Query-preserving graph fusion** adds graph evidence orthogonal to the original role query.
5. **Full-context span pointer** predicts argument boundaries on the original contextual representation.

## Environment

```bash
conda create -n lpg-eae python=3.8 -y
conda activate lpg-eae
pip install -r requirements.txt
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_sm
```

The code was checked with Python 3.8, PyTorch 2.4.1, Transformers 4.46.3, and spaCy 3.1.7. Place a local `fastcoref` checkpoint at `fcoref_model/`. If FastCoref is unavailable, the code logs a warning and continues without coreference edges.

Place RoBERTa-large in `checkpoints/roberta-large/`, or pass another local path through `MODEL_PATH`.

## Data

Dataset files are not redistributed. Follow [DATA.md](DATA.md) and place the converted RAMS, WikiEvents, MLEE, and ACE05 files under `data/`. Prompt templates and role metadata are included.

## Training

```bash
MODEL_PATH=./checkpoints/roberta-large CUDA_VISIBLE_DEVICES=0 bash scripts/train_wikievent_roberta.sh
MODEL_PATH=./checkpoints/roberta-large CUDA_VISIBLE_DEVICES=0 bash scripts/train_rams_roberta.sh
MODEL_PATH=./checkpoints/roberta-large CUDA_VISIBLE_DEVICES=0 bash scripts/train_mlee_roberta.sh
MODEL_PATH=./checkpoints/roberta-large GPU_ID=0 bash scripts/train_ace05_roberta.sh
```

Module ablations:

```bash
MODEL_PATH=./checkpoints/roberta-large GPU_ID=0 bash scripts/train_wikievent_module_ablation.sh
MODEL_PATH=./checkpoints/roberta-large GPU_ID=0 bash scripts/train_rams_module_ablation.sh
```

The four ablations disable Light Prompt, the trigger graph, the dependency/coreference graph, or the complete graph module.

## Repository layout

```text
models/       LPG-EAE and graph/light-prompt modules
processors/   dataset conversion and offline graph caching
runner/       training, evaluation, and component profiling
scripts/      training and ablation entry points
data/         prompt/role metadata and dataset placeholders
```

## Acknowledgements

This implementation builds on the public DEEIA and PAIE codebases. Their papers and repositories should be cited when using this code. Citation information for LPG-EAE will be added after anonymous review.
