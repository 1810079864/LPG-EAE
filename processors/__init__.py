import json
from .processor_multiarg import MultiargProcessor


_DATASET_DIR = {
    "rams": {
        "train_file": './data/RAMS_1.0/data_final/train.jsonlines',
        "dev_file": './data/RAMS_1.0/data_final/dev.jsonlines',
        "test_file": './data/RAMS_1.0/data_final/test.jsonlines',
        "max_span_num_file": "./data/dset_meta/role_num_rams.json",
    },
    "wikievent": {
        "train_file": './data/WikiEvent/data_final/train.jsonl',
        "dev_file": './data/WikiEvent/data_final/dev.jsonl',
        "test_file": './data/WikiEvent/data_final/test.jsonl',
        "max_span_num_file": "./data/dset_meta/role_num_wikievent.json",
    },
}

def get_processor(config, tokenizer):
    return MultiargProcessor(config, tokenizer)


def build_processor(args, tokenizer):
    if args.dataset_type not in _DATASET_DIR:
        raise NotImplementedError("Please use valid dataset name")
    args.train_file=_DATASET_DIR[args.dataset_type]['train_file']
    args.dev_file = _DATASET_DIR[args.dataset_type]['dev_file']
    args.test_file = _DATASET_DIR[args.dataset_type]['test_file']

    args.role_name_mapping = None

    if args.model_type=="base":
        with open(_DATASET_DIR[args.dataset_type]['max_span_num_file']) as f:
            args.max_span_num_dict = json.load(f)

    processor = MultiargProcessor(args, tokenizer)
    return processor
