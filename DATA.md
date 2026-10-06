# Data preparation

The repository includes prompt templates and role descriptions, but not the original datasets.

Expected files:

```text
data/
├── RAMS_1.0/data_final/{train,dev,test}.jsonlines
├── WikiEvent/data_final/{train,dev,test}.jsonl
├── MLEE/
│   ├── MLEE_role_name_mapping.json
│   └── data_final/{train,test}.json
└── ace_eeqa/{train_convert,dev_convert,test_convert}.json
```

For MLEE, `scripts/train_mlee_roberta.sh` creates a deterministic document-disjoint development split from `train.json` using `scripts/split_mlee_train_dev.py`.

ACE05 is licensed and must be obtained from the Linguistic Data Consortium. Convert it to the PAIE-style JSON format before training.

Features and typed sparse graphs are generated on the first run and cached under the experiment cache directory. Subsequent seeds reuse the same cache.
