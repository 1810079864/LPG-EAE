# Data preparation

The repository includes prompt templates and role descriptions, but not the original datasets.

Expected files:

```text
data/
├── RAMS_1.0/data_final/{train,dev,test}.jsonlines
└── WikiEvent/data_final/{train,dev,test}.jsonl
```

Features and typed sparse graphs are generated on the first run and cached under the experiment cache directory. Subsequent seeds reuse the same cache.
