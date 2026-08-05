# Fine-Tuning

Fine-tuning scripts and configs. **No tutorials here yet.**

Planned ground: LoRA / QLoRA adapters, dataset preparation, PEFT training loops, and merging adapters back into a base model for serving.

## Layout to follow

```
fine-tuning/<tutorial>/
├── README.md        # what it covers + how to run
├── notebooks/       # the tutorial, runs top-to-bottom
└── scripts/         # the same content as demo_*() functions
```

Add the tutorial's dependencies as a new group in [`pyproject.toml`](../pyproject.toml), then list it here and in the [root README](../README.md). Training data belongs in [`data/`](../data/), not in the tutorial folder.
