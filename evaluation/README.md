# Evaluation

Benchmarks, metrics, and model comparisons. Every tutorial here scores model output using a **local** Qwen2.5-7B via Ollama, so no evaluation data leaves the machine.

## Tutorials

| Folder | What it covers | Install |
|---|---|---|
| [`ragas/`](ragas/) | RAG-pipeline metrics: faithfulness, context precision/recall, factual correctness, semantic similarity, custom metrics | `uv sync --group ragas` |
| [`deepeval/`](deepeval/) | LLM-output metrics: relevancy, hallucination, bias, toxicity, G-Eval, batch evaluation | `uv sync --group deepeval` |

Each folder has its own `README.md` with the section list, how to run it, and the gotchas specific to that framework.

## Layout

```
evaluation/<tutorial>/
├── README.md        # what it covers + how to run
├── notebooks/       # the tutorial, runs top-to-bottom
└── scripts/         # the same content as demo_*() functions
```

See the [root README](../README.md#adding-a-tutorial) for how to add one.
