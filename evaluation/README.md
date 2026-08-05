# Evaluation

Benchmarks, metrics, and model comparisons. Every tutorial here scores model output using a **local** Qwen2.5-7B via Ollama, so no evaluation data leaves the machine.

## Why evaluate at all?

Prompt and model changes have no compiler and no stack trace. A tweak that fixes one case silently regresses four others, and "it seems better" does not survive a second reviewer or a rollback three weeks later. Evaluation converts that into a number you can compare across runs and gate a deploy on.

Both tutorials use **LLM-as-judge**: a model scores the output, so you need no human labels and can run on every change. The cost is that your instrument is itself a language model — pin it, keep `temperature=0`, and re-baseline whenever it changes. **A 7B local judge, as used here, is directionally useful and no more** — right for learning the metrics, too noisy to gate a release on.

## Which one?

| Your question | Reach for | Because |
|---|---|---|
| Is my RAG failing at retrieval or at generation? | [`ragas/`](ragas/) | Metrics decompose the pipeline — faithfulness is generation-side, context precision/recall is retrieval-side |
| Did this prompt change regress quality? | [`deepeval/`](deepeval/) | Thresholded, pytest-native assertions that fail a build |
| Is the output biased or toxic? | [`deepeval/`](deepeval/) | Ships safety metrics; RAGAS does not |
| Does it satisfy a rubric I can only describe in words? | [`deepeval/`](deepeval/) | G-Eval turns plain-English criteria into a scorer |
| Which chunk size / embedding model / `k` wins? | [`ragas/`](ragas/) | Built for comparing retrieval configurations |

They overlap on faithfulness and relevancy, and running both on the same case is a reasonable sanity check — agreement raises confidence, disagreement usually means the judge is out of its depth.

Neither replaces a deterministic check. If correctness means valid JSON or an exact identifier, assert it with [Pydantic](../inference/pydantic/README.md) instead: cheaper, instant, and never flaky.

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
