# RAGAS — Evaluating RAG Pipelines

Evaluating retrieval-augmented generation with [RAGAS](https://github.com/explodinggradients/ragas), scored by a local Qwen2.5-7B model.

## Contents

| Path | What it is |
|---|---|
| [`notebooks/ragas-evaluation-tutorial.ipynb`](notebooks/ragas-evaluation-tutorial.ipynb) | The tutorial, 7 sections, runs top-to-bottom |
| [`scripts/ragas_evaluation.py`](scripts/ragas_evaluation.py) | Same content as `demo_*()` coroutines, `main()` awaits them in order |

## How to run

```bash
ollama pull qwen2.5:7b         # once
uv sync --group ragas

uv run python evaluation/ragas/scripts/ragas_evaluation.py
uv run jupyter lab             # or the notebook
```

## Sections

1. **Faithfulness** — are the response's claims supported by the retrieved context?
2. **Context Precision** — is the relevant context ranked highly?
3. **Context Recall** — did retrieval get everything the reference needs?
4. **Factual Correctness** — response vs reference, by claim
5. **Semantic Similarity** — embedding-based, no LLM in the loop
6. **Custom Metric** — a `DiscreteMetric` with your own rubric
7. **End-to-End Batch Evaluation** — several metrics across a dataset

## Notes

- **Everything is async.** Metrics are called as `await scorer.ascore(...)`, and the script's `main()` runs under `asyncio.run()`. In the notebook the cells `await` directly.
- The LLM comes from `llm_factory("qwen2.5:7b", client=AsyncOpenAI(base_url="http://localhost:11434/v1", ...))` — the `openai` package is just an HTTP client for Ollama.
- **Embeddings do not go through Ollama.** Section 5 uses a local `sentence-transformers/all-MiniLM-L6-v2` model, which is why this group pulls `sentence-transformers` (and with it, torch).
- The group pins `langchain-community>=0.3,<0.4`: ragas 0.4 imports `langchain_community.chat_models.vertexai`, which was removed in 0.4, so a newer version breaks `import ragas` outright.
