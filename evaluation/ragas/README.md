# RAGAS — Evaluating RAG Pipelines

Evaluating retrieval-augmented generation with [RAGAS](https://github.com/explodinggradients/ragas), scored by a local Qwen2.5-7B model.

## Why RAGAS?

**The problem it addresses.** A RAG pipeline has two failure modes that look identical from the outside. The answer is wrong because retrieval never fetched the right passage — or because retrieval worked and the model ignored it. Eyeballing outputs cannot tell you which, so you end up tuning the prompt when the chunk size was the problem. And "it looks better" does not survive contact with a second reviewer or a code change three weeks later.

**What it gives you.** Metrics that decompose the pipeline instead of scoring it end to end. Faithfulness asks whether the answer's claims are grounded in the retrieved context — a generation-side measure. Context precision and recall ask whether retrieval surfaced the right material and ranked it well — a retrieval-side measure. Read together they point at the component to fix. Most are computed by an LLM judge, so they need no human labels, which is what makes them practical to run on every change.

**Use it when**

- You are iterating on a RAG pipeline and need to know which half to fix.
- You are comparing chunking strategies, embedding models, or `k` values and want a number.
- You want a regression gate on a golden set before a retrieval change ships.

**Skip it when**

- There is no retrieval. For plain generation quality, reach for [DeepEval](../deepeval/README.md).
- You already have relevance labels. Classic IR metrics — recall@k, nDCG, MRR — are deterministic, free, and not subject to judge noise.
- You need an absolute quality number to report. These are comparative signals, best read as deltas between two configurations of your own system.

**In production.** The judge is the instrument, so treat it like one: pin the model, keep `temperature=0`, and re-baseline whenever you change it. Judge with a model at least as strong as the one under test — **a 7B local judge, as used here, is directionally useful and no more**, which is fine for learning the metrics and too noisy to gate a release. Track scores over time rather than against a fixed threshold, and keep a small human-reviewed set to sanity-check the judge itself.

## Alternatives

| Instead of | Consider | Why |
|---|---|---|
| RAGAS | [DeepEval](../deepeval/README.md) | Broader metric set, pytest-native assertions, safety metrics |
| LLM-judged retrieval | recall@k, nDCG, MRR | Deterministic and free when you have relevance labels |
| A metrics library | [TruLens](https://www.trulens.org/), [Phoenix](https://phoenix.arize.com/) | Evaluation plus tracing — see the retrieval that produced the score |
| Python-defined evals | [promptfoo](https://www.promptfoo.dev/) | Config-driven evals and red-teaming, CI-friendly |
| Rolling your own | [MLflow LLM Evaluate](https://mlflow.org/docs/latest/llms/llm-evaluate/) | If experiment tracking already lives in MLflow |

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
