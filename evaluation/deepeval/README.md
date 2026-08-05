# DeepEval — Evaluating LLM Outputs

Evaluating LLM outputs with [DeepEval](https://github.com/confident-ai/deepeval), judged by a local Qwen2.5-7B model instead of a hosted API.

## Contents

| Path | What it is |
|---|---|
| [`notebooks/deepeval-evaluation-tutorial.ipynb`](notebooks/deepeval-evaluation-tutorial.ipynb) | The tutorial, 9 sections + custom model setup, runs top-to-bottom |
| [`scripts/deepeval_evaluation.py`](scripts/deepeval_evaluation.py) | Same content as `demo_*()` functions, `main()` runs them in order |

## How to run

```bash
ollama pull qwen2.5:7b         # once
uv sync --group deepeval

uv run python evaluation/deepeval/scripts/deepeval_evaluation.py
uv run jupyter lab             # or the notebook
```

## Sections

0. **Custom Model Setup** — the `OllamaLLM` adapter (read this first, everything depends on it)
1. **Faithfulness** — claims grounded in the retrieval context
2. **Answer Relevancy** — does the answer address the input?
3. **Contextual Precision** — relevant context ranked highly
4. **Contextual Recall** — expected information actually retrieved
5. **Contextual Relevancy** — proportion of retrieved context that is useful
6. **Hallucination** — contradictions against provided context
7. **Bias & Toxicity** — safety-oriented metrics
8. **G-Eval** — custom criteria evaluated by the LLM itself
9. **End-to-End Batch Evaluation** — `evaluate()` across several test cases

## Notes

- DeepEval has **no first-class Ollama backend**, so the tutorial hand-writes an `OllamaLLM(DeepEvalBaseLLM)` adapter. It appends the JSON schema to the prompt and runs the reply through `_extract_json()`, which strips markdown fences — a local 7B model does not reliably return bare JSON.
- **Every metric must be passed `model=llm` explicitly.** Omit it and DeepEval falls back to OpenAI and fails without an API key.
- `evaluate()` writes a `.deepeval/` cache directory at the repo root; it is gitignored.
- On DeepEval 4.x, `LLMTestCaseParams` emits a deprecation warning in favour of `SingleTurnParams`. The tutorial still uses the older name and works.
