# DeepEval — Evaluating LLM Outputs

Evaluating LLM outputs with [DeepEval](https://github.com/confident-ai/deepeval), judged by a local Qwen2.5-7B model instead of a hosted API.

## Why DeepEval?

**The problem it addresses.** Prompt and model changes have no compiler. You tweak a system prompt to fix one case, ship it, and quietly regress four others — and nothing fails, because there is no test to fail. Traditional assertions do not help either: you cannot `assertEqual` a paragraph, and a regex that pins the exact wording breaks the moment the model rephrases.

**What it gives you.** LLM evaluation shaped like unit testing. Metrics carry a `threshold`, produce a pass/fail plus a reason, and run under pytest, so quality checks live in CI next to everything else. The metric library spans correctness (faithfulness, relevancy, hallucination) and safety (bias, toxicity), and **G-Eval** lets you state a bespoke rubric in plain English — "penalise answers that give medical advice" — and have the judge apply it consistently.

**Use it when**

- You want a build to fail when a prompt change degrades output quality.
- Your quality bar is a rubric, not a fixed string — G-Eval is the reason to be here.
- You need safety screening (bias, toxicity) as a gate rather than an afterthought.
- You are comparing two models or two prompts and want per-metric evidence.

**Skip it when**

- A deterministic check would do. If correctness means valid JSON or an exact ID, validate it with [Pydantic](../../inference/pydantic/README.md) or a regex — cheaper, instant, and never flaky.
- Your question is specifically *where* a RAG pipeline fails. [RAGAS](../ragas/README.md) decomposes retrieval from generation more directly.
- You cannot afford the calls. Every metric on every test case is one or more LLM invocations; a broad suite gets expensive against a paid API and slow against a local one.

**In production.** Judged scores drift when the judge changes, so pin the judge model and treat any change to it as a re-baseline. Set thresholds from an observed distribution rather than intuition — start by recording scores without failing the build. Keep the gating suite small and the exploratory suite separate; you want CI minutes spent on the cases that actually catch regressions. Note that DeepEval will reach for OpenAI unless every metric is passed `model=`, which is the adapter this tutorial exists to build.

## Alternatives

| Instead of | Consider | Why |
|---|---|---|
| DeepEval for RAG | [RAGAS](../ragas/README.md) | Purpose-built to separate retrieval failures from generation failures |
| Python test files | [promptfoo](https://www.promptfoo.dev/) | Declarative YAML evals and red-teaming, language-agnostic |
| Local-only evaluation | [LangSmith](https://docs.smith.langchain.com/), [Phoenix](https://phoenix.arize.com/) | Datasets, tracing and eval history as a platform |
| LLM-judged metrics | Human review on a small set | Still the ground truth every judge is calibrated against |
| A framework | A hand-written judge prompt | For one or two criteria, a prompt plus a threshold is honest and transparent |

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
