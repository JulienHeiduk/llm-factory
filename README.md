# llm-factory

LLM experimentation workspace: evaluation, deployment, quantization, and more.

All experiments use **Qwen2.5-7B** served locally with [Ollama](https://ollama.com/) — no API key needed, fully offline.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) from `pyproject.toml`, with one dependency group per tutorial so you install only what you are running.

```bash
# 1. Pull the model (once)
ollama pull qwen2.5:7b

# 2. Create .venv with the base deps + Jupyter
uv sync

# 3. Add the group for whichever tutorial you want
uv sync --group langchain
uv sync --group ragas
uv sync --group deepeval
```

`uv sync` alone is enough for the **Pydantic** tutorial — `openai` and `pydantic[email]` are base dependencies.

Run scripts with `uv run`, which uses `.venv` automatically:

```bash
uv run python inference/pydantic/scripts/pydantic_tutorial.py
```

For notebooks, launch JupyterLab from the same environment, or point your IDE's interpreter at `.venv/bin/python`:

```bash
uv run jupyter lab
```

> `uv sync` prunes anything not in `uv.lock`, so install extras through the groups rather than `pip install` into `.venv`.

## Structure

```
llm-factory/
├── pyproject.toml        # dependency groups, one per tutorial
├── uv.lock               # pinned, committed
├── evaluation/           # Benchmarks, metrics, model comparisons
│   ├── ragas/
│   └── deepeval/
├── inference/            # Building with LLMs, inference optimization
│   ├── langchain/
│   ├── langgraph-memory/
│   ├── pydantic/
│   └── vllm/
├── deployment/           # Serving, APIs, containerization      (empty)
├── fine-tuning/          # Fine-tuning scripts and configs      (empty)
├── quantization/         # Model compression experiments        (empty)
├── data/
│   ├── raw/              # Raw datasets
│   └── processed/        # Cleaned/prepared datasets
└── configs/              # Model and experiment configurations
```

Every folder has a `README.md`. Each tutorial is a folder holding its own notebook, script, and docs:

```
<area>/<tutorial>/
├── README.md        # what it covers + how to run
├── notebooks/       # the tutorial, runs top-to-bottom
└── scripts/         # the same content as demo_*() functions
```

## Evaluation

Tutorials for evaluating RAG pipelines and LLM outputs using local models — see [`evaluation/`](evaluation/).

### RAGAS

Evaluate retrieval-augmented generation with the [RAGAS](https://github.com/explodinggradients/ragas) framework. Covers faithfulness, context precision/recall, factual correctness, semantic similarity, and custom metrics.

- Notebook: [`evaluation/ragas/notebooks/ragas-evaluation-tutorial.ipynb`](evaluation/ragas/notebooks/ragas-evaluation-tutorial.ipynb)
- Script: [`evaluation/ragas/scripts/ragas_evaluation.py`](evaluation/ragas/scripts/ragas_evaluation.py)

```bash
uv sync --group ragas
uv run python evaluation/ragas/scripts/ragas_evaluation.py
```

### DeepEval

Evaluate LLM outputs with [DeepEval](https://github.com/confident-ai/deepeval). Covers faithfulness, answer relevancy, contextual precision/recall/relevancy, hallucination, bias, toxicity, G-Eval custom criteria, and batch evaluation.

- Notebook: [`evaluation/deepeval/notebooks/deepeval-evaluation-tutorial.ipynb`](evaluation/deepeval/notebooks/deepeval-evaluation-tutorial.ipynb)
- Script: [`evaluation/deepeval/scripts/deepeval_evaluation.py`](evaluation/deepeval/scripts/deepeval_evaluation.py)

```bash
uv sync --group deepeval
uv run python evaluation/deepeval/scripts/deepeval_evaluation.py
```

## Inference

Building with LLMs and optimising how they run — see [`inference/`](inference/).

### LangChain

Build LLM applications with [LangChain](https://python.langchain.com/) and local Ollama models. Covers chat models, prompt templates, output parsers, LCEL chains, RAG pipelines, conversation memory, and agents with tools.

- Notebook: [`inference/langchain/notebooks/langchain-tutorial.ipynb`](inference/langchain/notebooks/langchain-tutorial.ipynb)
- Script: [`inference/langchain/scripts/langchain_tutorial.py`](inference/langchain/scripts/langchain_tutorial.py)

```bash
uv sync --group langchain
uv run python inference/langchain/scripts/langchain_tutorial.py
```

### LangGraph — Long-Term Agentic Memory

Give an agent memory that survives the conversation, with [LangGraph](https://langchain-ai.github.io/langgraph/) and [LangMem](https://langchain-ai.github.io/langmem/). An email assistant that triages, drafts replies, and rewrites its own instructions from your feedback.

The through-line: "memory" is three separate mechanisms — **semantic** (facts it looks up), **episodic** (past decisions replayed as few-shot examples), and **procedural** (instructions it follows and rewrites). Only the third one actually changes behaviour.

- Notebook: [`inference/langgraph-memory/notebooks/langgraph-memory-tutorial.ipynb`](inference/langgraph-memory/notebooks/langgraph-memory-tutorial.ipynb)
- Script: [`inference/langgraph-memory/scripts/langgraph_memory_tutorial.py`](inference/langgraph-memory/scripts/langgraph_memory_tutorial.py)

```bash
uv sync --group langgraph-memory
uv run python inference/langgraph-memory/scripts/langgraph_memory_tutorial.py
```

> Embeddings come from sentence-transformers, not Ollama — the semantic store needs an embedding endpoint and Ollama only serves one when started with `--embeddings`.

### vLLM

High-throughput LLM inference with [vLLM](https://docs.vllm.ai/). Covers offline inference, sampling parameters, chat completion, batch inference, OpenAI-compatible API server, streaming, and structured output (guided decoding).

- Notebook: [`inference/vllm/notebooks/vllm-tutorial.ipynb`](inference/vllm/notebooks/vllm-tutorial.ipynb)
- Script: [`inference/vllm/scripts/vllm_tutorial.py`](inference/vllm/scripts/vllm_tutorial.py)

```bash
uv pip install vllm      # Linux + NVIDIA only, so it is not in uv.lock
uv run python inference/vllm/scripts/vllm_tutorial.py
```

> Requires an NVIDIA GPU (~16 GB VRAM for 7B in float16). vLLM loads models directly from HuggingFace — no Ollama needed.

### Pydantic

Validate LLM output with [Pydantic](https://docs.pydantic.dev/). Covers models and field constraints, validation errors, schema-guided generation via `response_format`, the `.parse()` helper, retry loops driven by validation feedback, nested models, and custom validators.

The through-line: schema-guided decoding constrains the **shape** of the output, Pydantic judges its **meaning**, and the two disagree more than you would expect — enums and array bounds are enforced during decoding, numeric ranges are not, and `min_length` on a string is "enforced" by padding the answer with junk.

- Notebook: [`inference/pydantic/notebooks/pydantic-tutorial.ipynb`](inference/pydantic/notebooks/pydantic-tutorial.ipynb)
- Script: [`inference/pydantic/scripts/pydantic_tutorial.py`](inference/pydantic/scripts/pydantic_tutorial.py)
- Reference: [`inference/pydantic/pydantic-reference.md`](inference/pydantic/pydantic-reference.md)

```bash
uv sync                  # base dependencies are all this tutorial needs
uv run python inference/pydantic/scripts/pydantic_tutorial.py
```

## Adding a tutorial

1. Create `<area>/<tutorial>/` with `notebooks/` and `scripts/` inside it.
2. Write the pair: `<topic>-tutorial.ipynb` and `<topic>_tutorial.py`. Keep them structurally identical — notebook `## N. Section Title` headings map one-to-one onto script `demo_section_title()` functions, and `main()` calls them in the same order.
3. Add a `README.md` in the tutorial folder: what it covers, how to run it, and anything that will surprise the next person.
4. Add its dependencies as a new group in `pyproject.toml`, then `uv lock`.
5. Link it from the area README and from the list above.

## Prerequisites

- [Ollama](https://ollama.com/) running locally with the model pulled: `ollama pull qwen2.5:7b`
- [uv](https://docs.astral.sh/uv/) for dependency management: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Python 3.12+ (uv will fetch one if needed)

> The tutorials reference the model as `qwen2.5:7b`. If you already have the
> equivalent `qwen2.5:7b-instruct` pulled, alias it instead of downloading again:
> `ollama cp qwen2.5:7b-instruct qwen2.5:7b`