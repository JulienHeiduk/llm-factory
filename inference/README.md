# Inference

Building with LLMs and optimising how they run. All tutorials use a **local** Qwen2.5-7B via Ollama — except vLLM, which loads from HuggingFace and needs an NVIDIA GPU.

## Tutorials

| Folder | What it covers | Install |
|---|---|---|
| [`langchain/`](langchain/) | Chat models, prompts, output parsers, LCEL chains, RAG, memory, agents | `uv sync --group langchain` |
| [`pydantic/`](pydantic/) | Validating LLM output: field constraints, schema-guided generation, retry loops, nested models | `uv sync` |
| [`vllm/`](vllm/) | Offline inference, sampling, batching, the API server, streaming, guided decoding | `uv pip install vllm` (GPU box) |

Each folder has its own `README.md` with the section list, how to run it, and the gotchas specific to that framework.

## Layout

```
inference/<tutorial>/
├── README.md        # what it covers + how to run
├── notebooks/       # the tutorial, runs top-to-bottom
└── scripts/         # the same content as demo_*() functions
```

`pydantic/` also carries a `pydantic-reference.md` API cheatsheet — a tutorial folder can hold extra docs alongside its two artifact directories.

See the [root README](../README.md#adding-a-tutorial) for how to add one.
