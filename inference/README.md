# Inference

Building with LLMs and optimising how they run. All tutorials use a **local** Qwen2.5-7B via Ollama — except vLLM, which loads from HuggingFace and needs an NVIDIA GPU.

## These three are layers, not competitors

A common mistake is to treat this folder as a menu. It is a stack — a serious application uses all three at once, at different levels:

| Layer | Tutorial | Question it answers |
|---|---|---|
| **Serving** | [`vllm/`](vllm/) | How do I run the model fast enough, on hardware I control? |
| **Orchestration** | [`langchain/`](langchain/) | How do I wire prompts, retrieval, memory and tools into a pipeline? |
| **Data contract** | [`pydantic/`](pydantic/) | How do I turn the model's text into typed data I can trust? |

Read them in the order that matches your problem. If output keeps arriving in the wrong shape, start with **Pydantic** — it is the smallest and pays off immediately, whether or not you ever adopt a framework. If you are assembling a RAG or agent pipeline, **LangChain**. If a working prototype is too slow or too expensive per GPU, **vLLM**.

The bottom two also work fine without the middle: the Pydantic tutorial calls the model through the plain `openai` client and no framework at all, which is exactly how a lot of production code should look.

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
