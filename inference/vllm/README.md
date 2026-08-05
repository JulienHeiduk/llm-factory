# vLLM — High-Throughput Inference

High-throughput LLM inference with [vLLM](https://docs.vllm.ai/). **The one tutorial in this repo that does not use Ollama** — vLLM loads `Qwen/Qwen2.5-7B-Instruct` straight from HuggingFace.

> Requires an NVIDIA GPU (~16 GB VRAM for 7B in float16). It will not run on macOS.

## Why vLLM?

**The problem it addresses.** Serving a model with a naive `model.generate()` loop wastes most of your GPU. Each request reserves a contiguous KV-cache block sized for the *longest* output it might produce, so memory fragments and sits idle; requests are batched only if they arrive together and finish together, so one long generation stalls everything behind it. The result is a GPU at low utilisation that still cannot serve many concurrent users.

**What it gives you.** Two ideas do most of the work. **PagedAttention** stores the KV cache in fixed-size non-contiguous pages — the same trick an OS uses for virtual memory — which removes fragmentation and lets many sequences share memory. **Continuous batching** admits new requests into the running batch as others finish, instead of waiting for the whole batch to complete. Together they typically buy an order of magnitude more throughput than a naive loop, and you get an OpenAI-compatible server for free, so existing client code points at it unchanged.

**Use it when**

- You are self-hosting an open model and serving real concurrent traffic.
- Throughput per GPU is what your bill depends on.
- You need to run inference over a large dataset offline, as fast as the hardware allows.
- You want to drop in behind code that already speaks the OpenAI API.

**Skip it when**

- You are one developer on a laptop. Ollama or llama.cpp start in seconds, run on CPU or Apple Silicon, and are why the rest of this repo uses Ollama.
- You have no NVIDIA GPU. (There is CPU and non-CUDA support, but that is not where vLLM earns its keep.)
- Your traffic is low and bursty — a hosted API is often cheaper than a GPU you keep warm.

**In production.** Tune `--gpu-memory-utilization` and `--max-model-len` together; the defaults reserve more KV cache than a short-context workload needs. Turn on prefix caching when many requests share a long system prompt — it is close to free latency. Reach for quantization (AWQ, GPTQ, FP8) before reaching for a bigger GPU, and use tensor parallelism to shard a model that genuinely will not fit. Scrape the `/metrics` endpoint: queue time and preemption count tell you when you are out of KV cache long before users do.

## Alternatives

| Instead of | Consider | Why |
|---|---|---|
| vLLM for local dev | [Ollama](https://ollama.com/), [llama.cpp](https://github.com/ggml-org/llama.cpp) | Trivial setup, CPU/Metal, GGUF quantization — far lower throughput |
| vLLM for serving | [SGLang](https://docs.sglang.ai/) | Comparable or better throughput; RadixAttention shines on shared prefixes |
| vLLM for serving | [TGI](https://huggingface.co/docs/text-generation-inference) | HuggingFace-native serving with a similar feature set |
| Maximum speed at any cost | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Fastest on NVIDIA, by a wide margin in engineering effort too |
| Running it yourself | A hosted API | No GPU, no ops — you trade cost per token for cost per engineer |

## Contents

| Path | What it is |
|---|---|
| [`notebooks/vllm-tutorial.ipynb`](notebooks/vllm-tutorial.ipynb) | The tutorial, 7 sections (no stored outputs — needs a GPU) |
| [`scripts/vllm_tutorial.py`](scripts/vllm_tutorial.py) | Same content as `demo_*()` functions, `main()` runs them in order |

## How to run

vLLM is **not in `uv.lock`** — it is Linux + NVIDIA only and would drag a CUDA dependency tree into the lockfile for every other machine. Install it directly on the GPU box:

```bash
uv pip install vllm
uv run python inference/vllm/scripts/vllm_tutorial.py
```

Sections 5 and 6 talk to a server over the OpenAI-compatible API, so start one in a second terminal first:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
```

Without it, those two sections print a hint and return rather than failing.

## Sections

1. **Offline Inference** — `LLM.generate()`, no server involved
2. **Sampling Parameters** — greedy vs creative, nucleus sampling, `n`, stop sequences
3. **Chat Completion** — `LLM.chat()`, single and multi-turn
4. **Batch Inference** — throughput measurement across a batch of prompts
5. **OpenAI-Compatible API Server** — requires `vllm serve`
6. **Streaming** — requires `vllm serve`
7. **Structured Output** — guided decoding by JSON schema, regex, and choice

## Notes

- Model load happens at **import time** (`llm = LLM(model=MODEL, ...)` at module level), so even running a single `demo_*()` pulls the weights onto the GPU.
- Section 7's guided decoding is the vLLM counterpart to the [Pydantic tutorial](../pydantic/README.md) — same idea, different engine.
