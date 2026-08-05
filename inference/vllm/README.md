# vLLM — High-Throughput Inference

High-throughput LLM inference with [vLLM](https://docs.vllm.ai/). **The one tutorial in this repo that does not use Ollama** — vLLM loads `Qwen/Qwen2.5-7B-Instruct` straight from HuggingFace.

> Requires an NVIDIA GPU (~16 GB VRAM for 7B in float16). It will not run on macOS.

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
