# Deployment

Serving, APIs, and containerization. **No tutorials here yet.**

Planned ground: wrapping a model behind FastAPI, containerizing an Ollama or vLLM server, batching and concurrency under load, and health/latency monitoring.

Note that the [vLLM tutorial](../inference/vllm/README.md) already covers running an OpenAI-compatible server; deployment tutorials should pick up where that stops — packaging, scaling, and operating it.

## Layout to follow

```
deployment/<tutorial>/
├── README.md        # what it covers + how to run
├── notebooks/       # the tutorial, runs top-to-bottom
└── scripts/         # the same content as demo_*() functions
```

Add the tutorial's dependencies as a new group in [`pyproject.toml`](../pyproject.toml), then list it here and in the [root README](../README.md).
