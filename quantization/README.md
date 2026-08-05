# Quantization

Model compression and quantization experiments. **No tutorials here yet.**

Planned ground: GGUF conversion and `ollama create` from a Modelfile, GPTQ / AWQ quantization, bitsandbytes 8-bit and 4-bit loading, and measuring the quality/latency/VRAM trade-off each one buys.

## Layout to follow

```
quantization/<tutorial>/
├── README.md        # what it covers + how to run
├── notebooks/       # the tutorial, runs top-to-bottom
└── scripts/         # the same content as demo_*() functions
```

Add the tutorial's dependencies as a new group in [`pyproject.toml`](../pyproject.toml), then list it here and in the [root README](../README.md).
