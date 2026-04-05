# llm-factory

LLM experimentation workspace: evaluation, deployment, quantization, and more.

All experiments use **Qwen2.5-7B** served locally with [Ollama](https://ollama.com/) — no API key needed, fully offline.

## Structure

```
llm-factory/
├── evaluation/           # Benchmarks, metrics, model comparisons
│   ├── notebooks/
│   └── scripts/
├── deployment/           # Serving, APIs, containerization
│   ├── notebooks/
│   └── scripts/
├── quantization/         # Model compression and quantization experiments
│   ├── notebooks/
│   └── scripts/
├── fine-tuning/          # Fine-tuning scripts and configs
│   ├── notebooks/
│   └── scripts/
├── inference/            # Inference optimization and benchmarks
│   ├── notebooks/
│   └── scripts/
├── data/
│   ├── raw/              # Raw datasets
│   └── processed/        # Cleaned/prepared datasets
└── configs/              # Model and experiment configurations
```

## Evaluation

Tutorials for evaluating RAG pipelines and LLM outputs using local models.

### RAGAS

Evaluate retrieval-augmented generation with the [RAGAS](https://github.com/explodinggradients/ragas) framework. Covers faithfulness, context precision/recall, factual correctness, semantic similarity, and custom metrics.

- Notebook: [`evaluation/notebooks/ragas-evaluation-tutorial.ipynb`](evaluation/notebooks/ragas-evaluation-tutorial.ipynb)
- Script: [`evaluation/scripts/ragas_evaluation.py`](evaluation/scripts/ragas_evaluation.py)

```bash
pip install ragas openai sentence-transformers
python evaluation/scripts/ragas_evaluation.py
```

### DeepEval

Evaluate LLM outputs with [DeepEval](https://github.com/confident-ai/deepeval). Covers faithfulness, answer relevancy, contextual precision/recall/relevancy, hallucination, bias, toxicity, G-Eval custom criteria, and batch evaluation.

- Notebook: [`evaluation/notebooks/deepeval-evaluation-tutorial.ipynb`](evaluation/notebooks/deepeval-evaluation-tutorial.ipynb)
- Script: [`evaluation/scripts/deepeval_evaluation.py`](evaluation/scripts/deepeval_evaluation.py)

```bash
pip install deepeval openai
python evaluation/scripts/deepeval_evaluation.py
```

## Inference

### LangChain

Build LLM applications with [LangChain](https://python.langchain.com/) and local Ollama models. Covers chat models, prompt templates, output parsers, LCEL chains, RAG pipelines, conversation memory, and agents with tools.

- Notebook: [`inference/notebooks/langchain-tutorial.ipynb`](inference/notebooks/langchain-tutorial.ipynb)
- Script: [`inference/scripts/langchain_tutorial.py`](inference/scripts/langchain_tutorial.py)

```bash
pip install langchain langchain-ollama langchain-community chromadb langgraph
python inference/scripts/langchain_tutorial.py
```

### vLLM

High-throughput LLM inference with [vLLM](https://docs.vllm.ai/). Covers offline inference, sampling parameters, chat completion, batch inference, OpenAI-compatible API server, streaming, and structured output (guided decoding).

- Notebook: [`inference/notebooks/vllm-tutorial.ipynb`](inference/notebooks/vllm-tutorial.ipynb)
- Script: [`inference/scripts/vllm_tutorial.py`](inference/scripts/vllm_tutorial.py)

```bash
pip install vllm openai
python inference/scripts/vllm_tutorial.py
```

> Requires an NVIDIA GPU (~16 GB VRAM for 7B in float16). vLLM loads models directly from HuggingFace — no Ollama needed.

## Prerequisites

```bash
# Install Ollama and pull the model
ollama pull qwen2.5:7b
```