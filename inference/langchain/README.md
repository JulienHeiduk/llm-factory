# LangChain — Building LLM Applications

Building LLM applications with [LangChain](https://python.langchain.com/) against a local Qwen2.5-7B model served by Ollama.

## Why LangChain?

**The problem it addresses.** An LLM call is one HTTP request; an LLM *application* is everything around it — templating prompts, parsing replies into typed data, chunking and embedding documents, retrieving them, threading conversation history, letting the model call tools, and retrying when any of that fails. Written by hand that is a few hundred lines of glue per project, rewritten every time you change provider or vector store.

**What it gives you.** One interface — the `Runnable`, composed with the `|` operator — over models, prompts, parsers, retrievers and tools. Because every piece speaks the same protocol, swapping `ChatOllama` for `ChatOpenAI`, or Chroma for Qdrant, is a one-line change instead of a rewrite, and every chain inherits streaming, async and batching for free.

**Use it when**

- You are exploring — trying five retrieval strategies or three providers this week.
- The provider is genuinely undecided, or you must support several at once.
- You want RAG, memory, or tool-calling agents without writing the plumbing.
- The integration you need already exists (there are hundreds).

**Skip it when**

- You have one provider and a handful of prompts. The provider SDK plus three functions is less code — and every layer you skip is a layer you never have to debug.
- You are latency- or cost-sensitive and want to see exactly what goes over the wire.
- The abstraction costs more than it saves. Stack traces through generic `Runnable` machinery are a real tax when something misbehaves.

**In production.** Pin versions hard: LangChain moves fast and relocates interfaces between majors — this repo pins the 0.3 series precisely because 1.x moved both `Chroma` and `create_react_agent` out from under this tutorial. Prefer LCEL (`prompt | llm | parser`) over the legacy `Chain` classes, and build agents on LangGraph, whose explicit state machine is far easier to reason about than the old `AgentExecutor`. Add tracing early — an agent looping silently is invisible without it.

## Alternatives

| Instead of | Consider | Why |
|---|---|---|
| The framework at all | The provider SDK directly | One provider, narrow scope — usually less total code |
| LangChain for RAG | [LlamaIndex](https://www.llamaindex.ai/) | Indexing- and retrieval-first; better defaults for document QA |
| LangChain for pipelines | [Haystack](https://haystack.deepset.ai/) | Explicit pipeline graph, strong deployment story |
| LangChain agents | [LangGraph](https://langchain-ai.github.io/langgraph/) alone | You want the state machine, not the rest of the framework |
| Hand-tuned prompts | [DSPy](https://dspy.ai/) | Optimises prompts programmatically against a metric |
| Untyped chains | [Pydantic AI](https://ai.pydantic.dev/) | Type-first agents — pairs with the [Pydantic tutorial](../pydantic/README.md) |

## Contents

| Path | What it is |
|---|---|
| [`notebooks/langchain-tutorial.ipynb`](notebooks/langchain-tutorial.ipynb) | The tutorial, 7 sections, runs top-to-bottom |
| [`scripts/langchain_tutorial.py`](scripts/langchain_tutorial.py) | Same content as `demo_*()` functions, `main()` runs them in order |

## How to run

```bash
ollama pull qwen2.5:7b            # once
uv sync --group langchain

uv run python inference/langchain/scripts/langchain_tutorial.py
uv run jupyter lab                # or the notebook
```

## Sections

1. **Chat Models** — `ChatOllama`, structured messages, streaming
2. **Prompt Templates** — `ChatPromptTemplate`, few-shot prompting
3. **Output Parsers** — `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser`
4. **LCEL Chains** — pipe composition, `RunnableParallel`, multi-step chains
5. **RAG Pipeline** — text splitting, `OllamaEmbeddings`, Chroma retrieval
6. **Conversation Memory** — `RunnableWithMessageHistory` over an in-memory store
7. **Agents & Tools** — `@tool` functions driven by `create_react_agent`

## Notes

- The RAG section builds an **in-memory** Chroma collection and calls `vectorstore.delete_collection()` at the end, so reruns stay clean.
- Embeddings come from `qwen2.5:7b` via `OllamaEmbeddings` — no embedding-specific model needed.
- The dependency group pins the **LangChain 0.3 series** (`langchain>=0.3,<1.0`, `langchain-community>=0.3,<0.4`, `langgraph<1.0`). LangChain 1.x moves `langchain_community.vectorstores.Chroma` and the `create_react_agent` entry point this tutorial uses, so raising those bounds breaks it.
