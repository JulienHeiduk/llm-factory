# LangChain — Building LLM Applications

Building LLM applications with [LangChain](https://python.langchain.com/) against a local Qwen2.5-7B model served by Ollama.

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
