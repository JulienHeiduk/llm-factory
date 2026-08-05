# Long-Term Agentic Memory — LangGraph + LangMem

An email assistant that gets better the more you correct it, built with [LangGraph](https://langchain-ai.github.io/langgraph/) and [LangMem](https://langchain-ai.github.io/langmem/) on a local Qwen2.5-7B.

## Why long-term memory?

**The problem it addresses.** A stateless agent makes the same mistake forever. You tell it to sign emails as "John Doe", it complies for the rest of the conversation, and tomorrow it has forgotten. Stuffing corrections into the system prompt by hand does not scale past a handful, does not survive multiple users, and gives you no way to tell which correction is responsible for which behaviour.

**What it gives you.** A store the agent reads at every turn and writes to as it works — and, crucially, the recognition that "memory" is **three different mechanisms** that people routinely conflate:

| Kind | Holds | Written by | Makes the agent |
|---|---|---|---|
| **Semantic** | facts — who Alice is, what she owns | the agent, via tools | *informed* |
| **Episodic** | past decisions, replayed as few-shot examples | your corrections | *consistent* |
| **Procedural** | the instructions it follows | an optimizer, from your feedback | **able to change** |

Only the third one changes behaviour. An agent with a vector store full of facts and no procedural memory looks like it remembers and still repeats every mistake you have ever corrected.

**Use it when**

- The assistant is long-lived and personal — the same user, over weeks.
- Corrections arrive as feedback ("stop doing that") rather than as code changes.
- You need per-user behaviour from one shared agent definition.
- Classification quality should improve from examples rather than from prompt edits.

**Skip it when**

- The task is one-shot. A stateless call with a good prompt is simpler and cheaper.
- Conversation-scoped memory is enough — that is checkpointing, not long-term memory, and LangGraph gives it to you with a checkpointer alone.
- You cannot audit what gets stored. Self-rewriting prompts are genuinely hard to debug: behaviour changes with no diff in your repository.

**In production.** Swap `InMemoryStore` for a persistent store on day one — the interface is identical, so nothing else changes. Version stored prompts and keep the previous value: an optimizer that rewrites a rule badly is a production incident with no rollback unless you planned one. Namespace strictly by user, since a leak here is a data-protection problem rather than a bug. And gate procedural updates behind review before letting an agent rewrite its own instructions unattended.

## Alternatives

| Instead of | Consider | Why |
|---|---|---|
| Long-term memory | A LangGraph **checkpointer** | If you only need state within a thread, this is the whole answer |
| LangMem | [mem0](https://mem0.ai/) | Dedicated memory layer with its own extraction and consolidation |
| LangMem | [Zep](https://www.getzep.com/) | Temporal knowledge graph over conversation history |
| A memory framework | A table plus `WHERE user_id = ?` | Underrated — most "semantic memory" is a lookup you can write yourself |
| Self-rewriting prompts | [DSPy](https://dspy.ai/) | Optimises prompts against a metric offline, where you can inspect the result |

## Contents

| Path | What it is |
|---|---|
| [`notebooks/langgraph-memory-tutorial.ipynb`](notebooks/langgraph-memory-tutorial.ipynb) | The tutorial, 7 sections, runs top-to-bottom |
| [`scripts/langgraph_memory_tutorial.py`](scripts/langgraph_memory_tutorial.py) | Same content as `demo_*()` functions, `main()` runs them in order |

## How to run

```bash
ollama pull qwen2.5:7b               # once
uv sync --group langgraph-memory

uv run python inference/langgraph-memory/scripts/langgraph_memory_tutorial.py
uv run jupyter lab                   # or the notebook
```

The first run downloads the sentence-transformers embedding model (~90 MB). A full pass takes a few minutes — most of it is the agent making real tool calls.

## Sections

1. **Three Kinds of Memory** — the store, the namespaces, what lives where
2. **Semantic Memory** — LangMem's `manage_memory` / `search_memory` tools
3. **Episodic Memory** — past triage decisions retrieved semantically as few-shot examples
4. **The Triage Router** — structured output forces one of three labels; rules come from memory
5. **The Email Agent** — five tools behind `create_react_agent`, prompt rebuilt from the store per invocation
6. **Procedural Memory** — the optimizer rewrites stored instructions from one line of feedback
7. **Watching Behaviour Change** — feedback aimed at triage, and what actually happens

## What a local 7B can and cannot do here

Every row was observed, not assumed:

| Component | Verdict on qwen2.5:7b |
|---|---|
| Triage with `with_structured_output` | Works well |
| ReAct agent driving 5 tools | Works well |
| Semantic + episodic retrieval | Works well — embeddings are local and deterministic |
| **Prompt optimizer** | **Unreliable** |

The optimizer has to decide *which* stored prompt a piece of feedback belongs to, and that is where a small model falls over. Running one identical feedback three times routed it to `triage-respond`, then `triage-ignore`, then `main_agent` — **despite `temperature=0`**, because the optimizer makes several internal calls and only one has to wobble. Roughly one run in three it fails differently: LangMem reads `result["responses"][0].which`, the model emits no structured choice, and it raises `IndexError` rather than degrading. The tutorial catches that and reports it.

The fix is a one-line change — `create_multi_prompt_optimizer(llm, ...)` takes any chat model, so point that at something stronger while the agent stays local.

## Notes

- **Embeddings are sentence-transformers, not Ollama.** The semantic store needs an embedding endpoint; Ollama only serves one when started with `--embeddings`, and returns `501` otherwise. Loading `all-MiniLM-L6-v2` in-process keeps this runnable on a default install. (The same constraint affects the RAG section of the [LangChain tutorial](../langchain/README.md).)
- **`clean_prompt` is load-bearing.** LangMem wraps the prompt being optimised in `<current_prompt>` tags, and a 7B occasionally copies the opening tag into its rewrite — which then gets stored and fed back, compounding each turn.
- **Adapted from** DeepLearning.AI's *Long-Term Agentic Memory with LangGraph*, lesson 5. Changes: `gpt-4o` / `gpt-4o-mini` / `claude-3-5-sonnet` → local `qwen2.5:7b`; OpenAI embeddings → sentence-transformers; the missing `prompts` module inlined; the "implement the remaining stores" exercise completed so triage feedback is not silently discarded; course-platform boilerplate, `dotenv`, and 35 empty trailing cells removed.
