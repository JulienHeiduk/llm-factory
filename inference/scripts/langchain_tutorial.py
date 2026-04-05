"""
LangChain Tutorial (using Qwen2.5 locally)

Requirements: pip install langchain langchain-ollama langchain-community chromadb langgraph
LLM: Qwen2.5-7B served locally with Ollama

Pull the model before running:
    ollama pull qwen2.5:7b
"""

import math

from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

llm = ChatOllama(model="qwen2.5:7b", temperature=0)


# ---------------------------------------------------------------------------
# 1. Chat Models
# ---------------------------------------------------------------------------


def demo_chat_models():
    print("=" * 60)
    print("1. CHAT MODELS")
    print("=" * 60)

    # Simple invocation
    response = llm.invoke("What is LangChain in one sentence?")
    print(f"Simple: {response.content}\n")

    # Structured messages
    messages = [
        SystemMessage(
            content="You are a concise Python tutor. Answer in 2-3 sentences max."
        ),
        HumanMessage(content="Explain list comprehensions."),
    ]
    response = llm.invoke(messages)
    print(f"With messages: {response.content}\n")

    # Streaming
    print("Streaming: ", end="")
    for chunk in llm.stream("Name 3 benefits of local LLM inference."):
        print(chunk.content, end="", flush=True)
    print("\n")


# ---------------------------------------------------------------------------
# 2. Prompt Templates
# ---------------------------------------------------------------------------


def demo_prompt_templates():
    print("=" * 60)
    print("2. PROMPT TEMPLATES")
    print("=" * 60)

    # Simple template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert in {domain}. Be concise."),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke(
        {"domain": "machine learning", "question": "What is overfitting?"}
    )
    print(f"Template: {response.content}\n")

    # Few-shot prompting
    few_shot_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Translate English to French."),
            ("human", "Hello"),
            ("ai", "Bonjour"),
            ("human", "How are you?"),
            ("ai", "Comment allez-vous ?"),
            ("human", "{input}"),
        ]
    )
    chain = few_shot_prompt | llm
    response = chain.invoke({"input": "Good evening, my friend."})
    print(f"Few-shot: {response.content}\n")


# ---------------------------------------------------------------------------
# 3. Output Parsers
# ---------------------------------------------------------------------------


def demo_output_parsers():
    print("=" * 60)
    print("3. OUTPUT PARSERS")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert in {domain}. Be concise."),
            ("human", "{question}"),
        ]
    )

    # StrOutputParser
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke(
        {"domain": "databases", "question": "What is an index?"}
    )
    print(f"StrOutputParser (type={type(result).__name__}): {result}\n")

    # JsonOutputParser
    json_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Always respond with valid JSON. "
                "No markdown, no code fences, just raw JSON.",
            ),
            (
                "human",
                "Give me a JSON object with the name, year, and language of the "
                "programming language: {language}",
            ),
        ]
    )
    chain = json_prompt | llm | JsonOutputParser()
    result = chain.invoke({"language": "Python"})
    print(f"JsonOutputParser (type={type(result).__name__}): {result}\n")

    # PydanticOutputParser
    class MovieReview(BaseModel):
        title: str = Field(description="Movie title")
        rating: float = Field(description="Rating from 0 to 10")
        summary: str = Field(description="One-sentence summary")

    parser = PydanticOutputParser(pydantic_object=MovieReview)
    pydantic_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a movie critic. Respond ONLY with the requested JSON format.\n"
                "{format_instructions}",
            ),
            ("human", "Review the movie: {movie}"),
        ]
    )
    chain = pydantic_prompt | llm | parser
    review = chain.invoke(
        {
            "movie": "Inception",
            "format_instructions": parser.get_format_instructions(),
        }
    )
    print(f"PydanticOutputParser (type={type(review).__name__}):")
    print(f"  Title: {review.title}")
    print(f"  Rating: {review.rating}")
    print(f"  Summary: {review.summary}\n")


# ---------------------------------------------------------------------------
# 4. LCEL Chains
# ---------------------------------------------------------------------------


def demo_lcel_chains():
    print("=" * 60)
    print("4. LCEL CHAINS")
    print("=" * 60)

    # Simple chain
    summarize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize the following text in one sentence."),
            ("human", "{text}"),
        ]
    )
    summarize_chain = summarize_prompt | llm | StrOutputParser()
    result = summarize_chain.invoke(
        {
            "text": (
                "LangChain is a framework for developing applications powered by "
                "large language models. It provides tools to connect LLMs with "
                "external data sources and enables building complex reasoning chains."
            )
        }
    )
    print(f"Summarize: {result}\n")

    # RunnableParallel
    pros_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "List 3 pros of the given technology. Be concise."),
            ("human", "{tech}"),
        ]
    )
    cons_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "List 3 cons of the given technology. Be concise."),
            ("human", "{tech}"),
        ]
    )
    pros_chain = pros_prompt | llm | StrOutputParser()
    cons_chain = cons_prompt | llm | StrOutputParser()
    parallel_chain = RunnableParallel(pros=pros_chain, cons=cons_chain)
    result = parallel_chain.invoke({"tech": "Kubernetes"})
    print("=== Pros ===")
    print(result["pros"])
    print("\n=== Cons ===")
    print(result["cons"])

    # Multi-step chain
    code_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Write a short Python function. Code only, no explanations."),
            ("human", "{task}"),
        ]
    )
    explain_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Explain the following code in plain English. 2-3 sentences.",
            ),
            ("human", "{code}"),
        ]
    )
    code_chain = code_prompt | llm | StrOutputParser()
    full_chain = (
        {"code": code_chain} | explain_prompt | llm | StrOutputParser()
    )
    result = full_chain.invoke({"task": "Fibonacci sequence generator"})
    print(f"\nMulti-step (code -> explain): {result}\n")


# ---------------------------------------------------------------------------
# 5. RAG Pipeline
# ---------------------------------------------------------------------------


def demo_rag_pipeline():
    print("=" * 60)
    print("5. RAG PIPELINE")
    print("=" * 60)

    documents = [
        Document(
            page_content=(
                "Ollama is a tool for running large language models locally. "
                "It supports models like Llama, Qwen, Mistral, and Gemma. "
                "Ollama provides an OpenAI-compatible API at localhost:11434. "
                "Models are downloaded and managed automatically."
            ),
            metadata={"source": "ollama-docs"},
        ),
        Document(
            page_content=(
                "Qwen2.5 is a series of large language models developed by Alibaba. "
                "The 7B variant offers a good balance of performance and efficiency. "
                "It supports multiple languages and excels at reasoning tasks. "
                "Qwen2.5 uses a transformer architecture with grouped-query attention."
            ),
            metadata={"source": "qwen-docs"},
        ),
        Document(
            page_content=(
                "LangChain is a framework for building applications with LLMs. "
                "It provides abstractions for prompts, chains, memory, and agents. "
                "LCEL (LangChain Expression Language) uses the pipe operator for composition. "
                "LangChain integrates with vector stores, document loaders, and tools."
            ),
            metadata={"source": "langchain-docs"},
        ),
        Document(
            page_content=(
                "ChromaDB is an open-source vector database for AI applications. "
                "It stores embeddings and supports similarity search. "
                "Chroma can run in-memory or persist to disk. "
                "It integrates natively with LangChain and other AI frameworks."
            ),
            metadata={"source": "chroma-docs"},
        ),
    ]

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    # Embed and store
    embeddings = OllamaEmbeddings(model="qwen2.5:7b")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Test retrieval
    results = retriever.invoke("What models does Ollama support?")
    print(f"Retrieved {len(results)} chunks:")
    for doc in results:
        print(f"  [{doc.metadata['source']}] {doc.page_content[:80]}...")

    # RAG chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question based only on the following context. "
                "If the context doesn't contain the answer, say so.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    for question in [
        "What architecture does Qwen2.5 use?",
        "How does ChromaDB store data?",
    ]:
        answer = rag_chain.invoke(question)
        print(f"\nQ: {question}")
        print(f"A: {answer}")

    # Cleanup
    vectorstore.delete_collection()
    print()


# ---------------------------------------------------------------------------
# 6. Conversation Memory
# ---------------------------------------------------------------------------


def demo_conversation_memory():
    print("=" * 60)
    print("6. CONVERSATION MEMORY")
    print("=" * 60)

    memory_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Be concise."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = memory_prompt | llm | StrOutputParser()

    session_store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = InMemoryChatMessageHistory()
        return session_store[session_id]

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "demo-session"}}

    r1 = chain_with_history.invoke(
        {"input": "My name is Alice. I work on NLP."}, config=config
    )
    print(f"Turn 1: {r1}")

    r2 = chain_with_history.invoke(
        {"input": "What's my name and what do I work on?"}, config=config
    )
    print(f"Turn 2: {r2}")

    print(
        f"History length: {len(session_store['demo-session'].messages)} messages\n"
    )


# ---------------------------------------------------------------------------
# 7. Agents & Tools
# ---------------------------------------------------------------------------


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number."""
    return math.sqrt(x)


def demo_agents():
    print("=" * 60)
    print("7. AGENTS & TOOLS")
    print("=" * 60)

    tools = [multiply, add, square_root]
    print("Available tools:")
    for t in tools:
        print(f"  - {t.name}: {t.description}")

    agent_llm = ChatOllama(model="qwen2.5:7b", temperature=0)
    agent = create_react_agent(agent_llm, tools)

    for question in [
        "What is the square root of 144 plus 8 times 3?",
        "Multiply 15 by 7, then add the square root of 49 to the result.",
    ]:
        print(f"\nQ: {question}")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        for msg in result["messages"]:
            role = msg.__class__.__name__
            if hasattr(msg, "content") and msg.content:
                print(f"  [{role}] {msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  [{role}] -> tool call: {tc['name']}({tc['args']})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    demo_chat_models()
    demo_prompt_templates()
    demo_output_parsers()
    demo_lcel_chains()
    demo_rag_pipeline()
    demo_conversation_memory()
    demo_agents()
    print("All demos completed!")


if __name__ == "__main__":
    main()
