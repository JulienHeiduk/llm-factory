"""
vLLM Tutorial (using Qwen2.5-7B)

Requirements: pip install vllm openai
Model: Qwen/Qwen2.5-7B-Instruct (downloaded from HuggingFace on first run)
Hardware: NVIDIA GPU with ~16 GB VRAM

Sections 1-4 and 7 use offline inference (no server needed).
Sections 5-6 require a running vLLM server:
    vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
"""

import json
import time

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

MODEL = "Qwen/Qwen2.5-7B-Instruct"

llm = LLM(model=MODEL, trust_remote_code=True)


# ---------------------------------------------------------------------------
# 1. Offline Inference
# ---------------------------------------------------------------------------


def demo_offline_inference():
    print("=" * 60)
    print("1. OFFLINE INFERENCE")
    print("=" * 60)

    outputs = llm.generate(
        ["The capital of France is"], SamplingParams(max_tokens=30)
    )
    for output in outputs:
        print(f"Prompt:    {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
    print()


# ---------------------------------------------------------------------------
# 2. Sampling Parameters
# ---------------------------------------------------------------------------


def demo_sampling_params():
    print("=" * 60)
    print("2. SAMPLING PARAMETERS")
    print("=" * 60)

    prompt = "Write a one-sentence summary of machine learning."

    # Greedy decoding
    result = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=100))
    print(f"Greedy (temp=0):\n  {result[0].outputs[0].text}\n")

    # Creative
    result = llm.generate(
        [prompt], SamplingParams(temperature=1.2, top_p=0.95, max_tokens=100)
    )
    print(f"Creative (temp=1.2):\n  {result[0].outputs[0].text}\n")

    # Nucleus sampling
    result = llm.generate(
        [prompt],
        SamplingParams(temperature=0.7, top_p=0.9, top_k=50, max_tokens=100),
    )
    print(f"Nucleus (top_p=0.9, top_k=50):\n  {result[0].outputs[0].text}\n")

    # Multiple sequences
    result = llm.generate(
        ["Explain gravity in one sentence."],
        SamplingParams(temperature=0.8, max_tokens=60, n=3),
    )
    print(f"Generated {len(result[0].outputs)} sequences:")
    for i, out in enumerate(result[0].outputs):
        print(f"  [{i + 1}] {out.text.strip()}")

    # Stop sequences
    params = SamplingParams(
        temperature=0,
        max_tokens=200,
        stop=["\n\n", "3."],
        repetition_penalty=1.1,
    )
    result = llm.generate(["List 5 programming languages:\n1."], params)
    print(f"\nStopped output: 1.{result[0].outputs[0].text}")
    print(f"Stop reason: {result[0].outputs[0].stop_reason}\n")


# ---------------------------------------------------------------------------
# 3. Chat Completion
# ---------------------------------------------------------------------------


def demo_chat_completion():
    print("=" * 60)
    print("3. CHAT COMPLETION")
    print("=" * 60)

    # Single turn
    messages = [
        {
            "role": "system",
            "content": "You are a concise Python tutor. Answer in 2-3 sentences max.",
        },
        {"role": "user", "content": "What is a decorator?"},
    ]
    result = llm.chat([messages], SamplingParams(temperature=0, max_tokens=150))
    print(f"Single turn:\n  {result[0].outputs[0].text}\n")

    # Multi-turn conversation
    conversation = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is the GIL in Python?"},
        {
            "role": "assistant",
            "content": (
                "The GIL (Global Interpreter Lock) is a mutex in CPython that allows "
                "only one thread to execute Python bytecode at a time, limiting true "
                "parallelism for CPU-bound tasks."
            ),
        },
        {"role": "user", "content": "How can I work around it?"},
    ]
    result = llm.chat(
        [conversation], SamplingParams(temperature=0, max_tokens=200)
    )
    print(f"Multi-turn:\n  {result[0].outputs[0].text}\n")


# ---------------------------------------------------------------------------
# 4. Batch Inference
# ---------------------------------------------------------------------------


def demo_batch_inference():
    print("=" * 60)
    print("4. BATCH INFERENCE")
    print("=" * 60)

    questions = [
        [{"role": "user", "content": "What is a neural network? One sentence."}],
        [{"role": "user", "content": "What is gradient descent? One sentence."}],
        [{"role": "user", "content": "What is backpropagation? One sentence."}],
        [
            {
                "role": "user",
                "content": "What is an activation function? One sentence.",
            }
        ],
        [{"role": "user", "content": "What is a loss function? One sentence."}],
        [{"role": "user", "content": "What is regularization? One sentence."}],
        [{"role": "user", "content": "What is dropout? One sentence."}],
        [
            {
                "role": "user",
                "content": "What is batch normalization? One sentence.",
            }
        ],
    ]

    params = SamplingParams(temperature=0, max_tokens=80)

    start = time.perf_counter()
    results = llm.chat(questions, params)
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(r.outputs[0].token_ids) for r in results)
    print(f"Batch of {len(questions)} prompts in {elapsed:.2f}s")
    print(f"Total output tokens: {total_tokens}")
    print(f"Throughput: {total_tokens / elapsed:.1f} tokens/s\n")

    for q, r in zip(questions, results):
        print(f"Q: {q[0]['content']}")
        print(f"A: {r.outputs[0].text.strip()}\n")


# ---------------------------------------------------------------------------
# 5. OpenAI-Compatible API Server
# ---------------------------------------------------------------------------


def demo_api_server():
    """Requires: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000"""
    print("=" * 60)
    print("5. OPENAI-COMPATIBLE API SERVER")
    print("=" * 60)

    try:
        from openai import OpenAI
    except ImportError:
        print("Install openai: pip install openai\n")
        return

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

    try:
        models = client.models.list()
    except Exception as e:
        print(f"Server not reachable ({e}). Start it with:")
        print(f"  vllm serve {MODEL} --host 0.0.0.0 --port 8000\n")
        return

    print("Available models:")
    for m in models.data:
        print(f"  - {m.id}")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is vLLM and why is it fast?"},
        ],
        temperature=0,
        max_tokens=200,
    )
    print(f"\n{response.choices[0].message.content}")
    print(
        f"\nUsage: {response.usage.prompt_tokens} prompt + "
        f"{response.usage.completion_tokens} completion tokens\n"
    )


# ---------------------------------------------------------------------------
# 6. Streaming
# ---------------------------------------------------------------------------


def demo_streaming():
    """Requires: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000"""
    print("=" * 60)
    print("6. STREAMING")
    print("=" * 60)

    try:
        from openai import OpenAI
    except ImportError:
        print("Install openai: pip install openai\n")
        return

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Explain PagedAttention in 3 sentences.",
                },
            ],
            temperature=0,
            max_tokens=200,
            stream=True,
        )
    except Exception as e:
        print(f"Server not reachable ({e}). Start it with:")
        print(f"  vllm serve {MODEL} --host 0.0.0.0 --port 8000\n")
        return

    print("Streaming response:")
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print("\n")


# ---------------------------------------------------------------------------
# 7. Structured Output (Guided Decoding)
# ---------------------------------------------------------------------------


def demo_structured_output():
    print("=" * 60)
    print("7. STRUCTURED OUTPUT (GUIDED DECODING)")
    print("=" * 60)

    # JSON schema guided generation
    movie_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "year": {"type": "integer"},
            "genre": {"type": "string"},
            "rating": {"type": "number", "minimum": 0, "maximum": 10},
        },
        "required": ["title", "year", "genre", "rating"],
    }

    params = SamplingParams(
        temperature=0,
        max_tokens=200,
        guided_decoding=GuidedDecodingParams(json=movie_schema),
    )
    messages = [
        {
            "role": "user",
            "content": "Give me info about the movie Inception as JSON.",
        },
    ]
    result = llm.chat([messages], params)
    raw = result[0].outputs[0].text
    parsed = json.loads(raw)
    print(f"JSON-guided: {parsed}")

    # Regex guided generation (date format)
    params = SamplingParams(
        temperature=0,
        max_tokens=20,
        guided_decoding=GuidedDecodingParams(regex=r"\d{4}-\d{2}-\d{2}"),
    )
    messages = [
        {
            "role": "user",
            "content": "When was Python 3.0 released? Answer with just the date.",
        },
    ]
    result = llm.chat([messages], params)
    print(f"Regex-guided: {result[0].outputs[0].text}")

    # Choice guided generation (sentiment)
    params = SamplingParams(
        temperature=0,
        max_tokens=10,
        guided_decoding=GuidedDecodingParams(
            choice=["positive", "negative", "neutral"]
        ),
    )
    messages = [
        {"role": "system", "content": "Classify the sentiment of the text."},
        {
            "role": "user",
            "content": "I absolutely loved this movie, it was fantastic!",
        },
    ]
    result = llm.chat([messages], params)
    print(f"Choice-guided sentiment: {result[0].outputs[0].text}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    demo_offline_inference()
    demo_sampling_params()
    demo_chat_completion()
    demo_batch_inference()
    demo_api_server()
    demo_streaming()
    demo_structured_output()
    print("All demos completed!")


if __name__ == "__main__":
    main()
