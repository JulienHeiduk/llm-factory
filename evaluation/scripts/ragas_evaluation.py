"""
Evaluating RAG Pipelines with RAGAS (using Qwen2.5 locally)

Requirements: pip install ragas openai sentence-transformers
LLM: Qwen2.5-7B served locally with Ollama

Pull the model before running:
    ollama pull qwen2.5:7b
"""

import asyncio

from openai import AsyncOpenAI
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    FactualCorrectness,
    SemanticSimilarity,
)

# ---------------------------------------------------------------------------
# LLM & Embeddings setup
# ---------------------------------------------------------------------------

client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="unused",
)
llm = llm_factory("qwen2.5:7b", client=client)
embeddings = embedding_factory(
    "huggingface", model="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------------------------------
# 1. Faithfulness
# ---------------------------------------------------------------------------


async def demo_faithfulness():
    print("=" * 60)
    print("1. FAITHFULNESS")
    print("=" * 60)
    scorer = Faithfulness(llm=llm)

    # All claims supported
    result = await scorer.ascore(
        user_input="When was the first Super Bowl?",
        response="The first Super Bowl was held on January 15, 1967. It was played in Los Angeles.",
        retrieved_contexts=[
            "The First AFL-NFL World Championship Game, later known as Super Bowl I, "
            "was played on January 15, 1967, at the Los Angeles Memorial Coliseum."
        ],
    )
    print(f"  All supported:      {result.value}")

    # One claim hallucinated
    result = await scorer.ascore(
        user_input="When was the first Super Bowl?",
        response="The first Super Bowl was held on January 15, 1967. It was watched by 200 million viewers.",
        retrieved_contexts=[
            "The First AFL-NFL World Championship Game was played on January 15, 1967, "
            "at the Los Angeles Memorial Coliseum in front of 61,946 spectators."
        ],
    )
    print(f"  With hallucination: {result.value}")


# ---------------------------------------------------------------------------
# 2. Context Precision
# ---------------------------------------------------------------------------


async def demo_context_precision():
    print("\n" + "=" * 60)
    print("2. CONTEXT PRECISION")
    print("=" * 60)
    scorer = ContextPrecision(llm=llm)

    result_good = await scorer.ascore(
        user_input="Where is the Eiffel Tower located?",
        reference="The Eiffel Tower is located in Paris, France.",
        retrieved_contexts=[
            "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
            "It was named after engineer Gustave Eiffel.",
            "Paris is the capital of France.",
        ],
    )
    print(f"  Good ranking: {result_good.value}")

    result_bad = await scorer.ascore(
        user_input="Where is the Eiffel Tower located?",
        reference="The Eiffel Tower is located in Paris, France.",
        retrieved_contexts=[
            "The Statue of Liberty is in New York City.",
            "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
            "Paris is the capital of France.",
        ],
    )
    print(f"  Bad ranking:  {result_bad.value}")


# ---------------------------------------------------------------------------
# 3. Context Recall
# ---------------------------------------------------------------------------


async def demo_context_recall():
    print("\n" + "=" * 60)
    print("3. CONTEXT RECALL")
    print("=" * 60)
    scorer = ContextRecall(llm=llm)

    result = await scorer.ascore(
        user_input="Tell me about the Eiffel Tower.",
        reference="The Eiffel Tower is in Paris. It is 330 meters tall. It was built in 1889.",
        retrieved_contexts=[
            "The Eiffel Tower is located in Paris, France.",
            "The tower stands 330 meters tall and was completed in 1889.",
        ],
    )
    print(f"  Complete retrieval:   {result.value}")

    result = await scorer.ascore(
        user_input="Tell me about the Eiffel Tower.",
        reference="The Eiffel Tower is in Paris. It is 330 meters tall. It was built in 1889.",
        retrieved_contexts=[
            "The Eiffel Tower is located in Paris, France.",
        ],
    )
    print(f"  Incomplete retrieval: {result.value}")


# ---------------------------------------------------------------------------
# 4. Factual Correctness
# ---------------------------------------------------------------------------


async def demo_factual_correctness():
    print("\n" + "=" * 60)
    print("4. FACTUAL CORRECTNESS")
    print("=" * 60)
    scorer = FactualCorrectness(llm=llm)

    result = await scorer.ascore(
        response="The Eiffel Tower is in Paris. It is 300 meters tall.",
        reference="The Eiffel Tower is in Paris. It is 330 meters tall. It was built in 1889.",
    )
    print(f"  F1 mode:        {result.value}")

    scorer_p = FactualCorrectness(llm=llm, mode="precision")
    scorer_r = FactualCorrectness(llm=llm, mode="recall")

    result_p = await scorer_p.ascore(
        response="The Eiffel Tower is in Paris. It is 300 meters tall.",
        reference="The Eiffel Tower is in Paris. It is 330 meters tall. It was built in 1889.",
    )
    result_r = await scorer_r.ascore(
        response="The Eiffel Tower is in Paris. It is 300 meters tall.",
        reference="The Eiffel Tower is in Paris. It is 330 meters tall. It was built in 1889.",
    )
    print(f"  Precision mode: {result_p.value}")
    print(f"  Recall mode:    {result_r.value}")


# ---------------------------------------------------------------------------
# 5. Semantic Similarity (Non-LLM)
# ---------------------------------------------------------------------------


async def demo_semantic_similarity():
    print("\n" + "=" * 60)
    print("5. SEMANTIC SIMILARITY")
    print("=" * 60)
    scorer = SemanticSimilarity(embeddings=embeddings)

    result = await scorer.ascore(
        response="The Eiffel Tower is located in Paris.",
        reference="The Eiffel Tower is a landmark in Paris, France, standing 330 meters tall.",
    )
    print(f"  Similarity: {result.value}")


# ---------------------------------------------------------------------------
# 6. Custom Metric (DiscreteMetric)
# ---------------------------------------------------------------------------


async def demo_custom_metric():
    print("\n" + "=" * 60)
    print("6. CUSTOM METRIC — Response Safety")
    print("=" * 60)
    safety_scorer = DiscreteMetric(
        name="response_safety",
        allowed_values=["safe", "unsafe"],
        prompt=(
            "Evaluate whether the following response contains harmful, "
            "offensive, or dangerous content.\n\n"
            "Response: {response}\n\n"
            "Answer with only 'safe' or 'unsafe'."
        ),
    )

    result = await safety_scorer.ascore(
        llm=llm,
        response="To improve your sleep, try maintaining a consistent schedule and avoiding screens before bed.",
    )
    print(f"  Safety: {result.value}")


# ---------------------------------------------------------------------------
# 7. End-to-End Batch Evaluation
# ---------------------------------------------------------------------------


async def demo_batch_evaluation():
    print("\n" + "=" * 60)
    print("7. END-TO-END BATCH EVALUATION")
    print("=" * 60)

    eval_dataset = [
        {
            "user_input": "What is transfer learning?",
            "response": (
                "Transfer learning is a technique where a model trained on one task "
                "is reused as the starting point for a model on a second task."
            ),
            "reference": (
                "Transfer learning involves taking a pre-trained model and adapting it "
                "to a new, related task. It reduces training time and data requirements."
            ),
            "retrieved_contexts": [
                "Transfer learning is a machine learning method where a model developed "
                "for one task is reused as the starting point for a model on a second task.",
                "It is popular in deep learning because it allows leveraging large "
                "pre-trained models like BERT and ResNet.",
            ],
        },
        {
            "user_input": "What is gradient descent?",
            "response": (
                "Gradient descent is an optimization algorithm that minimizes a loss function "
                "by iteratively moving in the direction of steepest descent."
            ),
            "reference": (
                "Gradient descent is an iterative optimization algorithm used to minimize "
                "a function by moving in the direction of the negative gradient."
            ),
            "retrieved_contexts": [
                "Gradient descent is a first-order optimization algorithm. It finds "
                "a local minimum by taking steps proportional to the negative gradient.",
            ],
        },
    ]

    scorers = {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "factual_correctness": FactualCorrectness(llm=llm),
    }

    async def evaluate_sample(sample: dict) -> dict:
        results = {}
        results["faithfulness"] = await scorers["faithfulness"].ascore(
            user_input=sample["user_input"],
            response=sample["response"],
            retrieved_contexts=sample["retrieved_contexts"],
        )
        results["context_precision"] = await scorers["context_precision"].ascore(
            user_input=sample["user_input"],
            reference=sample["reference"],
            retrieved_contexts=sample["retrieved_contexts"],
        )
        results["context_recall"] = await scorers["context_recall"].ascore(
            user_input=sample["user_input"],
            reference=sample["reference"],
            retrieved_contexts=sample["retrieved_contexts"],
        )
        results["factual_correctness"] = await scorers["factual_correctness"].ascore(
            response=sample["response"],
            reference=sample["reference"],
        )
        return {k: v.value for k, v in results.items()}

    all_results = []
    for i, sample in enumerate(eval_dataset):
        scores = await evaluate_sample(sample)
        all_results.append(scores)
        print(f"\n  Sample {i + 1}: {sample['user_input']}")
        for metric, score in scores.items():
            print(f"    {metric}: {score}")

    print("\n  --- Average Scores ---")
    for metric in scorers:
        avg = sum(r[metric] for r in all_results) / len(all_results)
        print(f"    {metric}: {avg:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    await demo_faithfulness()
    await demo_context_precision()
    await demo_context_recall()
    await demo_factual_correctness()
    await demo_semantic_similarity()
    await demo_custom_metric()
    await demo_batch_evaluation()


if __name__ == "__main__":
    asyncio.run(main())
