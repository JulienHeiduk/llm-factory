"""
Pydantic Tutorial — Validating LLM Output (using Qwen2.5 locally)

Requirements: uv sync
LLM: Qwen2.5-7B served locally with Ollama

Pull the model before running:
    ollama pull qwen2.5:7b

Structured output is two independent guarantees, and this tutorial keeps them
apart on purpose:

  1. The decoder constrains the *shape* of the JSON (types, required keys,
     enum members, array bounds). Ollama does this from the JSON Schema that
     Pydantic generates.
  2. Pydantic validates the *meaning* (numeric ranges, cross-field rules).

Section 5 shows exactly where the two disagree — that gap is why you still
validate output that a grammar already "guaranteed".
"""

import json

from openai import OpenAI
from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

MODEL = "qwen2.5:7b"

client = OpenAI(
    base_url="http://localhost:11434/v1",  # local Ollama server
    api_key="ollama",                      # required by the client, ignored by Ollama
)


def structured_call(schema_model, messages, temperature=0.0):
    """Ask the LLM for JSON constrained by a Pydantic model's JSON Schema.

    Returns the raw JSON string — deliberately unvalidated, so the caller can
    decide what to do with output that is well-formed but wrong.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_model.__name__,
                "schema": schema_model.model_json_schema(),
                "strict": True,
            },
        },
    )
    return response.choices[0].message.content


def describe_errors(exc: ValidationError) -> str:
    """Flatten a ValidationError into one line per offending field.

    Model-level errors (from a `model_validator`) carry an empty `loc`, so they
    are labelled explicitly rather than printed as a bare colon.
    """
    parts = []
    for err in exc.errors():
        location = ".".join(str(p) for p in err["loc"]) or "(whole model)"
        parts.append(f"{location}: {err['msg']}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Shared models
# ---------------------------------------------------------------------------


class MovieRecommendation(BaseModel):
    """The contract we want every movie recommendation to satisfy."""

    title: str = Field(min_length=1, max_length=100)
    genre: Literal[
        "action", "comedy", "drama", "sci-fi", "thriller", "horror", "romance"
    ]
    year: int = Field(ge=1900, le=2026)
    rating: float = Field(ge=0.0, le=10.0, description="Rating from 0 to 10")
    synopsis: str = Field(min_length=10, max_length=500)
    director: Optional[str] = None
    recommended_for: Optional[Literal["family", "adults", "teens"]] = None


# ---------------------------------------------------------------------------
# 1. Pydantic Basics
# ---------------------------------------------------------------------------


def demo_pydantic_basics():
    print("=" * 60)
    print("1. PYDANTIC BASICS")
    print("=" * 60)

    class UserInput(BaseModel):
        name: str
        email: EmailStr
        query: str
        order_id: Optional[int] = Field(None, ge=10000, le=99999)

    user = UserInput(
        name="Joe User",
        email="joe.user@example.com",
        query="I forgot my password.",
    )
    print(f"Instance:  {user}")
    print(f"As JSON:   {user.model_dump_json()}")
    print(f"order_id defaulted to: {user.order_id}\n")

    # Type coercion: strings that *look* like the target type are converted.
    coerced = UserInput(
        name="Joe User",
        email="joe.user@example.com",
        query="I need help.",
        order_id="12345",  # a string, not an int
    )
    print(f"Coerced '12345' -> {coerced.order_id!r} ({type(coerced.order_id).__name__})")

    # Extra keys are dropped by default rather than raising. This matters for
    # LLM output: a chatty model can add fields and validation still passes.
    extra = UserInput(
        name="Joe User",
        email="joe.user@example.com",
        query="I need help.",
        confidence=0.97,          # not on the model
        note="thinking out loud",  # not on the model
    )
    print(f"Extra keys dropped: {extra.model_dump_json()}\n")


# ---------------------------------------------------------------------------
# 2. Validation Errors
# ---------------------------------------------------------------------------


def demo_validation_errors():
    print("=" * 60)
    print("2. VALIDATION ERRORS")
    print("=" * 60)

    class UserInput(BaseModel):
        name: str
        email: EmailStr
        query: str

    bad_input = {"name": "Joe User", "email": "not-an-email"}  # query missing too

    try:
        UserInput(**bad_input)
    except ValidationError as exc:
        print(f"{len(exc.errors())} problem(s) found:")
        for err in exc.errors():
            field = ".".join(str(p) for p in err["loc"])
            print(f"  - {field}: {err['msg']}  (type={err['type']})")
        print(f"\nFlattened for a retry prompt:\n  {describe_errors(exc)}\n")


# ---------------------------------------------------------------------------
# 3. Schema-Guided Generation
# ---------------------------------------------------------------------------


def demo_schema_guided_generation():
    print("=" * 60)
    print("3. SCHEMA-GUIDED GENERATION")
    print("=" * 60)

    schema = MovieRecommendation.model_json_schema()
    print("JSON Schema Pydantic hands to the decoder (truncated):")
    print(json.dumps(schema, indent=2)[:400] + " ...\n")

    raw = structured_call(
        MovieRecommendation,
        [
            {"role": "system", "content": "You are a movie recommendation expert."},
            {"role": "user", "content": "Recommend a sci-fi movie for a relaxing weekend."},
        ],
    )
    print(f"Raw model output:\n{raw}\n")

    movie = MovieRecommendation.model_validate_json(raw)
    print(f"Validated -> {movie.title} ({movie.year}), {movie.rating}/10\n")


# ---------------------------------------------------------------------------
# 4. The .parse() Helper
# ---------------------------------------------------------------------------


def demo_parse_helper():
    print("=" * 60)
    print("4. THE .parse() HELPER")
    print("=" * 60)

    # .parse() sends the schema and validates the reply in one step, handing
    # back a typed object instead of a string.
    completion = client.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a movie recommendation expert."},
            {"role": "user", "content": "Suggest a comedy suitable for family viewing."},
        ],
        response_format=MovieRecommendation,
        temperature=0,
    )
    movie = completion.choices[0].message.parsed
    print(f"Type: {type(movie).__name__}")
    print(f"{movie.title} ({movie.year}) — {movie.genre}, {movie.rating}/10")
    print(f"Recommended for: {movie.recommended_for}\n")

    # The convenience has a sharp edge: validation happens *inside* the call,
    # so a constraint violation surfaces here as a ValidationError, not as a
    # return value you can inspect. Wrap it.
    try:
        completion = client.chat.completions.parse(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Rate the movie Inception on a scale from 0 to 100.",
                }
            ],
            response_format=MovieRecommendation,
            temperature=0,
        )
        print(f"Parsed: {completion.choices[0].message.parsed}\n")
    except ValidationError as exc:
        print(f"ValidationError raised by .parse(): {describe_errors(exc)}")
        print("The rating came back on a 0-100 scale; the model is capped at 10.\n")


# ---------------------------------------------------------------------------
# 5. What the Grammar Does and Doesn't Enforce
# ---------------------------------------------------------------------------


def demo_constraint_enforcement():
    print("=" * 60)
    print("5. WHAT THE GRAMMAR DOES AND DOESN'T ENFORCE")
    print("=" * 60)

    # (a) enum members ARE enforced — but enforcement is not correctness. Asked
    #     for a genre outside the Literal, the decoder picks an allowed one and
    #     answers confidently wrong rather than failing.
    class Genre(BaseModel):
        genre: Literal["action", "comedy", "drama"]

    raw = structured_call(
        Genre,
        [{"role": "user", "content": "The genre of 'Planet Earth' is documentary. Return it."}],
    )
    print(f"(a) enum, asked for 'documentary'   -> {raw.strip()}")
    print("    forced into the allowed set: confidently wrong, still 'valid'\n")

    # (b) array bounds ARE enforced.
    class Tags(BaseModel):
        tags: list[str] = Field(min_length=1, max_length=3)

    raw = structured_call(
        Tags, [{"role": "user", "content": "Give exactly 8 tags describing the ocean."}]
    )
    print(f"(b) maxItems=3, asked for 8 tags    -> {raw.strip()}\n")

    # (c) numeric ranges are NOT enforced. The schema says maximum=10 and the
    #     decoder happily emits 93 — this is the gap Pydantic exists to close.
    class Score(BaseModel):
        title: str
        score: float = Field(ge=0.0, le=10.0)

    raw = structured_call(
        Score,
        [{"role": "user", "content": "Rate the movie Inception from 0 to 100."}],
        temperature=0.9,
    )
    print(f"(c) maximum=10, asked for 0-100     -> {raw.strip()}")
    try:
        Score.model_validate_json(raw)
        print("    pydantic: accepted (the model happened to comply)\n")
    except ValidationError as exc:
        print(f"    pydantic REJECTS: {describe_errors(exc)}\n")

    # (d) string length IS enforced — destructively. The decoder cannot close
    #     the string before minLength, so it pads with whatever comes next.
    #     The result satisfies the schema *and* passes Pydantic, which is the
    #     strongest argument in this notebook against pushing semantic wishes
    #     into the grammar.
    class Summary(BaseModel):
        summary: str = Field(min_length=200, max_length=500)

    raw = structured_call(
        Summary, [{"role": "user", "content": "Summarize the ocean in exactly three words."}]
    )
    text = json.loads(raw)["summary"]
    print(f"(d) minLength=200, asked for 3 words -> {len(text)} chars")
    print(f"    {text[:160]}...")
    print("    padded to satisfy the grammar — and Pydantic accepts it\n")

    print("Rule of thumb: let the grammar fix the shape, let Pydantic judge the")
    print("meaning, and keep length/range wishes out of the decoder.\n")


# ---------------------------------------------------------------------------
# 6. Retry with Validation Feedback
# ---------------------------------------------------------------------------


def demo_retry_with_feedback():
    print("=" * 60)
    print("6. RETRY WITH VALIDATION FEEDBACK")
    print("=" * 60)

    class Score(BaseModel):
        title: str
        score: float = Field(ge=0.0, le=10.0, description="Rating from 0 to 10")

    messages = [
        {"role": "user", "content": "Rate the movie Inception from 0 to 100."}
    ]

    for attempt in range(1, 4):
        # Retries nudge the temperature up: at temperature=0 a rejected answer
        # is regenerated verbatim and the loop spins forever.
        raw = structured_call(Score, messages, temperature=0.0 if attempt == 1 else 0.4)
        print(f"Attempt {attempt}: {raw.strip()}")

        try:
            score = Score.model_validate_json(raw)
            print(f"  accepted -> {score.title}: {score.score}/10\n")
            break
        except ValidationError as exc:
            reasons = describe_errors(exc)
            print(f"  rejected -> {reasons}")
            messages += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"That response failed validation: {reasons}. "
                        "Fix only those fields and return corrected JSON."
                    ),
                },
            ]
    else:
        print("  gave up after 3 attempts\n")


# ---------------------------------------------------------------------------
# 7. Nested Models & Business Rules
# ---------------------------------------------------------------------------


class LineItem(BaseModel):
    product: str
    quantity: int = Field(ge=1)
    unit_price: float = Field(ge=0)


class SupportTicket(BaseModel):
    """Nested extraction target, with rules a schema alone cannot express."""

    customer_email: EmailStr
    category: Literal["refund_request", "information_request", "complaint", "other"]
    priority: Literal["low", "medium", "high"]
    is_complaint: bool
    items: list[LineItem]
    tags: list[str] = Field(max_length=4)
    order_id: Optional[int] = Field(None, ge=10000, le=99999)

    @field_validator("tags")
    @classmethod
    def normalise_tags(cls, tags: list[str]) -> list[str]:
        """Field validators clean values; they run per field, after coercion."""
        return [tag.strip().lower().replace(" ", "_") for tag in tags]

    @model_validator(mode="after")
    def complaints_are_never_low_priority(self):
        """Model validators see the whole object, so they can compare fields."""
        if self.is_complaint and self.priority == "low":
            raise ValueError("a complaint cannot be priority 'low'")
        return self


def demo_nested_models():
    print("=" * 60)
    print("7. NESTED MODELS & BUSINESS RULES")
    print("=" * 60)

    email = (
        "From: joe.user@example.com\n"
        "I ordered 2 mechanical keyboards at 89.99 each and 1 monitor stand at "
        "34.50 on order 12345. The monitor stand arrived cracked and I was "
        "charged twice. I want a refund immediately, this is unacceptable."
    )

    raw = structured_call(
        SupportTicket,
        [
            {"role": "system", "content": "Extract a structured support ticket from the email."},
            {"role": "user", "content": email},
        ],
    )
    print(f"Raw output:\n{raw}\n")

    ticket = SupportTicket.model_validate_json(raw)
    print(f"Category: {ticket.category} | priority: {ticket.priority}")
    print(f"Order:    {ticket.order_id} | tags: {ticket.tags}")
    for item in ticket.items:
        print(f"  {item.quantity} x {item.product} @ {item.unit_price}")
    total = sum(item.quantity * item.unit_price for item in ticket.items)
    print(f"Order total: {total:.2f}\n")

    # The cross-field rule fires on data the JSON Schema considers perfect.
    try:
        SupportTicket(
            customer_email="joe.user@example.com",
            category="complaint",
            priority="low",
            is_complaint=True,
            items=[LineItem(product="monitor stand", quantity=1, unit_price=34.5)],
            tags=["Broken Item"],
        )
    except ValidationError as exc:
        print(f"Business rule caught it: {describe_errors(exc)}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    demo_pydantic_basics()
    demo_validation_errors()
    demo_schema_guided_generation()
    demo_parse_helper()
    demo_constraint_enforcement()
    demo_retry_with_feedback()
    demo_nested_models()
    print("All demos completed!")


if __name__ == "__main__":
    main()
