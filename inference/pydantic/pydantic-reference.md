# Pydantic Reference — LLM Output Validation

Companion reference for [`inference/notebooks/pydantic-tutorial.ipynb`](notebooks/pydantic-tutorial.ipynb) and [`inference/scripts/pydantic_tutorial.py`](scripts/pydantic_tutorial.py). The tutorial teaches the ideas in order; this file is the lookup table you keep open while writing your own models.

Targets **Pydantic v2** against **Qwen2.5-7B on local Ollama**. Everything here was checked against that stack — where the behaviour differs from the hosted-OpenAI docs, the difference is called out.

## Contents

1. [Setup](#setup)
2. [Models and fields](#models-and-fields)
3. [Validation errors](#validation-errors)
4. [Three ways to get structured output](#three-ways-to-get-structured-output)
5. [Which constraints the decoder actually enforces](#which-constraints-the-decoder-actually-enforces)
6. [Validators](#validators)
7. [Nested models](#nested-models)
8. [Retry on validation failure](#retry-on-validation-failure)
9. [Quick reference](#quick-reference)

---

## Setup

```bash
uv sync                 # base deps include pydantic[email] — EmailStr needs that extra
ollama pull qwen2.5:7b
```

See the [README](../../README.md#setup) for the full uv workflow.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",  # local Ollama server
    api_key="ollama",                      # required by the client, ignored by Ollama
)
```

No API key and no network egress: the `openai` package is used purely as an HTTP client for the local server.

---

## Models and fields

A model is a class inheriting `BaseModel`. Annotated attributes become validated fields; `Field()` attaches constraints and documentation. Validation runs on construction.

```python
from pydantic import BaseModel, EmailStr, Field
from typing import Literal, Optional

class UserInput(BaseModel):
    name: str                                             # required
    email: EmailStr                                       # required, format-checked
    age: int = Field(ge=0, le=120)                        # required, bounded
    priority: Literal["low", "medium", "high"]            # required, fixed set
    order_id: Optional[int] = Field(None, ge=10000, le=99999)   # optional
    tags: list[str] = Field(default_factory=list, max_length=4)  # optional list
```

**Required vs optional** is decided by the presence of a default, not by `Optional[...]`:

| Declaration | Required? | Accepts `None`? |
|---|---|---|
| `x: int` | yes | no |
| `x: Optional[int]` | **yes** | yes |
| `x: Optional[int] = None` | no | yes |
| `x: int = Field(..., ge=0)` | yes (`...` = required) | no |

Use `default_factory=list` for mutable defaults, never `= []`.

### Key methods

| Method | Purpose |
|---|---|
| `Model(**data)` | Build and validate from a dict |
| `Model.model_validate(obj)` | Same, from an existing object/dict |
| `Model.model_validate_json(s)` | Parse **and** validate a JSON string in one step |
| `instance.model_dump()` | → `dict` |
| `instance.model_dump_json(indent=2)` | → JSON string |
| `Model.model_json_schema()` | → JSON Schema dict (this is what you hand the LLM) |

### Type coercion

Values that unambiguously represent the target type are converted, which absorbs harmless LLM sloppiness:

```python
UserInput(age="42")            # -> 42          (int)
UserInput(purchase_date="2026-01-31")  # -> datetime.date(2026, 1, 31)
```

Coercion is not unlimited — `age="not a number"` raises. For strict behaviour, use `model_config = ConfigDict(strict=True)`.

### Extra keys are dropped, not rejected

The v2 default is `extra="ignore"`. A model that invents `{"confidence": 0.97}` still validates, and the field vanishes silently. To catch it:

```python
from pydantic import ConfigDict

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")
```

---

## Validation errors

`ValidationError.errors()` returns one dict per problem — and reports **all** problems, not just the first.

```python
from pydantic import ValidationError

try:
    UserInput(name="Joe", email="not-an-email")
except ValidationError as exc:
    for err in exc.errors():
        print(err["loc"], err["msg"], err["type"])
```

```
('email',)  value is not a valid email address: An email address must have an @-sign.  value_error
('query',)  Field required                                                             missing
```

| Key | Meaning |
|---|---|
| `loc` | Field path as a tuple. Nested: `('items', 0, 'quantity')`. **Empty for `model_validator` errors.** |
| `msg` | Human-readable reason |
| `type` | Stable identifier — `missing`, `value_error`, `less_than_equal`, `string_too_short`, … |

> **v1 → v2:** the old dotted types (`value_error.email`, `value_error.missing`) are gone. Match on v2 names such as `value_error` and `missing`.

Flattening helper used throughout the tutorial — note the empty-`loc` case, which is easy to miss until a `model_validator` fires:

```python
def describe_errors(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        location = ".".join(str(p) for p in err["loc"]) or "(whole model)"
        parts.append(f"{location}: {err['msg']}")
    return "; ".join(parts)
```

---

## Three ways to get structured output

All three work against Ollama. They differ in how much the decoder is constrained.

### 1. `json_schema` response format — schema-constrained, raw string back

Most explicit, and the one to use when you want to inspect what the model produced before validating it.

```python
response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": MyModel.__name__,
            "schema": MyModel.model_json_schema(),
            "strict": True,
        },
    },
)
raw = response.choices[0].message.content     # str, unvalidated
obj = MyModel.model_validate_json(raw)        # your call when to validate
```

> Hosted OpenAI additionally requires `additionalProperties: false` on every object when `strict: true`, which `model_json_schema()` does not emit. Ollama does not care. Add it if you point the same code at api.openai.com.

### 2. `client.chat.completions.parse()` — schema + validation in one step

```python
completion = client.chat.completions.parse(
    model="qwen2.5:7b",
    messages=messages,
    response_format=MyModel,          # the class itself
)
obj = completion.choices[0].message.parsed    # already a MyModel
```

Validation happens **inside** the call, so a constraint violation raises `ValidationError` at the call site — there is no partial response to inspect. Always wrap it in `try/except`.

> On older SDKs this was `client.beta.chat.completions.parse()`. The `beta` path still works; the non-beta one is current.

### 3. `json_object` mode — JSON guaranteed, shape not

The decoder guarantees only that the output parses as JSON. Put the schema in the prompt and validate afterwards. Useful for models or servers without schema support.

```python
prompt = f"...\n\nReturn JSON matching this schema:\n{json.dumps(MyModel.model_json_schema())}"
response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
)
obj = MyModel.model_validate_json(response.choices[0].message.content)
```

If a model wraps JSON in markdown fences, strip them before validating — see `_extract_json()` in [`evaluation/scripts/deepeval_evaluation.py`](../../evaluation/deepeval/scripts/deepeval_evaluation.py), which does exactly this for a local model.

---

## Which constraints the decoder actually enforces

Your constraints all reach the JSON Schema. The decoder honours only some of them, and where it does, *enforced* is not the same as *correct*. Observed with Qwen2.5-7B on Ollama:

| Pydantic | JSON Schema | Enforced while decoding? | What actually happens |
|---|---|---|---|
| `Literal[...]` | `enum` | ✅ | Coerced into the allowed set. Asked for a value outside it, the model returns an allowed one — **confidently wrong, and undetectable downstream** |
| `max_length` on a list | `maxItems` | ✅ | Respected cleanly |
| `min_length` on a list | `minItems` | ✅ | Respected cleanly |
| `ge` / `le` / `gt` / `lt` | `minimum` / `maximum` | ❌ | Freely violated — ask for "0 to 100" against `le=10` and you get `93`. **This is what Pydantic catches** |
| `min_length` on a `str` | `minLength` | ⚠️ | Enforced *destructively*: the decoder cannot close the string early, so a short answer is padded to the quota with junk — which then passes validation |
| `max_length` on a `str` | `maxLength` | ✅ | Truncates |

Two practical consequences:

1. **Numeric ranges need a validation pass.** This is the gap that makes Pydantic-after-generation non-redundant.
2. **Do not push semantic wishes into the grammar.** `min_length=200` on a summary does not buy thoroughness, it buys 200 characters of *something*. Express intent in the prompt; use constraints to reject bad output, not to manufacture good output.

---

## Validators

For rules a schema cannot express.

```python
from pydantic import field_validator, model_validator

class SupportTicket(BaseModel):
    priority: Literal["low", "medium", "high"]
    is_complaint: bool
    tags: list[str]

    @field_validator("tags")
    @classmethod
    def normalise_tags(cls, tags: list[str]) -> list[str]:
        """Runs per field, after type coercion. Return the cleaned value."""
        return [tag.strip().lower().replace(" ", "_") for tag in tags]

    @model_validator(mode="after")
    def complaints_are_never_low_priority(self):
        """Runs once with the whole object, so it can compare fields."""
        if self.is_complaint and self.priority == "low":
            raise ValueError("a complaint cannot be priority 'low'")
        return self
```

| Hook | Runs | Sees | Must return |
|---|---|---|---|
| `@field_validator("x")` + `@classmethod` | Per field, after coercion | That field's value | The (possibly transformed) value |
| `@field_validator("x", mode="before")` | Per field, before coercion | The raw input | The value to coerce |
| `@model_validator(mode="after")` | Once, after all fields | `self` | `self` |

Raise `ValueError` inside a validator; Pydantic wraps it into the `ValidationError`. Errors from `model_validator` carry an **empty `loc`**.

---

## Nested models

A field typed as another `BaseModel` becomes a `$ref` + `$defs` in the schema. Ollama handles these correctly, so one call can extract a whole object graph:

```python
class LineItem(BaseModel):
    product: str
    quantity: int = Field(ge=1)
    unit_price: float = Field(ge=0)

class SupportTicket(BaseModel):
    customer_email: EmailStr
    items: list[LineItem]

ticket = SupportTicket.model_validate_json(raw)
total = sum(i.quantity * i.unit_price for i in ticket.items)   # ordinary Python objects
```

---

## Retry on validation failure

Because ranges survive decoding unchecked, the practical loop is generate → validate → feed the errors back:

```python
messages = [{"role": "user", "content": task}]

for attempt in range(1, 4):
    raw = structured_call(Model, messages, temperature=0.0 if attempt == 1 else 0.4)
    try:
        obj = Model.model_validate_json(raw)
        break
    except ValidationError as exc:
        reasons = describe_errors(exc)
        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content":
             f"That response failed validation: {reasons}. "
             "Fix only those fields and return corrected JSON."},
        ]
```

**Raise the temperature on retries.** At `temperature=0` the model regenerates the rejected answer verbatim and the loop burns every attempt on the same output — reproducibly so with Qwen2.5-7B.

---

## Quick reference

### Field types

```python
str, int, float, bool          # scalars
date, datetime                 # from datetime; accepts ISO strings
EmailStr                       # requires pydantic[email]
Optional[T]                    # may be None
list[T], dict[str, T]          # collections
Literal["a", "b"]              # fixed set -> enum in the schema
SomeModel                      # nested model -> $ref in the schema
```

### Field constraints

```python
Field(ge=N, le=N, gt=N, lt=N)        # numeric bounds       (NOT decoder-enforced)
Field(min_length=N, max_length=N)    # str/list length      (decoder-enforced)
Field(multiple_of=N)                 # numeric step
Field(pattern=r"^\d{4}$")            # regex on strings
Field(description="...")             # lands in the schema — the model reads it
Field(default_factory=list)          # mutable default
Field(...)                           # explicitly required
```

`description` is worth using generously: it is the one part of your model that reaches the LLM as instruction rather than as a constraint, and it is how you communicate the intent that `ge`/`le` cannot enforce.

### Config

```python
from pydantic import ConfigDict

class M(BaseModel):
    model_config = ConfigDict(
        extra="forbid",       # reject unknown keys (default: "ignore")
        strict=True,          # disable type coercion
        str_strip_whitespace=True,
    )
```

---

## Further reading

- [Pydantic documentation](https://docs.pydantic.dev/)
- [Ollama structured outputs](https://ollama.com/blog/structured-outputs)
- [OpenAI structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — for the hosted-API differences noted above
