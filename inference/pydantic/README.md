# Pydantic — Validating LLM Output

Using [Pydantic](https://docs.pydantic.dev/) as the contract between your code and a local Qwen2.5-7B model. The through-line: schema-guided decoding constrains the **shape** of the output, Pydantic judges its **meaning**, and the two disagree more than you would expect.

## Why Pydantic?

**The problem it addresses.** An LLM returns text. Your program needs data — with types, ranges, and invariants. The gap between those two is where LLM applications break: `json.loads()` succeeds on `{"rating": 93}` and your code happily writes a 93 into a column that promised 0–10. Hand-rolled checking (`isinstance`, `if 0 <= x <= 10`, a pile of `KeyError` handling) is verbose, easy to get wrong, and impossible to hand to a model as a specification.

**What it gives you.** Declare the shape once as a class and get four things from it: parsing, type coercion, constraint validation with precise per-field errors, and a **JSON Schema** you can hand straight to the model as `response_format`. That last point is what makes it more than a validation library here — the same declaration both instructs the model and audits its answer.

**Use it when**

- Anywhere model output feeds code. This is close to always.
- You want structured output — the schema comes free from the model definition.
- You need error messages good enough to feed back to the model for a retry (see section 6).
- You are defining tool/function-calling signatures, or FastAPI request/response bodies.

**Skip it when**

- The output is genuinely free text a human reads. Do not validate a poem.
- A plain `dataclass` or `TypedDict` covers it — you want structure but no runtime checking or coercion.
- You are tempted to express *quality* as a constraint. Validation catches malformed data, not wrong data (see the table below), and it is not a substitute for evaluation — that is what [RAGAS](../../evaluation/ragas/README.md) and [DeepEval](../../evaluation/deepeval/README.md) are for.

**In production.** Pydantic v2's core is compiled Rust, so validation is rarely your bottleneck. Prefer `model_validate_json()` over `json.loads()` + `model_validate()` — it parses and validates in a single pass. Put semantic rules in `field_validator`/`model_validator` rather than pushing them into the schema, for the reason section 5 demonstrates. And spend effort on `Field(description=...)`: it is the one part of your model the LLM actually reads.

## Alternatives

| Instead of | Consider | Why |
|---|---|---|
| Pydantic for plain structs | `dataclasses`, [attrs](https://www.attrs.org/) | Structure without runtime validation — lighter when you trust the input |
| Runtime validation | `TypedDict` + mypy | Static checking only; no cost at runtime, no protection from an LLM |
| Writing the retry loop | [Instructor](https://python.useinstructor.com/) | Wraps the client with Pydantic + automatic reask-on-failure (built on Pydantic) |
| Validating after generation | [Outlines](https://dottxt-ai.github.io/outlines/), [Guidance](https://github.com/guidance-ai/guidance), [XGrammar](https://github.com/mlc-ai/xgrammar) | Constrain decoding at the token level — complementary, with the same caveat that shape is not meaning |
| Type-driven schemas | [jsonschema](https://python-jsonschema.readthedocs.io/) | Validate raw JSON Schema directly when your schema is not Python-native |
| Maximum throughput | [msgspec](https://jcristharif.com/msgspec/) | Faster still, much smaller ecosystem |

## Contents

| Path | What it is |
|---|---|
| [`notebooks/pydantic-tutorial.ipynb`](notebooks/pydantic-tutorial.ipynb) | The tutorial, 7 sections, runs top-to-bottom |
| [`scripts/pydantic_tutorial.py`](scripts/pydantic_tutorial.py) | Same content as `demo_*()` functions, `main()` runs them in order |
| [`pydantic-reference.md`](pydantic-reference.md) | API cheatsheet — models, fields, errors, the three structured-output modes, validators |

## How to run

```bash
ollama pull qwen2.5:7b     # once
uv sync                    # base deps (openai + pydantic[email]) are all this needs

uv run python inference/pydantic/scripts/pydantic_tutorial.py    # whole script
uv run jupyter lab                                               # or the notebook
```

Run a single section instead of all seven:

```bash
uv run python -c "from inference.pydantic.scripts.pydantic_tutorial import demo_nested_models; demo_nested_models()"
```

## Sections

1. **Pydantic Basics** — `BaseModel`, `Field` constraints, type coercion, extra keys dropped silently
2. **Validation Errors** — `exc.errors()` structure, reporting every failure at once
3. **Schema-Guided Generation** — `model_json_schema()` → `response_format`
4. **The `.parse()` Helper** — typed object back; raises `ValidationError` at the call site
5. **What the Grammar Does and Doesn't Enforce** — the point of the tutorial (see below)
6. **Retry with Validation Feedback** — feed `exc.errors()` back to the model
7. **Nested Models & Business Rules** — `$defs`, `field_validator`, `model_validator`

## The finding in section 5

Each row was checked against Qwen2.5-7B on Ollama, not assumed:

| Constraint | Enforced while decoding? | What happens |
|---|---|---|
| `Literal[...]` → `enum` | ✅ | Coerced into the allowed set — confidently wrong, undetectable downstream |
| list `max_length` → `maxItems` | ✅ | Respected cleanly |
| `ge` / `le` → `minimum` / `maximum` | ❌ | Freely violated — ask "0 to 100" against `le=10` and you get `93` |
| str `min_length` → `minLength` | ⚠️ | Padded to quota with junk, which then **passes** validation |

So: let the grammar fix the shape, let Pydantic judge the meaning, and keep length/range wishes out of the decoder.
