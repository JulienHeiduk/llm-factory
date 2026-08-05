# Pydantic — Validating LLM Output

Using [Pydantic](https://docs.pydantic.dev/) as the contract between your code and a local Qwen2.5-7B model. The through-line: schema-guided decoding constrains the **shape** of the output, Pydantic judges its **meaning**, and the two disagree more than you would expect.

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
