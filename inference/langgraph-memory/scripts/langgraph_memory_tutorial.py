"""
Long-Term Agentic Memory with LangGraph (using Qwen2.5 locally)

Requirements: uv sync --group langgraph-memory
LLM: Qwen2.5-7B served locally with Ollama
Embeddings: sentence-transformers/all-MiniLM-L6-v2, loaded in-process

Pull the model before running:
    ollama pull qwen2.5:7b

An email assistant that gets better the more you correct it. The point is the
three kinds of long-term memory an agent needs, and that they are three
different mechanisms rather than one:

  semantic   — facts it looks up   (who is Alice? what did she ask last time?)
  episodic   — examples it imitates (past triage decisions, as few-shot)
  procedural — instructions it follows, and rewrites from your feedback

Everything lives in one LangGraph `InMemoryStore`, namespaced per user, so the
same code swaps to a persistent store without touching the agent.

Note: embeddings come from sentence-transformers rather than Ollama, because
Ollama only serves embeddings when started with `--embeddings`, and this keeps
the tutorial runnable on a default install.
"""

import json

from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from langmem import (
    create_manage_memory_tool,
    create_multi_prompt_optimizer,
    create_search_memory_tool,
)
from pydantic import BaseModel, Field
from typing import Annotated, Literal, TypedDict

# ---------------------------------------------------------------------------
# LLM, embeddings, and the store
# ---------------------------------------------------------------------------

MODEL = "qwen2.5:7b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIMS = 384
USER_ID = "john"

llm = ChatOllama(model=MODEL, temperature=0)

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# The index is what makes `store.search(..., query=...)` a semantic lookup
# rather than a plain listing.
store = InMemoryStore(index={"embed": embeddings, "dims": EMBED_DIMS})

profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

# The seed values for procedural memory. After the first run these live in the
# store, and the agent — not this file — decides what they say.
prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently.",
}

SAMPLE_EMAIL = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": (
        "Hi John,\n\n"
        "I was reviewing the API documentation for the new authentication service "
        "and noticed a few endpoints seem to be missing from the specs. Could you "
        "clarify whether that was intentional?\n\n"
        "Specifically: /auth/refresh and /auth/validate.\n\nThanks!\nAlice"
    ),
}


def clean_prompt(text: str) -> str:
    """Strip scaffolding a small model sometimes echoes into its answer.

    langmem wraps the prompt it is optimising in `<current_prompt>` tags. A 7B
    model will occasionally copy the opening tag into the rewritten prompt,
    which then gets stored and fed back on the next turn. Larger models do not
    do this; cleaning is cheap insurance.
    """
    for tag in ("<current_prompt>", "</current_prompt>"):
        text = text.replace(tag, "")
    return text.strip()


def get_prompt(key: str, default: str) -> str:
    """Read a prompt from procedural memory, seeding it on first access."""
    result = store.get((USER_ID,), key)
    if result is None:
        store.put((USER_ID,), key, {"prompt": default})
        return default
    return result.value["prompt"]


# ---------------------------------------------------------------------------
# 1. Three Kinds of Memory
# ---------------------------------------------------------------------------


def demo_memory_types():
    print("=" * 60)
    print("1. THREE KINDS OF MEMORY")
    print("=" * 60)

    # Procedural memory is seeded from the dict above on first read, and lives
    # in the store from then on.
    for key, default in [
        ("triage_ignore", prompt_instructions["triage_rules"]["ignore"]),
        ("triage_notify", prompt_instructions["triage_rules"]["notify"]),
        ("triage_respond", prompt_instructions["triage_rules"]["respond"]),
        ("agent_instructions", prompt_instructions["agent_instructions"]),
    ]:
        get_prompt(key, default)

    print(f"Namespace ({USER_ID!r},) now holds the agent's procedural memory:")
    for item in store.search((USER_ID,)):
        print(f"  {item.key:20} {item.value['prompt'][:60]}...")

    print("\nThe three memory types, and where each one lives:")
    print(f"  semantic   -> ('email_assistant', {USER_ID!r}, 'collection')  facts, written by tools")
    print(f"  episodic   -> ('email_assistant', {USER_ID!r}, 'examples')    past decisions, few-shot")
    print(f"  procedural -> ({USER_ID!r},)                               instructions, rewritten by an optimizer\n")


# ---------------------------------------------------------------------------
# 2. Semantic Memory
# ---------------------------------------------------------------------------

manage_memory_tool = create_manage_memory_tool(
    namespace=("email_assistant", "{langgraph_user_id}", "collection")
)
search_memory_tool = create_search_memory_tool(
    namespace=("email_assistant", "{langgraph_user_id}", "collection")
)


def demo_semantic_memory():
    print("=" * 60)
    print("2. SEMANTIC MEMORY")
    print("=" * 60)

    print(f"langmem gives the agent two tools: {manage_memory_tool.name}, {search_memory_tool.name}")
    print("The `{langgraph_user_id}` placeholder is filled from config at call time,")
    print("so one agent definition serves every user without leaking memory between them.\n")

    namespace = ("email_assistant", USER_ID, "collection")
    store.put(namespace, "alice", {"content": "Alice Smith owns the authentication service and prefers email over meetings."})
    store.put(namespace, "standup", {"content": "The team standup is 09:15 every weekday and John chairs it."})

    for query in ["who looks after auth?", "when do we meet?"]:
        hits = store.search(namespace, query=query, limit=1)
        print(f"search({query!r}) -> {hits[0].value['content']}")
    print()


# ---------------------------------------------------------------------------
# 3. Episodic Memory
# ---------------------------------------------------------------------------

EXAMPLE_TEMPLATE = """Email Subject: {subject}
Email From: {from_email}
Email To: {to_email}
Email Content:
```
{content}
```
> Triage Result: {result}"""


def format_few_shot_examples(examples):
    """Turn retrieved past decisions into a few-shot block for the prompt."""
    strs = ["Here are some previous examples:"]
    for eg in examples:
        strs.append(
            EXAMPLE_TEMPLATE.format(
                subject=eg.value["email"]["subject"],
                to_email=eg.value["email"]["to"],
                from_email=eg.value["email"]["author"],
                content=eg.value["email"]["email_thread"][:400],
                result=eg.value["label"],
            )
        )
    return "\n\n------------\n\n".join(strs)


def demo_episodic_memory():
    print("=" * 60)
    print("3. EPISODIC MEMORY")
    print("=" * 60)

    namespace = ("email_assistant", USER_ID, "examples")
    store.put(namespace, "ex-spam", {
        "email": {
            "author": "Marketing <promo@vendor.com>",
            "to": "John Doe <john.doe@company.com>",
            "subject": "50% off developer tools this week",
            "email_thread": "Limited time offer on our IDE bundle. Click to claim.",
        },
        "label": "ignore",
    })
    store.put(namespace, "ex-bug", {
        "email": {
            "author": "Bob <bob@company.com>",
            "to": "John Doe <john.doe@company.com>",
            "subject": "Production login failures",
            "email_thread": "Auth service is returning 500s for ~5% of logins since the deploy.",
        },
        "label": "respond",
    })

    # Retrieval is semantic, so the examples that reach the prompt are the ones
    # resembling the email being classified — not an arbitrary sample.
    hits = store.search(namespace, query=str({"email": SAMPLE_EMAIL}), limit=2)
    print(f"Retrieved {len(hits)} past decision(s) similar to the incoming email:")
    for hit in hits:
        print(f"  [{hit.value['label']:7}] {hit.value['email']['subject']}")
    print(f"\nFormatted into the triage prompt:\n{format_few_shot_examples(hits)[:300]}...\n")


# ---------------------------------------------------------------------------
# 4. The Triage Router
# ---------------------------------------------------------------------------

TRIAGE_SYSTEM_PROMPT = """
< Role >
You are {full_name}'s executive assistant.
</ Role >

< Background >
{user_profile_background}.
</ Background >

< Instructions >
{name} gets lots of emails. Categorise each email into one of three categories:
1. IGNORE  - not worth responding to or tracking
2. NOTIFY  - important information, but no response needed
3. RESPOND - needs a direct response from {name}
</ Instructions >

< Rules >
Emails that are not worth responding to:
{triage_no}

Emails {name} should know about but need no response:
{triage_notify}

Emails worth responding to:
{triage_email}
</ Rules >

< Few shot examples >
Follow these examples more than any instruction above.

{examples}
</ Few shot examples >
"""

# The course notebook imports this from a `prompts` module that ships with the
# course workspace; it is inlined here so the tutorial stands alone.
TRIAGE_USER_PROMPT = """
Please determine how to handle the below email thread:

From: {author}
To: {to}
Subject: {subject}
{email_thread}"""


class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(description="Step-by-step reasoning behind the classification.")
    classification: Literal["ignore", "respond", "notify"] = Field(
        description=(
            "The classification of an email: 'ignore' for irrelevant emails, "
            "'notify' for important information that doesn't need a response, "
            "'respond' for emails that need a reply"
        )
    )


llm_router = llm.with_structured_output(Router)


class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]


def triage_router(state: State, config, store) -> Command[Literal["response_agent", "__end__"]]:
    """Classify the email, pulling both rules and examples out of memory."""
    email_input = state["email_input"]
    user_id = config["configurable"]["langgraph_user_id"]

    examples = store.search(
        ("email_assistant", user_id, "examples"), query=str({"email": email_input})
    )

    system_prompt = TRIAGE_SYSTEM_PROMPT.format(
        **profile,
        triage_no=get_prompt("triage_ignore", prompt_instructions["triage_rules"]["ignore"]),
        triage_notify=get_prompt("triage_notify", prompt_instructions["triage_rules"]["notify"]),
        triage_email=get_prompt("triage_respond", prompt_instructions["triage_rules"]["respond"]),
        examples=format_few_shot_examples(examples),
    )
    user_prompt = TRIAGE_USER_PROMPT.format(
        author=email_input["author"],
        to=email_input["to"],
        subject=email_input["subject"],
        email_thread=email_input["email_thread"],
    )

    result = llm_router.invoke(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}]
    )

    if result.classification == "respond":
        print(f"  triage -> RESPOND ({result.reasoning[:70]}...)")
        return Command(
            goto="response_agent",
            update={"messages": [{
                "role": "user",
                "content": f"Respond to this email:\n\n{user_prompt}",
            }]},
        )

    print(f"  triage -> {result.classification.upper()} ({result.reasoning[:70]}...)")
    return Command(goto=END)


def demo_triage_router():
    print("=" * 60)
    print("4. THE TRIAGE ROUTER")
    print("=" * 60)
    print("Structured output forces one of three labels; the rules come from memory.\n")

    for label, email in [
        ("direct question", SAMPLE_EMAIL),
        ("marketing blast", {
            "author": "Deals <deals@vendor.com>",
            "to": "John Doe <john.doe@company.com>",
            "subject": "Your weekly newsletter",
            "email_thread": "Top 10 productivity hacks, plus 30% off our annual plan.",
        }),
    ]:
        print(f"{label}:")
        result = llm_router.invoke([
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT.format(
                **profile,
                triage_no=get_prompt("triage_ignore", prompt_instructions["triage_rules"]["ignore"]),
                triage_notify=get_prompt("triage_notify", prompt_instructions["triage_rules"]["notify"]),
                triage_email=get_prompt("triage_respond", prompt_instructions["triage_rules"]["respond"]),
                examples=format_few_shot_examples(
                    store.search(("email_assistant", USER_ID, "examples"), query=str(email))
                ),
            )},
            {"role": "user", "content": TRIAGE_USER_PROMPT.format(
                author=email["author"], to=email["to"],
                subject=email["subject"], email_thread=email["email_thread"])},
        ])
        print(f"  -> {result.classification}\n")


# ---------------------------------------------------------------------------
# 5. The Email Agent
# ---------------------------------------------------------------------------


@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    return f"Email sent to {to} with subject '{subject}'"


@tool
def schedule_meeting(
    attendees: list[str], subject: str, duration_minutes: int, preferred_day: str
) -> str:
    """Schedule a calendar meeting."""
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"


@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"


AGENT_SYSTEM_PROMPT = """
< Role >
You are {full_name}'s executive assistant.
</ Role >

< Tools >
1. write_email(to, subject, content)
2. schedule_meeting(attendees, subject, duration_minutes, preferred_day)
3. check_calendar_availability(day)
4. manage_memory - store anything worth remembering about contacts or decisions
5. search_memory - look up what you stored earlier
</ Tools >

< Instructions >
{instructions}
</ Instructions >
"""


def create_prompt(state, config, store):
    """Build the agent's system prompt from procedural memory, per invocation.

    Reading the prompt here rather than closing over a constant is what lets
    section 6 change the agent's behaviour without rebuilding the graph.
    """
    user_id = config["configurable"]["langgraph_user_id"]
    result = store.get((user_id,), "agent_instructions")
    instructions = (
        result.value["prompt"] if result else prompt_instructions["agent_instructions"]
    )
    return [{
        "role": "system",
        "content": AGENT_SYSTEM_PROMPT.format(instructions=instructions, **profile),
    }] + state["messages"]


response_agent = create_react_agent(
    llm,
    tools=[
        write_email,
        schedule_meeting,
        check_calendar_availability,
        manage_memory_tool,
        search_memory_tool,
    ],
    prompt=create_prompt,
    store=store,
)

email_agent = (
    StateGraph(State)
    .add_node(triage_router)
    .add_node("response_agent", response_agent)
    .add_edge(START, "triage_router")
    .compile(store=store)
)

CONFIG = {"configurable": {"langgraph_user_id": USER_ID}}


def run_agent(email_input):
    """Invoke the graph and print the tool calls it made."""
    response = email_agent.invoke({"email_input": email_input}, config=CONFIG)
    for message in response.get("messages", []):
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                print(f"  tool: {call['name']}({json.dumps(call['args'])[:90]})")
        elif getattr(message, "content", None) and message.__class__.__name__ == "AIMessage":
            print(f"  reply: {str(message.content)[:150]}")
    return response


def demo_email_agent():
    print("=" * 60)
    print("5. THE EMAIL AGENT")
    print("=" * 60)
    print("Triage decides whether to answer at all; the ReAct agent does the work.\n")
    run_agent(SAMPLE_EMAIL)
    print()


# ---------------------------------------------------------------------------
# 6. Procedural Memory
# ---------------------------------------------------------------------------


def optimize_prompts(response, feedback: str):
    """Rewrite stored instructions from one piece of user feedback."""
    optimizer = create_multi_prompt_optimizer(llm, kind="prompt_memory")

    keys = {
        "main_agent": "agent_instructions",
        "triage-ignore": "triage_ignore",
        "triage-notify": "triage_notify",
        "triage-respond": "triage_respond",
    }
    when_to_update = {
        "main_agent": "Update whenever there is feedback on how the agent should write emails or schedule events",
        "triage-ignore": "Update whenever there is feedback on which emails should be ignored",
        "triage-notify": "Update whenever there is feedback on which emails the user should be notified of",
        "triage-respond": "Update whenever there is feedback on which emails deserve a response",
    }
    prompts = [
        {
            "name": name,
            "prompt": store.get((USER_ID,), key).value["prompt"],
            "update_instructions": "keep the instructions short and to the point",
            "when_to_update": when_to_update[name],
        }
        for name, key in keys.items()
    ]

    try:
        updated = optimizer.invoke(
            {"trajectories": [(response["messages"], feedback)], "prompts": prompts}
        )
    except IndexError:
        # langmem decides what to rewrite via `result["responses"][0].which`. When
        # a small model fails to emit that structured choice the list is empty and
        # langmem raises instead of degrading. Treat it as "nothing selected"
        # rather than letting the tutorial die — it happens perhaps one run in
        # three with qwen2.5:7b.
        print("  optimizer returned no prompt selection this run (small-model failure)")
        return []

    # The course notebook only writes `main_agent` back and leaves the rest as
    # an exercise. All four are handled here, or feedback about triage would be
    # computed and then silently discarded.
    changed = []
    for new, old in zip(updated, prompts):
        if new["prompt"] != old["prompt"]:
            store.put((USER_ID,), keys[old["name"]], {"prompt": clean_prompt(new["prompt"])})
            changed.append(old["name"])
    return changed


def demo_procedural_memory():
    print("=" * 60)
    print("6. PROCEDURAL MEMORY")
    print("=" * 60)
    print("The agent rewrites its own instructions from plain-English feedback.\n")

    before = store.get((USER_ID,), "agent_instructions").value["prompt"]
    print(f"agent_instructions before:\n  {before}\n")

    response = email_agent.invoke({"email_input": SAMPLE_EMAIL}, config=CONFIG)
    changed = optimize_prompts(response, "Always sign your emails `John Doe`")

    print(f"\nFeedback: 'Always sign your emails `John Doe`'")
    print(f"Prompts rewritten: {changed or 'none'}")
    print(f"\nagent_instructions after:\n  {store.get((USER_ID,), 'agent_instructions').value['prompt']}\n")
    print("Only the prompts the feedback is about should change — the optimizer")
    print("decides which, which is why each one carries a `when_to_update` note.\n")


# ---------------------------------------------------------------------------
# 7. Watching Behaviour Change
# ---------------------------------------------------------------------------


def demo_behaviour_change():
    print("=" * 60)
    print("7. WATCHING BEHAVIOUR CHANGE")
    print("=" * 60)
    print("Feedback about triage should move a triage rule, not the agent prompt.\n")

    nuisance = {
        "author": "Alice Jones <alice.jones@bar.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": "Hi John,\n\nUrgent issue - your service is down. Is there a reason why?",
    }

    print("Before feedback:")
    response = run_agent(nuisance)

    print(f"\ntriage_ignore before:\n  {store.get((USER_ID,), 'triage_ignore').value['prompt']}")
    changed = optimize_prompts(response, "Ignore any emails from Alice Jones")
    print(f"\nFeedback: 'Ignore any emails from Alice Jones'")
    print(f"Prompts rewritten: {changed or 'none'}")
    print(f"\ntriage_ignore after:\n  {store.get((USER_ID,), 'triage_ignore').value['prompt']}")

    print("\nAfter feedback (same email again):")
    run_agent(nuisance)

    if "triage-ignore" in changed:
        print("\nThe feedback landed on the right prompt. Whether the 7B then acts on")
        print("the rewritten rule is a separate question — rerun to see it vary.\n")
    else:
        print(f"\nThe feedback was about *triage*, but the optimizer attributed it to")
        print(f"{changed or 'nothing'} instead, so `triage_ignore` never changed and the email")
        print("is still answered.")
        print("\nThis is the honest limit of running the whole loop on a 7B. Deciding")
        print("*which* prompt a piece of feedback belongs to is the hardest step, and")
        print("qwen2.5:7b picks a different answer almost every run — three trials of")
        print("this exact feedback gave triage-respond, triage-ignore and main_agent,")
        print("despite temperature=0. The agent runs fine locally; the optimizer is the")
        print("component to point at a stronger model first.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    demo_memory_types()
    demo_semantic_memory()
    demo_episodic_memory()
    demo_triage_router()
    demo_email_agent()
    demo_procedural_memory()
    demo_behaviour_change()
    print("All demos completed!")


if __name__ == "__main__":
    main()
