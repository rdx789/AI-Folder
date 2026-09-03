"""Role-based RAG, the naive way: one vector index per audience, gated by role.

Managers may read both the employee handbook and the manager playbook;
employees only the handbook. Access control is which corpora a role's PLAN step
may route to — tools.plan_tool restricts its source enum to ROLE_TOOLS[role], so
the model can't search what its role can't reach.

Unlike rag_tool.py's agent loop, this uses a plan -> retrieve -> answer flow:
a forced planning call first decides which corpora to search (see the comment on
that call for why it's forced), then we retrieve each, then the model answers
from the gathered context. That's what makes a single manager question reliably
draw on BOTH corpora.

It still doesn't scale: every audience is its own index — another store to
build, hold, and re-embed — and no single search ranks across corpora. Lesson 7
solves the same problem with ONE index and metadata filtering.

    python 02-rag-as-tool/ingest_kbs.py             # build the indexes (once)
    python 02-rag-as-tool/rag_by_role.py            # scripted demo (no role)
    python 02-rag-as-tool/rag_by_role.py employee   # interactive
    python 02-rag-as-tool/rag_by_role.py manager
"""
import os
import sys

import boto3
from dotenv import find_dotenv, load_dotenv

from tools import ROLE_TOOLS, format_context, label, load_tools, plan_tool, retrieve

load_dotenv(find_dotenv())

client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
MODEL_ID = os.environ["BEDROCK_MODEL_ID"]

PLAN_SYSTEM = (
    "You plan which NovaOps knowledge sources are needed to answer the user's "
    "question. List every source needed — a question spanning company policy AND "
    "manager how-to needs BOTH. For greetings or questions about what you can "
    "do, plan no sources."
)

ANSWER_SYSTEM = (
    "You are the NovaOps internal assistant. If context is provided, answer ONLY "
    "from it, and say so if it doesn't contain the answer. If no context is "
    "provided, answer directly and briefly."
)


def plan_sources(question: str, tool_names: list[str]) -> list[dict]:
    # THE FIX for multi-source retrieval. In rag_tool.py's agent loop the model
    # decides which tools to call — but Nova 2 Lite reliably emits only ONE tool
    # call per turn, so a manager question spanning two corpora would search just
    # one of them. Forcing a single plan_sources call (toolChoice: tool) makes the
    # model ENUMERATE every corpus it needs in one structured answer, which is
    # reliable for single- and dual-source questions alike (and empty for chit-chat).
    response = client.converse(
        modelId=MODEL_ID,
        system=[{"text": PLAN_SYSTEM}],
        messages=[{"role": "user", "content": [{"text": question}]}],
        # This toolConfig IS the access control, in two layers. Soft: plan_tool
        # scopes the `source` enum to THIS role's corpora, so an employee's plan
        # can't even name the playbook. Hard: load_tools (via setup) loads only the
        # role's own indexes, so a stray source can't leak — it isn't in memory.
        toolConfig={"tools": [plan_tool(tool_names)], "toolChoice": {"tool": {"name": "plan_sources"}}},
    )
    for block in response["output"]["message"]["content"]:
        if "toolUse" in block:
            return block["toolUse"]["input"].get("searches", [])
    return []


def generate(question: str, context: str | None) -> str:
    prompt = (
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer using only the context above."
        if context
        else question  # greeting / capability question — no retrieval
    )
    response = client.converse(
        modelId=MODEL_ID,
        system=[{"text": ANSWER_SYSTEM}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 600, "temperature": 0.2},
    )
    return "".join(b["text"] for b in response["output"]["message"]["content"] if "text" in b)


def print_scores(name: str, query: str, results: list[tuple[int, float, str, str]]) -> None:
    print(f'  [{label(name)}]  query: "{query}"')
    for chunk_id, score, _text, _source in results:
        print(f"      chunk {chunk_id:<3}  cos {score:.3f}")


def run(question: str, tool_names: list[str], loaded: dict) -> None:
    print(f"Question: {question}")
    plan = plan_sources(question, tool_names)

    if not plan:
        print("Plan: no search needed (greeting / capability question).")
        print(f"\nAnswer:\n{generate(question, None)}\n")
        return

    # The plan solves TWO problems at once: which corpora to search (routing) and
    # WHAT to search each for (a query the model rewrote per source). We prove the
    # second by also searching each source with the untouched original question —
    # its chunks usually score lower, and are NOT sent to the model.
    print("Plan: " + " + ".join(f'{label(s["source"])}' for s in plan))

    print("\nPlanned queries  (retrieved chunks INCLUDED in the answer prompt):")
    planned = []
    for s in plan:
        results = retrieve(s["source"], s["query"], loaded)
        planned.append((s["source"], results))
        print_scores(s["source"], s["query"], results)

    print("\nOriginal question on the same sources  (NOT included — comparison only):")
    for s in plan:
        print_scores(s["source"], question, retrieve(s["source"], question, loaded))

    context = "\n\n".join(format_context(name, results) for name, results in planned)
    print(f"\nAnswer:\n{generate(question, context)}\n")


# Scripted demo shown when no role is given: the access gap (same question, both
# roles), then two manager questions that each need BOTH corpora — reliable now
# that planning enumerates the sources.
DEMO = [
    ("employee", "How should I run 1:1s with my report?"),
    ("manager", "How should I run 1:1s with my report?"),
    # Two cross-source questions; each shows the tailored per-source query beating
    # the raw question on the corpus whose match the *other* half was diluting —
    # the playbook for severance, the handbook for parental leave.
    ("manager", "What is our severance policy in weeks of pay, and how should I handle the termination conversation with a report?"),
    ("manager", "What is our parental leave policy, and how should I onboard a report returning from leave?"),
]


def setup(role: str) -> tuple[list[str], dict]:
    tool_names = ROLE_TOOLS[role]
    return tool_names, load_tools(tool_names)


def run_demo() -> None:
    print("No role given — running a scripted demo.")
    print("(Pass 'employee' or 'manager' for an interactive session.)\n")
    cache: dict[str, tuple[list[str], dict]] = {}
    for role, question in DEMO:
        if role not in cache:
            cache[role] = setup(role)
        tool_names, loaded = cache[role]
        print(f"===== [{role}]  (tools: {', '.join(tool_names)}) =====")
        run(question, tool_names, loaded)


def run_interactive(role: str) -> None:
    tool_names, loaded = setup(role)
    print(f"Role: {role} — tools: {', '.join(tool_names)}")
    print("Empty line or 'quit' to exit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"quit", "exit", "q"}:
            break
        print()
        run(question, tool_names, loaded)


def main() -> None:
    role = sys.argv[1] if len(sys.argv) > 1 else None
    if role is None:
        run_demo()
    elif role in ROLE_TOOLS:
        run_interactive(role)
    else:
        raise SystemExit(f"Usage: python 02-rag-as-tool/rag_by_role.py [{' | '.join(ROLE_TOOLS)}]")


if __name__ == "__main__":
    main()
