"""Agentic RAG tool: retrieval is a tool the model DECIDES whether to call —
toolChoice is "auto", so chit-chat gets answered directly and only questions
that need the knowledge base trigger a search_knowledge_base call. Usable
standalone (main()) or imported by any app via answer_question().
"""
from bedrock_client import converse
from registry import get_registry
from search import search

MAX_TURNS = 5

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a search_knowledge_base tool. "
    "Use it for any question that needs facts from the knowledge bases; answer "
    "greetings and questions about what you can do directly, without searching. "
    "Answer ONLY from what the tool returns — if the results don't contain the "
    "answer, say you don't know rather than guessing."
)


def build_tool_spec(registry) -> list[dict]:
    """One generic search tool; the kb_name enum is the registry, so a new
    data/<kb>/ folder becomes searchable with no code change here."""
    kb_descriptions = "\n".join(f"- {kb.name}: {kb.description}" for kb in registry.values())
    return [{
        "toolSpec": {
            "name": "search_knowledge_base",
            "description": (
                "Search one knowledge base for chunks relevant to a query. "
                f"Available knowledge bases:\n{kb_descriptions}"
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "kb_name": {
                            "type": "string",
                            "enum": list(registry.keys()),
                            "description": "Which knowledge base to search.",
                        },
                        "query": {
                            "type": "string",
                            "description": "The search query, rephrased for retrieval if helpful.",
                        },
                    },
                    "required": ["kb_name", "query"],
                }
            },
        }
    }]


def execute_tool(tool_use: dict, registry) -> list[dict]:
    kb_name = tool_use["input"]["kb_name"]
    query = tool_use["input"]["query"]
    kb = registry.get(kb_name)
    if kb is None:
        raise RuntimeError(f"Model chose unknown knowledge base '{kb_name}'.")
    return search(kb, query)


def answer_question(question: str) -> str:
    """Agent loop: the model may call search_knowledge_base zero, one, or
    several times (even several in one turn) before giving a final answer."""
    registry = get_registry()
    if not registry:
        raise RuntimeError("No knowledge bases found under data/.")
    tool_spec = build_tool_spec(registry)

    messages = [{"role": "user", "content": [{"text": question}]}]

    for _ in range(MAX_TURNS):
        resp = converse(
            messages,
            system=SYSTEM_PROMPT,
            tool_config=tool_spec,
            tool_choice={"auto": {}},
        )
        message = resp["output"]["message"]
        messages.append(message)

        if resp["stopReason"] != "tool_use":
            for block in message["content"]:
                if "text" in block:
                    return block["text"]
            return "The model returned no text answer."

        # A single turn can contain several toolUse blocks (e.g. a question
        # spanning two knowledge bases) — every one needs a matching toolResult
        # in the next user message, or Converse rejects the history.
        result_blocks = []
        for block in message["content"]:
            if "toolUse" not in block:
                continue
            tool_use = block["toolUse"]
            print(f'  -> search_knowledge_base(kb={tool_use["input"]["kb_name"]!r}, query={tool_use["input"]["query"]!r})')
            chunks = execute_tool(tool_use, registry)
            content = [{"json": {"chunks": chunks}}] if chunks else [{"text": "No matching chunks found."}]
            result_blocks.append({"toolResult": {"toolUseId": tool_use["toolUseId"], "content": content}})
        messages.append({"role": "user", "content": result_blocks})

    return f"(no final answer after {MAX_TURNS} turns)"


def main():
    print("Agentic RAG — ask a question (type 'quit' or 'exit' to stop).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Bye.")
            break
        try:
            print(answer_question(question))
        except RuntimeError as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
