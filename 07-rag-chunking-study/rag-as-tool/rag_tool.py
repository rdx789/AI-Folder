"""Tool-based RAG: the model DECIDES when to search the handbook.

01-simple-rag/rag_answer.py retrieves on every question, no matter what — even
"hello" gets four handbook chunks stuffed into its prompt. Here retrieval
becomes a tool in Lesson 4's agent loop: chit-chat is answered directly, policy
questions trigger search_handbook (watch the '-> search_handbook(...)' lines),
and the model can search more than once for a multi-part question.

Retrieval lives in retrieval.py; the tool schema + operation live in tools.py.
This file is just the agent loop that ties them together.

    python 02-rag-as-tool/ingest_kbs.py     # build the corpus indexes (once)
    python 02-rag-as-tool/rag_tool.py
"""
import os
import re

import boto3
from dotenv import find_dotenv, load_dotenv

from tools import format_context, load_tools, retrieve, tool_spec

load_dotenv(find_dotenv())

client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
MODEL_ID = os.environ["BEDROCK_MODEL_ID"]
MAX_TURNS = 5

SYSTEM = (
    "You are the NovaOps employee assistant. For any question about company "
    "policy, benefits, titles, or how the company works, use search_handbook "
    "and answer ONLY from what it returns — if the results don't contain the "
    "answer, say so. Every context chunk is tagged with its source file "
    "(e.g. 'source: benefits-and-perks.md'). Every factual claim in your "
    "answer must cite the source file it came from, inline right after the "
    "claim, e.g. 'you get 20 vacation days a year (benefits-and-perks.md)' — "
    "if an answer draws on two files, cite each claim with its own file, not "
    "one citation for the whole answer. Greetings and questions about "
    "what you can do, answer directly without searching."
)

# Nova sometimes fires its own native citation-grounding on top of the
# explicit "(file.md)" text asked for above, producing back-to-back
# duplicates wrapped in zero-width spaces (invisible in a UI, but visible as
# doubled "(file.md) (file.md)" in plain-text output). Sanitize rather than
# fight it via prompt wording, which didn't change the model's behavior.
_ZERO_WIDTH_RE = re.compile("[​‌﻿]")
_DUPLICATE_CITATION_RE = re.compile(r"(\([\w.\- ]+?\.md\))(?:\s*\1)+")


def _clean_answer(text: str) -> str:
    text = _ZERO_WIDTH_RE.sub("", text)
    return _DUPLICATE_CITATION_RE.sub(r"\1", text)


def run_agent(question: str, loaded: dict, tool_config: dict) -> str:
    # Lesson 4's agent loop. The only thing new vs L4: the tool is retrieval, so
    # the model turns a chat turn into a search exactly when it needs facts.
    messages = [{"role": "user", "content": [{"text": question}]}]

    for _ in range(MAX_TURNS):
        response = client.converse(
            modelId=MODEL_ID,
            system=[{"text": SYSTEM}],
            messages=messages,
            toolConfig=tool_config,
            # Same low temperature as rag_answer.py and rag_by_role.py, so the only
            # thing that changes vs 01-simple-rag is tool-based retrieval, not
            # sampling. Low temp is fine (often better) for reliable tool calling.
            inferenceConfig={"maxTokens": 600, "temperature": 0.2},
        )
        message = response["output"]["message"]
        messages.append(message)

        if response["stopReason"] != "tool_use":
            raw = "".join(block["text"] for block in message["content"] if "text" in block)
            return _clean_answer(raw)

        # The model may emit SEVERAL toolUse blocks in one turn (a compare
        # question -> two searches). Every id needs a matching toolResult back in
        # the next user message, or Converse rejects the history.
        result_blocks = []
        for block in message["content"]:
            if "toolUse" not in block:
                continue
            tool_use = block["toolUse"]
            print(f'  -> {tool_use["name"]}("{tool_use["input"]["query"]}")')
            results = retrieve(tool_use["name"], tool_use["input"]["query"], loaded)
            # Same chunk ids + cosine scores 01-simple-rag prints, so you can see
            # what each search actually returned — and how the model's rewritten
            # query retrieves different chunks than the raw question would.
            retrieved = ", ".join(
                f"{cid} (cos {score:.3f}, source: {source})" for cid, score, _text, source in results
            )
            print(f"     retrieved chunks: {retrieved}")
            context = format_context(tool_use["name"], results)
            result_blocks.append(
                {"toolResult": {"toolUseId": tool_use["toolUseId"], "content": [{"text": context}]}}
            )
        messages.append({"role": "user", "content": result_blocks})

    return f"(no final answer after {MAX_TURNS} turns)"


def main() -> None:
    loaded = load_tools(["search_handbook"])
    tool_config = {"tools": [tool_spec("search_handbook")], "toolChoice": {"auto": {}}}

    print(f"Agent with a search_handbook tool over {loaded['search_handbook']['index'].ntotal} chunks.")
    print("Try a greeting first, then a policy question — watch when it searches.")
    print("Empty line or 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"quit", "exit", "q"}:
            break
        print(f"\n{run_agent(question, loaded, tool_config)}\n")


if __name__ == "__main__":
    main()
