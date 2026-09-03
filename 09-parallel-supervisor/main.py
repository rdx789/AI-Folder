"""CLI entry point for the NovaOps assistant.

    python main.py ["<request>"]   # run the whole flow in one process
    python main.py resume           # recover: a previous run's process died mid-approval

The run is one process — it pauses inline for the write approval, then finishes.
The checkpoint is still durable (`AsyncSqliteSaver` on disk), so if the process is
killed while waiting at the approval prompt, `resume` reopens that exact checkpoint
and continues.
"""

import asyncio
import sys
import uuid
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from mcp_client import discover_tools
from supervisor import build_graph

DEFAULT_REQUEST = "I'm E010 and Webex says I'm not licensed — can you help?"

CHECKPOINT_DB = Path(__file__).with_name("checkpoints.sqlite")
# Each run gets its own thread; the id is stashed here so `resume` can find the
# last one if a run was interrupted before it finished.
LAST_THREAD_FILE = Path(__file__).with_name(".last_thread")


def _interrupt_payload(result_or_state):
    """Pull the pending interrupt's value out of an ainvoke result or a state snapshot."""
    interrupts = getattr(result_or_state, "interrupts", None)
    if interrupts is None and isinstance(result_or_state, dict):
        interrupts = result_or_state.get("__interrupt__")
    if interrupts:
        return interrupts[0].value
    return None


def _ask_approval(payload: dict) -> dict:
    print("\n--- Approval required ---")
    print(f"Action: {payload['action']}")
    print(f"  employee_id: {payload['employee_id']}")
    print(f"  software: {payload['software']}")
    print(f"  justification: {payload['business_justification']}")
    if payload.get("reasoning"):
        print(f"  (planner reasoning: {payload['reasoning']})")
    try:
        choice = input("Approve this request? [y/n]: ").strip().lower()
    except EOFError:
        choice = "n"
    if choice == "y":
        return {"approved": True, "reason": None}
    reason = ""
    while not reason.strip():
        try:
            reason = input("Denial reason (required): ").strip()
        except EOFError:
            print("No reason given; aborting.")
            sys.exit(1)
    return {"approved": False, "reason": reason}


async def _drive(graph, first_input, config) -> dict:
    """Invoke the graph, servicing any approval interrupt inline, until it finishes."""
    result = await graph.ainvoke(first_input, config=config)
    while True:
        payload = _interrupt_payload(result)
        if payload is None:
            return result
        decision = _ask_approval(payload)
        result = await graph.ainvoke(Command(resume=decision), config=config)


async def _run(request: str) -> None:
    tools = await discover_tools()
    print(f"Discovered {len(tools)} MCP tools: {', '.join(sorted(tools))}")

    thread_id = uuid.uuid4().hex
    LAST_THREAD_FILE.write_text(thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        graph = build_graph(tools, saver)
        print(f"\nYou: {request}\n")
        result = await _drive(graph, {"request": request}, config)

    print(f"\nAssistant: {result.get('answer', '(no answer produced)')}")


async def _resume() -> None:
    if not (CHECKPOINT_DB.exists() and LAST_THREAD_FILE.exists()):
        print("Nothing to resume — no prior run found.")
        sys.exit(1)

    thread_id = LAST_THREAD_FILE.read_text().strip()
    config = {"configurable": {"thread_id": thread_id}}
    tools = await discover_tools()

    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        graph = build_graph(tools, saver)

        snapshot = await graph.aget_state(config)
        if _interrupt_payload(snapshot) is None:
            if snapshot.values.get("answer"):
                print("That run already finished.\n")
                print(f"Assistant: {snapshot.values['answer']}")
            else:
                print("No pending approval on the last run — nothing to resume.")
            return

        print(f"Resuming the last run (thread {thread_id}).")
        result = await _drive(graph, None, config)

    print(f"\nAssistant: {result.get('answer', '(no answer produced)')}")


def main() -> None:
    args = sys.argv[1:]
    try:
        if args and args[0] == "resume":
            asyncio.run(_resume())
        else:
            request = args[0] if args else DEFAULT_REQUEST
            asyncio.run(_run(request))
    except RuntimeError as exc:  # e.g. MCP server unreachable
        print(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
