"""Watching AgentState change, one node at a time.

This lesson is about what each stage puts in front of the model, and the visible half
of that is AgentState: Stage 0 has one field, Stage 03 has seven. Run either stage
with `--trace` and every node reports the update it returned, so you can watch state
grow, get overwritten, and — in Stage 03's rearm — shrink.

Two rules make the output readable, and they are the LangGraph rules themselves:

  `messages` has a reducer (`add_messages`), so an update APPENDS. Its line shows the
  count either side plus the message that arrived.

  Every other field has no reducer, so an update REPLACES it. A plan and a memory are
  printed IN FULL, one field per line — they are what the stage spent a model call to
  produce, so abbreviating them would hide the decision you came to watch. When such a
  field is replaced rather than set, only the fields that moved are printed, with the
  values that came and went.

Nothing in this module runs unless `--trace` was passed: `update()` and `tool_node()`
hand back exactly what they were given while tracing is off, and `evals/compare.py`
never turns it on. The measured path is the untraced one.
"""

import json
import sys
import textwrap

from langchain_core.messages import RemoveMessage


WIDTH = 88  # cap for inline previews (message content, END-state summaries)
ITEM_CHARS = 160  # cap for ONE value inside a plan or memory — ~2 folded lines

_ENABLED = False


def enable() -> None:
    global _ENABLED
    _ENABLED = True


def enable_from_argv(argv: list | None = None) -> bool:
    """Turn tracing on when `--trace` is present, and LEAVE the flag in argv.

    The flag is deliberately not popped from argv: the shared runner declares
    `--trace` too, because the same switch drives two different things. Here it turns
    on the per-node lines; in the runner it turns the per-turn summary from a row in a
    wide table into a labelled block. Neither side imports the other to find that out.
    """
    argv = sys.argv if argv is None else argv
    if "--trace" in argv:
        enable()
    return _ENABLED


def trace(node: str, message: str) -> None:
    """One labelled line. The label says which node produced it, so reading the lines
    in order IS reading the path taken through the graph."""

    if _ENABLED:
        print(f"  [{node:<7}] {message}")


def _text(message) -> str:
    """Flatten a message's content enough to measure and preview it.

    Display-only, which is why it does not reuse `agent.message_text`: keeping this
    module free of imports from `agent` lets `agent` import IT.
    """

    content = getattr(message, "content", message)
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content).strip()


def _short(value: str, width: int = WIDTH) -> str:
    value = " ".join(value.split())
    return value if len(value) <= width else value[: width - 1] + "…"


def _render(value, width: int = WIDTH) -> str:
    """A compact one-line view of any state field."""

    if value is None or value == {} or value == []:
        return "∅"
    if isinstance(value, dict):
        return _short("{" + ", ".join(f"{k}: {_render(v, 28)}" for k, v in value.items()) + "}", width)
    if isinstance(value, list):
        # Short lists read better in full; long ones only need their size.
        rendered = ", ".join(str(item) for item in value)
        if len(rendered) <= width:
            return f"[{rendered}]"
        return f"{len(value)} item" + ("s" if len(value) != 1 else "")
    return _short(str(value), width)


INDENT = " " * 12  # lines up under the `  [node   ] ` prefix that trace() writes


def _continued(line: str) -> None:
    """A wrapped continuation of the line above, with no node label of its own."""

    if _ENABLED:
        print(f"{INDENT}{line}")


def _wrapped(text, first: str, cont: str, width: int = 96) -> list[str]:
    """Fold a value across lines, capped at ITEM_CHARS with the overflow reported.

    The cap is a compromise between two failures. Cut a value to one short line and a
    planner quoting a 350-character policy paragraph into `relevant_facts` looks like
    any other fact — which is how that goes unnoticed. Print it whole and one bad field
    floods the screen. Two-ish lines plus "[+N more chars]" shows both that it is prose
    rather than a value, and how much you are paying to carry it.
    """

    text = str(text)
    if len(text) > ITEM_CHARS:
        text = f"{text[:ITEM_CHARS].rstrip()}… [+{len(text) - ITEM_CHARS} more chars]"
    body = textwrap.wrap(text, width=max(20, width - len(first))) or ["∅"]
    return [first + body[0]] + [cont + line for line in body[1:]]


def _field_lines(key: str, value) -> list[str]:
    """One state field per line, every entry shown.

    A plan or a memory IS the stage's output — the thing it spent a model call to
    produce — so no entry is ever dropped or summarised as a count. Individual values
    are capped at ITEM_CHARS, and say so when they are.
    """

    label, pad = f"{key:<22}", f"{'':<22}"
    if isinstance(value, list):
        if not value:
            return [f"{label}∅"]
        lines = []
        for index, item in enumerate(value):
            lines += _wrapped(item, label if index == 0 else pad, pad)
        return lines
    return _wrapped(value if value != "" else "∅", label, pad)


def _diff_lines(old: dict, new: dict, changed: list[str]) -> list[str]:
    """The changed fields of a dict, showing the actual values that came and went."""

    pad = f"{'':<22}"
    lines = []
    for key in changed:
        before, after = old.get(key), new[key]
        if isinstance(before, list) and isinstance(after, list):
            lines.append(f"{key:<22}{len(before)} → {len(after)}")
            for item in [i for i in after if i not in before]:
                lines += _wrapped(item, f"{pad}  + ", f"{pad}    ")
            for item in [i for i in before if i not in after]:
                lines += _wrapped(item, f"{pad}  − ", f"{pad}    ")
        else:
            lines += _wrapped(f"{before} → {after}", f"{key:<22}", pad)
    return lines


def describe(message) -> str:
    """What a single appended (or removed) message is, in a few words."""

    kind = type(message).__name__
    if isinstance(message, RemoveMessage):
        return f"−{kind}"
    calls = getattr(message, "tool_calls", None) or []
    if calls:
        wants = ", ".join(
            f"{call['name']}({json.dumps(call.get('args', {}))})" for call in calls
        )
        return f"+{kind} · wants {_short(wants, 64)}"
    if kind == "ToolMessage":
        return f"+{kind} {getattr(message, 'name', '?')} · {len(_text(message))} chars"
    body = _text(message)
    label = "answer" if kind == "AIMessage" else _short(body, 40)
    return f"+{kind} · {label}, {len(body)} chars" if kind == "AIMessage" else f"+{kind} · {label}"


def update(node: str, current: dict, changes: dict) -> dict:
    """Print what this node's return value does to state, then return it untouched.

    Wrapping the RETURN VALUE rather than writing a print by hand is deliberate: the
    line can never describe something the node did not actually do.
    """

    if not _ENABLED:
        return changes
    if not changes:
        trace(node, "no state change")
        return changes

    for key, value in changes.items():
        if key == "messages":
            before = len(current.get("messages") or [])
            removed = sum(1 for m in value if isinstance(m, RemoveMessage))
            after = before + len(value) - 2 * removed
            detail = ", ".join(describe(m) for m in value)
            trace(node, f"Δ messages  {before} → {after}   ({detail})")
            continue

        old = current.get(key)
        if old == value:
            continue  # a node may re-set a field to what it already was; that is not news

        # A plan and a memory are the stages' real output, so they get printed whole,
        # one field per line. Everything else fits on the line it is announced on.
        if isinstance(value, dict) and value:
            if not old:
                trace(node, f"Δ {key}  set")
                for line in [l for k, v in value.items() for l in _field_lines(k, v)]:
                    _continued(line)
            else:
                fields = [k for k in value if old.get(k) != value[k]]
                trace(node, f"Δ {key}  {len(fields)} field{'s' if len(fields) != 1 else ''} changed")
                for line in _diff_lines(old, value, fields):
                    _continued(line)
        elif isinstance(value, list) and len(", ".join(map(str, value))) > 70:
            trace(node, f"Δ {key}  {len(old or [])} → {len(value)}")
            for item in value:
                _continued(f"  {item}")
        else:
            trace(node, f"Δ {key}  {_render(old)} → {_render(value)}")
    return changes


def context(node: str, messages: list, current: dict | None, exposed, catalogue,
            rendered: bool = False) -> None:
    """The one line worth comparing across both stages: what this call receives.

    Stage 0 sends every message and every schema; Stage 03 sends a handful of each.
    Same format everywhere, so the two runs can be read side by side.

    `rendered=True` marks a planner or distiller call, and the distinction matters:
    those do not receive the history AS messages, they receive it flattened into one
    block of text. Counting messages there would describe the wrong thing twice over —
    the block is one "message", and how much of the conversation went into it is a
    decision the stage makes, not something the message count reveals. So we measure
    the block itself. A planner that renders the whole transcript makes this number
    climb all session; Stage 03 renders memory plus only what is newer than
    `distilled_upto`, and it stays flat. That contrast IS history distillation.
    """

    if not _ENABLED:
        return
    sent = sum(1 for m in messages if type(m).__name__ != "SystemMessage")
    held = len(current.get("messages") or []) if current else sent
    if rendered:
        # ~4 chars per token, the same rough convention the token meter uses.
        size = sum(len(_text(m)) for m in messages if type(m).__name__ != "SystemMessage") // 4
        trace(node, f"context = system + rendered block ~{size:,} est tokens "
                    f"({held} msgs in state) · no schemas")
        return
    tools = (
        f"{len(exposed or [])} of {len(catalogue)} schemas" if catalogue
        else f"{len(exposed or [])} schemas"
    )
    trace(node, f"context = system + {sent} of {held} msgs · {tools}")


def tool_node(node):
    """Wrap a ToolNode so tool results are visible — but only while tracing.

    With tracing off this returns the ToolNode itself, so the graph the evaluation
    runs is exactly the graph that existed before this module. With tracing on it
    returns an async wrapper: the MCP tools are async-only, so `ainvoke` is the path
    that works.
    """

    if not _ENABLED:
        return node

    async def traced_tools(current: dict) -> dict:
        result = await node.ainvoke(current)
        return update("tools", current, result)

    return traced_tools
