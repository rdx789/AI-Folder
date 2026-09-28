"""
Replaying a prepared session against a compiled graph.

Both stages present the SAME external interface — send one user message with a
`thread_id`, get an answer back — so one runner drives both. That is the
point: from the outside they are interchangeable, and everything that differs
is inside the graph, in what it chooses to put in front of the model.

Two entry points:
  run_session()  one session against one compiled app -> per-turn records
  run_stage()    the standalone `main()` every stage folder shares
"""

import argparse
import asyncio
import time
import uuid

from langchain_core.messages import HumanMessage, ToolMessage

from .evalrules import evidence_of, run_deterministic, run_judges
from .dataset import load_sessions
from .runtime import load_eval_tools, message_text, recent_turns
from .tokens import METER, SCHEMA_TOKENS, index_tools

ROW = "{t:>3} {kind:<10} {intent:<20} {inp:>7} {out:>6} {calls:>5} {tools:>5} {schema:>7} {score:>6} {sec:>5}  {called}"
HEADER = ROW.format(
    t="t", kind="kind", intent="intent", inp="input", out="output", calls="calls",
    tools="tools", schema="schema", score="score", sec="sec", called="tools called",
)


def mean_score(metrics: dict) -> float | None:
    """One number per turn: the mean of every metric that applied to it."""
    values = [s for s, _ in metrics.values()]
    return sum(values) / len(values) if values else None


def _state_line(values: dict) -> str:
    """AgentState on one line, at a turn boundary.

    The runner renders this rather than `tracing`, and the split is not arbitrary:
    `tracing` narrates NODES, which is the graph's business, while START and END are
    TURN boundaries, which are the runner's. A node cannot honestly print "START"
    anyway — the model node runs several times per turn, and only its first pass sees
    the state the turn began from.
    """
    if not values:
        return "∅ (new thread — nothing persisted yet)"
    parts = []
    for key, value in values.items():
        if key == "messages":
            parts.append(f"messages={len(value)}")
        elif isinstance(value, dict):
            inner = ", ".join(f"{k}: {v}" for k, v in value.items())
            parts.append(f"{key}={{{inner[:36] + '…' if len(inner) > 36 else inner}}}" if value
                         else f"{key}=∅")
        elif isinstance(value, list):
            rendered = ", ".join(str(item) for item in value)
            parts.append(f"{key}=[{rendered}]" if len(rendered) <= 36
                         else f"{key}={len(value)} items")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def block_rule(label: str, width: int = 84) -> str:
    """A labelled separator, so every block in a traced run is the same width."""
    head = f"  ── {label} "
    return head + "─" * max(3, width - len(head))


def _intent_line(turn: dict, record: dict) -> str:
    """The planner's classification, next to the thing it is actually graded against.

    Three different fields get confused here, so the block names all of them:
      `kind`            hand-written in sessions.json; decides which CHECKS apply.
      `expected_intent` also hand-written, and the ground truth for the topic. It may
                        list alternatives ("onboarding_status|employee_lookup") when
                        more than one reading of the turn is defensible.
      `intent`          what the stage's planner actually decided, at runtime.
    Only the last two are a matched pair — `intent_accuracy` scores one against the
    other — which is why they share a line and `kind` sits on its own.
    """
    actual = record.get("intent")
    if actual is None:
        return f"{'—':<20}(this stage has no planner)"

    expected = turn.get("expected_intent")
    if not expected:
        return f"{actual:<20}(no intent expected on this turn)"

    allowed = expected.split("|")
    verdict = "OK" if actual in allowed else "MISMATCH"
    return f"{actual:<20}(expected {' or '.join(allowed)})  {verdict}"


def print_turn_block(turn: dict, record: dict, total_tools: int) -> None:
    """One turn as two labelled blocks: what the graph did, then what it cost.

    The wide one-row-per-turn table is right for comparing stages side by side, and
    wrong for reading a single run — the columns are too terse to interpret and the
    trace lines land in the middle of them. This is the same data, laid out to be read.
    """
    available = f"{record['tools_exposed']} of {total_tools}"
    print()
    print(block_rule(f"turn {turn['n']} · what happened"))
    print(f"    turn kind         {record['kind']:<20}(dataset label: routes the checks)")
    print(f"    planner intent    {_intent_line(turn, record)}")
    print(f"    tools available   {available:<20}{record['schema_tokens_est']:,} schema tokens")
    print(f"    tool calls        {len(record['tools_called'])}")
    print(f"    tools called      {', '.join(record['tools_called']) or '—'}")
    if record["error"]:
        print(f"    ERROR             {record['error']}")
    score = f"{record['score']:.2f}" if record["score"] is not None else "—"
    calls = record["model_calls"]
    print(f"    cost              {record['input_tokens']:,} in · {record['output_tokens']:,} out"
          f" · {calls} model call{'s' if calls != 1 else ''}"
          f" · score {score} · {record['latency_s']}s")


async def run_session(app, session: dict, verbose: bool = True, judge: bool = False,
                      trace: bool = False) -> list[dict]:
    """Replay one session's turns in order on a single thread; return per-turn records.

    One `thread_id` for the whole session is what makes this a conversation rather
    than a series of unrelated questions: the checkpointer keeps the graph's state
    between turns, so each turn starts where the last one ended.
    """
    # recursion_limit bounds ONE turn's model↔tools cycle. LangGraph's default is
    # generous, and a degenerate turn — a user statement that isn't a question, with
    # a plan that wrongly says a lookup is needed — will happily loop the ReAct cycle
    # dozens of times. One such turn burned 1.2M tokens while this lab was being
    # built. Every legitimate turn here finishes in at most five model calls, so 12
    # supersteps is a comfortable ceiling and a cheap failure. Any agent loop you
    # ship needs a bound like this; the default is not one.
    config = {
        "configurable": {"thread_id": f"{session['id']}-{uuid.uuid4().hex[:8]}"},
        "recursion_limit": 12,
    }
    records = []
    # Evidence accumulates ACROSS the session, so a summary turn can be graded
    # against everything the session gathered rather than only its own turn.
    session_evidence: list[str] = []

    if verbose:
        print(f"\n── {session['id']} ── {session['persona']}\n")
        if not trace:
            print(HEADER)

    for turn in session["turns"]:
        if verbose and trace:
            # Block 1 opens here so the node lines that follow are visibly part of it.
            print()
            print(block_rule(f"turn {turn['n']} · trajectory and state"))
            # The checkpoint this turn resumes from — everything the graph remembers
            # before the new message is merged in, which is what makes this a
            # conversation rather than twelve unrelated questions.
            before = app.get_state(config).values
            carried = len(before.get("messages") or [])
            print(f"  [START  ] state: {_state_line(before)}")
            print(f"  [START  ] + the user's new message → messages={carried + 1}")
        METER.reset()
        started = time.perf_counter()
        try:
            result = await app.ainvoke(
                {"messages": [HumanMessage(turn["user"])]}, config=config
            )
            answer = message_text(result["messages"][-1])
            # Keep what the tools actually returned this turn — the faithfulness
            # judge grades the answer against it. Truncated because a policy
            # document is long and the judge only needs the part that was quotable.
            tool_results = [
                {"name": m.name or "?", "text": message_text(m)[:1500]}
                for m in recent_turns(result["messages"], keep=1)
                if isinstance(m, ToolMessage)
            ]
            error = None
        except Exception as exc:  # keep the run alive; the failure is reported, not hidden
            answer, tool_results, error = "", [], f"{type(exc).__name__}: {exc}"
            result = None

        if verbose and trace:
            # END is a sentinel, not a node: nothing ran there and nothing was written.
            # This is simply the state the turn finished holding.
            print(f"  [END    ] state: {_state_line(result or {})}")

        record = {
            "turn": turn["n"],
            "kind": turn.get("kind", "lookup"),
            "answer": answer,
            "tool_results": tool_results,
            "error": error,
            "judges_enabled": judge,
            "latency_s": round(time.perf_counter() - started, 2),
            **METER.turn_stats(),
        }
        record["metrics"] = run_deterministic(turn, record)
        if judge:
            record["metrics"].update(
                await run_judges(turn, record, session, session_evidence)
            )
        session_evidence.extend(evidence_of(record))
        record["score"] = mean_score(record["metrics"])
        records.append(record)

        if verbose and trace:
            print_turn_block(turn, record, len(SCHEMA_TOKENS))
        elif verbose:
            print(ROW.format(
                t=turn["n"],
                kind=turn.get("kind", "lookup")[:10],
                intent=(record.get("intent") or "—")[:20],
                inp=f"{record['input_tokens']:,}",
                out=f"{record['output_tokens']:,}",
                calls=record["model_calls"],
                tools=record["tools_exposed"],
                schema=f"{record['schema_tokens_est']:,}",
                score=f"{record['score']:.2f}" if record["score"] is not None else "—",
                sec=record["latency_s"],
                called=", ".join(record["tools_called"]) or ("ERROR" if error else "—"),
            ))

    if verbose and trace:
        scored = [r["score"] for r in records if r["score"] is not None]
        print()
        print(block_rule(f"{session['id']} · session total"))
        print(f"    {sum(r['input_tokens'] for r in records):,} in"
              f" · {sum(r['output_tokens'] for r in records):,} out"
              f" · {sum(r['model_calls'] for r in records)} model calls"
              f" · {sum(r['schema_tokens_est'] for r in records):,} schema tokens")
        print(f"    mean score {sum(scored) / len(scored):.2f}" if scored else "    mean score —",
              f"· {sum(r['latency_s'] for r in records):.1f}s")
    elif verbose:
        scored = [r["score"] for r in records if r["score"] is not None]
        print(ROW.format(
            t="", kind="", intent="TOTAL",
            inp=f"{sum(r['input_tokens'] for r in records):,}",
            out=f"{sum(r['output_tokens'] for r in records):,}",
            calls=sum(r["model_calls"] for r in records),
            tools="",
            schema=f"{sum(r['schema_tokens_est'] for r in records):,}",
            score=f"{sum(scored) / len(scored):.2f}" if scored else "—",
            sec=f"{sum(r['latency_s'] for r in records):.1f}",
            called="",
        ))
    return records


async def run_stage(build_graph, description: str, **build_options) -> None:
    """The standalone entry point shared by every stage.

    Every stage stays runnable on its own — `python graph.py --session S2` — so you
    can watch one policy behave before comparing it with the others.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--session", default="S2",
                        help="session id prefix to replay (default: S2). Use 'all' for every session.")
    parser.add_argument("--answers", action="store_true", help="print each answer in full")
    parser.add_argument("--judge", action="store_true",
                        help="also run the LLM judges (slower; evals/compare.py runs them by default)")
    # Declared here as well as read by tracing.enable_from_argv(): the same switch
    # turns on the per-node lines AND turns this summary into a readable block.
    parser.add_argument("--trace", action="store_true",
                        help="narrate every state change, and print each turn as a block")
    args = parser.parse_args()

    tools = await load_eval_tools()
    index_tools(tools)  # price each tool's schema once
    app = build_graph(tools, **build_options)

    for session in load_sessions(None if args.session == "all" else args.session):
        records = await run_session(app, session, judge=args.judge, trace=args.trace)
        if args.answers:
            for turn, record in zip(session["turns"], records):
                print(f"\n[{turn['n']}] You: {turn['user']}\n    Assistant: {record['answer']}")


def main(build_graph, description: str, **build_options) -> None:
    asyncio.run(run_stage(build_graph, description, **build_options))
