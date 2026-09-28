"""
Token accounting for evaluation and reporting.

Every claim this lab makes ("planning costs less than it saves", "tool schemas are
a third of your input on a short turn") is a token claim, so the measurement has to
be honest about what is exact and what is estimated:

  EXACT      input/output tokens, from Bedrock's own usage metadata on every
             response. This is what you are billed for.
  ESTIMATED  the share of those input tokens spent on TOOL SCHEMAS. The Converse
             API bills one input number for the whole request and does not break
             out the toolConfig block, so we estimate it at ~4 characters per
             token over the serialized schemas. Treat it as an order of magnitude,
             not an invoice — it is labelled "est." everywhere it is printed.

Usage: the graphs call METER.record(...) after every model call, and the runner
calls METER.reset() before each turn and METER.turn_stats() after it.
"""

import json
from dataclasses import dataclass, field

# Filled in once by index_tools(); maps tool name -> estimated schema tokens.
SCHEMA_TOKENS: dict[str, int] = {}


def estimate_tokens(text: str) -> int:
    """Rough token count for text we cannot get an exact number for. ~4 chars/token."""
    return len(text) // 4


def index_tools(tools: list) -> None:
    """Measure each tool's schema once, so exposing it can be priced later.

    What the model actually receives per tool is its name, its description (the
    docstring you wrote on the server) and its JSON argument schema. Long,
    helpful docstrings are good for tool selection and expensive in context —
    that trade-off is real, and this is where you can see it.
    """
    SCHEMA_TOKENS.clear()
    for tool in tools:
        payload = json.dumps(
            {"name": tool.name, "description": tool.description, "schema": tool.args}
        )
        SCHEMA_TOKENS[tool.name] = estimate_tokens(payload)


@dataclass
class Call:
    """One model call: what it cost, and what tools were on the table for it."""

    kind: str  # "answer" | "plan" | "distill" — so overhead can be split out
    input_tokens: int
    output_tokens: int
    tools_exposed: list[str]
    tool_calls: list[str]
    # The full trajectory: {name, args} per call, in order. Names alone cannot answer
    # "were the arguments usable?" or "did the write fire before the lookup?", which
    # are two of the failures worth catching — so we keep the arguments and the order.
    trajectory: list[dict]

    @property
    def schema_tokens(self) -> int:
        return sum(SCHEMA_TOKENS.get(name, 0) for name in self.tools_exposed)


@dataclass
class Meter:
    """Accumulates the calls made during ONE conversational turn."""

    calls: list[Call] = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    def reset(self) -> None:
        self.calls.clear()
        self.extras.clear()

    def note(self, key: str, value) -> None:
        """Record something a stage wants reported (intent, fallback re-arms, ...)."""
        self.extras[key] = value

    def bump(self, key: str, amount: int = 1) -> None:
        self.extras[key] = self.extras.get(key, 0) + amount

    def record(self, response, kind: str = "answer", exposed: list | None = None) -> None:
        """Log one model call from its response message.

        `response.usage_metadata` is LangChain's normalized view of the provider's
        token report — for Bedrock that's the `usage` block of the Converse
        response. It is absent only when a provider doesn't report usage, hence
        the `or {}`.
        """
        usage = getattr(response, "usage_metadata", None) or {}
        # Only an "answer" call's tool_calls are real tool use. with_structured_output
        # is implemented AS a tool call under the hood, so a planner response carries a
        # tool_call named after its schema — counting it would report the planner as
        # having used a NovaOps tool, which it never does.
        raw_calls = getattr(response, "tool_calls", None) or [] if kind == "answer" else []
        self.calls.append(
            Call(
                kind=kind,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                tools_exposed=list(exposed or []),
                tool_calls=[c["name"] for c in raw_calls],
                trajectory=[{"name": c["name"], "args": c.get("args", {})}
                            for c in raw_calls],
            )
        )

    def turn_stats(self) -> dict:
        """Everything measured for the turn that just finished."""
        answer_calls = [c for c in self.calls if c.kind == "answer"]
        overhead_calls = [c for c in self.calls if c.kind != "answer"]
        return {
            "input_tokens": sum(c.input_tokens for c in self.calls),
            "output_tokens": sum(c.output_tokens for c in self.calls),
            "total_tokens": sum(c.input_tokens + c.output_tokens for c in self.calls),
            "model_calls": len(self.calls),
            # Overhead = every token spent on planning/distilling rather than on
            # answering. This is the number that decides whether the extra stage
            # paid for itself.
            "overhead_tokens": sum(c.input_tokens + c.output_tokens for c in overhead_calls),
            "answer_input_tokens": sum(c.input_tokens for c in answer_calls),
            "schema_tokens_est": sum(c.schema_tokens for c in self.calls),
            "tools_exposed": max((len(c.tools_exposed) for c in self.calls), default=0),
            "tools_called": [name for c in self.calls for name in c.tool_calls],
            # Utilization: of the tools we paid to expose, how many did the model
            # actually reach for? A turn that exposes ten and uses one is paying nine
            # schemas for nothing — which is the whole argument for a per-turn tool loadout.
            "tool_utilization": round(
                len({n for c in self.calls for n in c.tool_calls})
                / max((len(c.tools_exposed) for c in self.calls), default=0), 3
            ) if any(c.tools_exposed for c in self.calls) else None,
            "trajectory": [step for c in self.calls for step in c.trajectory],
            **self.extras,
        }


METER = Meter()
