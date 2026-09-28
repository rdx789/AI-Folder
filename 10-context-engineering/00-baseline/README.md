# Stage 00 — baseline

The control implementation: no context policy at all. Everything the conversation
has ever contained goes into every model call, alongside all ten tool schemas.

One file, not two. Every later stage splits into `graph.py` (LangGraph mechanics)
plus `stageNN_policy.py` (the context policy) — here there is no policy to separate
out, which is exactly what makes this the baseline.

## Read the code in this order

1. [`AgentState`](graph.py) — the full persisted state is one growing message list.
2. `call_model` — the whole list is sent to the model, with every tool bound.
3. `should_continue` — a tool call enters the tool loop; otherwise the turn ends.
4. The graph wiring — five lines show the entire control flow.

The key problem is visible in `call_model`: **state and model context are the same
thing**. Every old message and tool result is sent again.

## Run

```bash
# the starting point you were given — twelve turns, nothing ever thrown away
python graph.py --session S2 --trace
```
