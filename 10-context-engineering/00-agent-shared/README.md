# Shared agent mechanics

Read this folder before Stage 00. [`agent.py`](agent.py) holds the integration seams
every stage shares, unchanged between them:

1. `ASSISTANT_PROMPT` — the same task prompt for every stage.
2. `get_model` and `load_tools` — the Bedrock and MCP integration seams.
3. `invoke_answer` and `invoke_structured` — the two ways a stage calls the model.
   Each one invokes, prices the call against the meter it is handed, and reports what
   the call received. The meter is a parameter rather than an import, so this folder
   depends on nothing in `evals/`.
4. `message_text` — normalizes Bedrock message content to plain text.

[`tracing.py`](tracing.py) is the second module here, and it is a study aid rather
than part of the agent. Run any stage with `--trace` and every node prints the update
it returned, so AgentState's growth — one field at Stage 00, seven at Stage 03 — is
something you watch rather than infer:

```bash
# the baseline: state and context are the same thing, on every single call
python 00-baseline/graph.py --session S1 --trace

# stage 03: the same session, with the plan, the loadout and the memory moving
python 03-history-distillation/graph.py --session S1 --trace
```

Tracing is off unless `--trace` is passed, and `evals/compare.py` never passes it —
so the measured graph is the untraced one.

Stage 00 is provided and runnable; stage 03 is the one built on top of it.

Context **policy** does not live here. Turn windows, transcript rendering, tool-use
checks, planning, loadouts, routing, and memory remain in the stage folders where
students can see why each capability exists.

Evaluation also does not live here or import this module. Datasets, runtime helpers,
meters, checks, judges, replay, and reports are introduced later under
[`evals/`](../evals/README.md).
