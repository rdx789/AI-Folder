<!-- Everything greyed out in this file is FOR YOU, the student, and is not part of
     the prompt. What is left in plain text IS the prompt.

     Read the prompt first. Then read "WHAT THIS PROMPT DOESN'T SAY" at the bottom —
     that is where the actual exercise lives — and fold your own answers into the
     prompt before you hand it to your agent. -->

# Build: a NovaOps assistant that chooses what to put in front of the model

`00-baseline/` is a working assistant. It answers correctly, and by turn twelve of a
support conversation it costs several times what turn one cost — because every call
re-sends the entire conversation and all ten tool schemas.

Build the version that doesn't, without making the assistant worse. One agent, three
capabilities:

- **it works out what each turn needs**, and sends only that;
- **it puts only that turn's tools** in front of the model;
- **it stops the conversation growing** without bound, however long the session runs.

**Provided, finished, and not to be edited:** `server/` (the NovaOps MCP server, ten
tools), `00-agent-shared/` (the prompt, model and tool seams), `00-baseline/` (your
starting point *and* your control — change it and you lose the thing you are measuring
against), and `evals/` (replays fixed sessions and scores what came back). Read them and
import from them.

Write it as `03-history-distillation/graph.py`, exposing `build_graph(tools)` the way
`00-baseline/graph.py` does, so the evaluation harness can load it beside the baseline:

```bash
python 03-history-distillation/graph.py --session S2 --trace
python evals/compare.py --stages 00,03 --no-judge
```

`--trace` makes every node print the state update it returned. Use it constantly.

<!-- ──────────────────────────────────────────────────────────────────────────
  WHAT THIS PROMPT DOESN'T SAY  — and what you have to decide before it will work

  The prompt above names the task. It does not say what "what this turn needs" is, how
  you know which tools that means, what each model call is allowed to see, what happens
  when a decision is wrong, or what survives when the messages are gone. Those are the
  exercise, and so are the TRADE-OFFS behind each. A blank where a decision should be
  is a wrong answer — write down what each choice buys and what it costs.

  ── Working out what the turn needs ─────────────────────────────────────────
  - What that "working out" produces. Something structured beats something you have to
    parse, and every field costs a model call to produce — so name the decision each
    field feeds. A field nothing routes on is paid for twice.
  - What the working-out call sees versus what the answering call sees. They have
    different jobs; giving them the same thing is the habit you are breaking.
  - How much raw conversation still has to go through verbatim, and what breaks at one
    turn less. Note where you are allowed to cut a message list at all.
  - What it costs: you just added a model call per turn. Find the session where that
    makes things worse, and be able to say why you would still ship it.

  ── Deciding which tools exist this turn ────────────────────────────────────
  - How a turn's tools get chosen. A table, a model call, both? Say who maintains it
    and what happens when it drifts.
  - What "uncertain" means operationally, and which way you fail when it happens.
    Reads and writes may not deserve the same answer.
  - Whether a turn can have zero tools, and how you would know.
  - How you detect that you narrowed too far, and what you do about it. Nothing errors
    when this goes wrong — you get a confident answer built on nothing.

  ── Not growing forever ─────────────────────────────────────────────────────
  - When to summarise, and what triggers it. Whatever you check must be cheap on the
    turns where it does not fire.
  - What the summary is SHAPED like. A paragraph and a set of named fields fail
    differently, and one of those failures loses a constraint the user stated.
  - How you know the summary did not silently drop something, and what you do when it
    did.
  - What the summariser reads, and what the turn-planning call reads afterwards. If the
    second one still grows with the session, you have not solved the problem.
  - Whether you delete anything. Deleting bounds storage and costs you every fallback
    you had; not deleting is cheap and bounds nothing.

  ── How to work ─────────────────────────────────────────────────────────────
  - Server in one terminal, your build in another.
  - The deliverable is ONE agent, but do not try to write it in one go. Get the
    turn-planning working and measure it; then add tool selection and measure again;
    then add the memory. Each capability is separately visible in --trace, so you
    always know which one you just broke.
  - Three capabilities is more than one sitting. One you can measure and defend beats
    three you cannot.
  - Write your answers to the questions above down as a DOCUMENT, not only as edits to
    this prompt. A context policy that exists only as code is a policy nobody can
    review, and reviewing it is how you find the constraint you forgot to keep. Keep it
    with your project (docs/context-policy.md is fine) - your final project needs the
    same document, for the same reasons.
  - Before you implement any of it, write down what you expect each change to do to
    tokens AND to quality. Then run the evals and find out. In this lab planning cost
    more in every full run, and policies that were close on tidy input separated
    sharply on messy input - neither was assumed in advance, both showed up because
    the numbers were collected before the result was interpreted.
─────────────────────────────────────────────────────────────────────────── -->

A. Shared Infrastructure (
00-agent-shared/
)
agent.py
: Provider seams shared across every stage so that benchmarks only test context policies:
Fixed ASSISTANT_PROMPT (NovaOps IT/HR support assistant rules).
get_model() (ChatBedrockConverse with Amazon Nova 2 Lite at temperature=0).
load_tools() (connects to FastMCP client).
invoke_answer() & invoke_structured() (calls the model and records token meters).
tracing.py
: Study tool enabled by --trace. Narrates state updates at every node (appends vs overwrites vs deletes).
B. The Two Stages
1. Stage 00 — Baseline (
00-baseline/graph.py
)
Concept: State = Context.
AgentState: Only messages: Annotated[list, add_messages].
Flow: START -> model -> (tools -> model | END).
Problem: Every past message, every raw tool output, and all 10 tool schemas are re-sent on every single model call. Tool schemas alone consume ~33% of input tokens on short turns.
2. Stage 03 — One agent, three capabilities (
03-history-distillation/graph.py
)
Context planning — State ≠ Context. The state retains full history, but two projections are generated:
planner_messages(state): Planner sees memory + the undistilled transcript flattened to text (tool outputs capped at 400 chars) with zero tool schemas. Produces a structured TurnPlan (current_intent, relevant_facts, relevant_constraints, requires_tools).
answer_messages(state, plan): Answer model receives the rendered plan + only the last 2 turns (recent_turns sliced at HumanMessage boundaries to keep tool call/result pairs intact).
Dynamic tool loadout — expose only the tools needed for the turn's intent:
AgentState adds loadout: list[str] and rearmed: bool.
Deterministic select node: no model call. Maps TurnPlan.current_intent against a static LOADOUTS dictionary.
Write gate: create_access_request is isolated in WRITE_TOOLS. It is only released if plan.current_intent == "access_request" AND plan.action_confirmed == True.
rearm fallback: if the planner claimed requires_tools = True but the model called no tools, the router routes to rearm, which re-opens all read tools, uses RemoveMessage to delete the ungrounded draft response, sets rearmed = True, and retries.
History distillation — bound the planner's context on long sessions:
AgentState adds memory: dict and distilled_upto: int.
Structured memory (ConversationMemory): not prose, but 5 named fields: current_intent, important_facts, active_constraints, decisions, and unresolved_items.
Trigger (needs_distillation): fires when the undistilled history grows past a token budget.
Safety check (memory_is_safe): validates that constraints weren't dropped and memory isn't empty. If invalid, the distillation is rejected and raw history is retained (the planner reads the raw history instead).
Planner boundary: the planner sees memory + messages[distilled_upto:], capping context growth regardless of session length.

build in this one agent in stages and let me check and aprove it
