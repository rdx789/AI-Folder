# Homework — Agentic Frameworks

Task 1 is a one-time environment fix; tasks 2–4 each push one architecture further and
produce a piece you can reuse in the NovaOps final project. They're useful project
extensions, not prerequisites for finishing it. The prompt hint drives a coding agent
through each one — copy it, or write your own. Keep the shared server running (`cd server
&& python server.py`).

## 1. Feel the version wall — what a framework costs you in dependency freedom

**Goal.** Before any orchestration, the lesson frameworks teach the hard way: adopting a
framework means inheriting its dependency lag. This one isn't hypothetical and it isn't
a bug you can route around — it's the reason every MCP lab in this course is written
against the **1.x** SDK.

**The situation.** The MCP SDK shipped a **2.0** — a new server API (`MCPServer` replacing
`FastMCP`) and a high-level `Client`. LangChain's MCP bridge, `langchain-mcp-adapters`,
hasn't caught up: it still imports `RequestContext` from `mcp.shared.context`, which 2.0
removed. So the course sits on 1.x — not because 1.x is better, but because the framework
can't follow. `requirements.txt` pins `mcp<2` and `setup.sh` installs it, which is why
everything just worked for you.

**What to do.**

1. **See it once.** Break the pin, run any exercise, and read the failure:

   ```bash
   pip install "mcp>=2"
   cd 02-langgraph-basics && python graph.py     # dies before your code runs
   ```

   ```
   ImportError: cannot import name 'RequestContext' from 'mcp.shared.context'
   ```

   Nothing you wrote is wrong. Restore with `pip install "mcp<2"`. It's the error you'll
   hit for real the first time you adopt a framework in anger.

2. **Now try to actually upgrade** — port this lesson's `server/server.py` **forward** to
   the 2.0 API and watch the wall from the other side. On mcp 2.x your upgraded server
   runs fine on its own; the four exercises still can't reach it. Downgrade back to 1.x
   and the exercises run but your upgraded server won't import. One virtualenv, two
   mutually exclusive halves — that's the wall, not a workaround away.

   **Prompt hint.**
   > "Port `server/server.py` from the mcp 1.x FastMCP API to the 2.0 API: `from mcp.server
   > import MCPServer`, `MCPServer(name)`, and `mcp.run(transport='streamable-http',
   > host=..., port=...)` instead of `FastMCP(name, host=..., port=...)` +
   > `mcp.run(transport='streamable-http')`. Keep every `@mcp.tool()` and its docstring
   > exactly as-is. Then, with `pip install \"mcp>=2\"`, start it and confirm it serves;
   > then run `02-langgraph-basics/graph.py` against it and show what happens. Then
   > `pip install \"mcp<2\"` and show what happens to each of the two."

Sit with the result: you gave up a whole major version of a dependency — permanently, for
as long as you use this framework — in exchange for orchestration you could have written
yourself. Direct SDK code (Lesson 8) had no such constraint. A pin in a `requirements.txt`
is a cheap fix here because it's one course; on a real system, *"we can't upgrade the SDK
until our agent framework catches up"* is a roadmap item, and it's the question to ask
**before** you adopt the framework, not after.

> **Timeliness (August 2026).** All of this holds only while `langchain-mcp-adapters` lacks
> mcp 2.0 support. Once it ships, the pin goes and this task disappears — check the
> adapter's changelog before assuming it's still true.

## 2. Give the supervisor a reflection loop

**Goal.** Turn Exercise 4's one-shot supervisor into a *reasoning* one that checks
whether it has enough to answer and gathers more if not — the orchestration pattern
behind the final project's **flagship workflow** (the assistant that keeps working a
request until it's resolved).

**What to build.** After the specialists report, add a **review** node that inspects the
merged findings and decides exactly one of two things: `enough` (go on to `finalize`) or
`need_more`. On `need_more`, it writes a **refined follow-up request** and runs the **same
specialists you originally selected** a second time with that sharper ask — you don't need
to pick a different team, just give the same team a better question. Add only the minimal
state this needs: a `round` counter and a `follow_up_request` string. Cap it at **two
rounds total** so the graph always terminates — a reflection loop with no ceiling is how
you get an agent that spins forever. Prove it with a deliberately under-specified request
that needs the second pass, and a well-specified one that finishes in the first.

**Prompt hint.**
> "In `04-parallel-supervisor/graph.py`, add a `review` node after `worker` that uses the
> model to decide `enough` vs `need_more`, and add `round` and `follow_up_request` to the
> state. On `enough`, go to `finalize`. On `need_more` — and only while `round` < 2 —
> increment `round`, write a refined `follow_up_request`, and re-dispatch the **same**
> specialists selected in round one with that follow-up. Show an under-specified request
> that triggers a second round and a clear one that finishes in the first."

## 3. Make approvals survive a restart

**Goal.** Exercise 3 pauses in memory — kill the process and the pending approval is gone.
Make the pause **durable**, which is what the final project's **human-approval gate** needs
to be a real workflow rather than a demo.

**What to build.** Swap `InMemorySaver` for a persistent checkpointer. The graph is invoked
asynchronously (`ainvoke`), so use the async saver — **`AsyncSqliteSaver`** (from
`langgraph.checkpoint.sqlite.aio`; you may need to `pip install langgraph-checkpoint-sqlite`
first). Drive it from **one script with two CLI modes**: `start` runs until the interrupt
and exits; `resume` reopens the same on-disk checkpoint and continues with the approval.
Run `start`, let it interrupt, **exit the program**, then run `resume` in a **fresh
process** against the *same* `thread_id` and watch it complete. That's a workflow that
outlives the request — and the process — that started it.

**Prompt hint.**
> "In `03-human-in-the-loop/graph.py`, replace `InMemorySaver` with `AsyncSqliteSaver`
> (`from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver`; `pip install
> langgraph-checkpoint-sqlite` if it's missing) writing to a local file, keyed by a fixed
> `thread_id`. Give it a single entry point with `start` and `resume` CLI modes: `start`
> runs until the interrupt and exits; `resume` reopens the same checkpoint and resumes with
> an approval. Show `start` in one process, then `resume` completing it from a new process."

## 4. A router for the whole assistant

**Goal.** Every exercise assumes the Webex access case. Build the layer above it: a
**top-level router** that reads any NovaOps request and sends it to the one handler that
fits (policy Q&A, IT ticket, access request, employee lookup). This is the assistant's
**top-level router** in the final project.

**What to build.** A LangGraph graph with a **`classify`** node that uses structured output
to label the incoming request as one of `policy_qa`, `it_ticket`, `access_request`,
`lookup`, or a general fallback, and a **conditional edge** that routes it to exactly one
matching handler. Unlike Exercise 4, there's no fan-out and no merging here — one request,
one handler; the lesson is clean single-path routing, not orchestration. A handler can be a
focused `create_agent` (given only the tools that category needs) or a plain deterministic
node when an agent is overkill. Test it on several different NovaOps requests and show each
reaching a different handler.

**Prompt hint.**
> "Build a top-level router graph (a new file): a `classify` node uses `with_structured_output`
> to label the request as `policy_qa` / `it_ticket` / `access_request` / `lookup` /
> `general`, and a conditional edge routes to exactly one handler. Each handler is either a
> focused `create_agent` over the relevant Lesson 8 tools or a simple deterministic node —
> no `Send`, no reducers, no parallel dispatch. Show it routing 'what's the expense
> policy?', 'I need Webex access', and 'who is E010?' to three different handlers."

## Optional stretch goals

Once the four tasks work, any of these push a concept further:

- **Reflection picks a fresh team.** Let Task 2's `review` step choose a *different* worker
  set for the second round, instead of re-running the same specialists — dynamic selection
  driven by what the first round was missing.
- **A new specialist.** Add another specialist agent to Exercise 4's parallel supervisor
  (say, an assets or tickets specialist) and confirm the planner only dispatches it when
  the request is relevant.
- **A fan-out router.** Allow Task 4's router to dispatch **multiple** independent handlers
  when a request spans categories, then merge their answers — the point where a router
  starts to look like a supervisor again.
- **A stateful sub-agent.** Convert one Exercise 4 worker from a `create_agent` into a
  hand-built stateful LangGraph subgraph, and run it as a node inside the supervisor.
