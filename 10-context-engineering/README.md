# Lesson 10 — Context Engineering

A NovaOps IT/HR support assistant built from a spec (`PROMPT.md`) as **one agent** on
top of a baseline, and measured against that baseline with a fixed eval harness.

| Folder                     | What it is                                                                                   |
| -------------------------- | -------------------------------------------------------------------------------------------- |
| `00-baseline/`             | the control: a working agent with **no** context policy. Every call re-sends everything      |
| `03-history-distillation/` | the agent: context planning + dynamic tool loadout + history distillation ([README](03-history-distillation/README.md)) |
| `docs/context-policy.md`   | the design: every decision, its trade-off, what was rejected and the measured effect         |
| `tests/`                   | offline tests for stage 03 (no server, no model)                                             |
| `00-agent-shared/`         | one prompt, one model, one tracing helper, held identical so a difference between the stages can only come from the context policy |
| `evals/`                   | replays fixed sessions against both stages and scores them                                   |
| `server/` + `data/`        | the NovaOps MCP server and its ten tools                                                     |

## 1. Setup

You need a `.env` (see `.env.example`) with `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`,
`AWS_REGION` (e.g. `us-east-1`), `BEDROCK_MODEL_ID` (`us.amazon.nova-2-lite-v1:0`) and
`BEDROCK_EMBEDDING_MODEL_ID` (`amazon.titan-embed-text-v2:0`). Then:

```bash
source setup.sh                  # venv + deps + Bedrock smoke test
```

> On Windows, use Git Bash (easiest via VS Code's integrated terminal), or `setup.ps1`.

## 2. Run

Two terminals, always: the server in one, the agent in the other.

```bash
# terminal 1
python server/server.py
```

```bash
# terminal 2
python 00-baseline/graph.py --session S2 --trace              # the control
python 03-history-distillation/graph.py --session S2 --trace  # the agent
```

Watch the `context = ...` line. On the baseline both message counts are equal on every
call and the schema count stays at 10 of 10. On stage 03 the answer model sees a short
window and only the turn's tools, and the planner's input stays flat once distillation
starts.

## 3. Measure

```bash
python evals/compare.py --stages 00,03 --no-judge   # fast, no LLM grading noise
python evals/compare.py --stages 00,03              # with judges: quality scores
python -m unittest discover -s tests -v             # offline tests
```

Results are written to `evals/results/`. Single runs swing 5–10 quality points with
identical code, so compare repeated runs, not one. How to read the numbers is in
[`evals/EVALS.md`](evals/EVALS.md); the measured results and known limits are in
[`docs/context-policy.md`](docs/context-policy.md).
