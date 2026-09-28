# Evals and reports

Cover this folder after the two agent stages (00 and 03). It contains measurement and replay
infrastructure, not agent behavior.

## Read in this order

1. [`data/sessions.json`](data/sessions.json) — prepared conversations and expected behavior.
2. [`checks.py`](checks.py) — deterministic checks over tool trajectories and facts.
3. [`judges.py`](judges.py) — LLM judges for answer qualities that code cannot assert.
4. [`evalrules.py`](evalrules.py) — which checks and judges apply to each turn kind.
5. [`runner.py`](runner.py) — replays one session and records per-turn evidence.
6. [`compare.py`](compare.py) — compares stages and produces the report artifacts.

[`runtime.py`](runtime.py) contains eval-owned copies of the provider, MCP, and
message-reading helpers. This deliberate duplication keeps the eval package from
importing the student-facing code in `00-agent-shared`.

[`EVALS.md`](EVALS.md) explains every metric, the weighted quality score, task
success, and the hard safety/runtime gates.

## Run

From the `code/` directory:

```bash
python evals/compare.py --session S2
python evals/compare.py
```

Reports are written to `evals/results/`.
