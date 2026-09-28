# Stage 03 — history distillation (the agent)

One agent that combines three capabilities: context planning, dynamic tool loadout and
history distillation. The design, and why each choice was made, is in
[`../docs/context-policy.md`](../docs/context-policy.md). This page is the short version.

```bash
python server/server.py                                    # terminal 1
python 03-history-distillation/graph.py --session S2 --trace
python evals/compare.py --stages 00,03                     # add --no-judge for a fast run
python -m unittest discover -s tests -v                    # 45 offline tests, no server or model
```

## What's there

`graph.py`: `START -> distill -> plan | skip_plan -> select -> model <-> tools / rearm -> END`

- **distill**: when the planner's undistilled tail passes ~1,000 est. tokens, it
  folds older whole turns (down to ~500) into a five-field `ConversationMemory`. A fold
  is rejected if it drops a constraint, an id, a date or a number. A rejected or failed
  fold keeps the raw history and never fails the turn.
- **plan**: a structured `TurnPlan` (`current_intent`, `also_needs`, `requires_tools`,
  facts, constraints, `action_confirmed`). The planner reads memory plus the raw tail
  only, with no tool schemas. Bare acknowledgements ("thanks", "ok") skip it
  (`skip_plan`).
- **select**: binds only the tools in the intent's `LOADOUTS` row, unioned with the
  `also_needs` row. A tool that needs an employee id brings `get_employee` along.
- **model / rearm**: the answer model sees the plan, the pinned employee ids and a
  two-turn window. If a lookup was planned but none was made, `rearm` reopens all read
  tools and retries once. After 3 tool rounds the model must answer without tools.
- **Write gate**: `create_access_request` is bound only after `authorize_plan`
  re-decides from the user's own messages. The authorising words must be in the newest
  message, and any doubt denies.

## Results (mean of three judged full sweeps, 2026-09-27)

| | vs baseline |
|---|---|
| outcome quality | about +1.2 points |
| total tokens | about −0.5% |
| tool-schema tokens | −77% (sweep `191722`) |
| tool arguments | 0.95–1.00 (baseline 0.80–0.90) |
| faithfulness | 0.88–0.97 (baseline 0.82–0.86) |
| completeness | 0.72–0.79 (baseline 0.83–0.89): the weakest metric |

The long session (S2) is 12–22% cheaper. On short sessions the planner costs more than
it saves. Single runs swing 5–10 quality points, so compare means, not one run.

## Known limits (details in the policy doc, section 6)

- **S5 (noisy inbox):** quality varies a lot from run to run.
- **S3 turn 8, "who was in my first question?":** answers with the most recent person
  instead of the first.
- **S2 turn 9:** still answers the Finance-approval question from memory in 2 of 6
  runs.
- **Planner overhead:** a fixed ~2.2k tokens per turn, which short sessions can't
  recover.
