# Maya graph model: nodes and routes

How one conversation turn moves through Maya's LangGraph workflow, as built by
`build_graph()` and routed by `route()` in [`code/graph.py`](code/graph.py).

```
                        MayaGraph.ainvoke({"newest_message": …}, config)
                                         │
                ┌────────────────────────┴─────────────────────────┐
                │ GATE (before the graph)                          │
                │ • caller from config, valid?   • subject E001 allowed?
                │ • thread_id present?           • same caller as this thread?
                └───────┬──────────────────────────────────┬───────┘
                   no ──┘                                  └── yes  (turns of one thread run one at a time)
                   ▼                                            ▼
           status=denied                              START
           (no node runs,                               │
            nothing read)                               ▼
                                  ┌──────────────────────────────────────────┐
                                  │ 1 plan                                   │
                                  │ pin facts (start date, Israel, hybrid,   │
                                  │ Q3 freeze) · close the Rachel tangent ·  │
                                  │ fold memory if long · LLM planner →      │
                                  │ ContextPlan · app overrides: recall /    │
                                  │ explicit source / full checklist /       │
                                  │ Webex filing                             │
                                  └────────────────────┬─────────────────────┘
                                                       ▼
                                  ┌──────────────────────────────────────────┐
                                  │ 2 retrieve  (skipped if not needed)      │
                                  │ ACL inside k-NN → 20 → rerank → 4        │
                                  │ + ≤1 refinement for missing named docs   │
                                  └────────────────────┬─────────────────────┘
                                                       ▼
                                  ┌──────────────────────────────────────────┐
                                  │ 3 select                                 │
                                  │ read-tool loadout for the intent,        │
                                  │ narrowed to the reads this turn needs    │
                                  │ (recall → zero tools)                    │
                                  └────────────────────┬─────────────────────┘
                                                       ▼
            ┌───────────────────────────────▶ ┌──────────────────────────────────────────┐
            │                                 │ 4 model                                  │
            │                                 │ phase 1: required reads (app-resolved)   │
            │                                 │ phase 2: pick candidate_ids from the     │
            │   rearm (once):                 │ cited catalogue (no free text)           │
            │   reads planned but none ran →  │ shortcuts: greeting → empty;             │
            │   retry with full READ-ONLY     │ supported Webex filing → no LLM call     │
            │   catalogue (never a write)     └───────┬───────────┬──────────────┬───────┘
            │                                         │           │              │
            └──────────────── "rearm" ────────────────┘    "tools"│     "finalize"│
                                                                  │  (errors, or  │
                                    route(state):                 │  no more reads)
                                    1. errors        → finalize   │              │
                                    2. tool calls &  → tools      ▼              │
                                       rounds < 2               ┌────────────────────────────────┐
                                    3. can rearm     → rearm    │ 5 tools                        │
                                    4. otherwise     → finalize │ validate whole batch (loadout, │
                                                                │ args, employee scope) → run    │
                                         ┌──────────────────────│ reads → Evidence snapshots     │
                                         │  back to model       └────────────────────────────────┘
                                         ▼
                                       (4 model)                                     │
                                                                                     ▼
                                  ┌──────────────────────────────────────────────────────────┐
                                  │ 6 finalize                                               │
                                  │ validate every task (exact quote, cited, status allowed) │
                                  │ precedence/conflicts → OnboardingChecklist · unresolved  │
                                  │ items · for an explicit supported Webex request:         │
                                  │ checkpoint a dispatch record BEFORE calling out          │
                                  └────────────────────────────┬─────────────────────────────┘
                                                               ▼
                                  ┌──────────────────────────────────────────────────────────┐
                                  │ 7 handoff                                                │
                                  │ only if a new dispatch was reserved: WebexPort           │
                                  │ .request_access() → PendingAccessResult (pending)        │
                                  │ replay of the same request → saved result, no new call   │
                                  └────────────────────────────┬─────────────────────────────┘
                                                               ▼
                                                              END → result["response"]
```

## Routes in practice

Step counts are from the S2 runs; the hard cap is `RECURSION_LIMIT = 12`.

| Route | Path | Steps | Example |
|---|---|---|---|
| Denied | gate only | 0 | `E002/UG_REGULAR` asking for Maya's offer letter |
| Documents only / recall / Webex filing | plan → retrieve → select → model → finalize → handoff | 6 | turn 2 offer letter, turns 7 and 12 recall, turn 10 filing (handoff calls the port) |
| One read round | … → model → **tools** → model → finalize → handoff | 8 | turn 1 employee record, turn 3 checklist, turn 8 Webex seats |
| Two read rounds | … model → tools → model → **tools** → model → … | 10 | the one-turn full checklist (employee + tasks, then the subscription named by the blocked task) |
| Rearm | … model —rearm→ model → tools → model → … | ≤ 12 | the planner wanted reads but the model requested none (covered by tests; not seen in S2) |

## What the shape guarantees

- **One way out.** Only the `handoff` node reaches outside the agent, and only for an
  explicit, already-evidenced Webex request. The dispatch is checkpointed before the call,
  so replaying turn 10 returns the saved pending result instead of calling twice.
- **No model-written claims.** The model only chooses ids from a catalogue of exact, cited
  statements; `finalize` re-validates every task against the cached evidence.
- **Bounded retries.** At most two read rounds and one read-only rearm, so every turn stays
  within 12 graph steps (10 at most in practice).
- **Access first.** An unauthorised caller is stopped at the gate, before planning,
  retrieval, reads or handoff; the access filter is also inside every OpenSearch query.
