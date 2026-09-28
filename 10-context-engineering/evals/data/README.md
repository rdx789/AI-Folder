# The dataset

Eight prepared NovaOps sessions — 54 turns in total — replayed identically against
every stage. Fixing the input is the only reason the token numbers mean anything.

| Session                   | Turns | What it is there to expose                                                       |
| ------------------------- | ----- | -------------------------------------------------------------------------------- |
| `S1-quick-policy`         | 4     | The control. Short and cheap, so a planner has nothing to save and still costs   |
| `S2-onboarding-maya`      | 12    | The hero. Early constraints, a dead branch, no-tool turns, RAG, records, a write |
| `S3-intent-shift`         | 9     | A resolved incident, then a hard pivot to a different person                     |
| `S4-constraint-carry`     | 9     | Four near-identical tool results, and a rule from turn 1 that governs turn 8     |
| `S5-noisy-inbox`          | 8     | Noise and retrieval pressure — see below                                         |
| `S6-boundary-clean`       | 4     | Clean write boundary: constrain, retrieve, draft, explicitly authorize           |
| `S7-boundary-noisy`       | 4     | The same four outcomes with moderate forwarding, typo and aside noise            |
| `S8-boundary-heavy-noise` | 4     | The same outcomes under nested threads, stale instructions and heavy noise       |

## S6/S7/S8 are a controlled noise curve

The three sessions have identical expected intents, tools, facts and workflow stages.
Only the presentation changes. S6 is clean, S7 adds moderate noise, and S8 adds nested
forwards, stale quoted instructions, competing names, boilerplate and interruptions.
[`compare.py`](../compare.py) reports every level against the clean control, so the
result is a noise curve rather than a comparison between unrelated tasks.

The curve also adds authorization-boundary cases: “draft words only” must not write,
while a later explicit “go ahead” must write. The do-not-file constraint is introduced,
retained and then satisfied across the four turns.

## S5 exists because the other four are too clean

Sessions 1–4 are written in tidy sentences that say exactly one thing. Real requests
do not arrive that way, and a context policy tuned only against tidy input will look
better than it is. S5 supplies what the others left out:

- **A pasted email**, signature block and confidentiality footer included. About 180
  words, of which roughly 15 are the request.
- **Typos and sloppy phrasing** — `whats the proces for vpn accesss agian`.
- **An irrelevant aside** ("the coffee machine is broken again") wrapped around a
  real question.
- **A genuinely ambiguous back-reference** — "that other thing I mentioned earlier"
  could be any of three topics. The correct behaviour is to ask, not to guess, and
  no tool can resolve it.
- **Four knowledge-base searches.** Each returns ~700 tokens of article text, so by
  turn 5 the baseline is carrying ~2,000 tokens of retrieved documents forward on
  every call — most of it about topics nobody is asking about any more. This is the
  disease the other sessions barely exercise.
- **A write-gate trap.** The last turn says "write it all up for the ticket". That is
  composing text, not authorising a filing — and an over-eager gate reads it as
  permission and files a request nobody asked for.

## Turn expectations

Every turn carries what a correct handling of it looks like. The fields are read by
[`checks.py`](../checks.py), [`judges.py`](../judges.py), and routed by
[`evalrules.py`](../evalrules.py).

- `expected_intent` — the plan's classification. `null` means "don't check": some
  turns are genuinely ambiguous, and testing the dataset author's taste is not
  testing the system.
- `should_use_tools` — whether *any* tool call is correct for this turn.
- `expected_tools` — tools that must appear among those called.
- `forbidden_tools` — tools that must not.
- `required_facts` — load-bearing text that must appear in the answer.
- `note` — why the turn exists. Read these; they are the map of the dataset.
- **`kind`** — `retrieval` · `lookup` · `reasoning` · `action` · `summary` · `trivial`.
  This decides **which evals run on the turn**; the routing table is
  [`evalrules.py`](../evalrules.py) and the reasoning is in
  [`EVALS.md`](../EVALS.md).
- `optional_tools` — reasonable but not required. Keeps `tool_necessity` from
  punishing ordinary caution.
- `arg_checks` — `{tool, arg, equals|contains}`. Right tool, right arguments: the
  failure a tool *name* cannot expose.
- `sequence` — tools that must appear in this relative order. Declared on the four
  write turns, where "did the lookup precede the write?" is the whole question.

Two metrics fire regardless of what a turn declares, because they are wrong on any
turn:

- `call_efficiency` — an identical repeat call is pure waste, and models make them
  more often than you would expect, usually when the first result is buried further
  up a long context.
- **faithfulness** — any turn that actually retrieved gets graded against what the
  tools returned, expected or not. Nobody writes a test case for the search they did
  not anticipate, and that is exactly the one worth grading.

**Alternatives:** any expectation string may contain `|`, meaning any one of the
alternatives satisfies it — `"2026-08-01|August 1|Aug 1"`, `"get_policy|search_knowledge_base"`.
This keeps the checks blunt but honest: they match ids, numbers and proper nouns,
so a paraphrase cannot game them and a different wording cannot fail them.

## Editing it

Add turns freely — everything downstream reads the file, nothing hard-codes a turn
count. Two rules worth keeping:

1. **Expectations must be satisfiable from the shipped data.** Check `data/` before
   asserting a number. An expectation the data cannot support makes every stage
   look equally broken, which tells you nothing about any of them.
2. **A turn that only ever passes is not paying rent.** The useful turns are the
   ones some stages fail — that is where the design differences show up.
