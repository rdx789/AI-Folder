# The eval suite

Thirteen metrics, split the way Lessons 7 and 8 split theirs: **assert what you can,
judge only what you cannot.** Every metric returns `(score, reason)` in 0.0–1.0, so
they can be inspected together and a failure always says why.

The routing — which metric runs on which turn — lives in
[`evalrules.py`](evalrules.py) and is the part worth reading first.

The stored repeated-run artifacts predate the retirement of the wording-based
`constraint_retention` metric, so that field remains in the raw JSON as historical
evidence. The current scorer ignores it and new runs no longer emit it.

## Deterministic — [`checks.py`](checks.py), no model calls

These score the **trajectory**: what the agent did, in what order, with what
arguments. Facts, not opinions, and free to compute.

| Metric                      | Question                                   | Catches                                        |
| --------------------------- | ------------------------------------------ | ---------------------------------------------- |
| `tool_selection`            | Were the required tools called? (RECALL)   | Over-filtered context starving the model       |
| `tool_necessity`            | Were the calls made warranted? (PRECISION) | Thrashing; an unprompted write                 |
| `tool_argument_correctness` | Right tool, right arguments?               | Right tool, wrong `employee_id`                |
| `tool_sequence`             | Did dependent calls happen in order?       | A write firing before the lookup               |
| `call_efficiency`           | Any identical repeat calls?                | A result buried too far up the context         |
| `forbidden_avoided`         | Did it avoid tools it must not use?        | The write gate misfiring                       |
| `fact_recall`               | Do required facts appear in the answer?    | A fact dropped by trimming                     |
| `intent_accuracy`           | Did the planner classify the turn right?   | The misclassification everything else inherits |

Selection and necessity are deliberately **recall and precision over the same set of
calls**, so they move in opposite directions: trim context too hard and selection
falls; trim too little and necessity falls. One "did it work" number hides both.

## Judged — [`judges.py`](judges.py), one model call each

Rubric as system prompt, labelled sections as the user message, structured
`{score, reason}` back. Same shape as Lesson 7's RAG judges.

| Metric                 | Question                                                         |
| ---------------------- | ---------------------------------------------------------------- |
| `faithfulness`         | Is every claim supported by the evidence the agent gathered?     |
| `session_faithfulness` | Same, but against **every** tool result in the session           |
| `context_relevance`    | What fraction of retrieved documents were on-topic?              |
| `completeness`         | What fraction of required facts does the answer state?           |
| `correctness`          | Does the answer satisfy the request and respect the constraints? |

**Faithfulness covers tools, not just RAG.** A database row is evidence exactly as a
document chunk is, so `faithfulness` grades against *all* tool output.
`context_relevance` is the retrieval-only one, because "what fraction of the chunks
were on-topic" needs chunks.

**`completeness` and `fact_recall` deliberately overlap.** One matches substrings and
cannot be fooled by paraphrase in the *answer*; the other survives paraphrase of the
*fact*. When they disagree, read the reason — that disagreement is signal.

## Routing: turn kinds

Scoring every turn identically is wrong in both directions — it wastes judge calls on
"ok, drop it" and never checks what matters on a closing summary. So each dataset
turn declares a `kind`:

| Kind        | What it is                         | Judges it earns                                       |
| ----------- | ---------------------------------- | ----------------------------------------------------- |
| `retrieval` | needs a document / KB lookup       | faithfulness · context_relevance · completeness       |
| `lookup`    | needs a structured record          | faithfulness · completeness                           |
| `reasoning` | answerable from context, no tools  | correctness · completeness                            |
| `action`    | performs or authorises a write     | correctness                                           |
| `summary`   | the closing answer for the session | correctness · completeness · **session_faithfulness** |
| `trivial`   | an acknowledgement                 | none — safety metrics only                            |

All deterministic metrics run on every kind except `trivial`, which is scored only on
`forbidden_avoided` (an acknowledgement can still do damage; it cannot be incomplete).

### Two rules that override the table

**1. Retrieval is graded every time it happens, not only when expected.** If a turn
called a retrieval tool, faithfulness and context_relevance run on it whatever its
declared kind. Nobody writes a test case for the search they did not expect, and an
unplanned retrieval is exactly the one worth grading.

**2. A summary is graded against the whole session's evidence.** Retrieval failures
rarely surface on the turn that retrieved — they surface three turns later, when the
closing answer confidently restates something no document said. So `summary` turns get
`session_faithfulness`, judged against every tool result the session gathered.

That second rule is the reason the suite tracks evidence across turns instead of
scoring each turn in isolation.

**Missing expected evidence scores zero.** A retrieval or lookup turn that gathers no
evidence still receives `faithfulness = 0`; a retrieval turn that retrieves no
documents receives `context_relevance = 0`. Omitting those metrics would reward the
stage that skipped the required lookup by removing a difficult score from its average.

## The outcome-quality score and hard gates

The original report averaged every applicable metric equally. That is useful while
debugging, but it is not a defensible product-quality score: a correct final answer is
more important than an internal intent label, and a forbidden write must not be
compensated for by several cheap successes.

[`compare.py`](compare.py) reports a weighted **outcome quality** score from 0–100:

| Outcome metric              | Weight | Why it belongs                                                   |
| --------------------------- | -----: | ---------------------------------------------------------------- |
| `correctness`               | 25%    | Did the answer satisfy the request and its active constraints?   |
| `completeness`              | 25%    | Did the answer contain everything the user needed?               |
| `faithfulness`              | 20%    | Was the turn grounded in evidence the agent gathered?            |
| `session_faithfulness`      | 5%     | Did the closing answer remain grounded across the whole session? |
| `tool_selection`            | 10%    | Were the required capabilities actually used?                    |
| `tool_argument_correctness` | 10%    | Were those capabilities invoked with usable arguments?           |
| `fact_recall`               | 5%     | Did exact load-bearing ids, dates, and numbers survive?          |

This is a **weighted macro-average of metric means**, not an average of turns. The
weights total 100 for the full suite. When a session has no applicable examples for a
metric, its remaining weights are renormalized.

The report also includes a strict **task success rate**. A substantive turn succeeds
only when every applicable deterministic outcome metric is 1.0 and every applicable
judged outcome metric is at least 0.8. This answers the more intuitive question “how
many complete turns worked?” and prevents a strong score on one dimension from
compensating for a broken one. Stage 03's measured rates, against the baseline, are in
`docs/context-policy.md` (section 5).

Four other metrics remain diagnostics rather than product-quality ingredients:
`intent_accuracy` explains routing failures, `context_relevance` diagnoses retrieval,
`tool_necessity` prices unnecessary calls, and `call_efficiency` catches repetition.
They are valuable for deciding what to change, but they are not themselves proof that
the user received a good answer.

The score is subordinate to a hard gate:

- any runtime error or empty non-result fails the run;
- any `forbidden_avoided < 1.0` fails the run;
- any `tool_sequence < 1.0` fails the run;
- any judge error marks the measurement `INVALID`, rather than silently scoring the
  candidate as bad.

Running with `--no-judge` leaves the deterministic diagnostics available but reports
no outcome score and marks the quality gate `PARTIAL`; it cannot accept a candidate.

A stage is therefore acceptable only when its gate is `PASS`; a high score can never
average away an unsafe write or a broken run.

## Turn-kind coverage in the shipped dataset

| Kind        | Turns  |
| ----------- | ------ |
| `lookup`    | 17     |
| `retrieval` | 14     |
| `reasoning` | 11     |
| `summary`   | 5      |
| `action`    | 5      |
| `trivial`   | 2      |
| **Total**   | **54** |

Plus 18 turns carrying `arg_checks` and 5 carrying a `sequence` — the five write turns,
where "did `get_employee` precede `create_access_request`?" is the whole question.

## Cost, and the honest limits

Judges run on most turns, so a full comparison is a few hundred extra model calls.
`--no-judge` skips all of them and leaves the deterministic suite, which is free and
catches most trajectory regressions on its own. Each stage's `graph.py` runs without
judges by default; pass `--judge` to include them.

What this suite is **not**: a labelled golden set. There is no Recall@k, no MRR, no
inter-rater agreement, no significance testing — and given the run-to-run variance
documented in `RESULTS.md`, small differences here are not measurements. This is
enough evidence to tell whether a context change helped, hurt, or did nothing. Lesson
11 is where evaluation becomes the subject rather than the instrument.
