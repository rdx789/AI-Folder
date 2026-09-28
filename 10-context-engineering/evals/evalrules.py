"""
The eval routing rules — which checks run on which turn, and why.

A conversation is not a list of interchangeable questions, so scoring every turn the
same way is wrong in both directions: it wastes judge calls on "ok, drop it", and it
never checks the thing that actually matters on a summary turn. This file is the
routing table that fixes that, and it is deliberately one readable page.

Every turn in the dataset declares a **kind**. The kind decides which metrics apply:

  kind        what it is                                metrics beyond the always-on set
  ----------  ----------------------------------------  --------------------------------
  retrieval   needs a document / knowledge-base lookup  faithfulness · context_relevance · completeness
  lookup      needs a structured record (DB) lookup     faithfulness · completeness
  reasoning   answerable from context; no tool needed   correctness · completeness
  action      performs or authorises a write            correctness
  summary     the closing answer for the session        correctness · completeness · SESSION faithfulness
  trivial     an acknowledgement; nothing to get right  safety metrics only

Always-on deterministic metrics (every kind except `trivial`):
  tool_selection · tool_necessity · tool_argument_correctness · tool_sequence
  call_efficiency · forbidden_avoided · fact_recall · intent_accuracy

Two rules override the table, and both exist because of how RAG actually fails:

1. **Retrieval is checked every time it happens, not only when it was expected.**
   If a turn called a retrieval tool, faithfulness and context_relevance run on it
   whatever its declared kind. An unplanned search is exactly the case you want
   graded — nobody writes a test case for the retrieval they did not expect.

   The judge also sees what the USER said, because the user is a source of truth
   too — a rule they stated is grounding, and without that the judge scores correct
   constraint-carrying as hallucination.

2. **A summary is checked against the WHOLE session's evidence.** Retrieval failures
   do not surface on the turn that retrieved; they surface three turns later when the
   closing answer confidently restates something no document said. So `summary` turns
   are judged for faithfulness against every tool result gathered in the session, not
   just their own turn's.

And one rule that saves money: **`trivial` turns are never judged.** An answer to
"fine, drop it" has nothing to be faithful to and nothing to complete. Scoring it
would add noise to the averages and cost a model call to do it.
"""

from . import checks, judges

# Tools whose output is a DOCUMENT — the only ones context_relevance can score,
# because "what fraction of the retrieved chunks were on-topic" needs chunks.
RETRIEVAL_TOOLS = {"search_knowledge_base", "search_hr_documents", "get_policy"}

# Which judges each kind earns. Rule 1 also adds retrieval judges dynamically when
# a different kind unexpectedly retrieves documents.
JUDGES_BY_KIND = {
    "retrieval": {"faithfulness", "context_relevance", "completeness"},
    "lookup":    {"faithfulness", "completeness"},
    "reasoning": {"correctness", "completeness"},
    "action":    {"correctness"},
    "summary":   {"correctness", "completeness", "session_faithfulness"},
    "trivial":   set(),
}


def turn_kind(turn: dict) -> str:
    """A turn's declared kind, defaulting to the most-checked interpretation.

    Defaulting to `lookup` rather than `trivial` is deliberate: a turn nobody
    classified should be over-checked, not silently skipped.
    """
    return turn.get("kind", "lookup")


def evidence_of(record: dict, retrieval_only: bool = False) -> list[str]:
    """What the tools actually returned this turn, as text for a judge.

    Faithfulness uses ALL of it — a database row is evidence exactly as a document
    chunk is, and "tool faithfulness" and "RAG faithfulness" are the same question
    asked of different backends. context_relevance uses only the document tools.
    """
    results = record.get("tool_results") or []
    if retrieval_only:
        results = [r for r in results if r["name"] in RETRIEVAL_TOOLS]
    return [f"[{r['name']}]\n{r['text']}" for r in results]


def run_deterministic(turn: dict, record: dict) -> dict:
    """Every metric that needs no model call. Returns {metric: (score, reason)}."""
    kind = turn_kind(turn)
    answer = record.get("answer", "")
    trajectory = record.get("trajectory") or []
    called = [c["name"] for c in trajectory]

    # A trivial turn still must not do damage — but nothing else about it is graded.
    results = {"forbidden_avoided": checks.forbidden_avoided(called, turn.get("forbidden_tools"))}
    if kind == "trivial":
        return {k: v for k, v in results.items() if v is not None}

    results.update({
        "tool_selection": checks.tool_selection(called, turn.get("expected_tools")),
        "tool_necessity": checks.tool_necessity(
            called, turn.get("expected_tools") or [], turn.get("optional_tools")
        ),
        "tool_argument_correctness": checks.tool_argument_correctness(
            trajectory, turn.get("arg_checks")
        ),
        "tool_sequence": checks.tool_sequence(trajectory, turn.get("sequence")),
        "call_efficiency": checks.call_efficiency(trajectory),
        "fact_recall": checks.fact_recall(answer, turn.get("required_facts")),
        "intent_accuracy": checks.intent_accuracy(
            record.get("intent"), turn.get("expected_intent")
        ),
    })
    return {k: v for k, v in results.items() if v is not None}


async def run_judges(turn: dict, record: dict, session: dict, session_evidence: list[str]) -> dict:
    """The model-scored metrics this turn earns. One call per metric that applies."""
    kind = turn_kind(turn)
    answer = record.get("answer", "")
    if kind == "trivial" or not answer:
        return {}

    wanted = set(JUDGES_BY_KIND.get(kind, set()))
    turn_evidence = evidence_of(record)
    documents = evidence_of(record, retrieval_only=True)

    # RULE 1 — retrieval is graded whenever it happened, expected or not.
    if documents:
        wanted |= {"faithfulness", "context_relevance"}
    if not turn.get("required_facts"):
        wanted.discard("completeness")

    request = turn["user"]
    history = "\n".join(f"- {t['user']}" for t in session["turns"] if t["n"] < turn["n"])
    results = {}

    # Expected evidence metrics must have stable coverage across stages. Previously,
    # a stage that skipped a required lookup simply did not receive a faithfulness
    # score, which could make a worse agent's average look better. Missing required
    # evidence is now an explicit zero; unplanned non-retrieval turns still do not
    # acquire these metrics.
    if "faithfulness" in wanted and not turn_evidence:
        results["faithfulness"] = (0.0, "no tool evidence gathered")
        wanted.discard("faithfulness")
    if "context_relevance" in wanted and not documents:
        results["context_relevance"] = (0.0, "no documents retrieved")
        wanted.discard("context_relevance")

    if "faithfulness" in wanted:
        results["faithfulness"] = await judges.faithfulness(
            request, turn_evidence, answer, stated_by_user=history
        )
    if "context_relevance" in wanted:
        results["context_relevance"] = await judges.context_relevance(request, documents)
    if "completeness" in wanted:
        results["completeness"] = await judges.completeness(
            request, turn["required_facts"], answer
        )
    if "correctness" in wanted:
        results["correctness"] = await judges.correctness(request, history, answer)
    # RULE 2 — the closing answer is graded against everything the session gathered.
    if "session_faithfulness" in wanted:
        if session_evidence:
            results["session_faithfulness"] = await judges.faithfulness(
                request, session_evidence, answer, stated_by_user=history
            )
        else:
            results["session_faithfulness"] = (0.0, "no session evidence gathered")
    return results
