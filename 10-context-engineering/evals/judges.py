"""
Four LLM-as-judge metrics, in the shape Lessons 7 and 8 established.

Every metric is one model call, built the same way so they are easy to read and to
add to:
  - a RUBRIC becomes the system prompt — the judge's role and what 0.0 vs 1.0 mean;
  - the user message is a few LABELLED sections (question / evidence / answer / facts);
  - the score comes back through a structured output, so we always get a clean
    {score, reason} instead of parsing a number out of prose.

The metrics, and the failure each one catches:

  faithfulness       — is every claim in the answer supported by the EVIDENCE the
                       agent actually gathered? This is the metric that catches a
                       context change quietly making things worse: trim the wrong
                       thing and an agent does not fall silent, it keeps answering
                       from parametric memory instead of from what it retrieved.
                       Evidence here is *every* tool result, not only RAG — a
                       database row is evidence exactly as a document chunk is.
  context_relevance  — what fraction of the RETRIEVED chunks were on-topic? Scores
                       the retrieval, not the answer. Only meaningful on turns that
                       actually searched.
  completeness       — what fraction of the facts the turn requires does the answer
                       actually state? The graded counterpart to the deterministic
                       substring check: it survives paraphrase, which the cheap one
                       cannot.
  correctness        — does the answer satisfy the request AND respect the
                       constraints in force at this point in the session?

A judge returns (score, reason) in 0.0-1.0, matching `checks.py`, so the harness can
average everything the same way.

Judges are cheap, need no labelled golden answers, and catch the failures assertions
cannot. They do NOT prove an answer is correct — they are evidence, not proof, and
Lesson 11 is where evaluation becomes the subject rather than the instrument.
"""

from pydantic import BaseModel, Field

from .runtime import get_judge_model


class Score(BaseModel):
    """The structured verdict every judge returns."""

    score: float = Field(description="A value from 0.0 (worst) to 1.0 (best), per the rubric.")
    reason: str = Field(description="One short sentence justifying the score.")


async def _run_judge(rubric: str, sections: list[tuple[str, str]]) -> tuple[float, str]:
    """One judge call. `rubric` is the system prompt; `sections` are the labelled
    blocks of the user message. Returns (score, reason), clamped to 0.0-1.0."""
    user_message = "\n\n".join(f"### {label}\n{body}" for label, body in sections)
    grader = get_judge_model().with_structured_output(Score)
    try:
        result = await grader.ainvoke([("system", rubric), ("human", user_message)])
        return max(0.0, min(1.0, float(result.score))), result.reason
    except Exception as exc:  # a judge failure must never sink a comparison run
        return 0.0, f"judge error: {type(exc).__name__}"


# --- the rubrics (each defines its own 0.0-1.0 scale) -------------------------

FAITHFULNESS_RUBRIC = (
    "You are a strict evaluator scoring FAITHFULNESS — whether the ANSWER is "
    "grounded in the EVIDENCE the assistant gathered.\n"
    "  1.0 = every factual claim in the answer is supported by the evidence.\n"
    "  0.0 = the answer states claims the evidence does not support.\n"
    "Score by the EVIDENCE only; ignore whether a claim happens to be true in the "
    "real world. An answer that correctly says the evidence does not cover something "
    "is FAITHFUL.\n"
    "IMPORTANT: anything the USER stated is also valid grounding. A rule, date or "
    "constraint the user supplied is supported even though no tool returned it — "
    "repeating it back is faithful, not invented."
)

CONTEXT_RELEVANCE_RUBRIC = (
    "You are a strict evaluator scoring CONTEXT RELEVANCE — the PROPORTION of "
    "retrieved documents that are on-topic for the REQUEST.\n"
    "Label each numbered document independently as relevant (it helps answer the "
    "request) or not, then:\n"
    "  score = (number of relevant documents) / (total number of documents).\n"
    "IMPORTANT: do NOT score by whether the answer is present. An off-topic document "
    "MUST lower the score even if the others already answer the request fully. "
    "Example: 2 on-topic and 1 off-topic = 0.67, never 1.0."
)

COMPLETENESS_RUBRIC = (
    "You are a strict evaluator scoring COMPLETENESS — whether the ANSWER states the "
    "facts the turn requires.\n"
    "  score = the fraction of the REQUIRED FACTS the answer actually states.\n"
    "  1.0 = all present, 0.0 = none. Each missing required fact lowers the score.\n"
    "Accept paraphrase and different formatting of the same fact — '1 August 2026' "
    "and '2026-08-01' are the same fact. Do not credit a fact the answer only alludes "
    "to without stating."
)

CORRECTNESS_RUBRIC = (
    "You are a strict evaluator scoring CORRECTNESS of one turn in a support "
    "conversation.\n"
    "  1.0 = the answer satisfies the request AND respects every constraint still in "
    "force at this point in the session.\n"
    "  0.0 = it misses the request, invents a fact, or ignores a stated constraint.\n"
    "Be strict about invented specifics and ignored constraints; be lenient about "
    "wording, format and brevity. A constraint can be lifted or satisfied by a later "
    "user message INCLUDING the one being graded — 'do not file until I say so' is "
    "satisfied the moment the user says so. When the request is genuinely ambiguous, "
    "asking the user which they meant is CORRECT and guessing is not."
)


# --- the judges: assemble sections, run the rubric ----------------------------

async def faithfulness(
    request: str, evidence: list[str], answer: str, stated_by_user: str = ""
) -> tuple[float, str]:
    """Every claim in `answer` supported by the evidence — tool output, or the user.

    `stated_by_user` matters more than it looks. Half this lab's sessions turn on a
    rule the USER supplied ("anything over 7,500 needs Finance"), and an answer that
    correctly carries that rule forward has no tool result behind it. Without this
    section the judge scores exactly the right behaviour as a hallucination — which
    it did, on four turns, before this argument existed.
    """
    sections = [("REQUEST", request), ("EVIDENCE FROM TOOLS", "\n\n".join(evidence))]
    if stated_by_user:
        sections.append(("STATED BY THE USER (also valid grounding)", stated_by_user))
    sections.append(("ANSWER", answer))
    return await _run_judge(FAITHFULNESS_RUBRIC, sections)


async def context_relevance(request: str, documents: list[str]) -> tuple[float, str]:
    """What fraction of the retrieved documents were worth retrieving."""
    numbered = "\n\n".join(f"[{i + 1}] {d}" for i, d in enumerate(documents))
    return await _run_judge(CONTEXT_RELEVANCE_RUBRIC, [
        ("REQUEST", request),
        ("RETRIEVED DOCUMENTS", numbered),
    ])


async def completeness(request: str, key_facts: list[str], answer: str) -> tuple[float, str]:
    """The graded version of the deterministic fact check — survives paraphrase."""
    facts = "\n".join(f"- {f}" for f in key_facts)
    return await _run_judge(COMPLETENESS_RUBRIC, [
        ("REQUEST", request),
        ("REQUIRED FACTS", facts),
        ("ANSWER", answer),
    ])


async def correctness(request: str, history: str, answer: str) -> tuple[float, str]:
    """Does the answer satisfy the turn, given what the user has already said?

    The judge sees the earlier USER messages — where constraints were stated — but
    never the assistant's context. It grades the output, so it stays neutral between
    the stages.
    """
    return await _run_judge(CORRECTNESS_RUBRIC, [
        ("EARLIER USER MESSAGES", history or "(none)"),
        ("THIS TURN'S REQUEST", request),
        ("ANSWER", answer),
    ])
