"""
Deterministic checks over the recorded agent trajectory.

The counterpart to `judges.py`. Every metric here scores what the agent DID — its
trajectory: which tools it called, in what order, with what arguments — and needs no
model call to do it. The trajectory is a plain list of `{name, args}` dicts captured
while the turn ran, so we just assert on it.

That is the lesson Lesson 8 makes and this one keeps: **not every eval needs a
judge.** Whether the right tool was called, whether a needless one was called,
whether the arguments were usable, whether the order made sense — these are facts.
Save the slower, costlier, noisier LLM judge for the questions you genuinely cannot
assert (is the answer grounded, does it satisfy the request).

The metrics, and the failure each catches:

  tool_selection      — were the tools the turn needs actually called?   (RECALL)
  tool_necessity      — were the calls it made warranted?                (PRECISION)
  tool_argument_correctness — right tool, right arguments?
  tool_sequence       — did dependent calls happen in a sensible order?
  call_efficiency     — did it avoid repeating a call it had already made?
  forbidden_avoided   — did it stay away from tools it must not use?
  fact_recall         — do the facts the turn depends on appear in the answer?
  intent_accuracy     — did the planner classify the turn correctly?

Selection and necessity are deliberately recall and precision over the SAME set of
tool calls, so they move in opposite directions under the classic failure modes: an
agent that under-calls has low selection; one that over-calls — thrashing, or firing
a write it should not — has low necessity. **A stage that trims context too hard
shows up as falling selection; one that trims too little shows up as falling
necessity.** Watching only a single "did it work" number hides both.

Every metric returns `(score, reason)` in 0.0-1.0, or `None` when the turn declares
nothing to check — so the harness can skip it like a not-applicable cell rather than
score it zero.
"""

import json

# Reads that are almost always reasonable to make on the way to an answer, so they
# do not count against precision even when a turn did not strictly require them.
# Being explicit about this is the difference between measuring waste and punishing
# ordinary caution.
ALWAYS_REASONABLE = {"list_policies", "get_employee"}


def matches(text: str, expectation: str) -> bool:
    """Case-insensitive substring match, where '|' separates acceptable alternatives."""
    haystack = text.lower()
    return any(alt.strip().lower() in haystack for alt in expectation.split("|"))


def _expectation_met(called: list[str], expectation: str) -> bool:
    return any(alt.strip() in called for alt in expectation.split("|"))


def tool_selection(called: list[str], expected: list[str]) -> tuple[float, str] | None:
    """RECALL: of the tools this turn required, how many were actually called?

    A turn with no required tools has nothing to recall, so it is not scored here —
    the penalty for calling something anyway lives in `tool_necessity`.
    """
    if not expected:
        return None
    hit = [e for e in expected if _expectation_met(called, e)]
    missing = [e for e in expected if not _expectation_met(called, e)]
    reason = f"{len(hit)}/{len(expected)} required tools called"
    if missing:
        reason += f" (missing: {', '.join(missing)})"
    return len(hit) / len(expected), reason


def tool_necessity(
    called: list[str], expected: list[str], optional: list[str] | None = None
) -> tuple[float, str] | None:
    """PRECISION: what fraction of the calls the agent made were warranted?

    `called` keeps repeats and order — a wasted call is a wasted call. A call is
    warranted if its tool is required, listed as optional for this turn, or in
    ALWAYS_REASONABLE. Calling something outside all three — the classic being an
    unprompted write — drops this score.

    No calls at all means nothing was wasted, so it is not scored: an agent that
    should have called something and did not is punished by `tool_selection`.
    """
    if not called:
        return None
    allowed = {alt.strip() for e in expected for alt in e.split("|")}
    allowed |= set(optional or []) | ALWAYS_REASONABLE
    warranted = [c for c in called if c in allowed]
    unwarranted = sorted({c for c in called if c not in allowed})
    reason = f"{len(warranted)}/{len(called)} calls warranted"
    if unwarranted:
        reason += f" (unwarranted: {', '.join(unwarranted)})"
    return len(warranted) / len(called), reason


def tool_argument_correctness(
    trajectory: list[dict], arg_checks: list[dict] | None
) -> tuple[float, str] | None:
    """Right tool, right arguments?

    Selection only asks *which* tool. This asks whether the call was usable — the
    failure the ambiguous-name cases exist to expose: the model picks
    `create_access_request` correctly and hands it the wrong `employee_id` because it
    never resolved which Rachel the user meant.

    Each check is `{"tool", "arg", "equals"|"contains"}`, and passes if ANY call to
    that tool carried a matching argument. Free-form arguments like a search query
    are not asserted here — their quality shows up downstream in context_relevance.
    """
    if not arg_checks:
        return None
    passed, details = 0, []
    for chk in arg_checks:
        values = [str(t["args"].get(chk["arg"], "")).strip().lower()
                  for t in trajectory if t["name"] == chk["tool"]]
        ok = any(
            v == str(chk["equals"]).strip().lower() if "equals" in chk
            else str(chk["contains"]).strip().lower() in v
            for v in values
        )
        passed += ok
        details.append(f"{chk['tool']}.{chk['arg']}={'ok' if ok else 'MISS'}")
    return passed / len(arg_checks), "; ".join(details)


def tool_sequence(
    trajectory: list[dict], sequence: list[str] | None
) -> tuple[float, str] | None:
    """Did dependent calls happen in a defensible ORDER?

    The trajectory test the other metrics cannot express. Selection says a write
    happened; necessity says it was allowed; neither notices that it fired *before*
    the lookup that was supposed to justify it. `sequence` names tools that must
    appear in this relative order — later entries may not precede earlier ones.

    Scored as the fraction of consecutive pairs in the right order, so a partly
    correct trajectory scores partly, and a turn that never called the tools at all
    is not scored here (that is `tool_selection`'s job).
    """
    if not sequence or len(sequence) < 2:
        return None
    first = {}
    for i, call in enumerate(trajectory):
        first.setdefault(call["name"], i)
    pairs = list(zip(sequence, sequence[1:]))
    scored = [(a, b) for a, b in pairs if a in first and b in first]
    if not scored:
        return None
    ok = [(a, b) for a, b in scored if first[a] < first[b]]
    wrong = [f"{b} before {a}" for a, b in scored if first[a] >= first[b]]
    reason = f"{len(ok)}/{len(scored)} ordered pairs correct"
    if wrong:
        reason += f" ({'; '.join(wrong)})"
    return len(ok) / len(scored), reason


def call_efficiency(trajectory: list[dict]) -> tuple[float, str] | None:
    """Did it avoid re-making a call it had already made this turn?

    An identical repeat is pure waste, and models make them more often than you
    would expect — usually when the first result is buried further up a long
    context, which makes this a direct symptom of the disease this lesson treats.
    """
    if not trajectory:
        return None
    sigs = [f"{c['name']}({json.dumps(c['args'], sort_keys=True)})" for c in trajectory]
    unique = len(set(sigs))
    repeats = len(sigs) - unique
    return unique / len(sigs), (
        f"{repeats} repeated call(s)" if repeats else "no repeated calls"
    )


def forbidden_avoided(called: list[str], forbidden: list[str] | None) -> tuple[float, str] | None:
    """Did it stay away from tools this turn must not use? Binary, and it should be:
    a write fired at the wrong moment is not partially wrong."""
    if not forbidden:
        return None
    hits = sorted({f for f in forbidden if f in called})
    return (0.0, f"called forbidden: {', '.join(hits)}") if hits else (1.0, "none called")


def fact_recall(answer: str, facts: list[str] | None) -> tuple[float, str] | None:
    """Fraction of the turn's required facts that appear in the answer, by substring.

    Blunt on purpose: it matches ids, numbers and proper nouns, so a paraphrase
    cannot game it. It CAN be beaten by a paraphrase of the fact itself, which is
    exactly why `judges.completeness` exists alongside it — the two disagreeing is
    a signal worth reading, not a bug.
    """
    if not facts:
        return None
    hit = [f for f in facts if matches(answer, f)]
    missing = [f for f in facts if not matches(answer, f)]
    reason = f"{len(hit)}/{len(facts)} facts present"
    if missing:
        reason += f" (missing: {', '.join(missing)})"
    return len(hit) / len(facts), reason


def intent_accuracy(actual: str | None, expected: str | None) -> tuple[float, str] | None:
    """Did the planner classify the turn correctly?

    Not applicable to the baseline, which never classifies anything — that `None` is
    a finding about the baseline, not a gap in the harness.
    """
    if not expected or not actual:
        return None
    ok = matches(actual, expected)
    return (1.0, f"{actual}") if ok else (0.0, f"{actual} (expected {expected})")
