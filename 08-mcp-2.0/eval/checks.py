"""Deterministic trajectory checks — the PROCESS metrics for tool use.

The counterpart to judges.py. Every metric here scores what the agent DID (its
trajectory: which tools it called, in what order, with what arguments) — and needs
NO model call to do it. The trajectory is a plain list of {name, input} dicts we
captured while running the agent, so we just assert on it.

That's the lesson: not every eval needs an LLM judge. Whether the right tool was
called, whether a needless one was called, whether the arguments were right — these
are facts about the trajectory. Save the (slower, costlier, noisier) LLM judge for
the questions you genuinely can't assert: is the ANSWER grounded, does it use the
results, does it address the question (that's judges.py).

The three metrics, and the failure each catches:
  - tool_selection  — did the agent call the tools the task needed?  (RECALL)
  - tool_necessity  — did it avoid calling tools it didn't need?     (PRECISION)
  - tool_argument_correctness — when it called a tool, were the arguments right?

Selection and necessity are deliberately recall and precision over the SAME set of
tool calls, so they move in opposite directions under the classic failure modes:
an agent that under-calls has low selection; one that over-calls (thrashing, firing
a write it shouldn't) has low necessity. Argument correctness is the one that catches
the sneakiest bug — right tool, wrong argument (e.g. the wrong employee_id after a
name that matched two people).

A metric returns (score, reason); tool_argument_correctness returns None when a case
declares no argument checks, so the harness can skip it (like a not-applicable cell).
"""


def tool_selection(called: set[str], expected: set[str]) -> tuple[float, str]:
    """RECALL: of the tools the task required, how many did the agent actually call?

    score = |called ∩ expected| / |expected|. A case with no required tools (e.g. a
    greeting that needs no tool at all) is trivially satisfied — nothing to recall —
    so it scores 1.0; the penalty for calling something anyway lives in tool_necessity.
    """
    if not expected:
        return 1.0, "no tools required"
    hit = called & expected
    missing = expected - called
    reason = f"{len(hit)}/{len(expected)} required tools called"
    if missing:
        reason += f" (missing: {', '.join(sorted(missing))})"
    return len(hit) / len(expected), reason


def tool_necessity(
    called_list: list[str], expected: set[str], optional: set[str]
) -> tuple[float, str]:
    """PRECISION: what fraction of the agent's tool calls were warranted?

    `called_list` keeps repeats and order — a wasted call is a wasted call. A call
    counts as warranted if its tool is in expected OR optional. `optional` is the
    escape hatch for tools that are reasonable-but-not-required (e.g. checking seat
    usage before filing a request): allowed, so they don't hurt precision, but not
    demanded by tool_selection either.

    score = (warranted calls) / (total calls). Calling a tool that's neither expected
    nor optional — the classic being an unprompted WRITE — drops this. No calls at all
    means nothing was wasted, so it scores 1.0 (an agent that SHOULD have called
    something but didn't is punished by tool_selection, not here).

    (Caveat worth knowing: repeating an *allowed* tool isn't penalized here — both
    calls count as warranted. Our cases don't lean on that; catching redundant repeats
    would need a per-tool call budget.)
    """
    allowed = expected | optional
    if not called_list:
        return 1.0, "no tool calls — nothing wasted"
    warranted = [name for name in called_list if name in allowed]
    unwarranted = [name for name in called_list if name not in allowed]
    reason = f"{len(warranted)}/{len(called_list)} calls warranted"
    if unwarranted:
        reason += f" (unwarranted: {', '.join(sorted(set(unwarranted)))})"
    return len(warranted) / len(called_list), reason


def tool_argument_correctness(
    trajectory: list[dict], arg_checks: list[dict]
) -> tuple[float, str] | None:
    """Did the agent pass the RIGHT arguments to the tools it called?

    Selection only asks *which* tool. This asks whether the call was actually usable —
    the failure the disambiguation cases are built to expose: the model picks
    create_access_request correctly but hands it the wrong employee_id because it
    didn't resolve which "Rachel" (or "Daniel") the user meant.

    Each check is {"tool", "arg", and one of "equals"/"contains"}; a check passes if
    ANY call to that tool carried a matching argument (case-insensitive). Free-form
    args like a search query aren't checked here — their quality shows up downstream
    in result utilization/faithfulness, not as an exact assertion.

    Returns None when the case declares no checks, so the harness skips the metric for
    that case instead of scoring it 0.
    """
    if not arg_checks:
        return None
    passed, details = 0, []
    for chk in arg_checks:
        inputs = [t["input"] for t in trajectory if t["name"] == chk["tool"]]
        ok = False
        for inp in inputs:
            value = str(inp.get(chk["arg"], "")).strip().lower()
            if "equals" in chk:
                ok = value == str(chk["equals"]).strip().lower()
            elif "contains" in chk:
                ok = str(chk["contains"]).strip().lower() in value
            if ok:
                break
        passed += ok
        details.append(f"{chk['tool']}.{chk['arg']}={'ok' if ok else 'MISS'}")
    return passed / len(arg_checks), "; ".join(details)
