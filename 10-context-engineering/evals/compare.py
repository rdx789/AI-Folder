"""
Run every stage folder over the same sessions and print what actually happened.

This is the evidence, and it is the reason the lab exists in this shape. Every
stage claims to help. Some of those claims are false on some sessions, and the only
way to know which is to replay identical input through all of them and read the
numbers — including the ones that make the engineered version look bad.

    python evals/compare.py                       # everything (every stage x every session)
    python evals/compare.py --session S2          # one session, all stages
    python evals/compare.py --stages 00,03        # baseline vs finished version
    python evals/compare.py --no-judge            # skip the LLM judge calls

Results are written to evals/results/ so class discussion needs no live re-run.

Read the table in this order:
  1. `total` on S1 vs S2. The engineered stages should LOSE on the short session.
  2. `overhead` — what planning and distillation cost, separated from answering.
  3. `schema est` — the tool-definition bill stage 03's tool loadout stops paying.
  4. `used` — tool utilization: of the schemas you paid to expose, what fraction did
     the model actually reach for? The baseline sits near 7%. That single number is
     the argument for a per-turn tool loadout.
  5. `final in` — input tokens on the LAST turn: how fast context grows.
  6. `quality` and `gate` — weighted user-visible outcome quality plus the hard
     safety/runtime gate. A stage that saves tokens and loses ten quality points has
     not simply become more efficient; it made a measurable trade. A forbidden write
     or broken run fails the gate regardless of the average.

And read S5 separately from the rest. Sessions 1-4 are written in tidy sentences;
S5 is pasted emails, typos and vague references. Every stage scores worse there, and
the stages that lean hardest on the planner's classification degrade the most —
which is the cost of engineering that nothing else in the table shows you.
"""

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
CODE_DIR = EVALS_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from evals.dataset import load_sessions
from evals.evalrules import JUDGES_BY_KIND
from evals.runner import run_session
from evals.runtime import load_eval_tools
from evals.tokens import index_tools

# Discovered rather than hardcoded: a stage is any "NN-name" folder with a graph.py.
# A fixed list broke the default run once stage folders were removed; the graph.py
# check also keeps shared-code folders such as 00-agent-shared out.
STAGES = sorted(
    p.name for p in CODE_DIR.iterdir()
    if p.is_dir() and p.name[:2].isdigit() and p.name[2:3] == "-" and (p / "graph.py").is_file()
)
RESULTS_DIR = EVALS_DIR / "results"

SUMMARY_ROW = ("{stage:<26} {inp:>9} {out:>7} {total:>9} {calls:>5} {tpc:>5} {util:>5} "
               "{schema:>10} {over:>9} {final:>8} {quality:>7} {success:>8} {gate:>7} "
               "{det:>6} {jud:>6} {sec:>6}")


def load_stage(name: str):
    """Import a stage's graph.py by path.

    The folders are named 00-baseline and friends, which are not importable module
    names, so we load them by file location instead of by import statement.
    """
    spec = importlib.util.spec_from_file_location(
        f"stage_{name.replace('-', '_')}", CODE_DIR / name / "graph.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


METRIC_ORDER = [
    "tool_selection", "tool_necessity", "tool_argument_correctness", "tool_sequence",
    "call_efficiency", "forbidden_avoided", "fact_recall", "intent_accuracy",
    "faithfulness", "session_faithfulness", "context_relevance", "completeness",
    "correctness",
]
DETERMINISTIC = set(METRIC_ORDER[:8])

# A quality score should reflect whether the user got a correct, complete, grounded
# result — not whether an internal classifier happened to use the expected label or
# whether schemas were cheap. We therefore macro-average these outcome metrics with
# explicit importance weights. The weights total 100 for the full suite and are
# renormalized when a session has no applicable examples.
QUALITY_WEIGHTS = {
    "correctness": 25,
    "completeness": 25,
    "faithfulness": 20,
    "session_faithfulness": 5,
    "tool_selection": 10,
    "tool_argument_correctness": 10,
    "fact_recall": 5,
}

# These are release gates, not ingredients a high average may compensate for.
# A candidate with a forbidden write, an invalid write sequence, or a runtime error
# fails even if every other answer is excellent.
HARD_GATE_METRICS = {"forbidden_avoided", "tool_sequence"}

# A strict per-turn complement to the weighted score. Deterministic requirements
# must be perfect; judge scores allow the rubric's small grading variation.
EXACT_SUCCESS_METRICS = {
    "tool_selection", "tool_argument_correctness", "fact_recall"
}
JUDGED_SUCCESS_METRICS = {
    "correctness", "completeness", "faithfulness", "session_faithfulness"
}


def metric_means(records: list[dict]) -> dict[str, tuple[float, int]]:
    """Per-metric (mean score, number of turns it applied to)."""
    out = {}
    for name in METRIC_ORDER:
        scores = [r["metrics"][name][0] for r in records if name in r.get("metrics", {})]
        if scores:
            out[name] = (sum(scores) / len(scores), len(scores))
    return out


def outcome_quality(means: dict[str, tuple[float, int]]) -> float | None:
    """Weighted 0–100 user-outcome score, independent of diagnostic metrics."""
    weighted = [
        (means[name][0], weight)
        for name, weight in QUALITY_WEIGHTS.items()
        if name in means
    ]
    if not weighted:
        return None
    return round(100 * sum(score * weight for score, weight in weighted)
                 / sum(weight for _, weight in weighted), 1)


def task_success(records: list[dict]) -> tuple[float | None, int, int]:
    """Strict end-to-end success rate over substantive, scorable turns."""
    passed = 0
    assessed = 0
    for record in records:
        applicable = {
            name: score
            for name, (score, _) in record.get("metrics", {}).items()
            if name in EXACT_SUCCESS_METRICS or name in JUDGED_SUCCESS_METRICS
        }
        if not applicable:
            continue
        assessed += 1
        exact_ok = all(
            score == 1.0 for name, score in applicable.items()
            if name in EXACT_SUCCESS_METRICS
        )
        judged_ok = all(
            score >= 0.8 for name, score in applicable.items()
            if name in JUDGED_SUCCESS_METRICS
        )
        if not record.get("error") and record.get("answer") and exact_ok and judged_ok:
            passed += 1
    rate = round(100 * passed / assessed, 1) if assessed else None
    return rate, passed, assessed


def quality_gate(records: list[dict], evaluation_complete: bool) -> tuple[str, list[str]]:
    """Return PASS/FAIL/INVALID/PARTIAL plus the concrete reasons.

    Judge failures invalidate the measurement rather than making the candidate look
    bad. Runtime and hard safety failures reject the candidate.
    """
    invalid = []
    failed = []
    for record in records:
        if record.get("error"):
            failed.append(f"turn {record['turn']}: runtime error")
        elif not record.get("answer"):
            failed.append(f"turn {record['turn']}: empty answer")
        for name, (score, reason) in record.get("metrics", {}).items():
            if str(reason).startswith("judge error:"):
                invalid.append(f"turn {record['turn']}: {name} {reason}")
            if name in HARD_GATE_METRICS and score < 1.0:
                failed.append(f"turn {record['turn']}: {name}={score:.2f}")
    if invalid:
        return "INVALID", sorted(set(invalid))
    if failed:
        return "FAIL", sorted(set(failed))
    if not evaluation_complete:
        return "PARTIAL", ["LLM judges were disabled"]
    return "PASS", []


def summarize(records: list[dict]) -> dict:
    scored = [r["score"] for r in records if r.get("score") is not None]
    means = metric_means(records)
    det = [m for n, (m, _) in means.items() if n in DETERMINISTIC]
    jud = [m for n, (m, _) in means.items() if n not in DETERMINISTIC]
    # New records carry the explicit flag. The metric inference keeps historical
    # result files readable after this field was introduced.
    evaluation_complete = any(r.get("judges_enabled") for r in records) or bool(jud)
    gate, gate_reasons = quality_gate(records, evaluation_complete)
    success_rate, successful_turns, assessed_turns = task_success(records)
    return {
        "score": round(sum(scored) / len(scored), 3) if scored else None,
        "score_deterministic": round(sum(det) / len(det), 3) if det else None,
        "score_judged": round(sum(jud) / len(jud), 3) if jud else None,
        "outcome_quality": outcome_quality(means) if evaluation_complete else None,
        "task_success_rate": success_rate if evaluation_complete else None,
        "successful_turns": successful_turns if evaluation_complete else None,
        "assessed_turns": assessed_turns if evaluation_complete else None,
        "quality_gate": gate,
        "quality_gate_reasons": gate_reasons,
        "metrics": {k: (round(v, 3), n) for k, (v, n) in means.items()},
        "input_tokens": sum(r["input_tokens"] for r in records),
        "output_tokens": sum(r["output_tokens"] for r in records),
        "total_tokens": sum(r["total_tokens"] for r in records),
        "model_calls": sum(r["model_calls"] for r in records),
        "overhead_tokens": sum(r["overhead_tokens"] for r in records),
        "schema_tokens_est": sum(r["schema_tokens_est"] for r in records),
        "tools_exposed_avg": round(
            sum(r["tools_exposed"] for r in records) / max(len(records), 1), 1
        ),
        "final_turn_input": records[-1]["input_tokens"] if records else 0,
        "latency_s": round(sum(r["latency_s"] for r in records), 1),
        "errors": sum(1 for r in records if r["error"]),
        "rearms": sum(r.get("rearms", 0) for r in records),
        # Of the tools we paid to expose, what fraction did the model reach for?
        "tool_utilization": round(
            sum(u for r in records if (u := r.get("tool_utilization")) is not None)
            / max(sum(1 for r in records if r.get("tool_utilization") is not None), 1), 2
        ),
    }


def print_table(title: str, rows: list[tuple[str, dict]]) -> None:
    """The COST table: what each stage spent."""
    print(f"\n{title}")
    print(SUMMARY_ROW.format(
        stage="stage", inp="input", out="output", total="total", calls="calls",
        tpc="tools", util="used", schema="schema est", over="overhead", final="final in",
        quality="quality", success="success", gate="gate", det="det", jud="judged", sec="sec",
    ))
    print("-" * 163)
    for name, s in rows:
        print(SUMMARY_ROW.format(
            stage=name,
            inp=f"{s['input_tokens']:,}",
            out=f"{s['output_tokens']:,}",
            total=f"{s['total_tokens']:,}",
            calls=s["model_calls"],
            tpc=s["tools_exposed_avg"],
            util=f"{s['tool_utilization']:.0%}" if s.get("tool_utilization") else "—",
            schema=f"{s['schema_tokens_est']:,}",
            over=f"{s['overhead_tokens']:,}",
            final=f"{s['final_turn_input']:,}",
            quality=f"{s['outcome_quality']:.1f}" if s.get("outcome_quality") is not None else "—",
            success=(f"{s['task_success_rate']:.1f}%"
                     if s.get("task_success_rate") is not None else "—"),
            gate=s["quality_gate"],
            det=f"{s['score_deterministic']:.2f}" if s.get("score_deterministic") else "—",
            jud=f"{s['score_judged']:.2f}" if s.get("score_judged") else "—",
            sec=s["latency_s"],
        ))


def print_metrics(title: str, stages: list[str], per_stage: dict[str, dict]) -> None:
    """The QUALITY table: every metric, every stage, side by side.

    `n` is how many turns the metric applied to — a metric that only fired twice is
    a hint, not a result, and printing the count keeps that visible.
    """
    print(f"\n{title}")
    width = max(len(s) for s in stages) if stages else 10
    print(f"{'metric':<26} {'n':>4}  " + "  ".join(f"{s[:12]:>12}" for s in stages))
    print("-" * (32 + 14 * len(stages)))
    for metric in METRIC_ORDER:
        cells, counts = [], []
        for st in stages:
            m = per_stage[st].get(metric)
            cells.append(f"{m[0]:>12.2f}" if m else f"{'—':>12}")
            if m:
                counts.append(m[1])
        if not counts:
            continue
        if metric == "faithfulness":
            print("- " * ((32 + 14 * len(stages)) // 2))
        print(f"{metric:<26} {max(counts):>4}  " + "  ".join(cells))


def merge_metrics(records: list[dict]) -> dict[str, tuple[float, int]]:
    return metric_means(records)


def print_variant_robustness(results: dict, sessions: list[dict], stages: list[str]) -> None:
    """Compare controlled semantic variants without folding noise into quality.

    The ordinary outcome score says whether each session worked. This report says how
    much the same task degraded as presentation noise increased.
    """
    groups: dict[str, list[dict]] = {}
    for session in sessions:
        group = session.get("variant_group")
        if group and session.get("variant"):
            groups.setdefault(group, []).append(session)

    # outcome_quality and task_success_rate exist only when the LLM judges ran.
    if groups and any(
        results[stage][session["id"]]["summary"]["outcome_quality"] is None
        for stage in stages for session in sessions
    ):
        print("\nCONTROLLED NOISE CURVE skipped: it compares judged quality (run without --no-judge).")
        return

    for group, variants in groups.items():
        variants.sort(key=lambda session: session.get("noise_level", 0))
        clean = next((session for session in variants if session["variant"] == "clean"), None)
        if not clean or len(variants) < 2:
            continue
        clean_id = clean["id"]
        print(f"\nCONTROLLED NOISE CURVE {group} — every level versus clean")
        print(f"{'stage':<26} {'variant':<10} {'noise':>5} {'quality':>8} {'q delta':>8} "
              f"{'success':>9} {'ok delta':>9} {'gate':>7}")
        print("-" * 94)
        for stage in stages:
            clean_summary = results[stage][clean_id]["summary"]
            for variant in variants:
                summary = results[stage][variant["id"]]["summary"]
                q_delta = summary["outcome_quality"] - clean_summary["outcome_quality"]
                success_delta = (
                    summary["task_success_rate"] - clean_summary["task_success_rate"]
                )
                summary["variant_comparison"] = {
                    "variant_group": group,
                    "clean_session": clean_id,
                    "noise_level": variant.get("noise_level", 0),
                    "outcome_quality_delta": round(q_delta, 1),
                    "task_success_delta": round(success_delta, 1),
                }
                print(
                    f"{stage:<26} {variant['variant']:<10} "
                    f"{variant.get('noise_level', 0):>5} {summary['outcome_quality']:>8.1f} "
                    f"{q_delta:>+8.1f} {summary['task_success_rate']:>8.1f}% "
                    f"{success_delta:>+8.1f}pp {summary['quality_gate']:>7}"
                )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the context policies (one per stage folder).")
    parser.add_argument(
        "--session", default=None,
        help="comma-separated session-id prefixes, for example S6,S7,S8",
    )
    parser.add_argument("--stages", default=None, help="comma-separated stage prefixes, e.g. 00,03")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judges (much faster)")
    args = parser.parse_args()

    sessions = load_sessions(args.session)
    stages = STAGES
    if args.stages:
        wanted = tuple(s.strip() for s in args.stages.split(","))
        stages = [s for s in STAGES if s.startswith(wanted)]
    if not stages:
        # Fail before the tool load and the paid model calls, not with an empty table.
        parser.error(f"no stage matches {args.stages!r}; available: {', '.join(STAGES) or 'none'}")

    tools = await load_eval_tools()
    index_tools(tools)
    print(f"Loaded {len(tools)} tools. Replaying {len(sessions)} session(s) "
          f"against {len(stages)} stage(s). Judges: {'off' if args.no_judge else 'on'}.")

    results: dict[str, dict] = {}
    for stage in stages:
        app = load_stage(stage).build_graph(tools)
        print(f"\n{'=' * 163}\n{stage}\n{'=' * 163}")
        results[stage] = {}
        for session in sessions:
            records = await run_session(app, session, judge=not args.no_judge)
            results[stage][session["id"]] = {"records": records, "summary": summarize(records)}

    # Per-session: this is where the honest result lives. The stages do not all win
    # on all sessions, and the shape of WHERE they win is the lesson.
    for session in sessions:
        rows = [(st, results[st][session["id"]]["summary"]) for st in stages]
        print_table(f"SESSION {session['id']} — {session['focus'][:78]}", rows)
        print_metrics(
            f"  metrics · {session['id']}", stages,
            {st: merge_metrics(results[st][session["id"]]["records"]) for st in stages},
        )

    print_variant_robustness(results, sessions, stages)

    if len(sessions) > 1:
        rows = []
        for stage in stages:
            everything = [r for s in sessions for r in results[stage][s["id"]]["records"]]
            totals = summarize(everything)
            totals["final_turn_input"] = sum(
                results[stage][s["id"]]["summary"]["final_turn_input"] for s in sessions
            )
            totals["tool_utilization"] = round(sum(
                results[stage][s["id"]]["summary"]["tool_utilization"] for s in sessions
            ) / len(sessions), 2)
            rows.append((stage, totals))
        print_table("ALL SESSIONS", rows)
        print_metrics("  metrics · ALL SESSIONS", stages, {
            st: merge_metrics([r for s in sessions for r in results[st][s["id"]]["records"]])
            for st in stages
        })

        # Tidy versus noisy — the split that shows what engineering context costs.
        # Preserve the original S1–S4 versus S5 comparison. Controlled semantic
        # variants have their own noise-curve report above and must not leak into it.
        tidy = [s for s in sessions if not s.get("variant_group")
                and not s["id"].startswith("S5")]
        noisy = [s for s in sessions if s["id"].startswith("S5")]
        if tidy and noisy:
            print("\noutcome quality, tidy sessions vs noisy session (weighted 0–100):")
            for stage in stages:
                t_recs = [r for s in tidy for r in results[stage][s["id"]]["records"]]
                n_recs = [r for s in noisy for r in results[stage][s["id"]]["records"]]
                t = summarize(t_recs)["outcome_quality"]
                n = summarize(n_recs)["outcome_quality"]
                if t is None or n is None:  # no judges, so no outcome quality
                    print(f"  {stage:<26} skipped (needs the LLM judges; run without --no-judge)")
                    continue
                print(f"  {stage:<26} tidy {t:>5.1f}   noisy {n:>5.1f}   drop {t - n:+.1f}")

    baseline = results.get("00-baseline")
    if baseline and len(stages) > 1:
        base_total = sum(baseline[s["id"]]["summary"]["total_tokens"] for s in sessions)
        print("\nversus baseline:")
        for stage in stages[1:]:
            stage_total = sum(results[stage][s["id"]]["summary"]["total_tokens"] for s in sessions)
            delta = (stage_total - base_total) / base_total * 100 if base_total else 0
            print(f"  {stage:<26} {delta:+6.1f}% total tokens")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"compare-{time.strftime('%Y%m%d-%H%M%S')}.json"
    payload = json.dumps(results, indent=2, default=str)
    path.write_text(payload, encoding="utf-8")
    latest_name = "latest-targeted.json" if args.session or args.stages else "latest.json"
    (RESULTS_DIR / latest_name).write_text(payload, encoding="utf-8")
    print(f"\nDetailed results → {path.relative_to(CODE_DIR)}")
    print(f"Latest pointer   → evals/results/{latest_name}")


if __name__ == "__main__":
    asyncio.run(main())
