"""Run the fixed question set (eval_questions.py) against every chunking
strategy's index through the ACTUAL rag_tool.py agent loop — not a
reimplementation — so what's measured here is exactly what a user would get
from `python rag_tool.py`. Prints, per index and per question, the retrieved
chunk ids + cosine scores + source files and the final cited answer, then
emits the README-format comparison table.

    python chunking-study/run_eval.py
"""
import json
import sys
from pathlib import Path

from common import DATA_DIR
from eval_questions import QUESTIONS, is_correct, is_refusal

RAG_TOOL_DIR = Path(__file__).resolve().parent.parent / "rag-as-tool"
sys.path.insert(0, str(RAG_TOOL_DIR))

import rag_tool  # noqa: E402  (path must be inserted first)
import tools  # noqa: E402

# Build-time call/cost accounting from build_indexes.py's run (see PLAN.md) —
# reused here rather than re-running the paid build just to fill this table.
BUILD_STATS = {
    "fixed_300w_50ov": {"chunks": 175, "embed_calls": 175, "nova_calls": 0, "cost_usd": 0.001686},
    "fixed_600w_100ov": {"chunks": 91, "embed_calls": 91, "nova_calls": 0, "cost_usd": 0.001671},
    "fixed_800w_200ov": {"chunks": 73, "embed_calls": 73, "nova_calls": 0, "cost_usd": 0.001804},
    "separator": {"chunks": 890, "embed_calls": 890, "nova_calls": 0, "cost_usd": 0.001412},
    "sentence_3": {"chunks": 802, "embed_calls": 802, "nova_calls": 0, "cost_usd": 0.001412},
    "sentence_5": {"chunks": 484, "embed_calls": 484, "nova_calls": 0, "cost_usd": 0.001414},
    "sentence_8": {"chunks": 306, "embed_calls": 306, "nova_calls": 0, "cost_usd": 0.001415},
    "semantic_llm": {"chunks": 196, "embed_calls": 196, "nova_calls": 28, "cost_usd": 0.015707},
}
STRATEGIES = list(BUILD_STATS)


def capture_retrieve(strategy_results: list):
    """Wrap tools.retrieve so each call's results are recorded for scoring,
    while still returning them untouched to rag_tool.run_agent — the agent
    loop's actual behavior (what it searches, what it does with results) is
    unmodified; only observation is added."""
    original = tools.retrieve

    def wrapper(name, query, loaded, top_k=tools.TOP_K):
        results = original(name, query, loaded, top_k)
        strategy_results.extend(results)
        return results

    return wrapper


def run_one_strategy(name: str) -> dict:
    tools.KB_REGISTRY["search_handbook"]["index_dir"] = DATA_DIR / f"faiss_index_{name}"
    loaded = tools.load_tools(["search_handbook"])
    tool_config = {"tools": [tools.tool_spec("search_handbook")], "toolChoice": {"auto": {}}}

    print(f"\n{'=' * 70}\n[{name}]  {loaded['search_handbook']['index'].ntotal} chunks\n{'=' * 70}")

    per_question = []
    right_source_hits = 0
    hit_eligible = 0
    answers_correct = 0
    for q in QUESTIONS:
        print(f'\nQ: "{q["question"]}"')
        call_results: list = []
        rag_tool.retrieve = capture_retrieve(call_results)
        answer = rag_tool.run_agent(q["question"], loaded, tool_config)
        rag_tool.retrieve = tools.retrieve  # restore, so the wrapper doesn't nest across questions

        retrieved_sources = {source for _cid, _score, _text, source in call_results}
        hit = q["expected_source"] is not None and q["expected_source"] in retrieved_sources
        if q["expected_source"] is not None:
            hit_eligible += 1
            right_source_hits += int(hit)
        correct = is_correct(q, answer)
        answers_correct += int(correct)

        print(f"A: {answer}")
        print(
            f"   expected_source={q['expected_source']!r}  retrieved_sources={sorted(retrieved_sources)}  "
            f"hit={hit}  correct={correct}"
        )
        per_question.append(
            {
                "question": q["question"],
                "expected_source": q["expected_source"],
                "retrieved": [
                    {"chunk_id": cid, "score": score, "source": source} for cid, score, _text, source in call_results
                ],
                "answer": answer,
                "right_source_hit": hit,
                "correct": correct,
                "refusal": is_refusal(answer),
            }
        )

    return {
        "strategy": name,
        "right_source_hits": right_source_hits,
        "hit_eligible": hit_eligible,
        "answers_correct": answers_correct,
        "total_questions": len(QUESTIONS),
        "per_question": per_question,
    }


def print_comparison_table(eval_results: list[dict]) -> None:
    header = (
        f"| {'strategy':<18} | {'#chunks':>7} | {'embed calls':>11} | {'nova calls':>10} | "
        f"{'~$':>8} | {'right-source hits':>18} | {'answers correct':>16} |"
    )
    sep = "|" + "|".join("-" * (len(c) + 2) for c in header.strip("|").split("|")) + "|"
    print("\n" + header)
    print(sep)
    for r in eval_results:
        b = BUILD_STATS[r["strategy"]]
        hits = f"{r['right_source_hits']}/{r['hit_eligible']}"
        correct = f"{r['answers_correct']}/{r['total_questions']}"
        print(
            f"| {r['strategy']:<18} | {b['chunks']:>7} | {b['embed_calls']:>11} | {b['nova_calls']:>10} | "
            f"${b['cost_usd']:>7.4f} | {hits:>18} | {correct:>16} |"
        )


def main() -> None:
    eval_results = []
    for name in STRATEGIES:
        eval_results.append(run_one_strategy(name))

    print_comparison_table(eval_results)

    out_path = Path(__file__).resolve().parent / "eval_results.json"
    out_path.write_text(json.dumps(eval_results, indent=2), encoding="utf-8")
    print(f"\nFull per-question results written to {out_path.name}")


if __name__ == "__main__":
    main()
