"""Runs the pipeline over data/eval_questions.jsonl and scores it two ways:

  - retrieval (labeled): Recall@k and MRR, checking whether the reranked top-k's
    chunk `key`s contain the question's `expected_source` file(s).
  - generation (LLM judges): faithfulness / context-relevance / completeness on
    answerable questions, plus a refusal-accuracy check on access/unanswerable ones.

Call run_eval(label, **pipeline_kwargs) to score one config (e.g. a chunk size,
retrieve-N, or rerank top-k) — use it to baseline, then tweak and re-run to compare.
"""
import json
import statistics
from pathlib import Path

from env_check import validate_env

validate_env()

import pipeline  # noqa: E402
from judges import completeness, context_relevance, faithfulness, refused  # noqa: E402

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_questions.jsonl"


def load_questions() -> list[dict]:
    with open(QUESTIONS_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def recall_and_rank(top_keys: list[str], expected_source: list[str]) -> tuple[int, float]:
    """Recall@k (0/1: was ANY expected source retrieved) and reciprocal rank of the
    first hit (0.0 if none)."""
    for rank, key in enumerate(top_keys, start=1):
        if key in expected_source:
            return 1, 1.0 / rank
    return 0, 0.0


def score_question(q: dict, **pipeline_kwargs) -> dict:
    result = pipeline.run(q["question"], q["audience"], verbose=False, **pipeline_kwargs)
    answer = result["answer"]
    row = {"id": q["id"], "category": q.get("category", ""), "expect_refusal": q["expect_refusal"]}

    if q["expect_refusal"]:
        row["refused_correctly"] = refused(answer)
    else:
        recall, rr = recall_and_rank(result["top_keys"], q["expected_source"])
        row["recall@k"] = recall
        row["mrr"] = rr
        f_score, _ = faithfulness(q["question"], result["contexts"], answer)
        c_score, _ = context_relevance(q["question"], result["contexts"])
        comp_score, _ = completeness(q["question"], q["key_facts"], answer)
        row["faithfulness"] = f_score
        row["context_relevance"] = c_score
        row["completeness"] = comp_score
    return row


def run_eval(label: str, **pipeline_kwargs) -> list[dict]:
    questions = load_questions()
    rows = [score_question(q, **pipeline_kwargs) for q in questions]

    answerable = [r for r in rows if not r["expect_refusal"]]
    refusal_cases = [r for r in rows if r["expect_refusal"]]

    print(f"\n=== {label} ===")
    print(f"{'id':<20} {'cat':<12} {'recall@k':>8} {'mrr':>6} {'faith':>6} {'ctx_rel':>8} {'complete':>9}")
    for r in rows:
        if r["expect_refusal"]:
            print(f"{r['id']:<20} {r['category']:<12} {'refused=' + str(r['refused_correctly']):>8}")
        else:
            print(f"{r['id']:<20} {r['category']:<12} {r['recall@k']:>8} {r['mrr']:>6.2f} "
                  f"{r['faithfulness']:>6.2f} {r['context_relevance']:>8.2f} {r['completeness']:>9.2f}")

    if answerable:
        print(f"\nRecall@k:          {statistics.mean(r['recall@k'] for r in answerable):.2f}")
        print(f"MRR:               {statistics.mean(r['mrr'] for r in answerable):.2f}")
        print(f"Faithfulness:      {statistics.mean(r['faithfulness'] for r in answerable):.2f}")
        print(f"Context-relevance: {statistics.mean(r['context_relevance'] for r in answerable):.2f}")
        print(f"Completeness:      {statistics.mean(r['completeness'] for r in answerable):.2f}")
    if refusal_cases:
        acc = statistics.mean(1.0 if r["refused_correctly"] else 0.0 for r in refusal_cases)
        print(f"Refusal accuracy:  {acc:.2f}  ({len(refusal_cases)} cases)")

    return rows


if __name__ == "__main__":
    run_eval("baseline (N=20, k=4)")
