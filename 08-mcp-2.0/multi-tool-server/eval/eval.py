"""Evaluate the agent's TOOL USE against a labelled question set.

Exercise 3's agent picks tools on its own. This asks: does it pick the RIGHT ones,
skip the wrong ones, pass the right arguments — and then actually use what comes
back? It runs each question in ../../data/eval_tool_use.jsonl through the same MCP
tool-use loop as agent.py, but instrumented to record the TRAJECTORY (every tool
call and its result), then scores six metrics split across two files:

  PROCESS  (checks.py — deterministic asserts on the trajectory, no model call)
    selection   — did it call the tools the task needed?      (recall)
    necessity   — did it avoid tools it didn't need?          (precision)
    arg_correct — were the arguments right? (the disambiguation bug)

  OUTCOME  (judges.py — one Nova LLM-judge call each)
    utilization — is the answer built from the tool results?
    faithful    — is the answer supported by the tool results?
    relevance   — does the answer address the question?

The six form a chain: selection → arguments → utilization → faithful → relevance.
A break upstream caps everything downstream — which is the whole reason to score the
trajectory and not just the final answer.

Run (with server.py already running in another terminal):
    python eval.py            # the whole set
    python eval.py 3          # only the first 3 cases (a quick, cheap smoke test)

Every case makes several Bedrock calls (the agent loop + three judges), so the full
set takes a few minutes.
"""
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import boto3
from dotenv import find_dotenv, load_dotenv
from mcp import Client

# Only the eval's own siblings — no import from agent.py or client.py. A real eval
# pipeline is a separate system from the agent it grades, so this folder stands alone.
from checks import tool_argument_correctness, tool_necessity, tool_selection
from judges import answer_relevance, result_faithfulness, result_utilization

load_dotenv(find_dotenv())

# --- Standalone eval wiring -------------------------------------------------------
# The eval re-runs the agent loop itself (see run_agent), so it needs the same Bedrock
# and MCP plumbing the agent uses — carried HERE rather than imported, so eval/ is a
# self-contained pipeline you could lift out and point at any running MCP server.
# Trade-off: this duplicates the agent's seam (get_client / mcp_tools_to_bedrock) and a
# couple of helpers; if you change how the agent talks to Bedrock, mirror it here.
REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-2-lite-v1:0")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:9876/mcp")
MAX_TURNS = 10

# Must match agent.py's SYSTEM_PROMPT — the eval only means something if it runs the
# agent under the SAME instructions that ship. Keep the two copies in sync.
SYSTEM_PROMPT = (
    "You are the NovaOps internal assistant. Answer employee questions using the "
    "MCP tools available to you. Discover what you need — list things before you "
    "fetch them — then give a clear, final answer grounded in what the tools return."
)


def get_client():
    """boto3 Bedrock runtime client (SigV4 auth from the AWS_* env vars)."""
    return boto3.client("bedrock-runtime", region_name=REGION)


def mcp_tools_to_bedrock(mcp_tools) -> list[dict]:
    """Convert MCP tool definitions into the Bedrock Converse `toolConfig` shape.
    The same bridge as the agent's client.py, copied here to keep the eval standalone.
    (tool.input_schema is snake_case in MCP 2.0; the outer "inputSchema" is Bedrock's key.)"""
    return [
        {
            "toolSpec": {
                "name": tool.name,
                "description": tool.description or tool.name,
                "inputSchema": {"json": tool.input_schema},
            }
        }
        for tool in mcp_tools
    ]


def result_text(result) -> str:
    """Join the text of EVERY content block in an MCP tool result (a tool that returns a
    list comes back as one block per item, so content[0] alone would drop the rest)."""
    return "\n".join(block.text for block in result.content if hasattr(block, "text"))


def final_text(message: dict) -> str:
    """Pull the model's plain-text answer out of a Converse message."""
    return "".join(b["text"] for b in message["content"] if "text" in b).strip()


# data/ sits two levels up now (eval/ -> 03-multi-tool-server/ -> code/).
QUESTIONS = Path(__file__).resolve().parent.parent.parent / "data" / "eval_tool_use.jsonl"

# The six metric columns, in chain order. None in a row means "not applicable to this
# case" (e.g. no arg checks declared, or no tools were called to judge results on).
METRICS = ["selection", "necessity", "arg_correct", "utilization", "faithful", "relevance"]

# Full column names (for headers) and a one-line "what it does" (for the legend).
LABELS = {
    "selection":   "Selection",
    "necessity":   "Necessity",
    "arg_correct": "Arg-correct",
    "utilization": "Utilization",
    "faithful":    "Faithfulness",
    "relevance":   "Relevance",
}
LEGEND = {
    "selection":   ("check", "recall — did it call the tools the task needed?"),
    "necessity":   ("check", "precision — did it avoid tools it didn't need?"),
    "arg_correct": ("check", "were the tool arguments right (e.g. the right employee_id)?"),
    "utilization": ("judge", "is the answer built from what the tools returned?"),
    "faithful":    ("judge", "is the answer consistent with the tool results?"),
    "relevance":   ("judge", "does the answer address the question?"),
}


def load_cases(limit: int | None) -> list[dict]:
    cases = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    return cases[:limit] if limit else cases


async def run_agent(client: Client, tool_config: dict, question: str) -> tuple[str, list[dict], list[tuple[str, str]]]:
    """Run the agent's tool-use loop for one question, INSTRUMENTED.

    Same three moves as agent.py's answer() — ask the model, run any tools it wants,
    feed the results back — but here we also record what happened so we can score it:
      - trajectory:   [{"name", "input"}, ...] every tool call, in order
      - tool_results: [(name, text), ...]      what each call returned
    (agent.py itself stays unchanged; instrumenting the loop is the eval's job.)
    """
    bedrock = get_client()
    messages = [{"role": "user", "content": [{"text": question}]}]
    trajectory: list[dict] = []
    tool_results: list[tuple[str, str]] = []

    for _ in range(MAX_TURNS):
        response = bedrock.converse(
            modelId=MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig=tool_config,
        )
        output = response["output"]["message"]
        messages.append(output)

        if response["stopReason"] != "tool_use":
            return final_text(output), trajectory, tool_results

        result_blocks = []
        for block in output["content"]:
            if "toolUse" not in block:
                continue
            call = block["toolUse"]
            trajectory.append({"name": call["name"], "input": call["input"]})
            result = await client.call_tool(call["name"], call["input"])
            text = result_text(result)
            tool_results.append((call["name"], text))
            result_blocks.append({
                "toolResult": {
                    "toolUseId": call["toolUseId"],
                    "content": [{"text": text}],
                    **({"status": "error"} if result.is_error else {}),
                }
            })
        messages.append({"role": "user", "content": result_blocks})

    return "(Reached the tool-call limit without a final answer.)", trajectory, tool_results


def score_case(case: dict, answer: str, trajectory: list[dict], tool_results: list[tuple[str, str]]) -> tuple[dict, dict]:
    """Score one case on all six metrics. Returns (scores, reasons): scores are floats
    (or None where a metric doesn't apply), reasons are the checker/judge's one-line
    justification for each — kept so the worst-case report can explain the number."""
    called_list = [t["name"] for t in trajectory]
    expected = set(case["expected_tools"])
    optional = set(case.get("optional_tools", []))

    scores: dict[str, float | None] = {}
    reasons: dict[str, str] = {}

    scores["selection"], reasons["selection"] = tool_selection(set(called_list), expected)
    scores["necessity"], reasons["necessity"] = tool_necessity(called_list, expected, optional)
    arg = tool_argument_correctness(trajectory, case.get("arg_checks", []))
    scores["arg_correct"], reasons["arg_correct"] = arg if arg else (None, "no argument checks for this case")

    # The result-based judges need results to judge; skip them when no tool ran.
    if tool_results:
        results_text = "\n\n".join(f"[{name}]\n{text}" for name, text in tool_results)
        scores["utilization"], reasons["utilization"] = result_utilization(case["question"], results_text, answer)
        scores["faithful"], reasons["faithful"] = result_faithfulness(case["question"], results_text, answer)
    else:
        scores["utilization"], reasons["utilization"] = None, "no tools called — nothing to judge"
        scores["faithful"], reasons["faithful"] = None, "no tools called — nothing to judge"
    scores["relevance"], reasons["relevance"] = answer_relevance(case["question"], answer)
    return scores, reasons


async def evaluate(client: Client, tool_config: dict, cases: list[dict]) -> list[dict]:
    """Run every case, keeping the full record (scores, reasons, answer, trajectory)
    so report() can print per-question rows and drill into the worst case. Prints a
    progress heartbeat because each case is several Bedrock calls."""
    records = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i:>2}/{len(cases)}] {case['id']} ({case['difficulty']}) ...", flush=True)
        answer, trajectory, tool_results = await run_agent(client, tool_config, case["question"])
        scores, reasons = score_case(case, answer, trajectory, tool_results)
        records.append({
            "case": case, "scores": scores, "reasons": reasons,
            "answer": answer, "trajectory": trajectory, "tool_results": tool_results,
        })
    return records


def _fmt(v: float | None) -> str:
    return " -- " if v is None else f"{v:.2f}"


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _group_means(subset: list[dict]) -> dict:
    """Mean of each metric across a set of records, skipping the None (N/A) cells."""
    return {m: _mean([r["scores"][m] for r in subset]) for m in METRICS}


def _case_mean(scores: dict) -> float:
    """One case's overall score — mean of the metrics that applied. Used to pick the
    single worst case. A case with no applicable metrics counts as a perfect 1.0."""
    present = [v for v in scores.values() if v is not None]
    return sum(present) / len(present) if present else 1.0


def report(records: list[dict]) -> None:
    IDW, DIFW, MW = 20, 8, 13          # id / difficulty / per-metric column widths
    GW = IDW + DIFW                     # group label spans id+diff so columns align
    LINEW = GW + MW * len(METRICS)
    metric_header = "".join(f"{LABELS[m]:>{MW}}" for m in METRICS)

    def cells(scores: dict) -> str:
        return "".join(f"{_fmt(scores[m]):>{MW}}" for m in METRICS)

    # 1) Per-question results — the discriminating cases (medium + hard).
    print("\n" + "=" * LINEW)
    print("PER-QUESTION RESULTS — medium + hard")
    print("=" * LINEW)
    print(f"{'id':<{IDW}}{'diff':<{DIFW}}" + metric_header)
    print("-" * LINEW)
    for r in records:
        c = r["case"]
        if c["difficulty"] not in ("medium", "hard"):
            continue
        print(f"{c['id']:<{IDW}}{c['difficulty']:<{DIFW}}" + cells(r["scores"]))
        called = ", ".join(t["name"] for t in r["trajectory"]) or "(none)"
        print(f"    tools: {called}")
    print("(easy cases run too, but omitted here — see the ALL / easy rows below.)")

    # 2) Averages: overall, then by difficulty so the easy→hard gap is visible.
    print("\n" + "=" * LINEW)
    print("AVERAGES  (a metric is skipped where it doesn't apply)")
    print("=" * LINEW)
    print(f"{'group':<{GW}}" + metric_header)
    print("-" * LINEW)
    print(f"{'ALL':<{GW}}" + cells(_group_means(records)))
    print("-" * LINEW)
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_diff[r["case"]["difficulty"]].append(r)
    for diff in ("easy", "medium", "hard"):
        if by_diff[diff]:
            print(f"{diff:<{GW}}" + cells(_group_means(by_diff[diff])))

    # 3) Legend — what each column measures, in one line each.
    print("\nWhat each check/judge does:")
    for m in METRICS:
        kind, meaning = LEGEND[m]
        print(f"  {LABELS[m]:<12} ({kind}) {meaning}")

    # 4) The single worst case, with the dataset's expectations, what the agent did,
    #    its answer, and every metric's score + the reason behind it.
    worst = min(records, key=lambda r: _case_mean(r["scores"]))
    c, scores, reasons = worst["case"], worst["scores"], worst["reasons"]
    print("\n" + "=" * LINEW)
    print(f"WORST CASE — {c['id']} ({c['difficulty']})   overall {_case_mean(scores):.2f}")
    print("=" * LINEW)
    print(f"Question : {c['question']}")
    print(f"\nDataset says the trajectory should be:")
    print(f"  expected tools : {c['expected_tools'] or '(none)'}")
    if c.get("optional_tools"):
        print(f"  optional tools : {c['optional_tools']}")
    if c.get("arg_checks"):
        print(f"  arg checks     : {c['arg_checks']}")
    print("\nWhat the agent actually did:")
    if worst["trajectory"]:
        for i, t in enumerate(worst["trajectory"], 1):
            print(f"  {i}. {t['name']}({json.dumps(t['input'], ensure_ascii=False)})")
    else:
        print("  (no tools called)")
    print("\nAnswer:")
    for line in worst["answer"].splitlines() or [""]:
        print(f"  {line}")
    print("\nScore vs each metric (with the checker/judge's reason):")
    for m in METRICS:
        print(f"  {LABELS[m]:<12} {_fmt(scores[m]):>5}  {reasons[m]}")
    if c.get("note"):
        print(f"\nWhy this case exists: {c['note']}")


async def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    cases = load_cases(limit)
    # High-level Client (MCP 2.0): connect + initialize on context entry, one client for
    # the whole eval run — the same MCP plumbing the agent uses, so the eval grades the
    # agent under identical conditions.
    async with Client(MCP_SERVER_URL) as client:
        tools = (await client.list_tools()).tools
        tool_config = {"tools": mcp_tools_to_bedrock(tools)}
        print(f"Connected. Evaluating {len(cases)} cases against tools: "
              f"{', '.join(t.name for t in tools)}\n")
        report(await evaluate(client, tool_config, cases))


if __name__ == "__main__":
    asyncio.run(main())
