"""Shared config, paths, and Bedrock call/cost accounting for the chunking study.

Every chunker/indexer in this package imports from here so call counts and
$ costs are tracked the same way regardless of strategy.
"""
import json
import os
from pathlib import Path

import boto3
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

STUDY_DIR = Path(__file__).resolve().parent
HOMEWORK_DIR = STUDY_DIR.parent
# Handbook source lives in the sibling lab, not copied — see PLAN.md "Path wiring note".
HANDBOOK_DIR = HOMEWORK_DIR.parent / "code" / "data" / "handbook"
DATA_DIR = HOMEWORK_DIR / "data"

client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
EMBEDDING_MODEL_ID = os.environ["BEDROCK_EMBEDDING_MODEL_ID"]
NOVA_MODEL_ID = os.environ["BEDROCK_MODEL_ID"]

# Bedrock on-demand pricing, USD per 1,000 tokens (us-east-1, checked 2026-08-09
# via web search — see PLAN.md). Titan Text Embeddings V2 has no output-token price
# (embeddings have no generated tokens); Nova 2 Lite is priced $0.30/$2.50 per 1M
# input/output tokens = $0.0003/$0.0025 per 1K.
PRICE_PER_1K_TOKENS = {
    "titan_embed_input": 0.00002,
    "nova_lite_input": 0.0003,
    "nova_lite_output": 0.0025,
}


class CallTracker:
    """Counts Bedrock calls and tokens for one index build, and estimates $ cost."""

    def __init__(self, strategy: str):
        self.strategy = strategy
        self.embed_calls = 0
        self.embed_input_chars = 0  # Titan: no usage field, so cost is estimated from input length
        self.nova_calls = 0
        self.nova_input_tokens = 0
        self.nova_output_tokens = 0

    def record_embed(self, text: str) -> None:
        """Log one Titan embedding call and the character length of its input
        (Titan returns no usage field, so cost is later estimated from this)."""
        self.embed_calls += 1
        self.embed_input_chars += len(text)

    def record_nova(self, usage: dict) -> None:
        """Log one Nova call using Bedrock's own reported input/output token
        counts (the "usage" field of a converse() response)."""
        self.nova_calls += 1
        self.nova_input_tokens += usage.get("inputTokens", 0)
        self.nova_output_tokens += usage.get("outputTokens", 0)

    def estimated_titan_tokens(self) -> int:
        """Estimate Titan input tokens at ~4 chars/token — the standard rough
        conversion AWS docs use for English text, since Titan embeddings give
        no exact usage field to read from."""
        return round(self.embed_input_chars / 4)

    def cost_usd(self) -> float:
        """Estimated $ cost of this build: Titan embedding calls (from the
        char-based token estimate) plus Nova calls (from exact usage tokens),
        both priced at PRICE_PER_1K_TOKENS."""
        titan_cost = (self.estimated_titan_tokens() / 1000) * PRICE_PER_1K_TOKENS["titan_embed_input"]
        nova_cost = (self.nova_input_tokens / 1000) * PRICE_PER_1K_TOKENS["nova_lite_input"] + (
            self.nova_output_tokens / 1000
        ) * PRICE_PER_1K_TOKENS["nova_lite_output"]
        return titan_cost + nova_cost

    def summary(self) -> dict:
        """This tracker's counts and estimated cost as a plain dict, for
        printing or for the cross-strategy comparison table."""
        return {
            "strategy": self.strategy,
            "embed_calls": self.embed_calls,
            "nova_calls": self.nova_calls,
            "nova_input_tokens": self.nova_input_tokens,
            "nova_output_tokens": self.nova_output_tokens,
            "estimated_titan_tokens": self.estimated_titan_tokens(),
            "cost_usd": round(self.cost_usd(), 6),
        }

    def print_summary(self) -> None:
        """One-line human-readable printout of summary()."""
        s = self.summary()
        print(
            f"  [{s['strategy']}] embed calls: {s['embed_calls']} | nova calls: {s['nova_calls']} "
            f"| ~titan tokens: {s['estimated_titan_tokens']} | nova tokens: "
            f"{s['nova_input_tokens']}in/{s['nova_output_tokens']}out | est. cost: ${s['cost_usd']:.6f}"
        )


def embed_text(text: str, tracker: CallTracker) -> list[float]:
    """Embed one chunk with Titan (unit-normalized, so index search can use
    plain dot product as cosine similarity), logging the call on tracker."""
    tracker.record_embed(text)
    response = client.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True}),
    )
    return json.loads(response["body"].read())["embedding"]


def call_nova(system: str, user: str, tracker: CallTracker, max_tokens: int = 500) -> str:
    """One-shot Nova 2 Lite call (used by the semantic chunker), tracked for cost."""
    response = client.converse(
        modelId=NOVA_MODEL_ID,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
    )
    tracker.record_nova(response["usage"])
    message = response["output"]["message"]
    return "".join(block["text"] for block in message["content"] if "text" in block)


def load_handbook_files() -> list[tuple[str, str]]:
    """(filename, text) for every handbook .md file, sorted for reproducibility."""
    return [(path.name, path.read_text(encoding="utf-8")) for path in sorted(HANDBOOK_DIR.glob("*.md"))]
