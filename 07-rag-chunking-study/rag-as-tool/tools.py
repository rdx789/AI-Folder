"""The search tools: their schemas, the operation behind them, and which roles
may use which.

rag_tool.py hands the model a single tool from here; rag_by_role.py hands each
role a different subset. Keeping the toolSpecs and the role map in one place is
what makes the access-control model easy to see (and easy to break, in Lesson 7).
"""
from pathlib import Path

from retrieval import TOP_K, load_kb, search

CODE_DIR = Path(__file__).resolve().parents[1]
BUILD_CMD = "python 02-rag-as-tool/ingest_kbs.py"

# tool name -> the corpus it searches. Each corpus is its own index, built by
# ingest_kbs.py as faiss_index_<subfolder>. `label` tags chunks in the prompt so
# handbook and playbook chunks are distinguishable (chunk ids are per-corpus).
KB_REGISTRY = {
    "search_handbook": {
        "index_dir": CODE_DIR / "data" / "faiss_index_handbook",
        "label": "handbook",
        "description": (
            "The NovaOps employee handbook — company-wide policies and facts for "
            "all employees: pay, benefits, vacation and leave, severance, the "
            "moonlighting policy, device rules, and job titles and levels."
        ),
    },
    "search_manager_playbook": {
        "index_dir": CODE_DIR / "data" / "faiss_index_manager_playbook",
        "label": "manager playbook",
        "description": (
            "The NovaOps manager playbook — how-to guidance for managers working "
            "with their reports: running 1:1s, giving feedback, coaching, hiring, "
            "onboarding a report, performance reviews, promotions, and handling "
            "underperformance or termination conversations."
        ),
    },
}

# Which tools each role's agent is handed — the ENTIRE access-control model of
# rag_by_role.py. The employee agent never even learns the playbook tool exists,
# so there's nothing to jailbreak; but it only works because each corpus is its
# own index. Lesson 7 replaces this with one index + a metadata filter.
ROLE_TOOLS = {
    "employee": ["search_handbook"],
    "manager": ["search_handbook", "search_manager_playbook"],
}


def tool_spec(name: str) -> dict:
    """The Bedrock toolSpec for one search tool, as rag_tool.py's agent loop uses
    it. All tools share the same single-`query` schema; only the name and
    description differ."""
    return {
        "toolSpec": {
            "name": name,
            "description": KB_REGISTRY[name]["description"],
            "inputSchema": {
                "json": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string", "description": "What to look up, e.g. 'vacation policy'."}
                    },
                }
            },
        }
    }


def plan_tool(allowed: list[str]) -> dict:
    """A FORCED planning tool, used by rag_by_role.py. The model fills `searches`
    with one entry per corpus it needs (empty for none). The `source` enum is
    restricted to THIS role's tools, so a plan can't route to a corpus the role
    may not read — that restriction is the access control in the planned flow."""
    source_desc = "Which corpus to search. Options:\n" + "\n".join(
        f"- {name}: {KB_REGISTRY[name]['description']}" for name in allowed
    )
    return {
        "toolSpec": {
            "name": "plan_sources",
            "description": (
                "Plan which knowledge sources to search before answering. List "
                "EVERY source the question needs — a question with a company-policy "
                "part AND a manager how-to part needs BOTH. Return an empty list "
                "for greetings or questions about what you can do."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["searches"],
                    "properties": {
                        "searches": {
                            "type": "array",
                            "description": "One entry per source to search; empty if none needed.",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["source", "query"],
                                "properties": {
                                    "source": {"type": "string", "enum": allowed, "description": source_desc},
                                    "query": {"type": "string", "description": "What to look up in that source."},
                                },
                            },
                        }
                    },
                }
            },
        }
    }


def load_tools(names: list[str]) -> dict:
    """Load the index behind each named tool. Returns name -> loaded KB, so the
    demo only opens the corpora its tools actually cover."""
    return {name: load_kb(KB_REGISTRY[name]["index_dir"], BUILD_CMD) for name in names}


def label(name: str) -> str:
    """Short human tag for a tool's corpus, e.g. 'handbook'."""
    return KB_REGISTRY[name]["label"]


def retrieve(name: str, query: str, loaded: dict, top_k: int = TOP_K) -> list[tuple[int, float, str, str]]:
    """Search the corpus behind `name` with `query`. Returns scored chunks
    (chunk_id, cosine similarity, text, source_file). source_file is None for
    indexes built without per-chunk source tracking."""
    return search(loaded[name], query, top_k)


def format_context(name: str, results: list[tuple[int, float, str, str]]) -> str:
    """Format retrieved chunks as prompt context, each tagged with its corpus
    label and source filename (when known) so the model can tell handbook
    chunks from playbook chunks — the chunk ids are per-corpus and would
    otherwise collide — and can cite the exact file an answer came from."""
    tags = []
    for cid, _score, text, source in results:
        tag = f"{label(name)} chunk {cid}" + (f", source: {source}" if source else "")
        tags.append(f"[{tag}]\n{text}")
    return "\n\n".join(tags)
