"""
Loading the prepared sessions.

The dataset is fixed and replayed identically against every stage — that is the
only reason the token numbers mean anything. Nothing here is clever; it exists so
no stage grows its own copy of the loading code.
"""

import json
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent / "data" / "sessions.json"


def load_sessions(only: str | None = None) -> list[dict]:
    """Return the prepared sessions, optionally narrowed by comma-separated prefixes.

    Each value matches a prefix, so 'S2' finds 'S2-onboarding-maya' and
    'S6,S7,S8' selects the controlled noise curve.
    """
    sessions = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["sessions"]
    if only:
        prefixes = tuple(value.strip().lower() for value in only.split(",") if value.strip())
        sessions = [s for s in sessions if s["id"].lower().startswith(prefixes)]
        if not sessions:
            raise SystemExit(f"No session id starts with any prefix in {only!r}.")
    return sessions
