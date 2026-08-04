"""Shared JSON loader for tools/ — not a tool itself (leading underscore)."""
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_ROOT, "data")
_cache = {}


def load(name):
    """Load data/<name>.json, caching the parsed result for the process lifetime."""
    if name not in _cache:
        path = os.path.join(_DATA_DIR, f"{name}.json")
        with open(path) as f:
            _cache[name] = json.load(f)
    return _cache[name]
