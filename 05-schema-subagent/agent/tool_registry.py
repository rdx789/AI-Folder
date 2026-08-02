"""Discovers every tool module in tools/ automatically — nothing is named by hand.

A tool module is any tools/*.py (except __init__.py) exposing TOOL_SPEC and handle().
"""
import importlib
import pkgutil
import sys

import tools


def discover_tools() -> tuple[list[dict], dict]:
    """Return (list of TOOL_SPEC dicts, {name: handle callable}).

    A broken tool module (import error, missing dependency) is skipped with a
    warning rather than crashing discovery for every other tool.
    """
    specs = []
    handlers = {}
    for _, module_name, _ in pkgutil.iter_modules(tools.__path__):
        try:
            module = importlib.import_module(f"tools.{module_name}")
        except Exception as exc:
            print(f"warning: skipping tools.{module_name} — failed to import: {exc}", file=sys.stderr)
            continue
        spec = getattr(module, "TOOL_SPEC", None)
        handle = getattr(module, "handle", None)
        if spec is None or handle is None:
            continue
        specs.append(spec)
        handlers[spec["toolSpec"]["name"]] = handle
    return specs, handlers
