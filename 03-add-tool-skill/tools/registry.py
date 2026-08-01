"""Discovers every tool module in tools/ and exposes specs + dispatch."""

import importlib
import pkgutil

import tools

_TOOLS = {}


def _discover():
    if _TOOLS:
        return _TOOLS
    for module_info in pkgutil.iter_modules(tools.__path__):
        name = module_info.name
        if name in ("registry",) or name.startswith("_"):
            continue
        module = importlib.import_module(f"tools.{name}")
        if hasattr(module, "TOOL_SPEC") and hasattr(module, "handle"):
            tool_name = module.TOOL_SPEC["toolSpec"]["name"]
            _TOOLS[tool_name] = module
    return _TOOLS


def get_tool_config():
    tools_list = [module.TOOL_SPEC for module in _discover().values()]
    return {"tools": tools_list} if tools_list else None


def dispatch(tool_name: str, tool_input: dict):
    module = _discover()[tool_name]
    return module.handle(**tool_input)