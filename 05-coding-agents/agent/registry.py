"""Auto-discovers tools from the tools/ package — no tool is named here."""
import importlib
import pkgutil

import tools


def discover_tools():
    """Return (tool_specs, handlers) by importing every module in tools/."""
    tool_specs = []
    handlers = {}
    for _, module_name, is_pkg in pkgutil.iter_modules(tools.__path__):
        if is_pkg or module_name.startswith("_"):
            continue
        module = importlib.import_module(f"tools.{module_name}")
        if hasattr(module, "TOOL_SPEC") and hasattr(module, "handle"):
            name = module.TOOL_SPEC["toolSpec"]["name"]
            tool_specs.append(module.TOOL_SPEC)
            handlers[name] = module.handle
    return tool_specs, handlers
