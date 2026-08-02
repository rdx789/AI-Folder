"""Validate one tool file and run its tests — the "try to break it" gate.

    python .claude/skills/create-agent-tool/scripts/check_tool.py <tool_name>

Checks that tools/<tool_name>.py has a well-formed TOOL_SPEC (the Bedrock
toolConfig shape: name, description, and an inputSchema.json with
additionalProperties:false, a required list, and a description on every
property) plus a handle(), then runs tests/test_<tool_name>.py under pytest.
Exits non-zero on any failure so the skill knows to fix and re-run.
"""
import importlib
import subprocess
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(1)


def check_spec(name: str) -> None:
    sys.path.insert(0, str(Path.cwd()))
    try:
        mod = importlib.import_module(f"tools.{name}")
    except Exception as e:  # import error = the tool file is broken
        fail(f"could not import tools/{name}.py: {e}")

    spec = getattr(mod, "TOOL_SPEC", None)
    if not spec:
        fail(f"tools/{name}.py has no TOOL_SPEC")
    ts = spec.get("toolSpec", {})
    for key in ("name", "description", "inputSchema"):
        if not ts.get(key):
            fail(f"toolSpec missing '{key}'")
    schema = ts["inputSchema"].get("json", {})
    if schema.get("additionalProperties") is not False:
        fail("inputSchema must set additionalProperties: false")
    if "required" not in schema:
        fail("inputSchema must have a 'required' list")
    props = schema.get("properties", {})
    if not props:
        fail("inputSchema has no properties")
    for field, p in props.items():
        if not p.get("description"):
            fail(f"field '{field}' has no description")
    if not hasattr(mod, "handle"):
        fail(f"tools/{name}.py has no handle()")
    print(f"✓ TOOL_SPEC for '{name}' is well-formed")


def run_tests(name: str) -> None:
    test_file = Path("tests") / f"test_{name}.py"
    if not test_file.exists():
        fail(f"missing {test_file}")
    result = subprocess.run([sys.executable, "-m", "pytest", str(test_file), "-q"])
    if result.returncode != 0:
        fail(f"tests failed for '{name}'")
    print(f"✓ tests pass for '{name}'")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        fail("usage: check_tool.py <tool_name>")
    tool = sys.argv[1]
    check_spec(tool)
    run_tests(tool)
    print(f"✓ {tool} is good")
