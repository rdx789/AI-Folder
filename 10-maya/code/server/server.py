"""Local policy/operational reads. Document searches use the authorized retriever.

The host graph must authorize operational calls using trusted caller configuration.
This local development server does not implement public-client authentication.
"""
if __name__ == '__main__' and not __package__:
    # Started by file path (python code/server/server.py): relative imports need the package, so
    # re-run this file as the installed module `maya.server.server`.
    import runpy
    try:
        runpy.run_module('maya.server.server', run_name='__main__', alter_sys=True)
    except ImportError as exc:  # runpy wraps the ModuleNotFoundError for the package
        if 'maya' not in (getattr(exc, 'name', None), getattr(exc.__cause__, 'name', None)):
            raise
        raise SystemExit("maya is not installed for this Python. Use the project's interpreter: "
                         ".venv/bin/python (or `source .venv/bin/activate`), or run ./setup.sh")
    raise SystemExit

import os
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP

from . import db
from ..paths import DATA_PATH, ENV_FILE

load_dotenv(ENV_FILE, override=False)

POLICIES_DIR = DATA_PATH / "policies"

PORT = int(os.environ.get("MAYA_MCP_PORT", "9878"))
mcp = FastMCP("maya-reads", host="127.0.0.1", port=PORT)


@mcp.tool()
def list_policies() -> list[str]:
    """List the names of all available NovaOps company policies.

    Call this first to discover which policies exist before fetching one.
    """
    return sorted(p.stem for p in POLICIES_DIR.glob("*.md"))


@mcp.tool()
def get_policy(name: str) -> str:
    """Return the full text of one NovaOps policy.

    Args:
        name: A policy name as returned by list_policies, e.g. 'equipment_policy'.
    """
    if name not in list_policies():
        raise ValueError("Unknown policy name")
    path = POLICIES_DIR / f"{name}.md"
    if not path.is_file():
        raise ValueError(f"No policy named {name!r}. Call list_policies for valid names.")
    return path.read_text(encoding="utf-8")


@mcp.tool()
def get_employee(query: str) -> list[dict]:
    """Look up NovaOps employees by employee id, email, or (partial) name.

    Args:
        query: An employee id (e.g. 'E001'), an email, or a name to match.
    """
    return db.get_employee(query)


@mcp.tool()
def check_software_subscription(software: str) -> list[dict]:
    """Check a SaaS subscription's seat usage — is it at or over its seat limit?

    Use this when someone can't get access to a tool (e.g. 'Webex says I'm not
    licensed') to see whether seats are available or the subscription is full.
    Also returns the annual cost and renewal date.

    Args:
        software: The software or vendor name, e.g. 'Webex' or 'Salesforce'.
    """
    return db.check_software_subscription(software)


@mcp.tool()
def list_onboarding_tasks(employee_id: str) -> list[dict]:
    """List an employee's onboarding checklist tasks and their statuses.

    The checklist is the system of record for onboarding — use it to answer
    "what's left?" and to see which tasks are blocked.

    Args:
        employee_id: The employee id, e.g. 'E001'.
    """
    return db.list_onboarding_tasks(employee_id)


@mcp.tool()
def check_asset_inventory(asset_type: str = "", location: str = "") -> list[dict]:
    """Check hardware inventory — what equipment is on hand and where.

    Use this for equipment questions (laptops, monitors, docks). Both arguments are
    optional filters; omit them to list everything.

    Args:
        asset_type: e.g. 'Laptop' or 'Monitor'. Omit for all types.
        location: e.g. 'Israel'. Omit for all locations.
    """
    return db.check_asset_inventory(asset_type, location)


@mcp.tool()
def list_employee_tickets(employee_id: str, status: str = "") -> list[dict]:
    """List the IT tickets raised by or for one employee.

    Use this to see what has already been reported before opening anything new.

    Args:
        employee_id: The employee id, e.g. 'E010'.
        status: Optional status filter, e.g. 'open'. Omit for all statuses.
    """
    return db.list_employee_tickets(employee_id, status)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Maya's read-only operational MCP server (streamable HTTP).")
    parser.add_argument("--port", type=int, default=PORT, help=f"port to listen on (default {PORT}, or MAYA_MCP_PORT)")
    args = parser.parse_args(argv)  # --help exits here, before the server starts
    mcp.settings.port = args.port
    print(f"Maya read-only MCP server → http://127.0.0.1:{args.port}/mcp (Ctrl-C to stop)")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
