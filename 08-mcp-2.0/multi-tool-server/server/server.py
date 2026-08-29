"""
NovaOps MCP server — policies + knowledge base + database.

Nine tools over three backends, and to the agent they are all just a name and a
schema — that uniformity is the whole point of MCP:

  policy files (data/policies/*.md)
    list_policies() / get_policy(name)

  knowledge base (OpenSearch Serverless, via rag.py)
    search_knowledge_base(query)

  NovaOps database (SQLite, via db.py)
    get_employee(query)                    read
    check_software_subscription(software)  read
    check_asset_inventory(type, location)  read
    list_employee_tickets(employee_id)     read
    create_ticket(...)                     WRITE — file an IT support ticket
    create_access_request(...)             WRITE — file a system access request

agent.py and client.py never change as tools are added: only this file grows.

Run:
  python server.py        # serves at http://127.0.0.1:9876/mcp
"""

from pathlib import Path

from mcp.server import MCPServer

import db
import rag

POLICIES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "policies"

# MCP 2.0: host/port move from the constructor to run() (see bottom of file).
mcp = MCPServer("novaops-assistant")


@mcp.tool()
def list_policies() -> list[str]:
    """List the names of all NovaOps company policies — the official rules and
    requirements (what's allowed, what's required). These are formal policy
    documents, NOT how-to or troubleshooting guides.

    Call this first to see which policies exist, then fetch one with get_policy.
    """
    return sorted(p.stem for p in POLICIES_DIR.glob("*.md"))


@mcp.tool()
def get_policy(name: str) -> str:
    """Return the full text of one NovaOps policy — the official rules and
    requirements on a topic (not step-by-step instructions).

    Args:
        name: A policy name as returned by list_policies, e.g. 'remote_work_policy'.
    """
    path = POLICIES_DIR / f"{name}.md"
    if not path.is_file():
        raise ValueError(f"No policy named {name!r}. Call list_policies for valid names.")
    return path.read_text(encoding="utf-8")


@mcp.tool()
def search_knowledge_base(query: str) -> list[dict]:
    """Search the NovaOps IT knowledge base for articles relevant to a question.

    Use this for how-to and troubleshooting questions — password reset, VPN access,
    MFA, laptop provisioning, and the like. Returns the most relevant articles.

    Args:
        query: A natural-language description of the problem or question.
    """
    return rag.search_kb(query)


@mcp.tool()
def get_employee(query: str) -> list[dict]:
    """Look up NovaOps employees by employee id, email, or (partial) name.

    Args:
        query: An employee id (e.g. 'E009'), an email, or a name to match.
    """
    return db.get_employee(query)


@mcp.tool()
def check_software_subscription(software: str) -> list[dict]:
    """Check a SaaS subscription's seat usage — is it at or over its seat limit?

    Use this when someone can't get access to a tool (e.g. 'Webex says I'm not
    licensed') to see whether seats are available or the subscription is full.

    Args:
        software: The software or vendor name, e.g. 'Webex' or 'Salesforce'.
    """
    return db.check_software_subscription(software)


@mcp.tool()
def check_asset_inventory(asset_type: str = "", location: str = "") -> list[dict]:
    """Check hardware inventory — what spare equipment is in stock and unassigned.

    Use this for "are there any spare laptops / monitors / phones" questions, and
    filter by location when the question names one (e.g. 'in the US').

    Args:
        asset_type: The kind of asset, e.g. 'Laptop' or 'Monitor'. Optional.
        location: Where the stock should be, e.g. 'United Kingdom'. Optional.
    """
    return db.check_asset_inventory(asset_type, location)


@mcp.tool()
def list_employee_tickets(employee_id: str) -> list[dict]:
    """List every support ticket an employee has filed, newest first.

    Args:
        employee_id: The employee id, e.g. 'E010'.
    """
    return db.list_employee_tickets(employee_id)


@mcp.tool()
def create_ticket(
    employee_id: str, subject: str, description: str, category: str
) -> dict:
    """File a support ticket for an employee (status: open).

    This WRITES to the database. Use it after confirming who the employee is
    (get_employee). The ticket is routed to a team based on its category; there is
    no approval step — an open ticket just enters the queue.

    Args:
        employee_id: The employee the ticket is for, e.g. 'E010'.
        subject: A short one-line summary of the issue.
        description: The full description of the problem.
        category: The issue category, e.g. 'Hardware', 'Access', 'Network', 'Security'.
    """
    return db.create_ticket(employee_id, subject, description, category)


@mcp.tool()
def create_access_request(
    employee_id: str, software: str, business_justification: str
) -> dict:
    """File an access request for an employee to a system (status: pending_approval).

    This WRITES to the database. Use it after confirming who the employee is
    (get_employee) and that access is warranted. The request is created pending a
    human approval decision — it does not grant access on its own.

    Args:
        employee_id: The employee id the access is for, e.g. 'E010'.
        software: The software or system to request, e.g. 'Webex'.
        business_justification: A short reason for the request.
    """
    return db.create_access_request(employee_id, software, business_justification)


if __name__ == "__main__":
    # Confirm the KB backend is reachable before serving. The embedding model calls
    # for search happen here on the server side, never in the agent.
    print("Checking the IT KB (OpenSearch Serverless)...")
    print(f"KB index reachable — {rag.warm_index()} chunks searchable.")
    print("NovaOps MCP server → http://127.0.0.1:9876/mcp  (Ctrl-C to stop)")
    mcp.run(transport="streamable-http", host="127.0.0.1", port=9876)
