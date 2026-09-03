"""Discover the NovaOps MCP server's tools at runtime — no hand-written schemas."""

from langchain_mcp_adapters.client import MultiServerMCPClient

from config import MCP_SERVER_URL

_CONNECTIONS = {
    "novaops": {
        "url": MCP_SERVER_URL,
        "transport": "streamable_http",
    }
}


async def discover_tools() -> dict:
    """Return {tool_name: BaseTool} for every tool the server exposes.

    Raises RuntimeError with a clear message if the server can't be reached.
    """
    client = MultiServerMCPClient(_CONNECTIONS)
    try:
        tools = await client.get_tools()
    except Exception as exc:  # unreachable server, handshake failure, etc.
        raise RuntimeError(
            f"Could not reach the NovaOps MCP server at {MCP_SERVER_URL}. "
            f"Start it with `python server/server.py` first. ({exc})"
        ) from exc

    if not tools:
        raise RuntimeError(
            f"The MCP server at {MCP_SERVER_URL} returned no tools — is it the right server?"
        )
    return {tool.name: tool for tool in tools}
