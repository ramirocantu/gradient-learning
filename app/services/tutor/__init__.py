"""Tutor-facing service functions, surfaced over HTTP for the MCP server.

Moved from mcp_server/tools/ in ticket 9.0 when the MCP server pivoted from
library-mode (direct SQLAlchemy + Anthropic SDK access) to HTTP-client-mode.
The backend now owns the data layer; the MCP server is a thin httpx proxy.
"""
