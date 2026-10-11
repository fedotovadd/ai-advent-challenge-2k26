"""Compatibility launcher for python github_mcp_server.py."""

from deepseek_app.mcp.github_mcp_server import mcp

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
