"""Compatibility launcher for python pipeline_mcp_server.py."""

from deepseek_app.pipeline.pipeline_mcp_server import mcp

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
