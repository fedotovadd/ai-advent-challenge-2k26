"""Discovery-only client for the public DeepWiki MCP server."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable


DEEPWIKI_MCP_URL = "https://mcp.deepwiki.com/mcp"
DEFAULT_DESCRIPTION = "Описание не предоставлено"
HTTP_TIMEOUT_SECONDS = 10.0


@asynccontextmanager
async def _streamable_http_transport(
    url: str, *, timeout: float
) -> AsyncIterator[tuple[Any, Any, Any]]:
    try:
        import httpx
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as error:
        raise RuntimeError("Для работы MCP-клиента установите зависимость mcp.") from error

    async with httpx.AsyncClient(timeout=timeout) as http_client:
        async with streamable_http_client(url, http_client=http_client) as streams:
            yield streams


def _client_session_factory(read_stream: Any, write_stream: Any) -> Any:
    try:
        from mcp import ClientSession
    except ImportError as error:
        raise RuntimeError("Для работы MCP-клиента установите зависимость mcp.") from error
    return ClientSession(read_stream, write_stream)


async def _list_public_tools(
    url: str,
    *,
    transport_factory: Callable[..., Any] = _streamable_http_transport,
    session_factory: Callable[[Any, Any], Any] = _client_session_factory,
) -> list[dict[str, str]]:
    """Initialize an MCP session and return only its advertised tools."""
    async with transport_factory(url, timeout=HTTP_TIMEOUT_SECONDS) as streams:
        read_stream, write_stream, *_ = streams
        async with session_factory(read_stream, write_stream) as session:
            await session.initialize()
            tool_result = await session.list_tools()

    return [
        {
            "name": tool.name,
            "description": (
                tool.description
                if isinstance(getattr(tool, "description", None), str) and tool.description.strip()
                else DEFAULT_DESCRIPTION
            ),
        }
        for tool in tool_result.tools
    ]


def list_tools() -> list[dict[str, str]]:
    """Synchronously discover the tools offered by DeepWiki."""
    return asyncio.run(_list_public_tools(DEEPWIKI_MCP_URL))
