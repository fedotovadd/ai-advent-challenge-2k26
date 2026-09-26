"""Short-lived MCP sessions for discovery and tool invocation."""

import asyncio
import ipaddress
import json
from urllib.parse import urlsplit, urlunsplit
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable


DEEPWIKI_MCP_URL = "https://mcp.deepwiki.com/mcp"
DEFAULT_DESCRIPTION = "Описание не предоставлено"
HTTP_TIMEOUT_SECONDS = 10.0
OPERATION_TIMEOUT_SECONDS = 15.0
LOCAL_GITHUB_URL = "http://127.0.0.1:8001/mcp"
MAX_RESULT_BYTES = 16_384


@asynccontextmanager
async def _streamable_http_transport(
    url: str, *, timeout: float
) -> AsyncIterator[tuple[Any, Any, Any]]:
    try:
        import httpx
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as error:
        raise RuntimeError("Для работы MCP-клиента установите зависимость mcp.") from error

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as http_client:
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
    include_schema: bool = False,
) -> list[dict[str, str]]:
    """Initialize an MCP session and return only its advertised tools."""
    async with transport_factory(url, timeout=HTTP_TIMEOUT_SECONDS) as streams:
        read_stream, write_stream, *_ = streams
        async with session_factory(read_stream, write_stream) as session:
            await session.initialize()
            tool_result = await session.list_tools()

    return [
        {
            **({"inputSchema": tool.inputSchema} if include_schema else {}),
            "name": tool.name,
            "description": (
                tool.description
                if isinstance(getattr(tool, "description", None), str) and tool.description.strip()
                else DEFAULT_DESCRIPTION
            ),
        }
        for tool in tool_result.tools
    ]


def validate_url(url):
    if not isinstance(url, str) or len(url) > 2048 or any(ord(c) <= 32 for c in url):
        raise ValueError("Укажите корректный URL MCP-сервера.")
    if url == LOCAL_GITHUB_URL:
        return url
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
        if (parts.scheme != "https" or not host or parts.username is not None
                or parts.password is not None or "?" in url or "#" in url or "\\" in url
                or host.rstrip(".").lower() == "localhost" or host.endswith(".localhost")):
            raise ValueError()
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise ValueError()
        if "." not in host or "%" in host:
            raise ValueError()
        return urlunsplit(("https", host.lower() + (f":{port}" if port and port != 443 else ""), parts.path or "/", "", ""))
    except ValueError:
        raise ValueError("Поддерживаются публичный HTTPS URL и http://127.0.0.1:8001/mcp.") from None


async def _bounded(operation):
    return await asyncio.wait_for(operation, timeout=OPERATION_TIMEOUT_SECONDS)


def list_tools(url=None):
    """No-argument legacy discovery retains its name/description contract."""
    if url is None:
        return asyncio.run(_bounded(_list_public_tools(DEEPWIKI_MCP_URL)))
    return asyncio.run(_bounded(_list_public_tools(validate_url(url), include_schema=True)))


async def _call_tool(url, name, arguments, *, transport_factory=_streamable_http_transport,
                     session_factory=_client_session_factory):
    async with transport_factory(url, timeout=HTTP_TIMEOUT_SECONDS) as streams:
        async with session_factory(*streams[:2]) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments=arguments)
    data = {
        "content": [item.model_dump(mode="json") for item in result.content],
        "structuredContent": getattr(result, "structuredContent", None),
        "isError": bool(result.isError),
    }
    if len(json.dumps(data, ensure_ascii=False).encode("utf-8")) > MAX_RESULT_BYTES:
        return {"content": [{"type": "text", "text": "Результат MCP превышает 16 KiB."}],
                "structuredContent": None, "isError": True}
    return data


def call_tool(url, name, arguments):
    return asyncio.run(_bounded(_call_tool(validate_url(url), name, arguments)))
