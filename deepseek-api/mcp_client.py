"""Discovery-only client for public MCP Streamable HTTP servers."""

import asyncio
import ipaddress
import socket
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Iterable
from urllib.parse import urlsplit, urlunsplit


DEFAULT_DESCRIPTION = "Описание не предоставлено"
HTTP_TIMEOUT_SECONDS = 10.0

Resolver = Callable[[str], Iterable[str | ipaddress.IPv4Address | ipaddress.IPv6Address]]


def _resolve_host(host: str) -> Iterable[str]:
    try:
        return {result[4][0] for result in socket.getaddrinfo(host, None)}
    except socket.gaierror as error:
        raise ValueError("Не удалось разрешить имя хоста MCP-сервера.") from error


def normalize_server_url(value: str, *, resolver: Resolver = _resolve_host) -> str:
    """Validate a public HTTPS endpoint and return its canonical URL."""
    if not isinstance(value, str) or not value:
        raise ValueError("Укажите абсолютный HTTPS URL MCP-сервера.")
    if value != value.strip() or any(character.isspace() for character in value):
        raise ValueError("URL MCP-сервера не должен содержать пробелы.")

    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise ValueError("Некорректный URL MCP-сервера.") from error

    if parsed.scheme.lower() != "https":
        raise ValueError("Допускается только HTTPS URL MCP-сервера.")
    if not parsed.netloc or not host:
        raise ValueError("Укажите абсолютный HTTPS URL MCP-сервера.")
    if "@" in parsed.netloc:
        raise ValueError("URL MCP-сервера не должен содержать учетные данные.")
    if parsed.query:
        raise ValueError("URL MCP-сервера не должен содержать параметры запроса.")
    if parsed.fragment:
        raise ValueError("URL MCP-сервера не должен содержать фрагмент.")

    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("В URL MCP-сервера нельзя использовать IP-адрес.")

    try:
        resolved_addresses = tuple(resolver(host))
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("Не удалось разрешить имя хоста MCP-сервера.") from error

    if not resolved_addresses:
        raise ValueError("Не удалось разрешить имя хоста MCP-сервера.")

    for resolved_address in resolved_addresses:
        try:
            address = ipaddress.ip_address(resolved_address)
        except ValueError as error:
            raise ValueError("Не удалось проверить адрес MCP-сервера.") from error
        if (
            not address.is_global
            or address.is_loopback
            or address.is_private
            or address.is_link_local
            or address.is_multicast
            or address.is_unspecified
            or address.is_reserved
        ):
            raise ValueError("Хост MCP-сервера должен разрешаться только в общедоступные адреса.")

    canonical_host = host.encode("idna").decode("ascii").lower()
    netloc = canonical_host if ":" not in canonical_host else f"[{canonical_host}]"
    if port is not None and port != 443:
        netloc = f"{netloc}:{port}"

    return urlunsplit(("https", netloc, parsed.path or "/", "", ""))


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


async def list_public_tools(
    url: str,
    *,
    transport_factory: Callable[..., Any] = _streamable_http_transport,
    session_factory: Callable[[Any, Any], Any] = _client_session_factory,
    resolver: Resolver = _resolve_host,
) -> list[dict[str, str]]:
    """Initialize a public MCP session and return only its advertised tools."""
    server_url = normalize_server_url(url, resolver=resolver)

    async with transport_factory(server_url, timeout=HTTP_TIMEOUT_SECONDS) as streams:
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


def list_tools(url: str) -> list[dict[str, str]]:
    """Synchronously discover tools for use in ordinary request threads."""
    return asyncio.run(list_public_tools(url))
