import unittest
from unittest.mock import AsyncMock, patch

import mcp_client


class FakeTransport:
    def __init__(self, events):
        self.events = events

    async def __aenter__(self):
        self.events.append("transport-enter")
        return "read", "write", "session-id"

    async def __aexit__(self, exc_type, exc, traceback):
        self.events.append("transport-exit")


class FakeSession:
    def __init__(self, read, write, events):
        self.events = events
        self.events.append(("session-created", read, write))

    async def __aenter__(self):
        self.events.append("session-enter")
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.events.append("session-exit")

    async def initialize(self):
        self.events.append("initialize")

    async def list_tools(self):
        self.events.append("list-tools")
        return type(
            "ToolList",
            (),
            {
                "tools": [
                    type("Tool", (), {"name": "weather", "description": "Погода"})(),
                    type("Tool", (), {"name": "clock", "description": ""})(),
                    type("Tool", (), {"name": "news"})(),
                ]
            },
        )()


class PublicToolDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_initializes_session_before_listing_tools_and_normalizes_descriptions(self):
        events = []

        def transport_factory(url, **kwargs):
            events.append(("transport-created", url, kwargs))
            return FakeTransport(events)

        def session_factory(read, write):
            return FakeSession(read, write, events)

        result = await mcp_client._list_public_tools(
            mcp_client.DEEPWIKI_MCP_URL,
            transport_factory=transport_factory,
            session_factory=session_factory,
        )

        self.assertEqual(result, [
            {"name": "weather", "description": "Погода"},
            {"name": "clock", "description": "Описание не предоставлено"},
            {"name": "news", "description": "Описание не предоставлено"},
        ])
        self.assertEqual(events, [
            ("transport-created", mcp_client.DEEPWIKI_MCP_URL, {"timeout": 10.0}),
            "transport-enter",
            ("session-created", "read", "write"),
            "session-enter",
            "initialize",
            "list-tools",
            "session-exit",
            "transport-exit",
        ])


class WebToolDiscoveryEntryPointTests(unittest.TestCase):
    def test_list_tools_always_uses_fixed_deepwiki_endpoint(self):
        discovery = AsyncMock(return_value=[])

        with patch("mcp_client._list_public_tools", discovery):
            self.assertEqual(mcp_client.list_tools(), [])

        discovery.assert_awaited_once_with(mcp_client.DEEPWIKI_MCP_URL)


if __name__ == "__main__":
    unittest.main()

class McpInvocationTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_call_initializes_and_preserves_structured_result(self):
        events = []
        class Session(FakeSession):
            async def call_tool(self, name, arguments):
                events.append((name, arguments))
                return type('Result', (), {'content': [], 'structuredContent': {'stars': 12}, 'isError': False})()
        result = await mcp_client._call_tool(
            'http://127.0.0.1:8001/mcp', 'get_repository', {'owner': 'x'},
            transport_factory=lambda *a, **k: FakeTransport(events),
            session_factory=lambda r,w: Session(r,w,events))
        self.assertEqual(result['structuredContent'], {'stars': 12})
        self.assertLess(events.index('initialize'), events.index(('get_repository', {'owner': 'x'})))
        self.assertEqual(events[-1], 'transport-exit')

    async def test_discovery_with_schema(self):
        class Session(FakeSession):
            async def list_tools(self):
                from types import SimpleNamespace
                return SimpleNamespace(tools=[SimpleNamespace(name='repo', description='x', inputSchema={'type':'object'})])
        result = await mcp_client._list_public_tools('https://example.com/mcp', include_schema=True,
            transport_factory=lambda *a, **k: FakeTransport([]),
            session_factory=lambda r,w: Session(r,w,[]))
        self.assertEqual(result[0]['inputSchema'], {'type': 'object'})
