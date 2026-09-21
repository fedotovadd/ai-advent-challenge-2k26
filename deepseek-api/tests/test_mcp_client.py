import asyncio
import unittest

import mcp_client


class ServerUrlNormalizationTests(unittest.TestCase):
    def test_accepts_a_public_https_url_and_canonicalizes_it(self):
        url = mcp_client.normalize_server_url(
            "HTTPS://Example.COM:443/tools",
            resolver=lambda host: ["8.8.8.8"],
        )

        self.assertEqual(url, "https://example.com/tools")

    def test_rejects_scheme_credentials_query_and_fragment(self):
        cases = {
            "http://example.test/": "HTTPS",
            "https://user:password@example.test/": "учетные данные",
            "https://example.test/?page=1": "параметры запроса",
            "https://example.test/#anchor": "фрагмент",
        }

        for value, message in cases.items():
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, message):
                    mcp_client.normalize_server_url(value, resolver=lambda host: ["8.8.8.8"])

    def test_rejects_ip_literals_even_when_they_are_global(self):
        for value in ("https://8.8.8.8/", "https://[2606:4700:4700::1111]/"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "IP-адрес"):
                    mcp_client.normalize_server_url(value, resolver=lambda host: ["8.8.8.8"])

    def test_rejects_all_non_global_resolved_ipv4_and_ipv6_addresses(self):
        forbidden = {
            "IPv4 loopback": "127.0.0.1",
            "IPv4 private": "10.0.0.1",
            "IPv4 link-local": "169.254.1.1",
            "IPv4 multicast": "224.0.0.1",
            "IPv4 unspecified": "0.0.0.0",
            "IPv4 reserved": "240.0.0.1",
            "IPv6 loopback": "::1",
            "IPv6 private": "fc00::1",
            "IPv6 link-local": "fe80::1",
            "IPv6 multicast": "ff00::1",
            "IPv6 unspecified": "::",
            "IPv6 reserved": "2001:db8::1",
        }

        for label, address in forbidden.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "общедоступ"):
                    mcp_client.normalize_server_url(
                        "https://public.example/",
                        resolver=lambda host, address=address: [address],
                    )


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

        result = await mcp_client.list_public_tools(
            "https://example.test/",
            transport_factory=transport_factory,
            session_factory=session_factory,
            resolver=lambda host: ["8.8.8.8"],
        )

        self.assertEqual(result, [
            {"name": "weather", "description": "Погода"},
            {"name": "clock", "description": "Описание не предоставлено"},
            {"name": "news", "description": "Описание не предоставлено"},
        ])
        self.assertEqual(events, [
            ("transport-created", "https://example.test/", {"timeout": 10.0}),
            "transport-enter",
            ("session-created", "read", "write"),
            "session-enter",
            "initialize",
            "list-tools",
            "session-exit",
            "transport-exit",
        ])


if __name__ == "__main__":
    unittest.main()
