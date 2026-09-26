import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

from mcp_registry import McpRegistry
from mcp_client import validate_url

TOOL = {'name': 'get_repository', 'description': 'Repo', 'inputSchema': {
    'type': 'object', 'properties': {'owner': {'type': 'string'}},
    'required': ['owner'], 'additionalProperties': False}}


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'mcp.json'
        self.discovery = Mock(return_value=[TOOL])
        self.caller = Mock(return_value={'content': [], 'structuredContent': {'stars': 1}, 'isError': False})
        self.registry = McpRegistry(self.path, self.discovery, self.caller)

    def test_connect_snapshot_route_disconnect_and_restore(self):
        server = self.registry.connect('http://127.0.0.1:8001/mcp')
        snapshot = self.registry.snapshot()
        name = snapshot[0]['function']['name']
        self.assertEqual(snapshot[0]['function']['parameters'], TOOL['inputSchema'])
        result = self.registry.call(name, {'owner': 'test'}, snapshot)
        self.assertFalse(result['isError'])
        self.caller.assert_called_once_with(server['url'], 'get_repository', {'owner': 'test'})
        restored = McpRegistry(self.path, self.discovery, self.caller)
        self.assertEqual(restored.servers()[0]['status'], 'disconnected')
        self.assertEqual(restored.snapshot(), [])
        self.registry.disconnect(server['id'])
        self.assertEqual(self.registry.snapshot(), [])
        with self.assertRaises(ValueError):
            self.registry.call(name, {'owner': 'test'}, snapshot)

    def test_names_unique_and_invalid_arguments_do_not_reach_server(self):
        self.registry.connect('https://one.example/mcp')
        self.registry.connect('https://two.example/mcp')
        snapshot = self.registry.snapshot()
        self.assertEqual(len({t['function']['name'] for t in snapshot}), 2)
        with self.assertRaises(ValueError):
            self.registry.call(snapshot[0]['function']['name'], {}, snapshot)
        self.caller.assert_not_called()
        self.registry.connect('https://one.example/mcp')
        self.assertEqual(len(self.registry.servers()), 2)

    def test_disconnect_wins_over_pending_connect(self):
        server = self.registry.connect('https://one.example/mcp')
        entered, release = threading.Event(), threading.Event()
        def slow(url):
            entered.set()
            release.wait(2)
            return [TOOL]
        self.discovery.side_effect = slow
        thread = threading.Thread(target=lambda: self.registry.connect(server['url']))
        thread.start()
        self.assertTrue(entered.wait(2))
        self.registry.disconnect(server['id'])
        release.set()
        thread.join(2)
        self.assertEqual(self.registry.snapshot(), [])

    def test_failed_connect_is_not_active_and_bad_file_is_ignored(self):
        self.discovery.side_effect = RuntimeError('unavailable')
        with self.assertRaises(RuntimeError):
            self.registry.connect('https://one.example/mcp')
        self.assertEqual(self.registry.snapshot(), [])
        self.path.write_text('oops')
        self.assertEqual(McpRegistry(self.path).servers(), [])

    def test_url_policy(self):
        for bad in ['http://example.com/mcp', 'http://127.0.0.1:9000/mcp',
                    'https://localhost/mcp', 'https://127.0.0.1/mcp',
                    'https://user:pass@example.com/mcp', 'https://example.com/mcp?q=1',
                    'https://example.com/mcp#x', 'https://example.com/\n']:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_url(bad)
        self.assertEqual(validate_url('http://127.0.0.1:8001/mcp'), 'http://127.0.0.1:8001/mcp')
