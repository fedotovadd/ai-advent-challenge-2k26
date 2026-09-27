import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import httpx
from mcp.shared.memory import create_connected_server_and_client_session
from test_pipeline_models import SEARCH, SUMMARY
from test_pipeline_tools import GITHUB_RESPONSE


class PipelineMCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_protocol_schemas_calls_and_exact_forwarding(self):
        from pipeline_mcp_server import create_server
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=GITHUB_RESPONSE)
        )) as client:
            with TemporaryDirectory() as temp:
                server = create_server(client=client, reports_dir=Path(temp))
                self.assertEqual((server.settings.host, server.settings.port, server.settings.streamable_http_path),
                                 ('127.0.0.1', 8003, '/mcp'))
                async with create_connected_server_and_client_session(server) as session:
                    tools = (await session.list_tools()).tools
                    self.assertEqual([tool.name for tool in tools], ['search', 'summarize', 'saveToFile'])
                    for tool in tools:
                        self.assertFalse(tool.inputSchema['additionalProperties'])
                        self.assertFalse(tool.outputSchema['additionalProperties'])
                    self.assertTrue(tools[0].annotations.readOnlyHint)
                    self.assertTrue(tools[1].annotations.readOnlyHint)
                    self.assertFalse(tools[2].annotations.readOnlyHint)
                    self.assertFalse(tools[2].annotations.idempotentHint)
                    search = await session.call_tool('search', {'query': 'mcp'})
                    summary = await session.call_tool('summarize', {'search_result': search.structuredContent})
                    saved = await session.call_tool('saveToFile', {'report': summary.structuredContent})
                    for result in [search, summary, saved]:
                        self.assertFalse(result.isError)
                        self.assertEqual(json.loads(result.content[0].text), result.structuredContent)
                        self.assertLessEqual(len(json.dumps(result.model_dump(mode='json'), ensure_ascii=False).encode()), 16384)
                    self.assertEqual(Path(saved.structuredContent['path']).read_text(), summary.structuredContent['markdown'])

    async def test_invalid_inputs_extra_fields_and_safe_tool_errors(self):
        from pipeline_mcp_server import create_server
        requests = []
        def handler(req):
            requests.append(req)
            return httpx.Response(401, text='SECRET')
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with TemporaryDirectory() as temp:
                async with create_connected_server_and_client_session(create_server(client=client, reports_dir=Path(temp))) as session:
                    for name, args in [
                        ('search', {'query': 'mcp', 'limit': True}),
                        ('search', {'query': 'mcp', 'path': '../x'}),
                        ('search', {'query': 'mcp', 'limit': '3'}),
                        ('summarize', {'search_result': dict(SEARCH, extra=1)}),
                        ('summarize', {'search_result': dict(SEARCH, total_count='10')}),
                        ('summarize', {'search_result': json.dumps(SEARCH)}),
                        ('saveToFile', {'report': json.dumps(SUMMARY)}),
                        ('saveToFile', {'report': SUMMARY, 'path': '../x'}),
                        ('saveToFile', {'report': dict(SUMMARY, markdown='я' * 3001)}),
                    ]:
                        result = await session.call_tool(name, args)
                        self.assertTrue(result.isError, (name, args))
                    self.assertEqual(requests, [])
                    result = await session.call_tool('search', {'query': 'mcp'})
                    self.assertTrue(result.isError)
                    self.assertIn('токен', result.content[0].text)
                    self.assertNotIn('SECRET', result.content[0].text)
                    self.assertEqual(list(Path(temp).iterdir()), [])

    async def test_complete_envelope_limit_accounts_for_json_escaping(self):
        from pipeline_mcp_server import bounded_result
        # Fits the Markdown UTF-8 limit, but duplication and JSON escaping do not fit MCP.
        from pipeline_models import SummaryResult
        report = SummaryResult(**dict(SUMMARY, markdown='\x00' * 6000))
        with self.assertRaisesRegex(ValueError, '16 KiB'):
            bounded_result(report)
        report = SummaryResult(**dict(SUMMARY, markdown='я' * 2800))
        result = bounded_result(report)
        self.assertLessEqual(len(json.dumps(result.model_dump(mode='json'), ensure_ascii=False).encode()), 16384)
