import importlib.util
import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import httpx
from mcp.shared.memory import create_connected_server_and_client_session


PAYLOAD = {
    'full_name': 'octocat/Hello-World', 'description': None, 'language': None,
    'stargazers_count': 42, 'forks_count': 7, 'default_branch': 'main',
    'html_url': 'https://github.com/octocat/Hello-World', 'archived': False,
    'pushed_at': None, 'private': False, 'irrelevant': 'discard me',
}


class ImplementationTests(unittest.TestCase):
    def test_github_modules_exist(self):
        self.assertIsNotNone(importlib.util.find_spec('github_api'))
        self.assertIsNotNone(importlib.util.find_spec('github_mcp_server'))


class GitHubAPITests(unittest.IsolatedAsyncioTestCase):
    async def fetch(self, handler, owner='octocat', repo='Hello-World'):
        from github_api import get_repository
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
            return await get_repository(owner, repo, client=client)

    async def test_gets_public_metadata_with_bounded_request_and_utc_timestamp(self):
        seen = []
        def handler(request):
            seen.append(request)
            return httpx.Response(200, json=PAYLOAD)
        with patch.dict('os.environ', {'GITHUB_TOKEN': 'test-secret'}):
            data = await self.fetch(handler)
        self.assertEqual(str(seen[0].url), 'https://api.github.com/repos/octocat/Hello-World')
        self.assertEqual(seen[0].method, 'GET')
        self.assertEqual(seen[0].headers['Authorization'], 'Bearer test-secret')
        self.assertEqual(seen[0].extensions['timeout']['read'], 10.0)
        self.assertEqual(set(data), (set(PAYLOAD) - {'private', 'irrelevant'}) | {'fetched_at'})
        self.assertIsNone(data['description'])
        self.assertIsNone(data['language'])
        self.assertIsNone(data['pushed_at'])
        self.assertEqual(data['stargazers_count'], 42)
        timestamp = datetime.fromisoformat(data['fetched_at'].replace('Z', '+00:00'))
        self.assertEqual(timestamp.utcoffset(), timezone.utc.utcoffset(timestamp))
        self.assertLess(abs((datetime.now(timezone.utc) - timestamp).total_seconds()), 5)

    async def test_token_is_optional(self):
        def handler(request):
            self.assertNotIn('Authorization', request.headers)
            return httpx.Response(200, json=PAYLOAD)
        with patch.dict('os.environ', {}, clear=True):
            await self.fetch(handler)

    async def test_rejects_invalid_identifiers_without_sending_request(self):
        from github_api import GitHubAPIError
        def handler(request):
            self.fail('Invalid input must not reach GitHub')
        for value in ['', ' ', '.', '..', 'a/b', 'a\\b', 'a\nb', 'a\x00b', 'a\x7fb']:
            for field in ['owner', 'repo']:
                with self.subTest(value=value, field=field):
                    args = {'owner': 'octocat', 'repo': 'Hello-World', field: value}
                    with self.assertRaises(GitHubAPIError):
                        await self.fetch(handler, **args)

    async def test_status_errors_are_safe_and_distinguish_rate_limit_from_permissions(self):
        from github_api import GitHubAPIError
        cases = [
            (401, {}, 'токен'), (403, {}, 'доступ'),
            (403, {'x-ratelimit-remaining': '0'}, 'лимит'),
            (403, {'retry-after': '60'}, 'лимит'), (404, {}, 'не найден'),
            (429, {}, 'лимит'), (500, {}, 'недоступ'), (503, {}, 'недоступ'),
            (301, {'location': 'https://evil.example'}, 'перенаправ'),
        ]
        for status, headers, expected in cases:
            with self.subTest(status=status, headers=headers):
                calls = []
                def handler(request):
                    calls.append(request)
                    return httpx.Response(status, headers=headers, text='SECRET technical body')
                with self.assertRaises(GitHubAPIError) as caught:
                    await self.fetch(handler)
                self.assertIn(expected, str(caught.exception).lower())
                self.assertNotIn('SECRET', str(caught.exception))
                self.assertEqual(len(calls), 1)

    async def test_secondary_rate_limit_is_recognized_without_exposing_body(self):
        from github_api import GitHubAPIError
        with self.assertRaisesRegex(GitHubAPIError, 'лимит'):
            await self.fetch(lambda req: httpx.Response(403, json={
                'message': 'You have exceeded a secondary rate limit. SECRET'
            }))

    async def test_timeout_and_network_errors_do_not_expose_exception_text(self):
        from github_api import GitHubAPIError
        for error in [httpx.ReadTimeout, httpx.ConnectError]:
            def handler(request):
                raise error('SECRET token and URL', request=request)
            with self.subTest(error=error), self.assertRaises(GitHubAPIError) as caught:
                await self.fetch(handler)
            self.assertNotIn('SECRET', str(caught.exception))

    async def test_private_or_malformed_response_is_rejected(self):
        from github_api import GitHubAPIError
        invalid = [dict(PAYLOAD, private=True), [], {}, dict(PAYLOAD, stargazers_count='many')]
        for body in invalid:
            with self.subTest(body=body), self.assertRaises(GitHubAPIError):
                await self.fetch(lambda req: httpx.Response(200, json=body))
        with self.assertRaises(GitHubAPIError):
            await self.fetch(lambda req: httpx.Response(200, text='SECRET not JSON'))


class MCPProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_lists_tool_schema_and_calls_real_registered_tool(self):
        from github_mcp_server import create_server
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=PAYLOAD)
        )) as http_client:
            server = create_server(client=http_client)
            self.assertEqual(server.settings.host, '127.0.0.1')
            self.assertEqual(server.settings.port, 8001)
            self.assertEqual(server.settings.streamable_http_path, '/mcp')
            async with create_connected_server_and_client_session(server) as session:
                tools = (await session.list_tools()).tools
                self.assertEqual([tool.name for tool in tools], ['get_repository'])
                schema = tools[0].inputSchema
                self.assertEqual(set(schema['required']), {'owner', 'repo'})
                self.assertFalse(schema['additionalProperties'])
                for field in ['owner', 'repo']:
                    self.assertEqual(schema['properties'][field]['type'], 'string')
                    self.assertTrue(schema['properties'][field]['description'])
                result = await session.call_tool('get_repository', {'owner': 'octocat', 'repo': 'Hello-World'})
                self.assertFalse(result.isError)
                self.assertEqual(result.structuredContent['full_name'], PAYLOAD['full_name'])
                self.assertEqual(json.loads(result.content[0].text), result.structuredContent)

    async def test_tool_errors_are_mcp_errors_and_extra_arguments_are_rejected(self):
        from github_mcp_server import create_server
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(401, text='SECRET')
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            async with create_connected_server_and_client_session(create_server(client=http_client)) as session:
                result = await session.call_tool('get_repository', {'owner': 'octocat', 'repo': 'Hello-World'})
                self.assertTrue(result.isError)
                self.assertIn('токен', result.content[0].text.lower())
                self.assertNotIn('SECRET', result.content[0].text)
                for arguments in [
                    {'owner': '../evil', 'repo': 'x'},
                    {'owner': 'x', 'repo': 'y', 'extra': True},
                    {'owner': 'x'},
                ]:
                    invalid = await session.call_tool('get_repository', arguments)
                    self.assertTrue(invalid.isError)
                self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
