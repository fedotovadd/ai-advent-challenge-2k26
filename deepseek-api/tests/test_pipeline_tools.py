import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx
from test_pipeline_models import SEARCH, SUMMARY


GITHUB_ITEM = dict(SEARCH['items'][0], private=False)
GITHUB_ITEM['stargazers_count'] = GITHUB_ITEM.pop('stars')
GITHUB_ITEM['html_url'] = GITHUB_ITEM.pop('url')
GITHUB_RESPONSE = {'total_count': 10, 'incomplete_results': False, 'items': [GITHUB_ITEM]}


class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def search(self, handler, query='mcp', limit=3):
        from github_api import search_repositories
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await search_repositories(query, limit=limit, client=client)

    async def test_public_search_is_bounded_and_filters_response_fields(self):
        seen = []
        def handler(request):
            seen.append(request)
            return httpx.Response(200, json=dict(GITHUB_RESPONSE, items=[dict(
                GITHUB_ITEM, description='я' * 400, irrelevant='discard')]))
        result = await self.search(handler, '  mcp language:python  ', 2)
        request = seen[0]
        self.assertEqual(request.url.path, '/search/repositories')
        self.assertEqual(request.url.params['q'], 'mcp language:python is:public')
        self.assertEqual(request.url.params['per_page'], '2')
        self.assertEqual(request.extensions['timeout']['read'], 10.0)
        self.assertEqual(result['query'], 'mcp language:python')
        self.assertEqual(len(result['items'][0]['description']), 300)
        self.assertTrue(result['items'][0]['description'].endswith('…'))
        self.assertEqual(set(result['items'][0]), set(SEARCH['items'][0]))

    async def test_private_results_and_invalid_search_envelopes_are_rejected(self):
        from github_api import GitHubAPIError
        for payload in ({}, [], dict(GITHUB_RESPONSE, total_count='10'),
                        dict(GITHUB_RESPONSE, items=[dict(GITHUB_ITEM, private=True)]),
                        dict(GITHUB_RESPONSE, items=[dict(GITHUB_ITEM, stargazers_count=True)])):
            with self.subTest(payload=payload), self.assertRaises(GitHubAPIError):
                await self.search(lambda req: httpx.Response(200, json=payload))

    async def test_empty_results_and_incomplete_search_are_valid(self):
        result = await self.search(lambda req: httpx.Response(200, json={
            'total_count': 0, 'incomplete_results': True, 'items': []}))
        self.assertEqual(result['items'], [])
        self.assertTrue(result['incomplete_results'])

    async def test_invalid_input_never_reaches_github(self):
        for query, limit in [('', 3), ('x' * 201, 3), ('mcp', True), ('mcp', 4)]:
            with self.subTest(query=query, limit=limit), self.assertRaises(ValueError):
                await self.search(lambda req: self.fail('No request expected'), query, limit)

    async def test_errors_reuse_safe_github_messages_and_do_not_follow_redirects(self):
        from github_api import GitHubAPIError
        for code, message in [(401, 'токен'), (429, 'лимит'), (503, 'недоступ'), (301, 'перенаправ')]:
            seen = []
            def handler(req):
                seen.append(req)
                return httpx.Response(code, headers={'Location': 'https://example.com'}, text='SECRET')
            with self.subTest(code=code), self.assertRaisesRegex(GitHubAPIError, message):
                await self.search(handler)
            self.assertEqual(len(seen), 1)
        for error in (httpx.ReadTimeout, httpx.ConnectError):
            def fail(req):
                raise error('SECRET', request=req)
            with self.assertRaises(GitHubAPIError) as caught:
                await self.search(fail)
            self.assertNotIn('SECRET', str(caught.exception))


class SummaryAndSaveTests(unittest.TestCase):
    def test_summary_is_deterministic_includes_sources_and_actual_count(self):
        from pipeline_tools import summarize
        first = summarize(SEARCH)
        self.assertEqual(first, summarize(SEARCH))
        self.assertEqual(first['source_count'], 1)
        self.assertEqual(first['query'], SEARCH['query'])
        for text in ('owner/repo', 'Python', '123', 'Useful package', 'https://github.com/owner/repo'):
            self.assertIn(text, first['markdown'])
        incomplete = summarize(dict(SEARCH, incomplete_results=True))
        self.assertIn('непол', incomplete['markdown'].lower())
        empty = summarize(dict(SEARCH, total_count=0, items=[]))
        self.assertEqual(empty['source_count'], 0)
        self.assertIn('Ничего не найдено', empty['markdown'])

    def test_summary_treats_markup_as_data(self):
        from pipeline_tools import summarize
        data = dict(SEARCH, items=[dict(SEARCH['items'][0], description='<script>x</script>\n# injected')])
        text = summarize(data)['markdown']
        self.assertNotIn('<script>', text)
        self.assertNotIn('\n# injected', text)

    def test_save_writes_exact_utf8_bytes_and_unique_files(self):
        from pipeline_tools import save_to_file
        with TemporaryDirectory() as temp:
            first = save_to_file(SUMMARY, reports_dir=Path(temp))
            second = save_to_file(SUMMARY, reports_dir=Path(temp))
            expected = SUMMARY['markdown'].encode('utf-8')
            self.assertNotEqual(first['path'], second['path'])
            self.assertEqual(Path(first['path']).parent, Path(temp).resolve())
            self.assertEqual(Path(first['path']).read_bytes(), expected)
            self.assertEqual(first['bytes_written'], len(expected))
            self.assertEqual(first['sha256'], hashlib.sha256(expected).hexdigest())

    def test_save_rejects_paths_and_excess_bytes_without_creating_files(self):
        from pipeline_tools import save_to_file
        with TemporaryDirectory() as temp:
            for changes in ({'markdown': 'я' * 3001}, {'path': '../evil.md'}, {'markdown': ''}):
                with self.assertRaises(ValueError):
                    save_to_file(dict(SUMMARY, **changes), reports_dir=Path(temp))
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_collision_preserves_existing_file_and_write_failure_cleans_partial(self):
        from pipeline_tools import save_to_file
        with TemporaryDirectory() as temp, patch('pipeline_tools.uuid4', return_value='fixed'):
            existing = Path(temp) / 'fixed.md'
            existing.write_bytes(b'keep me')
            with self.assertRaises(OSError):
                save_to_file(SUMMARY, reports_dir=Path(temp))
            self.assertEqual(existing.read_bytes(), b'keep me')
            existing.unlink()
            original_open = Path.open
            class BrokenWriter:
                def __enter__(self):
                    self.file = original_open(existing, 'xb')
                    return self
                def write(self, content):
                    self.file.write(content[:1])
                    raise OSError('disk full')
                def __exit__(self, *args):
                    self.file.close()
            with patch('pipeline_tools.Path.open', return_value=BrokenWriter()):
                with self.assertRaises(OSError):
                    save_to_file(SUMMARY, reports_dir=Path(temp))
            self.assertEqual(list(Path(temp).iterdir()), [])
