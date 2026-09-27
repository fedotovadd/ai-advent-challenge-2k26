import importlib.util
import unittest

from pydantic import ValidationError


SEARCH = {'query': 'mcp', 'fetched_at': '2026-09-27T10:00:00Z', 'total_count': 10,
          'incomplete_results': False, 'items': [
              {'full_name': 'owner/repo', 'description': 'Useful package', 'language': 'Python',
               'stars': 123, 'url': 'https://github.com/owner/repo'}]}
SUMMARY = {'query': 'mcp', 'source_count': 1, 'markdown': '# Отчёт\n'}


class PipelineModelsTests(unittest.TestCase):
    def test_models_are_available(self):
        self.assertIsNotNone(importlib.util.find_spec('pipeline_models'))

    def test_search_requires_exact_types_fields_and_bounds(self):
        from pipeline_models import SearchResult
        self.assertEqual(SearchResult.model_validate(SEARCH).model_dump(), SEARCH)
        for change in ({'total_count': '10'}, {'total_count': True}, {'total_count': -1},
                       {'items': SEARCH['items'] * 4}, {'incomplete_results': 1},
                       {'extra': 'no'}, {'query': ''}, {'query': 'x' * 201},
                       {'fetched_at': 'not-a-date'}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                SearchResult.model_validate(dict(SEARCH, **change))
        for change in ({'stars': '123'}, {'stars': True}, {'description': 'x' * 301},
                       {'extra': 'no'}, {'url': 'https://evil.example/repo'}, {'language': 12}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                SearchResult.model_validate(dict(SEARCH, items=[dict(SEARCH['items'][0], **change)]))
        self.assertIsNone(SearchResult.model_validate(dict(SEARCH, items=[dict(
            SEARCH['items'][0], description=None, language=None)])).items[0].description)

    def test_query_normalizes_only_command_input_and_limit_is_strict(self):
        from pipeline_models import SearchArguments, normalize_query
        self.assertEqual(normalize_query('  mcp language:python  '), 'mcp language:python')
        for query in ('', '  ', 'x' * 201, None, 123):
            with self.subTest(query=query), self.assertRaises(ValueError):
                normalize_query(query)
        for limit in (0, 4, True, 1.0, '3'):
            with self.subTest(limit=limit), self.assertRaises(ValidationError):
                SearchArguments(query='mcp', limit=limit)

    def test_markdown_limit_is_utf8_bytes_and_saved_file_is_bounded(self):
        from pipeline_models import SummaryResult, SavedFile
        self.assertEqual(SummaryResult(**dict(SUMMARY, markdown='я' * 3000)).source_count, 1)
        for change in ({'markdown': ''}, {'markdown': 'я' * 3001}, {'source_count': True},
                       {'source_count': 4}, {'path': '/tmp/unwanted'}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                SummaryResult.model_validate(dict(SUMMARY, **change))
        saved = {'path': '/tmp/reports/abc.md', 'bytes_written': 42, 'sha256': 'a' * 64}
        self.assertEqual(SavedFile(**saved).model_dump(), saved)
        for change in ({'bytes_written': True}, {'bytes_written': 6001}, {'sha256': 'invalid'}, {'extra': 1}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                SavedFile.model_validate(dict(saved, **change))
