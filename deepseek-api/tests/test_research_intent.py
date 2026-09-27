import importlib.util
import json
import unittest
from unittest.mock import Mock

from errors import MissingApiKeyError


class ResearchIntentTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_intent'), 'natural-language research is missing')
        import research_intent
        return research_intent

    def test_legacy_and_explicit_raw_queries_skip_model(self):
        module = self.module()
        for text, expected in [('mcp language:python', 'mcp language:python'), ('repo:owner/name', 'repo:owner/name'), ('mcp', 'mcp'), ('--query telegram bot language:python', 'telegram bot language:python')]:
            self.assertEqual(module.raw_search_arguments(text), {'query': expected, 'limit': 3})
        for text in ['Найди две библиотеки для Telegram-ботов на Python', 'Find three Python libraries', 'Найди mcp language:python', 'Summarize repo:owner/name', 'Find language:python']:
            self.assertIsNone(module.raw_search_arguments(text))

    def test_one_bounded_call_translates_only_current_request(self):
        module = self.module()
        ask = Mock(return_value={'content': json.dumps({'action': 'search', 'query': 'telegram bot language:python', 'limit': 2, 'message': None}), 'usage': {'prompt_tokens': 120, 'completion_tokens': 30, 'total_tokens': 150}})
        result = module.interpret_research('Найди две библиотеки для Telegram-ботов на Python', ask, 'deepseek-v4-flash')
        self.assertEqual(result['arguments'], {'query': 'telegram bot language:python', 'limit': 2})
        self.assertIsNone(result['error'])
        self.assertEqual(result['record']['usage']['total_tokens'], 150)
        self.assertEqual(result['record']['status'], 'success')
        ask.assert_called_once()
        payload = ask.call_args.args[0]
        self.assertEqual(len(payload['messages']), 2)
        self.assertEqual(payload['messages'][-1]['content'], 'Найди две библиотеки для Telegram-ботов на Python')
        self.assertEqual(ask.call_args.kwargs['response_format'], {'type': 'json_object'})
        self.assertEqual(ask.call_args.kwargs['max_retries'], 0)
        self.assertEqual(ask.call_args.kwargs['timeout'], 30)
        self.assertNotIn('tools', ask.call_args.kwargs)

    def test_ambiguous_or_unsupported_request_stops_before_search(self):
        module = self.module()
        for action in ['clarify', 'unsupported']:
            result = module.interpret_research('Напиши кратко что в репозитории', Mock(return_value=json.dumps({'action': action, 'query': None, 'limit': None, 'message': 'Укажите репозиторий или тему поиска.'})), 'glm-4.7-flash')
            self.assertIsNone(result['arguments'])
            self.assertEqual(result['error'], 'Укажите репозиторий или тему поиска.')
            self.assertEqual(result['record']['status'], 'success')

    def test_invalid_outputs_cannot_select_tools_paths_or_exceed_limits(self):
        module = self.module()
        good = {'action': 'search', 'query': 'python', 'limit': 3, 'message': None}
        for raw in ['not JSON', '[]', json.dumps({**good, 'limit': True}), json.dumps({**good, 'limit': 4}), json.dumps({**good, 'query': ' '}), json.dumps({**good, 'query': 'x\ny'}), json.dumps({**good, 'filename': '/tmp/a'}), json.dumps({**good, 'query': 'x' * 201})]:
            result = module.interpret_research('Найди Python', Mock(return_value=raw), 'deepseek-v4-flash')
            self.assertIsNone(result['arguments'])
            self.assertIsNotNone(result['error'])
            self.assertEqual(result['record']['status'], 'error')

    def test_provider_errors_are_safe_and_missing_key_is_actionable(self):
        module = self.module()
        result = module.interpret_research('Найди Python', Mock(side_effect=MissingApiKeyError('DEEPSEEK_API_KEY')), 'deepseek-v4-flash')
        self.assertIn('DEEPSEEK_API_KEY', result['error'])
        self.assertIsNone(result['arguments'])
        ask = Mock(side_effect=RuntimeError('SECRET key=value'))
        result = module.interpret_research('Найди Python', ask, 'deepseek-v4-flash')
        self.assertNotIn('SECRET', result['error'])
        ask.assert_called_once()
