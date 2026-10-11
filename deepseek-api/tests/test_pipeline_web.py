import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

from deepseek_app.server.web import ChatServer


class PipelineWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ask = Mock(return_value='Обычный ответ')
        self.server = ChatServer(('127.0.0.1', 0), self.ask, Path(self.temp.name) / 'agents.json')
        self.calls = []
        self.server.pipeline_runner = self.run_pipeline
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def run_pipeline(self, query, *, emit, limit=3):
        self.calls.append(query)
        self.last_limit = limit
        for step, data in [('search', {'items': []}), ('summarize', {'source_count': 0, 'markdown': 'Ничего не найдено'}), ('saveToFile', {'path': '/tmp/test.md', 'bytes_written': 10, 'sha256': 'a'*64})]:
            emit({'type': 'step_started', 'step': step, 'arguments': {'query': query}, 'server_url': 'http://127.0.0.1:8003/mcp'})
            emit({'type': 'step_completed', 'step': step, 'result': {'content': [], 'structuredContent': data, 'isError': False}, 'duration_ms': 1})
        return {'status': 'success', 'saved_file': data, 'failed_step': None, 'error': None}

    def request(self, text):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        connection.request('POST', '/api/agents/agent-1/messages', json.dumps({'text': text}), {'Content-Type': 'application/json'})
        return connection, connection.getresponse()

    def test_research_streams_three_steps_and_saves_card_without_llm(self):
        conn, response = self.request('/research mcp language:python')
        self.assertIn('application/x-ndjson', response.getheader('Content-Type'))
        events = [json.loads(line) for line in response]
        conn.close()
        self.assertEqual([e['type'] for e in events], ['pipeline_started', 'step_started', 'step_completed', 'step_started', 'step_completed', 'step_started', 'step_completed', 'pipeline_completed'])
        self.assertEqual([e['seq'] for e in events], list(range(1,9)))
        self.assertEqual(events[-1]['agent']['messages'][-1]['pipeline']['status'], 'success')
        self.assertEqual(self.calls, ['mcp language:python'])
        self.ask.assert_not_called()

    def test_invalid_command_is_400_and_normal_chat_unchanged(self):
        conn, response = self.request('/research   ')
        self.assertEqual(response.status, 400)
        response.read(); conn.close()
        self.assertEqual(self.calls, [])
        self.ask.assert_not_called()
        conn, response = self.request('/researching test')
        self.assertEqual(response.status, 200)
        self.assertIn('application/json', response.getheader('Content-Type'))
        response.read(); conn.close()
        self.ask.assert_called_once()

    def test_natural_request_resolves_streams_and_persists_usage(self):
        self.ask.return_value = {'content': json.dumps({'action': 'search', 'query': 'telegram language:python', 'limit': 2, 'message': None}), 'usage': {'prompt_tokens': 120, 'completion_tokens': 30, 'total_tokens': 150}}
        conn, response = self.request('/research Найди две библиотеки для Telegram на Python')
        events = [json.loads(line) for line in response]; conn.close()
        self.assertEqual([e['type'] for e in events[:3]], ['pipeline_started', 'interpretation_started', 'interpretation_completed'])
        self.assertEqual(events[-1]['type'], 'pipeline_completed')
        self.assertEqual(self.calls, ['telegram language:python'])
        self.assertEqual(self.last_limit, 2)
        final = events[-1]['agent']
        self.assertEqual(final['messages'][-1]['pipeline']['interpretation']['status'], 'success')
        self.assertEqual(final['metadata']['usage']['totalTokens'], 150)
        self.assertEqual(final['metrics']['totals']['totalTokens'], 150)
        self.assertEqual(final['metrics']['lastMessage']['totalTokens'], 150)
        self.assertEqual(len(final['metrics']['calls']), 1)
        self.assertEqual(len(final['metadata']['modelRequests']), 1)
        from deepseek_app.agent.agent import AgentRegistry
        restored = AgentRegistry(self.ask, Path(self.temp.name) / 'agents.json').get('agent-1').snapshot()
        self.assertEqual(restored['messages'], final['messages'])
        self.assertEqual(restored['metadata'], final['metadata'])
        self.assertEqual(restored['metrics'], final['metrics'])
        self.ask.assert_called_once()

    def test_interpretation_failure_skips_all_tools_and_explains(self):
        self.ask.return_value = json.dumps({'action': 'clarify', 'query': None, 'limit': None, 'message': 'Укажите репозиторий.'})
        conn, response = self.request('/research Напиши кратко что в репозитории')
        events = [json.loads(line) for line in response]; conn.close()
        self.assertEqual(events[-1]['type'], 'pipeline_failed')
        card = events[-1]['agent']['messages'][-1]['pipeline']
        self.assertEqual(card['error'], 'Укажите репозиторий.')
        self.assertEqual(card['interpretation']['status'], 'error')
        self.assertEqual([s['status'] for s in card['steps']], ['skipped'] * 3)
        self.assertEqual(self.calls, [])

    def test_explicit_complex_query_skips_model_and_rejects_oversize(self):
        conn, response = self.request('/research --query telegram bot language:python')
        list(response); conn.close()
        self.assertEqual(self.calls, ['telegram bot language:python'])
        self.ask.assert_not_called()
        conn, response = self.request('/research --query ' + 'a' * 201)
        self.assertEqual(response.status, 400)
        response.read(); conn.close()

    def test_progress_arrives_before_runner_finishes_and_conflicts_are_rejected(self):
        reached = threading.Event(); release = threading.Event()
        self.addCleanup(release.set)
        def blocked(query, *, emit):
            emit({'type': 'step_started', 'step': 'search', 'arguments': {}, 'server_url': 'local'})
            emit({'type': 'step_completed', 'step': 'search', 'result': {'isError': False, 'structuredContent': {'items': []}, 'content': []}, 'duration_ms': 1})
            reached.set()
            release.wait(5)
            return {'status': 'error', 'error': 'Тестовая остановка', 'saved_file': None, 'failed_step': 'summarize'}
        self.server.pipeline_runner = blocked
        conn, response = self.request('/research demo')
        self.assertIn('ndjson', response.getheader('Content-Type'))
        first = [json.loads(response.readline()) for _ in range(3)]
        self.assertEqual(first[-1]['type'], 'step_completed')
        self.assertTrue(reached.wait(1))
        conn2, response2 = self.request('/research duplicate')
        self.assertEqual(response2.status, 409)
        response2.read(); conn2.close()
        release.set()
        remaining = [json.loads(line) for line in response]
        conn.close()
        self.assertEqual(remaining[-1]['type'], 'pipeline_failed')
        self.assertEqual(remaining[-1]['agent']['messages'][-1]['pipeline']['steps'][1]['status'], 'skipped')
        self.ask.assert_not_called()

    def test_disconnected_browser_does_not_repeat_or_cancel_tools(self):
        # Exercise the socket-failure branch deterministically; a TCP close can
        # otherwise arrive after a successful local write depending on buffering.
        from deepseek_app.pipeline.pipeline_http import handle_research
        from types import SimpleNamespace
        handler = SimpleNamespace(server=self.server, close_connection=False,
            send_response=Mock(), send_header=Mock(), end_headers=Mock(),
            wfile=SimpleNamespace(write=Mock(side_effect=BrokenPipeError), flush=Mock()))
        self.server.operation_locks['agent-1'] = {'active': 1, 'research': False}
        handle_research(handler, 'agent-1', '/research disconnected', 'disconnected')
        self.assertEqual(self.calls, ['disconnected'])
        card = self.server.registry.get('agent-1').snapshot()['messages'][-1]['pipeline']
        self.assertEqual(card['status'], 'success')
        self.assertEqual([s['status'] for s in card['steps']], ['success'] * 3)
        self.assertEqual(handler.wfile.write.call_count, 1)
        self.ask.assert_not_called()
