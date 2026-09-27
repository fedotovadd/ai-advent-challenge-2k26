"""Real HTTP Streamable MCP sessions with a mocked external GitHub service."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import http.client
import json
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import Mock, patch

import httpx
import uvicorn

import mcp_client
import pipeline_tools
from pipeline_mcp_server import create_server
from pipeline_runner import run_pipeline
from test_pipeline_tools import GITHUB_RESPONSE


@contextmanager
def running_mcp(payload=GITHUB_RESPONSE, *, status=200):
    github_requests = []
    def handler(request):
        github_requests.append(request)
        return httpx.Response(status, json=payload)
    with TemporaryDirectory() as directory:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        app = create_server(client=client, reports_dir=Path(directory)).streamable_http_app()
        listener = socket.socket()
        try:
            listener.bind(('127.0.0.1', 0))
        except OSError:
            listener.close()
            asyncio.run(client.aclose())
            raise
        port = listener.getsockname()[1]
        started = threading.Event()
        class Server(uvicorn.Server):
            async def startup(self, sockets=None):
                await super().startup(sockets=sockets)
                started.set()
        server = Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='error', access_log=False))
        thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
        thread.start()
        try:
            if not started.wait(5):
                raise RuntimeError('Test MCP HTTP server failed to start')
            # Test-only transport injection: the production URL allowlist stays exact.
            def invoke(_url, name, arguments):
                return asyncio.run(mcp_client._bounded(mcp_client._call_tool(
                    f'http://127.0.0.1:{port}/mcp', name, arguments)))
            yield invoke, Path(directory), github_requests
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            listener.close()
            asyncio.run(client.aclose())
            if thread.is_alive():
                raise RuntimeError('Test MCP HTTP server failed to stop')


class PipelineHTTPIntegrationTests(unittest.TestCase):
    def test_natural_language_to_real_mcp_chain_and_saved_file(self):
        from web import ChatServer
        ask = Mock(return_value={'content': json.dumps({'action': 'search', 'query': 'mcp language:python', 'limit': 2, 'message': None}), 'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}})
        with running_mcp() as (invoke, directory, requests), TemporaryDirectory() as state_dir:
            chat = ChatServer(('127.0.0.1', 0), ask, Path(state_dir) / 'agents.json')
            chat.pipeline_runner = lambda query, emit, limit=3: run_pipeline(query, limit=limit, invoke=invoke, emit=emit)
            thread = threading.Thread(target=chat.serve_forever, daemon=True); thread.start()
            connection = http.client.HTTPConnection(*chat.server_address, timeout=10)
            try:
                connection.request('POST', '/api/agents/agent-1/messages', json.dumps({'text': '/research Найди два Python-репозитория про MCP и сохрани обзор'}), {'Content-Type': 'application/json'})
                events = [json.loads(line) for line in connection.getresponse()]
                self.assertEqual(events[-1]['type'], 'pipeline_completed')
                card = events[-1]['agent']['messages'][-1]['pipeline']
                self.assertEqual(card['steps'][0]['arguments'], {'query': 'mcp language:python', 'limit': 2})
                self.assertEqual(requests[0].url.params['per_page'], '2')
                self.assertEqual(card['steps'][1]['arguments']['search_result'], card['steps'][0]['result']['structuredContent'])
                self.assertEqual(card['steps'][2]['arguments']['report'], card['steps'][1]['result']['structuredContent'])
                self.assertEqual(Path(card['saved_file']['path']).read_bytes(), card['steps'][1]['result']['structuredContent']['markdown'].encode())
                self.assertEqual(len(list(directory.iterdir())), 1)
                ask.assert_called_once()
            finally:
                connection.close(); chat.shutdown(); chat.server_close(); thread.join(timeout=5)

    def test_sequential_http_calls_publish_search_before_blocked_summarize_and_save_exact_report(self):
        reached_summary, release_summary, published_search = (threading.Event() for _ in range(3))
        summary_inputs, save_inputs, calls, events = [], [], [], []
        original_summarize, original_save = pipeline_tools.summarize, pipeline_tools.save_to_file
        def blocked_summary(search_result):
            summary_inputs.append(search_result)
            reached_summary.set()
            if not release_summary.wait(5):
                raise RuntimeError('Test barrier timed out')
            return original_summarize(search_result)
        def record_save(report, **kwargs):
            save_inputs.append(report)
            return original_save(report, **kwargs)
        def emit(event):
            events.append(event)
            if event['type'] == 'step_completed' and event['step'] == 'search':
                published_search.set()
        with running_mcp() as (http_invoke, directory, requests):
            def invoke(url, name, args):
                calls.append((name, args))
                return http_invoke(url, name, args)
            with patch('pipeline_tools.summarize', blocked_summary), patch('pipeline_tools.save_to_file', record_save):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_pipeline, 'mcp', invoke=invoke, emit=emit)
                    try:
                        self.assertTrue(reached_summary.wait(5))
                        self.assertTrue(published_search.is_set())
                        self.assertFalse(future.done())
                        self.assertEqual(list(directory.iterdir()), [])
                    finally:
                        release_summary.set()
                    result = future.result(timeout=5)
            self.assertEqual(result['status'], 'success', result)
            self.assertEqual([name for name, _ in calls], ['search', 'summarize', 'saveToFile'])
            self.assertEqual(len(requests), 1)
            search = result['trace'][0]['result']['structuredContent']
            report = result['trace'][1]['result']['structuredContent']
            self.assertEqual(summary_inputs, [search])
            self.assertEqual(save_inputs, [report])
            self.assertEqual(calls[1][1], {'search_result': search})
            self.assertEqual(calls[2][1], {'report': report})
            saved = result['saved_file']
            content = Path(saved['path']).read_bytes()
            self.assertEqual(content, report['markdown'].encode('utf-8'))
            self.assertEqual(saved['bytes_written'], len(content))
            self.assertEqual(saved['sha256'], hashlib.sha256(content).hexdigest())
            self.assertEqual(len(list(directory.iterdir())), 1)

    def test_empty_search_still_completes_all_steps_and_writes_empty_report(self):
        with running_mcp({'total_count': 0, 'incomplete_results': False, 'items': []}) as (invoke, directory, requests):
            result = run_pipeline('mcp', invoke=invoke)
            self.assertEqual(result['status'], 'success')
            self.assertEqual(len(result['trace']), 3)
            self.assertIn('Ничего не найдено', Path(result['saved_file']['path']).read_text())
            self.assertEqual(len(list(directory.iterdir())), 1)
            self.assertEqual(len(requests), 1)

    def test_github_error_stops_after_search_without_report(self):
        with running_mcp({'message': 'SECRET'}, status=503) as (invoke, directory, requests):
            result = run_pipeline('mcp', invoke=invoke)
            self.assertEqual(result['status'], 'error')
            self.assertEqual(result['failed_step'], 'search')
            self.assertEqual(len(result['trace']), 1)
            self.assertNotIn('SECRET', result['error'])
            self.assertEqual(list(directory.iterdir()), [])
            self.assertEqual(len(requests), 1)

    def test_chat_command_streams_real_mcp_progress_and_restores_saved_card_without_llm(self):
        from agent import AgentRegistry
        from web import ChatServer
        reached_summary, release_summary = threading.Event(), threading.Event()
        original_summarize = pipeline_tools.summarize
        def blocked_summary(search_result):
            reached_summary.set()
            if not release_summary.wait(5):
                raise RuntimeError('Test barrier timed out')
            return original_summarize(search_result)
        with running_mcp() as (invoke, directory, requests), TemporaryDirectory() as state_dir:
            state_path = Path(state_dir) / 'agents.json'
            ask = Mock(side_effect=AssertionError('Research must not call an LLM'))
            chat = ChatServer(('127.0.0.1', 0), ask, state_path)
            chat.pipeline_runner = lambda query, emit: run_pipeline(query, invoke=invoke, emit=emit)
            thread = threading.Thread(target=chat.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection(*chat.server_address, timeout=5)
            try:
                with patch('pipeline_tools.summarize', blocked_summary):
                    connection.request('POST', '/api/agents/agent-1/messages', json.dumps({'text': '/research mcp'}),
                                       {'Content-Type': 'application/json'})
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertIn('application/x-ndjson', response.getheader('Content-Type'))
                    events = [json.loads(response.readline()) for _ in range(3)]
                    self.assertEqual([event['type'] for event in events],
                                     ['pipeline_started', 'step_started', 'step_completed'])
                    self.assertEqual(events[-1]['step'], 'search')
                    self.assertTrue(reached_summary.wait(5))
                    self.assertEqual(list(directory.iterdir()), [])
                    release_summary.set()
                    events.extend(json.loads(line) for line in response)
                self.assertEqual(events[-1]['type'], 'pipeline_completed', events[-1])
                self.assertEqual([event['seq'] for event in events], list(range(1, 9)))
                agent = events[-1]['agent']
                self.assertEqual(len(agent['messages']), 2)
                card = agent['messages'][-1]['pipeline']
                self.assertEqual(card['status'], 'success')
                steps = card['steps']
                self.assertEqual([step['status'] for step in steps], ['success'] * 3)
                self.assertEqual(steps[1]['arguments']['search_result'], steps[0]['result']['structuredContent'])
                self.assertEqual(steps[2]['arguments']['report'], steps[1]['result']['structuredContent'])
                saved = card['saved_file']
                content = Path(saved['path']).read_bytes()
                self.assertEqual(content, steps[1]['result']['structuredContent']['markdown'].encode())
                self.assertEqual(saved['sha256'], hashlib.sha256(content).hexdigest())
                restored = AgentRegistry(ask, state_path).get('agent-1').snapshot()
                self.assertEqual(restored['messages'][-1]['pipeline'], card)
                self.assertEqual(len(restored['metadata']['toolCalls']), 3)
                self.assertEqual(restored['metadata']['modelRequests'], [])
                self.assertEqual(len(requests), 1)
                ask.assert_not_called()
            finally:
                release_summary.set()
                connection.close()
                chat.shutdown()
                chat.server_close()
                thread.join(timeout=5)
