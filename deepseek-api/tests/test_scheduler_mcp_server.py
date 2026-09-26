import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from mcp.shared.memory import create_connected_server_and_client_session

from scheduler import ScheduleStore, Scheduler


class SchedulerMCPExistsTests(unittest.TestCase):
    def test_scheduler_mcp_is_implemented(self):
        self.assertIsNotNone(importlib.util.find_spec('scheduler_mcp_server'))

    def test_chat_client_accepts_only_exact_scheduler_url(self):
        from mcp_client import validate_url
        self.assertEqual(validate_url('http://127.0.0.1:8002/mcp'), 'http://127.0.0.1:8002/mcp')
        for url in ['http://127.0.0.1:8002/other', 'http://127.0.0.1:8003/mcp', 'http://127.0.0.1:8002/mcp?x=1']:
            with self.assertRaises(ValueError):
                validate_url(url)


@unittest.skipUnless(importlib.util.find_spec('scheduler_mcp_server'), 'implementation pending')
class SchedulerMCPTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ScheduleStore(Path(self.temp.name) / 'db.sqlite3')

    async def test_mcp_discovery_schedule_summary_and_cancel(self):
        from scheduler_mcp_server import create_server
        server = create_server(self.store)
        async with create_connected_server_and_client_session(server) as session:
            tools = (await session.list_tools()).tools
            self.assertEqual({t.name for t in tools}, {
                'schedule_repository', 'list_schedules', 'get_schedule_summary', 'cancel_schedule'})
            for tool in tools:
                self.assertFalse(tool.inputSchema['additionalProperties'])
                self.assertTrue(tool.description)
            result = await session.call_tool('schedule_repository', {
                'owner': 'octocat', 'repo': 'Hello-World', 'delay_seconds': 30, 'interval_seconds': 60})
            self.assertFalse(result.isError)
            job = result.structuredContent['job']
            self.assertEqual(job['interval_seconds'], 60)
            listed = await session.call_tool('list_schedules', {})
            self.assertEqual(listed.structuredContent['total'], 1)
            summary = await session.call_tool('get_schedule_summary', {'job_id': job['id']})
            self.assertEqual(summary.structuredContent['success_count'], 0)
            self.assertEqual(json.loads(summary.content[0].text), summary.structuredContent)
            cancelled = await session.call_tool('cancel_schedule', {'job_id': job['id']})
            self.assertEqual(cancelled.structuredContent['job']['status'], 'cancelled')

    async def test_invalid_arguments_and_unknown_job_return_mcp_errors(self):
        from scheduler_mcp_server import create_server
        async with create_connected_server_and_client_session(create_server(self.store)) as session:
            for args in [
                {'owner': 'a', 'repo': 'b', 'delay_seconds': -1},
                {'owner': 'a', 'repo': 'b', 'interval_seconds': True},
                {'owner': 'a', 'repo': 'b', 'delay_seconds': '10'},
                {'owner': 'a', 'repo': 'b', 'unexpected': True},
                {'owner': '../x', 'repo': 'b'},
            ]:
                result = await session.call_tool('schedule_repository', args)
                self.assertTrue(result.isError, args)
            result = await session.call_tool('get_schedule_summary', {'job_id': 'missing'})
            self.assertTrue(result.isError)
            self.assertIn('не найдено', result.content[0].text)
            self.assertEqual(self.store.list_schedules()['total'], 0)

    async def test_long_running_summaries_fit_chat_transport_limit(self):
        from scheduler_mcp_server import create_server
        from mcp_client import MAX_RESULT_BYTES
        now = 1000
        self.store.clock = lambda: now
        async def fetch(owner, repo):
            return {'full_name': owner + '/' + repo, 'stargazers_count': 42, 'forks_count': 7,
                    'html_url': 'https://github.com/' + owner + '/' + repo,
                    'description': 'Большое описание' * 10000}
        worker = Scheduler(self.store, fetch=fetch, emit=lambda value: None)
        jobs = [self.store.schedule('a' * 100, 'b' * 100, interval_seconds=10) for _ in range(4)]
        for _ in range(105):
            await worker.run_due()
            now += 10
        async with create_connected_server_and_client_session(create_server(self.store)) as session:
            for name, args in [('get_schedule_summary', {'job_id': jobs[0]['id']}), ('list_schedules', {})]:
                result = await session.call_tool(name, args)
                self.assertFalse(result.isError)
                self.assertLess(len(json.dumps(result.model_dump(mode='json'), ensure_ascii=False).encode()), MAX_RESULT_BYTES)
            history = await session.call_tool('get_schedule_summary', {'job_id': jobs[0]['id'], 'history_offset': 3})
            self.assertEqual(len(history.structuredContent['recent_runs']), 3)

    async def test_worker_runs_without_mcp_client_and_stops_with_application(self):
        from scheduler_mcp_server import create_app
        completed = asyncio.Event()
        async def fetch(owner, repo):
            return {'stargazers_count': 12, 'forks_count': 3}
        worker = Scheduler(self.store, fetch=fetch, emit=lambda value: completed.set())
        job = self.store.schedule('a', 'b')
        app = create_app(self.store, worker=worker)
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(completed.wait(), 2)
            self.assertEqual(self.store.summary(job['id'])['success_count'], 1)
            self.assertFalse(app.state.worker_task.done())
        self.assertTrue(app.state.worker_task.done())

    async def test_dashboard_and_api_show_saved_results_and_reject_foreign_host(self):
        from scheduler_mcp_server import create_app
        self.store.schedule('octocat', 'Hello-World', delay_seconds=300)
        app = create_app(self.store)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1:8002') as client:
            page = await client.get('/')
            self.assertEqual(page.status_code, 200)
            self.assertIn('Сводки по расписанию', page.text)
            response = await client.get('/api/schedules')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['total'], 1)
            self.assertEqual(response.headers['cache-control'], 'no-store')
            self.assertEqual((await client.get('/api/schedules?offset=-1')).status_code, 400)
            self.assertEqual((await client.post('/api/schedules')).status_code, 405)
            self.assertEqual((await client.get('/api/schedules', headers={'host': 'evil.example'})).status_code, 400)
