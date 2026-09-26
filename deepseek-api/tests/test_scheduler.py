import asyncio
import importlib.util
import tempfile
import unittest
from pathlib import Path


class SchedulerExistsTests(unittest.TestCase):
    def test_scheduler_is_implemented(self):
        self.assertIsNotNone(importlib.util.find_spec('scheduler'))


@unittest.skipUnless(importlib.util.find_spec('scheduler'), 'implementation pending')
class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from scheduler import ScheduleStore, Scheduler
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'jobs.sqlite3'
        self.now = 1000.0
        self.store = ScheduleStore(self.path, clock=lambda: self.now)
        self.calls = []
        self.emitted = []

        async def fetch(owner, repo):
            self.calls.append((owner, repo))
            return {'full_name': f'{owner}/{repo}', 'stargazers_count': 40 + len(self.calls),
                    'forks_count': 7 + len(self.calls), 'html_url': f'https://github.com/{owner}/{repo}'}
        self.worker = Scheduler(self.store, fetch=fetch, emit=self.emitted.append)

    def schedule(self, **kwargs):
        return self.store.schedule('octocat', 'Hello-World', **kwargs)

    async def test_delayed_once_survives_restart_and_never_runs_early_or_twice(self):
        from scheduler import ScheduleStore, Scheduler
        job = self.schedule(delay_seconds=30)
        self.assertEqual(await self.worker.run_due(), 0)
        self.store = ScheduleStore(self.path, clock=lambda: self.now)
        self.worker = Scheduler(self.store, fetch=self.worker.fetch, emit=self.emitted.append)
        self.now += 30
        self.assertEqual(await self.worker.run_due(), 1)
        self.assertEqual(await self.worker.run_due(), 0)
        result = self.store.summary(job['id'])
        self.assertEqual(result['job']['status'], 'completed')
        self.assertEqual(result['success_count'], 1)
        self.assertEqual(len(result['recent_runs']), 1)
        self.assertEqual(len(self.emitted), 1)

    async def test_periodic_aggregation_and_missed_intervals_are_coalesced(self):
        job = self.schedule(interval_seconds=10)
        await self.worker.run_due()
        self.now += 95
        self.assertEqual(await self.worker.run_due(), 1)
        summary = self.store.summary(job['id'])
        self.assertEqual(summary['success_count'], 2)
        self.assertEqual(summary['stars_delta'], 1)
        self.assertEqual(summary['forks_delta'], 1)
        self.assertEqual(summary['latest']['stargazers_count'], 42)
        self.assertEqual(summary['job']['next_run_at'], '1970-01-01T00:18:20Z')
        self.assertIn('2', summary['text'])

    async def test_failure_is_persisted_and_next_interval_recovers(self):
        from github_api import GitHubAPIError
        job = self.schedule(interval_seconds=10)
        original = self.worker.fetch
        async def fail(*args):
            raise GitHubAPIError('Превышен лимит запросов GitHub.')
        self.worker.fetch = fail
        await self.worker.run_due()
        result = self.store.summary(job['id'])
        self.assertEqual(result['error_count'], 1)
        self.assertIsNone(result['latest'])
        self.assertIsNone(result['stars_delta'])
        self.assertIn('лимит', result['last_error'])
        self.worker.fetch = original
        self.now += 10
        await self.worker.run_due()
        self.assertEqual(self.store.summary(job['id'])['success_count'], 1)

    async def test_cancel_keeps_history_and_stops_future_work(self):
        job = self.schedule(interval_seconds=10)
        await self.worker.run_due()
        self.store.cancel(job['id'])
        self.store.cancel(job['id'])
        self.now += 10
        self.assertEqual(await self.worker.run_due(), 0)
        result = self.store.summary(job['id'])
        self.assertEqual(result['job']['status'], 'cancelled')
        self.assertEqual(result['success_count'], 1)

    async def test_two_workers_do_not_execute_same_job(self):
        from scheduler import ScheduleStore, Scheduler
        self.schedule()
        other = Scheduler(ScheduleStore(self.path, clock=lambda: self.now),
                          fetch=self.worker.fetch, emit=self.emitted.append)
        await asyncio.gather(self.worker.run_due(), other.run_due())
        self.assertEqual(len(self.calls), 1)

    async def test_expired_lease_is_recovered_and_stale_completion_rejected(self):
        job = self.schedule()
        stale = self.store.claim()
        self.assertIsNone(self.store.claim())
        self.now += 61
        await self.worker.run_due()
        self.assertFalse(self.store.finish(stale, data={'stargazers_count': 999, 'forks_count': 0}))
        self.assertEqual(self.store.summary(job['id'])['success_count'], 1)

    async def test_cancel_during_fetch_cannot_reactivate_job(self):
        job = self.schedule(interval_seconds=10)
        original = self.worker.fetch
        async def fetch(*args):
            self.store.cancel(job['id'])
            return await original(*args)
        self.worker.fetch = fetch
        await self.worker.run_due()
        self.assertEqual(self.store.summary(job['id'])['job']['status'], 'cancelled')
        self.now += 10
        self.assertEqual(await self.worker.run_due(), 0)

    def test_cancelled_job_is_not_shown_running_forever_after_worker_crash(self):
        job = self.schedule()
        self.store.claim()
        self.store.cancel(job['id'])
        self.assertTrue(self.store.summary(job['id'])['job']['running'])
        self.now += 61
        self.assertFalse(self.store.summary(job['id'])['job']['running'])

    async def test_recent_history_is_bounded_but_aggregation_keeps_baseline(self):
        job = self.schedule(interval_seconds=10)
        for _ in range(105):
            await self.worker.run_due()
            self.now += 10
        summary = self.store.summary(job['id'])
        self.assertEqual(len(summary['recent_runs']), 100)
        self.assertEqual(summary['success_count'], 105)
        self.assertEqual(summary['stars_delta'], 104)

    async def test_unexpected_failure_is_safe_and_does_not_stop_other_jobs(self):
        self.schedule()
        self.schedule()
        async def fail(*args):
            raise RuntimeError('SECRET')
        self.worker.fetch = fail
        self.assertEqual(await self.worker.run_due(), 2)
        self.assertNotIn('SECRET', str(self.store.list_schedules()))
        self.assertEqual(len(self.emitted), 2)

    async def test_timeout_is_recorded(self):
        job = self.schedule()
        async def hang(*args):
            await asyncio.sleep(10)
        self.worker.fetch = hang
        self.worker.fetch_timeout = .01
        await self.worker.run_due()
        self.assertEqual(self.store.summary(job['id'])['error_count'], 1)

    def test_validation_and_unknown_ids(self):
        for options in [{'delay_seconds': -1}, {'delay_seconds': True},
                        {'interval_seconds': 0}, {'interval_seconds': 1},
                        {'interval_seconds': 1.5}, {'delay_seconds': float('nan')}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.schedule(**options)
        for owner in ['', '../x', 'x/y']:
            with self.assertRaises(ValueError):
                self.store.schedule(owner, 'repo')
        for action in [self.store.cancel, self.store.summary]:
            with self.assertRaises(ValueError):
                action('missing')
