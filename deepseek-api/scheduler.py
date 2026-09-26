"""Durable schedules and a worker independent of chat/MCP client sessions."""

import asyncio
import json
import logging
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import github_api

LEASE_SECONDS = 60
MAX_SECONDS = 31_536_000
HISTORY_LIMIT = 100
logger = logging.getLogger(__name__)


def utc(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace('+00:00', 'Z') if value is not None else None


class ScheduleStore:
    def __init__(self, path, *, clock=time.time):
        self.path = Path(path)
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, repo TEXT NOT NULL,
                    created_at REAL NOT NULL, next_run REAL, interval_seconds INTEGER,
                    status TEXT NOT NULL, token TEXT, lease_until REAL,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    error_count INTEGER NOT NULL DEFAULT 0,
                    first_data TEXT, latest_data TEXT,
                    first_success REAL, last_success REAL, last_error TEXT
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                    scheduled_at REAL NOT NULL, finished_at REAL NOT NULL,
                    data TEXT, error TEXT, UNIQUE(job_id, scheduled_at)
                );
                CREATE INDEX IF NOT EXISTS due_jobs ON jobs(status, next_run);
                CREATE INDEX IF NOT EXISTS job_runs ON runs(job_id, id);
            ''')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def public_job(self, row):
        return {key: row[key] for key in ('id', 'owner', 'repo', 'interval_seconds', 'status')} | {
            'created_at': utc(row['created_at']), 'next_run_at': utc(row['next_run']),
            'running': row['token'] is not None and row['lease_until'] > self.clock(),
        }

    @staticmethod
    def require_job(db, job_id):
        row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row is None:
            raise ValueError('Задание не найдено. Получите ID через list_schedules.')
        return row

    def schedule(self, owner, repo, *, delay_seconds=0, interval_seconds=None):
        for value in (owner, repo):
            if (not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', value)
                    or value in {'.', '..'}):
                raise ValueError('Укажите корректные owner и repo GitHub без URL и разделителей пути.')
        for value, minimum in ((delay_seconds, 0), (interval_seconds, 10)):
            if value is None and minimum == 10:
                continue
            if type(value) is not int or not minimum <= value <= MAX_SECONDS:
                raise ValueError('Задержка: 0–31536000 секунд; период: 10–31536000 секунд или null.')
        job_id, now = uuid.uuid4().hex, self.clock()
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT count(*) FROM jobs WHERE status='active'").fetchone()[0] >= 100:
                raise ValueError('Допускается до 100 активных заданий. Отмените ненужные.')
            db.execute('''INSERT INTO jobs(id,owner,repo,created_at,next_run,interval_seconds,status)
                          VALUES(?,?,?,?,?,?, 'active')''',
                       (job_id, owner, repo, now, now + delay_seconds, interval_seconds))
            return self.public_job(self.require_job(db, job_id))

    def cancel(self, job_id):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self.require_job(db, job_id)
            db.execute("UPDATE jobs SET status='cancelled',next_run=NULL WHERE id=? AND status='active'", (job_id,))
            return self.public_job(self.require_job(db, job_id))

    def claim(self):
        now = self.clock()
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('''SELECT * FROM jobs WHERE status='active' AND next_run<=?
                                AND (token IS NULL OR lease_until<=?) ORDER BY next_run,id LIMIT 1''', (now, now)).fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex
            db.execute('UPDATE jobs SET token=?,lease_until=? WHERE id=?', (token, now + LEASE_SECONDS, row['id']))
            return dict(row) | {'token': token}

    def finish(self, claim, *, data=None, error=None):
        now = self.clock()
        encoded = json.dumps(data, ensure_ascii=False) if data is not None else None
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = self.require_job(db, claim['id'])
            if row['token'] != claim['token']:
                return False
            db.execute('INSERT INTO runs(job_id,scheduled_at,finished_at,data,error) VALUES(?,?,?,?,?)',
                       (row['id'], claim['next_run'], now, encoded, error))
            status, next_run = row['status'], None
            if status == 'active':
                interval = row['interval_seconds']
                if interval is None:
                    status = 'completed'
                else:
                    steps = max(1, int((now - claim['next_run']) // interval) + 1)
                    next_run = claim['next_run'] + steps * interval
            db.execute('UPDATE jobs SET status=?,next_run=?,token=NULL,lease_until=NULL WHERE id=?',
                       (status, next_run, row['id']))
            if error is None:
                db.execute('''UPDATE jobs SET success_count=success_count+1,
                              first_data=COALESCE(first_data,?),latest_data=?,
                              first_success=COALESCE(first_success,?),last_success=? WHERE id=?''',
                           (encoded, encoded, now, now, row['id']))
            else:
                db.execute('UPDATE jobs SET error_count=error_count+1,last_error=? WHERE id=?', (error, row['id']))
            db.execute('''DELETE FROM runs WHERE job_id=? AND id NOT IN
                          (SELECT id FROM runs WHERE job_id=? ORDER BY id DESC LIMIT ?)''',
                       (row['id'], row['id'], HISTORY_LIMIT))
            return True

    def summary(self, job_id, *, include_runs=True, history_limit=100, history_offset=0):
        if (type(history_limit) is not int or not 1 <= history_limit <= 100
                or type(history_offset) is not int or history_offset < 0):
            raise ValueError('Некорректная страница истории.')
        with self.connection() as db:
            db.execute('BEGIN')  # Counts and recent runs must describe the same snapshot.
            row = self.require_job(db, job_id)
            first = json.loads(row['first_data']) if row['first_data'] else None
            latest = json.loads(row['latest_data']) if row['latest_data'] else None
            result = {
                'job': self.public_job(row), 'success_count': row['success_count'],
                'error_count': row['error_count'], 'latest': latest,
                'first_observed_at': utc(row['first_success']), 'last_observed_at': utc(row['last_success']),
                'stars_delta': latest['stargazers_count'] - first['stargazers_count'] if latest else None,
                'forks_delta': latest['forks_count'] - first['forks_count'] if latest else None,
                'last_error': row['last_error'],
            }
            if include_runs:
                result['recent_runs'] = [
                    {'scheduled_at': utc(run['scheduled_at']), 'finished_at': utc(run['finished_at']),
                     'data': json.loads(run['data']) if run['data'] else None, 'error': run['error']}
                    for run in db.execute('SELECT * FROM runs WHERE job_id=? ORDER BY id DESC LIMIT ? OFFSET ?',
                                          (job_id, history_limit, history_offset))
                ]
                result['history_total'] = min(row['success_count'] + row['error_count'], HISTORY_LIMIT)
                result['history_offset'] = history_offset
        text = f"{row['owner']}/{row['repo']}: успешных проверок — {row['success_count']}, ошибок — {row['error_count']}."
        if latest:
            text += (f" Звёзды: {latest['stargazers_count']} ({result['stars_delta']:+d}),"
                     f" форки: {latest['forks_count']} ({result['forks_delta']:+d}) с первого наблюдения.")
        else:
            text += ' Успешных наблюдений пока нет.'
        result['text'] = text
        return result

    def list_schedules(self, *, limit=100, offset=0):
        if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or offset < 0:
            raise ValueError('limit: 1–100; offset: целое неотрицательное число.')
        with self.connection() as db:
            ids = [row['id'] for row in db.execute('SELECT id FROM jobs ORDER BY created_at DESC,id LIMIT ? OFFSET ?', (limit, offset))]
            total = db.execute('SELECT count(*) FROM jobs').fetchone()[0]
        return {'schedules': [self.summary(job_id, include_runs=False) for job_id in ids],
                'total': total, 'limit': limit, 'offset': offset}


def emit_summary(summary):
    print(json.dumps({'event': 'schedule_summary', **summary}, ensure_ascii=False), flush=True)


class Scheduler:
    def __init__(self, store, *, fetch=github_api.get_repository, emit=emit_summary):
        self.store, self.fetch, self.emit = store, fetch, emit
        self.fetch_timeout = 20

    async def run_due(self):
        # Bounded batch lets shutdown and API requests run even under sustained load.
        count = 0
        for _ in range(100):
            claim = self.store.claim()
            if claim is None:
                break
            data, error = None, None
            try:
                data = await asyncio.wait_for(self.fetch(claim['owner'], claim['repo']), self.fetch_timeout)
                # Persist only bounded fields needed for this aggregate, not arbitrary descriptions.
                data = {key: data[key] for key in ('stargazers_count', 'forks_count')}
                if any(type(value) is not int or value < 0 for value in data.values()):
                    raise ValueError('Некорректные счётчики GitHub.')
            except github_api.GitHubAPIError as exc:
                error = str(exc)
            except asyncio.TimeoutError:
                error = 'Проверка GitHub превысила время ожидания.'
            except Exception:
                error = 'Не удалось выполнить проверку GitHub.'
            if self.store.finish(claim, data=data, error=error):
                count += 1
                try:
                    self.emit(self.store.summary(claim['id'], include_runs=False))
                except Exception:
                    logger.error('Не удалось вывести сводку; результат сохранён в SQLite.')
        return count

    async def run(self, *, poll_seconds=1):
        while True:
            try:
                await self.run_due()
            except sqlite3.Error:
                logger.error('Хранилище планировщика недоступно; повтор через интервал опроса.')
            await asyncio.sleep(poll_seconds)
