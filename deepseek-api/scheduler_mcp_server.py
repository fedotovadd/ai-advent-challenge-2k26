"""Day 18: python scheduler_mcp_server.py; dashboard at http://127.0.0.1:8002/."""

import argparse
import asyncio
import json
import sqlite3
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import HTMLResponse, JSONResponse

from scheduler import MAX_SECONDS, ScheduleStore, Scheduler

DEFAULT_DB = Path(__file__).parent / 'data' / 'scheduler.sqlite3'
DASHBOARD = Path(__file__).parent / 'static' / 'scheduler.html'


def result_of(operation, *args, **kwargs):
    try:
        data = operation(*args, **kwargs)
    except (ValueError, sqlite3.Error) as error:
        message = str(error) if isinstance(error, ValueError) else 'Хранилище планировщика недоступно.'
        return CallToolResult(isError=True, content=[TextContent(type='text', text=message)])
    return CallToolResult(content=[TextContent(type='text', text=json.dumps(data, ensure_ascii=False))], structuredContent=data)


def create_server(store: ScheduleStore, *, port=8002):
    mcp = FastMCP('Scheduled GitHub summaries', host='127.0.0.1', port=port,
                  streamable_http_path='/mcp', instructions=(
                      'Фоновые проверки GitHub. Создавайте расписание только по просьбе пользователя. '
                      'Расписания общие для всех агентов, выполняются даже без чата. '
                      'Для остановки вызовите cancel_schedule. Сводка доступна через get_schedule_summary '
                      'и автоматически обновляемую страницу на корневом URL сервера.'))
    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    @mcp.tool(description=(
        'Создать фоновую проверку публичного GitHub-репозитория. Возвращает ID, расписание и пока пустую сводку. '
        'delay_seconds — задержка первого запуска (0 = сразу); interval_seconds=null — один запуск, '
        'иначе повторять с этим периодом. Данные и сводки сохраняются в SQLite. '
        'Каждый вызов создаёт новое задание; перед повтором при сомнении проверьте list_schedules.'),
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True))
    def schedule_repository(
        owner: Annotated[str, Field(description='Владелец публичного репозитория GitHub.')],
        repo: Annotated[str, Field(description='Имя репозитория без URL и владельца.')],
        delay_seconds: Annotated[int, Field(strict=True, ge=0, le=MAX_SECONDS, description='Задержка первого запуска в секундах.')] = 0,
        interval_seconds: Annotated[int, Field(strict=True, ge=10, le=MAX_SECONDS, description='Период в секундах, от 10; null для однократного запуска.')] | None = None,
    ) -> CallToolResult:
        def create():
            job = store.schedule(owner, repo, delay_seconds=delay_seconds, interval_seconds=interval_seconds)
            return store.summary(job['id'], history_limit=3)
        return result_of(create)

    @mcp.tool(description='Список общих фоновых заданий с накопительными сводками; limit/offset для страниц.', annotations=read_only)
    def list_schedules(
        limit: Annotated[int, Field(strict=True, ge=1, le=3, description='Размер страницы, до 3 заданий, чтобы сводки помещались в контекст чата.')] = 3,
        offset: Annotated[int, Field(strict=True, ge=0, description='Число пропускаемых заданий.')] = 0,
    ) -> CallToolResult:
        return result_of(store.list_schedules, limit=limit, offset=offset)

    @mcp.tool(description=(
        'Получить сохранённую агрегированную сводку: успешные проверки, ошибки, звёзды/форки '
        'и их изменение с первого успешного наблюдения, даты и историю попыток по 3 записи на страницу (хранятся последние 100). '
        'Не делает новый запрос GitHub. Ноль изменений не означает отсутствие новых коммитов.'), annotations=read_only)
    def get_schedule_summary(
        job_id: Annotated[str, Field(description='ID задания из schedule_repository или list_schedules.')],
        history_offset: Annotated[int, Field(strict=True, ge=0, description='Пропустить столько записей истории; 0 — последние три.')] = 0,
    ) -> CallToolResult:
        return result_of(store.summary, job_id, history_limit=3, history_offset=history_offset)

    @mcp.tool(description=(
        'Отменить будущие запуски задания по ID. История сохраняется; уже начавшаяся проверка может завершиться. '
        'Отключение MCP в чате само по себе не отменяет расписание.'),
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
    def cancel_schedule(
        job_id: Annotated[str, Field(description='ID задания, которое нужно остановить.')],
    ) -> CallToolResult:
        return result_of(lambda: {'job': store.cancel(job_id)})

    for name in ('schedule_repository', 'list_schedules', 'get_schedule_summary', 'cancel_schedule'):
        tool = mcp._tool_manager.get_tool(name)
        tool.fn_metadata.arg_model.model_config['extra'] = 'forbid'
        tool.fn_metadata.arg_model.model_rebuild(force=True)
        tool.parameters = tool.fn_metadata.arg_model.model_json_schema()

    @mcp.custom_route('/', methods=['GET'])
    async def dashboard(request):
        return HTMLResponse(DASHBOARD.read_text(encoding='utf-8'), headers={'Cache-Control': 'no-store'})

    @mcp.custom_route('/api/schedules', methods=['GET'])
    async def schedules(request):
        try:
            offset = int(request.query_params.get('offset', '0'))
            data = store.list_schedules(offset=offset)
        except ValueError:
            return JSONResponse({'error': 'Некорректная страница заданий.'}, status_code=400)
        except sqlite3.Error:
            return JSONResponse({'error': 'Хранилище планировщика недоступно.'}, status_code=503)
        return JSONResponse(data, headers={'Cache-Control': 'no-store'})

    return mcp


def create_app(store=None, *, worker=None, port=8002):
    store = store if store is not None else ScheduleStore(DEFAULT_DB)
    worker = worker if worker is not None else Scheduler(store)
    app = create_server(store, port=port).streamable_http_app()
    mcp_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with mcp_lifespan(app):
            task = asyncio.create_task(worker.run(), name='github-scheduler')
            app.state.worker_task = task
            try:
                yield
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    app.router.lifespan_context = lifespan
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]'])
    return app


def main():
    parser = argparse.ArgumentParser(description='Фоновый MCP-планировщик GitHub и страница сводок.')
    parser.add_argument('--db', type=Path, default=DEFAULT_DB, help='Путь к SQLite (по умолчанию data/scheduler.sqlite3).')
    parser.add_argument('--port', type=int, default=8002)
    args = parser.parse_args()
    uvicorn.run(create_app(ScheduleStore(args.db), port=args.port), host='127.0.0.1', port=args.port, access_log=False)


if __name__ == '__main__':
    main()
