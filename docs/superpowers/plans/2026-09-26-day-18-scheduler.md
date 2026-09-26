# Day 18 Scheduler Implementation Plan

**Goal:** Агент создаёт сохраняемые фоновые проверки GitHub через MCP и получает агрегированную сводку.

**Architecture:** SQLite store + async worker + самостоятельный FastMCP HTTP server с общей application lifespan. Read-only dashboard показывает сохранённые сводки без модели.

**Tech Stack:** Python, sqlite3, asyncio, существующие httpx/MCP v1, Starlette/Uvicorn из MCP.

### 1. Расписание и хранилище
- [x] Написать и запустить падающие тесты `deepseek-api/tests/test_scheduler.py`.
- [x] Создать `deepseek-api/scheduler.py`: транзакции, claim/lease, история, сводка, cancel и worker.
- [x] Проверить deadline, restart, coalescing, гонки и ошибки на управляемых часах.

### 2. MCP и фоновый процесс
- [x] Написать падающие тесты `deepseek-api/tests/test_scheduler_mcp_server.py`.
- [x] Создать `deepseek-api/scheduler_mcp_server.py`: четыре инструмента, строгие схемы, общий lifespan, CLI.
- [x] Проверить настоящий протокол MCP и запуск worker без клиентской сессии.

### 3. Сводки и запуск
- [x] Создать `deepseek-api/static/scheduler.html`, read-only HTTP API и обновление каждые 5 секунд.
- [x] Добавить пример URL и ссылку на сводки в существующую вкладку MCP.
- [x] Документировать запуск, примеры запросов, SQLite, отмену, launchd и ограничения 24/7 в README.

### 4. Проверка результата
- [x] Выполнить полную unittest-регрессию, проверить HTTP/JS интерфейса и настоящий HTTP MCP-сценарий с отложенным запуском.
- [x] Проверить код и записать доказательства в `docs/day-18/scheduler-results.md`.

Визуальная проверка браузером заблокирована автоматической проверкой разрешений; подробности в docs/day-18/scheduler-results.md. Выполнены реальные прогоны с GitHub и DeepSeek.
