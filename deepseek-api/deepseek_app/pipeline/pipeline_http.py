"""HTTP adapter for persisted, incremental MCP research runs."""
import copy
import json
from functools import wraps
from urllib.parse import urlparse


def guarded_agent_request(method):
    """An active research excludes mutations; ordinary requests retain their semantics."""
    @wraps(method)
    def wrapped(self):
        parts = urlparse(self.path).path.split('/')
        if len(parts) < 4 or parts[:3] != ['', 'api', 'agents'] or not parts[3].startswith('agent-'):
            return method(self)
        key = parts[3]
        with self.server.operation_locks_guard:
            entry = self.server.operation_locks.setdefault(key, {'active': 0, 'research': False})
            admitted = not entry['research']
            if admitted:
                entry['active'] += 1
        if not admitted:
            self.close_connection = True
            self._send_json(409, {'error': 'Агент занят исследованием. Дождитесь завершения.', 'accepted': False})
            return
        self._pipeline_exclusive = False
        try:
            return method(self)
        finally:
            with self.server.operation_locks_guard:
                entry['active'] -= 1
                if self._pipeline_exclusive:
                    entry['research'] = False
                if entry['active'] == 0:
                    self.server.operation_locks.pop(key, None)
    return wrapped


def handle_research(handler, agent_id, text, query):
    # Import here to keep HTTP routing independent from the MCP SDK lifecycle.
    from deepseek_app.agent.agent import PersistenceError
    from deepseek_app.pipeline.research_intent import raw_search_arguments, interpret_research
    registry = handler.server.registry
    try:
        arguments = raw_search_arguments(query)
    except ValueError:
        handler._send_json(400, {'error': 'Запрос GitHub должен быть одной строкой от 1 до 200 символов.', 'accepted': False})
        return
    with handler.server.operation_locks_guard:
        entry = handler.server.operation_locks[agent_id]
        admitted = entry['active'] == 1 and not entry['research']
        if admitted:
            entry['research'] = True
            handler._pipeline_exclusive = True
    if not admitted:
        handler._send_json(409, {'error': 'Агент занят. Дождитесь завершения текущей операции.', 'accepted': False})
        return
    try:
        snapshot, card = registry.begin_pipeline(agent_id, text, arguments['query'] if arguments else query,
                                                  interpret=arguments is None)
    except KeyError:
        handler._send_json(404, {'error': 'Агент не найден.', 'accepted': False})
        return
    except PersistenceError:
        handler._send_json(500, {'error': 'Не удалось сохранить начало исследования.', 'accepted': False})
        return
    connected = True
    seq = 0
    handler.close_connection = True
    try:
        handler.send_response(200)
        handler.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
        handler.send_header('Cache-Control', 'no-cache')
        handler.end_headers()
    except (BrokenPipeError, ConnectionResetError, OSError):
        connected = False

    def send(event, current):
        nonlocal connected, seq
        seq += 1
        envelope = {**event, 'run_id': card['run_id'], 'agent_id': agent_id,
                    'branch_id': card['branch_id'], 'seq': seq, 'agent': current}
        if connected:
            try:
                handler.wfile.write((json.dumps(envelope, ensure_ascii=False) + '\n').encode('utf-8'))
                handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                connected = False

    def emit(event):
        current = registry.update_pipeline(agent_id, card['run_id'], event)
        send(event, current)

    send({'type': 'pipeline_started', 'query': query, 'limit': 3, 'steps': card['steps']}, snapshot)
    outcome = None
    try:
        if arguments is None:
            send({'type': 'interpretation_started'}, snapshot)
            interpreted = interpret_research(query, handler.server.ask_model, snapshot['settings']['model'])
            snapshot = registry.resolve_pipeline(agent_id, card['run_id'], interpreted)
            arguments = interpreted['arguments']
            send({'type': 'interpretation_completed' if arguments else 'interpretation_failed',
                  'arguments': arguments, 'error': interpreted['error']}, snapshot)
            if arguments is None:
                outcome = {'status': 'error', 'error': interpreted['error'], 'saved_file': None, 'failed_step': None}
                snapshot = registry.finish_pipeline(agent_id, card['run_id'], outcome)
                send({'type': 'pipeline_failed', **outcome}, snapshot)
                return
        runner = handler.server.pipeline_runner
        if runner is None:
            from deepseek_app.pipeline.pipeline_runner import run_pipeline
            runner = run_pipeline
        options = {'limit': arguments['limit']} if arguments['limit'] != 3 else {}
        outcome = runner(arguments['query'], emit=emit, **options)
        snapshot = registry.finish_pipeline(agent_id, card['run_id'], outcome)
        send({'type': 'pipeline_completed' if outcome['status'] == 'success' else 'pipeline_failed', **outcome}, snapshot)
    except Exception as error:
        message = ('Не удалось сохранить состояние исследования. Уже созданный файл не удалён; автоматического повтора нет.'
                   if isinstance(error, PersistenceError) else 'Исследование прервано. Итог последнего шага может быть неизвестен; автоматического повтора нет.')
        saved = outcome.get('saved_file') if outcome else None
        failure = {'status': 'error', 'error': message, 'failed_step': None, 'saved_file': saved}
        try:
            snapshot = registry.finish_pipeline(agent_id, card['run_id'], failure)
        except Exception:
            snapshot = registry.get(agent_id).snapshot()
        trace = copy.deepcopy(snapshot['messages'][-1].get('pipeline', {}).get('steps', []))
        send({'type': 'pipeline_failed', **failure, 'trace': trace}, snapshot)
