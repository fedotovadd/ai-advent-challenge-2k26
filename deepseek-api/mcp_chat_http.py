"""Optional NDJSON transport for showing real MCP calls in the chat."""
import json
from uuid import uuid4


def handle_mcp_chat(handler, agent_id, text, temperature):
    registry = handler.server.registry
    agent = registry.get(agent_id)
    if agent is None:
        handler._send_json(404, {'error': 'Агент не найден.'})
        return
    branch_id = agent.snapshot().get('context', {}).get('activeBranch')
    run_id = uuid4().hex
    seq = 0
    connected = True

    handler.close_connection = True
    handler.send_response(200)
    handler.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
    handler.send_header('X-Stream-Type', 'mcp-chat')
    handler.send_header('Cache-Control', 'no-cache')
    handler.end_headers()

    def send(event):
        nonlocal seq, connected
        if not connected:
            return
        seq += 1
        message = {
            'run_id': run_id, 'agent_id': agent_id, 'branch_id': branch_id,
            'seq': seq, **event,
        }
        try:
            handler.wfile.write((json.dumps(message, ensure_ascii=False) + '\n').encode('utf-8'))
            handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            connected = False

    try:
        snapshot = registry.respond(agent_id, text, temperature, on_tool_event=send)
        send({'type': 'mcp_chat_completed', 'agent': snapshot, 'metadata': snapshot.get('metadata')})
    except Exception as error:
        current = registry.get(agent_id)
        snapshot = current.snapshot() if current else None
        send({
            'type': 'mcp_chat_failed', 'agent': snapshot,
            'metadata': snapshot.get('metadata') if snapshot else None,
            'error': _public_error(error),
            'accepted': getattr(error, 'accepted', True),
            'http_status': _http_status(error),
            'probe': error.__class__.__name__ == 'OverflowProbeError',
        })


def _public_error(error):
    name = error.__class__.__name__
    if name == 'InvariantViolationError':
        return getattr(error, 'refusal', 'Ответ нарушает ограничения агента.')
    if name == 'MissingApiKeyError':
        return f"Не задан {getattr(error, 'key_name', 'ключ API')}."
    if name == 'ContextOverflowError':
        return str(error)
    if name == 'OverflowProbeError':
        return str(error)
    if name == 'TaskStateError':
        return str(error)
    if name == 'FactsUpdateError':
        return str(error)
    if name == 'PersistenceError':
        return 'Не удалось сохранить состояние агента.'
    if name == 'ContextSummaryError':
        return 'Не удалось обновить сводку истории.'
    if name == 'ValueError' and str(error) == 'Пустое сообщение.':
        return str(error)
    return 'Не удалось получить ответ DeepSeek.'


def _http_status(error):
    return {
        'InvariantViolationError': 400,
        'TaskStateError': 400,
        'ValueError': 400 if str(error) in {'Пустое сообщение.', 'Поле temperature должно быть числом от 0 до 2.'} else 502,
        'MissingApiKeyError': 503,
        'ContextOverflowError': 409,
        'OverflowProbeError': 422,
        'FactsUpdateError': 502,
        'ContextSummaryError': 502,
        'PersistenceError': 500,
    }.get(error.__class__.__name__, 502)
