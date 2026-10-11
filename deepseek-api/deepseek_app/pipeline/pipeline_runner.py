"""A fixed three-call MCP pipeline, independent of chat, persistence and LLMs."""

import json
from time import perf_counter

from pydantic import ValidationError

from deepseek_app.mcp import mcp_client
from deepseek_app.pipeline.pipeline_models import MAX_RESULT_BYTES, SavedFile, SearchArguments, SearchResult, SummaryResult

PIPELINE_URL = mcp_client.LOCAL_PIPELINE_URL
STEPS = ('search', 'summarize', 'saveToFile')


class PipelineResultError(ValueError):
    """A bounded message explaining an MCP or schema failure."""


def _validate_result(result, model):
    if not isinstance(result, dict) or type(result.get('isError')) is not bool:
        raise PipelineResultError('MCP вернул некорректный ответ.')
    try:
        size = len(json.dumps(result, ensure_ascii=False).encode('utf-8'))
    except (TypeError, ValueError, UnicodeError):
        raise PipelineResultError('MCP вернул некорректный JSON-ответ.') from None
    if size > MAX_RESULT_BYTES:
        raise PipelineResultError('Результат MCP превышает 16 KiB.')
    if result['isError']:
        content = result.get('content', [])
        message = next((item['text'] for item in content if isinstance(item, dict)
                        and item.get('type') == 'text' and isinstance(item.get('text'), str)), '') if isinstance(content, list) else ''
        raise PipelineResultError(message[:500] or 'MCP-инструмент завершился с ошибкой.')
    if not isinstance(result.get('content'), list) or not isinstance(result.get('structuredContent'), dict):
        raise PipelineResultError('MCP не вернул structuredContent в ожидаемом формате.')
    try:
        model.model_validate(result['structuredContent'])
    except ValidationError:
        raise PipelineResultError('Структурированный результат MCP не соответствует схеме.') from None
    # Validation is a check only: forward the exact received object.
    return result['structuredContent']


def run_pipeline(query, *, limit=3, invoke=None, emit=None):
    arguments = SearchArguments(query=query, limit=limit).model_dump()
    invoke = invoke if invoke is not None else mcp_client.call_tool
    trace = []
    previous = None
    for name, model, input_key in (
        ('search', SearchResult, None), ('summarize', SummaryResult, 'search_result'),
        ('saveToFile', SavedFile, 'report'),
    ):
        if input_key is not None:
            arguments = {input_key: previous}
        if emit is not None:
            emit({'type': 'step_started', 'step': name, 'arguments': arguments, 'server_url': PIPELINE_URL})
        started = perf_counter()
        result = None
        error = None
        try:
            result = invoke(PIPELINE_URL, name, arguments)
            previous = _validate_result(result, model)
        except PipelineResultError as caught:
            error = str(caught)
        except Exception:
            error = ('Не удалось выполнить MCP-вызов. Проверьте, что pipeline_mcp_server.py запущен '
                     'на http://127.0.0.1:8003/mcp.')
            if name == 'saveToFile':
                error += ' Итог сохранения неизвестен; автоматического повтора не будет.'
        duration = round((perf_counter() - started) * 1000)
        entry = {'name': name, 'arguments': arguments, 'result': result,
                 'status': 'error' if error else 'success', 'durationMs': duration, 'serverUrl': PIPELINE_URL}
        if error:
            entry['error'] = error
        trace.append(entry)
        # Persistence callback exceptions deliberately escape this function: the
        # host cannot safely continue after failing to record an earlier step.
        if emit is not None:
            event = {'type': 'step_failed' if error else 'step_completed', 'step': name, 'duration_ms': duration}
            if result is not None:
                event['result'] = result
            if error:
                event['error'] = error
            emit(event)
        if error:
            return {'status': 'error', 'trace': trace, 'saved_file': None,
                    'failed_step': name, 'error': error}
    return {'status': 'success', 'trace': trace, 'saved_file': previous,
            'failed_step': None, 'error': None}
