"""Bounded model → MCP → model exchange, separate from conversation storage."""
import copy
import json
import time

MAX_TOOL_ROUNDS = 2
MAX_TOOL_CALLS = 3


def _usage_dict(usage):
    if usage is None or isinstance(usage, dict):
        return usage
    return {key: getattr(usage, key, None) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}


def run_tools(ask_model, registry, payload, options, tools, metadata, check_context):
    messages = copy.deepcopy(payload['messages'])
    options = {**options, 'tools': copy.deepcopy(tools),
               'extra_body': {'thinking': {'type': 'disabled'}}}
    count = 0
    for round_index in range(MAX_TOOL_ROUNDS + 1):
        final_round = round_index == MAX_TOOL_ROUNDS or count >= MAX_TOOL_CALLS
        request_options = {**options, 'tool_choice': 'none' if final_round else 'auto'}
        request = {**payload, 'messages': copy.deepcopy(messages)}
        actual_payload = {**request, **request_options}
        check_context(actual_payload)
        record = {'payload': actual_payload, 'usage': None, 'response': None, 'status': 'pending'}
        metadata['modelRequests'].append(record)
        started = time.monotonic()
        try:
            response = ask_model(request, **request_options)
            if isinstance(response, str):
                response = {'content': response, 'usage': None}
            record['usage'] = _usage_dict(response.get('usage'))
            calls = response.get('tool_calls') or []
            assistant = {'role': 'assistant', 'content': response.get('content')}
            if calls:
                assistant['tool_calls'] = copy.deepcopy(calls)
            record['response'] = copy.deepcopy(assistant)
            record['status'] = 'success'
        except Exception:
            record['status'] = 'error'
            raise
        finally:
            record['durationMs'] = round((time.monotonic() - started) * 1000)
        if not calls:
            answer = response.get('content')
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError('Модель вернула пустой ответ.')
            return answer
        if final_round:
            raise ValueError('Модель запросила инструмент после исчерпания лимита вызовов.')
        if (len(calls) > 20 or any(not isinstance(c, dict) or not isinstance(c.get('id'), str)
                                 or not isinstance(c.get('function'), dict) for c in calls)
                or len({c['id'] for c in calls}) != len(calls)):
            raise ValueError('Некорректный ответ модели с вызовами инструментов.')
        messages.append(assistant)
        for call in calls:
            name = call['function'].get('name')
            raw_arguments = call['function'].get('arguments')
            trace = {'serverUrl': None, 'name': name, 'modelToolName': name,
                     'toolCallId': call['id'], 'arguments': raw_arguments}
            started = time.monotonic()
            try:
                if not isinstance(raw_arguments, str) or len(raw_arguments) > 16384:
                    raise ValueError('Некорректные аргументы инструмента.')
                try:
                    arguments = json.loads(raw_arguments)
                except ValueError:
                    raise ValueError('Аргументы инструмента не являются JSON.') from None
                trace['arguments'] = arguments
                trace['serverUrl'], trace['name'] = registry.resolve(name)
                if count >= MAX_TOOL_CALLS:
                    raise ValueError('Лимит вызовов инструмента на ход исчерпан.')
                count += 1
                result = registry.call(name, arguments, tools)
            except ValueError as error:
                result = {'content': [{'type': 'text', 'text': str(error)}], 'structuredContent': None, 'isError': True}
            except Exception:
                result = {'content': [{'type': 'text', 'text': 'Не удалось вызвать MCP-инструмент. Проверьте подключение сервера.'}],
                          'structuredContent': None, 'isError': True}
            trace.update(result=copy.deepcopy(result), status='error' if result['isError'] else 'success',
                         durationMs=round((time.monotonic() - started) * 1000))
            metadata['toolCalls'].append(trace)
            messages.append({'role': 'tool', 'tool_call_id': call['id'],
                             'content': json.dumps(result, ensure_ascii=False)})
    raise ValueError('Не получен итоговый ответ модели.')
