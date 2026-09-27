"""Bounded model → MCP → model exchange, separate from conversation storage."""
import copy
import hashlib
import json
import time
import mcp_client

MAX_TOOL_ROUNDS = 6
MAX_TOOL_CALLS = 6


def _usage_dict(usage):
    if usage is None or isinstance(usage, dict):
        return usage
    return {key: getattr(usage, key, None) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}


def run_tools(ask_model, registry, payload, options, tools, metadata, check_context, on_tool_event=None):
    messages = copy.deepcopy(payload['messages'])
    options = {**options, 'tools': copy.deepcopy(tools),
               'extra_body': {'thinking': {'type': 'disabled'}}}
    count = 0
    flow_state = {'search_result': None, 'search_repositories': set(), 'summary_result': None,
                  'selected_repository': None, 'scheduled_jobs': {}, 'schedule_failed': False}
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
            call_started = False
            if count >= MAX_TOOL_CALLS:
                result = {'content': [{'type': 'text', 'text': 'Лимит вызовов инструмента на ход исчерпан.'}],
                          'structuredContent': None, 'isError': True}
            else:
                # Charge the budget before parsing or resolving so bad calls cannot evade the cap.
                count += 1
                try:
                    if not isinstance(raw_arguments, str) or len(raw_arguments) > 16384:
                        raise ValueError('Некорректные аргументы инструмента.')
                    try:
                        arguments = json.loads(raw_arguments)
                    except ValueError:
                        raise ValueError('Аргументы инструмента не являются JSON.') from None
                    trace['arguments'] = arguments
                    trace['serverUrl'], trace['name'] = registry.resolve(name)
                    _restore_flow_input(trace['serverUrl'], trace['name'], arguments, flow_state)
                    _validate_flow_input(trace['serverUrl'], trace['name'], arguments, flow_state)
                    call_started = True
                    _emit_tool_event(on_tool_event, {
                        'type': 'mcp_tool_started', 'toolCallId': call['id'],
                        'serverUrl': trace['serverUrl'], 'name': trace['name'],
                        'arguments': _bounded_event_value(arguments),
                    })
                    result = registry.call(name, arguments, tools)
                    if not result['isError']:
                        _record_flow_result(trace['serverUrl'], trace['name'], arguments, result, flow_state)
                except ValueError as error:
                    result = {'content': [{'type': 'text', 'text': str(error)}], 'structuredContent': None, 'isError': True}
                except Exception:
                    result = {'content': [{'type': 'text', 'text': 'Не удалось вызвать MCP-инструмент. Проверьте подключение сервера.'}],
                              'structuredContent': None, 'isError': True}
            if (trace['serverUrl'] == mcp_client.LOCAL_SCHEDULER_URL
                    and trace['name'] == 'schedule_repository' and result['isError']):
                flow_state['schedule_failed'] = True
            trace.update(result=copy.deepcopy(result), status='error' if result['isError'] else 'success',
                         durationMs=round((time.monotonic() - started) * 1000))
            metadata['toolCalls'].append(trace)
            if call_started:
                _emit_tool_event(on_tool_event, {
                    'type': 'mcp_tool_failed' if result['isError'] else 'mcp_tool_completed',
                    'toolCallId': call['id'], 'serverUrl': trace['serverUrl'],
                    'name': trace['name'], 'durationMs': trace['durationMs'],
                })
            messages.append({'role': 'tool', 'tool_call_id': call['id'],
                             'content': json.dumps(result, ensure_ascii=False)})
    raise ValueError('Не получен итоговый ответ модели.')


def _bounded_event_value(value):
    try:
        encoded = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return {'note': 'Аргументы скрыты.'}
    if len(encoded) > 4000:
        return {'note': 'Аргументы скрыты: слишком большой объём.'}
    return value


def _emit_tool_event(callback, event):
    if callback is None:
        return
    try:
        callback(copy.deepcopy(event))
    except Exception:
        # Progress output is best effort; it must never change tool execution.
        return


def _restore_flow_input(server_url, name, arguments, state):
    """Pass prior structured results through verbatim instead of asking the model to copy them."""
    if not isinstance(arguments, dict):
        return
    if server_url == mcp_client.LOCAL_PIPELINE_URL and name == 'summarize' and state['search_result'] is not None:
        arguments['search_result'] = copy.deepcopy(state['search_result'])
    elif server_url == mcp_client.LOCAL_PIPELINE_URL and name == 'saveToFile' and state['summary_result'] is not None:
        arguments['report'] = copy.deepcopy(state['summary_result'])


def _validate_flow_input(server_url, name, arguments, state):
    if server_url == mcp_client.LOCAL_SCHEDULER_URL and state['schedule_failed']:
        raise ValueError('После ошибки schedule_repository инструменты Scheduler в этом ходе не вызываются.')
    if server_url == mcp_client.LOCAL_PIPELINE_URL and name == 'summarize':
        expected = state['search_result']
        if expected is not None and arguments.get('search_result') != expected:
            raise ValueError('summarize должен получить исходный structuredContent инструмента search.')
    elif server_url == mcp_client.LOCAL_PIPELINE_URL and name == 'saveToFile':
        expected = state['summary_result']
        if expected is not None and arguments.get('report') != expected:
            raise ValueError('saveToFile должен получить исходный structuredContent инструмента summarize.')
    elif server_url == mcp_client.LOCAL_GITHUB_URL and name == 'get_repository':
        repositories = state['search_repositories']
        if (state['search_result'] is not None
                and f"{arguments.get('owner')}/{arguments.get('repo')}" not in repositories):
            raise ValueError('Репозиторий не найден в результатах предыдущего search.')
    elif server_url == mcp_client.LOCAL_SCHEDULER_URL and name == 'schedule_repository':
        repositories = state['search_repositories']
        selected = state['selected_repository']
        candidate = (arguments.get('owner'), arguments.get('repo'))
        if selected is not None and candidate != selected:
            raise ValueError('Расписание должно относиться к репозиторию, выбранному через get_repository.')
        if (selected is None and state['search_result'] is not None
                and f'{candidate[0]}/{candidate[1]}' not in repositories):
            raise ValueError('Расписание можно создать только для репозитория из предыдущего search.')


def _record_flow_result(server_url, name, arguments, result, state):
    structured = result.get('structuredContent')
    if server_url == mcp_client.LOCAL_PIPELINE_URL and name == 'search':
        state['search_result'] = structured
        state['summary_result'] = None
        state['selected_repository'] = None
        items = structured.get('items') if isinstance(structured, dict) else None
        if isinstance(items, list):
            state['search_repositories'] = {
                item['full_name'] for item in items
                if isinstance(item, dict) and isinstance(item.get('full_name'), str)
            }
    elif server_url == mcp_client.LOCAL_PIPELINE_URL and name == 'summarize':
        search = state['search_result']
        if (isinstance(search, dict) and isinstance(structured, dict)
                and (structured.get('query') != search.get('query')
                     or structured.get('source_count') != len(search.get('items', [])))):
            raise ValueError('summarize вернул обзор, не соответствующий результатам search.')
        state['summary_result'] = structured
    elif server_url == mcp_client.LOCAL_PIPELINE_URL and name == 'saveToFile':
        summary = state['summary_result']
        if isinstance(summary, dict) and isinstance(structured, dict):
            markdown = summary.get('markdown')
            if isinstance(markdown, str):
                encoded = markdown.encode('utf-8')
                if (structured.get('bytes_written') != len(encoded)
                        or structured.get('sha256') != hashlib.sha256(encoded).hexdigest()):
                    raise ValueError('Сохранённый файл не совпадает с обзором по размеру или SHA-256.')
    elif server_url == mcp_client.LOCAL_GITHUB_URL and name == 'get_repository':
        state['selected_repository'] = (arguments.get('owner'), arguments.get('repo'))
    elif server_url == mcp_client.LOCAL_SCHEDULER_URL and name == 'schedule_repository':
        job = structured.get('job') if isinstance(structured, dict) else None
        if isinstance(job, dict) and isinstance(job.get('id'), str):
            state['scheduled_jobs'][job['id']] = (arguments.get('owner'), arguments.get('repo'))
    elif server_url == mcp_client.LOCAL_SCHEDULER_URL and name == 'get_schedule_summary':
        job = structured.get('job') if isinstance(structured, dict) else None
        expected = state['scheduled_jobs'].get(arguments.get('job_id'))
        if expected is not None and (
                not isinstance(job, dict) or (job.get('owner'), job.get('repo')) != expected
                or job.get('interval_seconds') is not None
                or structured.get('success_count') != 0 or structured.get('error_count') != 0):
            raise ValueError('Сводка не совпадает с созданным в этом ходе расписанием.')
