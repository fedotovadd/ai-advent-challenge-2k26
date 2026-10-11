"""Persist research progress and optional intent interpretation as bounded cards."""
import copy
import json
from uuid import uuid4

STEPS = ('search', 'summarize', 'saveToFile')
CARD_FIELDS = {'run_id', 'agent_id', 'branch_id', 'query', 'status', 'steps', 'saved_file', 'error'}


def valid_pipeline_card(card):
    if not isinstance(card, dict) or set(card) not in (CARD_FIELDS, CARD_FIELDS | {'interpretation'}):
        return False
    if 'interpretation' in card:
        value = card['interpretation']
        if (not isinstance(value, dict) or set(value) != {'status', 'request', 'query', 'limit', 'error', 'durationMs'}
                or value['status'] not in {'running', 'success', 'error', 'interrupted'}
                or not isinstance(value['request'], str) or not 1 <= len(value['request']) <= 2000
                or type(value['durationMs']) is not int or value['durationMs'] < 0
                or (value['error'] is not None and not isinstance(value['error'], str))):
            return False
        if value['status'] == 'success':
            if (not isinstance(value['query'], str) or not 1 <= len(value['query']) <= 200
                    or type(value['limit']) is not int or not 1 <= value['limit'] <= 3):
                return False
        elif value['query'] is not None or value['limit'] is not None:
            return False
    if not all(isinstance(card[k], str) and card[k] for k in ('run_id', 'agent_id', 'branch_id', 'query')):
        return False
    if card['status'] not in {'running', 'success', 'error', 'interrupted'}:
        return False
    steps = card['steps']
    if not isinstance(steps, list) or len(steps) != 3:
        return False
    for name, step in zip(STEPS, steps):
        if not isinstance(step, dict) or step.get('name') != name:
            return False
        if step.get('status') not in {'pending', 'running', 'success', 'error', 'skipped'}:
            return False
        if not isinstance(step.get('arguments'), dict) or not isinstance(step.get('serverUrl'), str):
            return False
        if step.get('result') is not None and not isinstance(step['result'], dict):
            return False
        if type(step.get('durationMs')) is not int or step['durationMs'] < 0:
            return False
    try:
        return len(json.dumps(card, ensure_ascii=False).encode('utf-8')) <= 180_000
    except (TypeError, ValueError):
        return False


def card_content(card):
    if card['status'] == 'success':
        return 'Исследование завершено. Файл: ' + card['saved_file']['path']
    if card['status'] == 'running':
        return 'Исследование: ' + card['query']
    return card.get('error') or 'Исследование прервано; итог последнего шага неизвестен.'


def interrupt_messages(messages):
    changed = False
    for message in messages:
        card = message.get('pipeline')
        if card and card['status'] == 'running':
            changed = True
            card['status'] = 'interrupted'
            card['error'] = 'Сервер перезапущен. Итог последнего шага неизвестен; автоматического повтора нет.'
            if card.get('interpretation', {}).get('status') == 'running':
                card['interpretation'].update(status='interrupted', error=card['error'])
            for step in card['steps']:
                if step['status'] == 'running':
                    step.update(status='error', error='Результат неизвестен.')
                elif step['status'] == 'pending':
                    step['status'] = 'skipped'
            message['content'] = card_content(card)
    return changed


class PipelineHistoryMixin:
    """Host supplies registry locks, Agent snapshots and atomic _save()."""
    def recover_pipelines(self):
        changed = False
        with self._lock:
            for agent in self._agents.values():
                with agent._lock:
                    interrupted = interrupt_messages(agent._messages)
                    changed = interrupted or changed
                    if interrupted and agent._messages[-1].get('pipeline', {}).get('status') == 'interrupted' and agent._metadata:
                        agent._metadata['status'] = {'kind': 'error', 'label': agent._messages[-1]['content']}
                    for group in ('branches', 'checkpoints'):
                        for branch in agent._context[group].values():
                            if branch.get('state') is not None:
                                changed = interrupt_messages(branch['state']['messages']) or changed
            if changed:
                self._save()

    def begin_pipeline(self, agent_id, text, query, *, interpret=False):
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                raise KeyError(agent_id)
            with agent._lock:
                before = agent.snapshot()
                card = {
                    'run_id': str(uuid4()), 'agent_id': agent_id,
                    'branch_id': agent._context['activeBranch'], 'query': query,
                    'status': 'running', 'saved_file': None, 'error': None,
                    'steps': [{'name': name, 'status': 'pending', 'arguments': {},
                               'result': None, 'durationMs': 0, 'serverUrl': ''} for name in STEPS],
                }
                if interpret:
                    card['interpretation'] = {'status': 'running', 'request': query, 'query': None,
                                              'limit': None, 'error': None, 'durationMs': 0}
                agent._messages.extend([{'role': 'user', 'content': text},
                                        {'role': 'assistant', 'content': card_content(card), 'pipeline': card}])
                agent._trim_messages()
                agent._metadata = {
                    'userPrompt': text, 'systemPrompt': '', 'payload': {},
                    'status': {'kind': 'pending', 'label': 'Исследование выполняется'},
                    'responseTimeMs': None, 'usage': None, 'cost': None,
                    'memoryLayers': {'shortTerm': [], 'working': None, 'longTerm': None},
                    'userProfile': None, 'toolCalls': [], 'modelRequests': [],
                }
                try:
                    self._save()
                except Exception:
                    agent._messages, agent._metadata, agent._context = before['messages'], before['metadata'], before['context']
                    raise
                return agent.snapshot(), copy.deepcopy(card)

    def _change_pipeline(self, agent_id, run_id, update, update_agent=None):
        with self._lock:
            agent = self._agents[agent_id]
            with agent._lock:
                message = next((m for m in reversed(agent._messages) if m.get('pipeline', {}).get('run_id') == run_id), None)
                if message is None:
                    raise ValueError('Карточка исследования не найдена.')
                card = message['pipeline']
                if card['branch_id'] != agent._context['activeBranch']:
                    raise ValueError('Ветка исследования изменилась.')
                update(card)
                if update_agent:
                    update_agent(agent)
                message['content'] = card_content(card)
                agent._metadata['toolCalls'] = copy.deepcopy([s for s in card['steps'] if s['status'] in {'success', 'error'}])
                agent._metadata['status'] = {'kind': card['status'], 'label': message['content']}
                agent._metadata['responseTimeMs'] = sum(s['durationMs'] for s in card['steps']) + card.get('interpretation', {}).get('durationMs', 0)
                self._save()
                return agent.snapshot()

    def resolve_pipeline(self, agent_id, run_id, interpreted):
        def update(card):
            if card['status'] != 'running' or card.get('interpretation', {}).get('status') != 'running':
                raise ValueError('Разбор запроса уже завершён.')
            args = interpreted['arguments']
            card['interpretation'].update(status='success' if args else 'error',
                query=args['query'] if args else None, limit=args['limit'] if args else None,
                error=interpreted['error'], durationMs=interpreted['record']['durationMs'])
            if args:
                card['query'] = args['query']

        def update_agent(agent):
            from deepseek_app.agent.agent import request_usage_and_cost, estimate_payload_tokens, estimate_text_tokens, MODEL_CAPABILITIES
            record = copy.deepcopy(interpreted['record'])
            agent._metadata['modelRequests'] = [record]
            agent._metadata['payload'] = record['payload']
            agent._metadata['systemPrompt'] = record['payload']['messages'][0]['content']
            response = record['response']
            usage, cost = request_usage_and_cost(agent._settings['model'], record['usage'],
                estimate_payload_tokens(record['payload']) if response else None,
                estimate_text_tokens(response['content']) if response else None)
            agent._metadata.update(usage=usage, cost=cost)
            if usage:
                for key in ('promptTokens', 'completionTokens', 'totalTokens'):
                    agent._metrics['totals'][key] += usage[key]
                if cost:
                    agent._metrics['totals']['usd'] += cost['usd']
                totals = agent._metrics['totals']
                estimate = estimate_payload_tokens(record['payload'])
                context_limit = MODEL_CAPABILITIES[agent._settings['model']]['contextLimit']
                metric = {
                    'messageNumber': len(agent._metrics['calls']) + 1,
                    'messageTokens': estimate_text_tokens(record['payload']['messages'][-1]['content']),
                    'historyBeforeTokens': 0, 'historyAfterTokens': 0,
                    'payloadEstimatedTokens': estimate, 'fullPayloadEstimatedTokens': estimate,
                    'compressedPayloadEstimatedTokens': estimate,
                    'compressionGrossSavedTokens': 0, 'summaryCallTokens': 0,
                    'compressionNetSavedTokens': 0, 'grossInputSavingsUsd': 0,
                    'summaryCostUsd': 0, 'netSavingsUsd': 0,
                    **usage, 'contextLimit': context_limit, 'contextPercent': estimate / context_limit,
                    'cost': copy.deepcopy(cost), 'cumulativePromptTokens': totals['promptTokens'],
                    'cumulativeCompletionTokens': totals['completionTokens'],
                    'cumulativeTotalTokens': totals['totalTokens'], 'cumulativeUsd': totals['usd'],
                }
                agent._metrics['calls'].append(metric)
                agent._metrics['lastMessage'] = copy.deepcopy(metric)
                agent._metrics['lastContextAttempt'] = None
        return self._change_pipeline(agent_id, run_id, update, update_agent)

    def update_pipeline(self, agent_id, run_id, event):
        def update(card):
            step = next(s for s in card['steps'] if s['name'] == event['step'])
            if card['status'] != 'running':
                raise ValueError('Исследование уже завершено.')
            if event['type'] == 'step_started':
                if step['status'] != 'pending' or any(s['status'] != 'success' for s in card['steps'][:STEPS.index(step['name'])]):
                    raise ValueError('Нарушен порядок шагов.')
                step.update(status='running', arguments=copy.deepcopy(event['arguments']), serverUrl=event['server_url'])
            elif event['type'] in {'step_completed', 'step_failed'}:
                if step['status'] != 'running':
                    raise ValueError('Шаг не был запущен.')
                step.update(status='success' if event['type'] == 'step_completed' else 'error',
                            result=copy.deepcopy(event.get('result')), durationMs=event['duration_ms'])
                if event.get('error'):
                    step['error'] = event['error']
            else:
                raise ValueError('Неизвестное событие исследования.')
        return self._change_pipeline(agent_id, run_id, update)

    def finish_pipeline(self, agent_id, run_id, outcome):
        def update(card):
            if outcome['status'] == 'success' and any(s['status'] != 'success' for s in card['steps']):
                raise ValueError('Не все шаги завершились успешно.')
            card.update(status=outcome['status'], saved_file=copy.deepcopy(outcome.get('saved_file')), error=outcome.get('error'))
            if card.get('interpretation', {}).get('status') == 'running':
                card['interpretation'].update(status='error', error=outcome.get('error'))
            for step in card['steps']:
                if step['status'] == 'pending':
                    step['status'] = 'skipped'
                elif step['status'] == 'running':
                    step.update(status='error', error=outcome.get('error') or 'Результат неизвестен.')
        return self._change_pipeline(agent_id, run_id, update)
