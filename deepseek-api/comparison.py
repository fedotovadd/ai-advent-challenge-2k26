"""Reproducible Day 10 experiment; uses the same real Agent as the web chat."""
import argparse
import copy
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from agent import Agent, default_settings, estimate_payload_tokens, request_usage_and_cost, estimate_text_tokens
from context_memory import STRATEGIES
from providers import ask_model

FIELDS = ['project', 'goal', 'audience', 'budget', 'deadline', 'platform', 'language',
          'registration', 'payments', 'notifications', 'export', 'hosting']
CHECK = ('Проверь все текущие договорённости. Верни только JSON с ключами ' + ', '.join(FIELDS)
         + '. budget — число рублей. Остальные значения — короткие строки. '
           'Если данных нет в доступном контексте, используй null. Не додумывай.')
SCENARIO = [
    'Собираем ТЗ: проект «Листок», цель — запись на занятия в студии керамики.',
    'Аудитория — взрослые новички. Бюджет на MVP 100000 рублей.',
    'Дедлайн — 20 ноября 2026. Платформа — веб-сайт, удобный на телефоне.',
    'Интерфейс только на русском. Регистрация по email, без социальных сетей.',
    'Онлайн-оплаты в MVP не будет: платят в студии. Уведомления только по email, без SMS.',
    'Нужен экспорт записей в CSV. Размещаем на собственном сервере студии.',
    'Список занятий должен показывать дату и количество свободных мест.',
    'Администратор может отменить занятие, письмо уходит всем записавшимся.',
    'Изменение договорённости: бюджет теперь 200000 рублей вместо 100000. Всё остальное остаётся.',
    'При заполнении группы запись закрывается. Лист ожидания пока не нужен.',
    CHECK,
    'В форме записи достаточно имени и email. Телефон не собираем.',
    'Добавь экран «Мои записи» и возможность отменить свою запись.',
    'Критерий приёмки: нельзя записать больше людей, чем вместимость группы.',
    'Составь итоговое ТЗ. ' + CHECK,
]
EXPECTED = {
    'project': ('листок',), 'goal': ('керамик',), 'audience': ('взросл', 'нович'),
    'budget': ('200000',), 'deadline': ('20', '2026'), 'platform': ('веб',),
    'language': ('рус',), 'registration': ('email',), 'payments': ('студи',),
    'notifications': ('email',), 'export': ('csv',), 'hosting': ('собствен',),
}
SYSTEM = ('Ты бизнес-аналитик, собирающий ТЗ. Запоминай явные решения пользователя. '
          'На обычное уточнение отвечай одним коротким предложением, не повторяй все старые требования. '
          'На проверку договорённостей и итоговое ТЗ верни запрошенный JSON, без Markdown. '
          'Исправления пользователя заменяют старые значения. Не выдумывай отсутствующие данные.')


def evaluate(answer):
    try:
        data = json.loads(answer.strip().removeprefix('```json').removesuffix('```').strip())
        if not isinstance(data, dict):
            raise ValueError('not object')
    except (TypeError, ValueError):
        return {'validJson': False, 'score': 0, 'total': len(EXPECTED), 'checks': {}, 'data': None}
    checks = {}
    for key, needles in EXPECTED.items():
        value = str(data.get(key, '')).lower().replace(' ', '').replace('\u00a0', '')
        checks[key] = all(needle in value for needle in needles)
    goal = str(data.get('goal') or '').lower()
    project = str(data.get('project') or '').lower()
    checks['goal'] = 'запис' in goal and 'занят' in goal and 'керамик' in (goal + project)
    checks['budget'] = type(data.get('budget')) in (int, float) and data['budget'] == 200000
    deadline = str(data.get('deadline') or '').lower()
    checks['deadline'] = ('20' in deadline and '2026' in deadline
                          and ('ноябр' in deadline or '2026-11-20' in deadline or '20.11.2026' in deadline))
    return {'validJson': True, 'score': sum(checks.values()), 'total': len(checks), 'checks': checks, 'data': data}


def run_strategy(strategy, provider, window_size=6, output=None):
    calls, turns = [], []
    def tracked(payload, **options):
        started = time.monotonic()
        response = provider(payload, **options)
        content = response if isinstance(response, str) else response['content']
        usage = None if isinstance(response, str) else response.get('usage')
        details, cost = request_usage_and_cost(payload['model'], usage,
                    estimate_payload_tokens({**payload, **options}), estimate_text_tokens(content))
        calls.append({'request': copy.deepcopy({**payload, **options}), 'answer': content,
                      'usage': details, 'cost': cost, 'seconds': round(time.monotonic() - started, 3)})
        return response
    settings = {**default_settings(), 'contextStrategy': strategy, 'windowSize': window_size,
                'systemPrompt': SYSTEM, 'maxTokens': 900}
    agent = Agent('agent-1', strategy, settings, tracked)
    checkpoint = None
    for number, text in enumerate(SCENARIO, 1):
        start = len(calls)
        state = agent.respond(text, temperature=0)
        turns.append({'turn': number, 'user': text, 'answer': state['messages'][-1]['content'],
                      'facts': copy.deepcopy(state['context']['facts']),
                      'callIndexes': list(range(start, len(calls)))})
        if strategy == 'branching' and number == 8:
            checkpoint = agent.create_checkpoint('После восьмого сообщения')['checkpointId']
        print(f'{strategy}: {number}/{len(SCENARIO)}', flush=True)
    comparison_calls = len(calls)
    result = {'strategy': strategy, 'model': settings['model'], 'temperature': 0, 'windowSize': window_size,
              'primaryThinking': 'provider default (enabled)', 'memoryThinking': 'disabled for DeepSeek',
              'turns': turns, 'checks': {str(n): evaluate(turns[n - 1]['answer']) for n in (11, 15)},
              'linearCalls': comparison_calls, 'branches': []}
    if checkpoint:
        for name, budget in [('Экономный', 90000), ('Расширенный', 250000)]:
            branch = agent.create_branch(checkpoint, name)['branchId']
            agent.switch_branch(branch)
            start = len(calls)
            agent.respond(f'В этой ветке бюджет меняется на {budget} рублей. Прежний бюджет отменён.', temperature=0)
            branch_state = agent.respond('Назови только актуальный бюджет этой ветки в рублях.', temperature=0)
            answer = branch_state['messages'][-1]['content']
            result['branches'].append({'name': name, 'budget': budget, 'answer': answer,
                                      'callIndexes': list(range(start, len(calls))),
                                      'isolated': str(budget) in re.sub(r'\s', '', answer)})
        agent.switch_branch('main')
        result['mainRestored'] = agent.snapshot()['messages'][-1]['content'] == turns[-1]['answer']
    result['calls'] = calls
    result['totals'] = totals(calls[:comparison_calls])
    result['memoryTotals'] = totals([c for c in calls[:comparison_calls] if c['request'].get('extra_body')])
    result['branchExtraTotals'] = totals(calls[comparison_calls:])
    # Includes facts/summary in the measured linear cost, excludes branch demonstration.
    if output:
        Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def totals(calls):
    return {'calls': len(calls),
            **{key: sum(call['usage'][key] for call in calls) for key in ('promptTokens', 'completionTokens', 'totalTokens')},
            'usd': sum(call['cost']['usd'] if call['cost'] else 0 for call in calls),
            'seconds': round(sum(call['seconds'] for call in calls), 3),
            'source': 'actual' if all(call['usage']['source'] == 'actual' for call in calls) else 'estimated'}


def report(results, directory):
    lines = ['# День 10 — сравнение стратегий', '',
             'Реальный прогон API, 15 одинаковых пользовательских сообщений; deepseek-v4-flash, параметр temperature=0, N=6 для оконных режимов. '
             'System prompt одинаковый. Ветвление сравнивается на линейной основной ветке; альтернативы измерены отдельно.', '',
             '| Стратегия | Проверка на ходе 11 | Итог на ходе 15 | Вызовы | Вход | Выход | Всего токенов | USD | Сумма времени, с |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        t = r['totals']
        lines.append(f"| {r['strategy']} | {r['checks']['11']['score']}/12 | {r['checks']['15']['score']}/12 | {t['calls']} | {t['promptTokens']} | {t['completionTokens']} | {t['totalTokens']} | {t['usd']:.6f} | {t['seconds']} |")
    lines += ['', 'В основных вызовах сохранён режим thinking провайдера по умолчанию; отправлен параметр temperature=0, но в этом режиме он не действует. В служебных вызовах facts/summary thinking отключён, лимиты 1200/512 токенов. См. [документацию DeepSeek](https://api-docs.deepseek.com/guides/thinking_mode/). Вызовы и токены включают извлечение facts и обновление summary. Числа берутся из usage API '
              '(поле source в JSON); стоимость рассчитана по тарифам, уже заданным в приложении, без учёта cache hit. '
              'Время — сумма длительностей вызовов внутри стратегии, режимы запускались независимо, часть прогонов — параллельно.', '',
              'Оценка — прозрачная проверка 12 полей JSON по ключевым признакам из исходного ТЗ. Для цели учитывается также предмет проекта из поля project; бюджет должен быть точным числом, дата включает месяц. '
              'Это проверка сохранения требований, а не независимая экспертная оценка качества текста. '
              'Ниже сохранены ответы и результаты каждого пункта. Один прогон не измеряет статистическую повторяемость.', '',
              '## Сценарий', '']
    scenario_heading = lines[-2:]
    lines = lines[:-2]
    lines += ['## Оценка ответа и стабильности', '']
    for r in results:
        a, b = r['checks']['11'], r['checks']['15']
        lost = [key for key, passed in a['checks'].items() if passed and not b['checks'].get(key)]
        lines += [f"- {r['strategy']}: {a['score']}/12 → {b['score']}/12; утрачены между проверками: {', '.join(lost) or 'нет по чек-листу'}. Валидный JSON на обеих проверках: {a['validJson'] and b['validJson']}."]
        data = b.get('data') or {}
        if data.get('goal') is None and 'запис' in str(data.get('project', '')).lower():
            lines += ['  Цель отражена внутри project, но goal=null: ошибка заполнения схемы, а не доказательство утраты смысла в памяти.']
    cheapest = min(results, key=lambda r: r['totals']['usd'])
    complete = [r for r in results if r['checks']['15']['score'] == 12]
    lines += ['', f"Минимальная расчётная стоимость: {cheapest['strategy']}. "]
    if complete:
        cheapest_complete = min(complete, key=lambda r: r['totals']['usd'])
        lines += [f"Среди режимов с 12/12 полей в итоге дешевле: {cheapest_complete['strategy']}."]
    lines += ['Facts требует дополнительного запроса после каждого user; Branching хранит полную историю. '
              'На более длинном диалоге соотношение расходов может измениться.', '', *scenario_heading]
    lines += [f'{i}. {text}' for i, text in enumerate(SCENARIO, 1)]
    for r in results:
        lines += ['', f"## {r['strategy']}", '', f"Полная трасса: [{r['strategy']}.json]({r['strategy']}.json)", '']
        for number in (11, 15):
            check = r['checks'][str(number)]
            lines += [f'### Ответ на ходе {number}', '', '```json', r['turns'][number-1]['answer'], '```', '',
                      'Сохранены: ' + ', '.join(k for k, v in check['checks'].items() if v) + '.',
                      'Не прошли проверку: ' + (', '.join(k for k, v in check['checks'].items() if not v) or 'нет') + '.']
        memory = r.get('memoryTotals', totals([]))
        lines += ['', f"Служебная память отдельно: {memory['calls']} вызовов, {memory['totalTokens']} токенов, ${memory['usd']:.6f}. Уже включена в основную таблицу."]
        if r['branches']:
            lines += ['', '### Две ветки от общего checkpoint', '']
            for branch in r['branches']:
                lines += [f"- {branch['name']}: {branch['answer']} (ожидалось {branch['budget']}; изоляция: {branch['isolated']})."]
            lines += [f"- Возврат к основной ветке сохранил её финальное ТЗ: {r['mainRestored']}.",
                      f"- Дополнительно: {r['branchExtraTotals']['totalTokens']} токенов, ${r['branchExtraTotals']['usd']:.6f}; не включено в основную таблицу."]
    lines += ['', '## Удобство и ограничения', '',
              '| Режим | Удобство | Ограничения |', '|---|---|---|',
              '| Sliding Window | Один параметр N; минимальное управление | Старые детали исчезают, приходится повторять; увеличение N не восстанавливает их |',
              '| Facts | Видимый словарь сохраняет явные договорённости | Дополнительный вызов на каждый user; ошибки извлечения и ограничение 32 факта |',
              '| Branching | Можно проверить альтернативы и вернуться к исходному ТЗ | Нужно явно создавать checkpoint; полная история каждой ветки увеличивает вход |',
              '| Summary | Длинный диалог продолжается с пакетным сжатием | Сводка может потерять точность; её обновление стоит токены |', '',
              'Системные тесты отдельно проверяют физическое удаление, исправление/удаление facts, '
              'неизменяемость checkpoint, отсутствие соседних веток в payload и восстановление с диска.']
    (directory / 'comparison.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/day-10')
    parser.add_argument('--reuse', action='store_true', help='reuse completed strategy files')
    args = parser.parse_args()
    directory = Path(args.output); directory.mkdir(parents=True, exist_ok=True)
    def run(strategy):
        path = directory / f'{strategy}.json'
        if args.reuse and path.exists():
            result = json.loads(path.read_text())
            result['memoryTotals'] = totals([c for c in result['calls'][:result['linearCalls']] if c['request'].get('extra_body')])
            result['checks'] = {str(n): evaluate(result['turns'][n-1]['answer']) for n in (11, 15)}
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            return result
        return run_strategy(strategy, ask_model, output=path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, STRATEGIES))
    (directory / 'manifest.json').write_text(json.dumps({'createdAt': datetime.now(timezone.utc).isoformat(), 'scenario': SCENARIO,
                         'strategies': list(STRATEGIES), 'systemPrompt': SYSTEM}, ensure_ascii=False, indent=2) + '\n')
    report(results, directory)
    print(directory / 'comparison.md')


if __name__ == '__main__':
    main()
