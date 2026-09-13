import copy
import json
import tempfile
import unittest
from pathlib import Path

from agent import Agent, AgentRegistry, default_settings


class StrategyTests(unittest.TestCase):
    def make_agent(self, strategy='sliding_window', size=4, responder=None):
        self.calls = []
        def ask(payload, **options):
            self.calls.append(copy.deepcopy({**payload, **options}))
            if responder:
                return responder(payload, options)
            return 'Принято'
        return Agent('agent-1', 'Тест', {**default_settings(), 'contextStrategy': strategy, 'windowSize': size}, ask)

    def test_window_discards_old_messages_and_counts_current_user(self):
        agent = self.make_agent(size=3)
        for text in ('секрет первого сообщения', 'второе', 'третье', 'четвёртое'):
            snapshot = agent.respond(text)
        self.assertEqual(len(snapshot['messages']), 3)
        self.assertEqual(len(self.calls[-1]['messages']), 4)  # system + exactly N
        self.assertNotIn('секрет', json.dumps(snapshot, ensure_ascii=False))
        self.assertEqual(len(self.calls), 4)  # no summary
        self.assertEqual(self.calls[-1]['messages'][-1]['content'], 'четвёртое')

    def test_changing_window_does_not_resurrect_discarded_messages(self):
        agent = self.make_agent(size=2)
        agent.respond('старое')
        agent.respond('новое')
        agent.update_settings({**agent.snapshot()['settings'], 'windowSize': 20})
        agent.respond('продолжим')
        self.assertNotIn('старое', json.dumps(self.calls[-1], ensure_ascii=False))

    def test_facts_update_replace_delete_and_survive_window(self):
        patches = iter([{'set': {'budget': '100', 'goal': 'магазин'}, 'delete': []},
                        {'set': {'budget': '200'}, 'delete': []},
                        {'set': {}, 'delete': ['goal']}])
        def respond(payload, options):
            if options.get('response_format'):
                return {'content': json.dumps(next(patches)), 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
            return {'content': 'Принято', 'usage': {'prompt_tokens': 20, 'completion_tokens': 5}}
        agent = self.make_agent('facts', size=2, responder=respond)
        agent.respond('Бюджет 100, цель магазин')
        agent.respond('Измени бюджет на 200')
        result = agent.respond('Удали цель')
        self.assertEqual(result['context']['facts'], {'budget': '200'})
        self.assertIn('200', self.calls[-1]['messages'][1]['content'])
        self.assertNotIn('магазин', self.calls[-1]['messages'][1]['content'])
        self.assertEqual(len(result['messages']), 2)
        self.assertEqual(result['context']['factsUsage']['calls'], 3)
        self.assertEqual(result['context']['factsUsage']['totalTokens'], 45)
        self.assertEqual(result['metrics']['totals']['totalTokens'], 75)

    def test_invalid_facts_keeps_old_facts_and_records_extraction_cost(self):
        def respond(payload, options):
            return {'content': '{"set": {"bad": []}, "delete": []}',
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
        agent = self.make_agent('facts', responder=respond)
        with self.assertRaisesRegex(RuntimeError, 'facts'):
            agent.respond('Запомни цель')
        state = agent.snapshot()
        self.assertEqual(state['context']['facts'], {})
        self.assertEqual(state['messages'], [])
        self.assertEqual(state['context']['factsUsage']['totalTokens'], 15)
        self.assertEqual(len(self.calls), 1)

    def test_facts_committed_even_when_primary_fails(self):
        def respond(payload, options):
            if options.get('response_format'):
                return '{"set":{"budget":"200"},"delete":[]}'
            raise RuntimeError('network')
        agent = self.make_agent('facts', responder=respond)
        with self.assertRaisesRegex(RuntimeError, 'network'):
            agent.respond('Бюджет 200')
        self.assertEqual(agent.snapshot()['context']['facts'], {'budget': '200'})
        self.assertEqual(agent.snapshot()['context']['factsUsage']['calls'], 1)

    def test_branches_are_independent_and_checkpoint_is_immutable(self):
        agent = self.make_agent('branching')
        agent.respond('Общая цель')
        checkpoint = agent.create_checkpoint('Общая точка')['checkpointId']
        a = agent.create_branch(checkpoint, 'Вариант А')['branchId']
        b = agent.create_branch(checkpoint, 'Вариант Б')['branchId']
        agent.switch_branch(a)
        state_a = agent.respond('Только вариант А')
        agent.switch_branch(b)
        self.assertEqual(len(agent.snapshot()['messages']), 2)
        state_b = agent.respond('Только вариант Б')
        self.assertNotIn('Только вариант А', json.dumps(self.calls[-1], ensure_ascii=False))
        self.assertIn('Общая цель', json.dumps(self.calls[-1], ensure_ascii=False))
        agent.switch_branch(a)
        self.assertEqual(agent.snapshot()['messages'], state_a['messages'])
        self.assertEqual(agent.snapshot()['metrics'], state_a['metrics'])
        self.assertNotEqual(state_a['messages'], state_b['messages'])
        self.assertEqual(len(self.calls), 3)
        c = agent.create_branch(checkpoint, 'Вариант В')['branchId']
        agent.switch_branch(c)
        self.assertEqual(len(agent.snapshot()['messages']), 2)

    def test_invalid_branch_actions_do_not_mutate_state(self):
        agent = self.make_agent('branching')
        before = agent.snapshot()
        for action in (lambda: agent.create_branch('missing', 'A'), lambda: agent.switch_branch('missing'),
                       lambda: agent.create_checkpoint(' ')):
            with self.assertRaises(ValueError):
                action()
            self.assertEqual(agent.snapshot(), before)

    def test_snapshot_and_restart_preserve_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            registry = AgentRegistry(lambda payload, **options: 'Ответ', path)
            registry.update_settings('agent-1', {**default_settings(), 'contextStrategy': 'branching'})
            registry.respond('agent-1', 'Начало')
            cp = registry.context_action('agent-1', 'checkpoint', {'name': 'Точка'})['checkpointId']
            a = registry.context_action('agent-1', 'branch', {'checkpointId': cp, 'name': 'A'})['branchId']
            registry.context_action('agent-1', 'switch', {'branchId': a})
            registry.respond('agent-1', 'Только А')
            before = registry.agents()
            restored = AgentRegistry(lambda payload, **options: 'Ответ', path)
            self.assertEqual(restored.agents(), before)
            result = restored.context_action('agent-1', 'switch', {'branchId': 'main'})
            self.assertEqual(len(result['agent']['messages']), 2)

    def test_validation_rejects_bad_strategy_and_window(self):
        agent = self.make_agent()
        for change in ({'contextStrategy': 'unknown'}, {'contextStrategy': []}, {'windowSize': True},
                       {'windowSize': 0}, {'windowSize': 101}, {'windowSize': 1.5}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                agent.update_settings({**default_settings(), **change})

    def test_corrupt_nested_state_is_rejected_without_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            good = self.make_agent('branching').snapshot()
            mutations = [lambda s: s.update(context=None),
                         lambda s: s['context'].update(branches=[]),
                         lambda s: s['context'].update(checkpoints=[]),
                         lambda s: s['context']['branches']['main']['state']['metrics'].update(calls=None)]
            for mutate in mutations:
                saved = copy.deepcopy(good)
                mutate(saved)
                path.write_text(json.dumps({'version': 4, 'nextId': 2, 'agents': [saved]}))
                restored = AgentRegistry(lambda payload, **options: 'Ответ', path)
                self.assertEqual(restored.agents()[0]['settings']['contextStrategy'], 'sliding_window')

    def test_failed_settings_save_does_not_discard_history(self):
        from unittest.mock import patch
        from agent import PersistenceError
        with tempfile.TemporaryDirectory() as directory:
            registry = AgentRegistry(lambda payload, **options: 'Ответ', Path(directory) / 'state.json')
            registry.update_settings('agent-1', {**default_settings(), 'contextStrategy': 'branching'})
            for text in ('один', 'два', 'три'):
                registry.respond('agent-1', text)
            before = registry.agents()
            with patch.object(registry, '_save', side_effect=PersistenceError('disk')):
                with self.assertRaises(PersistenceError):
                    registry.update_settings('agent-1', {**default_settings(), 'windowSize': 2})
            self.assertEqual(registry.agents(), before)

    def test_facts_limits_include_whitespace(self):
        from context_memory import apply_facts_patch, FactsUpdateError
        for patch in ({'set': {'goal': ' ' * 501 + 'x'}, 'delete': []},
                      {'set': {' ' * 81 + 'x': 'value'}, 'delete': []}):
            with self.assertRaises(FactsUpdateError):
                apply_facts_patch({}, json.dumps(patch))

    def test_history_after_metric_matches_retained_window(self):
        from agent import estimate_payload_tokens
        agent = self.make_agent(size=2)
        agent.respond('первое')
        result = agent.respond('второе')
        self.assertEqual(result['metrics']['lastMessage']['historyAfterTokens'],
                         estimate_payload_tokens({'messages': result['messages']}))

    def test_summary_output_limit_is_passed_as_provider_option(self):
        calls = []
        def respond(payload, options):
            calls.append((payload, options))
            return 'Сводка' if payload['temperature'] == 0 else 'Ответ'
        instance = self.make_agent('summary', responder=respond)
        for number in range(11):
            instance.respond(str(number))
        self.assertEqual(calls[-2][1].get('max_tokens'), 512)
        self.assertEqual(calls[-2][1].get('extra_body'), {'thinking': {'type': 'disabled'}})


if __name__ == '__main__':
    unittest.main()
