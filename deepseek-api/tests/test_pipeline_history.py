import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from deepseek_app.agent.agent import AgentRegistry, PersistenceError, _valid_message


class PipelineHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'agents.json'
        self.ask = Mock(return_value='Ответ')
        self.registry = AgentRegistry(self.ask, self.path)

    def begin(self):
        return self.registry.begin_pipeline('agent-1', '/research mcp', 'mcp')

    def test_progress_survives_reload_and_interrupted_is_not_success(self):
        self.assertTrue(hasattr(self.registry, 'begin_pipeline'), 'pipeline history is missing')
        snapshot, card = self.begin()
        self.assertEqual(len(snapshot['messages']), 2)
        self.assertTrue(_valid_message(snapshot['messages'][-1]))
        self.registry.update_pipeline('agent-1', card['run_id'], {'type': 'step_started', 'step': 'search', 'arguments': {'query': 'mcp'}, 'server_url': 'local'})
        restored = AgentRegistry(self.ask, self.path).get('agent-1').snapshot()
        self.assertEqual(restored['messages'][-1]['pipeline']['status'], 'interrupted')
        self.assertIn('неизвест', restored['messages'][-1]['content'])
        self.ask.assert_not_called()

    def test_completed_card_and_metadata_survive_without_model_cost(self):
        self.assertTrue(hasattr(self.registry, 'begin_pipeline'))
        before = self.registry.get('agent-1').snapshot()['metrics']
        snapshot, card = self.begin()
        result = {'content': [], 'structuredContent': {'items': []}, 'isError': False}
        self.registry.update_pipeline('agent-1', card['run_id'], {'type': 'step_started', 'step': 'search', 'arguments': {'query': 'mcp'}, 'server_url': 'local'})
        self.registry.update_pipeline('agent-1', card['run_id'], {'type': 'step_completed', 'step': 'search', 'result': result, 'duration_ms': 4})
        outcome = {'status': 'error', 'failed_step': 'summarize', 'error': 'Ошибка', 'saved_file': None}
        self.registry.finish_pipeline('agent-1', card['run_id'], outcome)
        restored = AgentRegistry(self.ask, self.path).get('agent-1').snapshot()
        self.assertEqual(restored['messages'][-1]['pipeline']['steps'][0]['result'], result)
        self.assertEqual(restored['metadata']['toolCalls'][0]['arguments'], {'query': 'mcp'})
        self.assertEqual(restored['metadata']['modelRequests'], [])
        self.assertEqual(restored['metrics'], before)
        self.assertEqual(restored['messages'][-1]['pipeline']['status'], 'error')

    def test_model_receives_only_role_content_after_pipeline(self):
        self.assertTrue(hasattr(self.registry, 'begin_pipeline'))
        snapshot, card = self.begin()
        self.registry.finish_pipeline('agent-1', card['run_id'], {'status': 'error', 'error': 'Нет подключения', 'failed_step': 'search', 'saved_file': None})
        self.registry.respond('agent-1', 'Привет')
        for message in self.ask.call_args.args[0]['messages']:
            self.assertEqual(set(message), {'role', 'content'})

    def test_failed_initial_save_rolls_back_and_old_messages_remain_valid(self):
        self.assertTrue(hasattr(self.registry, 'begin_pipeline'))
        with patch.object(self.registry, '_save', side_effect=PersistenceError('disk')):
            with self.assertRaises(PersistenceError):
                self.begin()
        self.assertEqual(self.registry.get('agent-1').snapshot()['messages'], [])
        self.assertTrue(_valid_message({'role': 'user', 'content': 'старое сообщение'}))
        self.assertFalse(_valid_message({'role': 'assistant', 'content': 'bad', 'pipeline': {}}))

    def test_fact_extraction_does_not_receive_pipeline_json(self):
        import json
        snapshot, card = self.begin()
        self.registry.finish_pipeline('agent-1', card['run_id'], {'status': 'error', 'error': 'Нет подключения', 'failed_step': 'search', 'saved_file': None})
        target = self.registry.get('agent-1')
        self.ask.return_value = '{"set": {}, "delete": []}'
        try:
            target._update_facts('Привет')
        except ValueError:
            pass
        dialogue = json.loads(self.ask.call_args.args[0]['messages'][-1]['content'])['dialogue']
        self.assertTrue(all(set(m) == {'role', 'content'} for m in dialogue))

    def test_restart_during_interpretation_marks_it_interrupted(self):
        self.registry.begin_pipeline('agent-1', '/research Найди Python', 'Найди Python', interpret=True)
        restored = AgentRegistry(self.ask, self.path).get('agent-1').snapshot()
        card = restored['messages'][-1]['pipeline']
        self.assertEqual(card['interpretation']['status'], 'interrupted')
        self.assertEqual([s['status'] for s in card['steps']], ['skipped'] * 3)
        self.assertTrue(_valid_message(restored['messages'][-1]))
        self.ask.assert_not_called()
