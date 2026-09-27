import copy
import unittest
from unittest.mock import Mock

from test_pipeline_models import SEARCH, SUMMARY

SAVED = {'path': '/tmp/reports/report.md', 'bytes_written': 12, 'sha256': 'a' * 64}


def envelope(data):
    import json
    return {'content': [{'type': 'text', 'text': json.dumps(data)}], 'structuredContent': data, 'isError': False}


class PipelineRunnerTests(unittest.TestCase):
    def test_three_calls_forward_original_structured_content_and_emit_in_order(self):
        from pipeline_runner import run_pipeline
        from mcp_client import LOCAL_PIPELINE_URL
        results = [envelope(copy.deepcopy(data)) for data in (SEARCH, SUMMARY, SAVED)]
        calls, events, timeline = [], [], []
        def invoke(url, name, arguments):
            calls.append((url, name, arguments))
            timeline.append('call:' + name)
            return results[len(calls) - 1]
        def emit(event):
            events.append(event)
            timeline.append(event['type'] + ':' + event['step'])
        result = run_pipeline(' mcp ', invoke=invoke, emit=emit)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['saved_file'], SAVED)
        self.assertIsNone(result['failed_step'])
        self.assertIsNone(result['error'])
        self.assertEqual([call[1] for call in calls], ['search', 'summarize', 'saveToFile'])
        self.assertTrue(all(call[0] == LOCAL_PIPELINE_URL for call in calls))
        self.assertEqual(calls[0][2], {'query': 'mcp', 'limit': 3})
        self.assertIs(calls[1][2]['search_result'], results[0]['structuredContent'])
        self.assertIs(calls[2][2]['report'], results[1]['structuredContent'])
        self.assertEqual(timeline, [f'{event}:{name}' for name in ('search', 'summarize', 'saveToFile')
                                    for event in ('step_started', 'call', 'step_completed')])
        for index, step in enumerate(result['trace']):
            self.assertIs(step['result'], results[index])
            self.assertEqual(step['status'], 'success')
            self.assertEqual(step['arguments'], calls[index][2])
            self.assertGreaterEqual(step['durationMs'], 0)
            self.assertIsInstance(step['durationMs'], int)
            self.assertEqual(step['serverUrl'], LOCAL_PIPELINE_URL)
        self.assertIs(events[1]['result'], results[0])
        self.assertTrue(all('run_id' not in event and 'seq' not in event for event in events))

    def test_each_step_stops_on_error_timeout_missing_data_and_invalid_schema(self):
        from pipeline_runner import run_pipeline
        names = ['search', 'summarize', 'saveToFile']
        for failed_index in range(3):
            for bad in [{'isError': True, 'content': [{'type': 'text', 'text': 'Tool failed'}]},
                        {'isError': False, 'content': []}, envelope({'wrong': 1}),
                        TimeoutError('SECRET'), envelope(None), {'isError': 'false', 'content': []}]:
                results = [envelope(copy.deepcopy(data)) for data in (SEARCH, SUMMARY, SAVED)]
                results[failed_index] = bad
                events = []
                invoke = Mock(side_effect=results)
                with self.subTest(step=names[failed_index], bad=bad):
                    result = run_pipeline('mcp', invoke=invoke, emit=events.append)
                    self.assertEqual(result['status'], 'error')
                    self.assertEqual(result['failed_step'], names[failed_index])
                    self.assertIsNone(result['saved_file'])
                    self.assertEqual(invoke.call_count, failed_index + 1)
                    self.assertEqual(len(result['trace']), failed_index + 1)
                    self.assertEqual(result['trace'][-1]['status'], 'error')
                    self.assertEqual(events[-1]['type'], 'step_failed')
                    self.assertNotIn('SECRET', result['error'])
                    if not isinstance(bad, Exception):
                        self.assertIs(result['trace'][-1]['result'], bad)

    def test_invalid_input_has_no_mcp_calls_or_events(self):
        from pipeline_runner import run_pipeline
        for query, limit in [('', 3), ('x' * 201, 3), ('mcp', True)]:
            invoke, emit = Mock(), Mock()
            with self.assertRaises(ValueError):
                run_pipeline(query, limit=limit, invoke=invoke, emit=emit)
            invoke.assert_not_called()
            emit.assert_not_called()

    def test_callback_failure_propagates_and_prevents_next_call(self):
        from pipeline_runner import run_pipeline
        for when, expected_calls in [('step_started', 0), ('step_completed', 1), ('step_failed', 1)]:
            invoke = Mock(return_value=envelope(SEARCH) if when != 'step_failed' else {'isError': True, 'content': []})
            def emit(event):
                if event['type'] == when:
                    raise OSError('persistence failed')
            with self.subTest(when=when), self.assertRaisesRegex(OSError, 'persistence failed'):
                run_pipeline('mcp', invoke=invoke, emit=emit)
            self.assertEqual(invoke.call_count, expected_calls)

    def test_runner_rejects_oversized_envelopes_without_forwarding_them(self):
        from pipeline_runner import run_pipeline
        response = envelope(SEARCH)
        response['content'][0]['text'] = 'я' * 9000
        invoke = Mock(return_value=response)
        result = run_pipeline('mcp', invoke=invoke)
        self.assertEqual(result['status'], 'error')
        self.assertIn('16 KiB', result['error'])
        self.assertEqual(invoke.call_count, 1)
