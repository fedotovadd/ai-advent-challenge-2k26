import unittest
from comparison import evaluate, totals

class ComparisonTests(unittest.TestCase):
    def test_goal_can_reference_subject_in_project_field(self):
        result = evaluate('{"project":"Листок — студия керамики","goal":"запись на занятия"}')
        self.assertTrue(result['checks']['goal'])

    def test_missing_goal_does_not_pass_using_project_alone(self):
        result = evaluate('{"project":"Листок — студия керамики","goal":null}')
        self.assertFalse(result['checks']['goal'])

    def test_budget_must_be_current_exact_numeric_value(self):
        result = evaluate('{"budget":"100000 вместо 200000"}')
        self.assertFalse(result['checks']['budget'])

    def test_invalid_json_is_explicit(self):
        self.assertFalse(evaluate('текст')['validJson'])

    def test_total_includes_memory_call_and_labels_estimates(self):
        calls = [{'usage': {'promptTokens': 10, 'completionTokens': 2, 'totalTokens': 12, 'source': 'actual'},
                  'cost': {'usd': 0.01}, 'seconds': 1},
                 {'usage': {'promptTokens': 20, 'completionTokens': 3, 'totalTokens': 23, 'source': 'estimated'},
                  'cost': {'usd': 0.02}, 'seconds': 2}]
        self.assertEqual(totals(calls)['totalTokens'], 35)
        self.assertEqual(totals(calls)['source'], 'estimated')
