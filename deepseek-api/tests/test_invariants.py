import unittest

from invariants import (
    InvariantCommandError, conflict_for_addition, invariant_prompt_block,
    parse_invariant_command, valid_invariants, violations_for_solution,
)


class InvariantCommandTests(unittest.TestCase):
    def test_free_form_add_and_local_commands(self):
        self.assertEqual(parse_invariant_command("/invariant Не использовать Python"), {
            "action": "add", "text": "Не использовать Python",
        })
        self.assertEqual(parse_invariant_command("/invariants"), {"action": "list"})
        self.assertEqual(parse_invariant_command("/remove-invariant 2"), {"action": "remove", "number": 2})
        self.assertEqual(parse_invariant_command("/clear-invariants"), {"action": "clear"})

    def test_schema_normalizes_and_rejects_duplicate_or_oversized_values(self):
        self.assertTrue(valid_invariants(["Только Kotlin"]))
        self.assertFalse(valid_invariants(["Kotlin", "kotlin"]))
        self.assertFalse(valid_invariants(["  Kotlin  "]))
        with self.assertRaisesRegex(InvariantCommandError, "укажите"):
            parse_invariant_command("/invariant")


class InvariantConstraintTests(unittest.TestCase):
    def test_stack_rules_reject_python_and_require_kotlin_and_ktor(self):
        rules = ["Использовать Kotlin и Ktor", "Не использовать Python"]
        self.assertIn("Не использовать Python", violations_for_solution("Реализуйте на Python", rules))
        self.assertIn("Использовать Kotlin и Ktor", violations_for_solution("Реализуйте на Kotlin", rules))
        self.assertTrue(conflict_for_addition(["Только Kotlin"], "Использовать Java"))

    def test_prompt_block_marks_rules_as_higher_priority_than_context(self):
        block = invariant_prompt_block(["Только Kotlin"])
        self.assertIn("[ИНВАРИАНТЫ]", block)
        self.assertIn("высшего приоритета", block)


if __name__ == "__main__":
    unittest.main()
