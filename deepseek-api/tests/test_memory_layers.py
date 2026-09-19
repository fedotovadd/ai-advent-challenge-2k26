import unittest

from memory_layers import (
    default_memory_layers,
    MemoryCommandError,
    parse_memory_command,
    stable_block,
    valid_long_term,
    valid_memory_layers,
    valid_working,
)


class MemoryLayerTests(unittest.TestCase):
    def test_defaults_are_separate_and_independent(self):
        first = default_memory_layers()
        second = default_memory_layers()

        self.assertEqual(first, {
            "working": [],
            "longTerm": [],
        })
        first["working"].append("catalog")
        self.assertEqual(second["working"], [])

    def test_working_requires_a_bounded_list_of_text_items(self):
        self.assertTrue(valid_working([]))
        self.assertTrue(valid_working(["Catalog", "B2B"] ))
        self.assertFalse(valid_working({"task": "Catalog", "data": {}}))
        self.assertFalse(valid_working([" "]))
        self.assertFalse(valid_working(["x" * 501]))
        self.assertFalse(valid_working([f"item-{index}" for index in range(25)]))

    def test_long_term_requires_a_bounded_list_of_text_items(self):
        self.assertTrue(valid_long_term(["Free APIs only", "catalog"] ))
        self.assertFalse(valid_long_term({"profile": {}, "decisions": [], "knowledge": {}}))
        self.assertFalse(valid_long_term([" "]))
        self.assertFalse(valid_long_term([f"item-{index}" for index in range(53)]))

    def test_memory_layer_validation_and_stable_block(self):
        layers = default_memory_layers()

        self.assertTrue(valid_memory_layers(layers))
        self.assertFalse(valid_memory_layers({"working": layers["working"]}))
        self.assertEqual(
            stable_block("WORKING_MEMORY", {"z": "last", "a": "first"}),
            '[WORKING_MEMORY]\n{"a":"first","z":"last"}',
        )

    def test_parse_memory_commands_and_reject_incomplete_ones(self):
        self.assertEqual(parse_memory_command("/working Дизайнерский проект"), {
            "action": "working", "text": "Дизайнерский проект",
        })
        self.assertEqual(parse_memory_command("/long Меня зовут Диана"), {
            "action": "long", "text": "Меня зовут Диана",
        })
        self.assertEqual(parse_memory_command("/clear-working"), {"action": "clear-working"})
        self.assertEqual(parse_memory_command("/clear-long"), {"action": "clear-long"})
        self.assertEqual(parse_memory_command("/clear-memory"), {"action": "clear-memory"})
        self.assertIsNone(parse_memory_command("/profile Имя: Диана"))
        for command in ("/working", "/long ", "/clear-long extra"):
            with self.subTest(command=command), self.assertRaises(MemoryCommandError):
                parse_memory_command(command)


if __name__ == "__main__":
    unittest.main()
