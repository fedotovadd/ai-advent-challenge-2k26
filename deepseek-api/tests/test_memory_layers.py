import unittest

from memory_layers import (
    default_memory_layers,
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
            "working": {"task": "", "data": {}},
            "longTerm": {"profile": {}, "decisions": [], "knowledge": {}},
        })
        first["working"]["data"]["goal"] = "catalog"
        self.assertEqual(second["working"]["data"], {})

    def test_working_requires_a_bounded_string_dictionary(self):
        self.assertTrue(valid_working({"task": "", "data": {}}))
        self.assertTrue(valid_working({"task": "Catalog", "data": {"audience": "B2B"}}))
        self.assertFalse(valid_working({"task": "Catalog", "data": {"audience": []}}))
        self.assertFalse(valid_working({"task": "Catalog", "data": {" ": "B2B"}}))
        self.assertFalse(valid_working({"task": "x" * 301, "data": {}}))
        self.assertFalse(valid_working({"task": "Catalog", "data": {
            f"key-{index}": "value" for index in range(25)
        }}))

    def test_long_term_keeps_categories_and_limits_separate(self):
        self.assertTrue(valid_long_term({
            "profile": {"style": "brief"},
            "decisions": ["Free APIs only"],
            "knowledge": {"product": "catalog"},
        }))
        self.assertFalse(valid_long_term({"profile": {}, "decisions": [], "knowledge": {}, "extra": {}}))
        self.assertFalse(valid_long_term({"profile": {}, "decisions": [" "], "knowledge": {}}))
        self.assertFalse(valid_long_term({"profile": {}, "decisions": ["x"] * 21, "knowledge": {}}))
        self.assertFalse(valid_long_term({
            "profile": {f"key-{index}": "value" for index in range(17)},
            "decisions": [], "knowledge": {},
        }))

    def test_memory_layer_validation_and_stable_block(self):
        layers = default_memory_layers()

        self.assertTrue(valid_memory_layers(layers))
        self.assertFalse(valid_memory_layers({"working": layers["working"]}))
        self.assertEqual(
            stable_block("WORKING_MEMORY", {"z": "last", "a": "first"}),
            '[WORKING_MEMORY]\n{"a":"first","z":"last"}',
        )


if __name__ == "__main__":
    unittest.main()
