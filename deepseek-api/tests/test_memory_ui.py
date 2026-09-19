import unittest
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "static" / "memory-controls.js"


class MemoryUiTests(unittest.TestCase):
    def test_panel_exposes_three_read_only_layers(self):
        source = SOURCE.read_text(encoding="utf-8")

        for text in (
            "memory-controls", "<h2>Память</h2>", "Краткосрочная память", "Рабочая память", "Долговременная память",
            "memory-short", "memory-working", "memory-long-term",
        ):
            with self.subTest(text=text):
                self.assertIn(text, source)
        for text in ("<input", "<textarea", "<button", "/memory/working", "Сохранить", "Очистить", "memory-trace", "Последний prompt по слоям"):
            with self.subTest(absent=text):
                self.assertNotIn(text, source)


if __name__ == "__main__":
    unittest.main()
