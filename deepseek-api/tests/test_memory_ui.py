import unittest
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "static" / "memory-controls.js"


class MemoryUiTests(unittest.TestCase):
    def test_panel_exposes_three_layers_and_explicit_save_actions(self):
        source = SOURCE.read_text(encoding="utf-8")

        for text in (
            "memory-controls", "Краткосрочная память", "Рабочая память", "Долговременная память",
            "Сохранить рабочую память", "Очистить рабочую память", "Сохранить профиль",
            "Сохранить решения", "Сохранить знания", "/memory/working",
            "/memory/long-term/profile", "/memory/long-term/decisions", "/memory/long-term/knowledge",
        ):
            with self.subTest(text=text):
                self.assertIn(text, source)


if __name__ == "__main__":
    unittest.main()
