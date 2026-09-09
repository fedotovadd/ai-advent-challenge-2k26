import unittest

import main
import web


class MainTests(unittest.TestCase):
    def test_create_server_uses_injected_model(self):
        def injected_model(payload, **kwargs):
            return "Ответ"

        server = main.create_server("127.0.0.1", 0, injected_model)
        self.addCleanup(server.server_close)

        self.assertIsInstance(server, web.ChatServer)
        self.assertIs(server.ask_model, injected_model)

    def test_main_does_not_reexport_feature_modules(self):
        for name in ("MODEL", "SYSTEM_PROMPT", "DAY_THREE_TASKS", "PAGE"):
            self.assertFalse(hasattr(main, name))


if __name__ == "__main__":
    unittest.main()
