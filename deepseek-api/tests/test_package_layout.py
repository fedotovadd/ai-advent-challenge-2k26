"""Checks that relocated modules remain importable and keep their resources."""

import importlib
import unittest
from pathlib import Path


class PackageLayoutTests(unittest.TestCase):
    def test_public_modules_live_in_responsibility_packages(self):
        for name in (
            "deepseek_app.agent.agent",
            "deepseek_app.providers.providers",
            "deepseek_app.mcp.mcp_client",
            "deepseek_app.pipeline.pipeline_runner",
            "deepseek_app.rag.document_index",
            "deepseek_app.server.web",
        ):
            with self.subTest(name=name):
                self.assertIsNotNone(importlib.import_module(name))

    def test_resources_still_point_at_project_root(self):
        from deepseek_app.agent.agent import DEFAULT_STATE_PATH
        from deepseek_app.pipeline.pipeline_tools import DEFAULT_REPORTS_DIR
        from deepseek_app.server.web import STATIC_PAGE

        root = Path(__file__).resolve().parents[1]
        self.assertEqual(DEFAULT_STATE_PATH, root / "data" / "agents.json")
        self.assertEqual(DEFAULT_REPORTS_DIR, root / "data" / "reports")
        self.assertEqual(STATIC_PAGE, root / "static" / "index.html")
